"""
Chuchu (Balanced + ChangePoints)
================================
Implementation of the decision-boundary exploration pipeline known as
"Chuchu".  This file is a direct port of the DelDel implementation shared by
our collaborators with naming updates so the API lives under the Chuchu
namespace.
"""

from __future__ import annotations

import copy
import logging
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =============================================================================
# Logging utilities
# =============================================================================

def _get_logger(name: str = "chuchu", level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with a simple stream handler."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# =============================================================================
# Helper utilities
# =============================================================================

_EPS = 1e-12

def _softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = np.asarray(z, float)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=axis, keepdims=True) + _EPS)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1.0 / (1.0 + np.exp(-z))


def _jsd(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence in the ``[0, 1]`` range."""

    P = np.clip(np.asarray(P, float), eps, 1.0)
    P /= P.sum()
    Q = np.clip(np.asarray(Q, float), eps, 1.0)
    Q /= Q.sum()
    M = 0.5 * (P + Q)

    def _kl(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.sum(A * (np.log2(A) - np.log2(B))))

    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)


# =============================================================================
# Score adaptor
# =============================================================================

class ScoreAdaptor:
    """Uniform adapter around sklearn-style models."""

    def __init__(self, model: Any, mode: str = "auto"):
        self.model = model
        self.mode = mode

    def scores(self, X: np.ndarray) -> np.ndarray:
        model = self.model
        mode = self.mode
        if mode == "auto":
            if hasattr(model, "predict_proba"):
                mode = "proba"
            elif hasattr(model, "decision_function"):
                mode = "decision"
            elif callable(model):
                mode = "callable"
            else:
                raise ValueError("Modelo no soportado.")
        X = np.asarray(X, float)
        if mode == "proba":
            return np.asarray(model.predict_proba(X), float)
        if mode == "decision":
            decision = np.asarray(model.decision_function(X), float)
            if decision.ndim == 1:
                p1 = _sigmoid(decision).reshape(-1, 1)
                return np.c_[1.0 - p1, p1]
            return _softmax(decision, axis=1)
        if mode == "callable":
            scores = np.asarray(model(X), float)
            if scores.ndim == 1:
                scores = np.c_[1.0 - scores, scores.reshape(-1, 1)]
            return scores
        raise ValueError("Modo desconocido.")


def macro_f1_ignore_rejects(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    reject_label: int = -1,
) -> float:
    """Compute the macro F1 score ignoring rejected samples.

    Parameters
    ----------
    y_true, y_pred:
        Arrays containing the ground-truth labels and the predictions.
    reject_label:
        Label used to mark rejected predictions. Samples with this
        prediction are ignored when computing the score.

    Returns
    -------
    float
        Macro-averaged F1 score over the non-rejected classes. When all
        samples are rejected the score defaults to ``0.0``.
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mask = y_pred_arr != reject_label
    if not np.any(mask):
        return 0.0

    labels = np.unique(y_true_arr[mask])
    if labels.size == 0:
        return 0.0

    f1_scores: List[float] = []
    for cls in labels:
        tp = np.sum((y_true_arr == cls) & (y_pred_arr == cls))
        fp = np.sum((y_true_arr != cls) & (y_pred_arr == cls))
        fn = np.sum((y_true_arr == cls) & (y_pred_arr != cls))
        if tp == 0 and (fp > 0 or fn > 0):
            f1_scores.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    if not f1_scores:
        return 0.0
    return float(np.mean(f1_scores))


# =============================================================================
# False-position with bisection refinement
# =============================================================================

def batch_false_position_flip(
    adaptor: ScoreAdaptor,
    A: np.ndarray,
    B: np.ndarray,
    yA: np.ndarray,
    yB: np.ndarray,
    iters: int = 2,
    final_bisect: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, d = A.shape
    SA = adaptor.scores(A)
    SB = adaptor.scores(B)
    hA = SA[np.arange(m), yA] - SA[np.arange(m), yB]
    hB = SB[np.arange(m), yA] - SB[np.arange(m), yB]

    denom = hA - hB
    t = np.clip(hA / (denom + 1e-12), 1e-4, 1 - 1e-4)
    X = (1.0 - t)[:, None] * A + t[:, None] * B
    S = adaptor.scores(X)
    y = np.argmax(S, axis=1)

    for _ in range(max(0, iters - 1)):
        hX = S[np.arange(m), yA] - S[np.arange(m), yB]
        maskA = (y == yA) | ((hX > 0) & (y != yA) & (y != yB))
        maskB = (y == yB) | ((hX <= 0) & (y != yA) & (y != yB))
        A = np.where(maskA[:, None], X, A)
        B = np.where(maskB[:, None], X, B)
        SA = np.where(maskA[:, None], S, SA)
        SB = np.where(maskB[:, None], S, SB)
        hA = SA[np.arange(m), yA] - SA[np.arange(m), yB]
        hB = SB[np.arange(m), yA] - SB[np.arange(m), yB]
        t = np.clip(hA / ((hA - hB) + 1e-12), 1e-4, 1 - 1e-4)
        X = (1.0 - t)[:, None] * A + t[:, None] * B
        S = adaptor.scores(X)
        y = np.argmax(S, axis=1)

    lo = np.zeros(m)
    hi = np.ones(m)
    XA, XB = A.copy(), B.copy()
    for _ in range(final_bisect):
        mid = 0.5 * (lo + hi)
        XM = (1.0 - mid)[:, None] * XA + mid[:, None] * XB
        SM = adaptor.scores(XM)
        yM = np.argmax(SM, axis=1)
        goA = yM == yA
        lo = np.where(goA, mid, lo)
        hi = np.where(goA, hi, mid)
        XA = np.where(goA[:, None], XM, XA)
        XB = np.where(~goA[:, None], XM, XB)
    Xstar = (1.0 - hi)[:, None] * A + hi[:, None] * B
    Sstar = adaptor.scores(Xstar)
    ystar = np.argmax(Sstar, axis=1)
    return Xstar, ystar, Sstar


# =============================================================================
# Records and configuration dataclasses
# =============================================================================

@dataclass
class DeltaRecord:
    index_a: int
    index_b: int
    method: str
    success: bool
    y0: int
    y1: int
    delta_norm_l2: float
    delta_norm_linf: float
    score_change: float
    distance_term: float
    change_term: float
    final_score: float
    time_ms: float
    x0: np.ndarray
    x1: np.ndarray
    delta: np.ndarray
    S0: np.ndarray
    S1: np.ndarray
    prob_swing: float = 0.0
    margin_gain: float = 0.0
    jsd_change: float = 0.0
    cp_t: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    cp_x: np.ndarray = field(default_factory=lambda: np.empty((0, 0), float))
    cp_y_left: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    cp_y_right: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    cp_count: int = 0


@dataclass
class ChuchuConfig:
    mode: str = "auto"
    log_level: int = logging.INFO
    pair_quota: int = 6
    pair_seed_quantile: float = 0.25
    max_pairs_total: int = 300
    secant_iters: int = 2
    final_bisect: int = 8
    distance_metric: str = "l2"
    alpha_change: float = 0.6
    min_pair_margin_end: float = 0.02
    min_logit_gain: float = 0.25
    random_state: int = 0
    prob_swing_weight: float = 0.7
    use_jsd: bool = False
    jsd_weight: float = 0.15


@dataclass
class ChangePointConfig:
    enabled: bool = True
    mode: str = "auto"
    per_record_max_points: int = 32
    only_success: bool = True
    topk_records: Optional[int] = None
    max_candidates: int = 512
    max_bisect_iters: int = 12
    base_samples: int = 128


# =============================================================================
# Change-point helpers
# =============================================================================

def _predict_labels(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return np.argmax(probs, axis=1)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            return (decision > 0).astype(int)
        return np.argmax(decision, axis=1)
    raise ValueError("El modelo no soporta predict/predict_proba/decision_function")


def _points_on_segment(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x0[None, :] + t[:, None] * (x1 - x0)[None, :]


def _unique_sorted(arr: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = np.sort(arr)
    keep = [0]
    for i in range(1, arr.size):
        if abs(arr[i] - arr[keep[-1]]) > tol:
            keep.append(i)
    return arr[keep]


def _iter_sklearn_trees(model: Any):
    if not hasattr(model, "estimators_"):
        return
    estimators = model.estimators_
    try:
        arr = np.array(estimators, dtype=object).flatten()
    except Exception:
        arr = estimators
    for est in arr:
        if hasattr(est, "tree_"):
            yield est.tree_


def _is_tree_ensemble(model: Any) -> bool:
    try:
        for _ in _iter_sklearn_trees(model):
            return True
        return False
    except Exception:
        return False


def _cross_ts_for_tree(tree: Any, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    feature = tree.feature
    thr = tree.threshold
    mask = feature >= 0
    if not np.any(mask):
        return np.empty(0, float)
    feature = feature[mask]
    thr = thr[mask]
    v0 = x0[feature]
    v1 = x1[feature]
    den = v1 - v0
    mask_nz = np.abs(den) > 1e-12
    if not np.any(mask_nz):
        return np.empty(0, float)
    feature = feature[mask_nz]
    v0 = v0[mask_nz]
    v1 = v1[mask_nz]
    thr = thr[mask_nz]
    den = den[mask_nz]
    sign0 = v0 - thr
    sign1 = v1 - thr
    mask_cross = (sign0 * sign1) < 0.0
    if not np.any(mask_cross):
        return np.empty(0, float)
    t = (thr[mask_cross] - v0[mask_cross]) / den[mask_cross]
    mask_01 = (t > 0.0) & (t < 1.0)
    return t[mask_01]


def _treefast_candidates(
    model: Any,
    x0: np.ndarray,
    x1: np.ndarray,
    max_candidates: Optional[int] = None,
) -> np.ndarray:
    ts: List[np.ndarray] = []
    for tree in _iter_sklearn_trees(model):
        ts.append(_cross_ts_for_tree(tree, x0, x1))
    if len(ts) == 0:
        return np.empty(0, float)
    t_all = np.concatenate(ts) if len(ts) > 1 else ts[0]
    if max_candidates is not None and t_all.size > max_candidates:
        idx = np.argsort(np.abs(t_all - 0.5))[:max_candidates]
        t_all = t_all[idx]
    return _unique_sorted(t_all, tol=1e-12)


def _bisection_labels_on_segment(
    model: Any,
    x0: np.ndarray,
    x1: np.ndarray,
    tL: np.ndarray,
    tR: np.ndarray,
    max_iters: int = 22,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tL = np.asarray(tL, float).copy()
    tR = np.asarray(tR, float).copy()
    YL = _predict_labels(model, _points_on_segment(x0, x1, tL + eps))
    YR = _predict_labels(model, _points_on_segment(x0, x1, tR - eps))
    for _ in range(max_iters):
        tm = 0.5 * (tL + tR)
        Ym = _predict_labels(model, _points_on_segment(x0, x1, tm))
        go_left = Ym == YL
        tL = np.where(go_left, tm, tL)
        YL = np.where(go_left, Ym, YL)
        tR = np.where(~go_left, tm, tR)
        YR = np.where(~go_left, Ym, YR)
    t_star = 0.5 * (tL + tR)
    return t_star, YL, YR


def _find_change_points_along_segment_treefast(
    model: Any,
    x0: np.ndarray,
    x1: np.ndarray,
    max_candidates: int = 4096,
    max_bisect_iters: int = 22,
) -> List[Dict[str, Any]]:
    x0 = np.asarray(x0, float)
    x1 = np.asarray(x1, float)
    if np.allclose(x0, x1):
        return []
    t_cand = _treefast_candidates(model, x0, x1, max_candidates=max_candidates)
    if t_cand.size == 0:
        return []
    tB = _unique_sorted(np.concatenate([[0.0], t_cand, [1.0]]), tol=1e-12)
    mids = 0.5 * (tB[:-1] + tB[1:])
    y_mid = _predict_labels(model, _points_on_segment(x0, x1, mids))
    change_idx = np.where(y_mid[1:] != y_mid[:-1])[0]
    if change_idx.size == 0:
        return []
    tL = tB[change_idx]
    tR = tB[change_idx + 1]
    t_star, yL, yR = _bisection_labels_on_segment(
        model, x0, x1, tL, tR, max_iters=max_bisect_iters
    )
    pts = _points_on_segment(x0, x1, t_star)
    out: List[Dict[str, Any]] = []
    for ts, point, yl, yr in zip(t_star, pts, yL, yR):
        out.append({"t": float(ts), "x": point.astype(float), "y_left": int(yl), "y_right": int(yr)})
    return out


def _find_change_points_along_segment_generic(
    model: Any,
    x0: np.ndarray,
    x1: np.ndarray,
    base_samples: int = 64,
    max_bisect_iters: int = 22,
) -> List[Dict[str, Any]]:
    x0 = np.asarray(x0, float)
    x1 = np.asarray(x1, float)
    if np.allclose(x0, x1):
        return []
    t_grid = np.linspace(0.0, 1.0, base_samples + 1)
    mids = 0.5 * (t_grid[:-1] + t_grid[1:])
    y_mid = _predict_labels(model, _points_on_segment(x0, x1, mids))
    change_idx = np.where(y_mid[1:] != y_mid[:-1])[0]
    if change_idx.size == 0:
        return []
    tL = t_grid[change_idx]
    tR = t_grid[change_idx + 1]
    t_star, yL, yR = _bisection_labels_on_segment(
        model, x0, x1, tL, tR, max_iters=max_bisect_iters
    )
    pts = _points_on_segment(x0, x1, t_star)
    out: List[Dict[str, Any]] = []
    for ts, point, yl, yr in zip(t_star, pts, yL, yR):
        out.append({"t": float(ts), "x": point.astype(float), "y_left": int(yl), "y_right": int(yr)})
    return out


def _find_change_points_along_segment(
    model: Any,
    x0: np.ndarray,
    x1: np.ndarray,
    cp: ChangePointConfig,
) -> List[Dict[str, Any]]:
    mode = cp.mode
    if mode == "auto":
        mode = "treefast" if _is_tree_ensemble(model) else "generic"
    if mode == "treefast":
        return _find_change_points_along_segment_treefast(
            model,
            x0,
            x1,
            max_candidates=cp.max_candidates,
            max_bisect_iters=cp.max_bisect_iters,
        )
    if mode == "generic":
        return _find_change_points_along_segment_generic(
            model,
            x0,
            x1,
            base_samples=cp.base_samples,
            max_bisect_iters=cp.max_bisect_iters,
        )
    raise ValueError("ChangePointConfig.mode debe ser 'auto' | 'treefast' | 'generic'")


# =============================================================================
# Core Chuchu implementation (pair balancing + change points)
# =============================================================================

class Chuchu:
    def __init__(self, config: ChuchuConfig, cp_config: Optional[ChangePointConfig] = None):
        self.cfg = config
        self.cp_cfg = cp_config or ChangePointConfig(enabled=False)
        self.logger = _get_logger("Chuchu", config.log_level)
        self.records_: List[DeltaRecord] = []
        self._adaptor: Optional[ScoreAdaptor] = None
        self._X: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def _pair_seeds_by_margin(self, a: int, b: int, quota: int) -> Tuple[np.ndarray, np.ndarray]:
        X = self._X
        P = self._P
        y = self._y
        assert X is not None and P is not None and y is not None
        m_ab = P[:, a] - P[:, b]
        Ia_all = np.where(y == a)[0]
        Ib_all = np.where(y == b)[0]
        if len(Ia_all) == 0 or len(Ib_all) == 0:
            return np.array([], int), np.array([], int)

        qa = np.quantile(m_ab[Ia_all], self.cfg.pair_seed_quantile)
        Ia = Ia_all[np.argsort(m_ab[Ia_all])[: max(1, min(quota * 2, len(Ia_all)))]]
        Ia = Ia[m_ab[Ia] <= qa + 1e-12]

        qb = np.quantile(m_ab[Ib_all], 1.0 - self.cfg.pair_seed_quantile)
        Ib = Ib_all[np.argsort(-m_ab[Ib_all])[: max(1, min(quota * 2, len(Ib_all)))]]
        Ib = Ib[m_ab[Ib] >= qb - 1e-12]

        if len(Ia) == 0:
            Ia = Ia_all[np.argsort(m_ab[Ia_all])[: max(1, min(quota, len(Ia_all)))]]
        if len(Ib) == 0:
            Ib = Ib_all[np.argsort(-m_ab[Ib_all])[: max(1, min(quota, len(Ib_all)))]]
        return Ia[: max(0, quota * 3)], Ib[: max(0, quota * 3)]

    def _pair_candidates_round_robin(
        self, labels: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
        quota = int(self.cfg.pair_quota)
        per_pair_lists: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        rng = np.random.RandomState(self.cfg.random_state)

        for a in labels:
            for b in labels:
                if a == b:
                    continue
                Ia, Ib = self._pair_seeds_by_margin(a, b, quota)
                if len(Ia) == 0 or len(Ib) == 0:
                    continue
                Xa = self._X[Ia]
                Xb = self._X[Ib]
                D = np.linalg.norm(Xa[:, None, :] - Xb[None, :, :], axis=2)
                order_b = np.argmin(D, axis=1)
                pairs: List[Tuple[int, int]] = []
                used_b: set[int] = set()
                for ii, jj in zip(Ia, Ib[order_b]):
                    if jj in used_b:
                        cand = [k for k in Ib if k not in used_b]
                        if not cand:
                            continue
                        jj = int(rng.choice(cand))
                    used_b.add(int(jj))
                    pairs.append((int(ii), int(jj)))
                    if len(pairs) >= quota:
                        break
                per_pair_lists[(a, b)] = pairs

        A_list: List[np.ndarray] = []
        B_list: List[np.ndarray] = []
        yA_list: List[int] = []
        yB_list: List[int] = []
        keys = list(per_pair_lists.keys())
        exhausted: set[Tuple[int, int]] = set()
        total_cap = int(self.cfg.max_pairs_total)
        idx_ptr = {k: 0 for k in keys}

        while len(A_list) < total_cap and len(exhausted) < len(keys):
            for k in keys:
                if k in exhausted:
                    continue
                lst = per_pair_lists[k]
                p = idx_ptr[k]
                if p >= len(lst):
                    exhausted.add(k)
                    continue
                ia, ib = lst[p]
                idx_ptr[k] = p + 1
                A_list.append(self._X[ia])
                B_list.append(self._X[ib])
                yA_list.append(k[0])
                yB_list.append(k[1])
                if len(A_list) >= total_cap:
                    break

        if len(A_list) == 0:
            return [], [], [], []
        return [np.vstack(A_list)], [np.vstack(B_list)], yA_list, yB_list

    def fit(self, X: np.ndarray, model: Any) -> "Chuchu":
        t0 = time.time()
        self._X = np.asarray(X, float)
        self._adaptor = ScoreAdaptor(model, mode=self.cfg.mode)
        self.records_.clear()

        self._P = self._adaptor.scores(self._X)
        self._y = np.argmax(self._P, axis=1)
        labels = sorted(np.unique(self._y).tolist())

        A_batches, B_batches, yA_list, yB_list = self._pair_candidates_round_robin(labels)
        if not A_batches:
            _get_logger("Chuchu", logging.WARNING).warning(
                "No se pudieron generar pares (verifica quotas y datos)."
            )
            return self

        all_records: List[DeltaRecord] = []
        t1 = time.time()
        for A, B in zip(A_batches, B_batches):
            yA = np.asarray(yA_list, int)
            yB = np.asarray(yB_list, int)
            Xstar, ystar, Sstar = batch_false_position_flip(
                self._adaptor,
                A,
                B,
                yA,
                yB,
                iters=self.cfg.secant_iters,
                final_bisect=self.cfg.final_bisect,
            )
            SA = self._adaptor.scores(A)
            cfg = self.cfg

            w_swing = float(getattr(cfg, "prob_swing_weight", 0.7))
            use_jsd = bool(getattr(cfg, "use_jsd", False))
            w_jsd = float(getattr(cfg, "jsd_weight", 0.15)) if use_jsd else 0.0

            for k in range(len(A)):
                x0 = A[k]
                x1 = Xstar[k]
                S0 = SA[k]
                S1 = Sstar[k]
                y0 = int(np.argmax(S0))
                y1 = int(np.argmax(S1))
                dvec = x1 - x0
                dn2 = float(np.linalg.norm(dvec, 2))
                dni = float(np.linalg.norm(dvec, np.inf))

                m1 = float(S1[y0] - S1[y1])

                logit_gain = float(
                    np.log((S1[y1] + 1e-12) / (S1[y0] + 1e-12))
                    - np.log((S0[y1] + 1e-12) / (S0[y0] + 1e-12))
                )

                robust = (
                    (y1 != y0)
                    and (m1 <= -self.cfg.min_pair_margin_end)
                    and (logit_gain >= self.cfg.min_logit_gain)
                )

                drop_a = max(0.0, float(S0[y0] - S1[y0]))
                gain_b = max(0.0, float(S1[y1] - S0[y1]))
                prob_swing = 0.5 * (drop_a + gain_b)

                m0 = float(S0[y0] - S0[y1])
                margin_gain = max(0.0, m0 - m1)

                jsd_val = _jsd(S0, S1) if use_jsd else 0.0

                strength_final_margin = max(0.0, -m1)

                change_mix = (
                    (w_swing * prob_swing)
                    + ((1.0 - w_swing) * strength_final_margin)
                    + (w_jsd * jsd_val)
                )

                rec = DeltaRecord(
                    index_a=-1,
                    index_b=-1,
                    method="pair_rr",
                    success=bool(robust),
                    y0=y0,
                    y1=y1,
                    delta_norm_l2=dn2,
                    delta_norm_linf=dni,
                    score_change=float(change_mix),
                    distance_term=0.0,
                    change_term=0.0,
                    final_score=0.0,
                    time_ms=float((time.time() - t1) * 1000.0 / max(1, len(A))),
                    x0=x0,
                    x1=x1,
                    delta=dvec,
                    S0=S0,
                    S1=S1,
                    prob_swing=float(prob_swing),
                    margin_gain=float(margin_gain),
                    jsd_change=float(jsd_val),
                )
                all_records.append(rec)

        by_pair: Dict[Tuple[int, int], List[int]] = {}
        for idx, record in enumerate(all_records):
            if record.success:
                by_pair.setdefault((record.y0, record.y1), []).append(idx)

        for key, idxs in by_pair.items():
            if self.cfg.distance_metric == "l2":
                dists = np.array([all_records[i].delta_norm_l2 for i in idxs])
            else:
                dists = np.array([all_records[i].delta_norm_linf for i in idxs])
            changes = np.array([all_records[i].score_change for i in idxs])
            d95 = np.percentile(dists, 95) + 1e-12
            c95 = np.percentile(changes, 95) + 1e-12
            d_term = np.clip(dists / d95, 0, 1)
            c_term = np.clip(changes / c95, 0, 1)
            alpha = float(self.cfg.alpha_change)
            final = alpha * c_term + (1 - alpha) * (1 - d_term)
            for ii, dt, ct, fs in zip(idxs, d_term, c_term, final):
                all_records[ii].distance_term = float(dt)
                all_records[ii].change_term = float(ct)
                all_records[ii].final_score = float(fs)

        self.records_ = sorted(all_records, key=lambda r: r.final_score, reverse=True)

        if self.cp_cfg.enabled:
            self._compute_change_points_for_records(model)

        self.logger.info(
            "Chuchu listo. Pares=%d | tiempo=%.1f ms",
            len(self.records_),
            (time.time() - t0) * 1000.0,
        )
        return self

    def _compute_change_points_for_records(self, model: Any) -> None:
        cp = self.cp_cfg
        recs = self.records_
        if cp.only_success:
            recs = [r for r in recs if r.success]
        if cp.topk_records is not None and cp.topk_records > 0:
            recs = recs[: cp.topk_records]

        for record in recs:
            pts = _find_change_points_along_segment(model, record.x0, record.x1, cp)
            if len(pts) == 0:
                record.cp_t = np.empty(0, float)
                record.cp_x = np.empty((0, record.x0.size), float)
                record.cp_y_left = np.empty(0, int)
                record.cp_y_right = np.empty(0, int)
                record.cp_count = 0
                continue
            if cp.per_record_max_points is not None and len(pts) > cp.per_record_max_points:
                idx = np.linspace(0, len(pts) - 1, cp.per_record_max_points).round().astype(int)
                pts = [pts[i] for i in np.unique(idx)]
            record.cp_t = np.array([p["t"] for p in pts], float)
            record.cp_x = np.vstack([p["x"] for p in pts]).astype(float)
            record.cp_y_left = np.array([p["y_left"] for p in pts], int)
            record.cp_y_right = np.array([p["y_right"] for p in pts], int)
            record.cp_count = int(len(pts))

    def results(self) -> List[DeltaRecord]:
        return self.records_

    def topk(self, k: int = 10) -> List[DeltaRecord]:
        return self.records_[: max(0, int(k))]

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records_]

    @property
    def X_(self):
        return self._X

    @property
    def model_(self):
        return None if self._adaptor is None else self._adaptor.model


@dataclass
class DeltaRecordLite:
    y0: int
    y1: int
    x0: np.ndarray
    x1: np.ndarray
    cp_x: np.ndarray
    cp_count: int


def _unique_rows(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if a.size == 0:
        return a.reshape(0, a.shape[-1] if a.ndim == 2 else 0)
    r = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
    _, idx = np.unique(r, axis=0, return_index=True)
    return a[np.sort(idx)]


def _stack_or_empty(lst: List[np.ndarray]) -> np.ndarray:
    return np.vstack(lst) if len(lst) else np.empty((0, 0), float)


class PCA3D:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        Z = X - self.mean_
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        self.components_ = Vt[:3, :]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mean_
        return Z @ self.components_.T


def build_sets_from_records(
    records: Iterable,
) -> Tuple[
    Dict[int, np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
]:
    interior: Dict[int, List[np.ndarray]] = {}
    frontier: Dict[Tuple[int, int], List[np.ndarray]] = {}
    directions: Dict[Tuple[int, int], List[np.ndarray]] = {}

    for r in records:
        y0, y1 = int(r.y0), int(r.y1)
        x0 = np.asarray(r.x0, float).reshape(-1)
        x1 = np.asarray(r.x1, float).reshape(-1)

        if y0 == y1:
            interior.setdefault(y0, []).append(x0)
            continue

        if getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
            F = np.asarray(r.cp_x, float)
        else:
            F = x1.reshape(1, -1)

        frontier.setdefault((y0, y1), []).append(F)
        directions.setdefault((y0, y1), []).append(F - x0.reshape(1, -1))

    interior_by_class = {c: _unique_rows(np.vstack(v)) for c, v in interior.items()}
    frontier_by_pair = {k: _unique_rows(np.vstack(v)) for k, v in frontier.items()}
    dir_by_pair = {k: np.vstack(v) for k, v in directions.items()}
    return interior_by_class, frontier_by_pair, dir_by_pair


def frontier_by_class(frontier_by_pair: Dict[Tuple[int, int], np.ndarray]) -> Dict[int, np.ndarray]:
    from collections import defaultdict

    by_class = defaultdict(list)
    for (a, b), P in frontier_by_pair.items():
        by_class[a].append(P)
        by_class[b].append(P)
    return {c: _unique_rows(np.vstack(v)) for c, v in by_class.items() if len(v)}


def choose_dims_topvar(
    frontier_by_pair: Dict[Tuple[int, int], np.ndarray],
    k: int = 3,
) -> Tuple[List[int], np.ndarray]:
    if not frontier_by_pair:
        return list(range(k)), np.eye(k, dtype=float)
    all_frontier = np.vstack([P for P in frontier_by_pair.values() if P.size > 0])
    var = np.var(all_frontier, axis=0)
    dims = np.argsort(var)[-k:]
    return dims.tolist(), var


def project_all_to_3d(
    interior_by_class: Dict[int, np.ndarray],
    frontier_by_pair: Dict[Tuple[int, int], np.ndarray],
    method: str = "pca",
    dims: Optional[Tuple[int, int, int]] = None,
):
    stacks = [v for v in interior_by_class.values()] + [v for v in frontier_by_pair.values()]
    Xall = np.vstack([v for v in stacks if v.size > 0]) if stacks else np.empty((0, 0))

    if method == "pca":
        pca = PCA3D().fit(Xall)
        proj = lambda A: pca.transform(A)
        info = {"method": "pca", "mean": pca.mean_, "components": pca.components_}
    elif method == "topvar":
        dims_auto, _ = choose_dims_topvar(frontier_by_pair, k=3)
        dims_use = tuple(dims_auto)
        proj = lambda A: A[:, dims_use]
        info = {"method": "topvar", "dims": dims_use}
    elif method == "dims":
        assert dims is not None and len(dims) == 3, "Para method='dims' debes pasar dims=(i,j,k)."
        dims_use = tuple(int(i) for i in dims)
        proj = lambda A: A[:, dims_use]
        info = {"method": "dims", "dims": dims_use}
    else:
        raise ValueError("method debe ser 'pca' | 'topvar' | 'dims'")

    interior3d = {c: proj(A) for c, A in interior_by_class.items()}
    frontier3d = {k: proj(A) for k, A in frontier_by_pair.items()}
    by_class3d = frontier_by_class(frontier3d)

    return {
        "interior3d": interior3d,
        "frontier3d_by_pair": frontier3d,
        "frontier3d_by_class": by_class3d,
        "projection_info": info,
    }


def convex_hull_faces(points3d: np.ndarray):
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return None
    if points3d.shape[0] < 4:
        return None
    hull = ConvexHull(points3d)
    return hull.simplices


# =============================================================================
# Weighted frontier fitting helpers
# =============================================================================

def build_weighted_frontier(
    records: Iterable,
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",
    gamma: float = 2.0,
    temp: float = 0.15,
    sigmoid_center: Optional[float] = None,
    density_k: Optional[int] = 8,
) -> Tuple[
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
    Dict[Tuple[int, int], np.ndarray],
]:
    from collections import defaultdict

    Ftmp, Btmp, Stmp = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        if success_only and not bool(getattr(r, "success", True)):
            continue
        a, b = int(r.y0), int(r.y1)
        if a == b:
            continue
        score = float(getattr(r, "final_score", 1.0))
        x0 = np.asarray(r.x0, float).reshape(1, -1)
        if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
            F = np.asarray(r.cp_x, float)
        else:
            F = np.asarray(r.x1, float).reshape(1, -1)
        m = F.shape[0]
        Ftmp[(a, b)].append(F)
        Btmp[(a, b)].append(np.repeat(x0, m, axis=0))
        Stmp[(a, b)].append(np.full(m, score, float))

    F_by = {k: np.vstack(v) for k, v in Ftmp.items()}
    B_by = {k: np.vstack(v) for k, v in Btmp.items()}
    S_by = {k: np.concatenate(v) for k, v in Stmp.items()}

    W_by = {}
    for k in F_by.keys():
        s = S_by[k].copy().astype(float)
        if s.size == 0:
            W_by[k] = s
            continue

        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0) ** float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            w = 1.0 / (1.0 + np.exp(-(s - cen) / (temp if temp > 1e-9 else 1e-9)))
        elif weight_map == "softmax":
            t = max(temp, 1e-9)
            z = (s - s.max()) / t
            w = np.exp(z)
        else:
            w = s

        if density_k is not None and density_k > 0 and F_by[k].shape[0] > density_k:
            P = F_by[k]
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, : density_k + 1]
            dens = D[np.arange(D.shape[0])[:, None], idx].mean(axis=1) + 1e-9
            w = w / dens

        W_by[k] = w / (w.sum() + 1e-12)

    return F_by, B_by, W_by


def fit_tls_plane_weighted(F: np.ndarray, w: np.ndarray):
    F = np.asarray(F, float)
    w = np.asarray(w, float).reshape(-1)
    assert F.shape[0] >= 3 and F.shape[0] == w.size
    w = w / (w.sum() + 1e-12)
    mu = (w[:, None] * F).sum(axis=0)
    Z = F - mu
    Sw = Z.T @ (w[:, None] * Z)
    evals, evecs = np.linalg.eigh(Sw)
    n = evecs[:, np.argmin(evals)]
    b = -float(n @ mu)
    return n.astype(float), float(b), mu.astype(float)


def compute_frontier_planes_weighted(
    records: Iterable,
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",
    gamma: float = 2.0,
    temp: float = 0.15,
    density_k: Optional[int] = 8,
    orient_with_bases: bool = True,
    eps_orient: float = 1e-3,
):
    F_by, B_by, W_by = build_weighted_frontier(
        records,
        prefer_cp,
        success_only,
        weight_map,
        gamma,
        temp,
        None,
        density_k,
    )
    planes = {}
    for key, F in F_by.items():
        w = W_by[key]
        if F.shape[0] < 3:
            continue
        n, b, mu = fit_tls_plane_weighted(F, w)
        if orient_with_bases and key in B_by:
            B = B_by[key]
            U = F - B
            U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
            side_plus = (F + eps_orient * U) @ n + b
            side_minus = (F - eps_orient * U) @ n + b
            if np.average(side_plus, weights=w) < np.average(side_minus, weights=w):
                n, b = -n, -b
            b = -float(n @ mu)
        planes[key] = {
            "n": n,
            "b": b,
            "mu": mu,
            "count": int(F.shape[0]),
            "points": F,
            "weights": w,
        }
    return planes


def fit_quadrics_from_records_weighted(
    records: Iterable,
    mode: str = "svd",
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",
    gamma: float = 2.0,
    temp: float = 0.15,
    density_k: Optional[int] = 8,
    eps: float = 1e-3,
    C: float = 10.0,
):
    def _standardize(X: np.ndarray):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        return (X - mu) / sd, mu, sd

    def _destandardize(Qz, rz, cz, mu, sd):
        D = np.diag(1.0 / sd)
        Qx = D @ Qz @ D
        r0 = D @ rz
        rx = r0 - 2.0 * (Qx @ mu)
        cx = float(mu.T @ Qx @ mu - r0.T @ mu + cz)
        return Qx, rx, cx

    def _unpack(theta, idx, d):
        Qz = np.zeros((d, d))
        for i in range(d):
            Qz[i, i] = theta[idx["diag"][i]]
        for k, (i, j) in enumerate(idx["pairs"]):
            Qz[i, j] = Qz[j, i] = 0.5 * theta[idx["off"][k]]
        rz = theta[idx["lin"][0] : idx["lin"][0] + d]
        cz = theta[idx["c"]]
        return Qz, rz, float(cz)

    def _poly2(Z: np.ndarray):
        n, d = Z.shape
        diag = [Z[:, i] ** 2 for i in range(d)]
        off = []
        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                off.append(2.0 * Z[:, i] * Z[:, j])
                pairs.append((i, j))
        lin = [Z[:, i] for i in range(d)]
        Phi = np.column_stack(diag + off + lin + [np.ones(n)])
        idx = {
            "diag": list(range(d)),
            "off": list(range(d, d + len(off))),
            "lin": list(range(d + len(off), d + len(off) + d)),
            "c": d + len(off) + d,
            "pairs": pairs,
        }
        return Phi, idx

    F_by, B_by, W_by = build_weighted_frontier(
        records,
        prefer_cp,
        success_only,
        weight_map,
        gamma,
        temp,
        None,
        density_k,
    )
    models = {}
    for key, F in F_by.items():
        w = W_by[key]
        if F.shape[0] < 3:
            continue

        if mode == "svd":
            Z, mu, sd = _standardize(F)
            Phi, idx = _poly2(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:, None]
            _, S, Vt = np.linalg.svd(Phi_w, full_matrices=False)
            theta = Vt[-1, :] / (np.linalg.norm(Vt[-1, :]) + 1e-12)
            d = F.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            models[key] = {
                "Q": Qx,
                "r": rx,
                "c": cx,
                "mode": "svd_w",
                "cond": S[-1] / (S[0] + 1e-12),
                "weights": w,
            }

        elif mode == "logistic":
            B = B_by[key]
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)
            eps_r = eps * (0.5 + 0.5 * (1.0 - (w / (w.max() + 1e-12))))
            Xa = F + (eps_r[:, None] * Uu)
            Xb = F - (eps_r[:, None] * Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack([np.ones(Xa.shape[0], int), -np.ones(Xb.shape[0], int)])
            w_lr = np.hstack([w, w])
            Z, mu, sd = _standardize(X)
            Phi, idx = _poly2(Z)
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=C, penalty="l2", max_iter=800)
            clf.fit(Phi, ybin, sample_weight=w_lr)
            wcoef = clf.coef_.reshape(-1)
            b0 = clf.intercept_[0]
            theta = np.r_[wcoef, b0]
            d = X.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            models[key] = {
                "Q": Qx,
                "r": rx,
                "c": cx,
                "mode": "logistic_w",
                "weights": w,
            }
        else:
            raise ValueError("mode debe ser 'svd' o 'logistic'")
    return models


def fit_cubic_from_records_weighted(
    records: Iterable,
    *,
    prefer_cp: bool = True,
    success_only: bool = True,
    mode: str = "svd",
    weight_map: str = "power",
    gamma: float = 2.0,
    temp: float = 0.15,
    sigmoid_center: float = None,
    density_k: int = 8,
    eps: float = 1e-3,
    C: float = 5.0,
):
    from collections import defaultdict

    def _unique_rows_tol(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        if a.size == 0:
            return a.reshape(0, a.shape[-1] if a.ndim == 2 else 0)
        q = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return a[np.sort(idx)]

    def _stack_frontier(records, prefer_cp=True, success_only=True):
        F_by, B_by, S_by = defaultdict(list), defaultdict(list), defaultdict(list)
        d = None
        for r in records:
            if success_only and not bool(getattr(r, "success", True)):
                continue
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            score = float(getattr(r, "final_score", 1.0))
            x0 = np.asarray(r.x0, float).reshape(1, -1)
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
                F = np.asarray(r.cp_x, float)
            else:
                F = np.asarray(r.x1, float).reshape(1, -1)
            if d is None:
                d = F.shape[1]
            m = F.shape[0]
            F_by[(a, b)].append(F)
            B_by[(a, b)].append(np.repeat(x0, m, axis=0))
            S_by[(a, b)].append(np.full(m, score, float))
        F_by = {k: _unique_rows_tol(np.vstack(v)) for k, v in F_by.items()}
        B_by = {k: np.vstack(v) for k, v in B_by.items()}
        S_by = {k: np.concatenate(v) for k, v in S_by.items()}
        return F_by, B_by, S_by, (0 if d is None else d)

    def _weights_from_scores_per_pair(scores: np.ndarray, P: np.ndarray):
        s = scores.copy().astype(float)
        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0) ** float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            w = 1.0 / (1.0 + np.exp(-(s - cen) / (temp if temp > 1e-9 else 1e-9)))
        elif weight_map == "softmax":
            t = max(temp, 1e-9)
            z = (s - s.max()) / t
            w = np.exp(z)
        else:
            w = s
        if density_k and density_k > 0 and P.shape[0] > density_k:
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, : density_k + 1]
            dens = D[np.arange(D.shape[0])[:, None], idx].mean(axis=1) + 1e-9
            w = w / dens
        return w / (w.sum() + 1e-12)

    def _standardize(X: np.ndarray):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Z = (X - mu) / sd
        return Z, mu, sd

    def _poly3_features(Z: np.ndarray):
        Z = np.asarray(Z, float)
        n, d = Z.shape
        cols = []
        catalog = {
            "lin": [],
            "quad_diag": [],
            "quad_off": [],
            "cubic_diag": [],
            "cubic_mixed2": [],
            "cubic_tri": [],
        }
        for i in range(d):
            cols.append(Z[:, i])
            catalog["lin"].append(("x", (i,)))
        for i in range(d):
            cols.append(Z[:, i] ** 2)
            catalog["quad_diag"].append(("x2", (i,)))
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Z[:, i] * Z[:, j])
                catalog["quad_off"].append(("xixj", (i, j)))
        for i in range(d):
            cols.append(Z[:, i] ** 3)
            catalog["cubic_diag"].append(("x3", (i,)))
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                cols.append((Z[:, i] ** 2) * Z[:, j])
                catalog["cubic_mixed2"].append(("xi2xj", (i, j)))
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(Z[:, i] * Z[:, j] * Z[:, k])
                    catalog["cubic_tri"].append(("xixjxk", (i, j, k)))
        cols.append(np.ones(n))
        Phi = np.column_stack(cols)
        return Phi, catalog

    F_by, B_by, S_by, d = _stack_frontier(records, prefer_cp, success_only)
    out = {}
    if d == 0:
        return out

    for key, F in F_by.items():
        if F.shape[0] < 5:
            continue
        B = B_by[key]
        scores = S_by[key]
        w = _weights_from_scores_per_pair(scores, F)

        if mode == "svd":
            Z, mu, sd = _standardize(F)
            Phi, catalog = _poly3_features(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:, None]
            _, S, Vt = np.linalg.svd(Phi_w, full_matrices=False)
            wvec = Vt[-1, :]
            wvec = wvec / (np.linalg.norm(wvec) + 1e-12)
            out[key] = {
                "w": wvec,
                "mu": mu,
                "sd": sd,
                "catalog": catalog,
                "mode": "svd_w",
                "cond": S[-1] / (S[0] + 1e-12),
                "weights": w,
            }

        elif mode == "logistic":
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)
            eps_r = eps * (0.5 + 0.5 * (1.0 - (w / (w.max() + 1e-12))))
            Xa = F + (eps_r[:, None] * Uu)
            Xb = F - (eps_r[:, None] * Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack([np.ones(Xa.shape[0], dtype=int), np.zeros(Xb.shape[0], dtype=int)])
            w_lr = np.hstack([w, w])

            Z, mu, sd = _standardize(X)
            Phi, catalog = _poly3_features(Z)

            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=C, penalty="l2", fit_intercept=False, max_iter=1000)
            clf.fit(Phi, ybin, sample_weight=w_lr)
            wcoef = clf.coef_.reshape(-1)
            out[key] = {
                "w": wcoef,
                "mu": mu,
                "sd": sd,
                "catalog": catalog,
                "mode": "logistic_w",
                "weights": w,
            }

        else:
            raise ValueError("mode debe ser 'svd' o 'logistic'")

    return out


def _poly3_features_eval(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, float)
    n, d = Z.shape
    cols = []
    for i in range(d):
        cols.append(Z[:, i])
    for i in range(d):
        cols.append(Z[:, i] ** 2)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(Z[:, i] * Z[:, j])
    for i in range(d):
        cols.append(Z[:, i] ** 3)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            cols.append((Z[:, i] ** 2) * Z[:, j])
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                cols.append(Z[:, i] * Z[:, j] * Z[:, k])
    cols.append(np.ones(n))
    return np.column_stack(cols) if cols else np.ones((n, 1))


def _cubic_g_from_model(model: Dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    mu = np.asarray(model["mu"], float)
    sd = np.asarray(model["sd"], float)
    Z = (X - mu) / sd
    Phi = _poly3_features_eval(Z)
    w = np.asarray(model["w"], float).reshape(-1)
    return Phi @ w


def eval_cubic_models(
    models: Dict[Tuple[int, int], Dict],
    X: np.ndarray,
    keys: Optional[Iterable[Tuple[int, int]]] = None,
    return_proba_when_logistic: bool = True,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    X = np.asarray(X, float)
    pairs = list(models.keys()) if keys is None else list(keys)
    out = {}
    for key in pairs:
        mdl = models[key]
        g = _cubic_g_from_model(mdl, X)
        p = None
        if return_proba_when_logistic and mdl.get("mode") == "logistic_w":
            p = 1.0 / (1.0 + np.exp(-g))
        out[key] = {"g": g, "p": p, "mode": mdl.get("mode", "")}
    return out


def find_segment_crossings_cubic(
    models: Dict[Tuple[int, int], Dict],
    key: Tuple[int, int],
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    n_sub: int = 128,
    near_zero: float = 1e-6,
    tol: float = 1e-7,
    max_iter: int = 50,
) -> List[Dict]:
    assert key in models, f"Par {key} no encontrado en models."
    mdl = models[key]
    x0 = np.asarray(x0, float).reshape(-1)
    x1 = np.asarray(x1, float).reshape(-1)
    d = x0.size
    assert x1.size == d

    def gx_of_t(t: np.ndarray) -> np.ndarray:
        X = x0[None, :] + t[:, None] * (x1 - x0)[None, :]
        return _cubic_g_from_model(mdl, X)

    ts = np.linspace(0.0, 1.0, n_sub + 1)
    gs = gx_of_t(ts)

    brackets = []
    nz_hits = [float(t) for t, g in zip(ts, gs) if abs(g) <= near_zero]
    for i in range(n_sub):
        gL, gR = gs[i], gs[i + 1]
        if np.sign(gL) == 0 and abs(gL) <= near_zero:
            continue
        if np.sign(gR) == 0 and abs(gR) <= near_zero:
            continue
        if gL == 0.0 or gR == 0.0:
            continue
        if np.sign(gL) != np.sign(gR):
            brackets.append((ts[i], ts[i + 1]))

    roots_t: List[float] = []

    def _append_unique(tcand: float, bag: List[float], atol: float = 1e-6):
        for ti in bag:
            if abs(ti - tcand) <= atol:
                return
        bag.append(float(tcand))

    for t_hit in nz_hits:
        _append_unique(t_hit, roots_t)

    for tL, tR in brackets:
        gL, gR = gx_of_t(np.array([tL, tR]))
        if np.sign(gL) == np.sign(gR):
            continue
        a, b = float(tL), float(tR)
        fa, fb = float(gL), float(gR)
        for _ in range(max_iter):
            c = 0.5 * (a + b)
            fc = float(gx_of_t(np.array([c]))[0])
            if abs(fc) <= tol or (b - a) <= tol:
                _append_unique(c, roots_t)
                break
            if np.sign(fa) * np.sign(fc) <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        else:
            ca, cb = abs(fa), abs(fb)
            c = a if ca < cb else b
            _append_unique(c, roots_t)

    roots_t.sort()
    out = []
    for t in roots_t:
        x = x0 + t * (x1 - x0)
        g = float(_cubic_g_from_model(mdl, x.reshape(1, -1))[0])
        out.append({"t": t, "x": x, "g": g})
    return out


# =============================================================================
# Interactive plotting (Plotly)
# =============================================================================

from typing import Dict as _Dict, Iterable as _Iterable, List as _List, Optional as _Optional, Tuple as _Tuple, Union as _Union


def plot_frontiers_implicit_interactive(
    records: _Iterable,
    X: np.ndarray,
    y: _Optional[np.ndarray] = None,
    *,
    planes: _Optional[_Dict[Tuple[int, int], Dict[str, np.ndarray]]] = None,
    quadrics: _Optional[_Dict[Tuple[int, int], Dict[str, np.ndarray]]] = None,
    cubic_models: _Optional[_Dict[Tuple[int, int], Dict[str, np.ndarray]]] = None,
    dims: Tuple[int, ...] = (0, 1, 2),
    dims_options: _Optional[_List[Tuple[int, ...]]] = None,
    feature_names: _Optional[_List[str]] = None,
    prefer_cp: bool = True,
    success_only: bool = True,
    show_X: _Optional[bool] = None,
    show_frontier: _Optional[bool] = None,
    show_planes: _Optional[bool] = None,
    show_quadrics: _Optional[bool] = None,
    show_cubics: _Optional[bool] = None,
    show_arrows: _Optional[bool] = None,
    detail: str = "auto",
    decimate_X: _Optional[int] = None,
    decimate_frontier: _Optional[int] = None,
    arrows_per_pair: _Optional[int] = None,
    arrow_scale: float = 1.0,
    grid_res_2d: _Optional[int] = None,
    grid_res_3d: _Optional[int] = None,
    quadric_alpha: float = 0.28,
    extend: _Union[float, Tuple[float, ...]] = 1.0,
    clamp_extend_to_X: bool = True,
    plane_mode: str = "fit_dims",
    slice_filter_tol: _Optional[float] = None,
    title: str = "ChuChu — Interiores, Fronteras y Superficies (auto)",
):
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio

    try:
        import google.colab  # type: ignore
        pio.renderers.default = "colab"
    except Exception:
        pass

    def _axis_label(idx: int) -> str:
        if feature_names and 0 <= idx < len(feature_names):
            return feature_names[idx]
        return f"x{idx}"

    def _pair_means(full_F: np.ndarray) -> np.ndarray:
        return full_F.mean(axis=0)

    def _g_quadric_eval(P: np.ndarray, Q: np.ndarray, r: np.ndarray, c: float) -> np.ndarray:
        return np.einsum("ni,ij,nj->n", P, Q, P) + P @ r + c

    def _g_cubic_eval(P: np.ndarray, model: dict) -> np.ndarray:
        Z = (P - model["mu"]) / model["sd"]
        n, d = Z.shape
        cols = []
        for i in range(d):
            cols.append(Z[:, i])
        for i in range(d):
            cols.append(Z[:, i] ** 2)
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Z[:, i] * Z[:, j])
        for i in range(d):
            cols.append(Z[:, i] ** 3)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                cols.append((Z[:, i] ** 2) * Z[:, j])
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(Z[:, i] * Z[:, j] * Z[:, k])
        cols.append(np.ones(n))
        Phi = np.column_stack(cols)
        w = model["w"]
        if w.shape[0] == Phi.shape[1] - 1:
            w = np.r_[w, 0.0]
        return Phi @ w

    def _box_from_points(P: np.ndarray, pad_ratio: float = 0.07):
        lo = P.min(axis=0)
        hi = P.max(axis=0)
        pad = pad_ratio * (hi - lo + 1e-12)
        return lo - pad, hi + pad

    def _apply_extend(lo: np.ndarray, hi: np.ndarray, dims_len: int, loX: np.ndarray, hiX: np.ndarray):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if isinstance(extend, (tuple, list, np.ndarray)):
            fac = np.asarray(extend, float)
            if fac.size not in (1, dims_len):
                raise ValueError("'extend' debe ser escalar o de longitud {dims_len}")
            if fac.size == 1:
                fac = np.repeat(fac, dims_len)
        else:
            fac = np.repeat(float(extend), dims_len)
        new_lo = mid - half * fac
        new_hi = mid + half * fac
        if clamp_extend_to_X:
            new_lo = np.maximum(new_lo, loX)
            new_hi = np.minimum(new_hi, hiX)
            mask = new_lo > new_hi
            if np.any(mask):
                fix = 0.5 * (new_lo[mask] + new_hi[mask])
                new_lo[mask] = fix
                new_hi[mask] = fix
        return new_lo, new_hi

    def _grid_points_2d(lo2: np.ndarray, hi2: np.ndarray, res: int):
        xs = np.linspace(lo2[0], hi2[0], res)
        ys = np.linspace(lo2[1], hi2[1], res)
        Xg, Yg = np.meshgrid(xs, ys)
        return Xg, Yg

    def _make_full_points_from_2d_grid(Xg, Yg, template: np.ndarray, dims_opt: Tuple[int, ...]):
        n_tot = Xg.size
        P = np.repeat(template.reshape(1, -1), n_tot, axis=0)
        P[:, dims_opt[0]] = Xg.reshape(-1)
        P[:, dims_opt[1]] = Yg.reshape(-1)
        return P

    def _grid_points_3d(lo3: np.ndarray, hi3: np.ndarray, res: int):
        xs = np.linspace(lo3[0], hi3[0], res)
        ys = np.linspace(lo3[1], hi3[1], res)
        zs = np.linspace(lo3[2], hi3[2], res)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="xy")
        return Xg, Yg, Zg

    def _make_full_points_from_3d_grid(Xg, Yg, Zg, template: np.ndarray, dims_opt: Tuple[int, ...]):
        n_tot = Xg.size
        P = np.repeat(template.reshape(1, -1), n_tot, axis=0)
        P[:, dims_opt[0]] = Xg.reshape(-1)
        P[:, dims_opt[1]] = Yg.reshape(-1)
        P[:, dims_opt[2]] = Zg.reshape(-1)
        return P

    if "compute_frontier_planes" not in globals():
        def compute_frontier_planes(*args, **kwargs):
            return {}

    def _stack_frontier_sets_from_records(records, prefer_cp=True, success_only=True):
        from collections import defaultdict

        F_by, B_by = defaultdict(list), defaultdict(list)
        d = None
        for r in records:
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            if success_only and not bool(getattr(r, "success", True)):
                continue
            x0 = np.asarray(r.x0, float).reshape(1, -1)
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
                F = np.asarray(r.cp_x, float)
            else:
                F = np.asarray(r.x1, float).reshape(1, -1)
            if d is None:
                d = F.shape[1]
            m = F.shape[0]
            F_by[(a, b)].append(F)
            B_by[(a, b)].append(np.repeat(x0, m, axis=0))

        def _unique_rows_tol(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
            if a.size == 0:
                return a.reshape(0, a.shape[-1] if a.ndim == 2 else 0)
            q = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
            _, idx = np.unique(q, axis=0, return_index=True)
            return a[np.sort(idx)]

        F_by = {k: _unique_rows_tol(np.vstack(v)) for k, v in F_by.items()}
        B_by = {k: np.vstack(v) for k, v in B_by.items()}
        return F_by, B_by, (0 if d is None else d)

    def _restrict_plane_to_dims(n: np.ndarray, b: float, mu: np.ndarray, dims_sel: Tuple[int, ...]):
        d = n.size
        dims_sel = tuple(int(i) for i in dims_sel)
        other = [j for j in range(d) if j not in dims_sel]
        b_eff = b + float(np.dot(n[other], mu[other])) if other else b
        n_sub = n[list(dims_sel)]
        return n_sub.astype(float), float(b_eff)

    X = np.asarray(X, float)
    d_total = X.shape[1]
    if len(dims) not in (2, 3):
        raise AssertionError("dims debe tener longitud 2 o 3")
    dims = tuple(int(i) for i in dims)
    for i in dims:
        if not 0 <= i < d_total:
            raise AssertionError(f"Índice dims fuera de rango: {i}")

    frontier_by_pair, bases_by_pair, d_detect = _stack_frontier_sets_from_records(records, prefer_cp, success_only)
    if d_detect == 0:
        raise ValueError("No hay puntos de frontera para graficar.")
    pair_keys = sorted(frontier_by_pair.keys())

    if planes is None:
        try:
            planes = compute_frontier_planes(records, prefer_cp=prefer_cp, success_only=success_only)  # type: ignore
        except Exception:
            planes = {}

    pair_templates = {p: _pair_means(frontier_by_pair[p]) for p in pair_keys}

    class_colors: Dict[int, str] = {}
    if (y is not None) and (len(y) == X.shape[0]):
        classes = np.unique(y)
        pal = px.colors.qualitative.Plotly
        for k, c in enumerate(classes):
            class_colors[int(c)] = pal[k % len(pal)]
    else:
        classes = np.array([0])
        class_colors[0] = "rgba(130,130,130,0.70)"
        y = np.zeros(X.shape[0], dtype=int)

    if dims_options is None:
        dims_options = [dims]
    else:
        dims_options = [tuple(map(int, opt)) for opt in dims_options if len(opt) in (2, 3)]

    def _resolve_detail(detail: str, nX: int, n_pairs: int, n_frontier_tot: int):
        if detail not in {"auto", "fast", "balanced", "high"}:
            detail = "balanced"
        if detail == "auto":
            load = nX + n_frontier_tot + 2000 * n_pairs
            detail = "fast" if load > 50_000 else "balanced"
        if detail == "fast":
            return dict(grid2d=120, grid3d=16, decX=4, decF=4, arrows=20)
        if detail == "balanced":
            return dict(grid2d=200, grid3d=28, decX=3, decF=3, arrows=40)
        return dict(grid2d=300, grid3d=40, decX=2, decF=2, arrows=60)

    n_frontier_tot = sum(frontier_by_pair[p].shape[0] for p in pair_keys)
    preset = _resolve_detail(detail, X.shape[0], len(pair_keys), n_frontier_tot)

    if grid_res_2d is None:
        grid_res_2d = preset["grid2d"]
    if grid_res_3d is None:
        grid_res_3d = preset["grid3d"]
    if decimate_X is None:
        decimate_X = preset["decX"]
    if decimate_frontier is None:
        decimate_frontier = preset["decF"]
    if arrows_per_pair is None:
        arrows_per_pair = preset["arrows"]

    if show_X is None:
        show_X = True
    if show_frontier is None:
        show_frontier = True
    if show_planes is None:
        show_planes = bool(planes)
    if show_quadrics is None:
        show_quadrics = bool(quadrics)
    if show_cubics is None:
        show_cubics = bool(cubic_models)
    if show_arrows is None:
        have_bases = any(k in bases_by_pair and bases_by_pair[k].size > 0 for k in pair_keys)
        show_arrows = have_bases and arrows_per_pair > 0

    all_traces = []
    all_vis_masks = []

    for opt_i, dims_opt in enumerate(dims_options):
        is_3d = len(dims_opt) == 3
        vis_here = []

        Xopt = X[:, dims_opt]
        loX = Xopt.min(axis=0)
        hiX = Xopt.max(axis=0)

        if show_X:
            idx_all = np.arange(X.shape[0])
            if isinstance(decimate_X, int) and decimate_X >= 2:
                idx_all = idx_all[::decimate_X]
            for c in classes:
                sel = idx_all[y[idx_all] == c]
                if sel.size == 0:
                    continue
                P = X[sel][:, dims_opt]
                if is_3d:
                    tr = go.Scatter3d(
                        x=P[:, 0],
                        y=P[:, 1],
                        z=P[:, 2],
                        mode="markers",
                        name=f"Clase {c} (X)",
                        legendgroup=f"class-{c}",
                        marker=dict(size=3, opacity=0.55, color=class_colors[int(c)]),
                        hovertemplate=f"Clase {int(c)}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}",
                    )
                else:
                    tr = go.Scatter(
                        x=P[:, 0],
                        y=P[:, 1],
                        mode="markers",
                        name=f"Clase {c} (X)",
                        legendgroup=f"class-{c}",
                        marker=dict(size=5, opacity=0.55, color=class_colors[int(c)]),
                        hovertemplate=f"Clase {int(c)}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}",
                    )
                all_traces.append(tr)
                vis_here.append(True)

        if show_frontier:
            for p in pair_keys:
                F = frontier_by_pair[p]
                if F.size == 0:
                    continue
                F_use = F
                if (
                    plane_mode == "slice"
                    and slice_filter_tol is not None
                    and planes
                    and p in planes
                ):
                    meta = planes[p]
                    other = [j for j in range(d_total) if j not in dims_opt]
                    if other:
                        mu_other = np.asarray(meta["mu"], float)[other]
                        mask = (
                            np.abs(F[:, other] - mu_other).mean(axis=1)
                            <= float(slice_filter_tol)
                        )
                        if np.any(mask):
                            F_use = F[mask]
                if F_use.size == 0:
                    continue
                if isinstance(decimate_frontier, int) and decimate_frontier >= 2:
                    F_use = F_use[::decimate_frontier]

                Qp = F_use[:, dims_opt]
                color_pair = "rgba(30,30,30,0.85)"
                outline = "rgba(0,0,0,0.6)"
                if is_3d:
                    tr = go.Scatter3d(
                        x=Qp[:, 0],
                        y=Qp[:, 1],
                        z=Qp[:, 2],
                        mode="markers",
                        name=f"Frontera {p}",
                        legendgroup=f"pair-{p}",
                        marker=dict(size=4, color=color_pair, line=dict(width=1, color=outline)),
                        hovertemplate=f"{p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}",
                    )
                else:
                    tr = go.Scatter(
                        x=Qp[:, 0],
                        y=Qp[:, 1],
                        mode="markers",
                        name=f"Frontera {p}",
                        legendgroup=f"pair-{p}",
                        marker=dict(size=6, color=color_pair, line=dict(width=1, color=outline)),
                        hovertemplate=f"{p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}",
                    )
                all_traces.append(tr)
                vis_here.append(True)

        if show_planes and planes:
            for p in pair_keys:
                meta = planes.get(p)
                if not meta:
                    continue
                n = np.asarray(meta["n"], float)
                b0 = float(meta["b"])
                mu = np.asarray(meta["mu"], float)
                n_sub, b_eff = _restrict_plane_to_dims(n, b0, mu, dims_opt)
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.05)
                lo, hi = _apply_extend(lo, hi, len(dims_opt), loX, hiX)

                if is_3d:
                    k = int(np.argmax(np.abs(n_sub)))
                    axes = [0, 1, 2]
                    axes.remove(k)
                    a, b = axes
                    xs = np.linspace(lo[a], hi[a], 14)
                    ys = np.linspace(hi[b], lo[b], 14)
                    Xg, Yg = np.meshgrid(xs, ys)

                    def solve_val(x, y):
                        num = -(n_sub[a] * x + n_sub[b] * y + b_eff)
                        den = n_sub[k] if abs(n_sub[k]) > 1e-12 else 1e-12
                        return num / den

                    if k == 2:
                        Zg = solve_val(Xg, Yg)
                        Xp_s, Yp_s, Zp_s = Xg, Yg, Zg
                    elif k == 1:
                        Yg2 = solve_val(Xg, Yg)
                        Xp_s, Yp_s, Zp_s = Xg, Yg2, Yg
                    else:
                        Xg2 = solve_val(Xg, Yg)
                        Xp_s, Yp_s, Zp_s = Xg2, Yg, Xg
                    tr = go.Surface(
                        x=Xp_s,
                        y=Yp_s,
                        z=Zp_s,
                        name=f"Plano {p}",
                        legendgroup=f"pair-{p}",
                        showscale=False,
                        opacity=0.25,
                        colorscale=[[0, "rgba(50,50,50,0.9)"], [1, "rgba(50,50,50,0.9)"]],
                    )
                else:
                    xs = np.linspace(lo[0], hi[0], 300)
                    if abs(n_sub[1]) > 1e-12:
                        ys = -(n_sub[0] * xs + b_eff) / n_sub[1]
                        tr = go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            name=f"Plano {p}",
                            legendgroup=f"pair-{p}",
                            line=dict(width=2, color="rgba(50,50,50,0.9)"),
                        )
                    else:
                        x0p = -b_eff / (n_sub[0] if abs(n_sub[0]) > 1e-12 else 1e-12)
                        tr = go.Scatter(
                            x=[x0p, x0p],
                            y=[lo[1], hi[1]],
                            mode="lines",
                            name=f"Plano {p}",
                            legendgroup=f"pair-{p}",
                            line=dict(width=2, color="rgba(50,50,50,0.9)"),
                        )
                all_traces.append(tr)
                vis_here.append(True)

        if show_quadrics and quadrics:
            for p in pair_keys:
                mdl = quadrics.get(p)
                if mdl is None:
                    continue
                Q = np.asarray(mdl["Q"], float)
                r = np.asarray(mdl["r"], float)
                cst = float(mdl["c"])
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.06)
                lo, hi = _apply_extend(lo, hi, len(dims_opt), loX, hiX)
                template = pair_templates[p].copy()

                if is_3d:
                    Xg, Yg, Zg = _grid_points_3d(lo, hi, grid_res_3d)
                    Pfull = _make_full_points_from_3d_grid(Xg, Yg, Zg, template, dims_opt)
                    G = _g_quadric_eval(Pfull, Q, r, cst).reshape(Xg.shape)
                    tr = go.Isosurface(
                        x=Xg.flatten(),
                        y=Yg.flatten(),
                        z=Zg.flatten(),
                        value=G.flatten(),
                        isomin=0.0,
                        isomax=0.0,
                        surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=False,
                        opacity=quadric_alpha,
                        name=f"Cuádrica {p}",
                        legendgroup=f"pair-{p}",
                        colorscale=[[0, "#444"], [1, "#444"]],
                    )
                else:
                    Xg, Yg = _grid_points_2d(lo, hi, grid_res_2d)
                    Pfull = _make_full_points_from_2d_grid(Xg, Yg, template, dims_opt)
                    G = _g_quadric_eval(Pfull, Q, r, cst).reshape(Xg.shape)
                    tr = go.Contour(
                        x=Xg[0, :],
                        y=Yg[:, 0],
                        z=G,
                        contours=dict(start=0.0, end=0.0, size=1.0),
                        showscale=False,
                        name=f"Cuádrica {p}",
                        legendgroup=f"pair-{p}",
                        line=dict(width=3, color="rgba(20,20,20,0.95)"),
                    )
                all_traces.append(tr)
                vis_here.append(True)

        if show_cubics and cubic_models:
            for p in pair_keys:
                mdl = cubic_models.get(p)
                if mdl is None:
                    continue
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.06)
                lo, hi = _apply_extend(lo, hi, len(dims_opt), loX, hiX)
                template = pair_templates[p].copy()
                if is_3d:
                    Xg, Yg, Zg = _grid_points_3d(lo, hi, max(16, grid_res_3d // 2))
                    Pfull = _make_full_points_from_3d_grid(Xg, Yg, Zg, template, dims_opt)
                    G = _g_cubic_eval(Pfull, mdl).reshape(Xg.shape)
                    tr = go.Isosurface(
                        x=Xg.flatten(),
                        y=Yg.flatten(),
                        z=Zg.flatten(),
                        value=G.flatten(),
                        isomin=0.0,
                        isomax=0.0,
                        surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=False,
                        opacity=quadric_alpha,
                        name=f"Cúbica {p}",
                        legendgroup=f"pair-{p}",
                        colorscale=[[0, "#777"], [1, "#777"]],
                    )
                else:
                    Xg, Yg = _grid_points_2d(lo, hi, max(100, grid_res_2d // 2))
                    Pfull = _make_full_points_from_2d_grid(Xg, Yg, template, dims_opt)
                    G = _g_cubic_eval(Pfull, mdl).reshape(Xg.shape)
                    tr = go.Contour(
                        x=Xg[0, :],
                        y=Yg[:, 0],
                        z=G,
                        contours=dict(start=0.0, end=0.0, size=1.0),
                        showscale=False,
                        name=f"Cúbica {p}",
                        legendgroup=f"pair-{p}",
                        line=dict(width=2, color="rgba(80,80,80,0.95)"),
                    )
                all_traces.append(tr)
                vis_here.append(True)

        if show_arrows:
            rng = np.random.RandomState(0)
            for p in pair_keys:
                F = frontier_by_pair[p]
                B = bases_by_pair.get(p)
                if F.size == 0 or B is None or B.size == 0:
                    continue
                m = F.shape[0]
                k = min(int(arrows_per_pair), m)
                if k <= 0:
                    continue
                idx = rng.choice(m, size=k, replace=False)
                Fp = F[idx][:, dims_opt]
                Bp = B[idx][:, dims_opt]
                U = (Fp - Bp) * float(arrow_scale)
                if is_3d:
                    xs, ys, zs = [], [], []
                    for i in range(k):
                        xs += [Bp[i, 0], Bp[i, 0] + U[i, 0], None]
                        ys += [Bp[i, 1], Bp[i, 1] + U[i, 1], None]
                        zs += [Bp[i, 2], Bp[i, 2] + U[i, 2], None]
                    tr = go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        name=f"Direcciones {p}",
                        legendgroup=f"pair-{p}",
                        showlegend=False,
                        line=dict(width=2, color="rgba(30,30,30,0.85)"),
                    )
                else:
                    xs, ys = [], []
                    for i in range(k):
                        xs += [Bp[i, 0], Bp[i, 0] + U[i, 0], None]
                        ys += [Bp[i, 1], Bp[i, 1] + U[i, 1], None]
                    tr = go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=f"Direcciones {p}",
                        legendgroup=f"pair-{p}",
                        showlegend=False,
                        line=dict(width=2, color="rgba(30,30,30,0.85)"),
                    )
                all_traces.append(tr)
                vis_here.append(True)

        all_vis_masks.append(vis_here)

    fig = go.Figure(data=all_traces)
    cnt = 0
    for v in all_vis_masks[0]:
        fig.data[cnt].visible = v
        cnt += 1

    ax_titles = [_axis_label(i) for i in dims_options[0]]
    if len(dims_options[0]) == 3:
        fig.update_layout(
            scene=dict(
                xaxis_title=ax_titles[0],
                yaxis_title=ax_titles[1],
                zaxis_title=ax_titles[2],
                aspectmode="cube",
            )
        )
    else:
        fig.update_xaxes(title_text=ax_titles[0])
        fig.update_yaxes(title_text=ax_titles[1])

    fig.update_layout(title=title, legend=dict(itemsizing="constant"))

    if len(dims_options) > 1:
        vis_by_opt = []
        base = 0
        for mask in all_vis_masks:
            block = [False] * base + mask + sum(([False] * len(m) for m in all_vis_masks[len(vis_by_opt) + 1 :]), [])
            vis_by_opt.append(block)
            base += len(mask)

        buttons = []
        for i, dims_opt in enumerate(dims_options):
            labels = [_axis_label(j) for j in dims_opt]
            if len(dims_opt) == 3:
                scene_or_axes = {
                    "scene": dict(
                        xaxis_title=labels[0],
                        yaxis_title=labels[1],
                        zaxis_title=labels[2],
                        aspectmode="cube",
                    )
                }
            else:
                scene_or_axes = {"xaxis": {"title": labels[0]}, "yaxis": {"title": labels[1]}}
            buttons.append(
                dict(
                    label=f"ejes {tuple(dims_opt)}",
                    method="update",
                    args=[{"visible": vis_by_opt[i]}, scene_or_axes],
                )
            )
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12, buttons=buttons)]
        )

    fig.show(config={"scrollZoom": True, "responsive": True})
    return fig


class _ChuchuBase:
    """Utility base class implementing shared helpers for the wrappers."""

    __artifact_version__ = "1.0"

    def __init__(
        self,
        config: Optional[ChuchuConfig] = None,
        *,
        base_estimator: Optional[Any] = None,
        cp_config: Optional[ChangePointConfig] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.config = config if config is not None else ChuchuConfig()
        self.base_estimator = base_estimator
        self.cp_config = cp_config
        self.random_state = self.config.random_state if random_state is None else random_state
        self.feature_names_: Optional[List[str]] = None
        self.estimator_: Optional[Any] = None
        self.chuchu_: Optional[Chuchu] = None
        self.records_: List[DeltaRecord] = []

    # ------------------------------------------------------------------
    def _setup_fit(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            return X.to_numpy(dtype=float)
        X_arr = np.asarray(X, dtype=float)
        self.feature_names_ = [f"x{j}" for j in range(X_arr.shape[1])]
        return X_arr

    # ------------------------------------------------------------------
    @staticmethod
    def _as_array(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    # ------------------------------------------------------------------
    def _make_estimator(self, default_factory):
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return default_factory()

    # ------------------------------------------------------------------
    def _compute_chuchu_records(self, X: np.ndarray) -> None:
        if self.estimator_ is None:
            return
        try:
            explorer = Chuchu(self.config, self.cp_config)
            explorer.fit(X, self.estimator_)
        except Exception as exc:  # pragma: no cover - safety net for optional deps
            warnings.warn(
                f"Fallo al ejecutar Chuchu; se continúa sin registros. Detalle: {exc}",
                RuntimeWarning,
            )
            self.chuchu_ = None
            self.records_ = []
        else:
            self.chuchu_ = explorer
            self.records_ = list(explorer.records_)

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.estimator_ is None:
            raise RuntimeError("Modelo no ha sido ajustado")

    # ------------------------------------------------------------------
    def save(self, filepath: str | Path) -> None:
        payload = {"__artifact_version__": self.__artifact_version__, "model": self}
        joblib.dump(payload, filepath)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, filepath: str | Path):
        payload = joblib.load(filepath)
        ver = payload.get("__artifact_version__")
        if ver != cls.__artifact_version__:
            raise ValueError(
                f"Artifact version mismatch: expected {cls.__artifact_version__}, got {ver}"
            )
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("Loaded object is not a valid Chuchu wrapper")
        return model


class ChuchuClassifier(_ChuchuBase):
    """Lightweight classifier wrapper exposing a scikit-learn style API."""

    def __init__(
        self,
        config: Optional[ChuchuConfig] = None,
        *,
        base_estimator: Optional[Any] = None,
        cp_config: Optional[ChangePointConfig] = None,
        random_state: Optional[int] = None,
        reject_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(
            config,
            base_estimator=base_estimator,
            cp_config=cp_config,
            random_state=random_state,
        )
        self.reject_threshold = reject_threshold
        self.classes_: Optional[np.ndarray] = None
        self._label_to_region: Dict[Any, int] = {}
        self._adaptor: Optional[ScoreAdaptor] = None

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "ChuchuClassifier":
        X_arr = self._setup_fit(X)
        y_arr = np.asarray(y)
        default_factory = lambda: RandomForestClassifier(
            n_estimators=150, random_state=self.random_state
        )
        estimator = self._make_estimator(default_factory)
        estimator.fit(X_arr, y_arr)
        self.estimator_ = estimator
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            classes = np.unique(y_arr)
        self.classes_ = np.asarray(classes)
        self._label_to_region = {cls: idx for idx, cls in enumerate(self.classes_)}
        self._adaptor = ScoreAdaptor(estimator, mode=self.config.mode)
        self._compute_chuchu_records(X_arr)
        return self

    # ------------------------------------------------------------------
    def fit_predict(self, X: Any, y: Any) -> np.ndarray:
        return self.fit(X, y).predict(X)

    # ------------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = self._as_array(X)
        proba = self.decision_function(X_arr)
        if proba.size == 0:
            return np.full(X_arr.shape[0], -1, dtype=int)
        idx = np.argmax(proba, axis=1)
        scores = proba[np.arange(proba.shape[0]), idx]
        labels = self.classes_[idx]
        if self.reject_threshold is not None:
            labels = labels.astype(object)
            reject_mask = scores < float(self.reject_threshold)
            labels[reject_mask] = -1
            return np.asarray(labels, dtype=int)
        return labels.astype(int)

    # ------------------------------------------------------------------
    def predict_proba(self, X: Any) -> Dict[Any, np.ndarray]:
        self._check_fitted()
        if self._adaptor is None:
            raise RuntimeError("Probability adaptor is not available")
        scores = self._adaptor.scores(self._as_array(X))
        return {cls: scores[:, idx] for idx, cls in enumerate(self.classes_)}

    # ------------------------------------------------------------------
    def decision_function(self, X: Any) -> np.ndarray:
        self._check_fitted()
        if self._adaptor is None:
            raise RuntimeError("Probability adaptor is not available")
        return self._adaptor.scores(self._as_array(X))

    # ------------------------------------------------------------------
    def transform(self, X: Any) -> np.ndarray:
        return self.decision_function(X)

    # ------------------------------------------------------------------
    def fit_transform(self, X: Any, y: Any) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    def membership(self, X: Any) -> Dict[Any, np.ndarray]:
        labels = self.predict(X)
        out: Dict[Any, np.ndarray] = {}
        for cls in self.classes_:
            out[int(cls)] = labels == int(cls)
        reject_mask = labels == -1
        if np.any(reject_mask):
            out[-1] = reject_mask
        return out

    # ------------------------------------------------------------------
    def predict_regions(self, X: Any) -> pd.DataFrame:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = None
        labels = self.predict(X)
        region_ids = np.full(labels.shape[0], -1, dtype=int)
        for idx, lab in enumerate(labels):
            if lab == -1:
                continue
            region_ids[idx] = self._label_to_region.get(int(lab), -1)
        return pd.DataFrame({"label": labels, "region_id": region_ids}, index=index)

    # ------------------------------------------------------------------
    def region_mask(self, X: Any) -> np.ndarray:
        df = self.predict_regions(X)
        return df["region_id"].to_numpy() >= 0

    # ------------------------------------------------------------------
    def score(self, X: Any, y: Any) -> float:
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return macro_f1_ignore_rejects(y_true, y_pred)

    # ------------------------------------------------------------------
    def plot_pairs(
        self,
        X: Any,
        max_pairs: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self._check_fitted()
        X_arr = self._as_array(X)
        if X_arr.shape[1] < 2:
            raise ValueError("Se requieren al menos dos características para plot_pairs")
        if feature_names is None:
            feature_names = self.feature_names_ if self.feature_names_ is not None else ["x0", "x1"]
        import matplotlib.pyplot as plt  # pragma: no cover - visual helper

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        labels = self.predict(X_arr)
        ax.scatter(X_arr[:, 0], X_arr[:, 1], c=labels, cmap="tab10", s=32, edgecolors="none")
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title("ChuchuClassifier - pares de características")
        return fig, [ax]

    # ------------------------------------------------------------------
    def plot_classes(
        self,
        X: Any,
        y: Any,
        grid_res: int = 200,
        contour_levels: Optional[Union[np.ndarray, List[float]]] = None,
        max_paths: int = 20,
        show_paths: bool = True,
    ):
        self._check_fitted()
        X_arr = self._as_array(X)
        if X_arr.shape[1] < 2:
            raise ValueError("Se requieren al menos dos características para plot_classes")
        import matplotlib.pyplot as plt  # pragma: no cover - visual helper

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        y_arr = np.asarray(y)
        sc = ax.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap="tab10", s=32, edgecolors="none")
        ax.set_xlabel(self.feature_names_[0] if self.feature_names_ else "x0")
        ax.set_ylabel(self.feature_names_[1] if self.feature_names_ else "x1")
        ax.set_title("ChuchuClassifier - distribución de clases")
        fig.colorbar(sc, ax=ax, label="Clase")
        return fig, [ax]

    # ------------------------------------------------------------------
    def plot_pair_3d(self, X: Any, dims: Tuple[int, int]):  # pragma: no cover - simple guard
        raise NotImplementedError("ChuchuClassifier no implementa plot_pair_3d")


class ChuchuRegressor(_ChuchuBase):
    """Regression counterpart of :class:`ChuchuClassifier`."""

    def __init__(
        self,
        config: Optional[ChuchuConfig] = None,
        *,
        base_estimator: Optional[Any] = None,
        cp_config: Optional[ChangePointConfig] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            config,
            base_estimator=base_estimator,
            cp_config=cp_config,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "ChuchuRegressor":
        X_arr = self._setup_fit(X)
        y_arr = np.asarray(y)
        default_factory = lambda: RandomForestRegressor(random_state=self.random_state)
        estimator = self._make_estimator(default_factory)
        estimator.fit(X_arr, y_arr)
        self.estimator_ = estimator
        # ``Chuchu`` currently focuses on classification; skip record extraction
        self.chuchu_ = None
        self.records_ = []
        return self

    # ------------------------------------------------------------------
    def fit_predict(self, X: Any, y: Any) -> np.ndarray:
        return self.fit(X, y).predict(X)

    # ------------------------------------------------------------------
    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = self._as_array(X)
        return np.asarray(self.estimator_.predict(X_arr), dtype=float)

    # ------------------------------------------------------------------
    def decision_function(self, X: Any) -> np.ndarray:
        return self.predict(X)

    # ------------------------------------------------------------------
    def transform(self, X: Any) -> np.ndarray:
        return self.decision_function(X)

    # ------------------------------------------------------------------
    def fit_transform(self, X: Any, y: Any) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    def predict_regions(self, X: Any) -> pd.DataFrame:
        preds = self.predict(X)
        if isinstance(X, pd.DataFrame):
            index = X.index
        else:
            index = None
        region_ids = np.zeros_like(preds, dtype=int)
        return pd.DataFrame({"label": preds, "region_id": region_ids}, index=index)

    # ------------------------------------------------------------------
    def region_mask(self, X: Any) -> np.ndarray:
        df = self.predict_regions(X)
        return df["region_id"].to_numpy() >= 0

    # ------------------------------------------------------------------
    def plot_pairs(
        self,
        X: Any,
        max_pairs: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self._check_fitted()
        X_arr = self._as_array(X)
        if X_arr.shape[1] < 2:
            raise ValueError("Se requieren al menos dos características para plot_pairs")
        preds = self.predict(X_arr)
        if feature_names is None:
            feature_names = self.feature_names_ if self.feature_names_ is not None else ["x0", "x1"]
        import matplotlib.pyplot as plt  # pragma: no cover - visual helper

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        sc = ax.scatter(X_arr[:, 0], X_arr[:, 1], c=preds, cmap="viridis", s=32, edgecolors="none")
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title("ChuchuRegressor - pares de características")
        fig.colorbar(sc, ax=ax, label="Predicción")
        return fig, [ax]

    # ------------------------------------------------------------------
    def plot_classes(
        self,
        X: Any,
        y: Any,
        grid_res: int = 200,
        contour_levels: Optional[Union[np.ndarray, List[float]]] = None,
        max_paths: int = 20,
        show_paths: bool = True,
    ):
        self._check_fitted()
        X_arr = self._as_array(X)
        if X_arr.shape[1] < 2:
            raise ValueError("Se requieren al menos dos características para plot_classes")
        import matplotlib.pyplot as plt  # pragma: no cover - visual helper

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        y_arr = np.asarray(y)
        sc = ax.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap="viridis", s=32, edgecolors="none")
        ax.set_xlabel(self.feature_names_[0] if self.feature_names_ else "x0")
        ax.set_ylabel(self.feature_names_[1] if self.feature_names_ else "x1")
        ax.set_title("ChuchuRegressor - valores objetivo")
        fig.colorbar(sc, ax=ax, label="y")
        return fig, [ax]

    # ------------------------------------------------------------------
    def plot_pair_3d(self, X: Any, dims: Tuple[int, int]):  # pragma: no cover - simple guard
        raise NotImplementedError("ChuchuRegressor no implementa plot_pair_3d")


def normalize_pair_order(records, *, make_copy=True):
    """Ensure ``y0 <= y1`` for each :class:`DeltaRecord`."""

    out = []
    for r in records:
        rr = copy.deepcopy(r) if make_copy else r

        y0 = int(rr.y0)
        y1 = int(rr.y1)
        if y0 > y1:
            rr.y0, rr.y1 = y1, y0

            x0_old = np.asarray(rr.x0, float).copy()
            x1_old = np.asarray(rr.x1, float).copy()
            rr.x0, rr.x1 = x1_old, x0_old

            try:
                rr.delta = rr.x1 - rr.x0
            except Exception:
                pass

            if hasattr(rr, "S0") and hasattr(rr, "S1"):
                S0_old = np.asarray(rr.S0).copy()
                S1_old = np.asarray(rr.S1).copy()
                rr.S0, rr.S1 = S1_old, S0_old

            if hasattr(rr, "cp_y_left") and hasattr(rr, "cp_y_right"):
                yl = np.asarray(rr.cp_y_left).copy()
                yr = np.asarray(rr.cp_y_right).copy()
                rr.cp_y_left, rr.cp_y_right = yr, yl

        out.append(rr)
    return out


__all__ = [
    "Chuchu",
    "ChuchuConfig",
    "ChangePointConfig",
    "ChuchuClassifier",
    "ChuchuRegressor",
    "macro_f1_ignore_rejects",
    "DeltaRecord",
    "DeltaRecordLite",
    "PCA3D",
    "ScoreAdaptor",
    "batch_false_position_flip",
    "build_sets_from_records",
    "build_weighted_frontier",
    "choose_dims_topvar",
    "compute_frontier_planes_weighted",
    "convex_hull_faces",
    "eval_cubic_models",
    "fit_cubic_from_records_weighted",
    "fit_quadrics_from_records_weighted",
    "fit_tls_plane_weighted",
    "find_segment_crossings_cubic",
    "normalize_pair_order",
    "plot_frontiers_implicit_interactive",
    "project_all_to_3d",
]

