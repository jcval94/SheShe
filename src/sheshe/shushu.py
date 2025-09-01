# -*- coding: utf-8 -*-
"""
ShuShu: Subespacios Heurísticos y Umbrales Sobre Hiperplanos Unificados.

This module is adapted from ``multiclass_gradient_maxima_demo_v2.py``.  It
implements a gradient-based search of maxima of scalar score functions with
multiple starting points.  A thin convenience wrapper automatically extends the
search to multiclass problems by running the optimizer once per class.

The implementation is largely a direct import of the original demo script with
minimal adaptations so it can live inside the ``sheshe`` package.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from matplotlib.lines import Line2D
from itertools import combinations

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from .region_interpretability import RegionInterpreter
from .sheshe import ClusterRegion

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# (Opcional) hull 2D si existe SciPy
try:  # pragma: no cover - optional dependency
    from scipy.spatial import ConvexHull
    _HAS_SCIPY_HULL = True
except Exception:  # pragma: no cover - SciPy no disponible
    _HAS_SCIPY_HULL = False


# =============================================================================
#                       CLUSTERER (MAXIMIZA UNA FUNCIÓN ESCALAR)
# =============================================================================

class _ShuShu:
    """
    Encuentra máximos de una función escalar ``f(X)->R`` con múltiples arranques.

    Este algoritmo utiliza un subespacio seleccionado vía ``RandomForest`` y
    optimización por gradiente (Adam con gradientes aproximados).  Los máximos se
    fusionan mediante ``DBSCAN`` y opcionalmente asigna todos los puntos a los
    centroides detectados.
    """

    def __init__(
        self,
        k: int = 25,
        k_per_dim: int = 1,
        importance_target: float = 0.85,
        rf_estimators: int = 35,
        importance_sample_size: Optional[int] = 500,
        max_iter: int = 80,
        lr: float = 0.06,
        tol_score: float = 1e-6,
        grad_eps_unit: float = 2e-2,
        patience: int = 8,
        merge_eps: Optional[float] = None,
        merge_alpha: float = 0.30,
        merge_kdist: int = 3,
        auto_eps_quantile: float = 0.90,
        grad_mode: str = "spsa",  # "spsa" | "forward" | "central"
        assign_all: bool = True,
        enable_cache: bool = True,
        cache_round_decimals: int = 6,
        max_score_evals: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.k = k
        self.k_per_dim = k_per_dim
        self.importance_target = importance_target
        self.rf_estimators = rf_estimators
        self.importance_sample_size = importance_sample_size
        self.max_iter = max_iter
        self.lr = lr
        self.tol_score = tol_score
        self.grad_eps_unit = grad_eps_unit
        self.patience = patience
        self.merge_eps = merge_eps
        self.merge_alpha = merge_alpha
        self.merge_kdist = merge_kdist
        self.auto_eps_quantile = auto_eps_quantile
        self.grad_mode = grad_mode
        self.assign_all = assign_all
        self.enable_cache = enable_cache
        self.cache_round_decimals = cache_round_decimals
        self.max_score_evals = max_score_evals
        self.random_state = random_state
        self.verbose = verbose

        # salidas / runtime
        self.selected_dims_: np.ndarray = np.array([], dtype=int)
        self.ignored_dims_: np.ndarray = np.array([], dtype=int)
        self.feature_importances_: Optional[np.ndarray] = None
        self.medians_: Optional[np.ndarray] = None
        self._mins: Optional[np.ndarray] = None
        self._spans: Optional[np.ndarray] = None
        self._medians_unit: Optional[np.ndarray] = None

        self.centroids_: Optional[np.ndarray] = None
        self.centroid_scores_: Optional[np.ndarray] = None
        self.paths_: Optional[List[List[Dict]]] = None
        self.start_points_: Optional[np.ndarray] = None
        self.start_to_centroid_: Optional[List[int]] = None
        self.endpoints_unit_: Optional[np.ndarray] = None
        self.clusters_: Optional[List[Dict]] = None

        self.timings_: Dict[str, float] = {}
        self.score_time_s_: float = 0.0
        self.n_score_evals_: int = 0
        self._cache: Dict[Tuple, float] = {}
        self._rng = np.random.RandomState(self.random_state)

    # --------- escalado a [0,1]^d ----------
    def _fit_scaler(self, X: np.ndarray):
        mins = X.min(axis=0)
        spans = X.max(axis=0) - mins + 1e-12
        self._mins, self._spans = mins, spans

    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mins) / self._spans

    def _from_unit(self, U: np.ndarray) -> np.ndarray:
        return U * self._spans + self._mins

    def _project_unit(self, U: np.ndarray) -> np.ndarray:
        U = np.minimum(np.maximum(U, 0.0), 1.0)
        if self.ignored_dims_.size > 0:
            U[:, self.ignored_dims_] = self._medians_unit[self.ignored_dims_]
        return U

    # --------- API principal ----------
    def fit(
        self, X: np.ndarray, score_fn: Callable[[np.ndarray], np.ndarray]
    ) -> "_ShuShu":
        t0 = time.perf_counter()
        rng = self._rng

        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.medians_ = np.median(X, axis=0)
        self._fit_scaler(X)
        self._medians_unit = self._to_unit(self.medians_[None, :])[0]

        # 1) score global (en datos originales)
        t1 = time.perf_counter()
        y_all = self._score_batch(score_fn, X)
        t2 = time.perf_counter()

        # 2) subespacio vía RF
        t2a = time.perf_counter()
        (
            self.selected_dims_,
            self.ignored_dims_,
        ) = self._select_dims_via_rf(X, y_all, self.importance_target, rng)
        t2b = time.perf_counter()

        # 3) arranques diversos
        t3a = time.perf_counter()
        k_eff = self._compute_k_eff(len(self.selected_dims_))
        starts_orig = self._sample_starts_diverse(X, k_eff, rng)
        self.start_points_ = starts_orig.copy()
        U0 = self._to_unit(starts_orig)
        if self.ignored_dims_.size > 0:
            U0[:, self.ignored_dims_] = self._medians_unit[self.ignored_dims_]
        t3b = time.perf_counter()

        # 4) optimización (Adam)
        t4a = time.perf_counter()
        paths, U_end, end_scores = self._multi_start_optimize(U0, score_fn)
        t4b = time.perf_counter()

        endpoints = self._from_unit(U_end) if U_end.size else np.zeros((0, d))
        self.paths_ = paths

        # 5) merge de máximos
        t5a = time.perf_counter()
        labels, unique_labels, reps_idx_per_label = self._merge_maxima_dbscan(
            endpoints, end_scores, U_end
        )
        t5b = time.perf_counter()

        # 6) centroides
        t6a = time.perf_counter()
        centroids, centroid_scores, label_to_centroid_idx = [], [], {}
        for ci, lab in enumerate(unique_labels):
            rep_idx = reps_idx_per_label[lab]
            centroids.append(endpoints[rep_idx])
            centroid_scores.append(end_scores[rep_idx])
            label_to_centroid_idx[lab] = ci
        self.centroids_ = np.vstack(centroids) if centroids else np.zeros((0, d))
        self.centroid_scores_ = (
            np.asarray(centroid_scores) if centroid_scores else np.zeros((0,))
        )
        self.start_to_centroid_ = [label_to_centroid_idx.get(lab, 0) for lab in labels]
        self.endpoints_unit_ = U_end
        t6b = time.perf_counter()

        # 7) clusters_ con área de influencia por decil del score
        t7a = time.perf_counter()
        self.clusters_ = self._build_influence_areas(X, y_all, paths, score_fn)
        t7b = time.perf_counter()

        # 8) asignación total (opcional)
        t8a = time.perf_counter()
        if self.assign_all and self.centroids_.shape[0] > 0:
            self.labels_ = self._assign_labels(X)
        t8b = time.perf_counter()

        t9 = time.perf_counter()
        self.timings_ = {
            "score_global_s": (t2 - t1),
            "select_dims_s": (t2b - t2a),
            "starts_s": (t3b - t3a),
            "optimize_s": (t4b - t4a),
            "merge_s": (t5b - t5a),
            "centroids_s": (t6b - t6a),
            "build_clusters_s": (t7b - t7a),
            "assign_all_s": (t8b - t8a),
            "total_fit_s": (t9 - t0),
            "score_fn_time_s": self.score_time_s_,
            "n_score_evals": int(self.n_score_evals_),
        }
        return self

    # --------- score & cache ----------
    def _score_batch(self, score_fn, X: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        y = score_fn(X)
        self.score_time_s_ += time.perf_counter() - t
        self.n_score_evals_ += X.shape[0]
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "score_fn debe devolver vector 1D (n_samples,) del mismo tamaño que X."
            )
        return y

    def _score_unit_cached(self, U: np.ndarray, score_fn) -> np.ndarray:
        X = self._from_unit(U)
        if not self.enable_cache or self.selected_dims_.size == 0:
            return self._score_batch(score_fn, X)
        keys = []
        misses_idx = []
        y = np.empty(U.shape[0], dtype=float)
        for i, u in enumerate(U):
            key = tuple(np.round(u[self.selected_dims_], self.cache_round_decimals))
            keys.append(key)
            if key in self._cache:
                y[i] = self._cache[key]
            else:
                misses_idx.append(i)
        if misses_idx:
            Xm = X[misses_idx]
            ym = self._score_batch(score_fn, Xm)
            for i, val in zip(misses_idx, ym):
                self._cache[keys[i]] = float(val)
                y[i] = float(val)
        return y

    # --------- selección & seeds ----------
    def _select_dims_via_rf(self, X, y, target, rng):
        m = (
            min(self.importance_sample_size, X.shape[0])
            if self.importance_sample_size is not None
            else X.shape[0]
        )
        idx = rng.choice(X.shape[0], size=m, replace=False)
        Xs, ys = X[idx], y[idx]
        rf = RandomForestRegressor(
            n_estimators=self.rf_estimators, random_state=self.random_state, n_jobs=-1
        )
        rf.fit(Xs, ys)
        imp = rf.feature_importances_
        self.feature_importances_ = imp
        order = np.argsort(imp)[::-1]
        cum = np.cumsum(imp[order])
        k = int(np.searchsorted(cum, target) + 1)
        selected = np.sort(order[:k])
        ignored = np.setdiff1d(np.arange(X.shape[1]), selected, assume_unique=False)
        return selected, ignored

    def _compute_k_eff(self, n_sel: int) -> int:
        extra = max(0, n_sel - 2) * self.k_per_dim
        return max(1, int(self.k + extra))

    def _sample_starts_diverse(self, X: np.ndarray, k_eff: int, rng) -> np.ndarray:
        n, d = X.shape
        if k_eff <= 1 or n <= 1:
            idx = rng.randint(0, n, size=min(1, n))
            return X[idx]
        U = self._to_unit(X)
        Z = U[:, self.selected_dims_] if self.selected_dims_.size > 0 else U
        idxs = [rng.randint(0, n)]
        last = Z[idxs[-1]]
        dists = np.linalg.norm(Z - last, axis=1)
        for _ in range(1, min(k_eff, n)):
            next_idx = int(np.argmax(dists))
            idxs.append(next_idx)
            last = Z[next_idx]
            dists = np.minimum(dists, np.linalg.norm(Z - last, axis=1))
        if k_eff > n:
            extra = k_eff - n
            extra_idx = rng.randint(0, n, size=extra)
            idxs = idxs + list(extra_idx)
        return X[idxs]

    # --------- gradientes ----------
    def _grad_spsa_batch(self, U: np.ndarray, score_fn, dims: np.ndarray, rng) -> np.ndarray:
        k, d = U.shape
        if dims.size == 0:
            return np.zeros_like(U)
        delta = rng.choice([-1.0, 1.0], size=(k, dims.size))
        eps = self.grad_eps_unit
        U_plus = U.copy()
        U_minus = U.copy()
        U_plus[:, dims] += eps * delta
        U_minus[:, dims] -= eps * delta
        U_plus = self._project_unit(U_plus)
        U_minus = self._project_unit(U_minus)
        f_plus = self._score_unit_cached(U_plus, score_fn)
        f_minus = self._score_unit_cached(U_minus, score_fn)
        g = np.zeros_like(U)
        denom = (2.0 * eps) * delta
        grads_sel = (f_plus[:, None] - f_minus[:, None]) / denom
        g[:, dims] = grads_sel
        return g

    def _grad_forward_batched(self, U: np.ndarray, score_fn, dims: np.ndarray) -> np.ndarray:
        k, d = U.shape
        if dims.size == 0:
            return np.zeros_like(U)
        eps = self.grad_eps_unit
        blocks = [U]
        for j in dims:
            Up = U.copy()
            Up[:, j] += eps
            blocks.append(self._project_unit(Up))
        bigU = np.vstack(blocks)
        f = self._score_unit_cached(bigU, score_fn)
        base = f[:k]
        g = np.zeros_like(U)
        for t, j in enumerate(dims, start=1):
            fp = f[t * k : (t + 1) * k]
            g[:, j] = (fp - base) / eps
        return g

    def _grad_central_batched(self, U: np.ndarray, score_fn, dims: np.ndarray) -> np.ndarray:
        k, d = U.shape
        if dims.size == 0:
            return np.zeros_like(U)
        eps = self.grad_eps_unit
        blocks = []
        for j in dims:
            Up = U.copy()
            Um = U.copy()
            Up[:, j] += eps
            Um[:, j] -= eps
            blocks.append(self._project_unit(Up))
            blocks.append(self._project_unit(Um))
        bigU = np.vstack(blocks)
        f = self._score_unit_cached(bigU, score_fn)
        g = np.zeros_like(U)
        for idx_j, j in enumerate(dims):
            fp = f[(2 * idx_j) * k : (2 * idx_j) * k + k]
            fm = f[(2 * idx_j + 1) * k : (2 * idx_j + 2) * k]
            g[:, j] = (fp - fm) / (2.0 * eps)
        return g

    # --------- optimización ----------
    def _multi_start_optimize(self, U0: np.ndarray, score_fn):
        rng = self._rng
        k, d = U0.shape
        dims = self.selected_dims_

        # Adam
        m = np.zeros_like(U0)
        v = np.zeros_like(U0)
        t = 0
        U = U0.copy()
        S = self._score_unit_cached(self._project_unit(U), score_fn)
        best_S = S.copy()
        best_U = U.copy()
        no_imp = np.zeros(k, dtype=int)
        paths: List[List[Dict]] = [[] for _ in range(k)]
        for i in range(k):
            paths[i].append({"x": self._from_unit(U[i : i + 1])[0].copy(), "score": float(S[i])})

        for it in range(self.max_iter):
            if self.max_score_evals is not None and self.n_score_evals_ >= self.max_score_evals:
                if self.verbose:
                    print(f"[stop] budget evals alcanzado ({self.n_score_evals_})")
                break

            if self.grad_mode == "spsa":
                G = self._grad_spsa_batch(U, score_fn, dims, rng)
            elif self.grad_mode == "central":
                G = self._grad_central_batched(U, score_fn, dims)
            else:
                G = self._grad_forward_batched(U, score_fn, dims)

            t += 1
            beta1, beta2 = 0.9, 0.999
            m[:, dims] = beta1 * m[:, dims] + (1 - beta1) * G[:, dims]
            v[:, dims] = beta2 * v[:, dims] + (1 - beta2) * (G[:, dims] ** 2)
            mhat = m[:, dims] / (1 - beta1 ** t)
            vhat = v[:, dims] / (1 - beta2 ** t)
            step_vec = self.lr * mhat / (np.sqrt(vhat) + 1e-8)
            step_vec = np.clip(step_vec, -0.10, 0.10)

            U_new = U.copy()
            U_new[:, dims] = U[:, dims] + step_vec
            U_new = self._project_unit(U_new)
            S_new = self._score_unit_cached(U_new, score_fn)

            improved = S_new > S + self.tol_score
            for i in range(k):
                if improved[i]:
                    U[i] = U_new[i]
                    S[i] = S_new[i]
                    no_imp[i] = 0
                    if S[i] > best_S[i]:
                        best_S[i] = S[i]
                        best_U[i] = U[i]
                else:
                    no_imp[i] += 1
                if (it + 1) % 5 == 0 or improved[i]:
                    paths[i].append({"x": self._from_unit(U[i : i + 1])[0].copy(), "score": float(S[i])})

            if np.all(no_imp >= self.patience):
                break

        for i in range(k):
            paths[i].append({"x": self._from_unit(best_U[i : i + 1])[0].copy(), "score": float(best_S[i])})
        return paths, best_U, best_S

    # --------- merge ----------
    def _auto_eps(self, Z: np.ndarray) -> float:
        k = min(self.merge_kdist, max(1, Z.shape[0] - 1))
        if Z.shape[0] <= 1:
            return 0.05
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(Z)
        dists, _ = nn.kneighbors(Z)
        kdist = np.sort(dists[:, -1])
        eps = float(np.quantile(kdist, self.auto_eps_quantile))
        return eps if eps > 0 else 0.05

    def _merge_maxima_dbscan(self, endpoints: np.ndarray, end_scores: np.ndarray, U_end: np.ndarray):
        if endpoints.shape[0] == 0:
            return np.array([], dtype=int), [], {}
        Z = U_end[:, self.selected_dims_] if self.selected_dims_.size > 0 else U_end
        s = (end_scores - end_scores.mean()) / (end_scores.std() + 1e-12)
        Z_mix = np.hstack([Z, (self.merge_alpha ** 0.5) * s[:, None]])
        eps = float(self.merge_eps) if self.merge_eps is not None else self._auto_eps(Z_mix)
        db = DBSCAN(eps=eps, min_samples=1, metric="euclidean")
        labels = db.fit_predict(Z_mix)
        unique = sorted(set(int(l) for l in labels))
        reps: Dict[int, int] = {}
        for lab in unique:
            idxs = np.where(labels == lab)[0]
            j = idxs[np.argmax(end_scores[idxs])]
            reps[lab] = int(j)
        return labels, unique, reps

    # --------- áreas de influencia (decil del centroide) ----------
    def _build_influence_areas(self, X, y_all, paths, score_fn):
        clusters = []
        if self.centroids_ is None or self.centroids_.shape[0] == 0:
            return clusters
        edges = np.quantile(y_all, np.linspace(0.0, 1.0, 11))

        def decile_of(v: float) -> int:
            return int(np.digitize([v], edges, right=True)[0])

        groups: Dict[int, List[np.ndarray]] = {ci: [] for ci in range(self.centroids_.shape[0])}
        for s_idx, ci in enumerate(self.start_to_centroid_):
            pts = np.vstack([st["x"] for st in paths[s_idx]])
            groups[ci].append(pts)

        for ci in range(self.centroids_.shape[0]):
            centroid = self.centroids_[ci]
            centroid_score = (
                float(self.centroid_scores_[ci]) if self.centroid_scores_ is not None else np.nan
            )
            dcent = decile_of(centroid_score) if np.isfinite(centroid_score) else 5

            all_pts = (
                np.vstack(groups[ci]) if len(groups[ci]) > 0 else centroid[None, :]
            )
            sc = self._score_batch(score_fn, all_pts)
            mask = np.digitize(sc, edges, right=True) == dcent
            area_points = all_pts[mask]

            boundary = None
            btype = None
            if self.selected_dims_.size >= 2 and _HAS_SCIPY_HULL and area_points.shape[0] >= 3:
                try:
                    hp = area_points[:, self.selected_dims_[:2]]
                    hull = ConvexHull(hp)
                    boundary = hp[hull.vertices]
                    btype = "convex_hull_2d_subspace"
                except Exception:
                    boundary = None
                    btype = None

            clusters.append(
                {
                    "label": int(ci),
                    "centroid": centroid,
                    "score": float(centroid_score),
                    "decile": int(dcent),
                    "area_points": area_points,
                    "boundary": boundary,
                    "boundary_type": btype,
                }
            )
        return clusters

    # --------- asignación ----------
    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None or self.centroids_.shape[0] == 0:
            return np.zeros(X.shape[0], dtype=int)
        Uc = (
            self._to_unit(self.centroids_)[:, self.selected_dims_]
            if self.selected_dims_.size > 0
            else self._to_unit(self.centroids_)
        )
        XU = self._to_unit(np.asarray(X, dtype=float))
        if self.ignored_dims_.size > 0:
            XU[:, self.ignored_dims_] = self._medians_unit[self.ignored_dims_]
        ZX = XU[:, self.selected_dims_] if self.selected_dims_.size > 0 else XU
        nn = NearestNeighbors(n_neighbors=1).fit(Uc)
        _, idxs = nn.kneighbors(ZX)
        return idxs[:, 0].astype(int)
# =============================================================================
#                         INTERFAZ UNIFICADA
# =============================================================================

class ShuShu:
    """Unified interface for the ShuShu optimizer.

    When ``y`` is provided the optimizer runs once per class using a score
    function per class (defaulting to ``LogisticRegression``).  When ``y`` is
    omitted it expects a scalar ``score_fn`` and behaves as the core optimizer
    returning the detected maxima directly.
    """

    def __init__(
        self,
        clusterer_factory: Callable[[], _ShuShu] | None = None,
        random_state: Optional[int] = None,
        **clusterer_kwargs,
    ):
        self.random_state = random_state
        if clusterer_factory is None:
            def _default_factory():
                return _ShuShu(random_state=self.random_state, **clusterer_kwargs)
            self.clusterer_factory = _default_factory
        else:
            self.clusterer_factory = clusterer_factory
        self.feature_names_: Optional[List[str]] = None
        self.clusterer_: Optional[_ShuShu] = None
        self.per_class_: Dict[int, Dict] = {}
        self.classes_: Optional[np.ndarray] = None
        self.model_ = None
        self.mode_: Optional[str] = None
        self.score_fn_: Optional[Callable[[np.ndarray], np.ndarray]] = None
        # expose discovered regions from underlying clusterers
        self.regions_: List[Dict] = []

    def _build_score_fns(
        self,
        X: np.ndarray,
        y: np.ndarray,
        score_model=None,
        score_fn_multi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        score_fn_per_class: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    ):
        if score_fn_per_class is not None:
            classes = np.unique(y)
            if len(score_fn_per_class) != len(classes):
                raise ValueError("score_fn_per_class debe tener una función por clase.")
            return score_fn_per_class, classes

        if score_fn_multi is not None:
            classes = np.unique(y)

            def make_c(ci: int):
                return lambda Z: np.asarray(score_fn_multi(Z))[:, ci]

            return [make_c(i) for i in range(len(classes))], classes

        if score_model is None:
            model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=self.random_state)
            model.fit(X, y)
            self.model_ = model
        else:
            model = score_model
            if not hasattr(model, "predict_proba"):
                raise ValueError("score_model debe tener predict_proba.")
            if not hasattr(model, "classes_"):
                model.fit(X, y)
            self.model_ = model

        classes = np.array(model.classes_)

        def make_c(ci: int):
            return lambda Z: np.asarray(model.predict_proba(Z))[:, ci]

        return [make_c(i) for i in range(len(classes))], classes

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        score_model=None,
        score_fn_multi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        score_fn_per_class: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    ):
        X = np.asarray(X, dtype=float)
        if y is None:
            if score_fn is None:
                raise ValueError("score_fn must be provided when y is None")
            clusterer = self.clusterer_factory()
            clusterer.fit(X, score_fn=score_fn)
            self.clusterer_ = clusterer
            self.centroids_ = clusterer.centroids_
            self.clusters_ = clusterer.clusters_
            self.timings_ = clusterer.timings_
            self.mode_ = "scalar"
            self.score_fn_ = score_fn
            self.regions_ = list(clusterer.clusters_)
            return self

        self.feature_names_ = (
            feature_names if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        )
        list_score_fns, classes = self._build_score_fns(
            X,
            y,
            score_model=score_model,
            score_fn_multi=score_fn_multi,
            score_fn_per_class=score_fn_per_class,
        )

        results = {}
        for ci, cls_label in enumerate(classes):
            score_fn_c = list_score_fns[ci]
            clusterer = self.clusterer_factory()
            clusterer.fit(X, score_fn=score_fn_c)

            imp = (
                clusterer.feature_importances_ if clusterer.feature_importances_ is not None else np.zeros(X.shape[1])
            )
            if clusterer.selected_dims_.size >= 2:
                dims2 = clusterer.selected_dims_[:2]
            else:
                dims2 = np.argsort(imp)[::-1][:2] if np.any(imp) else np.array([0, 1])

            results[ci] = {
                "label": cls_label,
                "clusterer": clusterer,
                "n_clusters": int(clusterer.centroids_.shape[0]) if clusterer.centroids_ is not None else 0,
                "dims2": dims2,
                "score_fn": score_fn_c,
            }
        self.per_class_ = results
        self.classes_ = classes
        self.mode_ = "multiclass"
        # flatten regions across classes for convenient access
        all_regions: List[Dict] = []
        for info in results.values():
            clus = info["clusterer"]
            cls_label = info["label"]
            for reg in getattr(clus, "clusters_", []) or []:
                r = dict(reg)
                r["cluster_id"] = len(all_regions)
                r["label"] = cls_label
                all_regions.append(r)
        self.regions_ = all_regions
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_kwargs) -> np.ndarray:
        """Convenience method that fits and then predicts ``X``.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Labels for multiclass mode.
        **fit_kwargs : dict
            Additional keyword arguments passed to :meth:`fit`.
        """
        self.fit(X, y, **fit_kwargs)
        return self.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for ``X``.

        Only available when the model was fit in multiclass mode.  Relies on
        the internal ``score_model`` trained during :meth:`fit`.
        """
        if self.mode_ != "multiclass" or self.model_ is None:
            raise RuntimeError("predict_proba only available after fitting with labels")
        X = np.asarray(X, dtype=float)
        if not hasattr(self.model_, "predict_proba"):
            raise RuntimeError("Underlying model lacks predict_proba")
        return np.asarray(self.model_.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or cluster ids for ``X``.

        * If fitted with ``y`` (multiclass mode) returns class labels using the
          internal model.
        * If fitted with ``score_fn`` (scalar mode) returns the index of the
          closest centroid.
        """
        if self.mode_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        if self.mode_ == "multiclass":
            if self.model_ is None:
                raise RuntimeError("Model missing for predict")
            return np.asarray(self.model_.predict(X))
        else:
            if self.clusterer_ is None:
                raise RuntimeError("Clusterer missing for predict")
            return self.clusterer_._assign_labels(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw decision scores for ``X``.

        In multiclass mode this delegates to the underlying ``score_model`` if
        it provides :meth:`decision_function`. In scalar mode it returns the
        negative distance to each centroid in the scaled feature space.
        """
        if self.mode_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        if self.mode_ == "multiclass":
            if self.model_ is None or not hasattr(self.model_, "decision_function"):
                raise RuntimeError("Underlying model lacks decision_function")
            return np.asarray(self.model_.decision_function(X))
        else:
            if self.clusterer_ is None or self.clusterer_.centroids_ is None:
                raise RuntimeError("Clusterer missing for decision_function")
            c = self.clusterer_
            if c.centroids_.shape[0] == 0:
                return np.zeros((X.shape[0], 0))
            Uc = c._to_unit(c.centroids_)
            if c.selected_dims_.size > 0:
                Uc = Uc[:, c.selected_dims_]
            XU = c._to_unit(X)
            if c.ignored_dims_.size > 0:
                XU[:, c.ignored_dims_] = c._medians_unit[c.ignored_dims_]
            ZX = XU[:, c.selected_dims_] if c.selected_dims_.size > 0 else XU
            dists = np.linalg.norm(ZX[:, None, :] - Uc[None, :, :], axis=2)
            return -dists

    def predict_regions(self, X: np.ndarray):
        """Predict class labels and cluster ids for ``X``.

        Returns a tuple ``(labels, cluster_ids)`` similar to
        :class:`ModalScoutEnsemble`.

        * In multiclass mode ``labels`` are class predictions and
          ``cluster_ids`` the id of the centroid of the predicted class.
        * In scalar mode both arrays contain the index of the closest centroid.
        """
        if self.mode_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        if self.mode_ == "multiclass":
            if self.model_ is None:
                raise RuntimeError("Model missing for predict_regions")
            labels = np.asarray(self.model_.predict(X))
            cluster_ids = np.full(X.shape[0], -1, dtype=int)
            for info in self.per_class_.values():
                cls_label = info["label"]
                mask = labels == cls_label
                if np.any(mask):
                    clus = info["clusterer"]
                    if clus.centroids_ is not None and clus.centroids_.shape[0] > 0:
                        cluster_ids[mask] = clus._assign_labels(X[mask])
            return labels, cluster_ids
        else:
            if self.clusterer_ is None:
                raise RuntimeError("Clusterer missing for predict_regions")
            labels = self.clusterer_._assign_labels(X)
            return labels, labels.copy()

    def plot_pairs(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        max_pairs: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        show_centroids: bool = True,
    ):
        """Simple 2D scatter plots for feature pairs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points used for the scatter plots.
        y : array-like, optional
            Labels used only for coloring points. If ``None`` all points are
            shown in the same color.
        max_pairs : int, optional
            Maximum number of feature pairs to plot. ``None`` plots all pairs.
        feature_names : sequence of str, optional
            Names to use for axis labels. Falls back to ``self.feature_names_``
            or generic ``xj`` names.
        show_centroids : bool, default ``True``
            Whether to draw cluster centroids when available.
        """

        import matplotlib.pyplot as plt  # pragma: no cover - optional dependency

        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if feature_names is None:
            feature_names = (
                self.feature_names_ if self.feature_names_ is not None else [f"x{j}" for j in range(d)]
            )

        pairs = list(combinations(range(d), 2))
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        if not pairs:
            raise ValueError("No hay pares de características para graficar")

        n_pairs = len(pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4))
        if n_pairs == 1:
            axes = [axes]

        for ax, (i, j) in zip(axes, pairs):
            if y is None:
                ax.scatter(X[:, i], X[:, j], s=18, alpha=0.6, label="data")
            else:
                uniq = np.unique(y)
                for cls in uniq:
                    mask = y == cls
                    ax.scatter(X[mask, i], X[mask, j], s=18, alpha=0.8, label=str(cls))
            if show_centroids and self.regions_:
                for reg in self.regions_:
                    ctr = np.asarray(reg.get("center"))
                    if ctr.shape[0] > max(i, j):
                        ax.scatter(ctr[i], ctr[j], marker="X", s=80, color="k", label="centro")
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel(feature_names[j])
            ax.legend(loc="best")

        fig.tight_layout()
        return fig, axes

    def plot_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        grid_res: int = 200,
        contour_levels: Optional[Union[np.ndarray, List[float]]] = None,
        max_paths: int = 20,
        show_paths: bool = True,
    ):
        """Plot class-wise score surfaces.

        Parameters
        ----------
        X, y:
            Data and labels used to generate the plots.
        grid_res:
            Resolution of the evaluation grid per axis.
        contour_levels:
            Optional levels for ``contourf``; defaults to 0..1.
        max_paths:
            Maximum number of optimization paths to show per class.
        show_paths:
            Whether to draw optimization paths (default ``True``).
        """
        X = np.asarray(X, dtype=float)
        uniq = np.unique(y)
        palette = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#a65628",
            "#f781bf",
            "#999999",
        ]
        class_colors = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}

        if contour_levels is None:
            contour_levels = np.linspace(0.0, 1.0, 12)
        else:
            contour_levels = np.asarray(contour_levels, dtype=float)

        for ci, info in self.per_class_.items():
            label = info["label"]
            col = class_colors[label]
            c = info["clusterer"]
            dims2 = info["dims2"]
            d1, d2 = int(dims2[0]), int(dims2[1])
            name1 = self.feature_names_[d1]
            name2 = self.feature_names_[d2]
            score_fn = info["score_fn"]

            x1_min, x1_max = X[:, d1].min(), X[:, d1].max()
            x2_min, x2_max = X[:, d2].min(), X[:, d2].max()
            pad1 = 0.05 * (x1_max - x1_min + 1e-12)
            pad2 = 0.05 * (x2_max - x2_min + 1e-12)
            x1 = np.linspace(x1_min - pad1, x1_max + pad1, grid_res)
            x2 = np.linspace(x2_min - pad2, x2_max + pad2, grid_res)
            XX1, XX2 = np.meshgrid(x1, x2)

            base = np.tile(c.medians_, (grid_res * grid_res, 1))
            base[:, d1] = XX1.ravel()
            base[:, d2] = XX2.ravel()
            ZZ = score_fn(base).reshape(grid_res, grid_res)

            fig, ax = plt.subplots()
            cf = ax.contourf(XX1, XX2, ZZ, levels=contour_levels)
            cb = fig.colorbar(cf, ax=ax)
            cb.set_label(f"P({label})")

            zmin, zmax = float(np.nanmin(ZZ)), float(np.nanmax(ZZ))
            frontier_val = 0.5 if (0.5 >= zmin and 0.5 <= zmax) else 0.5 * (zmin + zmax)
            ax.contour(
                XX1,
                XX2,
                ZZ,
                levels=[frontier_val],
                colors=[col],
                linewidths=2,
            )

            for j, yv in enumerate(uniq):
                mask = y == yv
                ax.scatter(
                    X[mask, d1],
                    X[mask, d2],
                    s=18,
                    alpha=0.8,
                    label=str(yv),
                    color=class_colors[yv],
                    edgecolor="k",
                    linewidths=0.3,
                )

            if c.centroids_ is not None and c.centroids_.shape[0] > 0:
                C2 = c.centroids_[:, [d1, d2]]
                ax.scatter(
                    C2[:, 0],
                    C2[:, 1],
                    marker="X",
                    s=80,
                    color=col,
                    edgecolor="k",
                    label=f"centro {label} ({c.centroids_.shape[0]})",
                )

            if show_paths and c.paths_:
                for i in range(min(max_paths, len(c.paths_))):
                    P = np.array([st["x"][[d1, d2]] for st in c.paths_[i]])
                    ax.plot(P[:, 0], P[:, 1], linewidth=1.0, color="k", alpha=0.5)

            if _HAS_SCIPY_HULL and c.clusters_:
                biggest = max(
                    c.clusters_,
                    key=lambda cl: cl["area_points"].shape[0] if cl["area_points"] is not None else 0,
                )
                ap = biggest["area_points"]
                if ap is not None and ap.shape[0] >= 3:
                    ap2 = ap[:, [d1, d2]]
                    try:
                        hull = ConvexHull(ap2)
                        H = ap2[hull.vertices]
                        ax.plot(
                            np.r_[H[:, 0], H[0, 0]],
                            np.r_[H[:, 1], H[0, 1]],
                            linewidth=2,
                            color=col,
                            alpha=0.9,
                        )
                    except Exception:
                        pass

            handles, labels_ = ax.get_legend_handles_labels()
            frontier_handle = Line2D(
                [], [], color=col, lw=2, label=f"frontera {label} ({frontier_val:.2f})"
            )
            handles.append(frontier_handle)
            ax.legend(handles=handles)

            ax.set_xlabel(name1)
            ax.set_ylabel(name2)
            ax.set_title(f"Prob. clase '{label}' vs ({name1},{name2})")
            fig.tight_layout()

    def plot_pair_3d(
        self,
        X: np.ndarray,
        pair: Tuple[int, int],
        class_label: Optional[Any] = None,
        grid_res: int = 50,
        alpha_surface: float = 0.6,
        engine: str = "matplotlib",
    ):
        """Visualize score surface for a pair of features in 3D."""
        if self.mode_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        d1, d2 = pair
        if self.mode_ == "multiclass":
            if class_label is None:
                raise ValueError("class_label required in multiclass mode")
            info = None
            for v in self.per_class_.values():
                if v["label"] == class_label:
                    info = v
                    break
            if info is None:
                raise ValueError(f"class_label {class_label} not found")
            c = info["clusterer"]
            score_fn = info["score_fn"]
            name1 = self.feature_names_[d1] if self.feature_names_ else f"x{d1}"
            name2 = self.feature_names_[d2] if self.feature_names_ else f"x{d2}"
            zlabel = f"P({class_label})"
            title = f"Prob. clase '{class_label}' vs ({name1},{name2})"
        else:
            if self.clusterer_ is None or self.score_fn_ is None:
                raise RuntimeError("score_fn missing for scalar mode")
            c = self.clusterer_
            score_fn = self.score_fn_
            name1 = self.feature_names_[d1] if self.feature_names_ else f"x{d1}"
            name2 = self.feature_names_[d2] if self.feature_names_ else f"x{d2}"
            zlabel = "score"
            title = f"score vs ({name1},{name2})"

        x1_min, x1_max = X[:, d1].min(), X[:, d1].max()
        x2_min, x2_max = X[:, d2].min(), X[:, d2].max()
        pad1 = 0.05 * (x1_max - x1_min + 1e-12)
        pad2 = 0.05 * (x2_max - x2_min + 1e-12)
        x1 = np.linspace(x1_min - pad1, x1_max + pad1, grid_res)
        x2 = np.linspace(x2_min - pad2, x2_max + pad2, grid_res)
        XX1, XX2 = np.meshgrid(x1, x2)

        base = np.tile(c.medians_, (grid_res * grid_res, 1))
        base[:, d1] = XX1.ravel()
        base[:, d2] = XX2.ravel()
        ZZ = score_fn(base).reshape(grid_res, grid_res)

        if engine == "plotly":
            try:
                import plotly.graph_objects as go
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("plotly is required when engine='plotly'") from exc
            fig = go.Figure(
                data=[
                    go.Surface(x=XX1, y=XX2, z=ZZ, colorscale="Viridis", opacity=alpha_surface)
                ]
            )
            fig.update_layout(
                title=title,
                scene=dict(xaxis_title=name1, yaxis_title=name2, zaxis_title=zlabel),
            )
            return fig

        if engine != "matplotlib":
            raise ValueError("engine must be 'matplotlib' or 'plotly'")

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XX1, XX2, ZZ, cmap="viridis", alpha=alpha_surface)
        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def interpretability_summary(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Summarize cluster regions using :class:`RegionInterpreter`."""
        if self.mode_ is None:
            raise RuntimeError("Model not fitted")

        if feature_names is None:
            if self.feature_names_ is not None:
                feature_names = self.feature_names_
            else:
                if self.mode_ == "multiclass":
                    any_cl = next(iter(self.per_class_.values()))["clusterer"]
                    d = any_cl.centroids_.shape[1] if any_cl.centroids_ is not None else 0
                else:
                    d = self.clusterer_.centroids_.shape[1] if self.clusterer_ and self.clusterer_.centroids_ is not None else 0
                feature_names = [f"x{j}" for j in range(d)]

        regions: List[ClusterRegion] = []
        if self.mode_ == "multiclass":
            for info in self.per_class_.values():
                cls_label = info["label"]
                c = info["clusterer"]
                for cl in c.clusters_ or []:
                    centroid = cl["centroid"]
                    ap = cl.get("area_points")
                    if ap is not None and ap.shape[0] > 0:
                        vecs = ap - centroid
                        radii = np.linalg.norm(vecs, axis=1)
                        mask = radii > 0
                        dirs = vecs[mask] / radii[mask][:, None]
                        radii = radii[mask]
                    else:
                        dirs = np.zeros((0, centroid.shape[0]))
                        radii = np.zeros((0,))
                    inf_pts = centroid[None, :] + dirs * radii[:, None]
                    regions.append(
                        ClusterRegion(
                            cluster_id=int(cl["label"]),
                            label=cls_label,
                            center=centroid,
                            directions=dirs,
                            radii=radii,
                            inflection_points=inf_pts,
                            inflection_slopes=np.zeros_like(radii),
                            peak_value_real=float(cl.get("score", np.nan)),
                            peak_value_norm=float(cl.get("decile", np.nan)) / 10.0,
                        )
                    )
        else:
            if self.clusterer_ is None:
                return pd.DataFrame()
            for cl in self.clusterer_.clusters_ or []:
                centroid = cl["centroid"]
                ap = cl.get("area_points")
                if ap is not None and ap.shape[0] > 0:
                    vecs = ap - centroid
                    radii = np.linalg.norm(vecs, axis=1)
                    mask = radii > 0
                    dirs = vecs[mask] / radii[mask][:, None]
                    radii = radii[mask]
                else:
                    dirs = np.zeros((0, centroid.shape[0]))
                    radii = np.zeros((0,))
                inf_pts = centroid[None, :] + dirs * radii[:, None]
                regions.append(
                    ClusterRegion(
                        cluster_id=int(cl["label"]),
                        label=0,
                        center=centroid,
                        directions=dirs,
                        radii=radii,
                        inflection_points=inf_pts,
                        inflection_slopes=np.zeros_like(radii),
                        peak_value_real=float(cl.get("score", np.nan)),
                        peak_value_norm=float(cl.get("decile", np.nan)) / 10.0,
                    )
                )

        if not regions:
            return pd.DataFrame()

        interpreter = RegionInterpreter(feature_names=feature_names)
        cards = interpreter.summarize(regions)
        try:
            return RegionInterpreter.to_dataframe(cards)
        except Exception:
            return pd.DataFrame(cards)

    def summary_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows_c = []
        rows_ct = []

        if not self.per_class_:
            return pd.DataFrame(), pd.DataFrame()

        for ci in sorted(self.per_class_.keys()):
            info = self.per_class_[ci]
            label = info["label"]
            c = info["clusterer"]

            n_clusters = int(c.centroids_.shape[0]) if getattr(c, "centroids_", None) is not None else 0
            rows_c.append(
                {
                    "class_index": ci,
                    "class_label": label,
                    "n_clusters": n_clusters,
                    "optimize_s": c.timings_.get("optimize_s", np.nan),
                    "total_fit_s": c.timings_.get("total_fit_s", np.nan),
                    "n_score_evals": c.timings_.get("n_score_evals", np.nan),
                }
            )

            if getattr(c, "centroids_", None) is not None and c.centroids_.shape[0] > 0 and getattr(
                c, "centroid_scores_", None
            ) is not None:
                for j, sc in enumerate(c.centroid_scores_):
                    rows_ct.append(
                        {
                            "class_index": ci,
                            "class_label": label,
                            "centroid_id": j,
                            "score": float(sc),
                        }
                    )

        per_class_df = pd.DataFrame(rows_c)
        per_centroid_df = pd.DataFrame(rows_ct)
        return per_class_df, per_centroid_df

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize this instance using ``joblib.dump``.

        Callables used during fitting (e.g. score functions) are not persisted;
        only the attributes required for inference are stored."""

        state: Dict[str, Any] = {
            "random_state": self.random_state,
            "feature_names_": self.feature_names_,
            "clusterer_": self.clusterer_,
            "per_class_": {
                k: {**{kk: vv for kk, vv in v.items() if kk != "score_fn"}, "score_fn": None}
                for k, v in self.per_class_.items()
            },
            "classes_": self.classes_,
            "model_": self.model_,
            "mode_": self.mode_,
            "score_fn_": None,
            "centroids_": getattr(self, "centroids_", None),
            "clusters_": getattr(self, "clusters_", None),
            "timings_": getattr(self, "timings_", None),
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ShuShu":
        """Load a previously saved instance with :meth:`save`."""

        state = joblib.load(filepath)
        obj = cls(random_state=state.get("random_state"))
        for k, v in state.items():
            setattr(obj, k, v)
        return obj

# =============================================================================
#                               EJEMPLOS DE USO
# =============================================================================


def demo_iris_default_logreg():  # pragma: no cover - demo utility
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    sh = ShuShu(
        k=20,
        k_per_dim=1,
        importance_target=0.85,
        rf_estimators=27,
        importance_sample_size=350,
        max_iter=60,
        lr=0.07,
        tol_score=1e-6,
        grad_eps_unit=0.02,
        patience=6,
        grad_mode="spsa",
        assign_all=True,
        random_state=0,
        verbose=0,
    )

    sh.fit(X, y, feature_names=feature_names)
    per_class_df, per_centroid_df = sh.summary_tables()

    print("\n=== Resumen por clase ===")
    print(per_class_df)
    print("\n=== Resumen por centroide ===")
    print(per_centroid_df.sort_values(["class_index", "score"], ascending=[True, False]).head(12))

    sh.plot_classes(X, y, grid_res=220, contour_levels=np.linspace(0, 1, 11), max_paths=25)
    plt.show()


def demo_iris_custom_model():  # pragma: no cover - demo utility
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    logreg = LogisticRegression(max_iter=500, C=0.8, solver="lbfgs", random_state=42)
    logreg.fit(X, y)

    sh = ShuShu(random_state=42)
    sh.fit(X, y, feature_names=feature_names, score_model=logreg)
    sh.plot_classes(X, y)
    plt.show()


def demo_iris_custom_score_fn():  # pragma: no cover - demo utility
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    aux_lr = LogisticRegression(max_iter=500, solver="lbfgs", random_state=7).fit(X, y)

    def score_fn_multi(Z: np.ndarray) -> np.ndarray:
        return aux_lr.predict_proba(Z)

    sh = ShuShu(random_state=7)
    sh.fit(X, y, feature_names=feature_names, score_fn_multi=score_fn_multi)
    sh.plot_classes(X, y)
    plt.show()


if __name__ == "__main__":  # pragma: no cover - demo execution
    print(">> Demo Iris con LogisticRegression por defecto (una corrida por clase)...")
    demo_iris_default_logreg()

    # Descomentar para probar variantes:
    # print("\n>> Demo Iris con modelo personalizado (LogReg con C=0.8)...")
    # demo_iris_custom_model()

    # print("\n>> Demo Iris con score_fn_multi (custom)...")
    # demo_iris_custom_score_fn()
