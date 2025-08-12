
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted


# =========================
# Numerical utilities
# =========================

def _rng(random_state: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(None if random_state is None else int(random_state))

def sample_unit_directions_gaussian(n: int, dim: int, random_state: Optional[int] = 42) -> np.ndarray:
    """Approximately uniform directions on :math:`S^{dim-1}` by normalizing Gaussian samples."""
    rng = _rng(random_state)
    U = rng.normal(size=(n, dim))
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return U

def sample_unit_directions_circle(n: int) -> np.ndarray:
    """2D: ``n`` evenly spaced angles."""
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def sample_unit_directions_sph_fibo(n: int) -> np.ndarray:
    """3D: nearly equal-area points on :math:`S^2` (spherical Fibonacci)."""
    ga = (1 + 5 ** 0.5) / 2  # golden ratio
    k = np.arange(n)
    z = 1 - (2*k + 1)/n
    phi = 2*np.pi * k / (ga)
    r = np.sqrt(np.maximum(0.0, 1 - z**2))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def finite_diff_gradient(f, x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Central difference gradient."""
    d = x.shape[0]
    g = np.zeros(d, dtype=float)
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0
        g[i] = (f(x + eps*e) - f(x - eps*e)) / (2.0*eps)
    return g

def project_step_with_barrier(x: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Zero out gradient components that push outside the domain when on the boundary.
    Prevents escaping and forces movement along other variables."""
    step = g.copy()
    for i in range(len(x)):
        if (x[i] <= lo[i] + 1e-12 and step[i] < 0) or (x[i] >= hi[i] - 1e-12 and step[i] > 0):
            step[i] = 0.0
    return step

def gradient_ascent(
    f, x0: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray],
    lr: float = 0.1, max_iter: int = 200, tol: float = 1e-5, eps_grad: float = 1e-2
) -> np.ndarray:
    """Gradient ascent with backtracking and boundary barriers."""
    lo, hi = bounds
    x = x0.copy()
    best = f(x)
    for _ in range(max_iter):
        g = finite_diff_gradient(f, x, eps=eps_grad)
        if np.linalg.norm(g) < tol:
            break
        g = project_step_with_barrier(x, g, lo, hi)
        if np.allclose(g, 0.0):
            break
        step = lr * g / (np.linalg.norm(g) + 1e-12)
        x_new = np.clip(x + step, lo, hi)
        v_new = f(x_new)
        if v_new <= best + 1e-12:
            # backtracking
            x_try = np.clip(x + 0.5*step, lo, hi)
            v_try = f(x_try)
            if v_try <= best + 1e-12:
                break
            x, best = x_try, v_try
        else:
            x, best = x_new, v_new
    return x

def second_diff(arr: np.ndarray) -> np.ndarray:
    s = np.zeros_like(arr)
    if len(arr) >= 3:
        s[1:-1] = arr[:-2] - 2*arr[1:-1] + arr[2:]
    return s

def find_inflection(ts: np.ndarray, vals: np.ndarray, direction: str) -> Tuple[float, float]:
    """Return ``(t_inf, slope_at_inf)``.

    Parameters
    ----------
    direction : {'center_out', 'outside_in'}
        Scanning strategy.

    Returns
    -------
    t_inf : float
        Parameter ``t`` in ``[0, T]``.
    slope_at_inf : float
        ``df/dt`` at ``t_inf`` (sign consistent with increasing ``t``).
    """
    if direction not in ("center_out", "outside_in"):
        raise ValueError("direction must be 'center_out' or 'outside_in'.")

    # Prepare series according to direction
    if direction == "outside_in":
        ts_scan = ts[::-1]
        vals_scan = vals[::-1]
    else:
        ts_scan = ts
        vals_scan = vals

    sd = second_diff(vals_scan)

    idx = None
    for j in range(1, len(sd)):
        if sd[j] >= 0 and sd[j-1] < 0:
            idx = j
            break

    def slope_at(idx0: int) -> float:
        # derivada central en el eje 'scan' (t creciente)
        if idx0 <= 0:
            return (vals_scan[1] - vals_scan[0]) / (ts_scan[1] - ts_scan[0] + 1e-12)
        if idx0 >= len(ts_scan)-1:
            return (vals_scan[-1] - vals_scan[-2]) / (ts_scan[-1] - ts_scan[-2] + 1e-12)
        return (vals_scan[idx0+1] - vals_scan[idx0-1]) / (ts_scan[idx0+1] - ts_scan[idx0-1] + 1e-12)

    if idx is not None and 1 <= idx < len(ts_scan):
        # interpolate exact position between idx-1 and idx
        j0, j1 = idx-1, idx
        a0, a1 = sd[j0], sd[j1]
        frac = float(np.clip(-a0 / (a1 - a0 + 1e-12), 0.0, 1.0))
        t_scan = ts_scan[j0] + frac * (ts_scan[j1] - ts_scan[j0])
        # slope (use nearest index)
        j_star = j0 if frac < 0.5 else j1
        m_scan = slope_at(j_star)
    else:
        # fallback: 50% drop from val[0]
        target = vals_scan[0] * 0.5
        t_scan = ts_scan[-1]
        m_scan = slope_at(len(ts_scan)//2)
        for j in range(1, len(vals_scan)):
            if vals_scan[j] <= target:
                t0, t1 = ts_scan[j-1], ts_scan[j]
                v0, v1 = vals_scan[j-1], vals_scan[j]
                α = float(np.clip((target - v0) / (v1 - v0 + 1e-12), 0.0, 1.0))
                t_scan = t0 + α*(t1 - t0)
                m_scan = slope_at(j)
                break

    # Convierte a t absoluto (0..T) coherente con ts original
    t_abs = t_scan if direction == "center_out" else (ts[-1] - t_scan)
    return float(t_abs), float(m_scan)


# =========================
# Output structures
# =========================

@dataclass
class ClusterRegion:
    label: Union[int, str]                 # class (or "NA" in regression)
    center: np.ndarray                     # local maximum
    directions: np.ndarray                 # (n_rays, d)
    radii: np.ndarray                      # (n_rays,)
    inflection_points: np.ndarray          # (n_rays, d)
    inflection_slopes: np.ndarray          # (n_rays,) df/dt at inflection
    peak_value_real: float                 # real prob/value at the center
    peak_value_norm: float                 # normalized value at the center [0,1]


# =========================
# Ray sampling plan
# =========================

def rays_count_auto(dim: int, base_2d: int = 8) -> int:
    """Suggested number of rays depending on dimension.

    - 2D: ``base_2d`` (default 8)
    - 3D: ``N ≈ 2 / (1 - cos(π/base_2d))`` (cap coverage; ~26 if ``base_2d=8``)
    - >3D: keep the cost bounded by using subspaces → return a small global count.
    """
    if dim <= 1:
        return 1
    if dim == 2:
        return int(base_2d)
    if dim == 3:
        theta = math.pi / base_2d  # ≈ 2D-like angular separation
        n = max(12, int(math.ceil(2.0 / max(1e-9, (1 - math.cos(theta))))))
        return min(64, n)  # cota superior razonable
    # For >3D return a few global ones; the rest via subspaces
    return 8

def generate_directions(dim: int, base_2d: int, random_state: Optional[int] = 42,
                        max_subspaces: int = 20) -> np.ndarray:
    """Set of directions.

    - 2D: 8 equally spaced angles (default)
    - 3D: ``~N`` from the cap formula + spherical Fibonacci
    - >3D: mixture of:
        * a few global (Gaussian) directions, and
        * directions embedded in 2D/3D subspaces (all or sampled)
    """
    if dim == 1:
        return np.array([[1.0]])
    if dim == 2:
        return sample_unit_directions_circle(rays_count_auto(2, base_2d))
    if dim == 3:
        n = rays_count_auto(3, base_2d)
        return sample_unit_directions_sph_fibo(n)

    # d > 3: subespacios
    rng = _rng(random_state)
    dirs = []

    # algunos globales
    dirs.append(sample_unit_directions_gaussian(rays_count_auto(dim, base_2d), dim, random_state))

    # choose subspaces of size 3 (or 2 if dim=4 and you want cheaper)
    sub_dim = 3 if dim >= 3 else 2
    total_combos = math.comb(dim, sub_dim)
    if max_subspaces >= total_combos:
        combos = list(itertools.combinations(range(dim), sub_dim))
    else:
        combos = set()
        while len(combos) < max_subspaces:
            combo = tuple(sorted(rng.choice(dim, size=sub_dim, replace=False)))
            combos.add(combo)
        combos = list(combos)
    rng.shuffle(combos)

    # nº de rays por subespacio
    if sub_dim == 3:
        n_local = rays_count_auto(3, base_2d)
        local_dirs = sample_unit_directions_sph_fibo(n_local)
    else:
        n_local = rays_count_auto(2, base_2d)
        local_dirs = sample_unit_directions_circle(n_local)

    for idxs in combos:
        block = np.zeros((n_local, dim))
        block[:, idxs] = local_dirs
        dirs.append(block)

    D = np.vstack(dirs)
    # normaliza por seguridad
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    return D


# =========================
# Clusterizador modal
# =========================

class ModalBoundaryClustering(BaseEstimator):
    """SheShe: Smart High-dimensional Edge Segmentation & Hyperboundary Explorer

    Clusters around local maxima on the probability surface (classification) or
    the predicted value (regression). Compatible with sklearn.

    Version 2 highlights:
      - Dynamic number of rays: 2D→8; 3D≈26; >3D reduced with 2D/3D subspaces
        plus a few global ones.
      - ``direction``: 'center_out' (default) or 'outside_in' to locate the
        inflection.
      - Slope at the inflection point (df/dt).
      - Ascent with boundary barriers.
    """

    def __init__(
        self,
        base_estimator: Optional['BaseEstimator'] = None,
        task: str = "classification",  # "classification" | "regression"
        base_2d_rays: int = 8,
        direction: str = "center_out",
        scan_radius_factor: float = 3.0,   # multiples of the global std
        scan_steps: int = 64,
        grad_lr: float = 0.2,
        grad_max_iter: int = 200,
        grad_tol: float = 1e-5,
        grad_eps: float = 1e-2,
        n_max_seeds: int = 5,
        random_state: Optional[int] = 42,
        max_subspaces: int = 20,
        verbose: bool = False,
        save_labels: bool = False,
        out_dir: Optional[Union[str, Path]] = None,
    ):
        if scan_steps < 2:
            raise ValueError("scan_steps must be at least 2")
        if n_max_seeds < 1:
            raise ValueError("n_max_seeds must be at least 1")

        self.base_estimator = base_estimator
        self.task = task
        self.base_2d_rays = base_2d_rays
        self.direction = direction
        self.scan_radius_factor = scan_radius_factor
        self.scan_steps = scan_steps
        self.grad_lr = grad_lr
        self.grad_max_iter = grad_max_iter
        self.grad_tol = grad_tol
        self.grad_eps = grad_eps
        self.n_max_seeds = n_max_seeds
        self.random_state = random_state
        self.max_subspaces = max_subspaces
        self.verbose = verbose
        self.save_labels = save_labels
        self.out_dir = Path(out_dir) if out_dir is not None else None

    # ---------- helpers ----------

    def _fit_estimator(self, X: np.ndarray, y: Optional[np.ndarray]):
        if self.base_estimator is None:
            if self.task == "classification":
                est = LogisticRegression(multi_class="auto", max_iter=1000)
            else:
                est = GradientBoostingRegressor(random_state=self.random_state)
        else:
            est = clone(self.base_estimator)

        self.pipeline_ = Pipeline([("scaler", StandardScaler()), ("estimator", est)])
        self.pipeline_.fit(X, y if y is not None else np.zeros(len(X)))
        self.estimator_ = self.pipeline_.named_steps["estimator"]
        self.scaler_ = self.pipeline_.named_steps["scaler"]

    def _predict_value_real(self, X: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        if self.task == "classification":
            if class_idx is None:
                raise ValueError("class_idx required for classification.")
            if hasattr(self.estimator_, "predict_proba"):
                proba = self.estimator_.predict_proba(Xs)
                return proba[:, class_idx]
            if hasattr(self.estimator_, "decision_function"):
                scores = self.estimator_.decision_function(Xs)
                if scores.ndim == 1:
                    # binary case -> two classes
                    if class_idx not in (0, 1):
                        raise ValueError("class_idx must be 0 or 1 for binary decision_function")
                    return scores if class_idx == 1 else -scores
                return scores[:, class_idx]
            raise NotImplementedError(
                "Base estimator must implement predict_proba or decision_function"
            )
        else:
            return self.estimator_.predict(Xs)

    def _build_value_fn(self, class_idx: Optional[int], norm_stats: Dict[str, float]):
        vmin, vmax = norm_stats["min"], norm_stats["max"]
        rng = vmax - vmin if vmax > vmin else 1.0
        def f(x: np.ndarray) -> float:
            val = float(self._predict_value_real(x.reshape(1, -1), class_idx=class_idx)[0])
            return (val - vmin) / rng
        return f

    def _bounds_from_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        span = hi - lo
        return lo - 0.05*span, hi + 0.05*span

    def _choose_seeds(self, X: np.ndarray, f, k: int) -> np.ndarray:
        vals = np.array([f(x) for x in X])
        idx = np.argsort(-vals)[:k]
        return X[idx]

    def _find_maximum(self, X: np.ndarray, f, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        seeds = self._choose_seeds(X, f, min(self.n_max_seeds, len(X)))
        if len(seeds) == 0:
            return X[0]
        best_x, best_v = seeds[0].copy(), f(seeds[0])
        for s in seeds:
            x_star = gradient_ascent(
                f, s, bounds, lr=self.grad_lr, max_iter=self.grad_max_iter,
                tol=self.grad_tol, eps_grad=self.grad_eps
            )
            v = f(x_star)
            if v > best_v:
                best_x, best_v = x_star, v
        return best_x

    def _scan_radii(self, center: np.ndarray, f, directions: np.ndarray, X_std: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each direction ``u``: radial scan ``t∈[0,T]`` and first inflection point
        according to ``direction``. Returns ``(radii, points, slopes)``.
        """
        d = center.shape[0]
        T = float(self.scan_radius_factor * np.linalg.norm(X_std))
        ts = np.linspace(0.0, T, self.scan_steps)

        radii = np.zeros(len(directions), dtype=float)
        pts = np.zeros((len(directions), d), dtype=float)
        slopes = np.zeros(len(directions), dtype=float)

        for i, u in enumerate(directions):
            vals = np.array([f(center + t*u) for t in ts], dtype=float)
            r, m = find_inflection(ts, vals, self.direction)
            radii[i] = r
            pts[i] = center + r*u
            slopes[i] = m
        return radii, pts, slopes

    def _build_norm_stats(self, X: np.ndarray, class_idx: Optional[int]) -> Dict[str, float]:
        vals = self._predict_value_real(X, class_idx=class_idx)
        return {"min": float(np.min(vals)), "max": float(np.max(vals))}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _maybe_save_labels(self, labels: np.ndarray, label_path: Optional[Union[str, Path]]) -> None:
        if label_path is None:
            if not self.save_labels:
                return
            label_path = Path(f"{self.__class__.__name__}.labels")
            if self.out_dir is not None:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                label_path = self.out_dir / label_path
        else:
            label_path = Path(label_path)
            if label_path.suffix != ".labels":
                label_path = label_path.with_suffix(".labels")
        try:
            np.savetxt(label_path, labels, fmt="%s")
        except Exception as exc:  # pragma: no cover - auxiliary logging
            self._log(f"Could not save labels to {label_path}: {exc}")

    # ---------- Public API ----------

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        start = time.perf_counter()
        try:
            X = np.asarray(X, dtype=float)
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None

            if self.task == "classification":
                if y is None:
                    raise ValueError("y cannot be None when task='classification'")
                y_arr = np.asarray(y)
                if np.unique(y_arr).size < 2:
                    raise ValueError("y must contain at least two classes for classification")

            self._fit_estimator(X, y)
            lo, hi = self._bounds_from_data(X)
            X_std = np.std(X, axis=0) + 1e-12
            dirs = generate_directions(self.n_features_in_, self.base_2d_rays, self.random_state, self.max_subspaces)

            self.regions_: List[ClusterRegion] = []
            self.classes_ = None

            if self.task == "classification":
                _ = self.pipeline_.predict(X[:2])  # asegura classes_
                self.classes_ = self.estimator_.classes_
                for ci, label in enumerate(self.classes_):
                    stats = self._build_norm_stats(X, class_idx=ci)
                    f = self._build_value_fn(class_idx=ci, norm_stats=stats)
                    center = self._find_maximum(X, f, (lo, hi))
                    radii, infl, slopes = self._scan_radii(center, f, dirs, X_std)
                    peak_real = float(self._predict_value_real(center.reshape(1, -1), class_idx=ci)[0])
                    peak_norm = float(f(center))
                    self.regions_.append(ClusterRegion(
                        label=label, center=center, directions=dirs, radii=radii,
                        inflection_points=infl, inflection_slopes=slopes,
                        peak_value_real=peak_real, peak_value_norm=peak_norm
                    ))
            else:
                stats = self._build_norm_stats(X, class_idx=None)
                f = self._build_value_fn(class_idx=None, norm_stats=stats)
                center = self._find_maximum(X, f, (lo, hi))
                radii, infl, slopes = self._scan_radii(center, f, dirs, X_std)
                peak_real = float(self._predict_value_real(center.reshape(1, -1), class_idx=None)[0])
                peak_norm = float(f(center))
                self.regions_.append(ClusterRegion(
                    label="NA", center=center, directions=dirs, radii=radii,
                    inflection_points=infl, inflection_slopes=slopes,
                    peak_value_real=peak_real, peak_value_norm=peak_norm
                ))
        except Exception as exc:
            self._log(f"Error in fit: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"fit completed in {runtime:.4f}s")
        return self

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return the prediction for ``X``.

        Common *sklearn* shortcut equivalent to calling :meth:`fit` and then
        :meth:`predict` on the same data.
        """
        self.fit(X, y)
        return self.predict(X)

    def _membership_matrix(self, X: np.ndarray) -> np.ndarray:
        """Build the membership matrix for the discovered regions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to evaluate in the original space.

        Returns
        -------
        ndarray of shape (n_samples, n_regions)
            Binary matrix ``R`` where ``R[i, k] = 1`` indicates sample ``i`` falls
            inside region ``k``.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> sh = ModalBoundaryClustering().fit(X, y)
        >>> sh._membership_matrix(X).shape
        (150, 3)
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        R = np.zeros((n, len(self.regions_)), dtype=int)
        for k, reg in enumerate(self.regions_):
            if reg.directions.size == 0:
                warnings.warn(
                    "Región sin direcciones; se marca como fuera de la región",
                    RuntimeWarning,
                )
                continue
            c = reg.center
            V = X - c
            norms = np.linalg.norm(V, axis=1) + 1e-12
            U = V / norms[:, None]
            dots = U @ reg.directions.T
            idx = np.argmax(dots, axis=1)
            r_boundary = reg.radii[idx]
            R[:, k] = (norms <= r_boundary + 1e-12).astype(int)
        return R

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        label_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Prediction for ``X``.

        Classification → label of the corresponding region (with a fallback to
        the base estimator if the point is outside all regions). Regression →
        predicted value from the base estimator.
        """
        start = time.perf_counter()
        try:
            check_is_fitted(self, "regions_")
            X = np.asarray(X, dtype=float)
            if self.task == "classification":
                M = self._membership_matrix(X)
                labels = np.array([reg.label for reg in self.regions_])
                pred = np.empty(len(X), dtype=labels.dtype)
                some = M.sum(axis=1) > 0
                for i in np.where(some)[0]:
                    ks = np.where(M[i] == 1)[0]
                    if len(ks) == 1:
                        pred[i] = labels[ks[0]]
                    else:
                        dists = [np.linalg.norm(X[i] - self.regions_[k].center) for k in ks]
                        pred[i] = labels[ks[np.argmin(dists)]]
                none = ~some
                if np.any(none):
                    base_pred = self.pipeline_.predict(X[none])
                    pred[none] = base_pred
                result = pred
            else:
                result = self.pipeline_.predict(X)
        except Exception as exc:
            self._log(f"Error in predict: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict completed in {runtime:.4f}s")
        self._maybe_save_labels(result, label_path)
        return result

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Classification: class probabilities or decision scores.

        For classification, this method returns ``predict_proba`` from the base
        estimator when available. If ``predict_proba`` is absent but
        ``decision_function`` exists, its output is returned instead (for binary
        problems the two-class scores are stacked as ``[-s, s]``). If neither is
        implemented a :class:`NotImplementedError` is raised.

        Regression: normalized value in ``[0, 1]``.
        """
        start = time.perf_counter()
        try:
            check_is_fitted(self, "regions_")
            Xs = self.scaler_.transform(np.asarray(X, dtype=float))
            if self.task == "classification":
                if hasattr(self.estimator_, "predict_proba"):
                    result = self.estimator_.predict_proba(Xs)
                elif hasattr(self.estimator_, "decision_function"):
                    scores = self.estimator_.decision_function(Xs)
                    if scores.ndim == 1:
                        scores = scores.reshape(-1, 1)
                        result = np.column_stack([-scores, scores])
                    else:
                        result = scores
                else:
                    raise NotImplementedError(
                        "Base estimator must implement predict_proba or decision_function"
                    )
            else:
                vals = self.estimator_.predict(Xs)
                vmin = min(reg.peak_value_real for reg in self.regions_)
                vmax = max(reg.peak_value_real for reg in self.regions_)
                rng = vmax - vmin if vmax > vmin else 1.0
                result = ((vals - vmin) / rng).reshape(-1, 1)
        except Exception as exc:
            self._log(f"Error in predict_proba: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict_proba completed in {runtime:.4f}s")
        return result

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Decision values from the base estimator with automatic fallback.

        If the underlying estimator provides :meth:`decision_function`, that
        output is returned. Otherwise we fall back to :meth:`predict_proba` for
        classification or :meth:`predict` for regression.

        Parameters
        ----------
        X : array-like
            Samples to evaluate.

        Returns
        -------
        ndarray
            Scores, probabilities or predictions depending on the fallback.

        Examples
        --------
        Classification with an estimator implementing ``decision_function``::

            >>> from sklearn.datasets import load_iris
            >>> from sklearn.linear_model import LogisticRegression
            >>> X, y = load_iris(return_X_y=True)
            >>> sh = ModalBoundaryClustering(LogisticRegression(max_iter=200),
            ...                             task="classification").fit(X, y)
            >>> sh.decision_function(X[:2]).shape
            (2, 3)

        Classification with a model lacking ``decision_function`` (uses
        ``predict_proba``)::

            >>> from sklearn.ensemble import RandomForestClassifier
            >>> sh = ModalBoundaryClustering(RandomForestClassifier(),
            ...                             task="classification").fit(X, y)
            >>> sh.decision_function(X[:2]).shape
            (2, 3)

        For regression the output comes from ``predict``::

            >>> from sklearn.datasets import make_regression
            >>> from sklearn.ensemble import RandomForestRegressor
            >>> X, y = make_regression(n_samples=10, n_features=4, random_state=0)
            >>> sh = ModalBoundaryClustering(RandomForestRegressor(),
            ...                             task="regression").fit(X, y)
            >>> sh.decision_function(X[:2]).shape
            (2,)
        """

        start = time.perf_counter()
        try:
            check_is_fitted(self, "regions_")
            Xs = self.scaler_.transform(np.asarray(X, dtype=float))
            if hasattr(self.estimator_, "decision_function"):
                result = self.estimator_.decision_function(Xs)
            else:
                if self.task == "classification" and hasattr(self.estimator_, "predict_proba"):
                    result = self.estimator_.predict_proba(Xs)
                else:
                    result = self.estimator_.predict(Xs)
        except Exception as exc:
            self._log(f"Error in decision_function: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"decision_function completed in {runtime:.4f}s")
        return result

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> float:
        """Return the sklearn metric delegating to the internal pipeline."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.score(np.asarray(X, dtype=float), y)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the current instance to ``filepath`` using ``joblib.dump``."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ModalBoundaryClustering":
        """Load a previously saved instance with :meth:`save`."""
        return joblib.load(filepath)

    def interpretability_summary(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Summarize centers and inflection points of each region in a ``DataFrame``.

        Parameters
        ----------
        feature_names : list of str, optional
            Feature names of length ``(n_features,)``. When ``None``, use the
            names seen during fitting or ``coord_i`` if unavailable.

        Returns
        -------
        DataFrame
            Table with one row per centroid and inflection point. Contains the
            columns ``['Type', 'Distance', 'Category', 'real_value',
            'norm_value', 'slope']`` plus one column per feature.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> sh = ModalBoundaryClustering().fit(X, y)
        >>> sh.interpretability_summary().head()

        """
        check_is_fitted(self, "regions_")
        d = self.n_features_in_
        if feature_names is None:
            feature_names = self.feature_names_in_ or [f"coord_{i}" for i in range(d)]

        rows = []
        for reg in self.regions_:
            # centroide
            row_c = {
                "Tipo": "centroide",
                "Distancia": 0.0,
                "Categoria": reg.label,
                "valor_real": reg.peak_value_real,
                "valor_norm": reg.peak_value_norm,
                "pendiente": np.nan,
            }
            for j in range(d):
                row_c[feature_names[j]] = float(reg.center[j])
            rows.append(row_c)
            # inflection points
            if self.task == "classification":
                cls_index = list(self.estimator_.classes_).index(reg.label)
            else:
                cls_index = None
            for r, p, m in zip(reg.radii, reg.inflection_points, reg.inflection_slopes):
                row_i = {
                    "Tipo": "inflexion_point",
                    "Distancia": float(r),
                    "Categoria": reg.label,
                    "valor_real": float(self._predict_value_real(p.reshape(1, -1), class_idx=cls_index)[0]),
                    "valor_norm": np.nan,
                    "pendiente": float(m),
                }
                for j in range(d):
                    row_i[feature_names[j]] = float(p[j])
                rows.append(row_i)
        return pd.DataFrame(rows)

    # -------- Visualization (2D pairs) --------

    def _plot_single_pair_classif(self, X: np.ndarray, y: np.ndarray, pair: Tuple[int, int],
                                  class_colors: Dict[Any, str], grid_res: int = 200, alpha_surface: float = 0.6):
        """Draw the probability surface for a pair of features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Class labels.
        pair : tuple of int
            Indices ``(i, j)`` of the features to plot.
        class_colors : dict
            Mapping from class to color for scatter points.
        grid_res : int, default=200
            Resolution of the mesh used for the surface.
        alpha_surface : float, default=0.6
            Surface transparency.

        Returns
        -------
        None

        Examples
        --------
        >>> sh = ModalBoundaryClustering().fit(X, y)
        >>> sh._plot_single_pair_classif(X, y, (0, 1), {0: 'red', 1: 'blue'})
        """
        i, j = pair
        xi, xj = X[:, i], X[:, j]
        xi_lin = np.linspace(xi.min(), xi.max(), grid_res)
        xj_lin = np.linspace(xj.min(), xj.max(), grid_res)
        XI, XJ = np.meshgrid(xi_lin, xj_lin)

        for reg in self.regions_:
            label = reg.label
            Z = np.zeros_like(XI, dtype=float)
            for r in range(grid_res):
                X_full = np.tile(np.mean(X, axis=0), (grid_res, 1))
                X_full[:, i] = XI[r, :]
                X_full[:, j] = XJ[r, :]
                Z[r, :] = self._predict_value_real(X_full, class_idx=list(self.classes_).index(label))

            plt.figure(figsize=(6, 5))
            plt.title(f"Prob. clase '{label}' vs (feat {i},{j})")
            cf = plt.contourf(XI, XJ, Z, levels=20, alpha=alpha_surface)
            plt.colorbar(cf, label=f"P({label})")

            # puntos
            for c in self.classes_:
                mask = (y == c)
                plt.scatter(X[mask, i], X[mask, j], s=18, c=class_colors[c], label=str(c), edgecolor='k', linewidths=0.3)

            # frontera (poli 2D)
            pts = reg.inflection_points[:, [i, j]]
            ctr = reg.center[[i, j]]
            ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
            order = np.argsort(ang)
            poly = pts[order]
            col = class_colors[label]
            plt.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=col, linewidth=2, label=f"frontera {label}")
            plt.scatter(ctr[0], ctr[1], c=col, marker='X', s=80, label=f"centro {label}")

            plt.xlabel(f"feat {i}")
            plt.ylabel(f"feat {j}")
            plt.legend(loc="best")
            plt.tight_layout()

    def _plot_single_pair_reg(self, X: np.ndarray, pair: Tuple[int, int],
                              grid_res: int = 200, alpha_surface: float = 0.6):
        """Draw the predicted-value surface for a pair of features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        pair : tuple of int
            Indices ``(i, j)`` of the features to plot.
        grid_res : int, default=200
            Resolution of the mesh used for the surface.
        alpha_surface : float, default=0.6
            Surface transparency.

        Returns
        -------
        None

        Examples
        --------
        >>> sh = ModalBoundaryClustering(task="regression").fit(X, y)
        >>> sh._plot_single_pair_reg(X, (0, 1))
        """
        i, j = pair
        xi, xj = X[:, i], X[:, j]
        xi_lin = np.linspace(xi.min(), xi.max(), grid_res)
        xj_lin = np.linspace(xj.min(), xj.max(), grid_res)
        XI, XJ = np.meshgrid(xi_lin, xj_lin)

        Z = np.zeros_like(XI, dtype=float)
        for r in range(grid_res):
            X_full = np.tile(np.mean(X, axis=0), (grid_res, 1))
            X_full[:, i] = XI[r, :]
            X_full[:, j] = XJ[r, :]
            Z[r, :] = self._predict_value_real(X_full, class_idx=None)

        plt.figure(figsize=(6, 5))
        plt.title(f"Valor predicho vs (feat {i},{j})")
        cf = plt.contourf(XI, XJ, Z, levels=20, alpha=alpha_surface)
        plt.colorbar(cf, label="y_pred")

        reg = self.regions_[0]
        pts = reg.inflection_points[:, [i, j]]
        ctr = reg.center[[i, j]]
        ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
        order = np.argsort(ang)
        poly = pts[order]
        plt.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color="black", linewidth=2, label="frontera")
        plt.scatter(ctr[0], ctr[1], c="black", marker='X', s=80, label="centro")

        plt.xlabel(f"feat {i}")
        plt.ylabel(f"feat {j}")
        plt.legend(loc="best")
        plt.tight_layout()

    def plot_pairs(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None,
                   max_pairs: Optional[int] = None):
        """Visualize 2D surfaces for feature pairs.

        Generates one figure for each ``(i, j)`` feature combination up to
        ``max_pairs``. In classification, the probability of each class is shown;
        in regression, the predicted value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to set the range of each axis.
        y : ndarray of shape (n_samples,), optional
            True labels; required when ``task='classification'``.
        max_pairs : int, optional
            Maximum number of combinations to plot. If ``None`` all possible
            combinations are generated.

        Returns
        -------
        None

        Examples
        --------
        Classification::

            >>> from sklearn.datasets import load_iris
            >>> X, y = load_iris(return_X_y=True)
            >>> sh = ModalBoundaryClustering().fit(X, y)
            >>> sh.plot_pairs(X, y, max_pairs=1)

        Regression::

            >>> from sklearn.datasets import make_regression
            >>> X, y = make_regression(n_samples=50, n_features=3, random_state=0)
            >>> sh = ModalBoundaryClustering(task="regression").fit(X, y)
            >>> sh.plot_pairs(X, max_pairs=1)
        """
        check_is_fitted(self, "regions_")
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        pairs = list(itertools.combinations(range(d), 2))
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        if self.task == "classification":
            assert y is not None, "y required to plot classification."
            assert len(y) == len(X), "X e y deben tener la misma longitud."
            palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                       "#ff7f00", "#a65628", "#f781bf", "#999999"]
            class_colors = {c: palette[i % len(palette)] for i, c in enumerate(self.classes_)}
            for pair in pairs:
                self._plot_single_pair_classif(X, y, pair, class_colors)
        else:
            for pair in pairs:
                self._plot_single_pair_reg(X, pair)
