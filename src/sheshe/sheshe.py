
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

from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted


# =========================
# Utilidades numéricas
# =========================

def _rng(random_state: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(None if random_state is None else int(random_state))

def sample_unit_directions_gaussian(n: int, dim: int, random_state: Optional[int] = 42) -> np.ndarray:
    """Direcciones ~uniformes en S^{dim-1} normalizando gaussianas."""
    rng = _rng(random_state)
    U = rng.normal(size=(n, dim))
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return U

def sample_unit_directions_circle(n: int) -> np.ndarray:
    """2D: n ángulos equiespaciados."""
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])

def sample_unit_directions_sph_fibo(n: int) -> np.ndarray:
    """3D: puntos casi equi-área en S^2 (Fibonacci esférico)."""
    ga = (1 + 5 ** 0.5) / 2  # golden ratio
    k = np.arange(n)
    z = 1 - (2*k + 1)/n
    phi = 2*np.pi * k / (ga)
    r = np.sqrt(np.maximum(0.0, 1 - z**2))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def finite_diff_gradient(f, x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Gradiente por diferencia central."""
    d = x.shape[0]
    g = np.zeros(d, dtype=float)
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0
        g[i] = (f(x + eps*e) - f(x - eps*e)) / (2.0*eps)
    return g

def project_step_with_barrier(x: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Anula componentes del gradiente que empujan fuera del dominio cuando estamos en el borde.
    Evita 'escaparse' y fuerza el movimiento por otras variables.
    """
    step = g.copy()
    for i in range(len(x)):
        if (x[i] <= lo[i] + 1e-12 and step[i] < 0) or (x[i] >= hi[i] - 1e-12 and step[i] > 0):
            step[i] = 0.0
    return step

def gradient_ascent(
    f, x0: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray],
    lr: float = 0.1, max_iter: int = 200, tol: float = 1e-5, eps_grad: float = 1e-2
) -> np.ndarray:
    """Ascenso con backtracking y barreras en los límites."""
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
    """
    Devuelve (t_inf, slope_at_inf). direction: 'center_out' | 'outside_in'.
    - t_inf: parámetro t en [0,T]
    - slope_at_inf: df/dt en t_inf (signo coherente con t creciente).
    """
    if direction not in ("center_out", "outside_in"):
        raise ValueError("direction debe ser 'center_out' u 'outside_in'.")

    # Prepara serie según dirección
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
        # interpola posición exacta entre idx-1 e idx
        j0, j1 = idx-1, idx
        a0, a1 = sd[j0], sd[j1]
        frac = float(np.clip(-a0 / (a1 - a0 + 1e-12), 0.0, 1.0))
        t_scan = ts_scan[j0] + frac * (ts_scan[j1] - ts_scan[j0])
        # pendiente (usar índice más cercano)
        j_star = j0 if frac < 0.5 else j1
        m_scan = slope_at(j_star)
    else:
        # fallback: 50% de caída desde val[0]
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
# Estructuras de salida
# =========================

@dataclass
class ClusterRegion:
    label: Union[int, str]                 # clase (o "NA" en regresión)
    center: np.ndarray                     # máximo local
    directions: np.ndarray                 # (n_rays, d)
    radii: np.ndarray                      # (n_rays,)
    inflection_points: np.ndarray          # (n_rays, d)
    inflection_slopes: np.ndarray          # (n_rays,) df/dt en inflexión
    peak_value_real: float                 # prob/valor real en el centro
    peak_value_norm: float                 # valor normalizado en el centro [0,1]


# =========================
# Plan de muestreo de rays
# =========================

def rays_count_auto(dim: int, base_2d: int = 8) -> int:
    """
    Nº de rays sugerido según dimensión:
      - 2D: base_2d (por defecto 8)
      - 3D: N ≈ 2 / (1 - cos(π/base_2d))  (cobertura por caps; ~26 si base_2d=8)
      - >3D: mantener coste acotado: usamos subespacios → devolver pequeño nº global.
    """
    if dim <= 1:
        return 1
    if dim == 2:
        return int(base_2d)
    if dim == 3:
        theta = math.pi / base_2d  # ≈ separación angular análoga a 2D
        n = max(12, int(math.ceil(2.0 / max(1e-9, (1 - math.cos(theta))))))
        return min(64, n)  # cota superior razonable
    # Para >3D devolvemos unos pocos globales; el resto irá por subespacios
    return 8

def generate_directions(dim: int, base_2d: int, random_state: Optional[int] = 42,
                        max_subspaces: int = 20) -> np.ndarray:
    """
    Conjunto de direcciones:
      - 2D: 8 equiángulos (por defecto)
      - 3D: ~N por fórmula de caps + Fibonacci esférico
      - >3D: mezcla de:
          * pequeños globales (gaussianos) y
          * direcciones embebidas en subespacios 2D/3D (todas o muestreadas)
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

    # elige subespacios de tamaño 3 (o 2 si dim=4 y quieres más baratas)
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
    """
    SheShe: Smart High-dimensional Edge Segmentation & Hyperboundary Explorer

    Clusterización por máximos locales sobre la superficie de probabilidad (clasificación)
    o del valor predicho (regresión). Compatible con sklearn.

    Novedades v2:
      - Nº de rays dinámico: 2D→8; 3D≈26; >3D se reduce con subespacios (2D/3D) + pocos globales.
      - `direction`: 'center_out' (default) o 'outside_in' para localizar la inflexión.
      - Pendiente en punto de inflexión (df/dt).
      - Ascenso con barreras en bordes.
    """

    def __init__(
        self,
        base_estimator: Optional['BaseEstimator'] = None,
        task: str = "classification",  # "classification" | "regression"
        base_2d_rays: int = 8,
        direction: str = "center_out",
        scan_radius_factor: float = 3.0,   # múltiplos del std global
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
                raise ValueError("class_idx requerido en clasificación.")
            proba = self.estimator_.predict_proba(Xs)
            return proba[:, class_idx]
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
        """
        Para cada dirección u: escaneo radial t∈[0,T] y primer punto de inflexión según `direction`.
        Devuelve (radii, points, slopes).
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
        except Exception as exc:  # pragma: no cover - logging auxiliar
            self._log(f"No se pudieron guardar etiquetas en {label_path}: {exc}")

    # ---------- API pública ----------

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        start = time.perf_counter()
        try:
            X = np.asarray(X, dtype=float)
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None

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
            self._log(f"Error en fit: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"fit completado en {runtime:.4f}s")
        return self

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Ajusta el modelo y devuelve la predicción para ``X``.

        Atajo común en *sklearn* que equivale a llamar a :meth:`fit` y
        posteriormente a :meth:`predict` sobre los mismos datos.
        """
        self.fit(X, y)
        return self.predict(X)

    def _membership_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = len(X)
        R = np.zeros((n, len(self.regions_)), dtype=int)
        for k, reg in enumerate(self.regions_):
            if reg.directions.size == 0:
                raise ValueError(
                    "Region con número de direcciones cero; revise base_2d_rays"
                )
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
        start = time.perf_counter()
        try:
            check_is_fitted(self, "regions_")
            X = np.asarray(X, dtype=float)
            M = self._membership_matrix(X)
            if self.task == "classification":
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
                result = M[:, 0]
        except Exception as exc:
            self._log(f"Error en predict: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict completado en {runtime:.4f}s")
        self._maybe_save_labels(result, label_path)
        return result

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Clasificación: proba por clase del estimador base. Regresión: valor normalizado [0,1]."""
        start = time.perf_counter()
        try:
            check_is_fitted(self, "regions_")
            Xs = self.scaler_.transform(np.asarray(X, dtype=float))
            if self.task == "classification":
                result = self.estimator_.predict_proba(Xs)
            else:
                vals = self.estimator_.predict(Xs)
                vmin = min(reg.peak_value_real for reg in self.regions_)
                vmax = max(reg.peak_value_real for reg in self.regions_)
                rng = vmax - vmin if vmax > vmin else 1.0
                result = ((vals - vmin) / rng).reshape(-1, 1)
        except Exception as exc:
            self._log(f"Error en predict_proba: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict_proba completado en {runtime:.4f}s")
        return result

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Valores de decisión del estimador base con *fallback* automático.

        Si el estimador subyacente dispone de :meth:`decision_function`, se
        devuelve dicha salida. En caso contrario se recurre a
        :meth:`predict_proba` para tareas de clasificación o a
        :meth:`predict` para regresión.

        Parameters
        ----------
        X:
            Muestras a evaluar.

        Returns
        -------
        ndarray
            Puntajes, probabilidades o predicciones dependiendo del *fallback*.

        Examples
        --------
        Clasificación con un estimador que implementa ``decision_function``::

            >>> from sklearn.datasets import load_iris
            >>> from sklearn.linear_model import LogisticRegression
            >>> X, y = load_iris(return_X_y=True)
            >>> sh = ModalBoundaryClustering(LogisticRegression(max_iter=200),
            ...                             task="classification").fit(X, y)
            >>> sh.decision_function(X[:2]).shape
            (2, 3)

        Clasificación con un modelo sin ``decision_function`` (usa
        ``predict_proba``)::

            >>> from sklearn.ensemble import RandomForestClassifier
            >>> sh = ModalBoundaryClustering(RandomForestClassifier(),
            ...                             task="classification").fit(X, y)
            >>> sh.decision_function(X[:2]).shape
            (2, 3)

        En regresión la salida proviene de ``predict``::

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
            self._log(f"Error en decision_function: {exc}")
            raise
        runtime = time.perf_counter() - start
        self._log(f"decision_function completado en {runtime:.4f}s")
        return result

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> float:
        """Devuelve la métrica de sklearn delegando en el pipeline interno."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.score(np.asarray(X, dtype=float), y)

    def save(self, filepath: Union[str, Path]) -> None:
        """Guarda la instancia actual en ``filepath`` usando ``joblib.dump``."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ModalBoundaryClustering":
        """Carga una instancia previamente guardada con :meth:`save`."""
        return joblib.load(filepath)

    def interpretability_summary(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
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
            # puntos de inflexión
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

    # -------- Visualización (pares 2D) --------

    def _plot_single_pair_classif(self, X: np.ndarray, y: np.ndarray, pair: Tuple[int, int],
                                  class_colors: Dict[Any, str], grid_res: int = 200, alpha_surface: float = 0.6):
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
        """Genera figuras para todas las combinaciones 2D (o hasta max_pairs)."""
        check_is_fitted(self, "regions_")
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        pairs = list(itertools.combinations(range(d), 2))
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        if self.task == "classification":
            assert y is not None, "y requerido para graficar clasificación."
            palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                       "#ff7f00", "#a65628", "#f781bf", "#999999"]
            class_colors = {c: palette[i % len(palette)] for i, c in enumerate(self.classes_)}
            for pair in pairs:
                self._plot_single_pair_classif(X, y, pair, class_colors)
        else:
            for pair in pairs:
                self._plot_single_pair_reg(X, pair)
