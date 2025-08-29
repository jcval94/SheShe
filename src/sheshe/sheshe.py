
from __future__ import annotations

import itertools
import math
import time
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence

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
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
)
try:  # optional dependency for density estimation
    import hnswlib
except Exception:  # pragma: no cover - handled at runtime
    hnswlib = None  # type: ignore
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
    """Central difference gradient with optional batch evaluation."""
    d = x.shape[0]
    E = np.eye(d) * eps
    P = np.vstack([x + E, x - E])
    if hasattr(f, "batch"):
        vals = f.batch(P)
        return (vals[:d] - vals[d:]) / (2.0 * eps)
    g = np.zeros(d, float)
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        g[i] = (f(x + eps * e) - f(x - eps * e)) / (2.0 * eps)
    return g


def finite_diff_hessian(f, x: np.ndarray, eps: float = 1e-2) -> np.ndarray:
    """Central difference Hessian with optional batch evaluation."""
    d = x.shape[0]
    H = np.zeros((d, d), float)
    if hasattr(f, "batch"):
        # Build all required evaluation points at once
        points = [x]
        for i in range(d):
            ei = np.zeros(d)
            ei[i] = eps
            points.extend([x + ei, x - ei])
            for j in range(i + 1, d):
                ej = np.zeros(d)
                ej[j] = eps
                points.extend(
                    [x + ei + ej, x + ei - ej, x - ei + ej, x - ei - ej]
                )
        vals = f.batch(np.vstack(points))
        idx = 1
        f0 = vals[0]
        for i in range(d):
            f_plus, f_minus = vals[idx : idx + 2]
            idx += 2
            H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (eps ** 2)
            for j in range(i + 1, d):
                fpp, fpm, fmp, fmm = vals[idx : idx + 4]
                idx += 4
                val = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
                H[i, j] = H[j, i] = val
    else:
        f0 = f(x)
        for i in range(d):
            ei = np.zeros(d)
            ei[i] = eps
            f_plus = f(x + ei)
            f_minus = f(x - ei)
            H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (eps ** 2)
            for j in range(i + 1, d):
                ej = np.zeros(d)
                ej[j] = eps
                fpp = f(x + ei + ej)
                fpm = f(x + ei - ej)
                fmp = f(x - ei + ej)
                fmm = f(x - ei - ej)
                val = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
                H[i, j] = H[j, i] = val
    return H


def spsa_gradient(
    f, x: np.ndarray, eps: float = 1e-2, random_state: Optional[int] = None
) -> np.ndarray:
    """Simultaneous perturbation stochastic approximation (SPSA) gradient."""
    rng = _rng(random_state)
    d = x.shape[0]
    delta = rng.choice([-1.0, 1.0], size=d)
    x_plus = x + eps * delta
    x_minus = x - eps * delta
    if hasattr(f, "batch"):
        vals = f.batch(np.vstack([x_plus, x_minus]))
        diff = vals[0] - vals[1]
    else:
        diff = f(x_plus) - f(x_minus)
    return diff / (2.0 * eps * delta)

def project_step_with_barrier(x: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Zero out gradient components that push outside the domain when on the boundary.
    Prevents escaping and forces movement along other variables."""
    step = g.copy()
    for i in range(len(x)):
        if (x[i] <= lo[i] + 1e-12 and step[i] < 0) or (x[i] >= hi[i] - 1e-12 and step[i] > 0):
            step[i] = 0.0
    return step

def gradient_ascent(
    f,
    x0: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    lr: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-5,
    eps_grad: float = 1e-2,
    gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    *,
    use_spsa: bool = False,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Gradient ascent with backtracking and boundary barriers.

    Parameters
    ----------
    f : callable
        Objective function.
    gradient : callable, optional
        Analytic gradient of ``f``. If ``None``, a finite difference
        approximation is used.
    """
    lo, hi = bounds
    x = x0.copy()
    best = f(x)
    no_improve = 0
    for _ in range(max_iter):
        if gradient is not None:
            g = gradient(x)
        else:
            g = (
                spsa_gradient(f, x, eps=eps_grad, random_state=random_state)
                if use_spsa
                else finite_diff_gradient(f, x, eps=eps_grad)
            )
        if np.linalg.norm(g) < tol:
            break
        g = project_step_with_barrier(x, g, lo, hi)
        if np.allclose(g, 0.0):
            break
        step = lr * g / (np.linalg.norm(g) + 1e-12)
        x_new = np.clip(x + step, lo, hi)
        v_new = f(x_new)
        if v_new > best + 1e-12:
            x, best = x_new, v_new
            no_improve = 0
            continue

        # backtracking
        x_try = np.clip(x + 0.5 * step, lo, hi)
        v_try = f(x_try)
        if v_try > best + 1e-12:
            x, best = x_try, v_try
            no_improve = 0
            continue

        no_improve += 1
        if no_improve >= 3:
            break
    return x


def newton_trust_region(
    f,
    x0: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    trust_radius: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    eps_grad: float = 1e-2,
    eps_hess: float = 1e-2,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Trust-region Newton method with boundary barriers.

    Falls back to :func:`gradient_ascent` if the Hessian is not negative
    definite (i.e. ``-H`` not positive definite).
    """
    lo, hi = bounds
    x = x0.copy()
    for _ in range(max_iter):
        if gradient is not None:
            g = gradient(x)
        else:
            g = finite_diff_gradient(f, x, eps=eps_grad)
        g = project_step_with_barrier(x, g, lo, hi)
        if np.linalg.norm(g) < tol:
            break
        H = hessian(x) if hessian is not None else finite_diff_hessian(f, x, eps=eps_hess)
        try:
            np.linalg.cholesky(-H)
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return gradient_ascent(
                f,
                x,
                bounds,
                lr=trust_radius,
                max_iter=max_iter,
                tol=tol,
                eps_grad=eps_grad,
                gradient=gradient,
                random_state=random_state,
            )
        step = project_step_with_barrier(x, step, lo, hi)
        norm_step = np.linalg.norm(step)
        if norm_step < 1e-12:
            break
        if norm_step > trust_radius:
            step = step / norm_step * trust_radius
        x = np.clip(x + step, lo, hi)
    return x

def second_diff(arr: np.ndarray) -> np.ndarray:
    s = np.zeros_like(arr)
    if len(arr) >= 3:
        s[1:-1] = arr[:-2] - 2*arr[1:-1] + arr[2:]
    return s

def _adaptive_scan_1d(
    f_line: Callable[[np.ndarray], np.ndarray],
    T: float,
    steps: int,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Escaneo adaptativo en ``[0, T]``.

    Comienza con una grilla gruesa y luego refina localmente alrededor del
    cambio de concavidad o de la caída del 50% del valor máximo observado.
    """

    steps = max(8, int(steps))

    # 1) Grilla gruesa
    n0 = max(8, steps // 3)
    ts = np.linspace(0.0, T, n0)
    vs = f_line(ts)

    # 2) Busca intervalo candidato mediante cambio de concavidad
    def second_diff(a: np.ndarray) -> np.ndarray:
        if len(a) < 3:
            return np.zeros_like(a)
        s = np.zeros_like(a)
        s[1:-1] = a[:-2] - 2 * a[1:-1] + a[2:]
        return s

    sd = second_diff(vs)
    j = None
    for k in range(1, len(sd)):
        if sd[k] >= 0 and sd[k - 1] < 0:
            j = k
            break

    # 3) Refinamiento local por bisección/densificación
    remaining = steps - len(ts)
    if j is not None and remaining > 0:
        a, b = max(0, j - 2), min(len(ts) - 1, j + 2)
        for _ in range(remaining):
            mids = (ts[a:b] + ts[a + 1 : b + 1]) * 0.5
            vm = f_line(mids)
            ts = np.sort(np.r_[ts, mids])
            # Re-evaluar para mantener el orden de ``vs``
            vs = f_line(ts)

    return ts, vs

def find_inflection(
    ts: np.ndarray,
    vals: np.ndarray,
    direction: str,
    smooth_window: int | None = None,
    drop_fraction: float = 0.5,
) -> Tuple[float, float]:
    """Return ``(t_inf, slope_at_inf)``.

    Parameters
    ----------
    direction : {'center_out', 'outside_in'}
        Scanning strategy.
    smooth_window : int | None, default=None
        If provided and >1, apply a moving average of this window size on the
        scanned values before computing the second derivative.
    drop_fraction : float, default=0.5
        Fallback fraction of the initial value used to determine the radius
        when no inflection is detected.

    Returns
    -------
    t_inf : float
        Parameter ``t`` in ``[0, T]``.
    slope_at_inf : float
        ``df/dt`` at ``t_inf`` (sign consistent with increasing ``t``).
    """
    if direction not in ("center_out", "outside_in"):
        raise ValueError("direction must be 'center_out' or 'outside_in'.")
    if not (0.0 < drop_fraction < 1.0):
        raise ValueError("drop_fraction must be in (0, 1)")

    # Prepare series according to direction
    if direction == "outside_in":
        ts_scan = ts[::-1]
        vals_scan = vals[::-1]
    else:
        ts_scan = ts
        vals_scan = vals

    if smooth_window is not None and smooth_window > 1:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        vals_pad = np.pad(vals_scan, pad_width=pad, mode="edge")
        kernel = np.ones(w) / w
        vals_scan = np.convolve(vals_pad, kernel, mode="valid")

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
        # fallback: drop from val[0] according to ``drop_fraction``
        target = vals_scan[0] * drop_fraction
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


def _eval_batch_safe(f, P: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Evalúa ``f`` sobre ``P`` en lotes para evitar picos de memoria.

    Si ``f`` dispone del método ``batch``, éste se utiliza directamente.
    En caso contrario se evalúa punto a punto.
    """
    P = np.asarray(P)
    n = len(P)
    out = np.empty(n, dtype=float)
    if hasattr(f, "batch"):
        for i in range(0, n, batch):
            out[i:i + batch] = f.batch(P[i:i + batch])
    else:
        for i in range(0, n, batch):
            out[i:i + batch] = np.array([f(p) for p in P[i:i + batch]])
    return out


def _first_drop_index(vals: np.ndarray, threshold: float) -> int:
    """Índice del primer valor que cae por debajo de ``threshold``.

    Si nunca cae, devuelve ``len(vals) - 1``.
    """
    keep = vals >= threshold
    if not keep.all():
        return int(np.argmax(~keep))
    return int(len(vals) - 1)


def _scan_ray_adaptive(center: np.ndarray, direction_unit: np.ndarray,
                       r_min: float, r_max: float, f,
                       threshold: float, batch_size: int = 8192,
                       coarse_steps: int = 12, refine_steps: int = 4,
                       early_exit_patience: int = 1) -> Tuple[float, float, int]:
    """Explora una dirección con malla gruesa y refinamiento local.

    Devuelve ``(r_boundary, slope, evals)`` donde ``r_boundary`` es el último
    radio por encima del umbral ``threshold`` y ``slope`` la pendiente
    estimada en la frontera.
    """
    rs = np.linspace(r_min, r_max, num=coarse_steps, dtype=float)
    P = center[None, :] + rs[:, None] * direction_unit[None, :]
    vals = _eval_batch_safe(f, P, batch=batch_size)
    idx = _first_drop_index(vals, threshold)

    evals = int(len(rs))
    if idx < len(rs) - 1:
        left = rs[max(0, idx - 1)]
        right = rs[idx]
        left_val = vals[max(0, idx - 1)]
        right_val = vals[idx]
    else:
        return float(rs[-1]), float((vals[-1] - vals[0]) / (rs[-1] - rs[0] + 1e-12)), evals

    for _ in range(refine_steps):
        mid = 0.5 * (left + right)
        Pm = center + mid * direction_unit
        mv = float(_eval_batch_safe(f, Pm[None, :], batch=batch_size)[0])
        evals += 1
        if mv >= threshold:
            left, left_val = mid, mv
        else:
            right, right_val = mid, mv
        if right - left <= 1e-9:
            break

    r_boundary = left
    slope = (right_val - left_val) / (right - left + 1e-12)
    return float(r_boundary), float(slope), evals


def find_percentile_drop(
    ts: np.ndarray,
    vals: np.ndarray,
    direction: str,
    percentiles: np.ndarray,
    drop_fraction: float = 0.5,
) -> Tuple[float, float]:
    """Return ``(t_drop, slope_at_drop)`` based on percentile decrease.

    This vectorized implementation avoids Python loops by relying on
    ``numpy`` operations. If no drop to a lower percentile bin is found,
    it falls back to a fractional drop of the initial value as in
    :func:`find_inflection`.
    """
    if direction not in ("center_out", "outside_in"):
        raise ValueError("direction must be 'center_out' or 'outside_in'.")
    if not (0.0 < drop_fraction < 1.0):
        raise ValueError("drop_fraction must be in (0, 1)")

    if direction == "outside_in":
        ts_scan = ts[::-1]
        vals_scan = vals[::-1]
    else:
        ts_scan = ts
        vals_scan = vals

    dec_idx = np.searchsorted(percentiles, vals_scan, side="right") - 1
    dec_drops = np.where(np.diff(dec_idx) < 0)[0]

    if dec_drops.size > 0:
        idx = int(dec_drops[0] + 1)
        t_scan = ts_scan[idx]
        m_scan = (vals_scan[idx] - vals_scan[idx - 1]) / (
            ts_scan[idx] - ts_scan[idx - 1] + 1e-12
        )
    else:
        target = vals_scan[0] * drop_fraction
        t_scan = ts_scan[-1]
        m_scan = (vals_scan[-1] - vals_scan[0]) / (
            ts_scan[-1] - ts_scan[0] + 1e-12
        )
        idxs = np.where(vals_scan <= target)[0]
        if idxs.size > 0:
            j = int(idxs[0])
            t0, t1 = ts_scan[j - 1], ts_scan[j]
            v0, v1 = vals_scan[j - 1], vals_scan[j]
            alpha = float(
                np.clip((target - v0) / (v1 - v0 + 1e-12), 0.0, 1.0)
            )
            t_scan = t0 + alpha * (t1 - t0)
            m_scan = (v1 - v0) / (t1 - t0 + 1e-12)

    t_abs = t_scan if direction == "center_out" else (ts[-1] - t_scan)
    return float(t_abs), float(m_scan)


# =========================
# Output structures
# =========================

@dataclass
class ClusterRegion:
    cluster_id: int                        # general cluster identifier
    label: Union[int, str]                 # class (or "NA" in regression)
    center: np.ndarray                     # local maximum
    directions: np.ndarray                 # (n_rays, d)
    radii: np.ndarray                      # (n_rays,)
    inflection_points: np.ndarray          # (n_rays, d)
    inflection_slopes: np.ndarray          # (n_rays,) df/dt at inflection
    peak_value_real: float                 # real prob/value at the center
    peak_value_norm: float                 # normalized value at the center [0,1]
    score: Optional[float] = None          # effectiveness metric for the cluster
    metrics: Dict[str, float] = field(default_factory=dict)  # optional extra metrics


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
        if base_2d <= 0:
            raise ValueError("base_2d must be positive for dim == 3")
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
# Utilidades auxiliares
# =========================

# --------- Fallback de direcciones (±ejes de dims con mayor varianza) ---------
def _fallback_axis_dirs(X_cluster: np.ndarray, k: int = 6) -> np.ndarray:
    """Retorna 2k direcciones ±canónicas de las k dimensiones con mayor varianza local."""
    d = X_cluster.shape[1]
    k = max(1, min(k, d))
    var = X_cluster.var(axis=0)
    idx = np.argsort(var)[::-1][:k]
    dirs = []
    for j in idx:
        e = np.zeros(d, dtype=float)
        e[j] = 1.0
        dirs.append(e)
        dirs.append(-e)
    return np.asarray(dirs, dtype=float)


def enforce_minimum_subspace(
    directions: np.ndarray | None,
    X_cluster: np.ndarray,
    k_fallback: int = 6,
    *,
    verbose: int = 0,
    log_fn: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Ensure at least one direction vector is available.

    When ``directions`` is ``None`` or empty, fall back to axis-aligned
    directions along the dimensions of highest variance.
    """
    if directions is None or directions.size == 0:
        if verbose and log_fn is not None:
            log_fn(f"Fallback to ±axis directions (k={k_fallback})")
        return _fallback_axis_dirs(X_cluster, k=k_fallback)
    return directions


# --------- Paso adaptativo por escala (0.15 σ proyectada) ---------
def per_direction_step_lengths(
    directions: np.ndarray,
    std_vec: np.ndarray,
    base_sigma_step: float = 0.15,
    mode: str = "diag",
    cov: np.ndarray | None = None,
) -> np.ndarray:
    """Calcula la longitud de paso por rayo según la desviación estándar proyectada."""
    if directions.ndim != 2:
        raise ValueError("directions must be 2D (m, d)")
    m, d = directions.shape

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    u = directions / norms

    if mode == "full":
        if cov is None:
            raise ValueError("cov must be provided when mode='full'")
        var_proj = np.einsum("ij,jk,ik->i", u, cov, u)
    else:
        var_proj = (u * (std_vec ** 2)).sum(axis=1)

    var_proj = np.clip(var_proj, 0.0, None)
    std_proj = np.sqrt(var_proj)
    steps = base_sigma_step * std_proj

    pos = steps[steps > 0]
    if pos.size == 0:
        steps[:] = 1e-3
    else:
        fallback = np.median(pos)
        steps = np.where(steps > 0, steps, np.maximum(1e-6, 0.25 * fallback))
    return steps


def cast_rays_with_censoring(
    center: np.ndarray,
    directions: np.ndarray,
    eval_fn,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    max_steps: int,
    step_lengths: np.ndarray,
    min_drop: float,
    min_slope: float,
    *,
    verbose: int = 0,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Cast rays from ``center`` and record the last point if no inflection.

    Parameters
    ----------
    verbose : int, optional
        ``0`` silences logs, ``1`` prints a summary, ``2`` logs every step.
    log_fn : callable, optional
        Function used for logging. Typically ``self._log`` from the estimator.
    """
    m, d = directions.shape
    lo = hi = None
    if bounds is not None:
        lo, hi = bounds

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    U = directions / norms

    radii = np.zeros(m, float)
    inf_pts = np.zeros((m, d), float)
    inf_slopes = np.zeros(m, float)
    censored = np.zeros(m, bool)
    n_eval = 0

    f0 = eval_fn(center)
    n_eval += 1

    for i in range(m):
        u = U[i]
        step = step_lengths[i]
        x = center.copy()
        prev_val = f0
        found = False

        for s in range(1, max_steps + 1):
            x_next = x + step * u

            if lo is not None:
                if (x_next < lo).any() or (x_next > hi).any():
                    censored[i] = True
                    inf_pts[i] = x
                    radii[i] = np.linalg.norm(x - center)
                    if s > 1:
                        inf_slopes[i] = (prev_val - f0) / (radii[i] + 1e-12)
                    if verbose >= 2 and log_fn is not None:
                        log_fn(f"ray {i}: exited bounds at step {s}")
                    break

            f_next = eval_fn(x_next)
            n_eval += 1
            drop = f0 - f_next
            slope = (prev_val - f_next) / (step + 1e-12)

            if verbose >= 2 and log_fn is not None:
                log_fn(
                    f"ray {i}: step {s} drop={drop:.4f} slope={slope:.4f}"
                )

            if (drop >= min_drop) and (abs(slope) >= min_slope):
                found = True
                inf_pts[i] = x_next
                radii[i] = np.linalg.norm(x_next - center)
                inf_slopes[i] = slope
                if verbose >= 2 and log_fn is not None:
                    log_fn(f"ray {i}: inflection at step {s}")
                break

            x = x_next
            prev_val = f_next

        if not found and not censored[i]:
            censored[i] = True
            inf_pts[i] = x
            radii[i] = np.linalg.norm(x - center)
            if max_steps > 0:
                inf_slopes[i] = (prev_val - f0) / (radii[i] + 1e-12)
            if verbose >= 2 and log_fn is not None:
                log_fn(f"ray {i}: censored after {max_steps} steps")

    metrics = {
        "n_rays": int(m),
        "n_eval": int(n_eval),
        "censored_mask": censored,
        "censored_frac": float(censored.mean()) if m > 0 else 0.0,
    }
    if verbose >= 1 and log_fn is not None:
        log_fn(
            f"cast {m} rays: n_eval={n_eval}, censored_frac={metrics['censored_frac']:.2f}"
        )
    return radii, inf_pts, inf_slopes, metrics


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
      - Optional smoothing of radial scans via ``smooth_window``.
      - Fallback radius via ``drop_fraction`` when no inflection is found.
      - ``verbose`` levels: ``0`` silencia, ``1`` tiempos resumidos, ``2`` detalle completo.
    """

    def __init__(
        self,
        base_estimator: Optional['BaseEstimator'] = None,
        task: str = "classification",  # "classification" | "regression"
        base_2d_rays: int = 32,
        direction: str = "center_out",
        stop_criteria: str = "inflexion",  # "inflexion" | "percentile"
        percentile_bins: int = 20,         # number of percentile bins when stop_criteria='percentile'
        scan_radius_factor: float = 3.0,   # multiples of the global std
        scan_steps: int = 24,
        smooth_window: int | None = None,
        drop_fraction: float = 0.5,
        grad_lr: float = 0.2,
        grad_max_iter: int = 80,
        grad_tol: float = 1e-5,
        grad_eps: float = 1e-3,
        n_max_seeds: int = 2,
        random_state: Optional[int] = 42,
        percentile_sample_size: Optional[int] = 50000,
        percentile_method: str = "auto",
        hist_bins: int = 2048,
        max_subspaces: int = 20,
        verbose: int = 0,
        save_labels: bool = False,
        prediction_within_region: bool = False,
        out_dir: Optional[Union[str, Path]] = None,
        auto_rays_by_dim: bool = True,
        ray_mode: str = "grad",
        use_spsa: bool = True,
        spsa_delta: float = 1e-2,
        spsa_avg: int = 4,
        ls_alpha0: float = 0.5,
        ls_shrink: float = 0.5,
        ls_min_alpha: float = 1e-3,
        arc_max_steps: int = 64,
        arc_len_max: float = 3.0,
        line_refine_steps: int = 8,
        use_adaptive_scan: Optional[bool] = None,
        batch_size: int = 16384,
        coarse_steps: int = 12,
        refine_steps: int = 4,
        early_exit_patience: int = 1,
        density_alpha: float = 0.0,
        density_k: int = 15,
        cluster_metrics_cls: Optional[Dict[str, Callable]] = None,
        cluster_metrics_reg: Optional[Dict[str, Callable]] = None,
        fast_membership: bool = False,
    ):
        if scan_steps < 2:
            raise ValueError("scan_steps must be at least 2")
        if smooth_window is not None and smooth_window < 1:
            raise ValueError("smooth_window must be None or >= 1")
        if n_max_seeds < 1:
            raise ValueError("n_max_seeds must be at least 1")
        if not (0.0 < drop_fraction < 1.0):
            raise ValueError("drop_fraction must be in (0, 1)")
        if stop_criteria not in ("inflexion", "percentile"):
            raise ValueError("stop_criteria must be 'inflexion' or 'percentile'")
        if ray_mode not in ("grid", "grad"):
            raise ValueError("ray_mode must be 'grid' or 'grad'")
        if percentile_bins < 1:
            raise ValueError("percentile_bins must be at least 1")
        if not (0.0 <= density_alpha <= 1.0):
            raise ValueError("density_alpha must be in [0, 1]")
        if density_k < 1:
            raise ValueError("density_k must be at least 1")

        self.base_estimator = base_estimator
        self.task = task
        self.base_2d_rays = base_2d_rays
        self.direction = direction
        self.stop_criteria = stop_criteria
        self.percentile_bins = percentile_bins
        self.scan_radius_factor = scan_radius_factor
        self.scan_steps = scan_steps
        self.smooth_window = smooth_window
        self.drop_fraction = drop_fraction
        self.grad_lr = grad_lr
        self.grad_max_iter = grad_max_iter
        self.grad_tol = grad_tol
        self.grad_eps = grad_eps
        self.n_max_seeds = n_max_seeds
        self.random_state = random_state
        self.percentile_sample_size = percentile_sample_size
        self.percentile_method = percentile_method
        self.hist_bins = hist_bins
        self.max_subspaces = max_subspaces
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # En algunos entornos (ej. notebooks o Google Colab) ``stderr`` se
            # presenta de forma distinta al flujo estándar.  Usamos ``stdout``
            # para asegurar que los mensajes se muestren siempre.
            self.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        if verbose >= 2:
            self.logger.setLevel(logging.DEBUG)
        elif verbose == 1:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        self.save_labels = save_labels
        self.prediction_within_region = prediction_within_region
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.auto_rays_by_dim = auto_rays_by_dim
        self.ray_mode = ray_mode
        self.use_spsa = use_spsa
        self.spsa_delta = spsa_delta
        self.spsa_avg = spsa_avg
        self.ls_alpha0 = ls_alpha0
        self.ls_shrink = ls_shrink
        self.ls_min_alpha = ls_min_alpha
        self.arc_max_steps = arc_max_steps
        self.arc_len_max = arc_len_max
        self.line_refine_steps = line_refine_steps
        self.use_adaptive_scan = use_adaptive_scan
        self.batch_size = batch_size
        self.coarse_steps = coarse_steps
        self.refine_steps = refine_steps
        self.early_exit_patience = early_exit_patience
        self.density_alpha = density_alpha
        self.density_k = density_k
        self.cluster_metrics_cls = cluster_metrics_cls
        self.cluster_metrics_reg = cluster_metrics_reg
        self.fast_membership = fast_membership
        self._P_all: Optional[np.ndarray] = None
        self._yhat_all: Optional[np.ndarray] = None
        self._X_train_shape: Optional[Tuple[int, int]] = None

    # ---------- helpers ----------

    def _fit_estimator(self, X: np.ndarray, y: Optional[np.ndarray]):
        if self.base_estimator is None:
            if self.task == "classification":
                est = LogisticRegression(max_iter=1000)
            else:
                est = GradientBoostingRegressor(random_state=self.random_state)
        else:
            est = clone(self.base_estimator)

        self.pipeline_ = Pipeline([("scaler", StandardScaler()), ("estimator", est)])
        self.pipeline_.fit(X, y if y is not None else np.zeros(len(X)))
        self.estimator_ = self.pipeline_.named_steps["estimator"]
        self.scaler_ = self.pipeline_.named_steps["scaler"]

    def _predict_value_real(self, X: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64, order="C")
        if self._X_train_shape is not None and X.shape == self._X_train_shape:
            if self.task == "classification":
                if class_idx is None:
                    raise ValueError("class_idx required for classification.")
                if self._P_all is not None:
                    return self._P_all[:, class_idx]
                if self._yhat_all is not None:
                    scores = self._yhat_all
                    if scores.ndim == 1:
                        if class_idx not in (0, 1):
                            raise ValueError("class_idx must be 0 or 1 for binary decision_function")
                        return scores if class_idx == 1 else -scores
                    return scores[:, class_idx]
            else:
                if self._yhat_all is not None:
                    return self._yhat_all
        Xs = self.scaler_.transform(X)
        if self.task == "classification":
            if class_idx is None:
                raise ValueError("class_idx required for classification.")
            if hasattr(self.estimator_, "predict_proba"):
                return self.estimator_.predict_proba(Xs)[:, class_idx]
            scores = self.estimator_.decision_function(Xs)
            if scores.ndim == 1:
                if class_idx not in (0, 1):
                    raise ValueError("class_idx must be 0 or 1 for binary decision_function")
                return scores if class_idx == 1 else -scores
            return scores[:, class_idx]
        else:
            return self.estimator_.predict(Xs)

    def _build_value_fn(self, class_idx: Optional[int], norm_stats: Dict[str, float]):
        vmin, vmax = norm_stats["min"], norm_stats["max"]
        rng = vmax - vmin if vmax > vmin else 1.0

        def _norm(v):
            return (v - vmin) / rng

        def f(x: np.ndarray) -> float:
            val = self._predict_value_real(x.reshape(1, -1), class_idx=class_idx)[0]
            v = float(_norm(val))
            if self.density_alpha > 0.0:
                dens = self._density(x.reshape(1, -1))[0]
                v *= float(dens ** self.density_alpha)
            return v

        def f_batch(X: np.ndarray) -> np.ndarray:
            vals = self._predict_value_real(np.asarray(X, float), class_idx=class_idx)
            v = _norm(vals)
            if self.density_alpha > 0.0:
                dens = self._density(np.asarray(X, float))
                v = v * (dens ** self.density_alpha)
            return v

        f.batch = f_batch  # type: ignore[attr-defined]
        return f

    def _bounds_from_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        span = hi - lo
        return lo - 0.05*span, hi + 0.05*span

    def _setup_density(self, X: np.ndarray) -> None:
        if self.density_alpha <= 0.0:
            return
        if hnswlib is None:
            raise ImportError("hnswlib is required when density_alpha > 0.0")
        Xs = self.scaler_.transform(np.asarray(X, float)).astype(np.float32)
        n_samples = len(Xs)
        if n_samples <= 1:
            self.density_k_ = 1
            self._nn_density = None
            self.density_min_ = self.density_max_ = 1.0
            return
        k = min(self.density_k, n_samples - 1)
        self.density_k_ = k
        dim = Xs.shape[1]
        index = hnswlib.Index(space="l2", dim=dim)
        index.init_index(max_elements=n_samples, ef_construction=200)
        index.add_items(Xs)
        index.set_ef(max(50, k * 3))
        self._nn_density = index
        _, dists = index.knn_query(Xs, k=k + 1)
        dists = np.sqrt(dists[:, -1])
        dens = 1.0 / (dists + 1e-12)
        self.density_min_ = float(np.min(dens))
        self.density_max_ = float(np.max(dens))

    def _density(self, X: np.ndarray) -> np.ndarray:
        if self.density_alpha <= 0.0:
            return np.ones(len(np.asarray(X, float)))
        Xs = self.scaler_.transform(np.asarray(X, float)).astype(np.float32)
        if self._nn_density is None:
            dens = np.ones(len(Xs))
        else:
            _, dists = self._nn_density.knn_query(Xs, k=self.density_k_)
            dists = np.sqrt(dists[:, -1])
            dens = 1.0 / (dists + 1e-12)
        rng = self.density_max_ - self.density_min_
        if rng <= 0:
            return np.ones_like(dens)
        dens_norm = (dens - self.density_min_) / rng
        return np.clip(dens_norm, 0.0, 1.0)

    def _choose_seeds(self, X: np.ndarray, f, k: int) -> np.ndarray:
        vals = f.batch(X) if hasattr(f, "batch") else np.array([f(x) for x in X])
        if len(vals) == 0 or k <= 0:
            return np.zeros((0, X.shape[1]))
        best_idx = int(np.argmax(vals))
        seeds = [X[best_idx]]
        if k == 1:
            return np.asarray(seeds)

        from sklearn.cluster import KMeans

        remaining = np.delete(X, best_idx, axis=0)
        n_clusters = min(k - 1, len(remaining))
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            kmeans.fit(remaining)
            centers = kmeans.cluster_centers_
            center_vals = f.batch(centers) if hasattr(f, "batch") else np.array([f(c) for c in centers])
            order = np.argsort(-center_vals)
            centers = centers[order]
            seeds.extend(list(centers))
        return np.asarray(seeds)[:k]

    def _find_maximum(self, X: np.ndarray, f, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        seeds = self._choose_seeds(X, f, min(self.n_max_seeds, len(X)))
        if len(seeds) == 0:
            return X[0]
        best_x, best_v = seeds[0].copy(), f(seeds[0])
        grad_fn = getattr(f, "grad", None)
        for s in seeds:
            x_star = gradient_ascent(
                f,
                s,
                bounds,
                lr=self.grad_lr,
                max_iter=self.grad_max_iter,
                tol=self.grad_tol,
                eps_grad=self.grad_eps,
                gradient=grad_fn,
                use_spsa=self.use_spsa,
                random_state=self.random_state,
            )
            v = f(x_star)
            if v > best_v:
                best_x, best_v = x_star, v
        return best_x

    def _scan_radii(
        self,
        center: np.ndarray,
        f,
        directions: np.ndarray,
        X_std: np.ndarray,
        percentiles: Optional[np.ndarray] = None,
        return_metrics: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Radial scans for ``directions`` starting at ``center``."""
        d = center.shape[0]
        lo, hi = getattr(self, "bounds_", (None, None))
        assert lo is not None and hi is not None, "bounds_ not initialized."

        if not return_metrics:
            T_base = float(self.scan_radius_factor * np.linalg.norm(X_std))

            def tmax_in_bounds(c: np.ndarray, u: np.ndarray) -> float:
                tmax = np.inf
                for k in range(d):
                    uk = u[k]
                    if abs(uk) < 1e-12:
                        continue
                    tk = (hi[k] - c[k]) / uk if uk > 0 else (lo[k] - c[k]) / uk
                    if tk > 0:
                        tmax = min(tmax, tk)
                if not np.isfinite(tmax) or tmax <= 0:
                    return 1e-9
                return float(tmax)

            if len(directions) == 0:
                return np.zeros(0), np.zeros((0, d)), np.zeros(0)

            use_adaptive = self.use_adaptive_scan
            if use_adaptive is None:
                use_adaptive = (len(directions) * self.coarse_steps > 512)

            n = len(directions)
            radii = np.zeros(n, float)
            slopes = np.zeros(n, float)
            pts = np.zeros((n, d), float)

            threshold = None
            if self.stop_criteria == "percentile":
                if percentiles is None:
                    raise ValueError("percentiles must be provided when stop_criteria='percentile'")
                v0 = float(f(center))
                idx0 = int(np.searchsorted(percentiles, v0, side="right") - 1)
                threshold = v0 * self.drop_fraction if idx0 <= 0 else float(percentiles[idx0 - 1])

            for i, u in enumerate(directions):
                T_dir = min(T_base, tmax_in_bounds(center, u))
                if self.stop_criteria == "percentile" and threshold is not None:
                    if use_adaptive:
                        r, m, _ = _scan_ray_adaptive(
                            center,
                            u,
                            0.0,
                            T_dir,
                            f,
                            threshold,
                            batch_size=self.batch_size,
                            coarse_steps=self.coarse_steps,
                            refine_steps=self.refine_steps,
                            early_exit_patience=self.early_exit_patience,
                        )
                    else:
                        n_steps = self.scan_steps
                        rs = np.linspace(0.0, T_dir, num=n_steps, dtype=float)
                        P = center[None, :] + rs[:, None] * u[None, :]
                        vals = _eval_batch_safe(f, P, batch=self.batch_size)
                        idx = _first_drop_index(vals, threshold)
                        if idx < len(rs):
                            r = rs[idx]
                            if idx > 0:
                                m = (vals[idx] - vals[idx - 1]) / (rs[idx] - rs[idx - 1] + 1e-12)
                            else:
                                m = 0.0
                        else:
                            r = rs[-1]
                            m = (vals[-1] - vals[-2]) / (rs[-1] - rs[-2] + 1e-12)
                    p = center + r * u
                    p = np.minimum(np.maximum(p, lo), hi)
                    radii[i], slopes[i], pts[i, :] = float(r), float(m), p
                else:
                    if use_adaptive:
                        def f_line(ts_arr: np.ndarray) -> np.ndarray:
                            P = center[None, :] + ts_arr[:, None] * u[None, :]
                            return _eval_batch_safe(f, P, batch=self.batch_size)

                        ts, vs = _adaptive_scan_1d(f_line, T_dir, self.scan_steps, self.direction)
                    else:
                        ts = np.linspace(0.0, T_dir, self.scan_steps)
                        P = center[None, :] + ts[:, None] * u[None, :]
                        vs = _eval_batch_safe(f, P, batch=self.batch_size)
                    r, m = find_inflection(
                        ts,
                        vs,
                        self.direction,
                        self.smooth_window,
                        self.drop_fraction,
                    )
                    p = center + r * u
                    p = np.minimum(np.maximum(p, lo), hi)
                    radii[i], slopes[i], pts[i, :] = float(r), float(m), p
            return radii, pts, slopes

        # --- New adaptive step algorithm with censoring ---
        if len(directions) == 0:
            empty_metrics = {
                "n_rays": 0,
                "n_eval": 0,
                "censored_mask": np.zeros(0, bool),
                "censored_frac": 0.0,
            }
            return np.zeros(0), np.zeros((0, d)), np.zeros(0), empty_metrics

        steps = per_direction_step_lengths(directions, X_std, base_sigma_step=0.15, mode="diag")
        v0 = float(f(center))
        if self.stop_criteria == "percentile":
            if percentiles is None:
                raise ValueError("percentiles must be provided when stop_criteria='percentile'")
            idx0 = int(np.searchsorted(percentiles, v0, side="right") - 1)
            thr = v0 * self.drop_fraction if idx0 <= 0 else float(percentiles[idx0 - 1])
            min_drop = max(0.0, v0 - thr)
        else:
            min_drop = (1.0 - self.drop_fraction) * v0

        log_fn = None
        if self.verbose >= 2:
            log_fn = lambda m: self._log(m, level=logging.DEBUG)
        elif self.verbose == 1:
            log_fn = lambda m: self._log(m, level=logging.INFO)

        radii, pts, slopes, metrics = cast_rays_with_censoring(
            center=center,
            directions=directions,
            eval_fn=f,
            bounds=(lo, hi),
            max_steps=self.scan_steps,
            step_lengths=steps,
            min_drop=min_drop,
            min_slope=0.0,
            verbose=self.verbose,
            log_fn=log_fn,
        )
        return radii, pts, slopes, metrics

    def _spsa_grad(self, f, x: np.ndarray, delta: Optional[float] = None, avg: int = 4) -> np.ndarray:
        """Estimate gradient of ``f`` at ``x`` using SPSA.

        Parameters
        ----------
        f : callable
            Function accepting batches.
        x : ndarray of shape (d,)
            Point where the gradient is estimated.
        delta : float, optional
            Relative perturbation size. Defaults to ``self.spsa_delta``.
        avg : int, optional
            Number of perturbation pairs to average. Defaults to ``self.spsa_avg``.
        """
        d = x.shape[0]
        delta = self.spsa_delta if delta is None else float(delta)
        scale = getattr(self, "_feature_scale", np.ones(d))
        step = delta * np.maximum(scale, 1e-8)

        K = max(1, int(avg))
        R = np.random.choice([-1.0, 1.0], size=(K, d))
        Xp = x[None, :] + R * step[None, :]
        Xm = x[None, :] - R * step[None, :]
        lb = getattr(self, "_lb", None)
        ub = getattr(self, "_ub", None)
        if lb is not None and ub is not None:
            Xp = np.minimum(ub, np.maximum(lb, Xp))
            Xm = np.minimum(ub, np.maximum(lb, Xm))

        fp = f(Xp)
        fm = f(Xm)
        denom = 2.0 * step[None, :]
        G = ((fp - fm)[:, None] / np.maximum(denom, 1e-12)) * R
        g = np.median(G, axis=0)
        return g

    def _stop_crossed_on_segment(self, x0: np.ndarray, x1: np.ndarray) -> bool:
        """Check whether the stopping criteria is crossed between ``x0`` and ``x1``."""
        S = min(32, max(8, self.line_refine_steps * 2))
        t = np.linspace(0.0, 1.0, S)
        Z = x0[None, :] + t[:, None] * (x1 - x0)[None, :]
        fz = self._f(Z)
        if self.stop_criteria == "percentile":
            thr = getattr(self, "_percentile_threshold", None)
            if thr is None:
                return False
            idx = np.nonzero(fz < thr)[0]
            return len(idx) > 0
        if self.stop_criteria == "inflexion":
            df = np.diff(fz)
            return np.any(df[:-1] * df[1:] < 0.0)
        return fz[-1] < fz[0] - 1e-2

    def _refine_boundary_bisection(self, x0: np.ndarray, x1: np.ndarray, steps: int = 8) -> np.ndarray:
        """Refine boundary between ``x0`` and ``x1`` using 1D bisection."""
        a, b = x0.copy(), x1.copy()
        fa = self._f(a[None, :])[0]
        fb = self._f(b[None, :])[0]
        thr = getattr(self, "_percentile_threshold", None)
        for _ in range(int(steps)):
            m = 0.5 * (a + b)
            fm = self._f(m[None, :])[0]
            if self.stop_criteria == "percentile" and thr is not None:
                crossed_left = (fa >= thr) and (fm < thr)
            else:
                crossed_left = fm < fa
            if crossed_left:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    def _trace_boundary_grad(self, x0: np.ndarray) -> np.ndarray:
        """Gradient-guided boundary tracing starting from ``x0``."""
        x = x0.copy()
        v_prev = None
        arc_len = 0.0
        pts: List[np.ndarray] = []
        f = self._f
        scale = getattr(self, "_feature_scale", np.ones_like(x))
        for it in range(self.arc_max_steps):
            if self.use_spsa:
                g = self._spsa_grad(f, x, delta=self.spsa_delta, avg=self.spsa_avg)
            else:
                g = np.random.randn(*x.shape)
            g_norm = float(np.linalg.norm(g))
            if g_norm < self.grad_eps:
                v = np.random.randn(*x.shape)
            else:
                v = g / g_norm
            if v_prev is not None:
                v = 0.2 * v_prev + 0.8 * v
                v /= (np.linalg.norm(v) + 1e-12)
            alpha = self.ls_alpha0
            moved = False
            while alpha >= self.ls_min_alpha:
                x_prop = x + alpha * scale * v
                lb = getattr(self, "_lb", None)
                ub = getattr(self, "_ub", None)
                if lb is not None and ub is not None:
                    x_prop = np.minimum(ub, np.maximum(lb, x_prop))
                if self._stop_crossed_on_segment(x, x_prop):
                    x_star = self._refine_boundary_bisection(x, x_prop, steps=self.line_refine_steps)
                    pts.append(x_star)
                    x = x_star
                    moved = True
                    break
                fx = f(x[None, :])[0]
                fxp = f(x_prop[None, :])[0]
                if fxp >= fx - 1e-12:
                    x = x_prop
                    moved = True
                    break
                alpha *= self.ls_shrink
            arc_len += alpha
            v_prev = v
            if not moved or arc_len >= self.arc_len_max:
                break
        return np.asarray(pts) if pts else np.empty((0, x.size))

    def _build_norm_stats(self, X: np.ndarray, class_idx: Optional[int]) -> Dict[str, float]:
        vals = self._predict_value_real(X, class_idx=class_idx)
        return {"min": float(np.min(vals)), "max": float(np.max(vals))}

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        self.logger.log(level, msg)

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
            self._log(f"Could not save labels to {label_path}: {exc}", level=logging.WARNING)

    # ---------- Public API ----------

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        start = time.perf_counter()
        self._log("Starting fit", level=logging.DEBUG)
        try:
            prep_start = time.perf_counter()
            X = np.asarray(X, dtype=float)
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None

            if self.task == "classification":
                if y is None:
                    raise ValueError("y cannot be None when task='classification'")
                y_arr = np.asarray(y)
                if np.unique(y_arr).size < 2:
                    raise ValueError("y must contain at least two classes for classification")
            prep_time = time.perf_counter() - prep_start
            self._log(f"Data preparation in {prep_time:.4f}s", level=logging.DEBUG)

            t = time.perf_counter()
            self._fit_estimator(X, y)
            self._log(f"Base estimator fit in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            if self.density_alpha > 0.0:
                t = time.perf_counter()
                self._setup_density(X)
                self._log(f"Density setup in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            t = time.perf_counter()
            Xs_all = self.scaler_.transform(np.asarray(X, dtype=float))
            P_all = None
            yhat_all = None
            if self.task == "classification":
                if hasattr(self.estimator_, "predict_proba"):
                    P_all = self.estimator_.predict_proba(Xs_all).astype(np.float32, copy=False)
                elif hasattr(self.estimator_, "decision_function"):
                    yhat_all = self.estimator_.decision_function(Xs_all).astype(np.float32, copy=False)
                else:
                    raise RuntimeError("Base estimator must implement predict_proba or decision_function")
            else:
                yhat_all = self.estimator_.predict(Xs_all).astype(np.float32, copy=False)
            self._log(f"Initial predictions in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            self._X_train_shape = Xs_all.shape
            self._P_all = P_all
            self._yhat_all = yhat_all
            lo, hi = self._bounds_from_data(X)
            self.bounds_ = (lo.copy(), hi.copy())  # store bounds for radial scans
            self._lb = lo.astype(float, copy=True)
            self._ub = hi.astype(float, copy=True)
            self._feature_scale = np.maximum(self._ub - self._lb, 1e-8)
            X_std = np.std(X, axis=0) + 1e-12
            d = X.shape[1]
            self.use_spsa = self.use_spsa or d >= 20
            base_rays_eff = self.base_2d_rays
            if self.auto_rays_by_dim:
                if d >= 65:
                    base_rays_eff = min(base_rays_eff, 12)
                elif d >= 25:
                    base_rays_eff = min(base_rays_eff, 16)
            dirs_raw = generate_directions(
                d, base_rays_eff, self.random_state, self.max_subspaces
            )
            fallback_dirs_used = (dirs_raw is None or dirs_raw.size == 0)
            log_fn = None
            if self.verbose >= 2:
                log_fn = lambda m: self._log(m, level=logging.DEBUG)
            elif self.verbose == 1:
                log_fn = lambda m: self._log(m, level=logging.INFO)
            dirs = enforce_minimum_subspace(
                dirs_raw,
                X,
                k_fallback=6,
                verbose=self.verbose,
                log_fn=log_fn,
            )

            rng = np.random.default_rng(self.random_state)

            def _pick_sample_idx(y_vals, n_total, n_sample):
                if n_sample is None or n_sample >= n_total:
                    return np.arange(n_total)
                if (y_vals is not None) and (self.task == "classification"):
                    cls, yi = np.unique(y_vals, return_inverse=True)
                    counts = np.bincount(yi)
                    props = counts / counts.sum()
                    take = np.maximum(1, np.floor(props * n_sample)).astype(int)
                    idx = []
                    for c in range(len(cls)):
                        ids = np.flatnonzero(yi == c)
                        k = min(len(ids), take[c])
                        idx.append(rng.choice(ids, size=k, replace=False))
                    out = np.concatenate(idx)
                    if out.size > n_sample:
                        out = rng.choice(out, size=n_sample, replace=False)
                    return np.sort(out)
                return np.sort(rng.choice(n_total, size=n_sample, replace=False))

            self.regions_: List[ClusterRegion] = []
            self.classes_ = None
            t = time.perf_counter()
            if self.task == "classification":
                _ = self.pipeline_.predict(X[:2])  # asegura classes_
                self.classes_ = self.estimator_.classes_
                P_s = None
                scores_s = None
                if self.stop_criteria == "percentile":
                    self.percentiles_ = {}
                    idx_sample = _pick_sample_idx(y, len(X), self.percentile_sample_size)
                    if self._P_all is not None:
                        P_s = self._P_all[idx_sample]
                    else:
                        if self._yhat_all is not None:
                            scores_s = self._yhat_all[idx_sample]
                        else:
                            scores_s = self.estimator_.decision_function(Xs_all[idx_sample])
                for ci, label in enumerate(self.classes_):
                    stats = self._build_norm_stats(X, class_idx=ci)
                    f = self._build_value_fn(class_idx=ci, norm_stats=stats)
                    center = self._find_maximum(X, f, (lo, hi))
                    self._f = lambda Z: _eval_batch_safe(f, np.asarray(Z, float), batch=self.batch_size)
                    if self.stop_criteria == "percentile":
                        if P_s is not None:
                            vals = P_s[:, ci].astype(np.float32, copy=False)
                        else:
                            scores = scores_s
                            if scores.ndim == 1:
                                if ci not in (0, 1):
                                    raise ValueError("class_idx must be 0 or 1 for binary decision_function")
                                vals = scores if ci == 1 else -scores
                            else:
                                vals = scores[:, ci]
                            vals = vals.astype(np.float32, copy=False)
                        vmin, vmax = float(vals.min()), float(vals.max())
                        rngv = (vmax - vmin) if vmax > vmin else 1.0
                        vals_norm = (vals - vmin) / rngv
                        perc = np.quantile(
                            vals_norm, np.linspace(0.0, 1.0, self.percentile_bins + 1)
                        ).astype(np.float32, copy=False)
                        self.percentiles_[ci] = perc
                        v0 = float(f(center))
                        idx0 = int(np.searchsorted(perc, v0, side="right") - 1)
                        thr = v0 * self.drop_fraction if idx0 <= 0 else float(perc[idx0 - 1])
                        self._percentile_threshold = thr
                        if self.ray_mode == "grad":
                            infl = self._trace_boundary_grad(center)
                            vecs = infl - center
                            radii = np.linalg.norm(vecs, axis=1)
                            dirs_reg = vecs / (radii[:, None] + 1e-12) if len(radii) else np.zeros((0, center.size))
                            slopes = np.zeros_like(radii)
                            dirs_use = dirs_reg
                            ray_metrics = {"n_rays": len(dirs_reg), "n_eval": 0, "censored_mask": np.zeros(len(dirs_reg), bool), "censored_frac": 0.0, "fallback_dirs_used": fallback_dirs_used}
                        else:
                            radii, infl, slopes, ray_metrics = self._scan_radii(center, f, dirs, X_std, percentiles=perc, return_metrics=True)
                            ray_metrics["fallback_dirs_used"] = fallback_dirs_used
                            dirs_use = dirs
                    else:
                        self._percentile_threshold = None
                        if self.ray_mode == "grad":
                            infl = self._trace_boundary_grad(center)
                            vecs = infl - center
                            radii = np.linalg.norm(vecs, axis=1)
                            dirs_reg = vecs / (radii[:, None] + 1e-12) if len(radii) else np.zeros((0, center.size))
                            slopes = np.zeros_like(radii)
                            dirs_use = dirs_reg
                            ray_metrics = {"n_rays": len(dirs_reg), "n_eval": 0, "censored_mask": np.zeros(len(dirs_reg), bool), "censored_frac": 0.0, "fallback_dirs_used": fallback_dirs_used}
                        else:
                            radii, infl, slopes, ray_metrics = self._scan_radii(center, f, dirs, X_std, return_metrics=True)
                            ray_metrics["fallback_dirs_used"] = fallback_dirs_used
                            dirs_use = dirs
                    peak_real = float(self._predict_value_real(center.reshape(1, -1), class_idx=ci)[0])
                    peak_norm = float(f(center))
                    if radii.size == 0 or dirs_use.size == 0:
                        if log_fn is not None:
                            log_fn("Skipping region without directions")
                        else:
                            warnings.warn(
                                "Región sin direcciones; se omite",
                                RuntimeWarning,
                            )
                        continue
                    self.regions_.append(ClusterRegion(
                        cluster_id=len(self.regions_),
                        label=label,
                        center=center,
                        directions=dirs_use,
                        radii=radii,
                        inflection_points=infl,
                        inflection_slopes=slopes,
                        peak_value_real=peak_real,
                        peak_value_norm=peak_norm,
                        metrics=ray_metrics,
                    ))
            else:
                stats = self._build_norm_stats(X, class_idx=None)
                f = self._build_value_fn(class_idx=None, norm_stats=stats)
                center = self._find_maximum(X, f, (lo, hi))
                self._f = lambda Z: _eval_batch_safe(f, np.asarray(Z, float), batch=self.batch_size)
                if self.stop_criteria == "percentile":
                    idx_sample = _pick_sample_idx(None, len(X), self.percentile_sample_size)
                    vals = (self._yhat_all[idx_sample] if self._yhat_all is not None
                            else self.estimator_.predict(Xs_all[idx_sample])).astype(np.float32, copy=False)
                    vmin, vmax = float(vals.min()), float(vals.max())
                    rngv = (vmax - vmin) if vmax > vmin else 1.0
                    vals_norm = (vals - vmin) / rngv
                    perc = np.quantile(
                        vals_norm, np.linspace(0.0, 1.0, self.percentile_bins + 1)
                    ).astype(np.float32, copy=False)
                    self.percentiles_ = perc
                    v0 = float(f(center))
                    idx0 = int(np.searchsorted(perc, v0, side="right") - 1)
                    thr = v0 * self.drop_fraction if idx0 <= 0 else float(perc[idx0 - 1])
                    self._percentile_threshold = thr
                    if self.ray_mode == "grad":
                        infl = self._trace_boundary_grad(center)
                        vecs = infl - center
                        radii = np.linalg.norm(vecs, axis=1)
                        dirs_reg = vecs / (radii[:, None] + 1e-12) if len(radii) else np.zeros((0, center.size))
                        slopes = np.zeros_like(radii)
                        dirs_use = dirs_reg
                        ray_metrics = {"n_rays": len(dirs_reg), "n_eval": 0, "censored_mask": np.zeros(len(dirs_reg), bool), "censored_frac": 0.0, "fallback_dirs_used": fallback_dirs_used}
                    else:
                        radii, infl, slopes, ray_metrics = self._scan_radii(
                            center, f, dirs, X_std, percentiles=perc, return_metrics=True
                        )
                        ray_metrics["fallback_dirs_used"] = fallback_dirs_used
                        dirs_use = dirs
                else:
                    self._percentile_threshold = None
                    if self.ray_mode == "grad":
                        infl = self._trace_boundary_grad(center)
                        vecs = infl - center
                        radii = np.linalg.norm(vecs, axis=1)
                        dirs_reg = vecs / (radii[:, None] + 1e-12) if len(radii) else np.zeros((0, center.size))
                        slopes = np.zeros_like(radii)
                        dirs_use = dirs_reg
                        ray_metrics = {"n_rays": len(dirs_reg), "n_eval": 0, "censored_mask": np.zeros(len(dirs_reg), bool), "censored_frac": 0.0, "fallback_dirs_used": fallback_dirs_used}
                    else:
                        radii, infl, slopes, ray_metrics = self._scan_radii(center, f, dirs, X_std, return_metrics=True)
                        ray_metrics["fallback_dirs_used"] = fallback_dirs_used
                        dirs_use = dirs
                peak_real = float(self._predict_value_real(center.reshape(1, -1), class_idx=None)[0])
                peak_norm = float(f(center))
                if radii.size == 0 or dirs_use.size == 0:
                    if log_fn is not None:
                        log_fn("Skipping region without directions")
                    else:
                        warnings.warn(
                            "Región sin direcciones; se omite",
                            RuntimeWarning,
                        )
                else:
                    self.regions_.append(ClusterRegion(
                        cluster_id=len(self.regions_),
                        label="NA",
                        center=center,
                        directions=dirs_use,
                        radii=radii,
                        inflection_points=infl,
                        inflection_slopes=slopes,
                        peak_value_real=peak_real,
                        peak_value_norm=peak_norm,
                        metrics=ray_metrics,
                    ))
            self.regions_ = [r for r in self.regions_ if r.directions.size > 0]
            region_time = time.perf_counter() - t
            self._log(f"Region discovery in {region_time:.4f}s", level=logging.DEBUG)
            # Calcular la efectividad de cada región (score)
            if y is not None:
                t = time.perf_counter()
                X_arr = np.asarray(X, float)
                M = self._membership_matrix(X_arr)
                for k, reg in enumerate(self.regions_):
                    mask = M[:, k] == 1
                    if not np.any(mask):
                        reg.score = float("nan")
                        continue
                    if self.task == "classification":
                        y_true = np.asarray(y)[mask]
                        y_pred = np.full(len(y_true), reg.label)
                        reg.score = float(accuracy_score(y_true, y_pred))
                        metrics = self.cluster_metrics_cls
                        if metrics is None:
                            metrics = {
                                "precision": lambda a, b: precision_score(a, b, average="macro", zero_division=0),
                                "recall": lambda a, b: recall_score(a, b, average="macro", zero_division=0),
                                "f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
                            }
                        for name, func in metrics.items():
                            try:
                                reg.metrics[name] = float(func(y_true, y_pred))
                            except Exception:
                                reg.metrics[name] = float("nan")
                    else:
                        y_true = np.asarray(y, float)[mask]
                        y_pred = self.pipeline_.predict(X_arr[mask])
                        if len(y_true) >= 2 and np.var(y_true) > 0:
                            reg.score = float(r2_score(y_true, y_pred))
                        else:
                            reg.score = float("nan")
                        metrics = self.cluster_metrics_reg
                        if metrics is None:
                            metrics = {
                                "mse": mean_squared_error,
                                "mae": mean_absolute_error,
                            }
                        for name, func in metrics.items():
                            try:
                                reg.metrics[name] = float(func(y_true, y_pred))
                            except Exception:
                                reg.metrics[name] = float("nan")
                score_time = time.perf_counter() - t
                self._log(f"Region scoring in {score_time:.4f}s", level=logging.DEBUG)
            # Guardar etiquetas de entrenamiento para compatibilidad con
            # la API estándar de clustering de scikit-learn
            t = time.perf_counter()
            save_flag = self.save_labels
            self.save_labels = False
            try:
                if self.prediction_within_region:
                    self.labels_ = self.predict_regions(X)
                else:
                    self.labels_ = self.predict(X)
            finally:
                self.save_labels = save_flag
            self._log(f"Label prediction in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
        except Exception as exc:
            self._log(f"Error in fit: {exc}", level=logging.ERROR)
            raise
        runtime = time.perf_counter() - start
        self._log(f"fit completed in {runtime:.4f}s", level=logging.INFO)
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
        n_regions = len(self.regions_)

        if self.fast_membership and n_regions:
            centers = np.stack([reg.center for reg in self.regions_])
            V = X[:, None, :] - centers[None, :, :]
            norms = np.linalg.norm(V, axis=2) + 1e-12
            R = np.zeros((n, n_regions), dtype=int)
            for k, reg in enumerate(self.regions_):
                if reg.directions.size == 0:
                    warnings.warn(
                        "Región sin direcciones; se marca como fuera de la región",
                        RuntimeWarning,
                    )
                    continue
                U = V[:, k, :] / norms[:, k][:, None]
                dots = U @ reg.directions.T
                idx = np.argmax(dots, axis=1)
                r_boundary = reg.radii[idx]
                R[:, k] = (norms[:, k] <= r_boundary + 1e-12).astype(int)
            return R

        R = np.zeros((n, n_regions), dtype=int)
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
        self._log("Starting predict", level=logging.DEBUG)
        try:
            check_is_fitted(self, "regions_")
            X = np.asarray(X, dtype=float)
            if self.task == "classification":
                t = time.perf_counter()
                M = self._membership_matrix(X)
                self._log(f"Membership matrix computed in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
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
                    t = time.perf_counter()
                    base_pred = self.pipeline_.predict(X[none])
                    self._log(f"Fallback base prediction in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
                    pred[none] = base_pred
                result = pred
            else:
                t = time.perf_counter()
                result = self.pipeline_.predict(X)
                self._log(f"Base estimator prediction in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
        except Exception as exc:
            self._log(f"Error in predict: {exc}", level=logging.ERROR)
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict completed in {runtime:.4f}s", level=logging.INFO)
        self._maybe_save_labels(result, label_path)
        return result

    def predict_regions(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        label_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Region membership prediction for ``X``.

        For each sample the method returns:

        - ``-1`` if the sample does not fall inside any region.
        - The ``cluster_id`` of the region if it falls in exactly one.
        - A list with all ``cluster_id`` values when the sample belongs to
          multiple regions.

        This method ignores the base estimator and solely relies on the
        discovered regions.
        """
        start = time.perf_counter()
        self._log("Starting predict_regions", level=logging.DEBUG)
        try:
            check_is_fitted(self, "regions_")
            X = np.asarray(X, dtype=float)
            t = time.perf_counter()
            M = self._membership_matrix(X)
            self._log(f"Membership matrix computed in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            ids = np.array([reg.cluster_id for reg in self.regions_])
            pred: np.ndarray = np.empty(len(X), dtype=object)
            for i in range(len(X)):
                ks = np.where(M[i] == 1)[0]
                if len(ks) == 0:
                    pred[i] = -1
                elif len(ks) == 1:
                    pred[i] = ids[ks[0]]
                else:
                    pred[i] = ids[ks].tolist()
            result = pred
        except Exception as exc:
            self._log(f"Error in predict_regions: {exc}", level=logging.ERROR)
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict_regions completed in {runtime:.4f}s", level=logging.INFO)
        self._maybe_save_labels(result, label_path)
        return result

    def get_cluster(self, cluster_id: int) -> Optional[ClusterRegion]:
        """Return the :class:`ClusterRegion` with the given ``cluster_id``.

        Parameters
        ----------
        cluster_id : int
            Identifier of the cluster to retrieve.

        Returns
        -------
        ClusterRegion or None
            Cluster object matching ``cluster_id`` or ``None`` if not found.
        """
        check_is_fitted(self, "regions_")
        for reg in self.regions_:
            if reg.cluster_id == cluster_id:
                return reg
        return None

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
        self._log("Starting predict_proba", level=logging.DEBUG)
        try:
            check_is_fitted(self, "regions_")
            t = time.perf_counter()
            Xs = self.scaler_.transform(np.asarray(X, dtype=float))
            self._log(f"Data scaling in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            if self.task == "classification":
                t = time.perf_counter()
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
                self._log(f"Estimator call in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            else:
                t = time.perf_counter()
                vals = self.estimator_.predict(Xs)
                self._log(f"Estimator call in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
                vmin = min(reg.peak_value_real for reg in self.regions_)
                vmax = max(reg.peak_value_real for reg in self.regions_)
                rng = vmax - vmin if vmax > vmin else 1.0
                result = ((vals - vmin) / rng).reshape(-1, 1)
        except Exception as exc:
            self._log(f"Error in predict_proba: {exc}", level=logging.ERROR)
            raise
        runtime = time.perf_counter() - start
        self._log(f"predict_proba completed in {runtime:.4f}s", level=logging.INFO)
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
        self._log("Starting decision_function", level=logging.DEBUG)
        try:
            check_is_fitted(self, "regions_")
            t = time.perf_counter()
            Xs = self.scaler_.transform(np.asarray(X, dtype=float))
            self._log(f"Data scaling in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
            t = time.perf_counter()
            if hasattr(self.estimator_, "decision_function"):
                result = self.estimator_.decision_function(Xs)
            else:
                if self.task == "classification" and hasattr(self.estimator_, "predict_proba"):
                    result = self.estimator_.predict_proba(Xs)
                else:
                    result = self.estimator_.predict(Xs)
            self._log(f"Estimator call in {time.perf_counter() - t:.4f}s", level=logging.DEBUG)
        except Exception as exc:
            self._log(f"Error in decision_function: {exc}", level=logging.ERROR)
            raise
        runtime = time.perf_counter() - start
        self._log(f"decision_function completed in {runtime:.4f}s", level=logging.INFO)
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
            # centroid
            row_c = {
                "Type": "centroid",
                "Distance": 0.0,
                "ClusterID": reg.cluster_id,
                "Category": reg.label,
                "real_value": reg.peak_value_real,
                "norm_value": reg.peak_value_norm,
                "slope": np.nan,
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
                    "Type": "inflection_point",
                    "Distance": float(r),
                    "ClusterID": reg.cluster_id,
                    "Category": reg.label,
                    "real_value": float(self._predict_value_real(p.reshape(1, -1), class_idx=cls_index)[0]),
                    "norm_value": np.nan,
                    "slope": float(m),
                }
                for j in range(d):
                    row_i[feature_names[j]] = float(p[j])
                rows.append(row_i)
        return pd.DataFrame(rows)

    # -------- Visualization (2D pairs) --------

    def _plot_single_pair_classif(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pair: Tuple[int, int],
        class_label: Any,
        class_colors: Dict[Any, str],
        feature_names: Sequence[str],
        grid_res: int = 200,
        alpha_surface: float = 0.6,
    ) -> plt.Figure:
        """Draw the probability surface for a pair of features and one class.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Class labels.
        pair : tuple of int
            Indices ``(i, j)`` of the features to plot.
        class_label : Any
            Class whose probability surface is drawn.
        class_colors : dict
            Mapping from class to color for scatter points.
        feature_names : sequence of str
            Names corresponding to the columns of ``X``.
        grid_res : int, default=200
            Resolution of the mesh used for the surface.
        alpha_surface : float, default=0.6
            Surface transparency.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            Figure containing the plot.

        Examples
        --------
        >>> sh = ModalBoundaryClustering().fit(X, y)
        >>> sh._plot_single_pair_classif(X, y, (0, 1), 0, {0: 'red', 1: 'blue'})
        """
        i, j = pair
        xi, xj = X[:, i], X[:, j]
        xi_lin = np.linspace(xi.min(), xi.max(), grid_res)
        xj_lin = np.linspace(xj.min(), xj.max(), grid_res)
        XI, XJ = np.meshgrid(xi_lin, xj_lin)

        cls_idx = list(self.classes_).index(class_label)
        d = X.shape[1]
        fixed = np.mean(X, axis=0)
        X_std = X.std(0)
        X_grid = np.tile(fixed, (grid_res * grid_res, 1))
        X_grid[:, i] = XI.ravel()
        X_grid[:, j] = XJ.ravel()
        Z = self._predict_value_real(X_grid, class_idx=cls_idx).reshape(XI.shape)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(
            f"Prob. clase '{class_label}' vs ({feature_names[i]},{feature_names[j]})"
        )
        cf = ax.contourf(XI, XJ, Z, levels=20, alpha=alpha_surface)
        fig.colorbar(cf, ax=ax, label=f"P({class_label})")

        # puntos
        for c in self.classes_:
            mask = (y == c)
            ax.scatter(
                X[mask, i],
                X[mask, j],
                s=18,
                c=class_colors[c],
                label=str(c),
                edgecolor='k',
                linewidths=0.3,
            )

        # fronteras de las regiones de esta clase
        for reg in [r for r in self.regions_ if r.label == class_label]:
            center2 = reg.center.copy()
            mask_others = np.ones(d, dtype=bool)
            mask_others[[i, j]] = False
            center2[mask_others] = fixed[mask_others]

            U2 = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            D = np.zeros((len(U2), d))
            D[:, i] = np.cos(U2)
            D[:, j] = np.sin(U2)

            f = self._build_value_fn(
                class_idx=cls_idx,
                norm_stats=self._build_norm_stats(X, cls_idx),
            )
            perc = self.percentiles_[cls_idx] if self.stop_criteria == "percentile" else None
            _, pts, _, _ = self._scan_radii(center2, f, D, X_std, percentiles=perc, return_metrics=True)
            pts = pts[:, [i, j]]
            ctr = center2[[i, j]]
            ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
            order = np.argsort(ang)
            poly = pts[order]
            col = class_colors[class_label]
            ax.fill(
                poly[:, 0],
                poly[:, 1],
                color=col,
                alpha=0.15,
                zorder=1,
            )
            ax.plot(
                np.r_[poly[:, 0], poly[0, 0]],
                np.r_[poly[:, 1], poly[0, 1]],
                color=col,
                linewidth=2,
                label=f"frontera {reg.cluster_id} ({class_label})",
            )
            ax.scatter(
                ctr[0],
                ctr[1],
                c=col,
                marker='X',
                s=80,
                label=f"centro {reg.cluster_id} ({class_label})",
            )

        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_xlim(xi.min(), xi.max())
        ax.set_ylim(xj.min(), xj.max())
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def _plot_single_pair_reg(
        self,
        X: np.ndarray,
        pair: Tuple[int, int],
        feature_names: Sequence[str],
        y: Optional[np.ndarray] = None,
        grid_res: int = 200,
        alpha_surface: float = 0.6,
        max_classes: Optional[int] = None,
    ) -> plt.Figure:
        """Draw predicted-value surface and decile regions for a feature pair.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        pair : tuple of int
            Indices ``(i, j)`` of the features to plot.
        feature_names : sequence of str
            Names corresponding to the columns of ``X``.
        y : ndarray of shape (n_samples,), optional
            Values used to compute deciles. If ``None``, the model predictions
            for ``X`` are used instead.
        grid_res : int, default=200
            Resolution of the mesh used for the surface.
        alpha_surface : float, default=0.6
            Surface transparency.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            Figure containing the plot.

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
        d = X.shape[1]
        fixed = np.mean(X, axis=0)
        X_grid = np.tile(fixed, (grid_res * grid_res, 1))
        X_grid[:, i] = XI.ravel()
        X_grid[:, j] = XJ.ravel()
        Z = self._predict_value_real(X_grid, class_idx=None).reshape(XI.shape)

        vals = y if y is not None else self._predict_value_real(X, class_idx=None)
        dec = np.percentile(vals, np.linspace(0, 100, 11))
        bins = dec[1:-1]
        ids = np.digitize(vals, bins, right=True)
        palette = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#a65628",
            "#f781bf",
            "#999999",
            "#66c2a5",
            "#ffd92f",
        ]
        n_dec = 10 if max_classes is None else min(10, max_classes)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(
            f"Deciles de y_pred vs ({feature_names[i]},{feature_names[j]})"
        )
        cf = ax.contourf(XI, XJ, Z, levels=20, alpha=alpha_surface)
        fig.colorbar(cf, ax=ax, label="y_pred")

        # decile boundaries
        levels = dec[1:n_dec]
        for lvl, col in zip(levels, palette[: len(levels)]):
            ax.contour(XI, XJ, Z, levels=[lvl], colors=[col], linewidths=2)

        # scatter points coloured by decile
        for k in range(10):
            mask = ids == k
            if not np.any(mask):
                continue
            ax.scatter(
                X[mask, i],
                X[mask, j],
                s=18,
                c=palette[k],
                label=f"dec {k+1}" if k < n_dec else None,
                edgecolor='k',
                linewidths=0.3,
            )

        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    def plot_pairs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        max_pairs: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        block_size: Optional[int] = None,
        max_classes: Optional[int] = None,
    ):
        """Visualize 2D surfaces for feature pairs.

        Generates one figure for each ``(i, j)`` feature combination up to
        ``max_pairs``. In classification, the probability of each class is shown;
        in regression, decile regions of the predicted value are drawn.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to set the range of each axis.
        y : ndarray of shape (n_samples,), optional
            True labels; required when ``task='classification'``.
        max_pairs : int, optional
            Maximum number of combinations to plot. If ``None`` all possible
            combinations are generated.
        feature_names : sequence of str, optional
            Names for each feature to use on axis labels. If ``None`` and ``X``
            is a ``DataFrame`` its column names are used; otherwise generic
            ``feat i`` labels are shown.
        block_size : int, optional
            If provided, figures are displayed and closed in blocks of this size
            to avoid keeping too many open at once. When set, plots are shown
            automatically and it is not necessary to call ``plt.show()``
            afterwards. If ``None`` (default), all figures remain open and the
            caller should invoke ``plt.show()``.
        max_classes : int, optional
            Maximum number of classes (or deciles in regression) to visualize.
            If ``None`` all are shown.

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
        if hasattr(X, "iloc"):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        d = X_arr.shape[1]
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(map(str, X.columns))
            else:
                feature_names = getattr(self, "feature_names_in_", None) or [f"feat {i}" for i in range(d)]
        else:
            if len(feature_names) != d:
                raise ValueError("feature_names length must match number of features")
        pairs = list(itertools.combinations(range(d), 2))
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        block_figs: List[plt.Figure] = []
        if self.task == "classification":
            assert y is not None, "y required to plot classification."
            assert len(y) == len(X), "X e y deben tener la misma longitud."
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
            class_colors = {c: palette[i % len(palette)] for i, c in enumerate(self.classes_)}
            labels = list(self.classes_)
            if max_classes is not None:
                labels = labels[:max_classes]
            for pair in pairs:
                for label in labels:
                    fig = self._plot_single_pair_classif(
                        X_arr, y, pair, label, class_colors, feature_names
                    )
                    if block_size:
                        block_figs.append(fig)
                        if len(block_figs) >= block_size:
                            plt.show()
                            for f in block_figs:
                                plt.close(f)
                            block_figs.clear()
        else:
            y_vals = y if y is not None else self._predict_value_real(X_arr, class_idx=None)
            for pair in pairs:
                fig = self._plot_single_pair_reg(
                    X_arr, pair, feature_names, y_vals, max_classes=max_classes
                )
                if block_size:
                    block_figs.append(fig)
                    if len(block_figs) >= block_size:
                        plt.show()
                        for f in block_figs:
                            plt.close(f)
                        block_figs.clear()

        if block_size and block_figs:
            plt.show()
            for f in block_figs:
                plt.close(f)

    def plot_pair_3d(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        pair: Tuple[int, int],
        class_label: Optional[Any] = None,
        grid_res: int = 50,
        alpha_surface: float = 0.6,
        engine: str = "matplotlib",
        feature_names: Optional[Sequence[str]] = None,
    ) -> Any:
        """Visualize probability (or predicted value) as 3D surface.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to define the range of each axis.
        pair : tuple of int
            Indices ``(i, j)`` of the features to plot.
        class_label : optional
            Class to visualize when ``task='classification'``.
        grid_res : int, default=50
            Resolution of the mesh used for the surface.
        alpha_surface : float, default=0.6
            Surface transparency.
        engine : {"matplotlib", "plotly"}, default="matplotlib"
            Rendering engine. ``"plotly"`` produces an interactive figure
            (requires the optional ``plotly`` dependency).
        feature_names : sequence of str, optional
            Names for each feature to use on axis labels. If ``None`` and ``X``
            is a ``DataFrame`` its column names are used; otherwise generic
            ``feat i`` labels are shown.

        Returns
        -------
        figure
            Matplotlib ``Figure`` or Plotly ``Figure`` depending on ``engine``.
        """
        check_is_fitted(self, "regions_")
        if hasattr(X, "iloc"):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        d = X_arr.shape[1]
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(map(str, X.columns))
            else:
                feature_names = getattr(self, "feature_names_in_", None) or [f"feat {i}" for i in range(d)]
        else:
            if len(feature_names) != d:
                raise ValueError("feature_names length must match number of features")
        i, j = pair
        xi, xj = X_arr[:, i], X_arr[:, j]
        xi_lin = np.linspace(xi.min(), xi.max(), grid_res)
        xj_lin = np.linspace(xj.min(), xj.max(), grid_res)
        XI, XJ = np.meshgrid(xi_lin, xj_lin)

        if self.task == "classification":
            assert class_label is not None, "class_label required for classification."
            class_idx = list(self.classes_).index(class_label)
            zlabel = f"P({class_label})"
            title = f"Prob. clase '{class_label}' vs ({feature_names[i]},{feature_names[j]})"
        else:
            class_idx = None
            zlabel = "y_pred"
            title = f"Valor predicho vs ({feature_names[i]},{feature_names[j]})"

        Z = np.zeros_like(XI, dtype=float)
        for r in range(grid_res):
            X_full = np.tile(np.mean(X_arr, axis=0), (grid_res, 1))
            X_full[:, i] = XI[r, :]
            X_full[:, j] = XJ[r, :]
            Z[r, :] = self._predict_value_real(X_full, class_idx=class_idx)

        if engine == "plotly":
            try:
                import plotly.graph_objects as go
            except Exception as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "plotly is required when engine='plotly'"
                ) from exc

            fig = go.Figure(
                data=[
                    go.Surface(
                        x=XI,
                        y=XJ,
                        z=Z,
                        colorscale="Viridis",
                        opacity=alpha_surface,
                    )
                ]
            )
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=feature_names[i],
                    yaxis_title=feature_names[j],
                    zaxis_title=zlabel,
                ),
            )
            return fig

        if engine != "matplotlib":
            raise ValueError("engine must be 'matplotlib' or 'plotly'")

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XI, XJ, Z, cmap="viridis", alpha=alpha_surface)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        plt.tight_layout()

        return fig
