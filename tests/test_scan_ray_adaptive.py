import numpy as np
from sheshe.sheshe import _scan_ray_adaptive


def _first_below(vals, threshold):
    below = np.flatnonzero(vals < threshold)
    return int(below[0]) if below.size else int(len(vals) - 1)


def scan_ray_dense(center, direction_unit, r_min, r_max, f, threshold, steps=1000):
    rs = np.linspace(r_min, r_max, steps)
    P = center[None, :] + rs[:, None] * direction_unit[None, :]
    vals = np.array([f(p) for p in P])
    idx = _first_below(vals, threshold)
    if idx < len(rs) - 1:
        left = rs[max(0, idx - 1)]
        right = rs[idx]
        left_val = vals[max(0, idx - 1)]
        right_val = vals[idx]
        slope = (right_val - left_val) / (right - left + 1e-12)
        return left, slope, len(rs)
    else:
        return (
            rs[-1],
            (vals[-1] - vals[0]) / (rs[-1] - rs[0] + 1e-12),
            len(rs),
        )


def test_scan_ray_adaptive_matches_dense():
    center = np.zeros(2)
    direction = np.array([1.0, 0.0])
    f = lambda x: float(np.exp(-np.linalg.norm(x)))
    threshold = np.exp(-2.0)
    r_min, r_max = 0.0, 3.0

    dense_r, dense_slope, _ = scan_ray_dense(
        center, direction, r_min, r_max, f, threshold, steps=1000
    )
    adaptive_r, adaptive_slope, _ = _scan_ray_adaptive(
        center,
        direction,
        r_min,
        r_max,
        f,
        threshold,
        coarse_steps=12,
        refine_steps=6,
    )

    assert np.isclose(adaptive_r, dense_r, atol=1e-2)
    assert np.isclose(adaptive_slope, dense_slope, atol=1e-3)


def test_scan_ray_adaptive_reduces_evaluations():
    center = np.zeros(2)
    direction = np.array([1.0, 0.0])
    f = lambda x: float(np.exp(-np.linalg.norm(x)))
    threshold = np.exp(-2.0)
    r_min, r_max = 0.0, 3.0

    _, _, dense_evals = scan_ray_dense(
        center, direction, r_min, r_max, f, threshold, steps=1000
    )
    _, _, adaptive_evals = _scan_ray_adaptive(
        center,
        direction,
        r_min,
        r_max,
        f,
        threshold,
        coarse_steps=12,
        refine_steps=6,
    )

    assert adaptive_evals < dense_evals


def test_scan_ray_adaptive_no_drop_returns_rmax():
    center = np.zeros(2)
    direction = np.array([1.0, 0.0])
    f = lambda x: 1.0
    threshold = 0.5
    r_min, r_max = 0.0, 2.0

    r, slope, _ = _scan_ray_adaptive(
        center,
        direction,
        r_min,
        r_max,
        f,
        threshold,
        coarse_steps=8,
        refine_steps=4,
    )

    assert np.isclose(r, r_max)
    assert np.isclose(slope, 0.0)
