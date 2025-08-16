"""Benchmark stop criteria implementations.

This script compares the previous vectorized implementation of
``find_percentile_drop`` with the current optimized loop version.  It also
measures the overall fitting time of :class:`ModalBoundaryClustering` when using
the ``inflection`` and ``percentile`` stop criteria.

Run with ``PYTHONPATH=src python experiments/benchmark_stop_criteria.py``.
"""

import time
import numpy as np
from sklearn.datasets import load_iris

from sheshe.sheshe import ModalBoundaryClustering, find_percentile_drop


def _vectorized_find_percentile_drop(ts, vals, direction, deciles, drop_fraction=0.5):
    """Vectorized version used previously (kept here for benchmarking)."""
    if direction == "outside_in":
        ts_scan = ts[::-1]
        vals_scan = vals[::-1]
    else:
        ts_scan = ts
        vals_scan = vals

    dec_idx = np.searchsorted(deciles, vals_scan, side="right") - 1
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
        for j in range(1, len(vals_scan)):
            if vals_scan[j] <= target:
                t0, t1 = ts_scan[j - 1], ts_scan[j]
                v0, v1 = vals_scan[j - 1], vals_scan[j]
                alpha = float(
                    np.clip((target - v0) / (v1 - v0 + 1e-12), 0.0, 1.0)
                )
                t_scan = t0 + alpha * (t1 - t0)
                m_scan = (v1 - v0) / (t1 - t0 + 1e-12)
                break

    t_abs = t_scan if direction == "center_out" else (ts[-1] - t_scan)
    return float(t_abs), float(m_scan)


def benchmark_functions():
    rng = np.random.default_rng(0)
    ts = np.linspace(0, 5, 500)
    vals = np.linspace(1.0, 0.0, 500) + rng.normal(scale=0.01, size=500)
    deciles = np.linspace(0.0, 1.0, 11)
    reps = 2000

    t0 = time.time()
    for _ in range(reps):
        _vectorized_find_percentile_drop(ts, vals, "center_out", deciles)
    t_vec = time.time() - t0

    t0 = time.time()
    for _ in range(reps):
        find_percentile_drop(ts, vals, "center_out", deciles)
    t_loop = time.time() - t0

    res_old = _vectorized_find_percentile_drop(ts, vals, "center_out", deciles)
    res_new = find_percentile_drop(ts, vals, "center_out", deciles)
    assert np.allclose(res_old, res_new)

    print(f"vectorized implementation: {t_vec:.4f}s")
    print(f"loop implementation:       {t_loop:.4f}s")
    if t_loop > 0:
        print(f"speedup: {t_vec / t_loop:.2f}x")


def benchmark_model_fit():
    X, y = load_iris(return_X_y=True)
    for crit in ("inflexion", "percentile"):
        sh = ModalBoundaryClustering(task="classification", stop_criteria=crit, random_state=0)
        t0 = time.time(); sh.fit(X, y); t = time.time() - t0
        print(f"ModalBoundaryClustering fit with stop_criteria='{crit}': {t:.4f}s")


if __name__ == "__main__":
    benchmark_functions()
    benchmark_model_fit()

