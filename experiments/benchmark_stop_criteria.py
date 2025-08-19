"""Benchmark stop criteria implementations.

This script compares the previous vectorized implementation of
``find_percentile_drop`` with the current optimized loop version.  It also
measures the overall fitting time of :class:`ModalBoundaryClustering` when using
the ``inflection`` and ``percentile`` stop criteria.

Run with ``PYTHONPATH=src python experiments/benchmark_stop_criteria.py``.
"""

import argparse
import csv
import os
import timeit

import numpy as np
from sklearn.datasets import load_iris

from sheshe.sheshe import ModalBoundaryClustering, find_percentile_drop


def _vectorized_find_percentile_drop(ts, vals, direction, percentiles, drop_fraction=0.5):
    """Vectorized version used previously (kept here for benchmarking)."""
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


def benchmark_functions(sizes, direction, reps, warmup):
    rng = np.random.default_rng(0)
    percentiles = np.linspace(0.0, 1.0, 21)
    results = []

    for n in sizes:
        ts = np.linspace(0, 5, n)
        vals = np.linspace(1.0, 0.0, n) + rng.normal(scale=0.01, size=n)

        vec_stmt = lambda: _vectorized_find_percentile_drop(
            ts, vals, direction, percentiles
        )
        loop_stmt = lambda: find_percentile_drop(ts, vals, direction, percentiles)

        assert np.allclose(vec_stmt(), loop_stmt())

        for _ in range(warmup):
            vec_stmt()
            loop_stmt()

        t_vec = timeit.repeat(vec_stmt, number=1, repeat=reps)
        t_loop = timeit.repeat(loop_stmt, number=1, repeat=reps)

        mean_vec = np.mean(t_vec)
        std_vec = np.std(t_vec, ddof=1) if reps > 1 else 0.0
        mean_loop = np.mean(t_loop)
        std_loop = np.std(t_loop, ddof=1) if reps > 1 else 0.0
        speedup = mean_vec / mean_loop if mean_loop > 0 else float("inf")

        results.extend(
            [
                {
                    "size": n,
                    "direction": direction,
                    "implementation": "vectorized",
                    "mean": mean_vec,
                    "std": std_vec,
                    "speedup": 1.0,
                },
                {
                    "size": n,
                    "direction": direction,
                    "implementation": "loop",
                    "mean": mean_loop,
                    "std": std_loop,
                    "speedup": speedup,
                },
            ]
        )

        print(
            f"n={n} direction={direction} -> "
            f"vectorized {mean_vec:.6f}s ± {std_vec:.6f}, "
            f"loop {mean_loop:.6f}s ± {std_loop:.6f}, "
            f"speedup {speedup:.2f}x"
        )

    return results


def benchmark_model_fit(reps, warmup):
    X, y = load_iris(return_X_y=True)
    times = {}
    results = []

    for crit in ("inflexion", "percentile"):
        def fit_once():
            sh = ModalBoundaryClustering(
                task="classification", stop_criteria=crit, random_state=0
            )
            sh.fit(X, y)

        for _ in range(warmup):
            fit_once()

        t = timeit.repeat(fit_once, number=1, repeat=reps)
        times[crit] = t
        mean_t = np.mean(t)
        std_t = np.std(t, ddof=1) if reps > 1 else 0.0
        results.append(
            {
                "size": X.shape[0],
                "direction": "N/A",
                "implementation": f"fit_{crit}",
                "mean": mean_t,
                "std": std_t,
                "speedup": None,
            }
        )
        print(
            f"ModalBoundaryClustering fit with stop_criteria='{crit}': "
            f"{mean_t:.4f}s ± {std_t:.4f}"
        )

    inf_mean = np.mean(times["inflexion"])
    pct_mean = np.mean(times["percentile"])
    fit_speedup = inf_mean / pct_mean if pct_mean > 0 else float("inf")
    for row in results:
        if row["implementation"] == "fit_inflexion":
            row["speedup"] = 1.0
        else:
            row["speedup"] = fit_speedup

    return results


def save_results(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["size", "direction", "implementation", "mean", "std", "speedup"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps", type=int, default=3, help="Number of repetitions for timing"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warm-up iterations before timing"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500],
        help="Sizes of the synthetic time series to test",
    )
    parser.add_argument(
        "--direction",
        choices=["center_out", "outside_in"],
        default="center_out",
        help="Direction used in find_percentile_drop",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = []
    rows.extend(benchmark_functions(args.sizes, args.direction, args.reps, args.warmup))
    rows.extend(benchmark_model_fit(args.reps, args.warmup))
    out_path = os.path.join("benchmark", "stop_criteria_results.csv")
    save_results(rows, out_path)
    print(f"Results written to {out_path}")

