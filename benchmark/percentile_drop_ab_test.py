import time
import numpy as np
import pandas as pd
import sys
sys.path.append("src")
from sheshe.sheshe import find_percentile_drop


def find_percentile_drop_old(ts, vals, direction, percentiles, drop_fraction=0.5):
    """Previous loop-based implementation of find_percentile_drop."""
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

    prev = int(np.searchsorted(percentiles, vals_scan[0], side="right") - 1)
    idx = None
    for j in range(1, len(vals_scan)):
        curr = int(np.searchsorted(percentiles, vals_scan[j], side="right") - 1)
        if curr < prev:
            idx = j
            break
        prev = curr

    if idx is not None:
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


def run_ab_test(n_reps: int = 5):
    sizes = [100, 1000, 10000]
    results = []
    rng = np.random.default_rng(0)
    for n in sizes:
        ts = np.linspace(0, 1, n)
        vals = np.linspace(1, 0, n) + rng.normal(scale=0.01, size=n)
        percentiles = np.quantile(vals, np.linspace(0, 1, 101))
        t_old = []
        t_new = []
        drop_diff = []
        slope_diff = []
        for _ in range(n_reps):
            start = time.perf_counter()
            td0, sl0 = find_percentile_drop_old(ts, vals, "center_out", percentiles)
            t_old.append(time.perf_counter() - start)
            start = time.perf_counter()
            td1, sl1 = find_percentile_drop(ts, vals, "center_out", percentiles)
            t_new.append(time.perf_counter() - start)
            drop_diff.append(abs(td0 - td1))
            slope_diff.append(abs(sl0 - sl1))
        results.append(
            {
                "n_points": n,
                "time_old_mean": float(np.mean(t_old)),
                "time_new_mean": float(np.mean(t_new)),
                "speedup": float(np.mean(t_old) / np.mean(t_new)),
                "t_drop_diff_max": float(np.max(drop_diff)),
                "slope_diff_max": float(np.max(slope_diff)),
            }
        )
    df = pd.DataFrame(results)
    df.to_csv("benchmark/percentile_drop_ab_test.csv", index=False)
    return df


if __name__ == "__main__":
    df = run_ab_test()
    print(df)
