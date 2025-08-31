import time
import numpy as np
import pandas as pd
import sys
from itertools import combinations

sys.path.append('src')
from sheshe import CheChe

try:  # pragma: no cover - optional dependency
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - SciPy not available
    _HAS_SCIPY = False


def shushu_frontiers(X: np.ndarray):
    """Compute 2D frontiers using shushu-style on-the-fly hulls."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    result = {}
    for i, j in combinations(range(d), 2):
        pts = X[:, [i, j]]
        boundary = None
        if _HAS_SCIPY and pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                boundary = pts[hull.vertices]
            except Exception:
                boundary = None
        if boundary is None:
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            boundary = np.array([
                [mins[0], mins[1]],
                [mins[0], maxs[1]],
                [maxs[0], maxs[1]],
                [maxs[0], mins[1]],
            ])
        result[(i, j)] = boundary
    return result


def run_ab_test(n_reps: int = 5, n_samples: int = 1000, n_features: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_reps):
        X = rng.normal(size=(n_samples, n_features))

        start = time.perf_counter()
        CheChe().fit(X)
        t_cheche = time.perf_counter() - start

        start = time.perf_counter()
        shushu_frontiers(X)
        t_shushu = time.perf_counter() - start

        rows.append({
            'time_cheche': t_cheche,
            'time_shushu': t_shushu,
        })

    df = pd.DataFrame(rows)
    summary = pd.DataFrame({
        'time_cheche_mean': [df['time_cheche'].mean()],
        'time_shushu_mean': [df['time_shushu'].mean()],
        'speedup': [df['time_shushu'].mean() / df['time_cheche'].mean()],
    })
    summary.to_csv('benchmark/cheche_vs_shushu_ab_test.csv', index=False)
    return summary


if __name__ == '__main__':
    result = run_ab_test()
    print(result)
