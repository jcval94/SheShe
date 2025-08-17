import sys
sys.path.append("src")
import pandas as pd
import numpy as np
import time
from sheshe.sheshe import ModalBoundaryClustering


def run_benchmark():
    sizes = [1_000, 10_000, 50_000]
    dims = 10
    results = []
    for n in sizes:
        X = pd.DataFrame(np.random.randn(n, dims))
        y = np.random.randint(0, 2, size=n)
        row = {"n_samples": n}
        for fast, label in [(False, "baseline"), (True, "fast")]:
            model = ModalBoundaryClustering(
                task="classification", random_state=0, fast_membership=fast
            )
            t0 = time.perf_counter()
            model.fit(X, y)
            row[f"fit_{label}"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            model.predict(X)
            row[f"predict_{label}"] = time.perf_counter() - t0
        results.append(row)
    df = pd.DataFrame(results)
    df.to_csv("benchmark/large_df_benchmark_results.csv", index=False)
    return df


if __name__ == "__main__":
    print(run_benchmark())
