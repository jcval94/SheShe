import sys
import time
import threading

import numpy as np
import pandas as pd
import psutil

sys.path.append("src")
from sheshe.sheshe import ModalBoundaryClustering


def _time_and_peak_memory(func, *args, **kwargs):
    """Return execution time and peak memory usage of ``func``.

    Memory usage is measured as the additional resident set size (MiB)
    observed during the execution of ``func``. ``psutil`` samples memory
    usage in a background thread every 10ms.
    """

    process = psutil.Process()
    peak = [process.memory_info().rss / (1024 ** 2)]  # MiB
    baseline = peak[0]
    running = True

    def monitor():
        while running:
            mem = process.memory_info().rss / (1024 ** 2)
            if mem > peak[0]:
                peak[0] = mem
            time.sleep(0.01)

    thread = threading.Thread(target=monitor)
    thread.start()
    start = time.perf_counter()
    func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    running = False
    thread.join()
    return elapsed, max(0.0, peak[0] - baseline)


def run_benchmark(n_reps: int = 3):
    sizes = [1_000, 10_000, 50_000, 100_000, 500_000]
    dims_list = [10, 50, 100]
    results = []
    for n in sizes:
        for dims in dims_list:
            X = pd.DataFrame(np.random.randn(n, dims))
            y = np.random.randint(0, 2, size=n)
            row = {"n_samples": n, "n_features": dims}
            times = {lbl: {"fit": [], "predict": []} for lbl in ["baseline", "fast"]}
            mems = {lbl: {"fit": [], "predict": []} for lbl in ["baseline", "fast"]}
            for _ in range(n_reps):
                for fast, label in [(False, "baseline"), (True, "fast")]:
                    model = ModalBoundaryClustering(
                        task="classification", random_state=0, fast_membership=fast,
                    )
                    t, m = _time_and_peak_memory(model.fit, X, y)
                    times[label]["fit"].append(t)
                    mems[label]["fit"].append(m)
                    t, m = _time_and_peak_memory(model.predict, X)
                    times[label]["predict"].append(t)
                    mems[label]["predict"].append(m)
            for label in ["baseline", "fast"]:
                for phase in ["fit", "predict"]:
                    row[f"{phase}_{label}_mean"] = float(np.mean(times[label][phase]))
                    row[f"{phase}_{label}_std"] = float(np.std(times[label][phase]))
                    row[f"{phase}_{label}_mem"] = float(np.max(mems[label][phase]))
            results.append(row)
    df = pd.DataFrame(results)
    df.to_csv("benchmark/large_df_benchmark_results.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run large dataframe benchmark")
    parser.add_argument("--n-reps", type=int, default=3, help="Number of repetitions")
    args = parser.parse_args()
    print(run_benchmark(n_reps=args.n_reps))

