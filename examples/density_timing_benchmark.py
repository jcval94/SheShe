import time
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_blobs

from sheshe import ModalBoundaryClustering


def time_fit(n: int) -> Tuple[float, float]:
    """Return fitting times without and with density for dataset of size n."""
    X, y = make_blobs(n_samples=n, centers=5, random_state=0)

    model_a = ModalBoundaryClustering(density_alpha=0, random_state=0)
    start = time.perf_counter()
    model_a.fit(X, y)
    t_a = time.perf_counter() - start

    model_b = ModalBoundaryClustering(density_alpha=1, random_state=0)
    start = time.perf_counter()
    model_b.fit(X, y)
    t_b = time.perf_counter() - start

    return t_a, t_b


def main() -> None:
    sizes: List[int] = [100 * 2 ** k for k in range(10)]
    print("n_samples, time_no_density(s), time_with_density(s), delta(s)")
    for n in sizes:
        t0, t1 = time_fit(n)
        print(f"{n:8d}, {t0:.4f}, {t1:.4f}, {t1 - t0:.4f}")


if __name__ == "__main__":
    main()
