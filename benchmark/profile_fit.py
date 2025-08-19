"""Profile fit and predict phases of ModalBoundaryClustering.

This script profiles the ``fit`` and ``predict`` methods using ``cProfile``.
Each phase is executed multiple times to obtain an average runtime and its
standard deviation.  The profiling results are stored in files sorted by
accumulated time (``profile_fit.stats`` and ``profile_predict.stats``) and a
comparative report highlighting the main hotspots is printed to stdout.
"""

from __future__ import annotations

import cProfile
import statistics
import time
from typing import Iterable, Tuple

import pstats
from sklearn.datasets import load_iris

from sheshe.sheshe import ModalBoundaryClustering


def _aggregate_stats(profiles: Iterable[cProfile.Profile]) -> pstats.Stats:
    """Aggregate multiple ``Profile`` objects into a single ``Stats`` object."""

    iterator = iter(profiles)
    first = next(iterator)
    stats = pstats.Stats(first)
    for pr in iterator:
        stats.add(pr)
    return stats


def _profile_function(func, runs: int) -> Tuple[float, float, pstats.Stats]:
    """Profile ``func`` ``runs`` times and return timing stats.

    Returns
    -------
    mean : float
        Average runtime across ``runs`` executions.
    stdev : float
        Standard deviation of the runtime.
    stats : :class:`pstats.Stats`
        Aggregated profiling statistics.
    """

    times = []
    profiles = []
    for _ in range(runs):
        pr = cProfile.Profile()
        start = time.time()
        pr.enable()
        func()
        pr.disable()
        times.append(time.time() - start)
        profiles.append(pr)

    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    stats = _aggregate_stats(profiles)
    return mean, stdev, stats


def profile_fit(model, X, y, runs: int = 5) -> Tuple[float, float, pstats.Stats]:
    """Profile the ``fit`` method of ``model``."""

    return _profile_function(lambda: model.fit(X, y), runs)


def profile_predict(model, X, runs: int = 5) -> Tuple[float, float, pstats.Stats]:
    """Profile the ``predict`` method of ``model``."""

    return _profile_function(lambda: model.predict(X), runs)


def save_stats(stats: pstats.Stats, filename: str) -> None:
    """Save profiling ``stats`` to ``filename`` sorted by cumulative time."""

    with open(filename, "w") as fh:
        stats.stream = fh
        stats.strip_dirs().sort_stats("cumtime").print_stats()


def extract_hotspots(stats: pstats.Stats, n: int = 5) -> Iterable[str]:
    """Return the top ``n`` hotspots from ``stats`` sorted by cumulative time."""

    stats.sort_stats("cumtime")
    entries = []
    for func, stat in stats.stats.items():
        ct = stat[3]  # cumulative time
        entries.append((ct, func))
    entries.sort(reverse=True)
    top = entries[:n]
    formatted = [f"{ct:.6f}s - {fn}:{ln} ({name})" for ct, (fn, ln, name) in top]
    return formatted


def main(runs: int = 5) -> None:
    X, y = load_iris(return_X_y=True)
    model = ModalBoundaryClustering(task="classification", random_state=42)

    fit_mean, fit_std, fit_stats = profile_fit(model, X, y, runs=runs)
    predict_mean, predict_std, predict_stats = profile_predict(model, X, runs=runs)

    save_stats(fit_stats, "profile_fit.stats")
    save_stats(predict_stats, "profile_predict.stats")

    print(f"fit:    {fit_mean:.3f}s ± {fit_std:.3f}s")
    print(f"predict: {predict_mean:.3f}s ± {predict_std:.3f}s")

    print("\nTop hotspots for fit:")
    for line in extract_hotspots(fit_stats):
        print("  ", line)

    print("\nTop hotspots for predict:")
    for line in extract_hotspots(predict_stats):
        print("  ", line)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

