import cProfile
import csv
from pathlib import Path
import pstats

from sklearn.datasets import load_iris

from sheshe import ModalBoundaryClustering


def main(out_csv: Path = Path(__file__).with_name("fit_profile.csv")) -> None:
    data = load_iris()
    X, y = data.data, data.target

    profiler = cProfile.Profile()
    profiler.enable()
    ModalBoundaryClustering().fit(X, y)
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")

    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["function", "ncalls", "tottime", "cumtime"])
        for (filename, line, func), stat in stats.stats.items():
            if "sheshe" not in filename:
                continue
            ncalls = stat[0]
            tottime = stat[2]
            cumtime = stat[3]
            writer.writerow([f"{filename}:{line}:{func}", ncalls, f"{tottime:.6f}", f"{cumtime:.6f}"])


if __name__ == "__main__":
    main()
