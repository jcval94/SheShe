"""Example of using SheShe and saving visualizations.

This script trains ``ModalBoundaryClustering`` on the Iris dataset and
stores the generated pair plots inside ``examples/images``.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sheshe import ModalBoundaryClustering


def main() -> None:
    """Entrena SheShe con Iris y guarda pares de gráficas en disco."""

    iris = load_iris()
    X, y = iris.data, iris.target

    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=1000),
        task="classification",
        base_2d_rays=8,  # número de rayos por cada plano 2D considerado
        random_state=0,
        drop_fraction=0.5,
    ).fit(X, y)

    # Create up to two pairwise plots
    sh.plot_pairs(X, y, max_pairs=2)

    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each generated figure
    for i, fig_num in enumerate(plt.get_fignums()):
        plt.figure(fig_num)
        plt.savefig(out_dir / f"pair_{i}.png")
        plt.close(fig_num)


if __name__ == "__main__":
    main()
