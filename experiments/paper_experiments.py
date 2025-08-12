"""Benchmark experiments for the manuscript.

This module provides reproducible routines to evaluate SheShe on several
scenarios: supervised comparison against classical algorithms, ablation
studies over internal hyperparameters and sensitivity analysis with respect
to the dimensionality of the data and injected noise.  Executing this script
will create CSV tables and figures under ``benchmark/`` ready to be included
in a paper or report.

Usage
-----
>>> python experiments/paper_experiments.py

The following artefacts are generated inside ``benchmark/``:
- ``supervised_results.csv`` and ``*_supervised.png``
- ``ablation_results.csv`` and ``ablation_accuracy.png``
- ``sensitivity_results.csv`` and ``sensitivity_heatmap.png``
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sheshe import ModalBoundaryClustering


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def supervised_comparison(out_dir: Path) -> pd.DataFrame:
    """Compare SheShe with supervised baselines.

    Datasets: Iris and Wine from scikit-learn.
    Algorithms: Logistic Regression, k-Nearest Neighbours, Random Forest and
    SheShe (using LogisticRegression as base estimator).
    Metrics: classification accuracy and training+prediction runtime.
    """

    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
    }
    algorithms = {
        "LogReg": LogisticRegression(max_iter=500),
        "KNN": KNeighborsClassifier(),
        "RF": RandomForestClassifier(random_state=0),
    }
    results = []
    for dname, (X, y) in datasets.items():
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        # Baselines
        for aname, algo in algorithms.items():
            start = time.perf_counter()
            algo.fit(Xtr, ytr)
            y_pred = algo.predict(Xte)
            runtime = time.perf_counter() - start
            acc = accuracy_score(yte, y_pred)
            results.append(
                {
                    "dataset": dname,
                    "algorithm": aname,
                    "accuracy": acc,
                    "runtime_sec": runtime,
                }
            )
        # SheShe
        start = time.perf_counter()
        sh = ModalBoundaryClustering(
            base_estimator=LogisticRegression(max_iter=500),
            task="classification",
            random_state=0,
        ).fit(Xtr, ytr)
        y_pred = sh.predict(Xte)
        runtime = time.perf_counter() - start
        acc = accuracy_score(yte, y_pred)
        results.append(
            {
                "dataset": dname,
                "algorithm": "SheShe",
                "accuracy": acc,
                "runtime_sec": runtime,
            }
        )

    df = pd.DataFrame(results)
    _save(df, out_dir / "supervised_results.csv")

    for dname in df["dataset"].unique():
        subset = df[df["dataset"] == dname]
        plt.figure()
        plt.bar(subset["algorithm"], subset["accuracy"], color="tab:blue")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"Supervised comparison on {dname}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{dname}_supervised.png")
        plt.close()
    return df


def ablation_study(out_dir: Path) -> pd.DataFrame:
    """Evaluate the effect of ``base_2d_rays`` and ``direction``.

    Dataset: Iris.  Metric: accuracy.
    """

    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    results = []
    for rays in [4, 8, 12]:
        for direction in ["center_out", "outside_in"]:
            start = time.perf_counter()
            sh = ModalBoundaryClustering(
                base_estimator=LogisticRegression(max_iter=500),
                task="classification",
                base_2d_rays=rays,
                direction=direction,
                random_state=0,
            ).fit(Xtr, ytr)
            y_pred = sh.predict(Xte)
            runtime = time.perf_counter() - start
            acc = accuracy_score(yte, y_pred)
            results.append(
                {
                    "base_2d_rays": rays,
                    "direction": direction,
                    "accuracy": acc,
                    "runtime_sec": runtime,
                }
            )
    df = pd.DataFrame(results)
    _save(df, out_dir / "ablation_results.csv")
    pivot = df.pivot(index="base_2d_rays", columns="direction", values="accuracy")
    pivot.plot(kind="bar")
    plt.ylabel("Accuracy")
    plt.title("Ablation on Iris")
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_accuracy.png")
    plt.close()
    return df


def sensitivity_analysis(out_dir: Path) -> pd.DataFrame:
    """Study sensitivity to dimensionality and noise.

    Uses synthetic datasets generated with ``make_classification`` and adds
    Gaussian noise with standard deviation in ``{0.0, 0.3, 0.6}``.
    """

    results = []
    rng = np.random.default_rng(0)
    for n_features in [2, 4, 8, 16]:
        X, y = make_classification(
            n_samples=400,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=0,
        )
        for noise in [0.0, 0.3, 0.6]:
            X_noisy = X + rng.normal(scale=noise, size=X.shape)
            Xtr, Xte, ytr, yte = train_test_split(
                X_noisy, y, test_size=0.3, random_state=0, stratify=y
            )
            start = time.perf_counter()
            sh = ModalBoundaryClustering(
                base_estimator=LogisticRegression(max_iter=500),
                task="classification",
                random_state=0,
            ).fit(Xtr, ytr)
            y_pred = sh.predict(Xte)
            runtime = time.perf_counter() - start
            acc = accuracy_score(yte, y_pred)
            results.append(
                {
                    "n_features": n_features,
                    "noise": noise,
                    "accuracy": acc,
                    "runtime_sec": runtime,
                }
            )
    df = pd.DataFrame(results)
    _save(df, out_dir / "sensitivity_results.csv")
    pivot = df.pivot(index="n_features", columns="noise", values="accuracy")
    plt.figure()
    im = plt.imshow(pivot, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Noise σ")
    plt.ylabel("n_features")
    plt.colorbar(im, label="Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_heatmap.png")
    plt.close()
    return df


def run_all(out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or Path(__file__).parent.parent / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    supervised_comparison(out_dir)
    ablation_study(out_dir)
    sensitivity_analysis(out_dir)
    print(f"Artifacts saved to {out_dir.resolve()}")


if __name__ == "__main__":
    run_all()
