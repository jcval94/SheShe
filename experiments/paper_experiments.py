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
from typing import Sequence

import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sheshe import ModalBoundaryClustering


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def supervised_comparison(out_dir: Path, seeds: Sequence[int]) -> pd.DataFrame:
    """Compare SheShe with supervised baselines using cross-validation.

    Datasets: Iris and Wine from scikit-learn.
    Algorithms: Logistic Regression, k-Nearest Neighbours, Random Forest,
    linear and kernel SVMs, Gradient Boosting and SheShe (using
    ``LogisticRegression`` as base estimator).
    Metrics: classification accuracy and separate fit/predict runtimes.
    """

    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
    }
    results = []
    for seed in seeds:
        algos = {
            "LogReg": LogisticRegression(max_iter=500, random_state=seed),
            "KNN": KNeighborsClassifier(),
            "RF": RandomForestClassifier(random_state=seed),
            "SVM-linear": SVC(kernel="linear"),
            "SVM-rbf": SVC(kernel="rbf"),
            "GBC": GradientBoostingClassifier(random_state=seed),
        }
        for dname, (X, y) in datasets.items():
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
                Xtr, Xte = X[tr_idx], X[te_idx]
                ytr, yte = y[tr_idx], y[te_idx]
                # Baselines
                for aname, algo in algos.items():
                    start = time.perf_counter()
                    algo.fit(Xtr, ytr)
                    fit_time = time.perf_counter() - start
                    start = time.perf_counter()
                    y_pred = algo.predict(Xte)
                    predict_time = time.perf_counter() - start
                    acc = accuracy_score(yte, y_pred)
                    results.append(
                        {
                            "dataset": dname,
                            "algorithm": aname,
                            "seed": seed,
                            "fold": fold,
                            "accuracy": acc,
                            "fit_time_sec": fit_time,
                            "predict_time_sec": predict_time,
                        }
                    )
                # SheShe
                start = time.perf_counter()
                sh = ModalBoundaryClustering(
                    base_estimator=LogisticRegression(max_iter=500, random_state=seed),
                    task="classification",
                    random_state=seed,
                ).fit(Xtr, ytr)
                fit_time = time.perf_counter() - start
                start = time.perf_counter()
                y_pred = sh.predict(Xte)
                predict_time = time.perf_counter() - start
                acc = accuracy_score(yte, y_pred)
                results.append(
                    {
                        "dataset": dname,
                        "algorithm": "SheShe",
                        "seed": seed,
                        "fold": fold,
                        "accuracy": acc,
                        "fit_time_sec": fit_time,
                        "predict_time_sec": predict_time,
                    }
                )

    df = pd.DataFrame(results)
    _save(df, out_dir / "supervised_results.csv")

    summary = (
        df.groupby(["dataset", "algorithm"])
        .agg(
            {
                "accuracy": ["mean", "std"],
                "fit_time_sec": ["mean", "std"],
                "predict_time_sec": ["mean", "std"],
            }
        )
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.reset_index(inplace=True)
    _save(summary, out_dir / "supervised_summary.csv")

    for dname in summary["dataset"].unique():
        subset = summary[summary["dataset"] == dname]
        plt.figure()
        plt.bar(
            subset["algorithm"],
            subset["accuracy_mean"],
            yerr=subset["accuracy_std"],
            capsize=4,
            color="tab:blue",
        )
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title(f"Supervised comparison on {dname}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{dname}_supervised.png")
        plt.close()
    return summary


def ablation_study(out_dir: Path, seeds: Sequence[int]) -> pd.DataFrame:
    """Evaluate the effect of ``base_2d_rays`` and ``direction``.

    Dataset: Iris.  Metric: accuracy.
    """

    X, y = load_iris(return_X_y=True)
    results = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        for rays in [4, 8, 12]:
            for direction in ["center_out", "outside_in"]:
                start = time.perf_counter()
                sh = ModalBoundaryClustering(
                    base_estimator=LogisticRegression(max_iter=500, random_state=seed),
                    task="classification",
                    base_2d_rays=rays,
                    direction=direction,
                    random_state=seed,
                ).fit(Xtr, ytr)
                y_pred = sh.predict(Xte)
                runtime = time.perf_counter() - start
                acc = accuracy_score(yte, y_pred)
                results.append(
                    {
                        "base_2d_rays": rays,
                        "direction": direction,
                        "seed": seed,
                        "accuracy": acc,
                        "runtime_sec": runtime,
                    }
                )
    df = pd.DataFrame(results)
    _save(df, out_dir / "ablation_results.csv")

    summary = (
        df.groupby(["base_2d_rays", "direction"])
        .agg({"accuracy": ["mean", "std"], "runtime_sec": ["mean", "std"]})
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.reset_index(inplace=True)
    _save(summary, out_dir / "ablation_summary.csv")

    pivot = summary.pivot(index="base_2d_rays", columns="direction", values="accuracy_mean")
    err = summary.pivot(index="base_2d_rays", columns="direction", values="accuracy_std")
    pivot.plot(kind="bar", yerr=err, capsize=4)
    plt.ylabel("Accuracy")
    plt.title("Ablation on Iris")
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_accuracy.png")
    plt.close()
    return summary


def sensitivity_analysis(out_dir: Path, seeds: Sequence[int]) -> pd.DataFrame:
    """Study sensitivity to dimensionality and noise.

    Uses synthetic datasets generated with ``make_classification`` and adds
    Gaussian noise with standard deviation in ``{0.0, 0.3, 0.6}``.
    """

    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for n_features in [2, 4, 8, 16]:
            X, y = make_classification(
                n_samples=400,
                n_features=n_features,
                n_informative=n_features // 2,
                n_redundant=0,
                n_clusters_per_class=1,
                random_state=seed,
            )
            for noise in [0.0, 0.3, 0.6]:
                X_noisy = X + rng.normal(scale=noise, size=X.shape)
                Xtr, Xte, ytr, yte = train_test_split(
                    X_noisy, y, test_size=0.3, random_state=seed, stratify=y
                )
                start = time.perf_counter()
                sh = ModalBoundaryClustering(
                    base_estimator=LogisticRegression(max_iter=500, random_state=seed),
                    task="classification",
                    random_state=seed,
                ).fit(Xtr, ytr)
                y_pred = sh.predict(Xte)
                runtime = time.perf_counter() - start
                acc = accuracy_score(yte, y_pred)
                results.append(
                    {
                        "n_features": n_features,
                        "noise": noise,
                        "seed": seed,
                        "accuracy": acc,
                        "runtime_sec": runtime,
                    }
                )
    df = pd.DataFrame(results)
    _save(df, out_dir / "sensitivity_results.csv")

    summary = (
        df.groupby(["n_features", "noise"])
        .agg({"accuracy": ["mean", "std"], "runtime_sec": ["mean", "std"]})
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.reset_index(inplace=True)
    _save(summary, out_dir / "sensitivity_summary.csv")

    pivot = summary.pivot(index="n_features", columns="noise", values="accuracy_mean")
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
    return summary


def run_all(out_dir: Path | None = None, seeds: Sequence[int] | None = None) -> None:
    out_dir = Path(out_dir or Path(__file__).parent.parent / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or range(5))
    supervised_comparison(out_dir, seeds)
    ablation_study(out_dir, seeds)
    sensitivity_analysis(out_dir, seeds)
    print(f"Artifacts saved to {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimentos para el paper")
    parser.add_argument("--runs", type=int, default=5, help="Número de repeticiones con distintas semillas")
    args = parser.parse_args()
    run_all(seeds=range(args.runs))
