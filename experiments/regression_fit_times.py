from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from sheshe import (
    CheChe,
    ModalBoundaryClustering,
    ModalScoutEnsemble,
    ShuShu,
)


def main() -> None:
    X, y = make_regression(n_samples=200, n_features=3, noise=0.1, random_state=0)

    predictors = [
        ("ModalBoundaryClustering", ModalBoundaryClustering(task="regression")),
        ("ShuShu", ShuShu()),
        ("CheChe", CheChe()),
        (
            "ModalScoutEnsemble",
            ModalScoutEnsemble(base_estimator=LinearRegression(), task="regression"),
        ),
    ]

    results = []
    for name, estimator in predictors:
        score_model = LinearRegression().fit(X, y)
        start = time.perf_counter()
        try:
            estimator.fit(X, y, score_model=score_model)
            elapsed = time.perf_counter() - start
            results.append({"predictor": name, "fit_time_s": elapsed, "status": "ok"})
        except Exception as exc:  # noqa: BLE001 - capture all for reporting
            results.append(
                {
                    "predictor": name,
                    "fit_time_s": float("nan"),
                    "status": type(exc).__name__,
                }
            )

    df = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent.parent / "benchmark" / "regression_fit_times.csv"
    df.to_csv(out_path, index=False)
    print(df)


if __name__ == "__main__":
    main()
