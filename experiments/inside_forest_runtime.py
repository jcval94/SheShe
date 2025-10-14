"""Runtime profiling for InsideForest using classification datasets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier

from sheshe import InsideForest


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[], Tuple]

    def load(self) -> Tuple:
        data = self.loader()
        if isinstance(data, tuple) and len(data) == 2:
            return data
        raise ValueError("Dataset loader must return (X, y)")


def _measure(callable_obj: Callable[[], None]) -> float:
    start = time.perf_counter()
    callable_obj()
    return time.perf_counter() - start


def _flatten_timings(prefix: str, timings: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}_s": value for key, value in timings.items()}


def _bottleneck(timings: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not timings:
        return None, 0.0
    name = max(timings, key=timings.get)
    return name, float(timings[name])


def profile_dataset(name: str, X, y) -> List[dict]:
    config = {
        "rf": {"random_state": 42, "n_estimators": 200, "min_samples_leaf": 2},
        "hypotheses": {"top_pairs": 5},
    }

    internal = InsideForest(config=config)
    fit_time_internal = _measure(lambda: internal.fit(X, y))
    transform_time_internal = _measure(lambda: internal.transform(X, mode="best"))
    explain_time_internal = _measure(lambda: internal.explain(top_k=5))
    hypotheses_time_internal = _measure(
        lambda: internal.generate_hypotheses(top_pairs=5)
    )

    rf_external = RandomForestClassifier(
        n_estimators=config["rf"]["n_estimators"],
        min_samples_leaf=config["rf"]["min_samples_leaf"],
        random_state=config["rf"]["random_state"],
        n_jobs=-1,
    )
    rf_external.fit(X, y)

    external = InsideForest(config=config)
    fit_time_external = _measure(
        lambda: external.fit(X, y, random_forest=rf_external)
    )
    transform_time_external = _measure(lambda: external.transform(X, mode="best"))
    explain_time_external = _measure(lambda: external.explain(top_k=5))
    hypotheses_time_external = _measure(
        lambda: external.generate_hypotheses(top_pairs=5)
    )

    timings_internal = internal.get_last_timings()
    timings_external = external.get_last_timings()

    fit_bottleneck_internal = _bottleneck(timings_internal.get("fit", {}))
    fit_bottleneck_external = _bottleneck(timings_external.get("fit", {}))
    transform_bottleneck_internal = _bottleneck(timings_internal.get("transform", {}))
    transform_bottleneck_external = _bottleneck(timings_external.get("transform", {}))
    explain_bottleneck_internal = _bottleneck(timings_internal.get("explain", {}))
    explain_bottleneck_external = _bottleneck(timings_external.get("explain", {}))
    hypo_bottleneck_internal = _bottleneck(
        timings_internal.get("generate_hypotheses", {})
    )
    hypo_bottleneck_external = _bottleneck(
        timings_external.get("generate_hypotheses", {})
    )

    return [
        {
            "dataset": name,
            "mode": "internal",
            "n_rules": len(internal.get_rules()),
            "n_regions": len(internal.get_regions()),
            "fit_time_s": fit_time_internal,
            "transform_time_s": transform_time_internal,
            "explain_time_s": explain_time_internal,
            "hypotheses_time_s": hypotheses_time_internal,
            "fit_bottleneck": fit_bottleneck_internal[0],
            "fit_bottleneck_time_s": fit_bottleneck_internal[1],
            "transform_bottleneck": transform_bottleneck_internal[0],
            "transform_bottleneck_time_s": transform_bottleneck_internal[1],
            "explain_bottleneck": explain_bottleneck_internal[0],
            "explain_bottleneck_time_s": explain_bottleneck_internal[1],
            "hypotheses_bottleneck": hypo_bottleneck_internal[0],
            "hypotheses_bottleneck_time_s": hypo_bottleneck_internal[1],
            **_flatten_timings("fit", timings_internal.get("fit", {})),
            **_flatten_timings("transform", timings_internal.get("transform", {})),
            **_flatten_timings("explain", timings_internal.get("explain", {})),
            **_flatten_timings(
                "generate_hypotheses",
                timings_internal.get("generate_hypotheses", {}),
            ),
        },
        {
            "dataset": name,
            "mode": "external",
            "n_rules": len(external.get_rules()),
            "n_regions": len(external.get_regions()),
            "fit_time_s": fit_time_external,
            "transform_time_s": transform_time_external,
            "explain_time_s": explain_time_external,
            "hypotheses_time_s": hypotheses_time_external,
            "fit_bottleneck": fit_bottleneck_external[0],
            "fit_bottleneck_time_s": fit_bottleneck_external[1],
            "transform_bottleneck": transform_bottleneck_external[0],
            "transform_bottleneck_time_s": transform_bottleneck_external[1],
            "explain_bottleneck": explain_bottleneck_external[0],
            "explain_bottleneck_time_s": explain_bottleneck_external[1],
            "hypotheses_bottleneck": hypo_bottleneck_external[0],
            "hypotheses_bottleneck_time_s": hypo_bottleneck_external[1],
            **_flatten_timings("fit", timings_external.get("fit", {})),
            **_flatten_timings("transform", timings_external.get("transform", {})),
            **_flatten_timings("explain", timings_external.get("explain", {})),
            **_flatten_timings(
                "generate_hypotheses",
                timings_external.get("generate_hypotheses", {}),
            ),
        },
    ]


def main() -> None:
    datasets = [
        DatasetSpec("iris", lambda: load_iris(return_X_y=True)),
        DatasetSpec("wine", lambda: load_wine(return_X_y=True)),
        DatasetSpec("digits", lambda: load_digits(return_X_y=True)),
    ]

    rows: List[dict] = []
    for spec in datasets:
        X, y = spec.load()
        rows.extend(profile_dataset(spec.name, X, y))

    df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent / "benchmark" / "inside_forest_runtime.csv"
    df.to_csv(out_path, index=False)
    print(df)


if __name__ == "__main__":
    main()
