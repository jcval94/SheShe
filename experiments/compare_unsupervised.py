"""
Compara SheShe con algoritmos no supervisados en varios conjuntos de datos.

Genera métricas de *clustering* (ARI, homogeneidad, completitud y V-measure)
para ocho dataframes distintos y diferentes configuraciones de parámetros.
Cada ejecución registra de manera separada los tiempos y memoria consumida en
las fases de entrenamiento y predicción, almacenados en ``fit_time_sec``,
``predict_time_sec``, ``fit_mem_mb`` y ``predict_mem_mb``.
Los resultados se guardan en ``benchmark/unsupervised_results.csv``.
"""
from pathlib import Path
import math
import time
import psutil
from typing import Iterable, Tuple, Optional, Sequence

import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)

from sheshe import ModalScoutEnsemble


Metric = Tuple[str, callable]


def _evaluate(y_true, y_pred, metrics: Iterable[Metric]) -> dict:
    """Calcula valores de métricas de agrupamiento.

    Parameters
    ----------
    y_true:
        Etiquetas verdaderas utilizadas como referencia.
    y_pred:
        Etiquetas predichas por el algoritmo de *clustering*.
    metrics:
        Iterable de tuplas ``(nombre, funcion)`` donde ``funcion`` recibe
        ``(y_true, y_pred)`` y devuelve un valor numérico.

    Returns
    -------
    dict
        Diccionario ``{nombre_metrica: valor}``. Si ``y_pred`` contiene una sola
        etiqueta distinta (p.ej. DBSCAN agrupando todo como ruido) se devuelven
        ``NaN`` en todas las métricas.
    """

    if len(set(y_pred)) <= 1:
        return {name: math.nan for name, _ in metrics}
    return {name: fn(y_true, y_pred) for name, fn in metrics}


def run(
    verbose: bool = False,
    save_labels: bool = True,
    out_dir: Optional[Path] = None,
    seeds: Sequence[int] | None = None,
) -> None:
    """Ejecuta el experimento de comparación y guarda resultados.

    Parameters
    ----------
    verbose:
        Si ``True``, muestra mensajes de progreso y tiempos de ejecución.
    save_labels:
        Cuando es ``True``, almacena las etiquetas predichas en ficheros ``.labels``.
    out_dir:
        Directorio de salida donde se guardan los resultados y etiquetas. Por
        defecto se usa ``benchmark/`` en la raíz del proyecto.
    seeds:
        Secuencia de semillas para repetir cada experimento. Cuando es ``None``
        se utilizan ``range(5)``.
    """

    seeds = list(seeds or range(5))

    def _datasets(seed: int):
        data = {
            "iris": load_iris(return_X_y=True),
            "wine": load_wine(return_X_y=True),
            "breast_cancer": load_breast_cancer(return_X_y=True),
            "digits": load_digits(return_X_y=True),
            "moons": make_moons(n_samples=300, noise=0.05, random_state=seed),
            "blobs": make_blobs(n_samples=300, centers=3, random_state=seed),
            "circles": make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=seed),
        }
        try:
            X_housing, y_housing = fetch_california_housing(return_X_y=True)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y_housing), size=300, replace=False)
            X_housing, y_housing = X_housing[idx], y_housing[idx]
            bins = np.quantile(y_housing, [0.25, 0.5, 0.75])
            y_housing = np.digitize(y_housing, bins)
            data["california_housing"] = (X_housing, y_housing)
        except Exception as exc:
            if verbose:
                print(f"[run] California housing no disponible: {exc}")
        return data

    metrics: Iterable[Metric] = [
        ("ARI", adjusted_rand_score),
        ("homogeneity", homogeneity_score),
        ("completeness", completeness_score),
        ("v_measure", v_measure_score),
    ]

    results = []
    out_dir = out_dir or Path(__file__).parent.parent / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_labels(y_pred, fname: str) -> None:
        if not save_labels:
            return
        label_path = out_dir / fname
        try:
            np.savetxt(label_path, y_pred, fmt="%s")
        except Exception as exc:  # pragma: no cover - logging auxiliar
            if verbose:
                print(f"No se pudieron guardar etiquetas en {label_path}: {exc}")

    for seed in seeds:
        datasets = _datasets(seed)
        for name, (X, y) in datasets.items():
            n_classes = len(set(y))

            # SheShe: barridos de C, max_order, jaccard_threshold y base_2d_rays
            def _run_sheshe(*, C: float, max_order: int, jacc: float, rays: int) -> None:
                try:
                    mem_before_fit = psutil.Process().memory_info().rss
                    start_fit = time.perf_counter()
                    sh = ModalScoutEnsemble(
                        base_estimator=LogisticRegression(
                            max_iter=500, C=C, random_state=seed
                        ),
                        task="classification",
                        random_state=seed,
                        max_order=max_order,
                        jaccard_threshold=jacc,
                        base_2d_rays=rays,
                        scout_kwargs={"max_order": max_order, "top_m": 4, "sample_size": None},
                        cv=2,
                    ).fit(X, y)
                    fit_time = time.perf_counter() - start_fit
                    fit_mem = (psutil.Process().memory_info().rss - mem_before_fit) / (1024 ** 2)

                    mem_before_pred = psutil.Process().memory_info().rss
                    start_pred = time.perf_counter()
                    y_pred = sh.predict(X)
                    predict_time = time.perf_counter() - start_pred
                    predict_mem = (psutil.Process().memory_info().rss - mem_before_pred) / (1024 ** 2)

                    metrics_dict = _evaluate(y, y_pred, metrics)
                    _save_labels(
                        y_pred,
                        f"{name}_SheShe_C-{C}_mo-{max_order}_jt-{jacc}_rays-{rays}_seed-{seed}.labels",
                    )
                except Exception as exc:
                    y_pred = []
                    metrics_dict = {name: math.nan for name, _ in metrics}
                    fit_time = predict_time = fit_mem = predict_mem = math.nan
                    if verbose:
                        print(
                            "SheShe falló en "
                            f"{name} (C={C}, max_order={max_order}, jt={jacc}, rays={rays}, seed={seed}): {exc}",
                        )
                runtime = (0 if math.isnan(fit_time) else fit_time) + (0 if math.isnan(predict_time) else predict_time)
                record = {
                    "dataset": name,
                    "algorithm": "SheShe",
                    "params": (
                        f"C={C},max_order={max_order},jaccard_threshold={jacc},",
                        f"base_2d_rays={rays}"
                    ),
                    "seed": seed,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    "fit_mem_mb": fit_mem,
                    "predict_mem_mb": predict_mem,
                    "runtime_sec": runtime,
                }
                record.update(metrics_dict)
                results.append(record)
                if verbose:
                    print(
                        "SheShe "
                        f"{name} C={C} max_order={max_order} jt={jacc} rays={rays} seed={seed} → {runtime:.4f}s",
                    )
            # Barrido sobre C (manteniendo el resto en valores por defecto)
            for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
                _run_sheshe(C=C, max_order=3, jacc=0.55, rays=24)

            # Barrido sobre max_order
            for max_order in [2]:
                _run_sheshe(C=1.0, max_order=max_order, jacc=0.55, rays=24)

            # Barrido sobre jaccard_threshold
            for jacc in [0.4, 0.7]:
                _run_sheshe(C=1.0, max_order=3, jacc=jacc, rays=24)

            # Barrido sobre base_2d_rays
            for rays in [16]:
                _run_sheshe(C=1.0, max_order=3, jacc=0.55, rays=rays)

            # KMeans: variar n_clusters (más valores)
            for k in [n_classes - 2, n_classes - 1, n_classes, n_classes + 1, n_classes + 2]:
                k = max(k, 1)
                try:
                    mem_before_fit = psutil.Process().memory_info().rss
                    start_fit = time.perf_counter()
                    km = KMeans(n_clusters=k, random_state=seed)
                    km.fit(X)
                    fit_time = time.perf_counter() - start_fit
                    fit_mem = (psutil.Process().memory_info().rss - mem_before_fit) / (1024 ** 2)

                    mem_before_pred = psutil.Process().memory_info().rss
                    start_pred = time.perf_counter()
                    y_pred = km.predict(X)
                    predict_time = time.perf_counter() - start_pred
                    predict_mem = (psutil.Process().memory_info().rss - mem_before_pred) / (1024 ** 2)

                    metrics_dict = _evaluate(y, y_pred, metrics)
                    _save_labels(y_pred, f"{name}_KMeans_k-{k}_seed-{seed}.labels")
                except Exception as exc:
                    y_pred = []
                    metrics_dict = {name: math.nan for name, _ in metrics}
                    fit_time = predict_time = fit_mem = predict_mem = math.nan
                    if verbose:
                        print(f"KMeans falló en {name} (k={k}, seed={seed}): {exc}")
                runtime = (0 if math.isnan(fit_time) else fit_time) + (0 if math.isnan(predict_time) else predict_time)
                record = {
                    "dataset": name,
                    "algorithm": "KMeans",
                    "params": f"n_clusters={k}",
                    "seed": seed,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    "fit_mem_mb": fit_mem,
                    "predict_mem_mb": predict_mem,
                    "runtime_sec": runtime,
                }
                record.update(metrics_dict)
                results.append(record)
                if verbose:
                    print(
                        f"KMeans {name} k={k} seed={seed} → {runtime:.4f}s",
                    )

            # DBSCAN: variar eps (rango ampliado)
            for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                try:
                    mem_before_fit = psutil.Process().memory_info().rss
                    start_fit = time.perf_counter()
                    db = DBSCAN(eps=eps, min_samples=5)
                    db.fit(X)
                    fit_time = time.perf_counter() - start_fit
                    fit_mem = (psutil.Process().memory_info().rss - mem_before_fit) / (1024 ** 2)

                    mem_before_pred = psutil.Process().memory_info().rss
                    start_pred = time.perf_counter()
                    y_pred = db.labels_
                    predict_time = time.perf_counter() - start_pred
                    predict_mem = (psutil.Process().memory_info().rss - mem_before_pred) / (1024 ** 2)

                    metrics_dict = _evaluate(y, y_pred, metrics)
                    _save_labels(y_pred, f"{name}_DBSCAN_eps-{eps}_seed-{seed}.labels")
                except Exception as exc:
                    y_pred = []
                    metrics_dict = {name: math.nan for name, _ in metrics}
                    fit_time = predict_time = fit_mem = predict_mem = math.nan
                    if verbose:
                        print(f"DBSCAN falló en {name} (eps={eps}, seed={seed}): {exc}")
                runtime = (0 if math.isnan(fit_time) else fit_time) + (0 if math.isnan(predict_time) else predict_time)
                record = {
                    "dataset": name,
                    "algorithm": "DBSCAN",
                    "params": f"eps={eps}",
                    "seed": seed,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    "fit_mem_mb": fit_mem,
                    "predict_mem_mb": predict_mem,
                    "runtime_sec": runtime,
                }
                record.update(metrics_dict)
                results.append(record)
                if verbose:
                    print(
                        f"DBSCAN {name} eps={eps} seed={seed} → {runtime:.4f}s",
                    )

    df = pd.DataFrame(results)
    out_path = out_dir / "unsupervised_results.csv"
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby(["dataset", "algorithm", "params"])
        .agg(
            {
                "fit_time_sec": ["mean", "std"],
                "predict_time_sec": ["mean", "std"],
                "fit_mem_mb": ["mean", "std"],
                "predict_mem_mb": ["mean", "std"],
                "runtime_sec": ["mean", "std"],
                "ARI": ["mean", "std"],
                "homogeneity": ["mean", "std"],
                "completeness": ["mean", "std"],
                "v_measure": ["mean", "std"],
            }
        )
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.reset_index(inplace=True)
    summary.to_csv(out_dir / "unsupervised_results_summary.csv", index=False)

    print(df.head())
    print(f"... guardado {len(df)} registros en {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación de algoritmos no supervisados")
    parser.add_argument("--runs", type=int, default=5, help="Número de repeticiones con distintas semillas")
    parser.add_argument("--verbose", action="store_true", help="Mostrar progreso")
    args = parser.parse_args()
    run(verbose=args.verbose, seeds=list(range(args.runs)))
