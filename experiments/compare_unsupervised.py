"""Compara SheShe con algoritmos no supervisados en varios conjuntos de datos.

Genera métricas de *clustering* (ARI, homogeneidad, completitud y V-measure)
para cinco dataframes distintos y diferentes configuraciones de parámetros.
Cada ejecución registra además el **tiempo de ejecución** de la fase de
entrenamiento/predicción, almacenado en ``runtime_sec``.
Los resultados se guardan en ``benchmark/unsupervised_results.csv``.
"""
from pathlib import Path
import math
import time
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_blobs,
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

from sheshe import ModalBoundaryClustering


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
    """

    datasets = {
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "moons": make_moons(n_samples=300, noise=0.05, random_state=0),
        "blobs": make_blobs(n_samples=300, centers=3, random_state=0),
    }

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

    for name, (X, y) in datasets.items():
        n_classes = len(set(y))

        # SheShe: barrido sobre C
        for C in [0.1, 1.0, 10.0]:
            start = time.perf_counter()
            try:
                sh = ModalBoundaryClustering(
                    base_estimator=LogisticRegression(max_iter=500, C=C),
                    task="classification",
                    random_state=0,
                ).fit(X, y)
                y_pred = sh.predict(X)
                metrics_dict = _evaluate(y, y_pred, metrics)
                _save_labels(y_pred, f"{name}_SheShe_C-{C}.labels")
            except Exception as exc:
                y_pred = []
                metrics_dict = {name: math.nan for name, _ in metrics}
                if verbose:
                    print(f"SheShe falló en {name} (C={C}): {exc}")
            runtime = time.perf_counter() - start
            record = {
                "dataset": name,
                "algorithm": "SheShe",
                "params": f"C={C}",
                "runtime_sec": runtime,
            }
            record.update(metrics_dict)
            results.append(record)
            if verbose:
                print(
                    f"SheShe {name} C={C} → {runtime:.4f}s"
                )

        # KMeans: variar n_clusters
        for k in [n_classes - 1, n_classes, n_classes + 1]:
            k = max(k, 1)
            start = time.perf_counter()
            try:
                km = KMeans(n_clusters=k, random_state=0)
                km.fit(X)
                y_pred = km.labels_
                metrics_dict = _evaluate(y, y_pred, metrics)
                _save_labels(y_pred, f"{name}_KMeans_k-{k}.labels")
            except Exception as exc:
                y_pred = []
                metrics_dict = {name: math.nan for name, _ in metrics}
                if verbose:
                    print(f"KMeans falló en {name} (k={k}): {exc}")
            runtime = time.perf_counter() - start
            record = {
                "dataset": name,
                "algorithm": "KMeans",
                "params": f"n_clusters={k}",
                "runtime_sec": runtime,
            }
            record.update(metrics_dict)
            results.append(record)
            if verbose:
                print(
                    f"KMeans {name} k={k} → {runtime:.4f}s"
                )

        # DBSCAN: variar eps
        for eps in [0.3, 0.5, 0.7]:
            start = time.perf_counter()
            try:
                db = DBSCAN(eps=eps, min_samples=5)
                db.fit(X)
                y_pred = db.labels_
                metrics_dict = _evaluate(y, y_pred, metrics)
                _save_labels(y_pred, f"{name}_DBSCAN_eps-{eps}.labels")
            except Exception as exc:
                y_pred = []
                metrics_dict = {name: math.nan for name, _ in metrics}
                if verbose:
                    print(f"DBSCAN falló en {name} (eps={eps}): {exc}")
            runtime = time.perf_counter() - start
            record = {
                "dataset": name,
                "algorithm": "DBSCAN",
                "params": f"eps={eps}",
                "runtime_sec": runtime,
            }
            record.update(metrics_dict)
            results.append(record)
            if verbose:
                print(
                    f"DBSCAN {name} eps={eps} → {runtime:.4f}s"
                )

    df = pd.DataFrame(results)
    out_path = out_dir / "unsupervised_results.csv"
    df.to_csv(out_path, index=False)
    print(df.head())
    print(f"... guardado {len(df)} registros en {out_path}")


if __name__ == "__main__":
    run()

