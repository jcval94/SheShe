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
from typing import Iterable, Tuple

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


def run() -> None:
    """Ejecuta el experimento de comparación y guarda un CSV con resultados."""

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

    for name, (X, y) in datasets.items():
        n_classes = len(set(y))

        # SheShe: barrido sobre C
        for C in [0.1, 1.0, 10.0]:
            start = time.perf_counter()
            sh = ModalBoundaryClustering(
                base_estimator=LogisticRegression(max_iter=500, C=C),
                task="classification",
                random_state=0,
            ).fit(X, y)
            y_pred = sh.predict(X)
            runtime = time.perf_counter() - start
            record = {
                "dataset": name,
                "algorithm": "SheShe",
                "params": f"C={C}",
                "runtime_sec": runtime,
            }
            record.update(_evaluate(y, y_pred, metrics))
            results.append(record)

        # KMeans: variar n_clusters
        for k in [n_classes - 1, n_classes, n_classes + 1]:
            k = max(k, 1)
            start = time.perf_counter()
            km = KMeans(n_clusters=k, random_state=0)
            km.fit(X)
            runtime = time.perf_counter() - start
            record = {
                "dataset": name,
                "algorithm": "KMeans",
                "params": f"n_clusters={k}",
                "runtime_sec": runtime,
            }
            record.update(_evaluate(y, km.labels_, metrics))
            results.append(record)

        # DBSCAN: variar eps
        for eps in [0.3, 0.5, 0.7]:
            start = time.perf_counter()
            db = DBSCAN(eps=eps, min_samples=5)
            db.fit(X)
            runtime = time.perf_counter() - start
            record = {
                "dataset": name,
                "algorithm": "DBSCAN",
                "params": f"eps={eps}",
                "runtime_sec": runtime,
            }
            record.update(_evaluate(y, db.labels_, metrics))
            results.append(record)

    df = pd.DataFrame(results)
    out_path = Path(__file__).parent.parent / "benchmark" / "unsupervised_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.head())
    print(f"... guardado {len(df)} registros en {out_path}")


if __name__ == "__main__":
    run()

