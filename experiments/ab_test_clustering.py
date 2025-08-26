# -*- coding: utf-8 -*-
"""
AB Testing: ModalBoundaryClustering (SheShe) vs 4 modelos no supervisados

- Datasets: Iris, Wine, Breast Cancer, Digits(64D), California Housing,
             Circles y un conjunto sintético multiclase
- Métricas: accuracy (Hungarian), macro/weighted F1, purity, BCubed,
            ARI/AMI/NMI, homogeneity/completeness/V-measure,
            Fowlkes–Mallows, silhouette
- Resultados: CSV + ranking en consola
"""

from __future__ import annotations

import time
import warnings
import platform
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any

import psutil
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# ---- Modelos no supervisados ----
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

# ---- Tu modelo (SheShe) ----
try:
    from sheshe.sheshe import ModalBoundaryClustering
except Exception:
    # Fallback si el módulo está en otro namespace
    from sheshe import ModalBoundaryClustering

# ---- Datasets ----
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_california_housing,
    make_circles,
    make_classification,
)

warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(suppress=True, linewidth=120)

PLATFORM_INFO = {
    "cpu": platform.processor(),
    "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
}

# ==========================
# Utilidades de evaluación
# ==========================

def _hungarian_best_map(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Mapea y_pred -> y_true optimizando exactitud con algoritmo Húngaro sobre la matriz de contingencia.
    Devuelve y_pred_mapeado y el diccionario de mapeo {cluster -> clase}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    # Asegura índices densos para la matriz
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    cm = contingency_matrix(
        [class_to_idx[v] for v in y_true],
        [cluster_to_idx[v] for v in y_pred],
        sparse=False,
    )
    # Maximizar exactitud => minimizamos costo negativo
    cost = cm.max() - cm
    r, c = linear_sum_assignment(cost)
    mapping = {clusters[cj]: classes[rj] for rj, cj in zip(r, c)}
    y_mapped = np.array([mapping[v] if v in mapping else v for v in y_pred])
    return y_mapped, mapping

def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_c max_l n_{c,l}
    Se calcula vía matriz de contingencia.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = contingency_matrix(y_true, y_pred, sparse=False)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

def _bcubed_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    BCubed P/R/F1 usando fórmula de matriz de contingencia (O(C*L)).
    Referencia:
      Precision_B3 = (1/N) * sum_{c,l} n_{c,l}^2 / n_{·,l}
      Recall_B3    = (1/N) * sum_{c,l} n_{c,l}^2 / n_{c,·}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = contingency_matrix(y_true, y_pred, sparse=False).astype(float)  # shape: [n_classes, n_clusters]
    N = cm.sum()
    cluster_sizes = cm.sum(axis=0) + 1e-12  # n_{·,l}
    class_sizes = cm.sum(axis=1) + 1e-12  # n_{c,·}
    sq = cm ** 2
    precision = (sq / cluster_sizes).sum() / N
    recall = (sq.T / class_sizes).sum() / N
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)

def _silhouette_safe(X: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Silhouette con protección ante clusters únicos o un solo punto por cluster.
    """
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_pred)) < 2:
        return np.nan
    # Silhouette suele ser más estable con X estandarizado
    Xs = StandardScaler().fit_transform(X)
    try:
        return float(silhouette_score(Xs, y_pred, metric="euclidean"))
    except Exception:
        return np.nan

def evaluar_labels(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """
    Calcula un set robusto de métricas externas e interna (silhouette).
    Para accuracy/F1 usa mapeo Húngaro de clusters->clases.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Mapeo óptimo para métricas tipo "clasificación"
    y_map, mapping = _hungarian_best_map(y_true, y_pred)

    metrics: Dict[str, float] = {}
    # Clasificación
    metrics["accuracy"] = float(np.mean(y_true == y_map))
    metrics["macro_f1"] = float(f1_score(y_true, y_map, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true, y_map, average="weighted"))

    # Clustering (externas)
    metrics["purity"] = _purity(y_true, y_pred)
    bP, bR, bF1 = _bcubed_scores(y_true, y_pred)
    metrics["bcubed_precision"] = bP
    metrics["bcubed_recall"] = bR
    metrics["bcubed_f1"] = bF1

    metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    metrics["ami"] = float(adjusted_mutual_info_score(y_true, y_pred))
    metrics["homogeneity"] = float(homogeneity_score(y_true, y_pred))
    metrics["completeness"] = float(completeness_score(y_true, y_pred))
    metrics["v_measure"] = float(v_measure_score(y_true, y_pred))
    metrics["fowlkes_mallows"] = float(fowlkes_mallows_score(y_true, y_pred))

    # Interna
    metrics["silhouette"] = _silhouette_safe(X, y_pred)

    return metrics

# ==========================
# Preparación de modelos
# ==========================

@dataclass
class Modelo:
    nombre: str
    tipo: str  # "supervisado" | "no_supervisado"
    creador: Callable[..., Any]

def preparar_modelos(n_clusters: int, random_state: int = 42) -> List[Modelo]:
    """
    Devuelve SheShe + 4 no supervisados con hiperparámetros razonables.
    Nota: para los no supervisados, n_clusters = nº de clases del dataset (conocido).
    """
    modelos: List[Modelo] = []

    # 1) SheShe (supervisado) - tu modelo
    def crear_sheshe():
        # Defaults optimizados (ya incorporados en tu lib); se puede ajustar base_2d_rays para d altos.
        return ModalBoundaryClustering(
            task="classification",
            random_state=random_state,
            # Si quieres forzar heurística por dimensión en tu lib: auto_rays_by_dim=True
        )

    modelos.append(Modelo("SheShe_ModalBoundaryClustering", "supervisado", crear_sheshe))

    # 2) KMeans
    def crear_kmeans():
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)

    modelos.append(Modelo("KMeans", "no_supervisado", crear_kmeans))

    # 3) GaussianMixture (usamos argmax de responsabilidades)
    def crear_gmm():
        return GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state, n_init=1)

    modelos.append(Modelo("GaussianMixture", "no_supervisado", crear_gmm))

    # 4) Agglomerative (Ward)
    def crear_aggl():
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")

    modelos.append(Modelo("Agglomerative", "no_supervisado", crear_aggl))

    # 5) Spectral
    def crear_spec():
        # defaults razonables; k-means en el embedding
        return SpectralClustering(n_clusters=n_clusters, affinity="rbf", assign_labels="kmeans", random_state=random_state, n_init=10)

    modelos.append(Modelo("SpectralClustering", "no_supervisado", crear_spec))

    return modelos

# ==========================
# Datasets
# ==========================

def cargar_datasets() -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Devuelve lista: (nombre, X, y)
    """
    data: List[Tuple[str, np.ndarray, np.ndarray]] = []
    X, y = load_iris(return_X_y=True)
    data.append(("Iris", X, y))
    X, y = load_wine(return_X_y=True)
    data.append(("Wine", X, y))
    X, y = load_breast_cancer(return_X_y=True)
    data.append(("BreastCancer30D", X, y))
    X, y = load_digits(return_X_y=True)
    data.append(("Digits64D", X, y))
    try:
        X, y = fetch_california_housing(return_X_y=True)
        rng = np.random.default_rng(123)
        idx = rng.choice(len(y), size=1200, replace=False)
        X, y = X[idx], y[idx]
        bins = np.quantile(y, [0.25, 0.5, 0.75])
        y = np.digitize(y, bins)
        data.append(("CaliforniaHousing", X, y))
    except Exception as exc:
        print(f"[cargar_datasets] California Housing no disponible: {exc}")
    X, y = make_circles(n_samples=1200, noise=0.05, factor=0.5, random_state=123)
    data.append(("Circles", X, y))
    # Sintético multiclase (más “realista” que blobs)
    X, y = make_classification(
        n_samples=1200,
        n_features=24,
        n_informative=14,
        n_redundant=6,
        n_repeated=0,
        n_classes=4,
        class_sep=1.2,
        flip_y=0.02,
        random_state=123,
    )
    data.append(("Synthetic24D_4c", X, y))
    return data

# ==========================
# Loop de benchmark
# ==========================

def fit_predict_unsupervised(model, X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Estandariza X y corre el modelo no supervisado devolviendo labels.
    """
    Xs = StandardScaler().fit_transform(X)
    # Si el modelo tiene fit_predict, úsalo
    if hasattr(model, "fit_predict"):
        return model.fit_predict(Xs)
    # Si no, fit + labels (predict o labels_)
    model.fit(Xs)
    if hasattr(model, "predict"):
        return model.predict(Xs)
    if hasattr(model, "labels_"):
        return model.labels_
    raise RuntimeError("El modelo no provee fit_predict/predict/labels_")

def fit_predict_sheshe(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Usa fit_predict del modelo SheShe (supervisado) para devolver labels de clúster.
    """
    if hasattr(model, "fit_predict"):
        return model.fit_predict(X, y)
    model.fit(X, y)
    if hasattr(model, "predict"):
        # predict devuelve "label cluster" por muestra (no la clase)
        return model.predict(X)
    raise RuntimeError("SheShe no expuso fit_predict/predict")

def benchmark_dataset(
    nombre_dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 3,
    random_state_base: int = 42,
) -> pd.DataFrame:
    """
    Corre todos los modelos sobre un dataset con múltiples seeds y devuelve un DataFrame de resultados promediados.
    """
    K = int(np.unique(y).size)
    registros: List[Dict[str, Any]] = []

    for rep in range(n_repeats):
        seed = random_state_base + rep
        modelos = preparar_modelos(K, random_state=seed)
        for m in modelos:
            t0 = time.perf_counter()
            try:
                if m.tipo == "supervisado":
                    modelo = m.creador()
                    y_pred = fit_predict_sheshe(modelo, X, y)  # usa y
                else:
                    modelo = m.creador()
                    y_pred = fit_predict_unsupervised(modelo, X, random_state=seed)
            except Exception as e:
                # En caso de fallo, registramos NaN
                elapsed = time.perf_counter() - t0
                registros.append(
                    {
                        "dataset": nombre_dataset,
                        "modelo": m.nombre,
                        "tipo": m.tipo,
                        "runtime_s": elapsed,
                        "error": str(e),
                        **{
                            k: np.nan
                            for k in [
                                "accuracy",
                                "macro_f1",
                                "weighted_f1",
                                "purity",
                                "bcubed_precision",
                                "bcubed_recall",
                                "bcubed_f1",
                                "ari",
                                "ami",
                                "nmi",
                                "homogeneity",
                                "completeness",
                                "v_measure",
                                "fowlkes_mallows",
                                "silhouette",
                            ]
                        },
                    }
                )
                continue

            elapsed = time.perf_counter() - t0
            mets = evaluar_labels(y, y_pred, X)
            row = {
                "dataset": nombre_dataset,
                "modelo": m.nombre,
                "tipo": m.tipo,
                "runtime_s": elapsed,
                "error": "",
            }
            row.update(mets)
            registros.append(row)

    # Promedio por modelo/dataset
    df = pd.DataFrame(registros)
    agg = {col: "mean" for col in df.columns if col not in ("dataset", "modelo", "tipo", "error")}
    df_avg = df.groupby(["dataset", "modelo", "tipo"], as_index=False).agg(agg)
    # Cuenta de fallos
    err_count = (
        df.groupby(["dataset", "modelo"])["error"].apply(lambda s: (s != "").sum()).reset_index(name="n_fallos")
    )
    df_avg = df_avg.merge(err_count, on=["dataset", "modelo"], how="left")
    return df_avg

def main():
    datasets = cargar_datasets()
    todo: List[pd.DataFrame] = []
    for nombre, X, y in datasets:
        print(f"\n>>> Benchmark en dataset: {nombre} (n={len(X)}, d={X.shape[1]}, clases={np.unique(y).size})")
        df_res = benchmark_dataset(nombre, X, y, n_repeats=3, random_state_base=42)
        # Ranking por macro_f1 y nmi como guías principales
        df_rank = df_res.sort_values(["macro_f1", "nmi"], ascending=False)
        print(
            df_rank[
                [
                    "dataset",
                    "modelo",
                    "tipo",
                    "macro_f1",
                    "nmi",
                    "purity",
                    "bcubed_f1",
                    "ari",
                    "runtime_s",
                    "n_fallos",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
        todo.append(df_res)

    df_final = pd.concat(todo, ignore_index=True)
    df_final = df_final.sort_values(["dataset", "macro_f1", "nmi"], ascending=[True, False, False])

    # Escribir CSV con metadatos de plataforma como encabezado comentado
    with open("ab_results_clustering.csv", "w", encoding="utf-8") as f:
        for k, v in PLATFORM_INFO.items():
            f.write(f"# {k}: {v}\n")
        df_final.to_csv(f, index=False)

    # Guardar los metadatos también en un JSON acompañante
    with open("ab_results_clustering_meta.json", "w") as f:
        json.dump(PLATFORM_INFO, f, indent=2)

    print("\n✅ Resultados guardados en: ab_results_clustering.csv (incluye metadatos de plataforma)")
    print("📄 Metadatos de plataforma guardados en: ab_results_clustering_meta.json")
    # Top-3 por dataset (macro_f1)
    print("\n🏁 Top-3 por dataset (macro_f1):")
    for nombre in sorted(df_final["dataset"].unique()):
        top3 = df_final[df_final["dataset"] == nombre].nlargest(
            3, "macro_f1"
        )[["modelo", "tipo", "macro_f1", "nmi", "runtime_s"]]
        print(f"\n{nombre}:\n{top3.round(4).to_string(index=False)}")
    print("\n🔧 Configuración de plataforma:")
    for k, v in PLATFORM_INFO.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
