# Experimentos para el manuscrito

Este directorio contiene scripts reproducibles que generan las tablas y figuras
utilizadas en el artículo de SheShe.

## Comparación no supervisada
- **Script:** `compare_unsupervised.py`
- **Datasets:** Iris, Wine, Breast Cancer, Moons y Blobs.
- **Algoritmos:** SheShe, KMeans y DBSCAN.
- **Métricas:** ARI, homogeneity, completeness, v_measure y tiempo de ejecución.
- **Ejecución:**
  ```bash
  python experiments/compare_unsupervised.py
  ```
  Los resultados se almacenan en `benchmark/unsupervised_results.csv` junto con
  las etiquetas predichas (`*.labels`).

## Experimentos para el paper
- **Script:** `paper_experiments.py`
- **Comparación supervisada:** Logistic Regression, KNN, Random Forest y SheShe
  (métrica: *accuracy*).
- **Ablation:** variación de `base_2d_rays` y `direction` sobre el conjunto de
  Iris.
- **Sensibilidad:** estudio de la dimensión (`n_features`) y robustez a ruido
  gaussiano.
- **Ejecución:**
  ```bash
  python experiments/paper_experiments.py
  ```
  Genera tablas (`*.csv`) y figuras (`*.png`) dentro de `benchmark/`.
