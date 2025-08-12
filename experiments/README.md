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
  python experiments/compare_unsupervised.py --runs 5
  ```
  El parámetro `--runs` controla la cantidad de repeticiones con distintas
  semillas. Los resultados individuales (`unsupervised_results.csv`) y el
  resumen estadístico (`unsupervised_results_summary.csv`) se almacenan en
  `benchmark/` junto con las etiquetas predichas (`*.labels`).

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
  python experiments/paper_experiments.py --runs 5
  ```
  El argumento `--runs` fija el número de semillas empleadas. Se guardan las
  ejecuciones individuales y un resumen con medias y desviaciones estándar para
  cada experimento (`*_summary.csv`). Además, se generan las figuras (`*.png`)
  dentro de `benchmark/`.
