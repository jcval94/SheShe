# Experimentos para el manuscrito

**📚 Documentación:** https://jcval94.github.io/SheShe/

Este directorio contiene scripts reproducibles que generan las tablas y figuras
utilizadas en el artículo de SheShe.

Las métricas reflejan el nuevo modo de rayos `grad`, que reemplaza al anterior
`grid` con mejoras significativas de rendimiento (ver
[benchmark/README.md](../benchmark/README.md)).

### Datasets

| Nombre | Enlace | Licencia | Citación |
| --- | --- | --- | --- |
| Iris | [UCI](https://archive.ics.uci.edu/ml/datasets/iris) | CC BY 4.0 | R. A. Fisher, 1936 |
| Wine | [UCI](https://archive.ics.uci.edu/ml/datasets/wine) | CC BY 4.0 | Forina et al., 1991 |
| Breast Cancer Wisconsin (Diagnostic) | [UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) | CC BY 4.0 | Street et al., 1993 |
| Digits | [UCI](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits) | CC BY 4.0 | C. Kaynak, 1995 |
| California Housing | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | CC BY 4.0 | Pace & Barry, 1997 |
| Moons (sintético) | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) | BSD-3-Clause | Pedregosa et al., 2011 |
| Blobs (sintético) | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) | BSD-3-Clause | Pedregosa et al., 2011 |
| Circles (sintético) | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) | BSD-3-Clause | Pedregosa et al., 2011 |
| Synthetic24D_4c / make_classification | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) | BSD-3-Clause | Pedregosa et al., 2011 |

## Comparación no supervisada
- **Script:** `compare_unsupervised.py`
- **Datasets:** Iris, Wine, Breast Cancer, Digits, California Housing, Moons, Blobs y Circles.
- **Algoritmos:** SheShe, KMeans y DBSCAN.
- **Métricas:** ARI, homogeneity, completeness, v_measure y tiempo de ejecución.
- **Preprocesamiento:** sin normalización ni división; cada algoritmo se entrena y evalúa sobre el conjunto completo.
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
- **Ablation:** variación de `base_2d_rays`, `direction`, `jaccard_threshold`,
  `drop_fraction` y `smooth_window` sobre el conjunto de Iris.
- **Sensibilidad:** estudio de la dimensión (`n_features`) y robustez a ruido
  gaussiano.
- **Preprocesamiento:**
  - Comparación supervisada: `StratifiedKFold` de 5 particiones sin escalado.
  - Ablation y sensibilidad: `train_test_split` (70/30) sin normalización.
- **Ejecución:**
  ```bash
  python experiments/paper_experiments.py --runs 5
  ```
  El argumento `--runs` fija el número de semillas empleadas. Se guardan las
  ejecuciones individuales y un resumen con medias y desviaciones estándar para
  cada experimento (`*_summary.csv`). Además, se generan las figuras (`*.png`)
  dentro de `benchmark/`.

## A/B testing de *clustering*
- **Script:** `ab_test_clustering.py`
- **Datasets:** Iris, Wine, Breast Cancer, Digits (64D), California Housing,
  Circles y un conjunto sintético multiclase.
- **Modelos:** ModalBoundaryClustering y cuatro algoritmos no supervisados
  (KMeans, AgglomerativeClustering, SpectralClustering y GaussianMixture).
- **Métricas:** accuracy (Hungarian), F1 macro/ponderado, purity, BCubed,
  ARI, AMI, NMI, homogeneity, completeness, V-measure, Fowlkes–Mallows y
  silhouette.
- **Preprocesamiento:** `StandardScaler` previo a los modelos no supervisados; SheShe usa las características crudas y no se realiza partición train/test.
- **Ejecución:**
  ```bash
  PYTHONPATH=src python experiments/ab_test_clustering.py
  ```
  Genera un archivo `ab_results_clustering.csv` con los resultados y muestra en
  consola un ranking por dataset.

## Benchmark de criterios de parada
- **Script:** `benchmark_stop_criteria.py`
- **Objetivo:** comparar la antigua implementación vectorizada de
  `find_percentile_drop` con la versión optimizada en bucle y medir el tiempo de
  ajuste de `ModalBoundaryClustering` usando `stop_criteria` `"inflexion"` y
  `"percentile"`.
- **Preprocesamiento:** utiliza el dataset completo de Iris sin escalado ni partición.
- **Ejecución:**
  ```bash
  PYTHONPATH=src python experiments/benchmark_stop_criteria.py \\
    --reps 5 --warmup 2 --sizes 500 1000 --directions center_out outside_in
  ```
  Los parámetros `--reps`, `--warmup`, `--sizes` y `--directions` permiten
  explorar distintas configuraciones del benchmark.

## Sugerencias de experimentos adicionales
- **Variación de parámetros:** explorar el impacto de `auto_rays_by_dim`,
  `drop_fraction`, `smooth_window` y distintos `stop_criteria` sobre la
  calidad y el tiempo de ajuste.
- **Nuevos algoritmos base:** incluir métodos como OPTICS, MeanShift o Birch
  para ampliar la comparación con ModalBoundaryClustering.
- **Métricas internas:** evaluar índices como Calinski–Harabasz y
  Davies–Bouldin, además de las métricas externas ya consideradas.
- **Efectos de normalización y paralelización:** medir el desempeño con
  diferentes escalados de características y valores de `n_jobs`.
- **Conjuntos de datos desafiantes:** probar datasets de mayor dimensionalidad
  o con ruido extremo para estudiar la robustez del algoritmo.

## Imágenes

Las imágenes utilizadas en este directorio se almacenan en [`../images/`](../images/). Actualmente la carpeta está vacía y se completará en el futuro. Para evitar que estos recursos binarios inflen el repositorio, se manejan con [Git LFS](https://git-lfs.com).
