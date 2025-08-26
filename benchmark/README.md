# Resultados de benchmarks

Este directorio contiene los resultados de pruebas de rendimiento y calidad para `SheShe` comparado con otros algoritmos clásicos.

## Calidad de clustering

Los siguientes resultados muestran el mejor `Adjusted Rand Index (ARI)` y `V-measure` alcanzados por cada algoritmo en distintos conjuntos de datos.

### ARI

| Dataset | SheShe | KMeans | DBSCAN |
| --- | --- | --- | --- |
| blobs | 0.794 | 0.767 | 0.356 |
| breast_cancer | 0.793 | 0.539 | n/a |
| iris | 0.922 | 0.716 | 0.590 |
| moons | 0.597 | 0.283 | 1.000 |
| wine | 0.799 | 0.371 | n/a |
| digits | 0.998 | 0.559 | 0.002 |
| california_housing | n/a | n/a | n/a |
| circles | 0.000 | -0.003 | 0.993 |

### V-measure

| Dataset | SheShe | KMeans | DBSCAN |
| --- | --- | --- | --- |
| blobs | 0.734 | 0.702 | 0.430 |
| breast_cancer | 0.683 | 0.471 | n/a |
| iris | 0.914 | 0.742 | 0.641 |
| moons | 0.490 | 0.338 | 1.000 |
| wine | 0.780 | 0.429 | n/a |
| digits | 0.997 | 0.695 | 0.047 |
| california_housing | n/a | n/a | n/a |
| circles | 0.000 | 0.000 | 0.986 |

## Rendimiento en conjuntos de datos grandes

Comparación entre la implementación base y una versión optimizada.

| n_samples | n_features | Speedup en `fit` | Speedup en `predict` |
| --- | --- | --- | --- |
| 1000 | 10 | 1.12× | 0.70× |
| 10000 | 10 | 1.03× | 0.98× |

## Prueba A/B del criterio de percentil

| n_points | Speedup |
| --- | --- |
| 100 | 0.85× |
| 1000 | 0.35× |
| 10000 | 0.17× |

## Criterios de parada

| Tamaño | Dirección | Implementación | Speedup |
| --- | --- | --- | --- |
| 500 | center_out | vectorized | 1.00× |
| 500 | center_out | loop | 2.73× |
| 500 | outside_in | vectorized | 1.00× |
| 500 | outside_in | loop | 3.64× |
| 150 | – | fit_inflexion | 1.00× |
| 150 | – | fit_percentile | 0.82× |

