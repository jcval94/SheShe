# SheShe  
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

Segmentación de bordes y exploración de hiper-fronteras basada en **máximos locales** de
la **probabilidad por clase** (clasificación) o del **valor predicho** (regresión).

---

## Instalación

Requiere **Python >=3.9** y se recomienda el uso de un entorno virtual.
Instala la versión publicada en
[PyPI](https://pypi.org/project/sheshe/):

```bash
pip install sheshe
```

Dependencias base: `numpy`, `pandas`, `scikit-learn>=1.1`, `matplotlib`

Para un entorno de desarrollo con pruebas:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

---

## API rápida

La librería expone cuatro objetos principales:

- `ModalBoundaryClustering`
- `ClusterRegion` – dataclass con la información de cada región
- `SubspaceScout`
- `ModalScoutEnsemble`

```python
from sheshe import (
    ModalBoundaryClustering,
    SubspaceScout,
    ModalScoutEnsemble,
    ClusterRegion,
)

# clasificación
clf = ModalBoundaryClustering(
    base_estimator=None,           # por defecto LogisticRegression
    task="classification",         # "classification" | "regression"
    base_2d_rays=8,                # nº de rayos en 2D (≈45°)
    direction="center_out",        # "center_out" | "outside_in"
    scan_radius_factor=3.0,
    scan_steps=64,
    smooth_window=None,             # ventana de suavizado opcional
    random_state=0
)

# regresión (ejemplo)
reg = ModalBoundaryClustering(task="regression")
```

### Métodos
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`  → clasificación: probas por clase; regresión: valor normalizado [0,1]
- `interpretability_summary(feature_names=None)` → DataFrame con:
  - `Tipo`: "centroide" | "inflexion_point"
  - `Distancia`: radio desde el centro al punto de inflexión
  - `Categoria`: clase (o "NA" en regresión)
  - `pendiente`: df/dt en el punto de inflexión
  - `valor_real` / `valor_norm`
  - `coord_0..coord_{d-1}` o nombres de features
- `plot_pairs(X, y=None, max_pairs=None)` → gráficos 2D para todas las combinaciones de pares
- `save(filepath)` → guarda el modelo mediante `joblib`
- `ModalBoundaryClustering.load(filepath)` → carga una instancia guardada

---

## ¿Cómo funciona?
1. Entrena/usa un **modelo base** de sklearn (clasificación con `predict_proba` o regresión con `predict`).
2. Busca **máximos locales** por **ascenso de gradiente** con barreras en los límites del dominio.
3. Desde el máximo, traza **rayos** (direcciones) en la hiperesfera:
   - 2D: 8 rayos por defecto
   - 3D: ~26 direcciones (cobertura por *caps* esféricos con muestreo Fibonacci)
   - >3D: mezcla de unas pocas direcciones globales + **subespacios** 2D/3D
4. Sobre cada rayo, **escanea radialmente** y calcula el **primer punto de inflexión** según `direction`:
    - `center_out`: desde el centro hacia fuera
    - `outside_in`: desde el exterior hacia el centro
   Opcionalmente aplica un promedio móvil (`smooth_window`) y registra además la
   **pendiente** (df/dt) en ese punto.
5. Conecta los puntos de inflexión para formar la **frontera** de la región de alta probabilidad/valor.

---

## Ejemplos
### Clasificación — Iris
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sheshe import ModalBoundaryClustering

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering(
    base_estimator=LogisticRegression(max_iter=1000),
    task="classification",
    base_2d_rays=8,
    random_state=0,
).fit(X, y)

print(sh.interpretability_summary(iris.feature_names).head())
sh.plot_pairs(X, y, max_pairs=3)   # genera las gráficas
plt.show()
```

### Clasificación con modelo ya entrenado
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sheshe import ModalBoundaryClustering

wine = load_wine()
X, y = wine.data, wine.target

# Entrena un modelo de forma independiente
base_model = RandomForestClassifier(n_estimators=200, random_state=0)
base_model.fit(X, y)

# Usa SheShe con ese modelo ya ajustado
sh = ModalBoundaryClustering(
    base_estimator=base_model,
    task="classification",
    base_2d_rays=8,
    random_state=0,
).fit(X, y)

sh.plot_pairs(X, y, max_pairs=2)
plt.show()
```

### Clasificación — blobs sintéticos con parámetros personalizados
```python
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sheshe import ModalBoundaryClustering

X, y = make_blobs(n_samples=400, centers=5, cluster_std=1.8, random_state=0)

sh = ModalBoundaryClustering(
    base_estimator=LogisticRegression(max_iter=200),
    task="classification",
    base_2d_rays=16,
    scan_steps=32,
    n_max_seeds=3,
    direction="outside_in",
    random_state=0,
).fit(X, y)

print(sh.predict(X[:5]))
```

### Regresión — Diabetes
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sheshe import ModalBoundaryClustering

diab = load_diabetes()
X, y = diab.data, diab.target

sh = ModalBoundaryClustering(
    base_estimator=GradientBoostingRegressor(random_state=0),
    task="regression",
    base_2d_rays=8,
    random_state=0,
).fit(X, y)

print(sh.interpretability_summary(diab.feature_names).head())
sh.plot_pairs(X, max_pairs=3)
plt.show()
```

### Guardado de gráficas
```python
from pathlib import Path
import matplotlib.pyplot as plt

# tras llamar a ``sh.plot_pairs(...)``
out_dir = Path("imagenes")
out_dir.mkdir(exist_ok=True)
for i, fig_num in enumerate(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(out_dir / f"pair_{i}.png")
    plt.close(fig_num)
```

### Guardar y cargar modelo
```python
from pathlib import Path
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering().fit(X, y)
ruta = Path("sheshe_model.joblib")
sh.save(ruta)
sh2 = ModalBoundaryClustering.load(ruta)
print((sh.predict(X) == sh2.predict(X)).all())
```

Para ejemplos más completos, consulta la carpeta `examples/`.

## SubspaceScout

`SubspaceScout` descubre subespacios de características (pares, tríos, …) antes
de ejecutar SheShe. Puede trabajar solo con información mutua o aprovechar
modelos opcionales como LightGBM+SHAP o EBM para puntuar interacciones.

```python
from sheshe import SubspaceScout

scout = SubspaceScout(
    # model_method='lightgbm',    # por defecto usa MI; LightGBM y SHAP son opcionales
    max_order=4,                # pares, tríos y cuartetos
    top_m=50,                   # recorta a las 50 features más informativas
    base_pairs_limit=12,        # semillas para órdenes >=3
    beam_width=10,              # combos que sobreviven por orden
    extend_candidate_pool=16,   # features aleatorias por padre
    branch_per_parent=4,        # extensiones por padre
    marginal_gain_min=1e-3,     # ganancia mínima aceptada
    max_eval_per_order=150,     # tope de evaluaciones por orden
    sample_size=4096,           # muestreo para el scout
    time_budget_s=None,         # p.ej., 15.0 para 15 segundos
    task='classification',
    random_state=0,
)
subspaces = scout.fit(X, y)
```

## ModalScoutEnsemble

`ModalScoutEnsemble` entrena varios modelos `ModalBoundaryClustering` en los mejores subespacios devueltos por `SubspaceScout` y combina sus predicciones.

```python
from sheshe import ModalScoutEnsemble
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target

mse = ModalScoutEnsemble(
    base_estimator=LogisticRegression(max_iter=200),
    task="classification",
    random_state=0,
    scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
    cv=2,
)
mse.fit(X, y)
print(mse.predict(X[:5]))
```

### Experimentos y benchmark

Los experimentos de comparación con algoritmos **no supervisados** se encuentran
en la carpeta [`experiments/`](experiments/). El script
[`compare_unsupervised.py`](experiments/compare_unsupervised.py) evalúa cinco
conjuntos de datos distintos, explora parámetros de **SheShe**, **KMeans** y
**DBSCAN**, y almacena cuatro métricas (`ARI`, `homogeneity`, `completeness`,
`v_measure`) junto con el tiempo de ejecución (`runtime_sec`).

```bash
python experiments/compare_unsupervised.py
cat benchmark/unsupervised_results.csv | head
```

se generan los resultados dentro de `benchmark/`.

Para los experimentos del manuscrito se incluye además
[`paper_experiments.py`](experiments/paper_experiments.py), que compara con
algoritmos **supervisados**, realiza estudios de *ablation* sobre
`base_2d_rays` y `direction`, y analiza la sensibilidad a la dimensión y al
ruido gaussiano. Ejecutando el script se obtienen tablas (`*.csv`) y figuras
(`*.png`) reproducibles en `benchmark/`:

```bash
python experiments/paper_experiments.py
```

---

## Parámetros clave
- `base_2d_rays` → controla la resolución angular en 2D (8 por defecto). 3D escala ~26; d>3 usa subespacios.
- `direction` → "center_out" | "outside_in" para localizar el punto de inflexión.
- `scan_radius_factor`, `scan_steps` → tamaño y resolución del escaneo radial.
- `grad_*` → hiperparámetros del ascenso (tasa, iteraciones, tolerancias).
- `max_subspaces` → nº máx. de subespacios considerados cuando d>3.
- `density_alpha` / `density_k` → penalización opcional de densidad calculada
  con una búsqueda k-NN HNSW (usando `hnswlib`) para mantener los centros
  dentro de la nube de datos. El valor normalizado se multiplica por
  `densidad(x)**density_alpha`; `density_alpha=0` desactiva la penalización.

---

## Limitaciones
- Depende de la **superficie** producida por el modelo base (puede ser rugosa en RF).
- En alta dimensión, la frontera es una **aproximación** (subespacios).
- Encuentra **máximos locales** (no garantiza el global), mitigado con múltiples *seeds*.

---

## Contribuir

Las mejoras son bienvenidas. Para proponer cambios:

1. Haz un *fork* del repositorio y crea una rama descriptiva.
2. Instala las dependencias de desarrollo y ejecuta los tests:

   ```bash
   pip install -e ".[dev]"
   PYTHONPATH=src pytest -q
   ```
3. Envía un *pull request* con una descripción clara del cambio.

---

## Licencia
MIT
