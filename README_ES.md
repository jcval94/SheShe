# SheShe  
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

Segmentación de bordes y exploración de hiper-fronteras basada en **máximos locales** de
la **probabilidad por clase** (clasificación) o del **valor predicho** (regresión).
Se trata de un algoritmo de clustering supervisado.

A diferencia de los métodos de clustering no supervisados que dependen solo de
la similitud entre características, SheShe aprovecha ejemplos etiquetados. Un
estimador base modela la relación entre entradas y objetivos, y el algoritmo
descubre regiones cuyas respuestas se mantienen altas para una clase o valor.
Los clusters siguen así la superficie de decisión supervisada en lugar de
métricas de distancia arbitrarias.

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
    base_2d_rays=24,               # nº de rayos en 2D (≈15°)
    direction="center_out",        # "center_out" | "outside_in"
    scan_radius_factor=3.0,
    scan_steps=24,
    smooth_window=None,             # ventana de suavizado opcional
    drop_fraction=0.5,              # caída requerida desde el pico
    stop_criteria="inflexion",     # o "percentile" para usar percentiles
    percentile_bins=20,             # número de cortes percentiles si stop_criteria="percentile"
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

### Métricas por clúster

Tras el ajuste, `ModalBoundaryClustering` guarda cada región descubierta en el
atributo `regions_`. Cada `ClusterRegion` incluye:

- `score`: eficacia del estimador sobre las muestras que pertenecen al clúster.
  Usa exactitud (accuracy) para clasificación y R² para regresión.
- `metrics`: diccionario opcional con métricas adicionales por clúster como
  precision, recall, F1, MSE o MAE.

### Descripciones en lenguaje natural con OpenAI

Instala la dependencia opcional ``openai`` (versión ``>=1``) y proporciona una
clave ya sea con el argumento ``api_key`` o mediante variables de entorno. El
intérprete busca ``OPENAI_API_KEY`` o ``OPENAI_KEY`` y, al ejecutarse en Google
Colab, también revisa ``google.colab.userdata``. Puedes fijar idioma y
temperatura por defecto en el intérprete y sobreescribirlos al llamar
``describe_cards``. Además, el parámetro ``layout`` permite sugerir un formato
específico o dejar que el modelo responda libremente.

```python
from sheshe import RegionInterpreter, OpenAIRegionInterpreter

cards = RegionInterpreter(feature_names=iris.feature_names).summarize(sh.regions_)
explicador = OpenAIRegionInterpreter(model="gpt-4o-mini", language="es", temperature=0.2)
textos = explicador.describe_cards(cards, layout="lista con viñetas", temperature=0.5)
print(textos[0])
```

---

## ¿Cómo funciona?
1. Entrena/usa un **modelo base** de sklearn (clasificación con `predict_proba` o regresión con `predict`).
2. Busca **máximos locales** por **ascenso de gradiente** con barreras en los límites del dominio.
3. Desde el máximo, traza **rayos** (direcciones) en la hiperesfera:
   - 2D: 24 rayos por defecto
   - 3D: ~26 direcciones (cobertura por *caps* esféricos con muestreo Fibonacci)
   - >3D: mezcla de unas pocas direcciones globales + **subespacios** 2D/3D
4. Sobre cada rayo, **escanea radialmente** y calcula el punto donde detenerse
   según `direction` y `stop_criteria`:
    - `center_out`: desde el centro hacia fuera
   - `outside_in`: desde el exterior hacia el centro
   Opcionalmente aplica un promedio móvil (`smooth_window`) y registra además la
   **pendiente** (df/dt) en ese punto. Con `stop_criteria="percentile"` el
   escaneo se detiene cuando el valor cae a un percentil inferior de la
   distribución (20 cortes por defecto). Si no se detecta un punto de inflexión
   o caída de percentil, se usa
   el primer valor donde la función cae por debajo de `drop_fraction` del
   máximo.
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
    base_2d_rays=24,
    random_state=0,
    drop_fraction=0.5,
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
    base_2d_rays=24,
    random_state=0,
    drop_fraction=0.5,
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
    drop_fraction=0.5,
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
    base_2d_rays=24,
    random_state=0,
    drop_fraction=0.5,
).fit(X, y)

print(sh.interpretability_summary(diab.feature_names).head())
sh.plot_pairs(X, max_pairs=3)
plt.show()
```

---

## Benchmark

La regla basada en percentiles evita el punto de inflexión y se detiene cuando
el valor cae a un percentil inferior (20 cortes por defecto). La implementación
actual en bucle es
considerablemente más rápida que la versión vectorizada anterior. En el dataset
de Iris:

```
$ PYTHONPATH=src python experiments/benchmark_stop_criteria.py
vectorized implementation: 0.0259s
loop implementation:       0.0121s
speedup: 2.14x
ModalBoundaryClustering fit with stop_criteria='inflexion': 0.1026s
ModalBoundaryClustering fit con stop_criteria='percentile': 0.1411s
```

Los valores exactos dependen de la máquina, pero el método en bucle optimizado
es mucho más rápido y arroja los mismos resultados.

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

#### `report()`

`report()` devuelve una lista con un elemento por cada subespacio entrenado,
ordenada por `weight`. Cada elemento es un diccionario con:

- `features`: tupla con los índices de las características del subespacio.
- `order`: número de características del subespacio.
- `scout_score`: puntuación asignada por `SubspaceScout`.
- `cv_score`: puntuación de validación cruzada del submodelo.
- `feat_importance`: importancia media de las características.
- `weight`: peso normalizado en el ensamble.

Ejemplo:

```python
from pprint import pprint

resumen = mse.report()
pprint([
    {k: row[k] for k in ("features", "order", "scout_score", "cv_score", "feat_importance", "weight")}
    for row in resumen[:2]
])
```

Salida:

```text
[{'cv_score': 0.9267,
  'feat_importance': 5.9886,
  'features': (3, 1),
  'order': 2,
  'scout_score': -0.2368,
  'weight': 0.4336},
{'cv_score': 0.8467,
  'feat_importance': 7.3800,
  'features': (2, 1),
  'order': 2,
  'scout_score': -0.1543,
  'weight': 0.4193}]
```

#### `plot_pairs(X, y=None, model_idx=0, max_pairs=None)`

Visualiza las superficies 2D de un submodelo concreto reutilizando las
funciones de graficado de `ModalBoundaryClustering`.

```python
feats = mse.features_[0]
mse.plot_pairs(X, y, model_idx=0, max_pairs=1)
```

#### `plot_pair_3d(X, pair, model_idx=0, class_label=None, grid_res=50, alpha_surface=0.6, engine="matplotlib")`

Muestra como superficie 3D la probabilidad (clasificación) o el valor predicho
(regresión) para un submodelo específico.

```python
feats = mse.features_[0]
mse.plot_pair_3d(X, (feats[0], feats[1]), model_idx=0, class_label=mse.classes_[0])
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
- `base_2d_rays` → controla la resolución angular en 2D (24 por defecto). 3D escala ~26; d>3 usa subespacios.
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
