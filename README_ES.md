# SheShe  
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

**SheShe** convierte cualquier modelo probabilístico en un explorador guiado de
su paisaje de decisión. Siguiendo los **máximos locales** de la **probabilidad
por clase** (clasificación) o del **valor predicho** (regresión), delimita
regiones nítidas e interpretables que se apoyan en la frontera supervisada del
problema. En lugar de agrupar por distancia, aprende de ejemplos etiquetados y
traza los clúesteres directamente sobre la superficie de decisión.

## Destacados

- Clustering supervisado impulsado por las propias probabilidades o predicciones del modelo.
- Soporte unificado para tareas de clasificación y regresión.
- Exploración de subespacios con `SubspaceScout` y ensamblados mediante `ModalScoutEnsemble`.
- Extracción de reglas interpretables a través de `RegionInterpreter`.
- Utilidades de graficado integradas para visualizaciones 2D y 3D.

*Figura de resumen de características omitida (no se permiten archivos binarios).*
Genera un gráfico equivalente en tu entorno:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
sh.plot_pairs(X, y, max_pairs=1)
plt.show()
```

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

La librería expone siete objetos principales:

- `ModalBoundaryClustering`
- `ClusterRegion` – dataclass con la información de cada región
- `SubspaceScout`
- `ModalScoutEnsemble`
- `RegionInterpreter` – convierte `ClusterRegion` en reglas interpretables
- `ShuShu` – búsqueda de máximos por gradiente
- `CheChe` – calcula fronteras 2D sobre pares de características seleccionados

Las figuras ilustrativas de estos objetos se omiten porque el repositorio no admite archivos binarios.
Puedes visualizar una superficie de decisión en 3D con:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0])
plt.show()
```

```python
from sheshe import (
    ModalBoundaryClustering,
    SubspaceScout,
    ModalScoutEnsemble,
    ClusterRegion,
    RegionInterpreter,
    ShuShu,
    CheChe,
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

### Matriz de métodos

| Método | ModalBoundaryClustering | ShuShu | CheChe | ModalScoutEnsemble | Descripción breve |
| --- | --- | --- | --- | --- | --- |
| `fit` | ✓ | ✓ | ✓ | ✓ | Ajusta el modelo al conjunto de datos. |
| `fit_predict` | ✓ | ✓ | ✓ | ✓ | Entrena y devuelve etiquetas o regiones asignadas. |
| `fit_transform` | ✓ | ✓ | ✓ | ✓ | Entrena y transforma los datos en una representación interna. |
| `transform` | ✓ | ✓ | ✓ | ✓ | Transforma nuevas muestras según el modelo entrenado. |
| `predict` | ✓ | ✓ | ✓ | ✓ | Predice etiquetas o valores para nuevas muestras. |
| `predict_proba` | ✓ | ✓ | ✓ | ✓ | Estima probabilidades o confianza de predicción. |
| `decision_function` | ✓ | ✓ | ✓ | ✓ | Devuelve la puntuación o distancia al límite de decisión. |
| `predict_regions` | ✓ | ✓ | ✓ | ✓ | Indica la región o clúster al que pertenece cada muestra. Similar a `get_cluster` en algunos predictores. |
| `score` | ✓ | ✓ | ✓ | ✓ | Calcula una métrica de desempeño sobre datos de prueba. |
| `save` | ✓ | ✓ | ✓ | ✓ | Guarda el modelo entrenado en disco. |
| `load` | ✓ | ✓ | ✓ | ✓ | Carga un modelo previamente guardado. |
| `plot_pairs` | ✓ | ✓ | ✓ | ✓ | Grafica pares de características con regiones o clústeres. |
| `plot_pair_3d` | ✓ | ✓ | ✓ | ✓ | Grafica en 3D dos características y la respuesta. |
| `interpretability_summary` | ✓ | ✓ | ✓ | ✓ | Resumen interpretativo de las regiones o del modelo. |

#### Ejemplos de uso

Los siguientes fragmentos asumen que `X` y `y` están definidos y que las clases se importaron desde `sheshe`.

##### `fit`
```python
shu = ShuShu(random_state=0).fit(X, y)
che = CheChe().fit(X, y)
sh = ModalBoundaryClustering().fit(X, y)
mse = ModalScoutEnsemble().fit(X, y)
```

##### `fit_predict`
```python
shu_labels = ShuShu(random_state=0).fit_predict(X, y)
che_labels = CheChe().fit_predict(X, y)
sh_labels = ModalBoundaryClustering().fit_predict(X, y)
mse_labels = ModalScoutEnsemble().fit_predict(X, y)
```

##### `fit_transform`
```python
shu_emb = ShuShu(random_state=0).fit_transform(X, y)
che_emb = CheChe().fit_transform(X, y)
sh_emb = ModalBoundaryClustering().fit_transform(X, y)
mse_emb = ModalScoutEnsemble().fit_transform(X, y)
```

##### `transform`
```python
shu_trans = ShuShu(random_state=0).fit(X, y).transform(X)
che_trans = CheChe().fit(X, y).transform(X)
sh_trans = ModalBoundaryClustering().fit(X, y).transform(X)
mse_trans = ModalScoutEnsemble().fit(X, y).transform(X)
```

##### `predict`
```python
shu_pred = ShuShu(random_state=0).fit(X, y).predict(X)
che_pred = CheChe().fit(X, y).predict(X)
sh_pred = ModalBoundaryClustering().fit(X, y).predict(X)
mse_pred = ModalScoutEnsemble().fit(X, y).predict(X)
```

##### `predict_proba`
```python
shu_proba = ShuShu(random_state=0).fit(X, y).predict_proba(X)
che_proba = CheChe().fit(X, y).predict_proba(X)
sh_proba = ModalBoundaryClustering().fit(X, y).predict_proba(X)
mse_proba = ModalScoutEnsemble().fit(X, y).predict_proba(X)
```

##### `decision_function`
```python
shu_score = ShuShu(random_state=0).fit(X, y).decision_function(X)
che_score = CheChe().fit(X, y).decision_function(X)
sh_score = ModalBoundaryClustering().fit(X, y).decision_function(X)
mse_score = ModalScoutEnsemble().fit(X, y).decision_function(X)
```

##### `predict_regions`
```python
shu_regions = ShuShu(random_state=0).fit(X, y).predict_regions(X)
che_regions = CheChe().fit(X, y).predict_regions(X)
sh_regions = ModalBoundaryClustering().fit(X, y).predict_regions(X)
mse_regions = ModalScoutEnsemble().fit(X, y).predict_regions(X)
```

##### `score`
```python
shu_sc = ShuShu(random_state=0).fit(X, y).score(X, y)
che_sc = CheChe().fit(X, y).score(X, y)
sh_sc = ModalBoundaryClustering().fit(X, y).score(X, y)
mse_sc = ModalScoutEnsemble().fit(X, y).score(X, y)
```

##### `save`
```python
ShuShu(random_state=0).fit(X, y).save("shu.joblib")
CheChe().fit(X, y).save("che.joblib")
ModalBoundaryClustering().fit(X, y).save("mbc.joblib")
ModalScoutEnsemble().fit(X, y).save("mse.joblib")
```

##### `load`
```python
shu = ShuShu.load("shu.joblib")
che = CheChe.load("che.joblib")
sh = ModalBoundaryClustering.load("mbc.joblib")
mse = ModalScoutEnsemble.load("mse.joblib")
```

##### `plot_pairs`
```python
ShuShu(random_state=0).fit(X, y).plot_pairs(X, y)
CheChe().fit(X, y).plot_pairs(X, y)
ModalBoundaryClustering().fit(X, y).plot_pairs(X, y)
ModalScoutEnsemble().fit(X, y).plot_pairs(X, y)
```

##### `plot_pair_3d`
```python
ShuShu(random_state=0).fit(X, y).plot_pair_3d(X, (0, 1))
CheChe().fit(X, y).plot_pair_3d(X, (0, 1))
ModalBoundaryClustering().fit(X, y).plot_pair_3d(X, (0, 1))
ModalScoutEnsemble().fit(X, y).plot_pair_3d(X, (0, 1))
```

##### `interpretability_summary`
```python
shu_int = ShuShu(random_state=0).fit(X, y).interpretability_summary()
che_int = CheChe().fit(X, y).interpretability_summary()
sh_int = ModalBoundaryClustering().fit(X, y).interpretability_summary()
mse_int = ModalScoutEnsemble().fit(X, y).interpretability_summary()
```

### Métodos
- `fit(X, y)`
- `predict(X)`
- `fit_predict(X, y=None)` → método de conveniencia equivalente a llamar a `fit` y luego a `predict` sobre los mismos datos
- `predict_proba(X)`  → clasificación: probas por clase; regresión: valor normalizado [0,1]
- `decision_function(X)` → valores de decisión del estimador base; recurre a `predict_proba` en clasificación o a `predict` en regresión
- `interpretability_summary(feature_names=None)` → DataFrame con:
  - `Tipo`: "centroide" | "inflexion_point"
  - `Distancia`: radio desde el centro al punto de inflexión
  - `Categoria`: clase (o "NA" en regresión)
  - `pendiente`: df/dt en el punto de inflexión
  - `valor_real` / `valor_norm`
  - `coord_0..coord_{d-1}` o nombres de features
- `predict_regions(X, label_path=None)` → ID(s) de clúster por muestra
- `get_cluster(cluster_id)` → obtiene la `ClusterRegion` con ese ID
- `plot_pairs(X, y=None, max_pairs=None)` → gráficos 2D para todas las combinaciones de pares
- `save(filepath)` → guarda el modelo mediante `joblib`
- `ModalBoundaryClustering.load(filepath)` → carga una instancia guardada

#### `predict_regions(X, label_path=None)`

Devuelve el identificador de clúster para cada muestra basándose únicamente en
las regiones descubiertas.

```python
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
print(sh.predict_regions(X[:3]))
```

#### Ejemplo de regresión con reentrenamiento

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sheshe import ModalBoundaryClustering

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# entrenamiento inicial con el estimador por defecto
reg = ModalBoundaryClustering(task="regression").fit(X_train, y_train)
print(reg.predict(X_test)[:3])

# reentrenar con un estimador base diferente
reg_reentrenado = ModalBoundaryClustering(
    base_estimator=RandomForestRegressor(random_state=0),
    task="regression",
).fit(X_train, y_train)
print(reg_reentrenado.predict(X_test)[:3])
```

#### `get_cluster(cluster_id)`

Obtiene la :class:`ClusterRegion` asociada al identificador indicado.

```python
reg = sh.get_cluster(0)
print(reg.center)
```

### Métricas por clúster

Tras el ajuste, `ModalBoundaryClustering` guarda cada región descubierta en el
atributo `regions_`. Cada `ClusterRegion` incluye:

- `score`: eficacia del estimador sobre las muestras que pertenecen al clúster.
  Usa exactitud (accuracy) para clasificación y R² para regresión.
- `metrics`: diccionario opcional con métricas adicionales por clúster como
  precision, recall, F1, MSE o MAE.

### Exploración de fronteras 2D con CheChe

`CheChe` evalúa pares de características y calcula el "convex hull" que
contiene las muestras para cada par seleccionado. Su método `plot_pairs`
superpone estas fronteras sobre gráficos de dispersión de los datos originales,
y el argumento opcional `mapping_level` permite muestrear solo una fracción de
los puntos antes de calcular las fronteras:

```python
from sklearn.datasets import load_iris
from sheshe import CheChe

X, y = load_iris(return_X_y=True)
ch = CheChe().fit(
    X,
    y,
    feature_names=["sepal length", "sepal width", "petal length", "petal width"],
    mapping_level=2,  # usa una de cada dos muestras
)
ch.plot_pairs(X, class_index=0)
```

El ejemplo anterior dibuja la frontera para el índice de clase ``0``. Si `fit`
se invoca sin etiquetas, `class_index` puede omitirse para graficar la frontera
en modo escalar.

## ShuShu – búsqueda de máximos por gradiente

`ShuShu` localiza máximos locales de una función escalar y ejecuta una
búsqueda por clase cuando se proporcionan etiquetas.

```python
from sklearn.datasets import load_iris
from sheshe import ShuShu

X, y = load_iris(return_X_y=True)

# Optimización multiclase: usa LogisticRegression internamente
sh = ShuShu(random_state=0).fit(X, y)
print(sh.summary_tables()[0][["class_label", "n_clusters"]])

# Ejemplo con función escalar
import numpy as np
def paraboloide(Z):
    return -np.linalg.norm(Z - 1.0, axis=1)

sc = ShuShu(random_state=0).fit(np.random.rand(100, 2), score_fn=paraboloide)
print(sc.centroids_)
```

## Interpretabilidad

### Interpretación de regiones

```python
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering, RegionInterpreter

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering().fit(X, y)
cards = RegionInterpreter(feature_names=iris.feature_names).summarize(sh.regions_)
RegionInterpreter.pretty_print(cards[:1])
```

Cada tarjeta incluye un `cluster_id` para identificar la región y la clase `label`.

### Descripciones en lenguaje natural con OpenAI

Instala la dependencia opcional ``openai`` (versión ``>=1``) y proporciona una
clave ya sea con el argumento ``api_key`` o mediante variables de entorno. El
intérprete busca ``OPENAI_API_KEY`` o ``OPENAI_KEY`` y, al ejecutarse en Google
Colab, también revisa ``google.colab.userdata``. Puedes fijar idioma y
temperatura por defecto en el intérprete y sobreescribirlos al llamar
``describe_cards``. El parámetro ``layout`` permite fijar un template general
(por ejemplo, ``"lista con viñetas"``) o dejar que el modelo responda libremente.

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

`ModalScoutEnsemble` entrena varios modelos `ModalBoundaryClustering` en los mejores subespacios devueltos por `SubspaceScout` y combina sus predicciones. Establece `ensemble_method="shushu"` para delegar el ensamble en el optimizador `ShuShu`.

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
    # ensemble_method="shushu" utilizaría el optimizador ShuShu
)
mse.fit(X, y)
print(mse.predict(X[:5]))
```

#### `predict_proba(X)`

Solo disponible para tareas de clasificación, devuelve la mezcla ponderada de
probabilidades de clase de todos los submodelos.

```python
mse.fit(X, y)
print(mse.predict_proba(X[:5]))
```

#### `predict_regions(X)`

Devuelve la etiqueta y el `cluster_id` estimados para cada muestra.

```python
labels, cluster_ids = mse.predict_regions(X[:3])
print(cluster_ids)
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
[`compare_unsupervised.py`](experiments/compare_unsupervised.py) evalúa ocho
conjuntos de datos distintos (Iris, Wine, Breast Cancer, Digits, California
Housing, Moons, Blobs y Circles), explora parámetros de **SheShe**, **KMeans** y
**DBSCAN**, y almacena cuatro métricas (`ARI`, `homogeneity`, `completeness`,
`v_measure`) junto con el tiempo de ejecución (`runtime_sec`).

```bash
python experiments/compare_unsupervised.py
cat benchmark/unsupervised_results.csv | head
```

se generan los resultados dentro de `benchmark/`.

Se añadió una comparación A/B de la búsqueda guiada por subespacios en
[benchmark/subspace_ab_results.csv](benchmark/subspace_ab_results.csv); la tabla
resumen muestra los tiempos promedio (segundos) sobre 5 semillas.

| dataset | baseline | subspace | subspace + light + escape |
| --- | --- | --- | --- |
| digits | 0.0567 | 0.0233 | 0.0222 |
| iris | 0.0040 | 0.00262 | 0.00268 |
=======
El nuevo modo de rayos `grad` reemplaza al anterior `grid`, logrando hasta
~9× de aceleración sin pérdida de exactitud (ver
[benchmark/README.md](benchmark/README.md)).

Para los experimentos del manuscrito se incluye además
[`paper_experiments.py`](experiments/paper_experiments.py), que compara con
algoritmos **supervisados**, realiza estudios de *ablation* sobre
`base_2d_rays`, `direction`, `jaccard_threshold`, `drop_fraction` y
`smooth_window`, y analiza la sensibilidad a la dimensión y al ruido gaussiano.
Ejecutando el script se obtienen tablas (`*.csv`) y figuras (`*.png`)
reproducibles en `benchmark/`:

```bash
python experiments/paper_experiments.py
```

---

## Parámetros clave
- `base_2d_rays` → controla la resolución angular en 2D (32 por defecto). 3D escala ~34; d>3 usa subespacios.
- `direction` → "center_out" | "outside_in" para localizar el punto de inflexión.
- `scan_radius_factor`, `scan_steps` → tamaño y resolución del escaneo radial.
- `optim_method` → `"gradient_ascent"` (por defecto) o `"trust_region_newton"`;
  la variante de región de confianza usa gradientes y hessianos para resolver
  subproblemas cuadráticos dentro de un radio adaptable y respeta los límites.
- `grad_*` → hiperparámetros del ascenso por gradiente (tasa, iteraciones,
  tolerancias; se usan solo si `optim_method="gradient_ascent"`).
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

## Imágenes

Las figuras se han omitido deliberadamente porque este repositorio no permite almacenar archivos binarios.

---

## Sobre el autor

SheShe es desarrollado por José Carlos Del Valle. Encuéntralo en [LinkedIn](https://www.linkedin.com/in/jose-carlos-del-valle/) o visita su [portafolio](https://jcval94.github.io/Portfolio/).

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
