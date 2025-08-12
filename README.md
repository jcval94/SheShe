# SheShe  
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

Segmentación de bordes y exploración de hiper-fronteras basada en **máximos locales** de
la **probabilidad por clase** (clasificación) o del **valor predicho** (regresión).

---

## Instalación (modo editable recomendado)

```bash
pip install -e .
```

Requisitos base: `numpy`, `pandas`, `scikit-learn>=1.1`, `matplotlib`

---

## API rápida

```python
from sheshe import ModalBoundaryClustering

# clasificación
clf = ModalBoundaryClustering(
    base_estimator=None,           # por defecto LogisticRegression
    task="classification",         # "classification" | "regression"
    base_2d_rays=8,                # nº de rayos en 2D (≈45°)
    direction="center_out",        # "center_out" | "outside_in"
    scan_radius_factor=3.0,
    scan_steps=64,
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
- `save(filepath)` → guarda la instancia entrenada con `joblib`
- `ModalBoundaryClustering.load(filepath)` → recupera una instancia guardada

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
   Registra además la **pendiente** (df/dt) en ese punto.
5. Conecta los puntos de inflexión para formar la **frontera** de la región de alta probabilidad/valor.

---

## Ejemplos

### Clasificación — Iris
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sheshe import ModalBoundaryClustering

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=1000)
sh = ModalBoundaryClustering(base_estimator=model, task="classification", base_2d_rays=8, random_state=0)
sh.fit(X, y)

print(sh.interpretability_summary(iris.feature_names).head())
sh.plot_pairs(X, y, max_pairs=3)   # genera gráficas por pares de variables
```

### Regresión — Diabetes
```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sheshe import ModalBoundaryClustering

diab = load_diabetes()
X, y = diab.data, diab.target

sh = ModalBoundaryClustering(base_estimator=GradientBoostingRegressor(random_state=0),
                             task="regression", base_2d_rays=8, random_state=0)
sh.fit(X, y)

print(sh.interpretability_summary(diab.feature_names).head())
sh.plot_pairs(X, max_pairs=3)
```

### Guardado y carga
```python
from pathlib import Path
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering(random_state=0).fit(X, y)
path = Path("mbc_model.joblib")
sh.save(path)
sh2 = ModalBoundaryClustering.load(path)
print((sh.predict(X[:5]) == sh2.predict(X[:5])).all())
```

Para ejemplos más completos, consulta la carpeta `examples/`.

### Visualización y guardado de gráficas

El script [`examples/iris_visualization.py`](examples/iris_visualization.py) muestra
cómo generar gráficas y almacenarlas en disco:

```bash
python examples/iris_visualization.py
ls examples/images
```

Las imágenes se guardan en `examples/images/`.

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

---

## Parámetros clave
- `base_2d_rays` → controla la resolución angular en 2D (8 por defecto). 3D escala ~26; d>3 usa subespacios.
- `direction` → "center_out" | "outside_in" para localizar el punto de inflexión.
- `scan_radius_factor`, `scan_steps` → tamaño y resolución del escaneo radial.
- `grad_*` → hiperparámetros del ascenso (tasa, iteraciones, tolerancias).
- `max_subspaces` → nº máx. de subespacios considerados cuando d>3.

---

## Limitaciones
- Depende de la **superficie** producida por el modelo base (puede ser rugosa en RF).
- En alta dimensión, la frontera es una **aproximación** (subespacios).
- Encuentra **máximos locales** (no garantiza el global), mitigado con múltiples *seeds*.

---

## Licencia
MIT
