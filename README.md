# SheShe  
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

Segmentación de bordes y exploración de hiper-fronteras basada en **máximos locales** de
la **probabilidad por clase** (clasificación) o del **valor predicho** (regresión).

---

## Instalación

Requiere **Python >=3.9** y se recomienda el uso de un entorno virtual.

```bash
pip install -e .
```

Dependencias base: `numpy`, `pandas`, `scikit-learn>=1.1`, `matplotlib`

Para un entorno de desarrollo con pruebas:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

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

Para ejemplos más completos, consulta la carpeta `examples/`.

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
