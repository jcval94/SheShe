# SheShe
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

Edge segmentation and hyperboundary exploration based on **local maxima** of
the **class probability** (classification) or the **predicted value**
(regression). It is a supervised clustering algorithm.

Unlike traditional unsupervised clustering methods that rely only on feature
similarity, SheShe learns from labeled examples. A base estimator models the
relationship between inputs and targets, and the algorithm discovers regions
whose responses remain consistently high for a given class or target value.
Clusters therefore follow the supervised decision surface instead of arbitrary
distance metrics.

---

## Installation

Requires **Python >=3.9** and it is recommended to work inside a virtual
environment. Install the latest release from
[PyPI](https://pypi.org/project/sheshe/):

```bash
pip install sheshe
```

Base dependencies: `numpy`, `pandas`, `scikit-learn>=1.1`, `matplotlib`

For a development environment with tests:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

---

## Quick API

The library exposes five main objects:

- `ModalBoundaryClustering`
- `ClusterRegion` – dataclass describing a discovered region
- `SubspaceScout`
- `ModalScoutEnsemble`
- `RegionInterpreter` – turn `ClusterRegion` objects into human-readable rules

```python
from sheshe import (
    ModalBoundaryClustering,
    SubspaceScout,
    ModalScoutEnsemble,
    ClusterRegion,
)

# classification
clf = ModalBoundaryClustering(
    base_estimator=None,           # default LogisticRegression
    task="classification",         # "classification" | "regression"
    base_2d_rays=24,
    direction="center_out",        # "center_out" | "outside_in"
    scan_radius_factor=3.0,
    scan_steps=24,
    smooth_window=None,             # optional moving average window
    drop_fraction=0.5,              # fallback drop from peak value
    stop_criteria="inflexion",     # or "percentile" for percentile-bin drop
    percentile_bins=20,             # number of percentile bins when stop_criteria="percentile"
    random_state=0
)

# regression (example)
reg = ModalBoundaryClustering(task="regression")
```

### Methods
- `fit(X, y)`
- `predict(X)`
- `fit_predict(X, y=None)` → convenience method equivalent to calling `fit` followed by `predict` on the same data
- `predict_proba(X)`  → classification: per-class probabilities; regression: normalized value [0,1]
- `decision_function(X)` → decision scores from the base estimator; falls back to `predict_proba` for classification or `predict` for regression
- `interpretability_summary(feature_names=None)` → DataFrame with:
  - `Type`: "centroid" | "inflection_point"
  - `Distance`: radius from the center to the inflection point
  - `Category`: class (or "NA" in regression)
  - `slope`: df/dt at the inflection point
  - `real_value` / `norm_value`
  - `coord_0..coord_{d-1}` or feature names
- `plot_pairs(X, y=None, max_pairs=None)` → 2D plots for all pair combinations
- `save(filepath)` → save the model using `joblib`
- `ModalBoundaryClustering.load(filepath)` → load a saved instance

Example of `fit_predict` usage:

```python
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
labels = ModalBoundaryClustering().fit_predict(X, y)
print(labels[:5])
```

#### `decision_function(X)`

Returns decision values from the underlying estimator. For classification it
prefers the estimator's ``decision_function`` but falls back to
``predict_proba`` when that method is missing. In regression the method relies
on ``predict`` as a fallback.

```python
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
print(sh.decision_function(X[:5]))
```

### RegionInterpreter – interpret cluster regions

```python
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering, RegionInterpreter

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering().fit(X, y)
cards = RegionInterpreter(feature_names=iris.feature_names).summarize(sh.regions_)
RegionInterpreter.pretty_print(cards[:1])
```

Each card includes a `cluster_id` to identify the region and the class `label`.

#### Visualización 3D
`plot_pair_3d` visualiza la probabilidad de una clase o el valor predicho como
una superficie tridimensional para un par de características.

Parámetros principales:
- `pair`: tupla `(i, j)` con los índices de las características a graficar.
- `class_label`: etiqueta de la clase a mostrar cuando `task='classification'`.
- `grid_res`: resolución de la malla usada para la superficie.
- `alpha_surface`: transparencia de la superficie.

Ejemplo mínimo:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering().fit(X, y)
sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0])
plt.show()
```

---

## How does it work?
1. Train/use a **base model** from sklearn (classification with `predict_proba`
   or regression with `predict`).
2. Find **local maxima** via **gradient ascent** with barriers at the domain
   boundaries.
3. From the maximum, trace **rays** (directions) on the hypersphere:
   - 2D: 24 rays by default
   - 3D: ~26 directions (coverage by spherical *caps* using Fibonacci sampling)
   - >3D: mixture of a few global directions + 2D/3D **subspaces**
4. Along each ray, **scan radially** and compute the **first inflection point**
   according to `direction` and `stop_criteria`:
    - `center_out`: from the center outward
   - `outside_in`: from the outside toward the center
   Optionally apply a moving average (`smooth_window`) and record the **slope**
   (df/dt) at that point. With `stop_criteria="percentile"` the scan stops
   when the value falls to a lower percentile bin of the dataset distribution
   (20 bins by default). If no
   stop is found, use the first point where the value drops below
   `drop_fraction` of the peak.
5. Connect the inflection points to form the **boundary** of the region with
   high probability/value.

---

## Examples
### Classification — Iris
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
sh.plot_pairs(X, y, max_pairs=3)   # generate the plots
plt.show()
```

### Classification with pre-trained model
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sheshe import ModalBoundaryClustering

wine = load_wine()
X, y = wine.data, wine.target

# Train a model independently
base_model = RandomForestClassifier(n_estimators=200, random_state=0)
base_model.fit(X, y)

# Use SheShe with that pre-fitted model
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

### Classification — synthetic blobs with custom parameters
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

### Regression — Diabetes
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

The percentile-based stopping rule avoids the point of inflection and scans
only until the value crosses into a lower percentile bin (20 bins by default).
The optimized loop
implementation is considerably faster than the previous vectorized version. On
the Iris dataset:

```
$ PYTHONPATH=src python experiments/benchmark_stop_criteria.py
vectorized implementation: 0.0259s
loop implementation:       0.0121s
speedup: 2.14x
ModalBoundaryClustering fit with stop_criteria='inflexion': 0.1026s
ModalBoundaryClustering fit with stop_criteria='percentile': 0.1411s
```

The exact numbers depend on the machine, but the optimized loop method is
substantially quicker while producing the same results.

### Saving figures
```python
from pathlib import Path
import matplotlib.pyplot as plt

# after calling ``sh.plot_pairs(...)``
out_dir = Path("images")
out_dir.mkdir(exist_ok=True)
for i, fig_num in enumerate(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(out_dir / f"pair_{i}.png")
plt.close(fig_num)
```

### Plotting with pandas DataFrames
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

sh = ModalBoundaryClustering().fit(df, iris.target)
sh.plot_pairs(df, iris.target, max_pairs=2)  # usa nombres de columnas en los ejes
plt.show()
```

### Visualizing interpretability summary
```python
import matplotlib.pyplot as plt

summary = sh.interpretability_summary(df.columns)
centroids = summary[summary["Type"] == "centroid"]
plt.scatter(centroids["coord_0"], centroids["coord_1"], c=centroids["Category"])
plt.xlabel("coord_0")
plt.ylabel("coord_1")
plt.show()
```

### Save and load model
```python
from pathlib import Path
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering().fit(X, y)
path = Path("sheshe_model.joblib")
sh.save(path)
sh2 = ModalBoundaryClustering.load(path)
print((sh.predict(X) == sh2.predict(X)).all())
```

For more complete examples, see the `examples/` folder.

## SubspaceScout

`SubspaceScout` helps discover informative feature subspaces (pairs, trios, ...)
before running SheShe. It can work purely with mutual information or leverage
optional models like LightGBM+SHAP or EBM to rank feature interactions.

```python
from sheshe import SubspaceScout

scout = SubspaceScout(
    # model_method='lightgbm',    # default uses MI; LightGBM and SHAP are optional
    max_order=4,                # explore pairs, trios and quartets
    top_m=50,                   # limit to top 50 informative features
    base_pairs_limit=12,        # seed pairs for orders >=3
    beam_width=10,              # combos kept per layer
    extend_candidate_pool=16,   # random candidate features per parent
    branch_per_parent=4,        # extensions per parent
    marginal_gain_min=1e-3,     # minimum gain to accept
    max_eval_per_order=150,     # cap MI evaluations per order
    sample_size=4096,           # subsample size
    time_budget_s=None,         # e.g., 15.0 for 15 seconds
    task='classification',
    random_state=0,
)
subspaces = scout.fit(X, y)
```

## ModalScoutEnsemble

`ModalScoutEnsemble` trains multiple ModalBoundaryClustering models on the top subspaces returned by `SubspaceScout` and combines their predictions.

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

#### `predict_proba(X)`

Only available for classification tasks, this method returns the weighted mixture
of class probabilities from all submodels in the ensemble.

```python
mse.fit(X, y)
print(mse.predict_proba(X[:5]))
```

#### `report()`

`report()` returns a list with one entry per trained subspace, sorted by
`weight`. Each entry is a dictionary containing:

- `features`: tuple with the indices of the features in that subspace.
- `order`: number of features (subspace order).
- `scout_score`: score assigned by `SubspaceScout`.
- `cv_score`: cross-validation score of the submodel.
- `feat_importance`: mean feature importance for the subspace.
- `weight`: normalized weight used by the ensemble.

Example:

```python
from pprint import pprint

summary = mse.report()
pprint([
    {k: row[k] for k in ("features", "order", "scout_score", "cv_score", "feat_importance", "weight")}
    for row in summary[:2]
])
```

Output:

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

### Experiments and benchmark

The experiments comparing against **unsupervised** algorithms are located in
the [`experiments/`](experiments/) folder. The script
[`compare_unsupervised.py`](experiments/compare_unsupervised.py) evaluates five
different datasets, explores parameters of **SheShe**, **KMeans** and
**DBSCAN**, and stores four metrics (`ARI`, `homogeneity`, `completeness`,
`v_measure`) along with the execution time (`runtime_sec`).

```bash
python experiments/compare_unsupervised.py --runs 5
cat benchmark/unsupervised_results_summary.csv | head
```

Results are generated inside `benchmark/` (valores por repetición y medias en
`*_summary.csv`).

For the manuscript we provide additional scripts in
[`paper_experiments.py`](experiments/paper_experiments.py) which perform
supervised comparisons, ablation studies over `base_2d_rays` and `direction`,
and sensitivity analyses w.r.t. dimensionality and Gaussian noise.  Executing
the script generates tables with todas las repeticiones y un resumen (`*_summary.csv`),
además de figuras (`*.png`) bajo `benchmark/`:

```bash
python experiments/paper_experiments.py --runs 5
```

---

## Key parameters
- `base_2d_rays` → controls angular resolution in 2D (24 by default). 3D scales
  to ~26; d>3 uses subspaces.
- `direction` → "center_out" | "outside_in" to locate the inflection point.
- `scan_radius_factor`, `scan_steps` → size and resolution of the radial scan.
- `grad_*` → hyperparameters of gradient ascent (rate, iterations, tolerances).
- `max_subspaces` → max number of subspaces considered when d>3.
- `density_alpha` / `density_k` → optional density penalty computed with an
  HNSW k‑NN search (via `hnswlib`) to keep centers inside the data cloud. The
  normalized value is multiplied by `(density(x))**density_alpha`; set
  `density_alpha=0` to disable.

---

## Performance tips
- Defaults favour speed: `base_2d_rays=24`, `scan_steps=24` and `n_max_seeds=2`.
- The heuristic `auto_rays_by_dim=True` (default) reduces rays for high dimensional datasets:
  - 25–64 features → `base_2d_rays` capped at 16.
  - 65+ features → `base_2d_rays` capped at 12.
  For 30D problems such as Breast Cancer this matches the recommended `base_2d_rays=16`.

---

## Limitations
- Depends on the **surface** produced by the base model (can be rough in RF).
- In high dimension, the boundary is an **approximation** (subspaces).
- Finds **local maxima** (does not guarantee the global one), mitigated with
  multiple seeds.

---

## Contribute

Improvements are welcome. To propose changes:

1. Fork the repository and create a descriptive branch.
2. Install development dependencies and run the tests:

   ```bash
   pip install -e ".[dev]"
   PYTHONPATH=src pytest -q
   ```
3. Submit a pull request with a clear description of the change.

---

## License
MIT

