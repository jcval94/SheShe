# SheShe
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

**SheShe** transforms any probabilistic model into a guided explorer of its own
decision landscape. By following the **local maxima** of the **class
probability** (classification) or the **predicted value** (regression), it
discovers crisp, human‑readable regions that obey the supervised boundary of the
problem. Rather than grouping samples by raw feature distance, SheShe learns
from labeled data and carves clusters directly on top of the model’s decision
surface.

## Highlights

- Supervised clustering driven by the model’s own probabilities or predictions.
- Unified support for classification and regression tasks.
- Subspace exploration with `SubspaceScout` and ensembles via `ModalScoutEnsemble`.
- Human-readable rule extraction through `RegionInterpreter`.
- Built-in plotting utilities for pairwise and 3D visualisations.

*Feature overview figure omitted (binary assets are not allowed).* 

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

Optional acceleration is available with [`numba`](https://numba.pydata.org/).
Install via `pip install sheshe[numba]` to enable JIT‑compiled finite-difference
gradients and Hessians.

## Reproducibility

This project was developed and tested on:

- **OS:** Ubuntu 24.04.2 LTS
- **CPU:** Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz
- **GPU:** None
- **Python:** 3.12.10

To recreate the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

---

## Quick API

The library exposes seven main objects:

- `ModalBoundaryClustering`
- `ClusterRegion` – dataclass describing a discovered region
- `SubspaceScout`
- `ModalScoutEnsemble`
- `RegionInterpreter` – turn `ClusterRegion` objects into human-readable rules
- `ShuShu` – gradient-based search for local maxima
- `CheChe` – compute 2D frontiers on selected feature pairs

Figures illustrating these objects are omitted because binary assets are not allowed in this repository.

```python
from sheshe import (
    ModalBoundaryClustering,
    SubspaceScout,
    ModalScoutEnsemble,
    ClusterRegion,
    RegionInterpreter,
    ShuShu,
    CheChe,
    plot_cheche,
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
- `predict_regions(X, label_path=None)` → cluster ID(s) for each sample
- `get_cluster(cluster_id)` → retrieve a stored `ClusterRegion`
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

#### Regression example with retraining

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sheshe import ModalBoundaryClustering

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# initial training with the default estimator
reg = ModalBoundaryClustering(task="regression").fit(X_train, y_train)
print(reg.predict(X_test)[:3])

# retrain using a different base estimator
reg_retrained = ModalBoundaryClustering(
    base_estimator=RandomForestRegressor(random_state=0),
    task="regression",
).fit(X_train, y_train)
print(reg_retrained.predict(X_test)[:3])
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

#### `predict_regions(X, label_path=None)`

Return cluster identifiers for each sample based solely on the discovered
regions.

```python
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
print(sh.predict_regions(X[:3]))
```

#### `get_cluster(cluster_id)`

Fetch a stored :class:`ClusterRegion` by its identifier.

```python
reg = sh.get_cluster(0)
print(reg.center)
```

### Per-cluster metrics

After fitting, `ModalBoundaryClustering` stores the discovered regions in the
`regions_` attribute. Each `ClusterRegion` includes:

- `score`: effectiveness of the estimator on samples inside the region
  (accuracy for classification, R² for regression)
- `metrics`: optional dictionary with additional per-cluster metrics such as
  precision, recall, F1, MSE or MAE

### 2D frontier exploration with CheChe

`CheChe` evaluates pairs of features and computes the convex hull enclosing
the samples for each selected pair. The helper `plot_cheche` overlays these
frontiers on scatter plots of the original data, and the optional
`mapping_level` argument can subsample the points before calculating the
frontiers:

```python
from sklearn.datasets import load_iris
from sheshe import CheChe, plot_cheche

X, y = load_iris(return_X_y=True)
ch = CheChe().fit(
    X,
    y,
    feature_names=["sepal length", "sepal width", "petal length", "petal width"],
    mapping_level=2,  # use every other sample
)
plot_cheche(ch, X, class_index=0)
```

The example above draws the frontier for class index ``0``. When ``fit`` is
called without labels, `class_index` can be omitted to plot the scalar mode
frontier.

## ShuShu – gradient-based maxima search

The `ShuShu` optimizer locates local maxima of a scalar score function and
automatically runs once per class when provided with labels.

```python
from sklearn.datasets import load_iris
from sheshe import ShuShu

X, y = load_iris(return_X_y=True)

# Multiclass optimisation: one run per class using LogisticRegression internally
sh = ShuShu(random_state=0).fit(X, y)
print(sh.summary_tables()[0][["class_label", "n_clusters"]])

# Scalar score function example
import numpy as np
def paraboloid(Z):
    return -np.linalg.norm(Z - 1.0, axis=1)

sc = ShuShu(random_state=0).fit(np.random.rand(100, 2), score_fn=paraboloid)
print(sc.centroids_)
```

## Interpretability

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

### OpenAIRegionInterpreter – describe regions with LLMs

Install the optional ``openai`` dependency (version ``>=1``) and provide an API
key using the ``api_key`` argument or via environment variables. The interpreter
looks for ``OPENAI_API_KEY`` or ``OPENAI_KEY`` and, when running on Google
Colab, also checks ``google.colab.userdata``. Language and temperature defaults
can be configured on the interpreter and overridden at call time. The ``layout``
parameter lets you enforce a general output template (for example, ``"bullet list"``)
or omit it for free‑form text. Then call ``describe_cards`` to obtain natural‑
language explanations for the region cards.

```python
from sheshe import OpenAIRegionInterpreter

expl = OpenAIRegionInterpreter(model="gpt-4o-mini", language="en", temperature=0.2)
texts = expl.describe_cards(cards, layout="bullet list", temperature=0.5)
print(texts[0])
```

## Visualización 3D
`plot_pair_3d` visualiza la probabilidad de una clase o el valor predicho como
una superficie tridimensional para un par de características.

Parámetros principales:
- `pair`: tupla `(i, j)` con los índices de las características a graficar.
- `class_label`: etiqueta de la clase a mostrar cuando `task='classification'`.
- `grid_res`: resolución de la malla usada para la superficie.
- `alpha_surface`: transparencia de la superficie.
- `engine`: `'matplotlib'` (por defecto) para una figura estática o `'plotly'`
  para un gráfico interactivo.

Ejemplo mínimo:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering().fit(X, y)
# Modo estático con Matplotlib
sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0])
plt.show()

# Modo interactivo con Plotly
fig = sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0], engine="plotly")
fig.show()
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

### Meta-optimization — Random search

The ``examples/meta_optimization.py`` script showcases a gradient-free meta-
optimization routine.  It evaluates random hyperparameter configurations for
``ModalBoundaryClustering`` directly instead of relying on first-order
approximations, providing a simple way to tune the algorithm.

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
    fig = plt.figure(fig_num)
    fig.savefig(out_dir / f"pair_{i}.png")
    plt.close(fig)
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

`ModalScoutEnsemble` trains multiple ModalBoundaryClustering models on the top subspaces returned by `SubspaceScout` and combines their predictions. Set `ensemble_method="shushu"` to delegate the ensemble to the `ShuShu` optimizer.

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
    # ensemble_method="shushu" would use the ShuShu optimizer
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

#### `predict_regions(X)`

Return the predicted label and cluster identifier for each sample.

```python
labels, cluster_ids = mse.predict_regions(X[:3])
print(cluster_ids)
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

#### `plot_pairs(X, y=None, model_idx=0, max_pairs=None)`

Visualize 2D decision surfaces of a given submodel using the same plotting
utilities as `ModalBoundaryClustering`.

```python
feats = mse.features_[0]
mse.plot_pairs(X, y, model_idx=0, max_pairs=1)
```

#### `plot_pair_3d(X, pair, model_idx=0, class_label=None, grid_res=50, alpha_surface=0.6, engine="matplotlib")`

Render probability (classification) or predicted value (regression) as a 3D
surface for a specific submodel.

```python
feats = mse.features_[0]
mse.plot_pair_3d(X, (feats[0], feats[1]), model_idx=0, class_label=mse.classes_[0])
```

### Experiments and benchmark

The experiments comparing against **unsupervised** algorithms are located in
the [`experiments/`](experiments/) folder. The script
[`compare_unsupervised.py`](experiments/compare_unsupervised.py) evaluates eight
different datasets (Iris, Wine, Breast Cancer, Digits, California Housing,
Moons, Blobs, Circles), explores parameters of **SheShe**, **KMeans** and
**DBSCAN**, and stores four metrics (`ARI`, `homogeneity`, `completeness`,
`v_measure`) along with the execution time (`runtime_sec`).

```bash
python experiments/compare_unsupervised.py --runs 5
cat benchmark/unsupervised_results_summary.csv | head
```

Results are generated inside `benchmark/` (valores por repetición y medias en
`*_summary.csv`).


An additional A/B comparison for the subspace-guided search is available in
[benchmark/subspace_ab_results.csv](benchmark/subspace_ab_results.csv); the table
below reports mean runtimes in seconds (5 seeds).

| dataset | baseline | subspace | subspace + light + escape |
| --- | --- | --- | --- |
| digits | 0.0567 | 0.0233 | 0.0222 |
| iris | 0.0040 | 0.00262 | 0.00268 |
=======
The new `grad` ray mode replaces the former `grid` approach, delivering up to
~9× faster fits with identical accuracy (see
[benchmark/README.md](benchmark/README.md)).


For the manuscript we provide additional scripts in
[`paper_experiments.py`](experiments/paper_experiments.py) which perform
supervised comparisons, ablation studies over `base_2d_rays`, `direction`,
`jaccard_threshold`, `drop_fraction` and `smooth_window`, and sensitivity
analyses w.r.t. dimensionality and Gaussian noise.  Executing the script
generates tables with todas las repeticiones y un resumen (`*_summary.csv`),
además de figuras (`*.png`) bajo `benchmark/`:

```bash
python experiments/paper_experiments.py --runs 5
```

---

## Key parameters
- `base_2d_rays` → controls angular resolution in 2D (32 by default). 3D scales
  to ~34; d>3 uses subspaces.
- `direction` → "center_out" | "outside_in" to locate the inflection point.
- `scan_radius_factor`, `scan_steps` → size and resolution of the radial scan.
- `optim_method` → `"gradient_ascent"` (default) or `"trust_region_newton"`; the
  trust-region variant uses gradients and Hessians to solve quadratic
  subproblems inside an adaptive radius and respects box constraints.
- `grad_*` → hyperparameters of gradient ascent (rate, iterations, tolerances;
  used only when `optim_method="gradient_ascent"`).
- `max_subspaces` → max number of subspaces considered when d>3.
- `density_alpha` / `density_k` → optional density penalty computed with an
  HNSW k‑NN search (via `hnswlib`) to keep centers inside the data cloud. The
  normalized value is multiplied by `(density(x))**density_alpha`; set
  `density_alpha=0` to disable.

---

## Performance tips
- Defaults favour speed: `base_2d_rays=32`, `scan_steps=24` and `n_max_seeds=2`.
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

## Images

Figures have been intentionally omitted because this repository does not permit storing binary assets.

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

