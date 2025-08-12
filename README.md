# SheShe
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

Edge segmentation and hyperboundary exploration based on **local maxima** of
the **class probability** (classification) or the **predicted value**
(regression).

---

## Installation

Requires **Python >=3.9** and it is recommended to work inside a virtual
environment.

```bash
pip install -e .
```

Base dependencies: `numpy`, `pandas`, `scikit-learn>=1.1`, `matplotlib`

For a development environment with tests:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

---

## Quick API

```python
from sheshe import ModalBoundaryClustering

# classification
clf = ModalBoundaryClustering(
    base_estimator=None,           # default LogisticRegression
    task="classification",         # "classification" | "regression"
    base_2d_rays=8,                # number of rays in 2D (≈45°)
    direction="center_out",        # "center_out" | "outside_in"
    scan_radius_factor=3.0,
    scan_steps=64,
    random_state=0
)

# regression (example)
reg = ModalBoundaryClustering(task="regression")
```

### Methods
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`  → classification: per-class probabilities; regression: normalized value [0,1]
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

---

## How does it work?
1. Train/use a **base model** from sklearn (classification with `predict_proba`
   or regression with `predict`).
2. Find **local maxima** via **gradient ascent** with barriers at the domain
   boundaries.
3. From the maximum, trace **rays** (directions) on the hypersphere:
   - 2D: 8 rays by default
   - 3D: ~26 directions (coverage by spherical *caps* using Fibonacci sampling)
   - >3D: mixture of a few global directions + 2D/3D **subspaces**
4. Along each ray, **scan radially** and compute the **first inflection point**
   according to `direction`:
   - `center_out`: from the center outward
   - `outside_in`: from the outside toward the center
   Also record the **slope** (df/dt) at that point.
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
    base_2d_rays=8,
    random_state=0,
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
    base_2d_rays=8,
    random_state=0,
).fit(X, y)

sh.plot_pairs(X, y, max_pairs=2)
plt.show()
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
    base_2d_rays=8,
    random_state=0,
).fit(X, y)

print(sh.interpretability_summary(diab.feature_names).head())
sh.plot_pairs(X, max_pairs=3)
plt.show()
```

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

### Plot samples

Running [examples/iris_visualization.py](examples/iris_visualization.py) generates
pairwise plots saved under `examples/images/` (e.g., `pair_0.png`, `pair_1.png`).

### Experiments and benchmark

The experiments comparing against **unsupervised** algorithms are located in
the [`experiments/`](experiments/) folder. The script
[`compare_unsupervised.py`](experiments/compare_unsupervised.py) evaluates five
different datasets, explores parameters of **SheShe**, **KMeans** and
**DBSCAN**, and stores four metrics (`ARI`, `homogeneity`, `completeness`,
`v_measure`) along with the execution time (`runtime_sec`). Each configuration
is executed multiple times with different random seeds, producing
`benchmark/unsupervised_results.csv` (raw runs) and
`benchmark/unsupervised_results_summary.csv` with averages and standard
deviations.

```bash
python experiments/compare_unsupervised.py
cat benchmark/unsupervised_results.csv | head
```

Results are generated inside `benchmark/`.

---

## Key parameters
- `base_2d_rays` → controls angular resolution in 2D (8 by default). 3D scales
  to ~26; d>3 uses subspaces.
- `direction` → "center_out" | "outside_in" to locate the inflection point.
- `scan_radius_factor`, `scan_steps` → size and resolution of the radial scan.
- `grad_*` → hyperparameters of gradient ascent (rate, iterations, tolerances).
- `max_subspaces` → max number of subspaces considered when d>3.

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

