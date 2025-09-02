# SheShe
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

SheShe transforms a probabilistic model into a guided explorer of its own decision landscape, uncovering human‑readable regions directly on the model's boundary.

```python
from sheshe import ModalBoundaryClustering
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
regions = sh.predict_regions(X)
```

## Documentation

Full highlights, installation instructions, examples and API reference are available in the [docs](docs/index.html) folder.

