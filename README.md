# SheShe
<div align="right"><a href="README_ES.md">ES</a></div>
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

SheShe turns probabilistic models into guided explorers of their decision surfaces, revealing human‑readable regions by following local maxima of class probability or predicted value.

## Features
- Supervised clustering for classification and regression
- Rule extraction and subspace exploration
- 2D/3D plotting utilities

## Mathematical Overview
SheShe approximates the optimisation problem <code>max_x f(x)</code> by climbing gradient-ascent paths toward local maxima and delineating neighbourhoods around them. Detailed derivations for each module are provided in the documentation.

## Installation
Requires Python ≥3.9. Install from [PyPI](https://pypi.org/project/sheshe/):

```bash
pip install sheshe
```

## Quickstart

```python
from sheshe import ModalBoundaryClustering
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target

sh = ModalBoundaryClustering(RandomForestClassifier(), task="classification").fit(X, y)
print(sh.predict(X[:5]))
```

## Documentation
See the [documentation](https://jcval94.github.io/SheShe/) for installation, API reference and guides.

## Author
SheShe is authored by José Carlos Del Valle – [LinkedIn](https://www.linkedin.com/in/jose-carlos-del-valle/) | [Portfolio](https://jcval94.github.io/Portfolio/)

## License
MIT
