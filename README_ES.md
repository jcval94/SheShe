# SheShe
**Smart High-dimensional Edge Segmentation & Hyperboundary Explorer**

SheShe convierte un modelo probabilístico en un explorador guiado de su propia superficie de decisión, revelando regiones interpretables directamente sobre la frontera del modelo.

```python
from sheshe import ModalBoundaryClustering
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
sh = ModalBoundaryClustering().fit(X, y)
regions = sh.predict_regions(X)
```

## Documentación

Los destacados, la instalación, los ejemplos y la referencia completa de la API se encuentran en la carpeta [docs](docs/index.html).

