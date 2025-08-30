import numpy as np
from sheshe import ModalBoundaryClustering

def test_bounds_margin_optional():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    sh = ModalBoundaryClustering(bounds_margin=0.0)
    lo, hi = sh._bounds_from_data(X)
    assert np.allclose(lo, [0.0, 0.0])
    assert np.allclose(hi, [1.0, 1.0])
    sh2 = ModalBoundaryClustering(bounds_margin=0.1)
    lo2, hi2 = sh2._bounds_from_data(X)
    assert np.all(lo2 < 0.0)
    assert np.all(hi2 > 1.0)
