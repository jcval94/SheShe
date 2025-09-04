import numpy as np
from sklearn.datasets import load_iris

from sheshe import CheChe


def test_transform_negative_distances():
    X, y = load_iris(return_X_y=True)
    ch = CheChe().fit(X, y)
    T = ch.transform(X)
    assert T.shape == (X.shape[0], len(ch.regions_))
    for idx, reg in enumerate(ch.regions_):
        dims = reg["dims"]
        center = reg["center"]
        dists = -np.linalg.norm(X[:, list(dims)] - center, axis=1)
        assert np.allclose(T[:, idx], dists)


def test_fit_transform_equivalent():
    X, y = load_iris(return_X_y=True)
    ch1 = CheChe()
    D1 = ch1.fit_transform(X, y)
    ch2 = CheChe().fit(X, y)
    D2 = ch2.transform(X)
    assert np.allclose(D1, D2)
