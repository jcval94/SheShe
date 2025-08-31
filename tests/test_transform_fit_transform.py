import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sheshe.sheshe import ModalBoundaryClustering


def test_transform_membership():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        ray_mode="grid",
    )
    sh.fit(X, y)
    T = sh.transform(X)
    assert T.shape == (X.shape[0], len(sh.regions_))
    M = sh._membership_matrix(X)
    assert np.array_equal((T >= 0).astype(int), M)


def test_fit_transform_equivalent():
    X, y = load_iris(return_X_y=True)
    params = dict(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        ray_mode="grid",
    )
    sh1 = ModalBoundaryClustering(**params)
    D1 = sh1.fit_transform(X, y)
    sh2 = ModalBoundaryClustering(**params)
    sh2.fit(X, y)
    D2 = sh2.transform(X)
    assert np.allclose(D1, D2)
