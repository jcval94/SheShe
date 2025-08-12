# tests/test_save_load.py
import numpy as np
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering


def test_save_and_load(tmp_path):
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    path = tmp_path / "model.joblib"
    sh.save(path)
    sh2 = ModalBoundaryClustering.load(path)
    assert np.all(sh.predict(X[:10]) == sh2.predict(X[:10]))
