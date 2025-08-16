import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering


def test_save_and_load(tmp_path: Path):
    data = load_iris()
    X, y = data.data, data.target
    model = ModalBoundaryClustering(random_state=0).fit(X, y)
    path = tmp_path / "mbc.joblib"
    model.save(path)
    assert path.exists()

    loaded = ModalBoundaryClustering.load(path)
    assert np.allclose(model.predict(X), loaded.predict(X))
