import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sheshe import ShuShu, CheChe, ModalScoutEnsemble


def test_shushu_save_load_score(tmp_path):
    data = load_iris()
    X, y = data.data, data.target
    sh = ShuShu(random_state=0).fit(X, y)
    df = sh.predict_regions(X[:3])
    assert set(df.columns) == {"label", "region_id"}
    assert isinstance(sh.score(X[:3], y[:3]), float)
    T = sh.transform(X[:3])
    assert T.shape[0] == 3
    assert np.allclose(T.sum(axis=1), 1.0)
    path = tmp_path / "sh.joblib"
    sh.save(path)
    loaded = ShuShu.load(path)
    assert isinstance(loaded, ShuShu)


def test_cheche_save_load_score(tmp_path):
    data = load_iris()
    X, y = data.data, data.target
    ch = CheChe().fit(X, y)
    df = ch.predict_regions(X[:2])
    assert set(df.columns) == {"label", "region_id"}
    assert isinstance(ch.score(X[:2], y[:2]), float)
    T = ch.transform(X[:2])
    assert T.shape == (2, len(ch.regions_))
    path = tmp_path / "ch.joblib"
    ch.save(path)
    loaded = CheChe.load(path)
    assert isinstance(loaded, CheChe)


def test_mse_save_load_score(tmp_path):
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    ).fit(X, y)
    df = mse.predict_regions(X[:5])
    assert set(df.columns) == {"label", "region_id"}
    assert isinstance(mse.score(X[:5], y[:5]), float)
    with pytest.raises(NotImplementedError):
        mse.transform(X[:5])
    path = tmp_path / "mse.joblib"
    mse.save(path)
    loaded = ModalScoutEnsemble.load(path)
    assert isinstance(loaded, ModalScoutEnsemble)
