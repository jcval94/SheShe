import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sheshe import ShuShu


def _small_shushu(random_state: int = 0) -> ShuShu:
    return ShuShu(
        k=5,
        rf_estimators=5,
        importance_sample_size=60,
        max_iter=5,
        random_state=random_state,
    )


def test_shushu_save_load_scalar(tmp_path: Path):
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    score_fn = lambda Z: model.predict_proba(Z)[:, 0]
    sh = _small_shushu(random_state=0)
    sh.fit(X, score_fn=score_fn)

    path = tmp_path / "shushu_scalar.joblib"
    sh.save(path)
    assert path.exists()

    loaded = ShuShu.load(path)
    assert np.array_equal(sh.predict(X[:10]), loaded.predict(X[:10]))
    assert np.allclose(sh.centroids_, loaded.centroids_)


def test_shushu_save_load_multiclass(tmp_path: Path):
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = _small_shushu(random_state=0)
    sh.fit(X, y, feature_names=iris.feature_names)

    path = tmp_path / "shushu_multi.joblib"
    sh.save(path)
    loaded = ShuShu.load(path)

    assert np.array_equal(sh.predict(X[:15]), loaded.predict(X[:15]))
    assert np.allclose(sh.predict_proba(X[:15]), loaded.predict_proba(X[:15]))
    for ci in sh.per_class_.keys():
        c1 = sh.per_class_[ci]["clusterer"]
        c2 = loaded.per_class_[ci]["clusterer"]
        if c1.centroids_ is not None and c1.centroids_.size > 0:
            assert np.allclose(c1.centroids_, c2.centroids_)
