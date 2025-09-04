import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from sheshe import ShuShu


def _small_clusterer(random_state: int = 0) -> ShuShu:
    return ShuShu(
        k=5,
        rf_estimators=5,
        importance_sample_size=60,
        max_iter=5,
        random_state=random_state,
    )


def test_shushu_clusterer_basic():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    score_fn = lambda Z: model.predict_proba(Z)[:, 0]
    cl = _small_clusterer(random_state=0)
    cl.fit(X, score_fn=score_fn)
    assert cl.centroids_.shape[1] == X.shape[1]
    assert isinstance(cl.clusters_, list)
    assert hasattr(cl, "regions_")
    assert isinstance(cl.regions_, list)
    labels = cl.predict(X[:3])
    assert labels.shape == (3,)
    df2 = cl.predict_regions(X[:3])
    assert np.array_equal(df2["label"].to_numpy(), df2["region_id"].to_numpy())


def test_shushu_multiclass_basic():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = _small_clusterer(random_state=0)
    sh.fit(X, y, feature_names=iris.feature_names)
    per_class_df, per_centroid_df = sh.summary_tables()
    sh.plot_classes(X, y, grid_res=20, max_paths=2, show_paths=False)
    titles = [plt.figure(n).axes[0].get_title() for n in plt.get_fignums()]
    for cid in sh.per_class_.keys():
        assert any(str(cid) in t for t in titles)
    plt.close("all")
    assert per_class_df.shape[0] == len(np.unique(y))
    assert set(per_class_df.columns).issuperset({"class_label", "n_clusters"})
    assert isinstance(per_centroid_df, type(per_class_df))
    yhat = sh.predict(X)
    assert yhat.shape == y.shape
    proba = sh.predict_proba(X[:5])
    assert np.allclose(proba.sum(axis=1), 1.0)
    df = sh.predict_regions(X[:5])
    assert df.shape == (5, 2)
    assert hasattr(sh, "regions_")
    assert isinstance(sh.regions_, list)


def test_transform_and_fit_transform():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh1 = _small_clusterer(random_state=0)
    T1 = sh1.fit_transform(X, y)
    sh2 = _small_clusterer(random_state=0)
    sh2.fit(X, y)
    T2 = sh2.transform(X)
    assert np.allclose(T1, T2)
    assert np.allclose(T1.sum(axis=1), 1.0)


def test_get_cluster_and_score():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = _small_clusterer(random_state=0)
    sh.fit(X, y)
    info = sh.get_cluster(0, with_geometry=True)
    assert set(info.keys()) == {"id", "label_mode", "support", "feature_stats", "geometry"}
    assert info["support"] > 0
    acc = sh.score(X, y)
    assert acc == pytest.approx(accuracy_score(y, sh.predict(X)))
    f1 = sh.score(X, y, metric="f1_macro")
    assert f1 == pytest.approx(f1_score(y, sh.predict(X), average="macro"))

