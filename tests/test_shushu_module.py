import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
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


def test_shushu_multiclass_basic():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = _small_clusterer(random_state=0)
    sh.fit(X, y, feature_names=iris.feature_names)
    per_class_df, per_centroid_df = sh.summary_tables()
    sh.plot_classes(X, y, grid_res=20, max_paths=2, show_paths=False)
    plt.close("all")
    assert per_class_df.shape[0] == len(np.unique(y))
    assert set(per_class_df.columns).issuperset({"class_label", "n_clusters"})
    assert isinstance(per_centroid_df, type(per_class_df))

