# tests/test_basic.py
import numpy as np
import pytest
import warnings
from sklearn.datasets import load_iris, make_regression, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sheshe import ModalBoundaryClustering

def test_import_and_fit():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(base_estimator=LogisticRegression(max_iter=200), task="classification", random_state=0)
    sh.fit(X, y)
    y_hat = sh.predict(X)
    assert y_hat.shape[0] == X.shape[0]
    proba = sh.predict_proba(X[:3])
    assert proba.shape[0] == 3
    df = sh.interpretability_summary(iris.feature_names)
    assert {"Type","Distance","Category","ClusterID"}.issubset(df.columns)
    score = sh.score(X, y)
    assert 0.0 <= score <= 1.0


def test_labels_attribute_after_fit():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        ray_mode="grid",
    )
    sh.fit(X, y)
    assert hasattr(sh, "labels_")
    assert sh.labels_.shape[0] == X.shape[0]
    assert np.array_equal(sh.labels_, sh.predict(X))


def test_fit_predict_sets_labels_attribute():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        ray_mode="grid",
    )
    labels = sh.fit_predict(X, y)
    assert hasattr(sh, "labels_")
    assert np.array_equal(labels, sh.predict(X))
    assert np.array_equal(sh.labels_, labels)


def test_prediction_within_region_labels():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        prediction_within_region=True,
    )
    sh.fit(X, y)
    assert hasattr(sh, "labels_")
    assert sh.labels_.shape[0] == X.shape[0]
    expected = sh.predict_regions(X)
    assert np.array_equal(sh.labels_, expected)
    far_point = np.array([[1000, 1000, 1000, 1000]])
    far_label = sh.predict_regions(far_point)[0]
    assert far_label == -1


def test_prediction_within_region_optional():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        prediction_within_region=False,
    )
    sh.fit(X, y)
    assert hasattr(sh, "labels_")
    assert np.array_equal(sh.labels_, sh.predict(X))


def test_score_regression():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=0)
    sh = ModalBoundaryClustering(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        task="regression",
        random_state=0,
    )
    sh.fit(X, y)
    score = sh.score(X, y)
    assert np.isfinite(score)


def test_cluster_region_metrics_classification():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        ray_mode="grid",
    )
    sh.fit(X, y)
    reg = sh.regions_[0]
    assert reg.score is not None
    assert {"precision", "recall", "f1"}.issubset(reg.metrics.keys())


def test_cluster_region_metrics_regression():
    X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=0)
    sh = ModalBoundaryClustering(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        task="regression",
        random_state=0,
    )
    sh.fit(X, y)
    reg = sh.regions_[0]
    assert reg.score is not None
    assert {"mse", "mae"}.issubset(reg.metrics.keys())


def test_predict_regression_returns_base_estimator_value():
    X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=0)
    sh = ModalBoundaryClustering(
        base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        task="regression",
        random_state=0,
    )
    sh.fit(X, y)
    expected = sh.pipeline_.predict(X[:5])
    y_hat = sh.predict(X[:5])
    assert np.allclose(y_hat, expected)


def test_smooth_window_param():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        smooth_window=5,
    )
    sh.fit(X, y)
    assert sh.smooth_window == 5


def test_decision_function_classifier_and_fallback():
    iris = load_iris()
    X, y = iris.data, iris.target
    # Estimador con decision_function
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
    )
    sh.fit(X, y)
    Xs = sh.scaler_.transform(X[:5])
    expected = sh.estimator_.decision_function(Xs)
    df_scores = sh.decision_function(X[:5])
    assert np.allclose(df_scores, expected)

    # Estimador sin decision_function → usa predict_proba
    sh2 = ModalBoundaryClustering(
        base_estimator=RandomForestClassifier(n_estimators=10, random_state=0),
        task="classification",
        random_state=0,
    )
    sh2.fit(X, y)
    Xs2 = sh2.scaler_.transform(X[:5])
    expected2 = sh2.estimator_.predict_proba(Xs2)
    df_scores2 = sh2.decision_function(X[:5])
    assert np.allclose(df_scores2, expected2)


def test_decision_function_regression_fallback():
    X, y = make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)
    sh = ModalBoundaryClustering(
        base_estimator=RandomForestRegressor(n_estimators=5, random_state=0),
        task="regression",
        random_state=0,
    )
    sh.fit(X, y)
    Xs = sh.scaler_.transform(X[:5])
    expected = sh.estimator_.predict(Xs)
    df_scores = sh.decision_function(X[:5])
    assert np.allclose(df_scores, expected)


def test_predict_proba_and_value_without_predict_proba():
    X, y = make_classification(n_samples=40, n_features=5, random_state=0)
    sh = ModalBoundaryClustering(
        base_estimator=SVC(kernel="linear"),
        task="classification",
        random_state=0,
    )
    sh.fit(X, y)
    Xs = sh.scaler_.transform(X[:5])
    scores = sh.estimator_.decision_function(Xs)
    proba = sh.predict_proba(X[:5])
    expected = np.column_stack([-scores.reshape(-1, 1), scores.reshape(-1, 1)])
    assert np.allclose(proba, expected)
    val1 = sh._predict_value_real(X[:5], class_idx=1)
    val0 = sh._predict_value_real(X[:5], class_idx=0)
    assert np.allclose(val1, scores)
    assert np.allclose(val0, -scores)


def test_scan_steps_minimum():
    with pytest.raises(ValueError):
        ModalBoundaryClustering(scan_steps=1)


def test_membership_matrix_no_directions():
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        base_2d_rays=0,
        random_state=0,
        ray_mode="grid",
    )
    sh.fit(X, y)
    assert all(reg.directions.size > 0 for reg in sh.regions_)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("error")
        M = sh._membership_matrix(X)
    assert len(record) == 0
    assert M.shape == (X.shape[0], len(sh.regions_))
    assert np.any(M != 0)
    pred = sh.predict(X)
    assert pred.shape == (X.shape[0],)


def test_base_2d_rays_zero_raises_in_3d():
    iris = load_iris()
    X, y = iris.data[:, :3], iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        base_2d_rays=0,
        random_state=0,
    )
    with pytest.raises(ValueError, match="base_2d"):
        sh.fit(X, y)


def test_n_max_seeds_minimum():
    with pytest.raises(ValueError, match="n_max_seeds"):
        ModalBoundaryClustering(n_max_seeds=0)


def test_fit_raises_when_y_none_classification():
    X = np.random.randn(10, 3)
    sh = ModalBoundaryClustering(task="classification")
    with pytest.raises(ValueError, match="y cannot be None"):
        sh.fit(X, None)


def test_fit_raises_with_single_class():
    X = np.random.randn(10, 3)
    y = np.zeros(10)
    sh = ModalBoundaryClustering(task="classification")
    with pytest.raises(ValueError, match="at least two classes"):
        sh.fit(X, y)


def test_drop_fraction_changes_radii():
    center = np.array([0.0])
    dirs = np.array([[1.0]])
    X_std = np.array([1.0])

    def f(point):
        return np.exp(-point[0])

    lo = np.array([0.0])
    hi = np.array([10.0])

    sh1 = ModalBoundaryClustering(drop_fraction=0.5)
    sh1.bounds_ = (lo, hi)
    r1, _, _ = sh1._scan_radii(center, f, dirs, X_std)

    sh2 = ModalBoundaryClustering(drop_fraction=0.2)
    sh2.bounds_ = (lo, hi)
    r2, _, _ = sh2._scan_radii(center, f, dirs, X_std)

    assert r2[0] > r1[0]


def test_percentile_stop_criteria():
    center = np.array([0.0])
    dirs = np.array([[1.0]])
    X_std = np.array([1.0])

    def f(point):
        return 0.94 - 0.02 * point[0]

    def f_batch(X):
        return 0.94 - 0.02 * X[:, 0]

    f.batch = f_batch  # type: ignore[attr-defined]

    lo = np.array([0.0])
    hi = np.array([10.0])

    percentiles = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.9, 0.92, 0.93, 0.94, 0.94, 1.0])

    sh = ModalBoundaryClustering(scan_steps=3, scan_radius_factor=2.0, stop_criteria="percentile")
    sh.bounds_ = (lo, hi)
    r, _, _ = sh._scan_radii(center, f, dirs, X_std, percentiles=percentiles)
    assert np.isclose(r[0], 1.0)


def test_get_cluster_by_id():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
    )
    sh.fit(X, y)
    first = sh.regions_[0]
    found = sh.get_cluster(first.cluster_id)
    assert found is first
    assert sh.get_cluster(-1) is None
