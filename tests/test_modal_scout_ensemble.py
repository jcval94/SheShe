import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sheshe import ModalScoutEnsemble


def test_modal_scout_ensemble_basic():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )
    yhat = mse.fit_predict(X, y)
    assert yhat.shape == y.shape
    proba = mse.predict_proba(X[:5])
    assert proba.shape[0] == 5
    assert np.isclose(mse.weights_.sum(), 1.0)
    report = mse.report()
    assert isinstance(report, list) and report


def test_modal_scout_ensemble_prediction_within_region():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
        prediction_within_region=True,
    )
    mse.fit(X, y)
    assert hasattr(mse, "labels_")
    assert hasattr(mse, "label2id_")
    assert mse.labels_.shape[0] == X.shape[0]
    labels, ids = mse.predict_regions(X)
    assert np.array_equal(labels, mse.labels_)
    assert np.array_equal(ids, mse.label2id_)
    far_point = np.array([[1000, 1000, 1000, 1000]])
    far_label, far_id = mse.predict_regions(far_point)
    assert far_label[0] == -1
    assert far_id[0] == -1


def test_modal_scout_ensemble_prediction_within_region_optional():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )
    yhat = mse.fit_predict(X, y)
    assert hasattr(mse, "labels_")
    assert np.array_equal(yhat, mse.labels_)
    assert np.array_equal(yhat, mse.predict(X))
    assert not hasattr(mse, "label2id_")


def test_modal_scout_ensemble_with_shushu():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        ensemble_method="shushu",
        shushu_kwargs={
            "k": 5,
            "rf_estimators": 5,
            "importance_sample_size": 60,
            "max_iter": 5,
        },
    )
    yhat = mse.fit_predict(X, y)
    assert yhat.shape == y.shape
    proba = mse.predict_proba(X[:5])
    assert proba.shape[0] == 5
    labels, ids = mse.predict_regions(X[:5])
    assert labels.shape == (5,)
    assert ids.shape == (5,)
