import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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
    df = mse.predict_regions(X)
    assert np.array_equal(df["label"].to_numpy(), mse.labels_)
    assert np.array_equal(df["region_id"].to_numpy(), mse.label2id_)
    far_point = np.array([[1000, 1000, 1000, 1000]])
    far_df = mse.predict_regions(far_point)
    assert far_df.iloc[0]["label"] == -1
    assert far_df.iloc[0]["region_id"] == -1


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
    df = mse.predict_regions(X[:5])
    assert df.shape == (5, 2)


def test_modal_scout_ensemble_decision_function():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )
    mse.fit(X, y)
    scores = mse.decision_function(X[:5])
    assert scores.shape == (5, len(mse.classes_))
    pred_from_scores = mse.classes_[np.argmax(scores, axis=1)]
    assert np.array_equal(pred_from_scores, mse.predict(X[:5]))


def test_modal_scout_ensemble_interpretability_summary():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )
    mse.fit(X, y)
    summary = mse.interpretability_summary(X, y, per_model=True)
    assert isinstance(summary, pd.DataFrame)
    # aggregate row is last
    assert summary.iloc[-1]["model_id"] == "aggregate"
    # weights align with report
    rep = mse.report()
    np.testing.assert_allclose(summary.iloc[:-1]["weight"].to_numpy(), [r["weight"] for r in rep])


def test_modal_scout_ensemble_save_load(tmp_path):
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )
    mse.fit(X, y)
    y_pred = mse.predict(X)
    path = tmp_path / "mse.pkl"
    mse.save(path)
    mse_loaded = ModalScoutEnsemble.load(path)
    y_pred_loaded = mse_loaded.predict(X)
    assert np.array_equal(y_pred, y_pred_loaded)


def test_modal_scout_ensemble_score():
    data = load_iris()
    X, y = data.data, data.target
    mse = ModalScoutEnsemble(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        random_state=0,
        scout_kwargs={"max_order": 2, "top_m": 4, "sample_size": None},
        cv=2,
    )
    mse.fit(X, y)
    expected = accuracy_score(y, mse.predict(X))
    assert np.isclose(mse.score(X, y), expected)
    # metric callable
    assert np.isclose(
        mse.score(X, y, metric=balanced_accuracy_score),
        balanced_accuracy_score(y, mse.predict(X)),
    )
