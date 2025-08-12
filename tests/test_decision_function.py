import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sheshe import ModalBoundaryClustering


def test_decision_function_uses_estimator_if_available():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(base_estimator=LogisticRegression(max_iter=200), task="classification")
    sh.fit(X, y)
    Xs = sh.scaler_.transform(X)
    expected = sh.estimator_.decision_function(Xs)
    np.testing.assert_allclose(sh.decision_function(X), expected)


def test_decision_function_fallback_predict_proba():
    iris = load_iris()
    X, y = iris.data, iris.target
    sh = ModalBoundaryClustering(base_estimator=RandomForestClassifier(n_estimators=10, random_state=0),
                                 task="classification", random_state=0)
    sh.fit(X, y)
    Xs = sh.scaler_.transform(X)
    expected = sh.estimator_.predict_proba(Xs)
    np.testing.assert_allclose(sh.decision_function(X), expected)


def test_decision_function_fallback_predict_regression():
    data = load_diabetes()
    X, y = data.data, data.target
    sh = ModalBoundaryClustering(base_estimator=RandomForestRegressor(n_estimators=10, random_state=0),
                                 task="regression", random_state=0)
    sh.fit(X, y)
    Xs = sh.scaler_.transform(X)
    expected = sh.estimator_.predict(Xs)
    np.testing.assert_allclose(sh.decision_function(X), expected)
