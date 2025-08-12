# tests/test_basic.py
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
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
    assert {"Tipo","Distancia","Categoria"}.issubset(df.columns)
    score = sh.score(X, y)
    assert 0.0 <= score <= 1.0


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
