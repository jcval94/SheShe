# tests/test_basic.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
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
