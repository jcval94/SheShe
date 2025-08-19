import matplotlib
matplotlib.use('Agg')

import pytest
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering


def test_plot_pair_3d_runs():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0])


def test_plot_pair_3d_plotly_runs():
    pytest.importorskip("plotly")
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    fig = sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0], engine="plotly")
    assert fig is not None


def test_plot_pair_3d_bad_engine():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        sh.plot_pair_3d(X, (0, 1), class_label=sh.classes_[0], engine="unknown")
