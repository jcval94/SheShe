import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_iris
from sklearn.datasets import make_regression
from sheshe import ModalBoundaryClustering


def test_plot_pairs_mismatched_y_length():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    with pytest.raises(AssertionError, match="misma longitud"):
        sh.plot_pairs(X, y[:-1])


def test_plot_pairs_feature_names():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    names = ['a', 'b', 'c', 'd']
    sh.plot_pairs(X, y, max_pairs=1, feature_names=names)
    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == 'a'
    assert ax.get_ylabel() == 'b'
    plt.close('all')


def test_plot_pairs_max_classes():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    sh.plot_pairs(X, y, max_pairs=1, max_classes=1)
    assert len(plt.get_fignums()) == 1
    plt.close('all')


def test_plot_pairs_regression_deciles():
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    sh = ModalBoundaryClustering(task="regression", random_state=0).fit(X, y)
    sh.plot_pairs(X, y, max_pairs=1, max_classes=5)
    assert len(plt.get_fignums()) == 1
    plt.close('all')


def test_plot_pairs_show_centroids_toggle():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    sh.plot_pairs(X, y, max_pairs=1, max_classes=1)
    ax = plt.gcf().axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("centro" in txt for txt in texts)
    plt.close('all')

    sh.plot_pairs(X, y, max_pairs=1, max_classes=1, show_centroids=False)
    ax = plt.gcf().axes[0]
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert all("centro" not in txt for txt in texts)
    plt.close('all')

