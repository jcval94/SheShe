import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import load_iris
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
