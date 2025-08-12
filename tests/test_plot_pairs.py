import pytest
from sklearn.datasets import load_iris
from sheshe import ModalBoundaryClustering


def test_plot_pairs_mismatched_y_length():
    X, y = load_iris(return_X_y=True)
    sh = ModalBoundaryClustering(random_state=0).fit(X, y)
    with pytest.raises(AssertionError, match="misma longitud"):
        sh.plot_pairs(X, y[:-1])
