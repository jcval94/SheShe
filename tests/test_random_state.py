import numpy as np
from sklearn.datasets import make_blobs

from sheshe import ModalBoundaryClustering


def test_fit_predict_reproducible():
    X, _ = make_blobs(n_samples=30, centers=3, random_state=0, cluster_std=0.5)
    mbc1 = ModalBoundaryClustering(task="regression", random_state=123, n_max_seeds=1, scan_steps=8)
    mbc2 = ModalBoundaryClustering(task="regression", random_state=123, n_max_seeds=1, scan_steps=8)
    y1 = mbc1.fit_predict(X)
    y2 = mbc2.fit_predict(X)
    assert np.array_equal(y1, y2)
