import numpy as np
from sklearn.cluster import KMeans

from sheshe import ModalBoundaryClustering


class _F:
    def __call__(self, x):
        return -np.linalg.norm(x)

    def batch(self, X):
        return -np.linalg.norm(X, axis=1)


def test_choose_seeds_kmeans_dispersion():
    rng = np.random.RandomState(0)
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
    X = np.vstack([c + 0.1 * rng.randn(20, 2) for c in centers])
    sh = ModalBoundaryClustering(random_state=0)
    seeds = sh._choose_seeds(X, _F(), k=3)

    vals = _F().batch(X)
    best_idx = np.argmax(vals)
    remaining = np.delete(X, best_idx, axis=0)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(remaining)
    centroids = km.cluster_centers_
    cent_vals = _F().batch(centroids)
    order = np.argsort(-cent_vals)
    expected = np.vstack([X[best_idx], centroids[order]])
    assert np.allclose(seeds, expected)
