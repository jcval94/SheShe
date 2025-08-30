import numpy as np
from sheshe import ModalBoundaryClustering
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KDTree
import sheshe.sheshe as sheshe_mod


def test_density_fallback(monkeypatch):
    # Force fallback path without hnswlib
    monkeypatch.setattr(sheshe_mod, "hnswlib", None)
    X = np.random.RandomState(0).randn(30, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    sh = ModalBoundaryClustering(
        base_estimator=LogisticRegression(max_iter=200),
        task="classification",
        density_alpha=0.5,
        density_k=3,
        random_state=0,
    )
    sh.fit(X, y)
    # Ensure KDTree is used as a lightweight fallback
    assert isinstance(sh._nn_density, KDTree)
    dens = sh._density(X[:1])
    assert dens.shape == (1,)
    assert np.isfinite(dens).all()
