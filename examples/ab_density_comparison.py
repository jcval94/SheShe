import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from sheshe import ModalBoundaryClustering


# -- Dataset ---------------------------------------------------------------
# A simple 2D synthetic dataset with 3 well separated clusters.
X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=0)

# Pre-compute dataset bounds for out-of-range checks
lo, hi = X.min(axis=0), X.max(axis=0)


def fit_model(alpha: float):
    """Train SheShe with a given density_alpha and return cluster centers."""
    model = ModalBoundaryClustering(
        density_alpha=alpha,
        density_k=15,
        random_state=0,
    )
    model.fit(X, y)
    centers = np.vstack([reg.center for reg in model.regions_])
    return centers


def centers_density(centers: np.ndarray) -> np.ndarray:
    """Estimate local density via 5-NN distances (inverse mean distance)."""
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    dists, _ = nn.kneighbors(centers)
    return 1.0 / (dists.mean(axis=1) + 1e-12)


def out_of_bounds(centers: np.ndarray) -> np.ndarray:
    """Boolean mask of centers that fall outside the original data bounds."""
    return np.any((centers < lo) | (centers > hi), axis=1)


# -- A/B configurations ----------------------------------------------------
centers_a = fit_model(alpha=0.0)  # A: no density penalty
centers_b = fit_model(alpha=1.0)  # B: density-aware

# Compute diagnostics
rho_a = centers_density(centers_a)
rho_b = centers_density(centers_b)

print("Baseline (density_alpha=0):")
print(f"  mean density: {rho_a.mean():.3f}")
print(f"  out-of-bounds centers: {out_of_bounds(centers_a).sum()}")

print("\nDensity-aware (density_alpha=1):")
print(f"  mean density: {rho_b.mean():.3f}")
print(f"  out-of-bounds centers: {out_of_bounds(centers_b).sum()}")
