"""Example comparing approximate areas of regions discovered by SheShe.

This script generates a simple two-dimensional classification dataset and fits
``ModalBoundaryClustering``.  For each discovered region the area of the
polygon defined by the boundary points is computed and compared against the
convex hull area of the samples belonging to that class.

Run with::

    PYTHONPATH=src python examples/compare_region_areas.py
"""

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.datasets import make_blobs

from sheshe import ModalBoundaryClustering


def _polygon_area(points: np.ndarray) -> float:
    """Return the area of a 2D polygon given ordered vertices."""
    if len(points) < 3:
        return 0.0
    x, y = points.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _hull_area(points: np.ndarray) -> float:
    """Area of the convex hull for a set of 2D points."""
    if len(points) < 3:
        return 0.0
    hull = ConvexHull(points)
    return float(hull.volume)


def main() -> None:
    # Synthetic 2D dataset with three classes
    X, y = make_blobs(
        n_samples=500,
        centers=[(-2, -2), (0, 2), (2, -1)],
        cluster_std=[0.8, 0.6, 0.7],
        random_state=0,
    )

    sh = ModalBoundaryClustering(task="classification", base_2d_rays=32, random_state=0)
    sh.fit(X, y)

    print("Approximate region areas (vs. convex hull of class samples):")
    for region in sh.regions_:
        boundary = region.center[None, :] + region.directions * region.radii[:, None]
        area = _polygon_area(boundary)
        true_area = _hull_area(X[y == region.label])
        print(
            f"Region {region.cluster_id} (label={region.label}): "
            f"area ≈ {area:.3f}, hull ≈ {true_area:.3f}"
        )


if __name__ == "__main__":
    main()
