# -*- coding: utf-8 -*-
"""CheChe: boundary computation for informative 2D subspaces.

This module intentionally mirrors the public API of :class:`ShuShu` so that
``CheChe`` can serve as a lightweight drop–in replacement in scenarios where we
only care about the geometric frontier induced by the data.  Most of the
attributes exposed by :class:`ShuShu` are provided here for compatibility even
though the underlying algorithm is much simpler.  Unlike :class:`ShuShu`,
``CheChe`` automatically selects a subset of promising 2D feature combinations
to evaluate, rather than exhaustively exploring all possible pairs.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple
from itertools import combinations

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - SciPy not available
    _HAS_SCIPY = False


class CheChe:
    """Compute convex hull frontiers for selected 2D subspaces."""

    def __init__(self) -> None:
        # Public attributes mimicking ``ShuShu`` for a familiar API ---------
        self.feature_names_: Optional[List[str]] = None
        self.clusterer_: Optional[object] = None  # ``CheChe`` has no clusterer
        self.per_class_: Dict[int, Dict] = {}
        self.classes_: Optional[np.ndarray] = None
        self.model_ = None
        self.mode_: Optional[str] = None
        self.score_fn_: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.regions_: List[Dict] = []
        self.deciles_: Optional[np.ndarray] = None

        # internal storage of frontiers and chosen pairs
        self.frontiers_: Dict[Tuple[int, int], np.ndarray] = {}
        self.pairs_: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    def _compute_frontiers(
        self, X: np.ndarray, pairs: Iterable[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Return frontiers for the provided 2D feature ``pairs`` in ``X``."""

        X = np.asarray(X, dtype=float)
        res: Dict[Tuple[int, int], np.ndarray] = {}
        for i, j in pairs:
            pts = X[:, [i, j]]
            if _HAS_SCIPY and pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts)
                    boundary = pts[hull.vertices]
                except Exception:  # pragma: no cover - degenerate input
                    boundary = None
            else:
                boundary = None

            if boundary is None:
                mins = pts.min(axis=0)
                maxs = pts.max(axis=0)
                boundary = np.array(
                    [
                        [mins[0], mins[1]],
                        [mins[0], maxs[1]],
                        [maxs[0], maxs[1]],
                        [maxs[0], mins[1]],
                    ]
                )
            res[(i, j)] = boundary
        return res

    # ------------------------------------------------------------------
    def _select_pairs(
        self, X: np.ndarray, max_pairs: Optional[int]
    ) -> List[Tuple[int, int]]:
        """Select the most informative 2D feature pairs.

        Pairs are ranked by the area of their bounding box and the top
        ``max_pairs`` with non-zero area are returned.  ``None`` means all
        pairs are considered.
        """

        d = X.shape[1]
        scored: List[Tuple[Tuple[int, int], float]] = []
        for i, j in combinations(range(d), 2):
            pts = X[:, [i, j]]
            area = np.prod(pts.max(axis=0) - pts.min(axis=0))
            scored.append(((i, j), float(area)))

        scored.sort(key=lambda x: x[1], reverse=True)
        pairs = [pair for pair, score in scored if score > 0]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        return pairs

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        score_model=None,
        score_fn_multi: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        score_fn_per_class: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        max_pairs: Optional[int] = 10,
    ) -> "CheChe":
        """Estimate frontiers for selected 2D combinations of features.

        The additional parameters mirror :meth:`ShuShu.fit` for API
        compatibility but are not used internally beyond being stored for
        reference.
        """

        X = np.asarray(X, dtype=float)
        self.feature_names_ = (
            feature_names if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        )
        self.score_fn_ = score_fn
        self.model_ = score_model
        self.frontiers_.clear()
        self.per_class_ = {}
        self.classes_ = None
        self.regions_ = []
        self.deciles_ = None

        self.pairs_ = self._select_pairs(X, max_pairs)

        if y is None:
            self.mode_ = "scalar"
            self.frontiers_ = self._compute_frontiers(X, self.pairs_)
            for cid, (dims, boundary) in enumerate(self.frontiers_.items()):
                self.regions_.append({"cluster_id": cid, "dims": dims, "frontier": boundary})
            return self

        # Determine whether we are in classification or regression mode
        y = np.asarray(y)
        unique = np.unique(y)
        is_classification = (
            np.issubdtype(y.dtype, np.integer) or unique.size <= 10
        )

        if is_classification:
            # Multiclass: compute frontiers per class ------------------
            self.mode_ = "multiclass"
            classes = unique
            self.classes_ = classes
            regions: List[Dict] = []
            per_class: Dict[int, Dict] = {}
            for ci, cls in enumerate(classes):
                subset = X[y == cls]
                fr = self._compute_frontiers(subset, self.pairs_)
                per_class[ci] = {"label": cls, "frontiers": fr}
                for dims, boundary in fr.items():
                    regions.append({
                        "cluster_id": len(regions),
                        "label": cls,
                        "dims": dims,
                        "frontier": boundary,
                    })
            self.per_class_ = per_class
            self.regions_ = regions
            return self

        # Regression: split into deciles and compute per-decile frontiers
        self.mode_ = "regression"
        n_dec = 10
        deciles = np.percentile(y, np.linspace(0, 100, n_dec + 1))
        self.deciles_ = deciles
        bins = deciles[1:-1]
        ids = np.digitize(y, bins, right=True)
        regions: List[Dict] = []
        per_class: Dict[int, Dict] = {}
        for k in range(n_dec):
            mask = ids == k
            if not np.any(mask):
                continue
            subset = X[mask]
            fr = self._compute_frontiers(subset, self.pairs_)
            label = (deciles[k], deciles[k + 1])
            per_class[k] = {"label": label, "frontiers": fr}
            for dims, boundary in fr.items():
                regions.append({
                    "cluster_id": len(regions),
                    "label": label,
                    "dims": dims,
                    "frontier": boundary,
                })
        self.per_class_ = per_class
        self.regions_ = regions
        self.classes_ = np.array(sorted(per_class.keys()))
        return self

    # ------------------------------------------------------------------
    def get_frontier(
        self, dims: Iterable[int], class_index: Optional[int] = None
    ) -> np.ndarray:
        """Return the frontier for a given pair of dimensions.

        Parameters
        ----------
        dims:
            Pair of feature indices.
        class_index:
            Index of the class or decile for which the frontier should be
            retrieved when the model was fit in multiclass or regression mode.
        """

        dims_t = tuple(dims)
        if self.mode_ in ("multiclass", "regression"):
            if class_index is None:
                raise ValueError("class_index required in multiclass mode")
            if class_index not in self.per_class_:
                raise KeyError(f"class_index {class_index} not found")
            fr = self.per_class_[class_index]["frontiers"]
            if dims_t not in fr:
                raise KeyError(f"Frontier for dims {dims_t} not available")
            return fr[dims_t]

        # scalar mode
        if dims_t not in self.frontiers_:
            raise KeyError(f"Frontier for dims {dims_t} not available")
        return self.frontiers_[dims_t]
