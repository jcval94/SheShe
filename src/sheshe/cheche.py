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

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Self
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score

try:  # pragma: no cover - optional dependency
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - SciPy not available
    _HAS_SCIPY = False


class CheChe:
    """Compute convex hull frontiers for selected 2D subspaces."""

    __artifact_version__ = "1.0"

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
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
        self._cache: Dict[str, Any] = {}

        # internal storage of frontiers and chosen pairs
        self.frontiers_: Dict[Tuple[int, int], np.ndarray] = {}
        self.pairs_: List[Tuple[int, int]] = []
        self.mapping_level_: Optional[int] = None

    # ------------------------------------------------------------------
    def _frontier_to_region(
        self,
        dims: Tuple[int, int],
        boundary: np.ndarray,
        cid: int,
        label: Optional[object] = None,
    ) -> Dict:
        """Convert a 2D frontier polygon into a region dictionary.

        The resulting structure mirrors the minimal requirements expected by
        :class:`RegionInterpreter`: a ``center`` in the 2D subspace, unit
        ``directions`` and their corresponding ``radii``.  Extra metadata such
        as ``dims`` and the original ``frontier`` are retained for downstream
        use.
        """

        center = boundary.mean(axis=0)
        vecs = boundary - center
        radii = np.linalg.norm(vecs, axis=1)
        directions = np.zeros_like(vecs)
        nonzero = radii > 0
        directions[nonzero] = vecs[nonzero] / radii[nonzero, None]

        region = {
            "cluster_id": cid,
            "dims": dims,
            "frontier": boundary,
            "center": center,
            "directions": directions,
            "radii": radii,
        }
        if label is not None:
            region["label"] = label
        return region

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
    def _subsample(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` downsampled according to ``mapping_level_``."""

        level = self.mapping_level_
        if level is None or level <= 1:
            return X
        return X[::level]

    # ------------------------------------------------------------------
    def fit(
        self,
        X: npt.NDArray[np.float_] | pd.DataFrame,
        y: Optional[npt.NDArray] = None,
        *,
        score_fn: Optional[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]] = None,
        feature_names: Optional[List[str]] = None,
        score_model=None,
        score_fn_multi: Optional[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]] = None,
        score_fn_per_class: Optional[List[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]]] = None,
        max_pairs: Optional[int] = 10,
        mapping_level: Optional[int] = None,
    ) -> Self:
        """Estimate frontiers for selected 2D combinations of features.

        The additional parameters mirror :meth:`ShuShu.fit` for API
        compatibility but are not used internally beyond being stored for
        reference.

        Parameters
        ----------
        mapping_level:
            Optional integer controlling the downsampling factor when
            computing frontiers. ``None`` or ``1`` uses all points, ``2`` uses
            every other point, ``3`` every third point, and so on.
        """

        X = np.asarray(X, dtype=float)
        self.mapping_level_ = mapping_level
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
            X_used = self._subsample(X)
            self.frontiers_ = self._compute_frontiers(X_used, self.pairs_)
            for cid, (dims, boundary) in enumerate(self.frontiers_.items()):
                self.regions_.append(
                    self._frontier_to_region(dims, boundary, cid)
                )
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
                subset = self._subsample(subset)
                fr = self._compute_frontiers(subset, self.pairs_)
                per_class[ci] = {"label": cls, "frontiers": fr}
                for dims, boundary in fr.items():
                    regions.append(
                        self._frontier_to_region(dims, boundary, len(regions), label=cls)
                    )
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
            subset = self._subsample(subset)
            fr = self._compute_frontiers(subset, self.pairs_)
            label = (deciles[k], deciles[k + 1])
            per_class[k] = {"label": label, "frontiers": fr}
            for dims, boundary in fr.items():
                regions.append(
                    self._frontier_to_region(dims, boundary, len(regions), label=label)
                )
        self.per_class_ = per_class
        self.regions_ = regions
        self.classes_ = np.array(sorted(per_class.keys()))
        return self

    # ------------------------------------------------------------------
    def fit_predict(
        self, X: npt.NDArray[np.float_] | pd.DataFrame, y: Optional[npt.NDArray] = None, **fit_kwargs
    ) -> npt.NDArray[np.int_]:
        """Fit the model and immediately return predictions for ``X``."""

        self.fit(X, y, **fit_kwargs)
        return self.predict(X)

    # ------------------------------------------------------------------
    def predict(self, X: npt.NDArray[np.float_] | pd.DataFrame) -> npt.NDArray[np.int_]:
        """Predict class labels or region ids for ``X``."""

        if self.mode_ is None:
            raise RuntimeError("Model not fitted")
        df = self.predict_regions(X)
        if self.mode_ == "scalar":
            return df["region_id"].to_numpy()
        return df["label"].to_numpy()

    # ------------------------------------------------------------------
    def predict_proba(self, X: npt.NDArray[np.float_] | pd.DataFrame) -> npt.NDArray[np.float_]:
        """Return class probabilities for ``X`` when in multiclass mode."""

        if self.mode_ != "multiclass" or self.classes_ is None:
            raise RuntimeError("predict_proba only available in multiclass mode")
        labels = self.predict(X)
        k = len(self.classes_)
        proba = np.zeros((len(labels), k), dtype=float)
        for j, cls in enumerate(self.classes_):
            proba[labels == cls, j] = 1.0
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return proba / row_sums

    # ------------------------------------------------------------------
    def predict_regions(
        self, X: npt.NDArray[np.float_] | pd.DataFrame
    ) -> pd.DataFrame:
        """Return labels and region ids for samples in ``X``."""

        if self.mode_ is None:
            raise RuntimeError("Model not fitted")

        from matplotlib.path import Path  # local import to keep dependency optional

        if isinstance(X, pd.DataFrame):
            index = X.index
            X_arr = X.to_numpy(dtype=float)
        else:
            index = None
            X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        labels = np.full(n, -1, dtype=object)
        cluster_ids = np.full(n, -1, dtype=int)

        for reg in self.regions_:
            dims = reg["dims"]
            boundary = reg["frontier"]
            path = Path(boundary)
            pts = X_arr[:, list(dims)]
            mask = path.contains_points(pts)
            if not np.any(mask):
                continue
            lab = reg.get("label", reg["cluster_id"])
            labels[mask] = lab
            cluster_ids[mask] = reg["cluster_id"]
        return pd.DataFrame({"label": labels.astype(int), "region_id": cluster_ids}, index=index)

    # ------------------------------------------------------------------
    def decision_function(
        self, X: npt.NDArray[np.float_] | pd.DataFrame
    ) -> npt.NDArray[np.float_]:
        """Return negative distances to region centers for ``X``.

        The output has shape ``(n_samples, n_regions)``.
        """

        if self.mode_ is None:
            raise RuntimeError("Model not fitted")

        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        m = len(self.regions_)
        if m == 0:
            return np.zeros((n, 0))

        scores = np.zeros((n, m), dtype=float)
        for idx, reg in enumerate(self.regions_):
            dims = reg["dims"]
            center = reg["center"]
            pts = X[:, list(dims)]
            dists = np.linalg.norm(pts - center[None, :], axis=1)
            scores[:, idx] = -dists
        return scores

    def transform(self, X: npt.NDArray[np.float_] | pd.DataFrame) -> npt.NDArray[np.float_]:
        """Feature transformation is not available for CheChe."""
        raise NotImplementedError("CheChe does not implement transform")

    def fit_transform(
        self,
        X: npt.NDArray[np.float_] | pd.DataFrame,
        y: Optional[npt.NDArray] = None,
        **fit_kwargs: Any,
    ) -> npt.NDArray[np.float_]:
        """Fit to data, then transform it.

        Notes
        -----
        ``CheChe`` lacks a transformation step and this method always raises
        ``NotImplementedError``.
        """
        raise NotImplementedError("CheChe does not implement fit_transform")

    def score(
        self,
        X: npt.NDArray[np.float_] | pd.DataFrame,
        y: npt.NDArray,
        *,
        metric: str | Callable[[npt.NDArray, npt.NDArray], float] = "auto",
        **metric_kwargs: Any,
    ) -> float:
        """Score predictions using ``metric``.

        ``metric='auto'`` defaults to accuracy.
        """
        if metric == "auto":
            metric_fn = accuracy_score
            y_pred = self.predict(X)
            return float(metric_fn(y, y_pred, **metric_kwargs))
        if isinstance(metric, str):
            from sklearn.metrics import get_scorer

            scorer = get_scorer(metric)
            return float(scorer(self, X, y))
        y_pred = self.predict(X)
        return float(metric(y, y_pred, **metric_kwargs))

    def save(self, filepath: str | Path) -> None:
        """Persist the estimator with joblib including version metadata."""
        payload = {"__artifact_version__": self.__artifact_version__, "model": self}
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str | Path) -> "CheChe":
        """Load an estimator saved with :meth:`save`."""
        payload = joblib.load(filepath)
        ver = payload.get("__artifact_version__")
        if ver != cls.__artifact_version__:
            raise ValueError(
                f"Artifact version mismatch: expected {cls.__artifact_version__}, got {ver}"
            )
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("Loaded object is not a CheChe instance")
        return model

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

    # ------------------------------------------------------------------
    def plot_pairs(
        self,
        X: np.ndarray,
        *,
        class_index: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        """Plot the frontiers discovered by this instance on ``X``.

        Parameters
        ----------
        X:
            Original data used for fitting. Only the dimensions present in the
            selected pairs are accessed.
        class_index:
            When the model was trained in multiclass or regression mode, this
            selects which class/decile frontier to plot.
        feature_names:
            Optional names for the features. If ``None``, ``self.feature_names_``
            is used.

        Returns
        -------
        (fig, axes):
            A tuple containing the created :class:`matplotlib.figure.Figure` and
            the list of :class:`matplotlib.axes.Axes` objects.
        """

        import matplotlib.pyplot as plt  # local import to keep optional dependency

        if feature_names is None:
            feature_names = self.feature_names_

        pairs = list(self.pairs_)
        if not pairs:
            raise ValueError("CheChe instance has no stored pairs. Did you call fit()?")

        n = len(pairs)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (i, j) in zip(axes, pairs):
            pts = X[:, [i, j]]
            ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, label="data")
            frontier = self.get_frontier((i, j), class_index=class_index)
            frontier = np.concatenate([frontier, frontier[:1]], axis=0)
            ax.plot(frontier[:, 0], frontier[:, 1], color="red", label="frontier")
            if feature_names is not None and len(feature_names) > max(i, j):
                ax.set_xlabel(feature_names[i])
                ax.set_ylabel(feature_names[j])
            ax.legend()

        fig.tight_layout()
        return fig, axes
