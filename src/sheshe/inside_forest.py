"""InsideForest: interpretable supervised clustering from random forests.

This module implements the :class:`InsideForest` estimator described in the
user specification.  The implementation focuses on providing a practical and
reasonably efficient reference that follows the proposed API and data model.

The algorithm extracts leaf rules from a fitted
``RandomForestClassifier``.  The rules are encoded as axis-aligned
hyper-rectangles which are clustered to form human-interpretable regions.
These regions expose aggregate metrics such as support, purity and lift.

The implementation avoids heavy third-party dependencies and only relies on
NumPy and scikit-learn which are already part of the repository's dependency
stack.  Computation intensive parts (e.g. pairwise Jaccard distances) are
implemented with NumPy vectorisation and block-wise loops to limit memory
usage.  A more optimised backend (Numba, blocks, etc.) can be added later
without altering the public API.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


Number = Union[int, float]


@dataclass
class Rule:
    """Representation of a single leaf rule."""

    id: int
    tree_id: int
    leaf_id: int
    a: np.ndarray  # lower bounds (float32)
    b: np.ndarray  # upper bounds (float32)
    mask: np.ndarray  # bool mask indicating which dimensions are active
    support: int
    class_counts: np.ndarray  # int32 counts per class
    distribution: np.ndarray  # float64 distribution per class
    weight: float
    gini: float


@dataclass
class Region:
    """Aggregated region composed of multiple rules."""

    id: int
    member_rule_ids: List[int] = field(default_factory=list)
    a: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    support: int = 0
    class_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    distribution: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    purity: float = 0.0
    lift: float = 0.0
    algo_info: Dict[str, Any] = field(default_factory=dict)


class InsideForest(BaseEstimator, TransformerMixin):
    """Interpretable supervised clustering built on top of random forests."""

    # ------------------------------------------------------------------
    # Construction & configuration
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = self._build_config(config or {})
        self.class_names = list(class_names) if class_names is not None else None
        self._label_encoder: Optional[LabelEncoder] = None
        self._rf: Optional[RandomForestClassifier] = None
        self._rules: List[Rule] = []
        self._regions: List[Region] = []
        self._feature_names: Optional[List[str]] = None
        self._fitted: bool = False
        self.global_mins_: Optional[np.ndarray] = None
        self.global_maxs_: Optional[np.ndarray] = None
        self.global_class_counts_: Optional[np.ndarray] = None
        self.global_class_distribution_: Optional[np.ndarray] = None
        self._region_arrays_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            None
        )
        self._rf_source: str = "internal"
        self._last_timings: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[Number]]],
        y: Optional[Union[np.ndarray, Sequence[Number]]],
        feature_names: Optional[Sequence[str]] = None,
        random_forest: Optional[RandomForestClassifier] = None,
        verbose: int = 0,
    ) -> "InsideForest":
        """Fit the estimator, extract rules and construct regions."""

        timings: Dict[str, float] = {}

        with self._time_block(timings, "fit", "coerce_features", verbose):
            X_arr = self._coerce_features(X)
            n_samples, n_features = X_arr.shape

            if feature_names is not None:
                if len(feature_names) != n_features:
                    raise ValueError("feature_names length must match number of features")
                self._feature_names = list(feature_names)
            else:
                self._feature_names = [f"x{i}" for i in range(n_features)]

            self.global_mins_ = X_arr.min(axis=0).astype(np.float32, copy=True)
            self.global_maxs_ = X_arr.max(axis=0).astype(np.float32, copy=True)

        if random_forest is not None:
            if y is None:
                raise ValueError(
                    "y must be provided when supplying a pretrained random forest"
                )
            if not isinstance(random_forest, RandomForestClassifier):
                raise TypeError(
                    "random_forest must be an instance of RandomForestClassifier"
                )
            if not hasattr(random_forest, "estimators_"):
                raise ValueError("Provided random forest must be fitted beforehand")
            if not hasattr(random_forest, "classes_"):
                raise ValueError(
                    "Provided random forest must expose classes_; ensure it is fitted"
                )
            if getattr(random_forest, "n_features_in_", n_features) != n_features:
                raise ValueError(
                    "random_forest expects a different number of features than X"
                )
            rf = random_forest
            self._rf_source = "external"
        else:
            if y is None:
                raise ValueError("y must be provided when training the internal forest")
            rf = self._make_random_forest()
            self._rf_source = "internal"

        with self._time_block(timings, "fit", "encode_target", verbose):
            y_arr = self._coerce_target(
                y, rf.classes_ if random_forest is not None else None
            )

        if random_forest is None:
            with self._time_block(timings, "fit", "train_random_forest", verbose):
                rf.fit(X_arr, y_arr)
        else:
            self._record_duration(
                timings,
                "fit",
                "train_random_forest",
                0.0,
                verbose,
                note="external forest reused",
            )
        self._rf = rf

        with self._time_block(timings, "fit", "compute_class_stats", verbose):
            encoder = self._label_encoder
            if encoder is None:
                raise RuntimeError("Label encoder not initialised")
            n_classes = len(encoder.classes_)
            self.global_class_counts_ = np.bincount(y_arr, minlength=n_classes)
            total = int(self.global_class_counts_.sum())
            if total > 0:
                self.global_class_distribution_ = self.global_class_counts_ / total
            else:
                self.global_class_distribution_ = np.zeros(n_classes, dtype=float)

        with self._time_block(timings, "fit", "extract_rules", verbose):
            rules = self._extract_rules(rf, X_arr, y_arr)
            self._rules = rules

        if not rules:
            # No rules survived the filtering; clear regions and exit early.
            self._regions = []
            self._record_duration(
                timings,
                "fit",
                "cluster_total",
                0.0,
                verbose,
                note="no rules available",
            )
            self._fitted = True
            self._last_timings["fit"] = timings
            return self

        # Cluster rules into regions.
        cluster_timings: Dict[str, float] = {}
        with self._time_block(timings, "fit", "cluster_total", verbose):
            regions = self._cluster_rules(rules, timings=cluster_timings, verbose=verbose)
        for key, value in cluster_timings.items():
            timings[f"cluster_{key}"] = value
        self._regions = regions

        self._fitted = True
        self._last_timings["fit"] = timings
        return self

    def transform(
        self,
        X: Union[np.ndarray, Sequence[Sequence[Number]]],
        mode: str = "best",
        verbose: int = 0,
    ) -> Union[np.ndarray, List[List[int]]]:
        """Assign each sample to the regions that cover it."""

        self._require_fitted()
        timings: Dict[str, float] = {}

        with self._time_block(timings, "transform", "coerce_features", verbose):
            X_arr = self._coerce_features(X, expect_fit=False)

        if not self._regions:
            self._record_duration(
                timings,
                "transform",
                "region_arrays",
                0.0,
                verbose,
                note="no regions available",
            )
            if mode == "best":
                result: Union[np.ndarray, List[List[int]]] = np.full(
                    (X_arr.shape[0],), -1, dtype=int
                )
            else:
                if mode != "all":
                    raise ValueError("mode must be either 'best' or 'all'")
                result = [[] for _ in range(X_arr.shape[0])]
            self._last_timings["transform"] = timings
            return result

        with self._time_block(timings, "transform", "region_arrays", verbose):
            a, b, mask = self._get_region_arrays()
        with self._time_block(timings, "transform", "points_in_regions", verbose):
            covers = self._points_in_regions(X_arr, a, b, mask)

        if mode == "all":
            with self._time_block(timings, "transform", "collect_all_regions", verbose):
                result_list: List[List[int]] = []
                for cover_row in covers:
                    region_ids = [
                        self._regions[idx].id for idx, flag in enumerate(cover_row) if flag
                    ]
                    result_list.append(region_ids)
            self._last_timings["transform"] = timings
            return result_list

        if mode != "best":
            raise ValueError("mode must be either 'best' or 'all'")

        with self._time_block(timings, "transform", "select_best_region", verbose):
            scores = np.array([r.support * r.purity for r in self._regions], dtype=float)
            best = np.full((X_arr.shape[0],), -1, dtype=int)
            for i, cover_row in enumerate(covers):
                if not np.any(cover_row):
                    continue
                idx = np.argmax(scores * cover_row)
                best[i] = self._regions[idx].id
        self._last_timings["transform"] = timings
        return best

    def predict(
        self,
        X: Union[np.ndarray, Sequence[Sequence[Number]]],
    ) -> np.ndarray:
        """Return class probabilities predicted by the underlying random forest."""

        self._require_fitted()
        if self._rf is None:
            raise RuntimeError("RandomForestClassifier has not been trained")
        X_arr = self._coerce_features(X, expect_fit=False)
        return self._rf.predict_proba(X_arr)

    def explain(self, top_k: int = 20, verbose: int = 0) -> List[Dict[str, Any]]:
        """Return a human-readable summary of the top-K regions."""

        self._require_fitted()
        if not self._regions:
            timings: Dict[str, float] = {}
            self._record_duration(
                timings,
                "explain",
                "no_regions",
                0.0,
                verbose,
                note="no regions available",
            )
            self._last_timings["explain"] = timings
            return []

        timings: Dict[str, float] = {}

        with self._time_block(timings, "explain", "rank_regions", verbose):
            ranking = sorted(
                self._regions,
                key=lambda r: (r.support * r.purity, r.support),
                reverse=True,
            )

        with self._time_block(timings, "explain", "build_summaries", verbose):
            summaries: List[Dict[str, Any]] = []
            for region in ranking[:top_k]:
                bounds = []
                for idx, name in enumerate(self._feature_names or []):
                    if not region.mask[idx]:
                        continue
                    bounds.append(
                        {
                            "feature": name,
                            "lower": float(region.a[idx]),
                            "upper": float(region.b[idx]),
                        }
                    )
                summaries.append(
                    {
                        "region_id": region.id,
                        "support": int(region.support),
                        "purity": float(region.purity),
                        "lift": float(region.lift),
                        "distribution": region.distribution.tolist(),
                        "bounds": bounds,
                    }
                )

        self._last_timings["explain"] = timings
        return summaries

    def export(self, path: Union[str, Path], format: str = "json") -> None:
        """Serialise the current state to *path* in the requested format."""

        self._require_fitted()
        path = Path(path)
        format = format.lower()

        payload = self._build_serialisation_payload()

        if format == "json":
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return
        if format == "npz":
            arrays = payload.pop("arrays")
            np.savez_compressed(path, **arrays, metadata=json.dumps(payload))
            return
        raise ValueError("Unsupported export format; use 'json' or 'npz'")

    def get_rules(self) -> List[Rule]:
        """Return the list of extracted rules."""

        self._require_fitted()
        return list(self._rules)

    def get_regions(self) -> List[Region]:
        """Return the list of aggregated regions."""

        self._require_fitted()
        return list(self._regions)

    def generate_hypotheses(
        self, top_pairs: Optional[int] = None, verbose: int = 0
    ) -> List[Dict[str, Any]]:
        """Return candidate contrasting region pairs."""

        self._require_fitted()
        if not self._regions:
            timings: Dict[str, float] = {}
            self._record_duration(
                timings,
                "generate_hypotheses",
                "no_regions",
                0.0,
                verbose,
                note="no regions available",
            )
            self._last_timings["generate_hypotheses"] = timings
            return []

        tau = self.config["hypotheses"].get("similarity_tau", 0.8)
        global_dist = self.global_class_distribution_
        assert global_dist is not None

        timings: Dict[str, float] = {}

        with self._time_block(
            timings, "generate_hypotheses", "scan_pairs", verbose
        ):
            pairs: List[Tuple[Tuple[int, int], float, float]] = []
            for i, j in combinations(range(len(self._regions)), 2):
                r_i = self._regions[i]
                r_j = self._regions[j]
                sim = self._region_jaccard(r_i, r_j)
                if sim < tau:
                    continue
                delta = abs(r_i.purity - r_j.purity)
                pairs.append(((r_i.id, r_j.id), sim, delta))

        pairs.sort(key=lambda item: (item[2], item[1]), reverse=True)
        if top_pairs is None:
            top_pairs = self.config["hypotheses"].get("top_pairs", 10)

        with self._time_block(
            timings, "generate_hypotheses", "build_results", verbose
        ):
            result: List[Dict[str, Any]] = []
            for (id_i, id_j), sim, delta in pairs[:top_pairs]:
                region_i = self._region_by_id(id_i)
                region_j = self._region_by_id(id_j)
                result.append(
                    {
                        "pair": (id_i, id_j),
                        "similarity": float(sim),
                        "purity_delta": float(delta),
                        "region_i": self._region_summary(region_i, global_dist),
                        "region_j": self._region_summary(region_j, global_dist),
                    }
                )

        self._last_timings["generate_hypotheses"] = timings
        return result

    def get_last_timings(self) -> Dict[str, Dict[str, float]]:
        """Return the most recent timings recorded for public methods."""

        return {scope: dict(values) for scope, values in self._last_timings.items()}

    # ------------------------------------------------------------------
    # Helpers: diagnostics & timing
    # ------------------------------------------------------------------
    def _log_verbose(self, verbose: int, scope: str, message: str) -> None:
        if verbose:
            print(f"[InsideForest][{scope}] {message}")

    def _record_duration(
        self,
        timings: Dict[str, float],
        scope: str,
        name: str,
        duration: float,
        verbose: int,
        note: Optional[str] = None,
    ) -> None:
        timings[name] = duration
        if verbose:
            suffix = f" ({note})" if note else ""
            print(f"[InsideForest][{scope}] {name}: {duration:.6f}s{suffix}")

    @contextmanager
    def _time_block(
        self,
        timings: Dict[str, float],
        scope: str,
        name: str,
        verbose: int,
        note: Optional[str] = None,
    ):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._record_duration(timings, scope, name, duration, verbose, note=note)

    # ------------------------------------------------------------------
    # Helpers: configuration & validation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_config(user_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        cfg = {
            "rf": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
            "rules": {
                "min_rule_support": 5,
                "min_rule_purity": 0.55,
                "max_rules_per_tree": None,
            },
            "clustering": {
                "cluster_method": "dbscan",
                "dbscan_eps": None,
                "dbscan_min_samples": 3,
                "kmeans_k": None,
                "fallback_threshold": 8000,
            },
            "regions": {
                "jaccard_merge_threshold": 0.85,
                "min_region_support": 10,
            },
            "hypotheses": {
                "similarity_tau": 0.8,
                "top_pairs": 10,
            },
            "performance": {
                "use_float32": True,
            },
        }

        for section, values in user_cfg.items():
            if section not in cfg:
                cfg[section] = values
                continue
            cfg[section].update(values)
        return cfg

    def _make_random_forest(self) -> RandomForestClassifier:
        rf_cfg = self.config["rf"].copy()
        # Parameters accepted by RandomForestClassifier
        rf_params = {
            key: rf_cfg[key]
            for key in [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "random_state",
                "n_jobs",
            ]
            if key in rf_cfg
        }
        return RandomForestClassifier(**rf_params)

    def _coerce_features(
        self,
        X: Union[np.ndarray, Sequence[Sequence[Number]]],
        expect_fit: bool = True,
    ) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array-like structure")
        dtype = np.float32 if self.config["performance"].get("use_float32", True) else np.float64
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        if expect_fit and arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        return arr

    def _coerce_target(
        self,
        y: Optional[Union[np.ndarray, Sequence[Number]]],
        rf_classes: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        if y is None:
            raise ValueError("y must contain at least one sample")
        arr = np.asarray(y)
        if arr.ndim != 1:
            raise ValueError("y must be a 1D array-like structure")
        if arr.shape[0] == 0:
            raise ValueError("y must contain at least one sample")
        self._label_encoder = LabelEncoder()
        if rf_classes is not None:
            self._label_encoder.fit(list(rf_classes))
            try:
                arr_enc = self._label_encoder.transform(arr)
            except ValueError as exc:  # noqa: F841
                raise ValueError(
                    "y contains classes that are incompatible with the provided random forest"
                ) from exc
        else:
            arr_enc = self._label_encoder.fit_transform(arr)

        if self.class_names is not None:
            if len(self.class_names) != len(self._label_encoder.classes_):
                raise ValueError(
                    "Provided class_names length does not match the number of classes"
                )
        else:
            self.class_names = [str(cls) for cls in self._label_encoder.classes_]
        return arr_enc.astype(np.int32, copy=False)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("InsideForest instance has not been fitted yet")

    # ------------------------------------------------------------------
    # Helpers: rule extraction
    # ------------------------------------------------------------------
    def _extract_rules(
        self,
        rf: RandomForestClassifier,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Rule]:
        n_features = X.shape[1]
        rules: List[Rule] = []
        global_min = np.asarray(self.global_mins_, dtype=np.float32)
        global_max = np.asarray(self.global_maxs_, dtype=np.float32)

        rule_cfg = self.config["rules"]
        min_support = int(rule_cfg.get("min_rule_support", 1))
        min_purity = float(rule_cfg.get("min_rule_purity", 0.0))
        max_per_tree = rule_cfg.get("max_rules_per_tree")

        rule_id = 0
        for tree_id, estimator in enumerate(rf.estimators_):
            tree = estimator.tree_
            children_left = tree.children_left
            children_right = tree.children_right
            features = tree.feature
            thresholds = tree.threshold
            impurities = tree.impurity
            n_node_samples = tree.n_node_samples

            stack: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
            root_a = global_min.copy()
            root_b = global_max.copy()
            root_mask = np.zeros(n_features, dtype=bool)
            stack.append((0, root_a, root_b, root_mask))
            tree_rules: List[Rule] = []

            while stack:
                node_id, a, b, mask = stack.pop()
                left = children_left[node_id]
                right = children_right[node_id]

                if left == -1 and right == -1:
                    support = int(n_node_samples[node_id])
                    if support < min_support:
                        continue
                    counts = tree.value[node_id][0].astype(np.int64)
                    total = counts.sum()
                    if total <= 0:
                        continue
                    distribution = counts / total
                    if distribution.max() < min_purity:
                        continue
                    rule = Rule(
                        id=rule_id,
                        tree_id=tree_id,
                        leaf_id=node_id,
                        a=a.copy(),
                        b=b.copy(),
                        mask=mask.copy(),
                        support=support,
                        class_counts=counts.astype(np.int64, copy=True),
                        distribution=distribution.astype(np.float64, copy=True),
                        weight=float(support * distribution.max()),
                        gini=float(impurities[node_id]),
                    )
                    tree_rules.append(rule)
                    rule_id += 1
                    continue

                feature = features[node_id]
                threshold = thresholds[node_id]
                if feature == -2:
                    # No split defined (should not happen in sklearn trees)
                    continue

                # Prepare left child (<= threshold)
                left_a = a.copy()
                left_b = b.copy()
                left_mask = mask.copy()
                left_b[feature] = min(left_b[feature], threshold)
                left_mask[feature] = True

                # Prepare right child (> threshold)
                right_a = a.copy()
                right_b = b.copy()
                right_mask = mask.copy()
                # Use nextafter to avoid overlaps due to float rounding
                right_a[feature] = max(
                    right_a[feature],
                    np.nextafter(threshold, np.float32(np.inf)),
                )
                right_mask[feature] = True

                stack.append((left, left_a, left_b, left_mask))
                stack.append((right, right_a, right_b, right_mask))

            if max_per_tree is not None and len(tree_rules) > int(max_per_tree):
                tree_rules.sort(
                    key=lambda r: (r.support * r.distribution.max(), r.support),
                    reverse=True,
                )
                tree_rules = tree_rules[: int(max_per_tree)]

            rules.extend(tree_rules)

        return rules

    # ------------------------------------------------------------------
    # Helpers: clustering & regions
    # ------------------------------------------------------------------
    def _cluster_rules(
        self,
        rules: List[Rule],
        timings: Optional[Dict[str, float]] = None,
        verbose: int = 0,
    ) -> List[Region]:
        clustering_cfg = self.config["clustering"]
        method = clustering_cfg.get("cluster_method", "dbscan").lower()

        local_timings: Dict[str, float] = timings if timings is not None else {}

        with self._time_block(local_timings, "cluster", "rules_to_matrix", verbose):
            V = self._rules_to_matrix(rules)
        if V.size == 0:
            return []

        if method == "dbscan" and len(rules) <= clustering_cfg.get("fallback_threshold", 8000):
            labels = self._cluster_dbscan(rules, V, timings=local_timings, verbose=verbose)
        else:
            labels = self._cluster_kmeans(rules, V, timings=local_timings, verbose=verbose)

        with self._time_block(local_timings, "cluster", "build_regions", verbose):
            regions = self._build_regions_from_labels(rules, labels)
        with self._time_block(local_timings, "cluster", "merge_filter", verbose):
            regions = self._merge_and_filter_regions(regions)
        return regions

    def _cluster_dbscan(
        self,
        rules: List[Rule],
        V: np.ndarray,
        timings: Optional[Dict[str, float]] = None,
        verbose: int = 0,
    ) -> np.ndarray:
        clustering_cfg = self.config["clustering"]
        min_samples = int(clustering_cfg.get("dbscan_min_samples", 3))
        eps = clustering_cfg.get("dbscan_eps")

        local_timings: Dict[str, float] = timings if timings is not None else {}

        with self._time_block(local_timings, "cluster", "distance_matrix", verbose):
            D = self._jaccard_distance_matrix(rules)

        if eps is None:
            with self._time_block(local_timings, "cluster", "eps_calibration", verbose):
                eps = self._auto_calibrate_eps(D, min_samples)

        with self._time_block(local_timings, "cluster", "dbscan_fit", verbose):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = dbscan.fit_predict(D)

        if np.all(labels == -1):
            # If DBSCAN fails (all noise), fallback to KMeans
            self._log_verbose(
                verbose,
                "cluster",
                "dbscan_fit produced only noise; switching to kmeans",
            )
            return self._cluster_kmeans(rules, V, timings=local_timings, verbose=verbose)
        return labels

    def _cluster_kmeans(
        self,
        rules: List[Rule],
        V: np.ndarray,
        timings: Optional[Dict[str, float]] = None,
        verbose: int = 0,
    ) -> np.ndarray:
        clustering_cfg = self.config["clustering"]
        local_timings: Dict[str, float] = timings if timings is not None else {}

        k = clustering_cfg.get("kmeans_k")
        if k is None or k <= 0:
            # Default heuristic: number of target classes or min(10, n_rules)
            k = min(max(len(self.class_names or []), 2), len(rules))
        if k <= 1:
            labels = np.zeros(len(rules), dtype=int)
            self._record_duration(
                local_timings,
                "cluster",
                "kmeans_fit",
                0.0,
                verbose,
                note="single cluster fallback",
            )
            return labels

        with self._time_block(local_timings, "cluster", "kmeans_fit", verbose):
            km = KMeans(
                n_clusters=k,
                n_init="auto",
                random_state=self.config["rf"].get("random_state", 42),
            )
            labels = km.fit_predict(V)
        return labels

    def _auto_calibrate_eps(self, D: np.ndarray, min_samples: int) -> float:
        if D.shape[0] <= min_samples:
            # Median of non-zero distances (if any)
            non_zero = D[D > 0]
            if non_zero.size == 0:
                return 0.1
            return float(np.percentile(non_zero, 50))

        k = max(min_samples, 1)
        kth_dist = np.partition(D, kth=k, axis=1)[:, k]
        # Robust percentile blend between 35 and 55 percentiles
        lo = np.percentile(kth_dist, 35)
        hi = np.percentile(kth_dist, 55)
        return float((lo + hi) / 2.0)

    def _rules_to_matrix(self, rules: List[Rule]) -> np.ndarray:
        if not rules:
            return np.empty((0, 0), dtype=np.float32)
        a = np.stack([rule.a for rule in rules]).astype(np.float32, copy=False)
        b = np.stack([rule.b for rule in rules]).astype(np.float32, copy=False)
        return np.concatenate([a, b], axis=1)

    def _build_regions_from_labels(self, rules: List[Rule], labels: np.ndarray) -> List[Region]:
        label_to_rules: Dict[int, List[Rule]] = {}
        for rule, label in zip(rules, labels):
            if label == -1:
                # Noise: treat rule as its own singleton region
                label = max(label_to_rules.keys(), default=-1) + 1
            label_to_rules.setdefault(label, []).append(rule)

        regions: List[Region] = []
        next_region_id = 0
        for label, members in label_to_rules.items():
            region = self._aggregate_region(next_region_id, members)
            region.algo_info["label"] = int(label)
            regions.append(region)
            next_region_id += 1
        return regions

    def _aggregate_region(self, region_id: int, members: List[Rule]) -> Region:
        n_features = members[0].a.shape[0]
        member_ids = [rule.id for rule in members]

        stacked_a = np.stack([rule.a for rule in members])
        stacked_b = np.stack([rule.b for rule in members])
        stacked_mask = np.stack([rule.mask for rule in members])

        region_mask = stacked_mask.any(axis=0)
        region_a = np.where(
            region_mask,
            stacked_a.min(axis=0),
            np.asarray(self.global_mins_, dtype=np.float32),
        )
        region_b = np.where(
            region_mask,
            stacked_b.max(axis=0),
            np.asarray(self.global_maxs_, dtype=np.float32),
        )

        support = int(sum(rule.support for rule in members))
        class_counts = np.sum([rule.class_counts for rule in members], axis=0)
        class_counts = class_counts.astype(np.int64, copy=False)
        total = class_counts.sum()
        distribution = (
            class_counts / total if total > 0 else np.zeros_like(class_counts, dtype=float)
        )
        purity = float(distribution.max()) if total > 0 else 0.0
        global_dist = self.global_class_distribution_
        assert global_dist is not None
        majority_idx = int(np.argmax(distribution)) if total > 0 else 0
        global_rate = float(global_dist[majority_idx]) if global_dist.size > 0 else 0.0
        lift = float(purity / global_rate) if global_rate > 0 else float("inf")

        return Region(
            id=region_id,
            member_rule_ids=member_ids,
            a=region_a.astype(np.float32, copy=False),
            b=region_b.astype(np.float32, copy=False),
            mask=region_mask.astype(bool, copy=False),
            support=support,
            class_counts=class_counts,
            distribution=distribution.astype(np.float64, copy=False),
            purity=purity,
            lift=lift,
            algo_info={},
        )

    def _merge_and_filter_regions(self, regions: List[Region]) -> List[Region]:
        if not regions:
            return []

        min_support = int(self.config["regions"].get("min_region_support", 1))
        merge_tau = float(self.config["regions"].get("jaccard_merge_threshold", 0.85))

        filtered = [region for region in regions if region.support >= min_support]
        if not filtered:
            return []

        merged: List[Region] = []
        visited = [False] * len(filtered)
        for idx, region in enumerate(filtered):
            if visited[idx]:
                continue
            current_members = [region]
            visited[idx] = True
            for jdx in range(idx + 1, len(filtered)):
                if visited[jdx]:
                    continue
                candidate = filtered[jdx]
                sim = self._region_jaccard(region, candidate)
                if sim >= merge_tau:
                    current_members.append(candidate)
                    visited[jdx] = True

            if len(current_members) == 1:
                merged.append(region)
                continue

            merged_region = self._aggregate_region(region.id, [
                rule
                for member in current_members
                for rule in self._rules
                if rule.id in member.member_rule_ids
            ])
            merged.append(merged_region)

        # Ensure deterministic ordering and consecutive IDs
        merged.sort(key=lambda r: r.id)
        for idx, region in enumerate(merged):
            region.id = idx
        self._region_arrays_cache = None
        return merged

    # ------------------------------------------------------------------
    # Helpers: geometry & distances
    # ------------------------------------------------------------------
    def _jaccard_distance_matrix(self, rules: List[Rule]) -> np.ndarray:
        m = len(rules)
        D = np.zeros((m, m), dtype=np.float32)
        block = 128
        for start in range(0, m, block):
            end = min(start + block, m)
            for i in range(start, end):
                D[i, i] = 0.0
                for j in range(i + 1, m):
                    sim = self._rule_jaccard(rules[i], rules[j])
                    dist = 1.0 - sim
                    D[i, j] = D[j, i] = dist
        return D

    def _rule_jaccard(self, r1: Rule, r2: Rule) -> float:
        global_min = np.asarray(self.global_mins_, dtype=np.float32)
        global_max = np.asarray(self.global_maxs_, dtype=np.float32)
        inter = 1.0
        union = 1.0
        eps = 1e-12
        for idx in range(r1.a.shape[0]):
            if not r1.mask[idx] and not r2.mask[idx]:
                length = max(global_max[idx] - global_min[idx], eps)
                inter *= length
                union *= length
                continue
            a1 = r1.a[idx] if r1.mask[idx] else global_min[idx]
            b1 = r1.b[idx] if r1.mask[idx] else global_max[idx]
            a2 = r2.a[idx] if r2.mask[idx] else global_min[idx]
            b2 = r2.b[idx] if r2.mask[idx] else global_max[idx]
            lower = max(a1, a2)
            upper = min(b1, b2)
            intersection = max(0.0, upper - lower)
            length1 = max(b1 - a1, eps)
            length2 = max(b2 - a2, eps)
            inter *= intersection
            union *= (length1 + length2 - intersection)
            if union <= eps:
                return 0.0
        return float(inter / max(union, eps))

    def _region_jaccard(self, r1: Region, r2: Region) -> float:
        tmp_rule1 = Rule(
            id=-1,
            tree_id=-1,
            leaf_id=-1,
            a=r1.a,
            b=r1.b,
            mask=r1.mask,
            support=r1.support,
            class_counts=r1.class_counts,
            distribution=r1.distribution,
            weight=0.0,
            gini=0.0,
        )
        tmp_rule2 = Rule(
            id=-2,
            tree_id=-1,
            leaf_id=-1,
            a=r2.a,
            b=r2.b,
            mask=r2.mask,
            support=r2.support,
            class_counts=r2.class_counts,
            distribution=r2.distribution,
            weight=0.0,
            gini=0.0,
        )
        return self._rule_jaccard(tmp_rule1, tmp_rule2)

    def _points_in_regions(
        self, X: np.ndarray, a: np.ndarray, b: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        # X shape: (n_samples, n_features)
        # a/b/mask shape: (n_regions, n_features)
        n_samples = X.shape[0]
        n_regions = a.shape[0]
        if n_regions == 0:
            return np.zeros((n_samples, 0), dtype=bool)

        X_exp = X[:, None, :]
        lower_ok = ~mask | (X_exp >= a[None, :, :])
        upper_ok = ~mask | (X_exp <= b[None, :, :])
        covers = np.all(lower_ok & upper_ok, axis=2)
        return covers

    def _get_region_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._region_arrays_cache is not None:
            return self._region_arrays_cache  # type: ignore[return-value]
        a = np.stack([region.a for region in self._regions])
        b = np.stack([region.b for region in self._regions])
        mask = np.stack([region.mask for region in self._regions])
        self._region_arrays_cache = (a, b, mask)
        return a, b, mask

    # ------------------------------------------------------------------
    # Helpers: serialisation & reporting
    # ------------------------------------------------------------------
    def _build_serialisation_payload(self) -> Dict[str, Any]:
        rf_state = None
        if self._rf is not None:
            rf_state = {
                "n_estimators": len(self._rf.estimators_),
                "params": self._rf.get_params(),
                "source": self._rf_source,
            }

        arrays = {}
        if self._regions:
            arrays.update(
                {
                    "regions_a": np.stack([r.a for r in self._regions]),
                    "regions_b": np.stack([r.b for r in self._regions]),
                    "regions_mask": np.stack([r.mask for r in self._regions]),
                    "regions_counts": np.stack([r.class_counts for r in self._regions]),
                }
            )
        if self._rules:
            arrays.update(
                {
                    "rules_a": np.stack([r.a for r in self._rules]),
                    "rules_b": np.stack([r.b for r in self._rules]),
                    "rules_mask": np.stack([r.mask for r in self._rules]),
                    "rules_counts": np.stack([r.class_counts for r in self._rules]),
                }
            )

        payload = {
            "config": self.config,
            "feature_names": self._feature_names,
            "class_names": self.class_names,
            "rf_state": rf_state,
            "rules": [
                {
                    "id": r.id,
                    "tree_id": r.tree_id,
                    "leaf_id": r.leaf_id,
                    "support": r.support,
                    "distribution": r.distribution.tolist(),
                    "weight": r.weight,
                    "gini": r.gini,
                }
                for r in self._rules
            ],
            "regions": [
                {
                    "id": r.id,
                    "member_rule_ids": r.member_rule_ids,
                    "support": r.support,
                    "distribution": r.distribution.tolist(),
                    "purity": r.purity,
                    "lift": r.lift,
                    "algo_info": r.algo_info,
                }
                for r in self._regions
            ],
            "global_mins": None if self.global_mins_ is None else self.global_mins_.tolist(),
            "global_maxs": None if self.global_maxs_ is None else self.global_maxs_.tolist(),
            "y_global_counts": None
            if self.global_class_counts_ is None
            else self.global_class_counts_.tolist(),
            "y_global_dist": None
            if self.global_class_distribution_ is None
            else self.global_class_distribution_.tolist(),
            "arrays": arrays,
        }
        return payload

    def _region_summary(self, region: Region, global_dist: np.ndarray) -> Dict[str, Any]:
        bounds = []
        for idx, name in enumerate(self._feature_names or []):
            if not region.mask[idx]:
                continue
            bounds.append(
                {
                    "feature": name,
                    "lower": float(region.a[idx]),
                    "upper": float(region.b[idx]),
                }
            )
        return {
            "id": region.id,
            "support": int(region.support),
            "purity": float(region.purity),
            "lift": float(region.lift),
            "distribution": region.distribution.tolist(),
            "bounds": bounds,
        }

    def _region_by_id(self, region_id: int) -> Region:
        for region in self._regions:
            if region.id == region_id:
                return region
        raise KeyError(f"Region with id {region_id} not found")


__all__ = ["InsideForest", "Rule", "Region"]

