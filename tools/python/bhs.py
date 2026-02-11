#!/usr/bin/env python3
"""
Binary Hierarchical Silhouette (BHS) clustering.

A divisive (top-down) hierarchical clustering method that recursively
bisects clusters using k-means (k=2), accepting splits only when the
mean silhouette score exceeds a configurable threshold.

Each leaf cluster receives a hierarchical path label encoding its
position in the binary split tree (e.g., "L", "R", "L.L", "R.L.R").

This is a general-purpose tool -- it accepts any numeric matrix and
is not specific to any particular domain.

Example
-------
    >>> from bhs import BHSClustering
    >>> import numpy as np
    >>> X = np.vstack([np.random.randn(50, 5) + 3,
    ...               np.random.randn(50, 5) - 3])
    >>> model = BHSClustering(silhouette_threshold=0.25)
    >>> labels = model.fit(X)
    >>> print(model.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("bhs")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitRecord:
    """Record of a single split decision in the BHS tree."""

    path: str
    n_observations: int
    silhouette: float | None
    accepted: bool
    reason: str
    child_sizes: tuple[int, int] | None = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BHSClustering:
    """
    Binary Hierarchical Silhouette clustering.

    Recursively bisects a dataset using k-means (k=2).  A split is
    accepted only when the mean silhouette score of the two child
    clusters exceeds ``silhouette_threshold``.  Recursion stops when a
    cluster is too small, when the maximum tree depth is reached, or
    when the silhouette criterion is not met.

    Parameters
    ----------
    silhouette_threshold : float, default 0.25
        Minimum mean silhouette score to accept a k=2 split.
    min_cluster_size : int, default 10
        Minimum number of observations required to attempt a split.
        Clusters smaller than this are automatically declared leaves.
    max_depth : int, default 20
        Maximum depth of the binary tree.
    random_state : int or None, default 42
        Seed for k-means reproducibility.
    n_init : int, default 10
        Number of k-means initialisations per split.

    Attributes
    ----------
    labels_ : pd.Series
        Cluster path labels for each observation (set after ``fit``).
    tree_ : list[SplitRecord]
        Ordered log of every split decision.
    n_leaves_ : int
        Number of leaf (terminal) clusters.
    """

    def __init__(
        self,
        silhouette_threshold: float = 0.25,
        min_cluster_size: int = 10,
        max_depth: int = 20,
        random_state: int | None = 42,
        n_init: int = 10,
    ) -> None:
        if silhouette_threshold < -1 or silhouette_threshold > 1:
            raise ValueError(
                f"silhouette_threshold must be in [-1, 1], got {silhouette_threshold}"
            )
        if min_cluster_size < 4:
            raise ValueError(
                f"min_cluster_size must be >= 4 (need >=2 per child for "
                f"silhouette), got {min_cluster_size}"
            )
        self.silhouette_threshold = silhouette_threshold
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_init = n_init

        # Populated by fit()
        self.labels_: pd.Series | None = None
        self.tree_: list[SplitRecord] = []
        self.n_leaves_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        """
        Run BHS clustering on *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.  If a DataFrame, its index is preserved in the
            returned labels Series.

        Returns
        -------
        pd.Series
            Mapping from row index to hierarchical cluster path string.

        Raises
        ------
        ValueError
            If *X* contains NaN, has fewer than 2 columns, or has too
            few rows.
        """
        # --- Input validation -------------------------------------------
        if isinstance(X, pd.DataFrame):
            index = X.index.copy()
            data = X.values.astype(float)
        else:
            data = np.asarray(X, dtype=float)
            index = pd.RangeIndex(len(data))

        if data.ndim != 2:
            raise ValueError(f"X must be 2-D, got {data.ndim}-D")
        if data.shape[1] < 2:
            raise ValueError(
                f"X must have >= 2 features, got {data.shape[1]}"
            )
        if data.shape[0] < self.min_cluster_size:
            raise ValueError(
                f"X has {data.shape[0]} rows, fewer than "
                f"min_cluster_size={self.min_cluster_size}"
            )
        if np.isnan(data).any():
            raise ValueError("X contains NaN values")

        # --- Initialise state -------------------------------------------
        self._data = data
        self._labels = np.empty(len(data), dtype=object)
        self.tree_ = []
        self.n_leaves_ = 0

        logger.info(
            "Starting BHS: %d observations x %d features, "
            "threshold=%.3f, min_size=%d",
            data.shape[0], data.shape[1],
            self.silhouette_threshold, self.min_cluster_size,
        )

        # --- Recurse ----------------------------------------------------
        self._split(np.arange(len(data)), path="ROOT", depth=0)

        # --- Package results --------------------------------------------
        self.labels_ = pd.Series(self._labels, index=index, name="cluster_id")
        logger.info("BHS complete: %d leaf clusters", self.n_leaves_)
        return self.labels_

    def summary(self) -> str:
        """Return a human-readable summary of the clustering result."""
        if self.labels_ is None:
            return "Model has not been fit yet."

        lines = [
            f"BHS Clustering Summary",
            f"  Parameters:",
            f"    silhouette_threshold = {self.silhouette_threshold}",
            f"    min_cluster_size     = {self.min_cluster_size}",
            f"    max_depth            = {self.max_depth}",
            f"    random_state         = {self.random_state}",
            f"  Results:",
            f"    Leaf clusters: {self.n_leaves_}",
            f"    Total observations: {len(self.labels_)}",
            f"",
            f"  Cluster sizes:",
        ]
        sizes = self.labels_.value_counts().sort_index()
        for path, n in sizes.items():
            lines.append(f"    {path:30s}  n={n}")
        return "\n".join(lines)

    def get_tree_diagram(self) -> str:
        """Return an ASCII diagram of the split tree."""
        if not self.tree_:
            return "No tree built yet."

        lines = ["BHS Split Tree:"]
        for rec in self.tree_:
            depth = 0 if rec.path == "ROOT" else rec.path.count(".") + 1
            indent = "  " * depth
            if rec.accepted:
                sil_str = f"sil={rec.silhouette:.3f}" if rec.silhouette is not None else ""
                child_str = ""
                if rec.child_sizes is not None:
                    child_str = f" -> [{rec.child_sizes[0]}, {rec.child_sizes[1]}]"
                lines.append(
                    f"{indent}{rec.path} (n={rec.n_observations}) "
                    f"SPLIT {sil_str}{child_str}"
                )
            else:
                sil_str = f"sil={rec.silhouette:.3f}" if rec.silhouette is not None else ""
                lines.append(
                    f"{indent}{rec.path} (n={rec.n_observations}) "
                    f"LEAF  {sil_str} [{rec.reason}]"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Recursive engine
    # ------------------------------------------------------------------

    def _split(
        self, indices: np.ndarray, path: str, depth: int
    ) -> None:
        """
        Attempt to bisect the observations at *indices*.

        If the split is accepted, recurse on each child.  Otherwise
        declare a leaf and assign the path label.
        """
        n = len(indices)

        # --- Stopping: too small ----------------------------------------
        if n < self.min_cluster_size:
            self._declare_leaf(indices, path, silhouette=None,
                               reason=f"n={n} < min_cluster_size={self.min_cluster_size}")
            return

        # --- Stopping: max depth ----------------------------------------
        if depth >= self.max_depth:
            self._declare_leaf(indices, path, silhouette=None,
                               reason=f"depth={depth} >= max_depth={self.max_depth}")
            return

        # --- k-means(k=2) ----------------------------------------------
        subset = self._data[indices]
        km = KMeans(
            n_clusters=2,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        km.fit(subset)
        sub_labels = km.labels_

        # --- Degenerate split check -------------------------------------
        sizes = [int((sub_labels == k).sum()) for k in range(2)]
        if min(sizes) < 2:
            self._declare_leaf(
                indices, path, silhouette=None,
                reason=f"degenerate split: child sizes {sizes}",
            )
            return

        # --- Silhouette -------------------------------------------------
        try:
            score = silhouette_score(subset, sub_labels)
        except ValueError as exc:
            self._declare_leaf(
                indices, path, silhouette=None,
                reason=f"silhouette error: {exc}",
            )
            return

        # --- Decision ---------------------------------------------------
        if score > self.silhouette_threshold:
            # Accept the split
            left_path = "L" if path == "ROOT" else f"{path}.L"
            right_path = "R" if path == "ROOT" else f"{path}.R"

            self.tree_.append(SplitRecord(
                path=path,
                n_observations=n,
                silhouette=score,
                accepted=True,
                reason="accepted",
                child_sizes=(sizes[0], sizes[1]),
            ))
            logger.info(
                "SPLIT %s: n=%d, sil=%.4f > %.4f -> [%d, %d]",
                path, n, score, self.silhouette_threshold,
                sizes[0], sizes[1],
            )

            left_idx = indices[sub_labels == 0]
            right_idx = indices[sub_labels == 1]
            self._split(left_idx, left_path, depth + 1)
            self._split(right_idx, right_path, depth + 1)
        else:
            self._declare_leaf(
                indices, path, silhouette=score,
                reason=f"sil={score:.4f} <= threshold={self.silhouette_threshold}",
            )

    def _declare_leaf(
        self,
        indices: np.ndarray,
        path: str,
        silhouette: float | None,
        reason: str,
    ) -> None:
        """Mark all *indices* as belonging to the leaf at *path*."""
        # Leaf label: use the path, but ROOT becomes "ROOT" only if
        # the entire dataset was never split.
        label = path if path != "ROOT" else "ROOT"
        self._labels[indices] = label
        self.n_leaves_ += 1

        self.tree_.append(SplitRecord(
            path=path,
            n_observations=len(indices),
            silhouette=silhouette,
            accepted=False,
            reason=reason,
        ))
        logger.info("LEAF  %s: n=%d [%s]", path, len(indices), reason)
