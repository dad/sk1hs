#!/usr/bin/env python3
"""
Cluster gene rank vectors using Binary Hierarchical Silhouette (BHS).

Loads the baseline-subtracted, median-centered rank vector matrix
produced by ``prepare_rank_vectors.py``, runs BHS clustering, and
writes cluster assignments with hierarchical path labels.

Input
-----
data/processed/rank_vectors.csv

Output
------
results/bhs_clusters.tsv
    Two columns: ``systematic_name`` and ``cluster_id`` (hierarchical
    path string, e.g. "L.R.L").
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Make tools/python/ importable
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools" / "python"))

from bhs import BHSClustering  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RANK_VECTORS_CSV = PROJECT / "data" / "processed" / "rank_vectors.csv"
OUT_TSV = PROJECT / "results" / "bhs_clusters.tsv"

# ---------------------------------------------------------------------------
# BHS parameters
# ---------------------------------------------------------------------------

SILHOUETTE_THRESHOLD = 0.20
MIN_CLUSTER_SIZE = 10
RANDOM_STATE = 42
MAX_DEPTH = 20


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_rank_vectors(path: Path) -> pd.DataFrame:
    """Load rank vector CSV and validate."""
    df = pd.read_csv(path, index_col="systematic_name")
    if df.isna().any().any():
        raise ValueError("Rank vector matrix contains NaN values")
    print(f"Loaded {path.name}: {df.shape[0]:,} genes x {df.shape[1]} conditions")
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(message)s",
    )

    # Load data
    vectors = load_rank_vectors(RANK_VECTORS_CSV)

    # Cluster
    model = BHSClustering(
        silhouette_threshold=SILHOUETTE_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    labels = model.fit(vectors)

    # Write output
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "systematic_name": labels.index,
        "cluster_id": labels.values,
    })
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nWrote {OUT_TSV.name}: {len(out):,} genes, "
          f"{model.n_leaves_} clusters")

    # Summary
    print()
    print(model.summary())
    print()
    print(model.get_tree_diagram())


if __name__ == "__main__":
    main()
