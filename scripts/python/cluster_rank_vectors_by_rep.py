#!/usr/bin/env python3
"""
BHS-cluster each biological replicate's rank vectors independently.

Produces one cluster assignment file per replicate, using the same
BHS parameters as the combined-replicate run.

Input
-----
data/processed/rank_vectors_rep1.csv
data/processed/rank_vectors_rep2.csv

Output
------
results/bhs_clusters_rep1.tsv
results/bhs_clusters_rep2.tsv
"""

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools" / "python"))

from bhs import BHSClustering  # noqa: E402

SILHOUETTE_THRESHOLD = 0.20
MIN_CLUSTER_SIZE = 10
RANDOM_STATE = 42
MAX_DEPTH = 20


def cluster_rep(rep: int) -> None:
    """Load rank vectors for one replicate, run BHS, write results."""
    in_csv = PROJECT / "data" / "processed" / f"rank_vectors_rep{rep}.csv"
    out_tsv = PROJECT / "results" / f"bhs_clusters_rep{rep}.tsv"

    vectors = pd.read_csv(in_csv, index_col="systematic_name")
    assert not vectors.isna().any().any(), f"NaN in {in_csv.name}"
    print(f"Loaded {in_csv.name}: {vectors.shape[0]:,} genes x {vectors.shape[1]} conditions")

    model = BHSClustering(
        silhouette_threshold=SILHOUETTE_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    labels = model.fit(vectors)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "systematic_name": labels.index,
        "cluster_id": labels.values,
    }).to_csv(out_tsv, sep="\t", index=False)

    print(f"Wrote {out_tsv.name}: {len(labels):,} genes, "
          f"{model.n_leaves_} clusters\n")
    print(model.summary())
    print()
    print(model.get_tree_diagram())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(message)s",
    )
    for rep in (1, 2):
        print(f"\n{'='*60}")
        print(f"  Replicate {rep}")
        print(f"{'='*60}\n")
        cluster_rep(rep)


if __name__ == "__main__":
    main()
