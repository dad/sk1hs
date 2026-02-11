#!/usr/bin/env python3
"""
Prepare per-replicate rank vectors for BHS clustering.

Generates one rank vector matrix per biological replicate by ranking
TPM values within each sample independently, then applying the same
baseline-subtraction and median-centering as prepare_rank_vectors.py.

Input
-----
data/processed/yeast_ORF_TPM_matrix_labeled.csv
    Wide TPM matrix (6,669 genes x 74 samples).

Outputs
-------
data/processed/rank_vectors_rep1.csv
data/processed/rank_vectors_rep2.csv
    Each: 6,669 genes x 37 columns (systematic_name + 36 numeric).
    Same column naming and ordering as rank_vectors.csv.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT / "data" / "processed" / "yeast_ORF_TPM_matrix_labeled.csv"
META_JSON = PROJECT / "data" / "processed" / "rank_vectors_meta.json"

BASELINE_TEMP = 30
BASELINE_TIME = 15
HEAT_TEMPS = [35, 37, 39, 42, 44, 46]
TIMES = [5, 15, 30, 60, 90, 120]


def main() -> None:
    # ---- Load wide TPM matrix -------------------------------------------
    df = pd.read_csv(IN_CSV)
    id_cols = ["systematic_name", "common_name"]
    sample_cols = [c for c in df.columns if c not in id_cols]
    print(f"Loaded {IN_CSV.name}: {len(df):,} genes, {len(sample_cols)} samples")

    # ---- Rank each sample column independently (1 = highest TPM) --------
    rank_wide = df[sample_cols].rank(ascending=False, method="average")
    rank_wide["systematic_name"] = df["systematic_name"].values

    # ---- Split samples by replicate -------------------------------------
    # Sample naming: "{temp}_{time}_{rep}"
    for rep in (1, 2):
        print(f"\n--- Replicate {rep} ---")

        # Identify columns for this rep
        rep_cols = [c for c in sample_cols
                    if c.split("_")[2] == str(rep)]
        print(f"  {len(rep_cols)} samples for rep {rep}")

        # Build a gene x (temp, time) rank matrix for this rep
        # Melt to long, then pivot
        long = rank_wide[["systematic_name"] + rep_cols].melt(
            id_vars=["systematic_name"],
            value_vars=rep_cols,
            var_name="sample",
            value_name="rank",
        )
        long[["temp", "time", "_rep"]] = (
            long["sample"].str.split("_", expand=True).astype(int)
        )

        wide = long.pivot_table(
            index="systematic_name",
            columns=["temp", "time"],
            values="rank",
            aggfunc="first",
        )
        assert wide.notna().all().all(), f"NaN in rep {rep} rank matrix"
        print(f"  Pivoted: {wide.shape[0]:,} genes x {wide.shape[1]} conditions")

        # ---- Subtract baseline ------------------------------------------
        baseline_col = (BASELINE_TEMP, BASELINE_TIME)
        baseline = wide[baseline_col]
        delta = wide.sub(baseline, axis=0).drop(columns=[baseline_col])
        print(f"  Baseline subtracted: {delta.shape[1]} conditions")

        # ---- Median-centre each gene ------------------------------------
        row_medians = delta.median(axis=1)
        centered = delta.sub(row_medians, axis=0)

        residual = centered.median(axis=1).abs().max()
        assert residual < 1e-8, f"Median centering failed: residual={residual}"

        # ---- Order and rename columns -----------------------------------
        ordered = [(t, m) for t in HEAT_TEMPS for m in TIMES]
        centered = centered[ordered]
        centered.columns = [f"t{t}_m{m}" for t, m in ordered]

        # ---- Write ------------------------------------------------------
        out_csv = PROJECT / "data" / "processed" / f"rank_vectors_rep{rep}.csv"
        centered.index.name = "systematic_name"
        centered.to_csv(out_csv)

        vals = centered.values
        print(f"  Wrote {out_csv.name}: {centered.shape[0]:,} x {centered.shape[1]}")
        print(f"  Value range: [{vals.min():.1f}, {vals.max():.1f}]")


if __name__ == "__main__":
    main()
