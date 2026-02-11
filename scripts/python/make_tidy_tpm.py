#!/usr/bin/env python3
"""
Build a tidy TPM + rank dataset from the wide ORF TPM matrix.

Input:  yeast_ORF_TPM_matrix_labeled.csv  (genes x samples, wide format)
Output: SK1_tidy_TPM.csv  (one row per gene x temp x time)

Columns in output:
  systematic_name, common_name,
  temp, time,
  TPM_rep1, TPM_rep2, mean_TPM,
  rank_per_sample, rank_mean_TPM

rank_per_sample : each of the 74 sample columns is ranked independently
                  (1 = highest TPM), then the two replicate ranks are averaged.
rank_mean_TPM   : mean_TPM is ranked within each temp x time group (1 = highest).
"""

from pathlib import Path
import pandas as pd

IN_CSV  = Path("yeast_ORF_TPM_matrix_labeled.csv")
OUT_CSV = Path("SK1_tidy_TPM.csv")

ID_COLS = ["systematic_name", "common_name"]


def main():
    # ---- Step 1: Load and validate ------------------------------------------------
    df = pd.read_csv(IN_CSV)
    sample_cols = [c for c in df.columns if c not in ID_COLS]
    n_genes = len(df)
    print(f"Loaded {IN_CSV.name}: {n_genes:,} genes, {len(sample_cols)} samples")
    assert len(sample_cols) == 74, f"Expected 74 sample columns, got {len(sample_cols)}"

    # ---- Step 2: Per-sample ranks (wide, before melting) --------------------------
    rank_wide = df[sample_cols].rank(ascending=False, method="average")
    rank_wide["systematic_name"] = df["systematic_name"].values

    # ---- Step 3: Melt TPM and rank DataFrames to long format ----------------------
    long = df.melt(id_vars=ID_COLS, value_vars=sample_cols,
                   var_name="sample", value_name="TPM")
    long[["temp", "time", "rep"]] = (
        long["sample"].str.split("_", expand=True).astype(int)
    )

    rank_long = rank_wide.melt(id_vars=["systematic_name"], value_vars=sample_cols,
                               var_name="sample", value_name="sample_rank")
    rank_long[["temp", "time", "rep"]] = (
        rank_long["sample"].str.split("_", expand=True).astype(int)
    )

    # ---- Step 4: Pivot replicates into TPM_rep1 / TPM_rep2 -----------------------
    tpm_pivot = long.pivot_table(
        index=["systematic_name", "common_name", "temp", "time"],
        columns="rep", values="TPM", aggfunc="first"
    ).reset_index()
    tpm_pivot.columns.name = None
    tpm_pivot = tpm_pivot.rename(columns={1: "TPM_rep1", 2: "TPM_rep2"})

    # Mean of available replicates (NaN-safe)
    tpm_pivot["mean_TPM"] = tpm_pivot[["TPM_rep1", "TPM_rep2"]].mean(axis=1)

    # ---- Step 5: Average per-sample ranks across replicates -----------------------
    mean_sample_rank = (
        rank_long
        .groupby(["systematic_name", "temp", "time"])["sample_rank"]
        .mean()
        .reset_index()
        .rename(columns={"sample_rank": "rank_per_sample"})
    )

    result = tpm_pivot.merge(mean_sample_rank,
                             on=["systematic_name", "temp", "time"], how="left")

    # ---- Step 6: Rank mean_TPM within each temp x time group ----------------------
    result["rank_mean_TPM"] = (
        result
        .groupby(["temp", "time"])["mean_TPM"]
        .rank(ascending=False, method="average")
    )

    # ---- Step 7: Sort and order columns -------------------------------------------
    result = result.sort_values(
        ["temp", "time", "rank_mean_TPM"]
    ).reset_index(drop=True)

    col_order = [
        "systematic_name", "common_name",
        "temp", "time",
        "TPM_rep1", "TPM_rep2", "mean_TPM",
        "rank_per_sample", "rank_mean_TPM",
    ]
    result = result[col_order]

    # ---- Step 8: Validate and write -----------------------------------------------
    n_conditions = long.groupby(["temp", "time"]).ngroups
    expected = n_genes * n_conditions
    assert result.shape[0] == expected, (
        f"Row count mismatch: {result.shape[0]} != {expected} "
        f"({n_genes} genes x {n_conditions} conditions)"
    )
    assert result["mean_TPM"].notna().all(), "mean_TPM has unexpected NaN values"
    assert result["rank_mean_TPM"].notna().all(), "rank_mean_TPM has NaN values"
    assert result["rank_per_sample"].notna().all(), "rank_per_sample has NaN values"

    result.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.name}: {result.shape[0]:,} rows x {result.shape[1]} cols")
    print(f"  {n_genes:,} genes x {n_conditions} conditions")
    print(f"  Temps: {sorted(result['temp'].unique())}")
    print(f"  Times: {sorted(result['time'].unique())}")


if __name__ == "__main__":
    main()
