#!/usr/bin/env python3
"""
Prepare baseline-subtracted, median-centered rank vectors for clustering.

Input
-----
data/processed/SK1_tidy_TPM.csv
    Tidy-format TPM data with ``rank_mean_TPM`` column (1 = highest
    expression within each temp x time group).

Outputs
-------
data/processed/rank_vectors.csv
    6,669 genes x 37 columns (systematic_name + 36 numeric).
    Column naming: ``t{temp}_m{time}`` -- e.g. ``t35_m5``, ``t42_m120``.
    Columns are ordered with time varying fastest within temperature
    blocks, so temperature-level or time-level randomisation is easy.

data/processed/rank_vectors_meta.json
    Documents the column structure including ``temp_blocks`` and
    ``time_slices`` index arrays for downstream randomisation.

Processing steps
    1. Pivot tidy data to gene x condition rank matrix (6,669 x 37).
    2. Subtract the 30 C / 15 min baseline from every column.
    3. Drop the (now-zero) baseline column -> 6,669 x 36.
    4. Subtract gene-wise median (centre each row).
    5. Re-order and rename columns; write CSV + JSON metadata.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT / "data" / "processed" / "SK1_tidy_TPM.csv"
OUT_CSV = PROJECT / "data" / "processed" / "rank_vectors.csv"
OUT_META = PROJECT / "data" / "processed" / "rank_vectors_meta.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_TEMP = 30
BASELINE_TIME = 15
HEAT_TEMPS = [35, 37, 39, 42, 44, 46]
TIMES = [5, 15, 30, 60, 90, 120]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_and_validate(path: Path) -> pd.DataFrame:
    """Load tidy TPM CSV and check for expected structure."""
    df = pd.read_csv(path)
    required = {"systematic_name", "temp", "time", "rank_mean_TPM"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_genes = df["systematic_name"].nunique()
    n_conditions = df.groupby(["temp", "time"]).ngroups
    print(f"Loaded {path.name}: {n_genes:,} genes, {n_conditions} conditions")

    # Verify baseline exists
    baseline = df[(df["temp"] == BASELINE_TEMP) & (df["time"] == BASELINE_TIME)]
    if baseline.empty:
        raise ValueError(
            f"Baseline condition ({BASELINE_TEMP}C, {BASELINE_TIME}min) not found"
        )
    return df


def pivot_to_rank_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy data to gene x (temp, time) matrix of rank_mean_TPM."""
    wide = df.pivot_table(
        index="systematic_name",
        columns=["temp", "time"],
        values="rank_mean_TPM",
        aggfunc="first",
    )
    assert wide.notna().all().all(), "Unexpected NaN in pivoted rank matrix"
    print(f"Pivoted to rank matrix: {wide.shape[0]:,} genes x {wide.shape[1]} conditions")
    return wide


def subtract_baseline(wide: pd.DataFrame) -> pd.DataFrame:
    """Subtract 30C/15min baseline and drop the baseline column."""
    baseline_col = (BASELINE_TEMP, BASELINE_TIME)
    if baseline_col not in wide.columns:
        raise ValueError(f"Baseline column {baseline_col} not found in matrix")

    baseline = wide[baseline_col]
    delta = wide.sub(baseline, axis=0)
    delta = delta.drop(columns=[baseline_col])

    print(f"Baseline subtracted: {delta.shape[1]} conditions remain")
    return delta


def center_by_gene_median(delta: pd.DataFrame) -> pd.DataFrame:
    """Subtract the row-wise (gene-wise) median from each row."""
    row_medians = delta.median(axis=1)
    centered = delta.sub(row_medians, axis=0)

    # Sanity: each row's median should now be ~0
    residual = centered.median(axis=1).abs()
    max_residual = residual.max()
    assert max_residual < 1e-8, (
        f"Median centering failed: max residual = {max_residual}"
    )
    print(f"Median-centered: max residual = {max_residual:.2e}")
    return centered


def order_and_rename(centered: pd.DataFrame) -> pd.DataFrame:
    """
    Re-order columns as time-within-temperature blocks and rename
    from (temp, time) tuples to ``t{temp}_m{time}`` strings.
    """
    ordered_tuples = [
        (temp, time) for temp in HEAT_TEMPS for time in TIMES
    ]
    # Verify all expected columns are present
    missing = [c for c in ordered_tuples if c not in centered.columns]
    if missing:
        raise ValueError(f"Missing expected condition columns: {missing}")

    centered = centered[ordered_tuples]
    centered.columns = [f"t{t}_m{m}" for t, m in ordered_tuples]
    return centered


def build_metadata(columns: list[str]) -> dict:
    """Build a metadata dict documenting column structure."""
    # temp_blocks: for each temperature, which column indices belong to it
    temp_blocks = {}
    for i, temp in enumerate(HEAT_TEMPS):
        start = i * len(TIMES)
        temp_blocks[str(temp)] = list(range(start, start + len(TIMES)))

    # time_slices: for each timepoint, which column indices (across temps)
    time_slices = {}
    for j, time in enumerate(TIMES):
        time_slices[str(time)] = [
            i * len(TIMES) + j for i in range(len(HEAT_TEMPS))
        ]

    return {
        "description": "Baseline-subtracted, median-centered rank vectors",
        "baseline": {"temp": BASELINE_TEMP, "time": BASELINE_TIME},
        "temperatures": HEAT_TEMPS,
        "times_min": TIMES,
        "n_genes": len(columns),  # will be overwritten with actual count
        "n_conditions": len(columns),
        "column_order": "time varies fastest within temperature blocks",
        "columns": columns,
        "temp_blocks": temp_blocks,
        "time_slices": time_slices,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_and_validate(IN_CSV)
    wide = pivot_to_rank_matrix(df)
    delta = subtract_baseline(wide)
    centered = center_by_gene_median(delta)
    result = order_and_rename(centered)

    # Final validation
    n_genes = result.shape[0]
    n_conds = result.shape[1]
    expected_conds = len(HEAT_TEMPS) * len(TIMES)
    assert n_conds == expected_conds, (
        f"Expected {expected_conds} condition columns, got {n_conds}"
    )
    assert result.notna().all().all(), "Unexpected NaN in final matrix"

    # Write CSV
    result.index.name = "systematic_name"
    result.to_csv(OUT_CSV)
    print(f"Wrote {OUT_CSV.name}: {n_genes:,} genes x {n_conds} conditions")

    # Write metadata
    meta = build_metadata(list(result.columns))
    meta["n_genes"] = n_genes
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {OUT_META.name}")

    # Quick summary of value range
    vals = result.values
    print(f"  Value range: [{vals.min():.1f}, {vals.max():.1f}]")
    print(f"  Mean: {vals.mean():.4f}, Std: {vals.std():.1f}")


if __name__ == "__main__":
    main()
