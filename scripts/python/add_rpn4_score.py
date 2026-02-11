#!/usr/bin/env python3
"""
Add Rpn4_score to regulon_scores.csv using info/Rpn4_target_genes.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

TPM = Path("yeast_ORF_TPM_matrix_labeled.csv")
REG = Path("regulon_scores.csv")
RPN4 = Path("info/Rpn4_target_genes.txt")

OUT = Path("regulon_scores.csv")  # overwrite in place


def canon(x):
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none"} else s.upper()


def main():
    for p in [TPM, REG, RPN4]:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    # ---- load Rpn4 genes ----
    rpn4_genes = {
        canon(line)
        for line in RPN4.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }

    print(f"Loaded {len(rpn4_genes)} Rpn4 target genes")

    # ---- load TPM matrix ----
    df = pd.read_csv(TPM)

    if "systematic_name" in df.columns:
        gene_col = "systematic_name"
    else:
        gene_col = df.columns[0]

    df[gene_col] = df[gene_col].astype(str).map(canon)

    sample_cols = [
        c for c in df.columns
        if c not in {gene_col, "common_name", "gene_category"}
    ]

    X = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.index = df[gene_col]

    # ---- log + gene-wise z-score ----
    X_log = np.log2(X + 1.0)
    mu = X_log.mean(axis=1)
    sd = X_log.std(axis=1).replace(0, np.nan)
    X_z = (X_log.sub(mu, axis=0)).div(sd, axis=0).fillna(0.0)

    # ---- subset Rpn4 genes ----
    present = sorted(set(X_z.index) & rpn4_genes)
    if not present:
        raise SystemExit("No Rpn4 genes found in TPM matrix")

    print(f"{len(present)} Rpn4 genes found in expression matrix")

    rpn4_score = X_z.loc[present].mean(axis=0)

    # ---- add to regulon matrix ----
    reg = pd.read_csv(REG, index_col=0)

    missing = set(rpn4_score.index) - set(reg.index)
    if missing:
        raise SystemExit(
            f"Sample mismatch between TPM and regulon_scores.csv. Example: {list(missing)[:5]}"
        )

    reg["Rpn4_score"] = rpn4_score.loc[reg.index]

    reg.to_csv(OUT)
    print("Updated regulon_scores.csv with Rpn4_score")


if __name__ == "__main__":
    main()
