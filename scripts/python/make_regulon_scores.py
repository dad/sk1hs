#!/usr/bin/env python3
"""
Create regulon_scores.csv from your annotated TPM matrix.

Inputs:
  - yeast_ORF_TPM_matrix_labeled_with_categories.csv
      columns: systematic_name, common_name, gene_category, <sample columns...>
  - info/sum1_target_genes.txt
  - info/ribi_genes.txt
  - info/Hsf1_target_genes_42.xlsx
  - info/Hac1 targets.xlsx
  - info/Msn24_target_genes.xlsx

Outputs:
  - regulon_scores.csv
    index = sample name
    columns: temp_C, time_min, rep, UMAP1, UMAP2 (if available), and regulon scores
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

TPM_CSV  = Path("yeast_ORF_TPM_matrix_labeled_with_categories.csv")

SUM1_TXT = Path("info/sum1_target_genes.txt")
RIBI_TXT = Path("info/ribi_genes.txt")

GCN4_TXT = Path("info/gcn4_target_genes.txt")
HSF1_XLSX = Path("info/Hsf1_target_genes_42.xlsx")
HAC1_XLSX = Path("info/Hac1 targets.xlsx")
MSN2_XLSX = Path("info/Msn24_target_genes.xlsx")

RPN4_TXT = Path("info/rpn4_target_genes.txt")
UMAP_CSV = Path("proj_umap_samples.csv")  # optional; if present we’ll join UMAP coords/metadata

OUT = Path("regulon_scores.csv")

# systematic name pattern (handles -A etc.)
SYS_RE = re.compile(r"^Y[A-P][LR]\d{3}[CW](?:-[A-Z])?$")

def canon(x) -> str:
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return s.upper()

def load_txt_gene_list(path: Path) -> set[str]:
    genes = set()
    for line in path.read_text().splitlines():
        t = canon(line)
        if t:
            genes.add(t)
    return genes

def extract_tokens_from_excel(xlsx_path: Path) -> set[str]:
    """
    Reads all sheets/cells; extracts gene-like tokens.
    Requires openpyxl.
      conda install -c conda-forge openpyxl
    """
    tokens: set[str] = set()
    xls = pd.ExcelFile(xlsx_path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet, dtype=str)
        for v in df.to_numpy().ravel():
            if v is None:
                continue
            s = canon(v)
            if not s:
                continue
            for part in re.split(r"[^A-Z0-9\-]+", s):
                part = part.strip()
                if not part:
                    continue
                # keep systematic OR plausible gene tokens
                if SYS_RE.match(part) or re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{1,}", part):
                    tokens.add(part)
    return tokens

def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Z-score each row (gene) across columns (samples)."""
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd

def main():
    for p in [TPM_CSV, SUM1_TXT, RIBI_TXT, HSF1_XLSX, HAC1_XLSX, MSN2_XLSX]:
        if not p.exists():
            raise SystemExit(f"ERROR: missing required file: {p}")

    print("Loading TPM matrix:", TPM_CSV)
    df = pd.read_csv(TPM_CSV)

    for c in ["systematic_name", "common_name"]:
        if c not in df.columns:
            raise SystemExit(f"ERROR: {TPM_CSV} missing required column: {c}")

    meta_cols = [c for c in ["systematic_name", "common_name", "gene_category"] if c in df.columns]
    sample_cols = [c for c in df.columns if c not in meta_cols]

    # numeric TPMs
    X = df[sample_cols].to_numpy(dtype=float)

    # log2(TPM+1) then gene-wise z-score
    X_log = np.log2(X + 1.0)
    X_z = zscore_rows(X_log)

    sys_names = df["systematic_name"].astype(str).map(canon)
    common_names = df["common_name"].astype(str).map(canon)

    # Load regulons
    sum1 = load_txt_gene_list(SUM1_TXT)
    ribi = load_txt_gene_list(RIBI_TXT)
    gcn4 = load_txt_gene_list(GCN4_TXT)
    hsf1 = extract_tokens_from_excel(HSF1_XLSX)
    hac1 = extract_tokens_from_excel(HAC1_XLSX)
    msn2 = extract_tokens_from_excel(MSN2_XLSX)

    rpn4 = load_txt_gene_list(RPN4_TXT)
    # Helper: match gene list against either systematic or common names
    def mask_from_set(gset: set[str]) -> np.ndarray:
        return np.array([(s in gset) or (c in gset) for s, c in zip(sys_names, common_names)], dtype=bool)

    masks = {
        "Sum1": mask_from_set(sum1),
        "Hsf1": mask_from_set(hsf1),
        "Hac1": mask_from_set(hac1),
        "Msn2": mask_from_set(msn2),
        "Gcn4": mask_from_set(gcn4),
        "Rpn4": mask_from_set(rpn4),
        "RiBi": mask_from_set(ribi),
        "RPG":  np.array([c.startswith("RPL") or c.startswith("RPS") for c in common_names], dtype=bool),
    }

    # Compute regulon scores (mean z across genes)
    scores = pd.DataFrame(index=sample_cols)
    for reg, m in masks.items():
        n = int(m.sum())
        if n == 0:
            scores[f"{reg}_score"] = np.nan
            print(f"{reg}: matched 0 genes (CHECK LIST)")
        else:
            scores[f"{reg}_score"] = X_z[m, :].mean(axis=0)
            print(f"{reg}: matched {n} genes")

    # If UMAP file exists, join temp/time/rep + UMAP coords
    if UMAP_CSV.exists():
        um = pd.read_csv(UMAP_CSV, index_col=0)
        # keep only what we need if present
        keep = [c for c in ["UMAP1", "UMAP2", "temp_C", "time_min", "rep"] if c in um.columns]
        um = um[keep]
        out = um.join(scores, how="left")
        if out.isna().any().any():
            # This usually means sample name mismatch
            missing = out.index[out[scores.columns[0]].isna()].tolist()[:10]
            raise SystemExit(f"ERROR: sample-name mismatch between {UMAP_CSV} and TPM columns. Examples: {missing}")
    else:
        out = scores.copy()

    out.to_csv(OUT)
    print(f"\nWrote {OUT} with shape {out.shape}")

if __name__ == "__main__":
    main()
