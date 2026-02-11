#!/usr/bin/env python3
"""
Add regulon/category labels to an existing TPM matrix.

Input:
  yeast_ORF_TPM_matrix_labeled.csv
    - must include two identifier columns:
        systematic_name (e.g., YAL005C)
        common_name     (e.g., SSA1; systematic repeated if no common)
    - plus TPM columns for samples

Gene categories (priority order):
  1) Hsf1 target
  2) Hac1 target
  3) Msn2 target
  4) Sum1 target
  5) RPG (common name starts with RPL or RPS)
  6) RiBi gene
  7) other

Files expected:
  info/Hsf1_target_genes_42.xlsx
  info/Hac1 targets.xlsx
  info/Msn24_target_genes.xlsx
  info/sum1_target_genes.txt
  info/ribi_genes.txt
"""

from pathlib import Path
import re
import sys
import pandas as pd

PROJECT_ROOT = Path(".")
IN_CSV  = PROJECT_ROOT / "yeast_ORF_TPM_matrix_labeled.csv"
OUT_CSV = PROJECT_ROOT / "yeast_ORF_TPM_matrix_labeled_with_categories.csv"

HSF1_XLSX = PROJECT_ROOT / "info" / "Hsf1_target_genes_42.xlsx"
HAC1_XLSX = PROJECT_ROOT / "info" / "Hac1 targets.xlsx"
MSN2_XLSX = PROJECT_ROOT / "info" / "Msn24_target_genes.xlsx"

SUM1_TXT  = PROJECT_ROOT / "info" / "sum1_target_genes.txt"
RIBI_TXT  = PROJECT_ROOT / "info" / "ribi_genes.txt"

# Systematic name pattern (handles -A etc.)
SYS_RE = re.compile(r"^Y[A-P][LR]\d{3}[CW](?:-[A-Z])?$")

def norm(x: str) -> str:
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return s.upper()

def read_txt_gene_list(path: Path) -> set[str]:
    genes: set[str] = set()
    if not path.exists():
        print(f"WARNING: missing {path} (set will be empty)", file=sys.stderr)
        return genes
    for line in path.read_text().splitlines():
        t = norm(line)
        if t:
            genes.add(t)
    return genes

def extract_tokens_from_excel(xlsx_path: Path) -> set[str]:
    """
    Read all sheets, all cells, and extract gene-like tokens.
    Keeps both systematic and common gene symbols (uppercased).
    """
    tokens: set[str] = set()
    if not xlsx_path.exists():
        print(f"WARNING: missing {xlsx_path} (set will be empty)", file=sys.stderr)
        return tokens

    xls = pd.ExcelFile(xlsx_path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet, dtype=str)
        for v in df.to_numpy().ravel():
            if v is None:
                continue
            s = norm(v)
            if not s:
                continue
            # split on non gene-ish delimiters
            for part in re.split(r"[^A-Z0-9\-]+", s):
                part = part.strip()
                if not part:
                    continue
                # Keep systematic names, and also plausible common symbols (A–Z/0–9/-)
                if SYS_RE.match(part) or re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{1,}", part):
                    tokens.add(part)
    return tokens

def find_name_cols(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    # Prefer explicit names
    sys_candidates = [c for c in cols if c.lower() in {"systematic_name", "systematic", "orf", "gene_id"}]
    common_candidates = [c for c in cols if c.lower() in {"common_name", "common", "gene_name", "symbol"}]
    if sys_candidates and common_candidates:
        return sys_candidates[0], common_candidates[0]
    # Fallback: first two columns
    if len(cols) < 2:
        raise ValueError("Input CSV must have at least two columns for gene identifiers.")
    print(f"NOTE: Using first two columns as identifiers: {cols[0]!r}, {cols[1]!r}", file=sys.stderr)
    return cols[0], cols[1]

def main():
    if not IN_CSV.exists():
        sys.exit(f"ERROR: missing input: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    sys_col, common_col = find_name_cols(df)

    sys_names = df[sys_col].astype(str).map(norm)
    common_names = df[common_col].astype(str).map(norm)

    # Load sets
    hsf1 = extract_tokens_from_excel(HSF1_XLSX)
    hac1 = extract_tokens_from_excel(HAC1_XLSX)
    msn2 = extract_tokens_from_excel(MSN2_XLSX)

    sum1 = read_txt_gene_list(SUM1_TXT)
    ribi = read_txt_gene_list(RIBI_TXT)

    cats = []
    for s, c in zip(sys_names, common_names):
        names = {s, c}

        if names & hsf1:
            cat = "Hsf1 target"
        elif names & hac1:
            cat = "Hac1 target"
        elif names & msn2:
            cat = "Msn2 target"
        elif names & sum1:
            cat = "Sum1 target"
        elif c.startswith("RPL") or c.startswith("RPS"):
            cat = "RPG"
        elif names & ribi:
            cat = "RiBi gene"
        else:
            cat = "other"

        cats.append(cat)

    # Insert gene_category right after common_name
    out = df.copy()
    insert_at = min(out.columns.get_loc(common_col) + 1, len(out.columns))
    out.insert(insert_at, "gene_category", cats)

    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV} (rows={out.shape[0]}, cols={out.shape[1]})")

if __name__ == "__main__":
    main()
