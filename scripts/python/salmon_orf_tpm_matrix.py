#!/usr/bin/env python3
"""
Build an ORF x sample TPM matrix from Salmon outputs, with:
- systematic_name (e.g., YAL005C)
- common_name (e.g., SSA1); if missing, repeats systematic_name
- sample columns labeled using info/Leah_samples_lookup.csv

This mapping is derived from the SAME reference used to build the Salmon index:
ref/orf_coding_all.fasta.gz
"""

from pathlib import Path
import pandas as pd
import re
import sys
import gzip

SALMON_DIR = Path("salmon_quant")
LOOKUP_CSV = Path("info/Leah_samples_lookup.csv")
FASTA_GZ = Path("ref/orf_coding_all.fasta.gz")
OUTFILE = Path("yeast_ORF_TPM_matrix_labeled.csv")

def extract_sample_id(name: str):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None

def looks_like_symbol(tok: str) -> bool:
    # True for typical yeast gene symbols like SSA1, HSP82, RPL3, etc.
    # False for tokens like SGDID:S000..., Chr, etc.
    if tok.startswith(("SGDID:", "Chr", "chr", "SGD:", "S000", "GeneID:", "protein")):
        return False
    if re.fullmatch(r"Y[A-P][LR]\d{3}[CW](?:-[A-Z])?", tok):
        # systematic name again -> treat as "no common name"
        return False
    return True

# ---- checks ----
for p in [SALMON_DIR, LOOKUP_CSV, FASTA_GZ]:
    if not p.exists():
        sys.exit(f"ERROR: required path not found: {p}")

# ---- load sample lookup ----
lk = pd.read_csv(LOOKUP_CSV)
required = {"sample id", "temp_time_rep"}
if not required.issubset(lk.columns):
    sys.exit(f"ERROR: {LOOKUP_CSV} must contain columns {sorted(required)}. Found: {list(lk.columns)}")
id_to_label = dict(zip(lk["sample id"].astype(int), lk["temp_time_rep"].astype(str)))

# ---- build systematic -> common name map from FASTA headers ----
sys_to_common = {}
with gzip.open(FASTA_GZ, "rt") as fh:
    for line in fh:
        if not line.startswith(">"):
            continue
        header = line[1:].strip()
        toks = header.split()
        if not toks:
            continue
        systematic = toks[0]
        common = systematic
        if len(toks) >= 2 and looks_like_symbol(toks[1]):
            common = toks[1]
        sys_to_common[systematic] = common

# ---- collect Salmon outputs (sorted by numeric sample id) ----
sample_dirs = [p for p in SALMON_DIR.iterdir() if p.is_dir()]
sample_dirs.sort(key=lambda p: (extract_sample_id(p.name) is None, extract_sample_id(p.name) or 10**9))

series_list = []
col_labels = []

for sample_dir in sample_dirs:
    quant = sample_dir / "quant.sf"
    if not quant.exists():
        continue

    sid = extract_sample_id(sample_dir.name)
    label = id_to_label.get(sid, sample_dir.name)

    df = pd.read_csv(quant, sep="\t", usecols=["Name", "TPM"])
    df["systematic_name"] = df["Name"].astype(str).str.split().str[0]
    tpm = df.groupby("systematic_name", as_index=True)["TPM"].sum()
    tpm.name = label

    series_list.append(tpm)
    col_labels.append(label)

if not series_list:
    sys.exit("ERROR: No quant.sf files found under salmon_quant/")

mat = pd.concat(series_list, axis=1).fillna(0.0)
mat = mat.loc[:, col_labels]  # lock order

# ---- add name columns, guaranteeing no blanks ----
mat.insert(0, "common_name", [sys_to_common.get(g, g) for g in mat.index])
mat.insert(0, "systematic_name", mat.index)

mat.to_csv(OUTFILE, index=False)
print(f"Wrote {OUTFILE} with shape {mat.shape}")
