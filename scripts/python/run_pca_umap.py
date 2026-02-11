#!/usr/bin/env python3
"""
Run PCA and UMAP on the TPM matrix and plot with:
- color = temperature
- marker shape = timepoint

Sample labels must look like: <temp>_<time>_<rep>
Example: 35_5_1
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap

IN_CSV = Path("yeast_ORF_TPM_matrix_labeled_with_categories.csv")
OUT_PREFIX = Path("proj")

# --------------------
# Helper: parse sample labels
# --------------------
def parse_label(lbl: str):
    m = re.match(r"^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+)$", lbl)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), m.group(3)

# --------------------
# Load + preprocess
# --------------------
df = pd.read_csv(IN_CSV)

meta_cols = ["systematic_name", "common_name", "gene_category"]
sample_cols = [c for c in df.columns if c not in meta_cols]

X = df[sample_cols].to_numpy(dtype=float)

# log2(TPM + 1)
X_log = np.log2(X + 1)

# Z-score genes
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X_log.T).T

# Samples as rows
X_samples = X_scaled.T

# --------------------
# Parse metadata
# --------------------
temps, times, reps = [], [], []
for lbl in sample_cols:
    t, ti, r = parse_label(lbl)
    temps.append(t)
    times.append(ti)
    reps.append(r)

meta = pd.DataFrame({
    "sample": sample_cols,
    "temp_C": temps,
    "time_min": times,
    "rep": reps
}).set_index("sample")

# --------------------
# PCA
# --------------------
pca = PCA()
X_pca = pca.fit_transform(X_samples)

pca_df = pd.DataFrame(
    X_pca[:, :5],
    index=sample_cols,
    columns=[f"PC{i+1}" for i in range(5)]
).join(meta)

pca_df.to_csv(f"{OUT_PREFIX}_pca_samples.csv")

# --------------------
# UMAP
# --------------------
um = umap.UMAP(
    n_neighbors=10,
    min_dist=0.3,
    metric="euclidean",
    random_state=42
)
X_umap = um.fit_transform(X_samples)

umap_df = pd.DataFrame(
    X_umap,
    index=sample_cols,
    columns=["UMAP1", "UMAP2"]
).join(meta)

umap_df.to_csv(f"{OUT_PREFIX}_umap_samples.csv")

# --------------------
# Plotting helper
# --------------------
def plot_embedding(df, x, y, fname):
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    unique_times = sorted(df["time_min"].dropna().unique())
    time_to_marker = {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(unique_times)}

    good = df["temp_C"].notna() & df["time_min"].notna()
    temps = df.loc[good, "temp_C"]

    norm = plt.Normalize(temps.min(), temps.max())
    cmap = plt.cm.viridis

    for t in unique_times:
        sub = df[df["time_min"] == t]
        colors = cmap(norm(sub["temp_C"]))
        ax.scatter(sub[x], sub[y], s=70, alpha=0.85,
                   marker=time_to_marker[t],
                   c=colors, edgecolors="none",
                   label=f"{int(t)} min")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title="Timepoint", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# --------------------
# Make plots
# --------------------
plot_embedding(pca_df, "PC1", "PC2", f"{OUT_PREFIX}_PCA_PC1_PC2.png")
plot_embedding(umap_df, "UMAP1", "UMAP2", f"{OUT_PREFIX}_UMAP.png")

print("PCA + UMAP complete.")
print("Wrote:")
print(f"  {OUT_PREFIX}_pca_samples.csv")
print(f"  {OUT_PREFIX}_umap_samples.csv")
print(f"  {OUT_PREFIX}_PCA_PC1_PC2.png")
print(f"  {OUT_PREFIX}_UMAP.png")
