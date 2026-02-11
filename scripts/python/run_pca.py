#!/usr/bin/env python3
"""
Run PCA on the TPM matrix and plot with:
- color = temperature
- marker shape = timepoint

Requires sample column names formatted like: <temp>_<time>_<rep>
Example: 35_5_1  -> temp=35, time=5, rep=1
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

IN_CSV = Path("yeast_ORF_TPM_matrix_labeled_with_categories.csv")
OUT_PREFIX = Path("pca")

# --------------------
# Helper: parse sample labels
# --------------------
def parse_label(lbl: str):
    """
    Expect labels like '35_5_1' (temp_time_rep).
    Returns (temp, time, rep) as strings; temp and time also returned as floats when possible.
    """
    s = str(lbl).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+)$", s)
    if not m:
        return None, None, None
    temp, time, rep = m.group(1), m.group(2), m.group(3)
    return float(temp), float(time), rep

# --------------------
# Load data
# --------------------
df = pd.read_csv(IN_CSV)

meta_cols = ["systematic_name", "common_name", "gene_category"]
sample_cols = [c for c in df.columns if c not in meta_cols]

X = df[sample_cols].to_numpy(dtype=float)

# log2(TPM+1)
X_log = np.log2(X + 1)

# Z-score genes (rows)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X_log.T).T  # scale per gene

# PCA on samples
pca = PCA()
X_pca = pca.fit_transform(X_scaled.T)

pca_df = pd.DataFrame(
    X_pca,
    index=sample_cols,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)

# Add parsed metadata
temps = []
times = []
reps = []
bad = []
for lbl in pca_df.index:
    t, ti, r = parse_label(lbl)
    if t is None:
        bad.append(lbl)
    temps.append(t)
    times.append(ti)
    reps.append(r)

pca_df["temp_C"] = temps
pca_df["time_min"] = times
pca_df["rep"] = reps

if bad:
    print("WARNING: Some sample labels did not match <temp>_<time>_<rep> and will not be colored/shaped:")
    for b in bad[:10]:
        print("  ", b)
    if len(bad) > 10:
        print(f"  ... ({len(bad)-10} more)")

# Save outputs
pca_df.to_csv(f"{OUT_PREFIX}_samples.csv")

var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "variance_explained": pca.explained_variance_ratio_
})
var_df.to_csv(f"{OUT_PREFIX}_variance_explained.csv", index=False)

# --------------------
# Plotting
# --------------------
def plot_pc(x, y, fname):
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    # marker shapes by timepoint (extend if you have more timepoints)
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    unique_times = sorted([t for t in pd.unique(pca_df["time_min"]) if pd.notna(t)])
    time_to_marker = {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(unique_times)}

    # color by temperature (continuous colormap)
    # We'll only color rows with a parsed temperature; others get gray.
    has_temp = pca_df["temp_C"].notna()
    temps = pca_df.loc[has_temp, "temp_C"].astype(float)

    # Normalize temps for colormap
    if len(temps) > 0:
        tmin, tmax = temps.min(), temps.max()
        norm = plt.Normalize(vmin=tmin, vmax=tmax)
        cmap = plt.cm.viridis
    else:
        norm = None
        cmap = None

    # Plot each timepoint as a separate scatter, so markers differ
    for t in unique_times:
        sub = pca_df[pca_df["time_min"] == t]
        if sub.empty:
            continue
        colors = cmap(norm(sub["temp_C"])) if cmap is not None else None
        ax.scatter(sub[x], sub[y], s=70, alpha=0.85, marker=time_to_marker[t], c=colors, edgecolors="none", label=f"{int(t)} min" if float(t).is_integer() else f"{t} min")

    # Unparsed samples (if any)
    sub_bad = pca_df[pca_df["temp_C"].isna() | pca_df["time_min"].isna()]
    if not sub_bad.empty:
        ax.scatter(sub_bad[x], sub_bad[y], s=70, alpha=0.6, marker="o", c="0.6", edgecolors="none", label="unparsed label")

    # Colorbar
    if cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Temperature (°C)")

    ax.set_xlabel(f"{x} ({pca.explained_variance_ratio_[int(x[2:])-1]*100:.1f}%)")
    ax.set_ylabel(f"{y} ({pca.explained_variance_ratio_[int(y[2:])-1]*100:.1f}%)")

    ax.legend(title="Timepoint", frameon=False, fontsize=9, title_fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

plot_pc("PC1", "PC2", f"{OUT_PREFIX}_PC1_PC2.png")
plot_pc("PC1", "PC3", f"{OUT_PREFIX}_PC1_PC3.png")

print("PCA complete.")
print("Wrote:")
print(f"  {OUT_PREFIX}_samples.csv")
print(f"  {OUT_PREFIX}_variance_explained.csv")
print(f"  {OUT_PREFIX}_PC1_PC2.png")
print(f"  {OUT_PREFIX}_PC1_PC3.png")
