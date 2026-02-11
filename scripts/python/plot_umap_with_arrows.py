#!/usr/bin/env python3
"""
Plot UMAP with points colored by temperature and shaped by timepoint,
plus arrows connecting timepoints within each temperature.

Input:  proj_umap_samples.csv
Output: proj_UMAP_with_arrows.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_CSV = Path("proj_umap_samples.csv")
OUT_PNG = Path("proj_UMAP_with_arrows.png")

df = pd.read_csv(IN_CSV, index_col=0)

required = {"UMAP1", "UMAP2", "temp_C", "time_min"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"ERROR: {IN_CSV} missing columns: {sorted(missing)}")

# Clean / ensure numeric
df["temp_C"] = pd.to_numeric(df["temp_C"], errors="coerce")
df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce")

# Marker shapes by timepoint
marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
unique_times = sorted([t for t in df["time_min"].dropna().unique()])
time_to_marker = {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(unique_times)}

# Color mapping by temperature
good = df["temp_C"].notna() & df["time_min"].notna()
temps = df.loc[good, "temp_C"]
norm = plt.Normalize(temps.min(), temps.max())
cmap = plt.cm.viridis

fig, ax = plt.subplots(figsize=(7.2, 6.4))

# Scatter points grouped by timepoint for marker control
for t in unique_times:
    sub = df[df["time_min"] == t]
    if sub.empty:
        continue
    colors = cmap(norm(sub["temp_C"]))
    ax.scatter(
        sub["UMAP1"], sub["UMAP2"],
        s=70, alpha=0.85,
        marker=time_to_marker[t],
        c=colors, edgecolors="none",
        label=f"{int(t)} min" if float(t).is_integer() else f"{t} min"
    )

# Arrows connecting timepoints within each temperature
# We group by *exact* temperature values present in the metadata.
for temp in sorted(df["temp_C"].dropna().unique()):
    dtemp = df[df["temp_C"] == temp].dropna(subset=["time_min"])
    if dtemp.shape[0] < 2:
        continue
    dtemp = dtemp.sort_values("time_min")

    xs = dtemp["UMAP1"].to_numpy()
    ys = dtemp["UMAP2"].to_numpy()

    # Draw arrows segment-by-segment
    for i in range(len(xs) - 1):
        ax.annotate(
            "", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.2,
                alpha=0.6,
                color="0.25"
            )
        )

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Temperature (°C)")

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(title="Timepoint", frameon=False, fontsize=9, title_fontsize=10, loc="best")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)

print(f"Wrote {OUT_PNG}")
