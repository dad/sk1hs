#!/usr/bin/env python3
"""
Plot regulon scores on the UMAP (with arrows), plus timecourses.

Inputs (expected in project root):
  - regulon_scores.csv   (written by scripts/make_regulon_scores.py)

Outputs (written in project root):
  - regulon_umap_<TF>.png
  - regulon_time_<TF>.png

Style:
  - points colored by score using viridis
  - timepoint legend uses gray markers
  - arrows per (temp, rep)
  - extra "origin" arrows from 30C_15min rep1/rep2 to the first timepoint of every (temp,rep) trajectory
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(".")
REGULON_CSV = ROOT / "regulon_scores.csv"

# Match your by-rep timepoint marker vocabulary
TIME_MARKERS = {
    5:   "o",
    15:  "s",
    30:  "^",
    60:  "D",
    90:  "v",
    120: "P",
}

def _as_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _legend_timepoints_gray(ax, fontsize=9):
    handles = []
    for t, m in TIME_MARKERS.items():
        handles.append(
            Line2D(
                [0], [0],
                marker=m, linestyle="None",
                markerfacecolor="0.6",
                markeredgecolor="0.6",
                markersize=9,
                label=f"{t} min",
            )
        )
    ax.legend(handles=handles, title="Timepoint", frameon=False, fontsize=fontsize)

def plot_umap_regulon(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    out_png: Path,
    cmap_name: str = "viridis",
):
    # Basic checks
    req = ["UMAP1", "UMAP2", "temp_C", "time_min", "rep", score_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing columns for UMAP plot: {missing}")

    d = df.dropna(subset=["UMAP1", "UMAP2", "temp_C", "time_min", "rep", score_col]).copy()
    _as_numeric(d, ["UMAP1", "UMAP2", "temp_C", "time_min", "rep", score_col])

    # Color scaling
    vals = d[score_col].to_numpy(dtype=float)
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = -1.0, 1.0

    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # --- Draw arrows per (temp, rep) using light gray
    arrow_color = "0.55"
    for (_, _), g in d.groupby(["temp_C", "rep"]):
        g = g.sort_values("time_min")
        if g.shape[0] < 2:
            continue
        xs = g["UMAP1"].to_numpy()
        ys = g["UMAP2"].to_numpy()
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.55, color=arrow_color),
            )

    # --- Add origin arrows: from 30C 15min rep{1,2} to earliest timepoint for each (temp,rep)
    # These should be subtle: same gray, slightly thinner, dashed
    origin_lw = 0.9
    origin_alpha = 0.50

    for rep in sorted(d["rep"].dropna().unique()):
        o = d[(d["temp_C"] == 30) & (d["time_min"] == 15) & (d["rep"] == rep)]
        if o.empty:
            continue
        ox, oy = float(o.iloc[0]["UMAP1"]), float(o.iloc[0]["UMAP2"])

        for temp, g in d[d["rep"] == rep].groupby("temp_C"):
            # Skip the origin condition itself (30C/15)
            g2 = g[~((g["temp_C"] == 30) & (g["time_min"] == 15))]
            if g2.empty:
                continue
            g2 = g2.sort_values("time_min")
            tx, ty = float(g2.iloc[0]["UMAP1"]), float(g2.iloc[0]["UMAP2"])

            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(ox, oy),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=origin_lw,
                    alpha=origin_alpha,
                    color=arrow_color,
                    linestyle="--",
                ),
            )

    # --- Scatter points by timepoint marker, colored by score
    for t, marker in TIME_MARKERS.items():
        g = d[d["time_min"] == t]
        if g.empty:
            continue
        ax.scatter(
            g["UMAP1"],
            g["UMAP2"],
            c=g[score_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=170,
            marker=marker,
            edgecolors="none",
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"{title} score (mean z)")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)

    # Timepoint legend with gray markers
    _legend_timepoints_gray(ax, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_timecourse(df: pd.DataFrame, score_col: str, title: str, out_png: Path):
    req = ["temp_C", "time_min", score_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing columns for timecourse plot: {missing}")

    d = df.dropna(subset=["temp_C", "time_min", score_col]).copy()
    _as_numeric(d, ["temp_C", "time_min", score_col])

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    agg = d.groupby(["temp_C", "time_min"], as_index=False)[score_col].mean()

    for temp, g in agg.groupby("temp_C"):
        g = g.sort_values("time_min")
        ax.plot(g["time_min"], g[score_col], marker="o", label=f"{temp:g}°C")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"{title} score (mean z)")
    ax.legend(title="Temperature", frameon=False, ncol=2, fontsize=9)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    if not REGULON_CSV.exists():
        raise SystemExit(f"ERROR: can't find {REGULON_CSV}. Run scripts/make_regulon_scores.py first.")

    df = pd.read_csv(REGULON_CSV)

    # Find all regulon score columns
    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        raise SystemExit("ERROR: no *_score columns found in regulon_scores.csv")

    # Make everything in one run
    for sc in score_cols:
        name = sc.replace("_score", "")
        out_umap = ROOT / f"regulon_umap_{name}.png"
        out_time = ROOT / f"regulon_time_{name}.png"

        plot_umap_regulon(df, sc, name, out_umap, cmap_name="viridis")
        plot_timecourse(df, sc, name, out_time)

    print(f"Done. Wrote {len(score_cols)} regulon UMAPs + timecourses.")

if __name__ == "__main__":
    main()
