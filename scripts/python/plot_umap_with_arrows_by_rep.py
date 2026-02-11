#!/usr/bin/env python3
"""
Plot UMAP with arrows showing time trajectories per replicate.

Input:  proj_umap_samples.csv
Output: proj_UMAP_with_arrows_by_rep.png

Features:
- Points are shaped by timepoint (5/15/30/60/90/120 min)
- Points are colored by temperature using a selectable colormap (default: turbo)
- Arrows connect timepoints within each (temp, rep) trajectory
- Optional: add "origin" arrows from (origin_temp, origin_time) for each replicate
  to the earliest timepoint of every other temperature trajectory in that replicate.

Example:
  python scripts/plot_umap_with_arrows_by_rep.py
  python scripts/plot_umap_with_arrows_by_rep.py --temp-cmap plasma --origin-temp 30 --origin-time 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


DEFAULT_IN = "proj_umap_samples.csv"
DEFAULT_OUT = "proj_UMAP_with_arrows_by_rep.png"


TIME_MARKERS = {
    5:  "o",
    15: "s",
    30: "^",
    60: "D",
    90: "v",
    120: "P",  # plus-filled
}


def plot_umap_with_arrows_by_rep(
    df: pd.DataFrame,
    out_png: Path,
    temp_cmap: str = "turbo",
    point_size: float = 140,
    arrow_lw: float = 1.2,
    arrow_alpha: float = 0.55,
    arrow_color: str = "0.25",
    add_origin_arrows: bool = True,
    origin_temp: float = 30.0,
    origin_time: float = 15.0,
    origin_arrow_style: str = "dashed",
    origin_arrow_alpha: float = 0.6,
    origin_arrow_lw: float = 1.4,
):
    # Basic checks
    required = {"UMAP1", "UMAP2", "temp_C", "time_min", "rep"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: missing columns in input CSV: {sorted(missing)}")

    # Colormap for temperature
    cmap = plt.get_cmap(temp_cmap)
    tmin = float(np.nanmin(df["temp_C"].values))
    tmax = float(np.nanmax(df["temp_C"].values))
    norm = colors.Normalize(vmin=tmin, vmax=tmax)

    fig, ax = plt.subplots(figsize=(8.6, 6.6))

    # Plot points grouped by timepoint, with temp-based coloring
    # (This keeps your legend as "Timepoint" using marker shapes)
    times_sorted = sorted({int(t) for t in df["time_min"].dropna().unique()})
    for t in times_sorted:
        marker = TIME_MARKERS.get(int(t), "o")
        d = df[df["time_min"] == t]
        ax.scatter(
            d["UMAP1"], d["UMAP2"],
            c=d["temp_C"], cmap=cmap, norm=norm,
            s=point_size, marker=marker,
            edgecolors="none",
            label=f"{int(t)} min"
        )

    # Arrows within each (temp, rep)
    for (_, _), d in df.groupby(["temp_C", "rep"]):
        d = d.dropna(subset=["time_min"]).sort_values("time_min")
        if d.shape[0] < 2:
            continue
        xs = d["UMAP1"].to_numpy()
        ys = d["UMAP2"].to_numpy()
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=arrow_lw,
                    alpha=arrow_alpha,
                    color=arrow_color,
                ),
            )

    # Optional: origin arrows from (origin_temp, origin_time) per rep to each temp trajectory's first timepoint
    if add_origin_arrows:
        # Find origin point for each rep
        origin_df = df[(df["temp_C"] == origin_temp) & (df["time_min"] == origin_time)].copy()

        if origin_df.empty:
            print(
                f"WARNING: no origin points found for temp={origin_temp} time={origin_time}. "
                "Skipping origin arrows."
            )
        else:
            # If multiple (shouldn't happen), take the first for each rep
            origin_by_rep = (
                origin_df.sort_values(["rep"])
                .groupby("rep", as_index=False)
                .head(1)
                .set_index("rep")
            )

            # For each replicate, connect origin to earliest timepoint of each temperature trajectory
            for rep, o in origin_by_rep.iterrows():
                ox, oy = float(o["UMAP1"]), float(o["UMAP2"])

                drep = df[df["rep"] == rep].copy()

                # For each temperature, get earliest timepoint sample
                for temp, dtemp in drep.groupby("temp_C"):
                    # Optionally skip drawing arrow to its own origin temperature trajectory
                    # (keeps things less cluttered)
                    if float(temp) == float(origin_temp):
                        continue

                    dtemp = dtemp.dropna(subset=["time_min"]).sort_values("time_min")
                    if dtemp.empty:
                        continue
                    first = dtemp.iloc[0]
                    fx, fy = float(first["UMAP1"]), float(first["UMAP2"])

                    ax.annotate(
                        "",
                        xy=(fx, fy),
                        xytext=(ox, oy),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=origin_arrow_lw,
                            alpha=origin_arrow_alpha,
                            color=arrow_color,
                            linestyle=origin_arrow_style,
                        ),
                    )

    # Colorbar for temperature
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    # Custom legend with gray markers (timepoint only)
    from matplotlib.lines import Line2D
    legend_handles = []
    for t, m in TIME_MARKERS.items():
        legend_handles.append(
            Line2D([0], [0], marker=m, linestyle='None',
                   markerfacecolor='0.6', markeredgecolor='0.6',
                   markersize=9, label=f"{t} min")
        )
    ax.legend(handles=legend_handles, title="Timepoint",
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default=DEFAULT_IN, help=f"Input CSV (default: {DEFAULT_IN})")
    ap.add_argument("--out-png", default=DEFAULT_OUT, help=f"Output PNG (default: {DEFAULT_OUT})")
    ap.add_argument("--temp-cmap", default="turbo",
                    help="Matplotlib colormap name for temperature (default: turbo). "
                         "Examples: plasma, inferno, magma, cividis, Spectral.")
    ap.add_argument("--point-size", type=float, default=140.0)
    ap.add_argument("--arrow-lw", type=float, default=1.2)
    ap.add_argument("--arrow-alpha", type=float, default=0.55)

    ap.add_argument("--add-origin-arrows", action="store_true",
                    help="Add arrows from origin (origin_temp, origin_time) per replicate to each temp trajectory start.")
    ap.add_argument("--no-origin-arrows", action="store_true",
                    help="Disable origin arrows (overrides --add-origin-arrows).")
    ap.add_argument("--origin-temp", type=float, default=30.0)
    ap.add_argument("--origin-time", type=float, default=15.0)
    ap.add_argument("--origin-arrow-style", default="dashed", choices=["solid", "dashed", "dashdot", "dotted"])
    ap.add_argument("--origin-arrow-alpha", type=float, default=0.6)
    ap.add_argument("--origin-arrow-lw", type=float, default=1.4)

    args = ap.parse_args()

    add_origin = True
    if args.no_origin_arrows:
        add_origin = False
    if args.add_origin_arrows:
        add_origin = True

    df = pd.read_csv(Path(args.in_csv), index_col=0)

    plot_umap_with_arrows_by_rep(
        df=df,
        out_png=Path(args.out_png),
        temp_cmap=args.temp_cmap,
        point_size=args.point_size,
        arrow_lw=args.arrow_lw,
        arrow_alpha=args.arrow_alpha,
        add_origin_arrows=add_origin,
        origin_temp=args.origin_temp,
        origin_time=args.origin_time,
        origin_arrow_style=args.origin_arrow_style,
        origin_arrow_alpha=args.origin_arrow_alpha,
        origin_arrow_lw=args.origin_arrow_lw,
    )


if __name__ == "__main__":
    main()
