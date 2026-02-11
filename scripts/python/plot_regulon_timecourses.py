#!/usr/bin/env python3
"""
Plot SK1 regulon timecourses with the special 30°C baseline rule.

Key rule (per your instructions):
- Do NOT plot a 30°C timecourse (only two 30_15_{rep} samples).
- Treat 30_15_1 and 30_15_2 as t=0 baselines for ALL other temperatures, replicate-matched.
  That is:
    baseline(rep1) = score at sample "30_15_1"
    baseline(rep2) = score at sample "30_15_2"
- For each other temperature (e.g. 35, 37, 39...), plot timepoints 5–120 min on a time axis:
    t_plot = [0, 5, 15, 30, 60, 90, 120]
  where t=0 is the replicate-matched 30_15 baseline point.

Outputs:
- timecourse_plots/per_regulon/<Regulon>.png/.pdf
- timecourse_plots/per_temperature/<Temp>C.png/.pdf

Style:
- Per regulon: color = temperature (discrete, non-viridis palette), marker = timepoint shape
- Thin replicate traces + thick mean trace
- Global y-limits consistent across ALL per-regulon panels
- Per temperature: color = regulon (qualitative palette), endpoint labels (minimal legend)

Assumptions about input:
- regulon_scores.csv has either:
  (A) a column like 'sample'/'sample_name' containing strings like "35_60_2"
      OR
  (B) sample names are in the index
- Regulon score columns end with "_score" (e.g., "Rpn4_score")

Usage:
  python plot_regulon_timecourses.py \
    --csv "/Users/pincus/Library/CloudStorage/Box-Box/Pincus Lab/Leah's SK1 RNA Seq/regulon_scores.csv"

Optional:
  --delta               plot Δscore relative to baseline (score - baseline)
  --no-endpoint-labels  don't label regulons at endpoint in per-temp plots
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SAMPLE_COL_CANDIDATES = ["sample", "sample_name", "Sample", "SampleName", "name"]
TIMEPOINTS = [0, 5, 15, 30, 60, 90, 120]

# Marker mapping used consistently across all plots (timepoint shape scheme)
TIME_MARKERS = {
    0: "o",
    5: "s",
    15: "^",
    30: "D",
    60: "v",
    90: "P",
    120: "X",
}

SAMPLE_RE = re.compile(r"^(?P<temp>-?\d+)[_\-](?P<time>\d+)[_\-](?P<rep>\d+)$")


def parse_sample_name(s: str):
    m = SAMPLE_RE.match(str(s).strip())
    if not m:
        return None
    return int(m.group("temp")), int(m.group("time")), int(m.group("rep"))


def find_sample_series(df: pd.DataFrame) -> pd.Series:
    for c in SAMPLE_COL_CANDIDATES:
        if c in df.columns:
            return df[c].astype(str)
    # fallback: index
    return df.index.to_series().astype(str)


def pick_score_columns(df: pd.DataFrame):
    score_cols = [c for c in df.columns if str(c).endswith("_score")]
    if not score_cols:
        raise ValueError("No regulon score columns found ending with '_score'.")
    return score_cols


def ensure_output_dirs(base: Path):
    (base / "per_regulon").mkdir(parents=True, exist_ok=True)
    (base / "per_temperature").mkdir(parents=True, exist_ok=True)


def discrete_palette(n: int, cmap_name: str):
    cmap = plt.get_cmap(cmap_name)
    # Evenly spaced discrete colors
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def compute_dataset_long(df: pd.DataFrame, sample_s: pd.Series, score_cols: list[str]) -> pd.DataFrame:
    """
    Build long-form table with columns: temp, time, rep, regulon, score

    Preferred path:
      If df already contains temp/time/rep columns (e.g. temp_C, time_min, rep),
      use those directly (more reliable than parsing sample strings).

    Fallback:
      Parse sample name strings like "35_60_2".
    """
    META_TEMP_COLS = ["temp_C", "temp", "temperature"]
    META_TIME_COLS = ["time_min", "time", "minute", "minutes"]
    META_REP_COLS  = ["rep", "Rep", "replicate"]

    def first_present(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    temp_col = first_present(META_TEMP_COLS)
    time_col = first_present(META_TIME_COLS)
    rep_col  = first_present(META_REP_COLS)

    if temp_col and time_col and rep_col:
        meta = df[[temp_col, time_col, rep_col]].rename(
            columns={temp_col: "temp", time_col: "time", rep_col: "rep"}
        )
        meta["temp"] = meta["temp"].astype(float).round().astype(int)
        meta["time"] = meta["time"].astype(float).round().astype(int)
        meta["rep"]  = meta["rep"].astype(int)

        long = (
            df[score_cols]
            .join(meta)
            .melt(id_vars=["temp", "time", "rep"], var_name="regulon", value_name="score")
        )
        return long

    parsed = sample_s.map(parse_sample_name)
    ok = parsed.notna()
    if ok.sum() == 0:
        raise ValueError(
            "Could not parse any sample names and did not find temp/time/rep columns."
        )

    df2 = df.loc[ok].copy()
    meta = parsed.loc[ok].apply(pd.Series)
    meta.columns = ["temp", "time", "rep"]

    long = (
        df2[score_cols]
        .join(meta)
        .reset_index(drop=True)
        .melt(id_vars=["temp", "time", "rep"], var_name="regulon", value_name="score")
    )
    return long


def build_baseline_augmented(long: pd.DataFrame, delta: bool) -> pd.DataFrame:
    """
    For each rep:
      baseline is (temp=30, time=15)
    For each other temp:
      include a synthetic timepoint at t_plot=0 equal to the baseline score (per rep, per regulon),
      and then include that temp's existing timepoints (5..120) at their times.
    Optionally compute delta vs baseline.
    """
    # Baselines per (rep, regulon)
    base = long[(long["temp"] == 30) & (long["time"] == 15)].copy()
    if base.empty:
        raise ValueError("No baseline rows found for (temp=30, time=15). Expected samples 30_15_1 and 30_15_2.")

    base = base[["rep", "regulon", "score"]].rename(columns={"score": "baseline_score"})

    # Only keep temps != 30 for plotting (since we don't want a 30°C timecourse)
    lt = long[long["temp"] != 30].copy()

    # Join baseline score to all rows by rep+regulon
    lt = lt.merge(base, on=["rep", "regulon"], how="left")
    if lt["baseline_score"].isna().any():
        missing = lt[lt["baseline_score"].isna()][["rep", "regulon"]].drop_duplicates()
        raise ValueError(
            "Missing baseline for some (rep, regulon) pairs. "
            f"Example missing rows:\n{missing.head(10)}"
        )

    # Add baseline point at time=0 for every (temp!=30, rep, regulon)
    temps = sorted(lt["temp"].unique().tolist())
    reps = sorted(lt["rep"].unique().tolist())
    regulons = sorted(lt["regulon"].unique().tolist())

    base_points = []
    for t in temps:
        for r in reps:
            # baseline scores for this rep across all regulons
            br = base[base["rep"] == r]
            if br.empty:
                continue
            btmp = br.copy()
            btmp["temp"] = t
            btmp["time"] = 0
            btmp["score"] = btmp["baseline_score"]
            base_points.append(btmp[["temp", "time", "rep", "regulon", "score", "baseline_score"]])
    base_points = pd.concat(base_points, ignore_index=True) if base_points else pd.DataFrame()

    # Keep only timepoints of interest from the non-30 temps
    lt = lt[lt["time"].isin([5, 15, 30, 60, 90, 120])].copy()

    aug = pd.concat([base_points, lt[["temp", "time", "rep", "regulon", "score", "baseline_score"]]], ignore_index=True)

    if delta:
        aug["score"] = aug["score"] - aug["baseline_score"]

    # Ensure ordering helpers
    aug["time"] = aug["time"].astype(int)
    aug["temp"] = aug["temp"].astype(int)
    aug["rep"] = aug["rep"].astype(int)

    return aug


def plot_per_regulon(aug: pd.DataFrame, outdir: Path, delta: bool):
    regulons = sorted(aug["regulon"].unique().tolist())
    temps = sorted(aug["temp"].unique().tolist())

    # Non-viridis discrete palette for temperature curves
    # Use a diverging-ish map; this is intentionally NOT viridis.
    # Temperature colors: match the intuitive "cool -> hot" scheme used in the UMAP panel
    cmap = plt.get_cmap("turbo")  # cool->hot; distinct from viridis
    tmin, tmax = min(temps), max(temps)
    denom = (tmax - tmin) if (tmax - tmin) != 0 else 1
    temp_to_color = {t: cmap((t - tmin) / denom) for t in temps}
    # Global y-lims across all per-regulon panels
    y_min = np.nanmin(aug["score"].values)
    y_max = np.nanmax(aug["score"].values)
    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
        ylims = (y_min - pad, y_max + pad)
    else:
        ylims = None

    for reg in regulons:
        sub = aug[aug["regulon"] == reg].copy()
        if sub.empty:
            continue

        fig = plt.figure(figsize=(7.8, 4.2))
        ax = plt.gca()

        # Plot each temperature: thin rep traces + thick mean trace
        for t in temps:
            st = sub[sub["temp"] == t]
            if st.empty:
                continue

            # Rep traces
            for r in sorted(st["rep"].unique().tolist()):
                sr = st[st["rep"] == r].sort_values("time")
                ax.plot(
                    sr["time"].values,
                    sr["score"].values,
                    linewidth=1.0,
                    alpha=0.5,
                    color=temp_to_color[t],
                )
                # markers per timepoint
                for tt, vv in zip(sr["time"].values, sr["score"].values):
                    ax.scatter(
                        [tt],
                        [vv],
                        marker=TIME_MARKERS.get(int(tt), "o"),
                        s=28,
                        color=temp_to_color[t],
                        alpha=0.8,
                        linewidths=0.0,
                    )

            # Mean trace
            mean = st.groupby("time", as_index=False)["score"].mean().sort_values("time")
            ax.plot(
                mean["time"].values,
                mean["score"].values,
                linewidth=2.6,
                alpha=0.95,
                color=temp_to_color[t],
                label=f"{t}°C",
            )

        ax.set_title(reg.replace("_score", ""), fontsize=12)
        ax.set_xlabel("Time (min; 0 = 30°C 15-min baseline)")
        ax.set_ylabel("Δ regulon score" if delta else "Regulon score")

        ax.set_xticks(TIMEPOINTS)
        ax.set_xlim(-2, 125)
        if ylims is not None:
            ax.set_ylim(*ylims)

        # tidy legend (temps only)
        ax.legend(
            title="Temperature",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=[0, 0, 0.92, 1])
        stem = reg.replace("_score", "")
        fig.savefig(outdir / "per_regulon" / f"{stem}.png", dpi=200)
        fig.savefig(outdir / "per_regulon" / f"{stem}.pdf")
        plt.close(fig)


def plot_per_temperature(aug: pd.DataFrame, outdir: Path, delta: bool, endpoint_labels: bool = True):
    temps = sorted(aug["temp"].unique().tolist())
    regulons = sorted(aug["regulon"].unique().tolist())

    # Qualitative palette for regulons
    reg_colors = discrete_palette(len(regulons), "tab20")
    reg_to_color = dict(zip(regulons, reg_colors))

    for t in temps:
        st = aug[aug["temp"] == t].copy()
        if st.empty:
            continue

        fig = plt.figure(figsize=(8.2, 4.6))
        ax = plt.gca()

        # For each regulon: thin rep traces + thick mean trace
        for reg in regulons:
            sr = st[st["regulon"] == reg]
            if sr.empty:
                continue

            # Rep traces
            for r in sorted(sr["rep"].unique().tolist()):
                srr = sr[sr["rep"] == r].sort_values("time")
                ax.plot(
                    srr["time"].values,
                    srr["score"].values,
                    linewidth=1.0,
                    alpha=0.35,
                    color=reg_to_color[reg],
                )
                for tt, vv in zip(srr["time"].values, srr["score"].values):
                    ax.scatter(
                        [tt],
                        [vv],
                        marker=TIME_MARKERS.get(int(tt), "o"),
                        s=22,
                        color=reg_to_color[reg],
                        alpha=0.7,
                        linewidths=0.0,
                    )

            # Mean trace
            mean = sr.groupby("time", as_index=False)["score"].mean().sort_values("time")
            ax.plot(
                mean["time"].values,
                mean["score"].values,
                linewidth=2.2,
                alpha=0.95,
                color=reg_to_color[reg],
                label=reg.replace("_score", ""),
            )
        # Boxed legend on the right (avoids overlapping endpoint text)
        ax.legend(
            title="Regulon",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.18, 0.5),
            borderaxespad=0.0,
        )


        ax.set_title(f"{t}°C", fontsize=12)
        ax.set_xlabel("Time (min; 0 = 30°C 15-min baseline)")
        ax.set_ylabel("Δ regulon score" if delta else "Regulon score")

        ax.set_xticks(TIMEPOINTS)
        ax.set_xlim(-2, 125)

        # no big legend
        fig.tight_layout(rect=[0, 0, 0.92, 1])
        fig.savefig(outdir / "per_temperature" / f"{t}C.png", dpi=200)
        fig.savefig(outdir / "per_temperature" / f"{t}C.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="regulon_scores.csv",
        help="Path to regulon_scores.csv (default: regulon_scores.csv in current directory).",
    )
    ap.add_argument(
        "--outdir",
        default="timecourse_plots",
        help="Output directory (default: timecourse_plots).",
    )
    ap.add_argument(
        "--delta",
        action="store_true",
        help="Plot Δscore relative to 30°C 15-min baseline (score - baseline).",
    )
    ap.add_argument(
        "--no-endpoint-labels",
        action="store_true",
        help="Disable endpoint labels in per-temperature plots.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_output_dirs(outdir)

    df = pd.read_csv(csv_path)
    sample_s = find_sample_series(df)
    score_cols = pick_score_columns(df)

    long = compute_dataset_long(df, sample_s, score_cols)
    aug = build_baseline_augmented(long, delta=args.delta)

    # Ensure we only plot timepoints in TIMEPOINTS order
    aug = aug[aug["time"].isin(TIMEPOINTS)].copy()

    plot_per_regulon(aug, outdir, delta=args.delta)
    plot_per_temperature(aug, outdir, delta=args.delta, endpoint_labels=(not args.no_endpoint_labels))

    print(f"Done. Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
