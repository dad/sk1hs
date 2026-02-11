#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

IN = Path("regulon_scores.csv")

REGS = ["Sum1", "Hsf1", "Msn2", "Hac1", "RPG", "RiBi"]
REG_COLS = [f"{r}_score" for r in REGS]

OUT_PCA_TABLE = Path("regulon_pca_coords.csv")
OUT_PCA_LOADINGS = Path("regulon_pca_loadings.csv")
OUT_DETOUR_TABLE = Path("umap_detour_metrics.csv")

def time_to_marker_map(times):
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    times = sorted(times)
    return {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(times)}

def plot_scatter_pc(df, out_png, title):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    times = sorted(df["time_min"].dropna().unique())
    t2m = time_to_marker_map(times)

    temps = sorted(df["temp_C"].dropna().unique())

    cmap = plt.get_cmap("tab10", max(len(temps), 3))
    temp2c = {t: cmap(i) for i, t in enumerate(temps)}

    for t in times:
        sub = df[df["time_min"] == t]
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=80, alpha=0.9,
            marker=t2m[t],
            c=[temp2c[x] for x in sub["temp_C"]],
            edgecolors="none",
            label=f"{int(t)} min" if float(t).is_integer() else f"{t} min",
        )

    ax.set_xlabel("PC1 (regulon space)")
    ax.set_ylabel("PC2 (regulon space)")
    ax.set_title(title)

    leg1 = ax.legend(title="Timepoint", frameon=False, fontsize=9, loc="upper right")
    ax.add_artist(leg1)

    handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                         markerfacecolor=temp2c[t], markeredgecolor='none', markersize=9)
               for t in temps]
    labels = [f"{t:g}°C" for t in temps]
    ax.legend(handles, labels, title="Temperature", frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_umap_colored(df, color_col, out_png, title):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    times = sorted(df["time_min"].dropna().unique())
    t2m = time_to_marker_map(times)

    vals = df[color_col].to_numpy()
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.viridis

    for t in times:
        sub = df[df["time_min"] == t]
        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            s=70, alpha=0.92,
            marker=t2m[t],
            c=cmap(norm(sub[color_col])),
            edgecolors="none",
            label=f"{int(t)} min" if float(t).is_integer() else f"{t} min",
        )

    for (_, _), d in df.groupby(["temp_C", "rep"]):
        d = d.dropna(subset=["time_min"]).sort_values("time_min")
        if d.shape[0] < 2:
            continue
        xs = d["UMAP1"].to_numpy()
        ys = d["UMAP2"].to_numpy()
        for i in range(len(xs) - 1):
            ax.annotate(
                "", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=0.55, color="0.25")
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(color_col)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="Timepoint", frameon=False, fontsize=9)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def orth_dist_to_line(P, A, B):
    v = B - A
    denom = np.linalg.norm(v)
    if denom == 0:
        return np.full((P.shape[0],), np.nan)
    PA = P - A
    cross = v[0]*PA[:,1] - v[1]*PA[:,0]
    return np.abs(cross) / denom

def projection_param(P, A, B):
    v = B - A
    vv = np.dot(v, v)
    if vv == 0:
        return np.full((P.shape[0],), np.nan)
    PA = P - A
    return (PA @ v) / vv

def pick_baseline_terminal(df):
    """
    Robust selection:
      baseline: lowest temp, earliest time available at that temp
      terminal: highest temp, latest time available at that temp
    """
    temps = sorted(df["temp_C"].dropna().unique())
    if not temps:
        raise SystemExit("ERROR: temp_C is all NaN")

    t_low = temps[0]
    t_high = temps[-1]

    df_low = df[df["temp_C"] == t_low].dropna(subset=["time_min"])
    df_high = df[df["temp_C"] == t_high].dropna(subset=["time_min"])

    if df_low.shape[0] == 0:
        raise SystemExit(f"ERROR: no rows for lowest temperature {t_low}")
    if df_high.shape[0] == 0:
        raise SystemExit(f"ERROR: no rows for highest temperature {t_high}")

    time_low = float(df_low["time_min"].min())
    time_high = float(df_high["time_min"].max())

    base = df[(df["temp_C"] == t_low) & (df["time_min"] == time_low)]
    term = df[(df["temp_C"] == t_high) & (df["time_min"] == time_high)]

    if base.shape[0] == 0:
        raise SystemExit(f"ERROR: couldn't find baseline rows at {t_low}°C, time={time_low}")
    if term.shape[0] == 0:
        raise SystemExit(f"ERROR: couldn't find terminal rows at {t_high}°C, time={time_high}")

    return (t_low, time_low, base), (t_high, time_high, term)

def main():
    if not IN.exists():
        raise SystemExit(f"ERROR: missing {IN}")

    df = pd.read_csv(IN, index_col=0)

    need = ["UMAP1", "UMAP2", "temp_C", "time_min", "rep"] + REG_COLS
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"ERROR: {IN} missing required column: {c}")

    # -------- A) PCA in regulon space --------
    X = df[REG_COLS].to_numpy(dtype=float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    pca = PCA(n_components=6, random_state=0)
    pcs = pca.fit_transform(X)

    pca_df = df[["UMAP1","UMAP2","temp_C","time_min","rep"]].copy()
    for i in range(6):
        pca_df[f"PC{i+1}"] = pcs[:, i]

    pca_df.to_csv(OUT_PCA_TABLE)
    print(f"Wrote {OUT_PCA_TABLE}")

    load = pd.DataFrame(
        pca.components_.T,
        index=REG_COLS,
        columns=[f"PC{i+1}" for i in range(6)]
    )
    load.to_csv(OUT_PCA_LOADINGS)
    print(f"Wrote {OUT_PCA_LOADINGS}")
    print("Explained variance ratios:", np.round(pca.explained_variance_ratio_, 3))

    plot_scatter_pc(pca_df, "regulon_PCA_PC1_PC2.png", "Regulon-space PCA (PC1 vs PC2)")
    plot_umap_colored(pca_df, "PC1", "umap_colored_PC1.png", "UMAP colored by regulon PC1")
    plot_umap_colored(pca_df, "PC2", "umap_colored_PC2.png", "UMAP colored by regulon PC2")
    print("Wrote PCA plots: regulon_PCA_PC1_PC2.png, umap_colored_PC1.png, umap_colored_PC2.png")

    # -------- B) Detour metric in UMAP space --------
    (t_low, time_low, base), (t_high, time_high, term) = pick_baseline_terminal(df)

    A = base[["UMAP1","UMAP2"]].to_numpy(dtype=float).mean(axis=0)
    B = term[["UMAP1","UMAP2"]].to_numpy(dtype=float).mean(axis=0)

    print(f"Baseline A: {t_low:g}°C @ {time_low:g} min (n={base.shape[0]}) -> {A}")
    print(f"Terminal B: {t_high:g}°C @ {time_high:g} min (n={term.shape[0]}) -> {B}")

    P = df[["UMAP1","UMAP2"]].to_numpy(dtype=float)
    detour = orth_dist_to_line(P, A, B)
    tproj = projection_param(P, A, B)

    det = df[["UMAP1","UMAP2","temp_C","time_min","rep","Sum1_score"]].copy()
    det["detour"] = detour
    det["stress_axis_t"] = tproj
    det.to_csv(OUT_DETOUR_TABLE)
    print(f"Wrote {OUT_DETOUR_TABLE}")

    # detour vs time
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    agg = det.groupby(["temp_C", "time_min"], as_index=False)["detour"].mean()
    for temp, d in agg.groupby("temp_C"):
        d = d.sort_values("time_min")
        ax.plot(d["time_min"], d["detour"], marker="o", label=f"{temp:g}°C")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Detour (orth. distance to low→high axis in UMAP)")
    ax.set_title("UMAP detour vs time")
    ax.legend(title="Temperature", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig("detour_vs_time.png", dpi=300)
    plt.close(fig)
    print("Wrote detour_vs_time.png")

    # detour vs Sum1
    valid = det[["detour","Sum1_score"]].dropna()
    r = np.corrcoef(valid["detour"], valid["Sum1_score"])[0,1] if valid.shape[0] > 2 else np.nan

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(valid["Sum1_score"], valid["detour"], s=60, alpha=0.85)
    ax.set_xlabel("Sum1 score (mean z)")
    ax.set_ylabel("Detour (UMAP orth distance)")
    ax.set_title(f"Detour vs Sum1 (r={r:.2f})" if np.isfinite(r) else "Detour vs Sum1")
    fig.tight_layout()
    fig.savefig("detour_vs_sum1.png", dpi=300)
    plt.close(fig)
    print("Wrote detour_vs_sum1.png")
    print("Done.")

if __name__ == "__main__":
    main()
