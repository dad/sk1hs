#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0,1])

def partial_corr(x, y, covars_df):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    C = covars_df.to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    if m.sum() < (C.shape[1] + 3):
        return np.nan

    X = C[m]
    X = np.column_stack([np.ones(X.shape[0]), X])  # intercept

    bx, *_ = np.linalg.lstsq(X, x[m], rcond=None)
    rx = x[m] - X @ bx

    by, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    ry = y[m] - X @ by

    return float(np.corrcoef(rx, ry)[0,1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="merged_regulon_detour_umap.csv")
    ap.add_argument("--detour_col", required=True)
    ap.add_argument("--temp_col", default="temp_C")
    ap.add_argument("--time_col", default="time_min")
    ap.add_argument("--hi_temps", default="42,44")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.detour_col not in df.columns:
        raise SystemExit(f"ERROR: detour column '{args.detour_col}' not found in {args.csv}")

    score_cols = [c for c in df.columns if c.endswith("_score")]
    if not score_cols:
        raise SystemExit("ERROR: no *_score columns found")

    # Overall
    rows = []
    for c in score_cols:
        rows.append({"regulon": c.replace("_score",""), "r_all": corr(df[c], df[args.detour_col])})
    out_all = pd.DataFrame(rows).sort_values("r_all", ascending=False)
    out_all.to_csv("tf_vs_detour_all.csv", index=False)
    print("Wrote tf_vs_detour_all.csv")

    # High temps
    hi = [float(x) for x in args.hi_temps.split(",") if x.strip()]
    if args.temp_col in df.columns:
        sub = df[df[args.temp_col].isin(hi)].copy()
        rows = []
        for c in score_cols:
            row = {"regulon": c.replace("_score",""), "r_hi": corr(sub[c], sub[args.detour_col])}
            if args.time_col in sub.columns:
                row["pcorr_hi_time"] = partial_corr(sub[c], sub[args.detour_col], sub[[args.time_col]])
            rows.append(row)
        out_hi = pd.DataFrame(rows).sort_values("r_hi", ascending=False)
        out_hi.to_csv("tf_vs_detour_hiTemps.csv", index=False)
        print("Wrote tf_vs_detour_hiTemps.csv")
    else:
        print(f"NOTE: '{args.temp_col}' not present; skipping high-temp subset")

if __name__ == "__main__":
    main()
