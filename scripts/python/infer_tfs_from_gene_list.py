#!/usr/bin/env python3
"""
TF enrichment from a gene list using local TF target sets.

Conveniences:
- If --genes file isn't found, tries scripts/<genes>.
- If --targets_dir isn't found, tries scripts/<targets_dir>.
- Alternatively, supply --ranked (ranked_genes_detour.csv) and it will
  create pos/neg top-N lists internally (no txt files needed).

Outputs:
- tf_enrichment_<label>.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact


def resolve_path(p: str) -> Path:
    """Resolve path with a fallback to scripts/ if needed."""
    path = Path(p)
    if path.exists():
        return path
    alt = Path("scripts") / p
    if alt.exists():
        return alt
    return path  # will fail later with a helpful error


def read_gene_list(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Gene list not found: {path}")
    genes = []
    for line in path.read_text().splitlines():
        g = line.strip()
        if g and not g.startswith("#"):
            genes.append(g)
    return set(genes)


def load_tf_targets(targets_dir: Path):
    if not targets_dir.exists():
        raise FileNotFoundError(f"targets_dir not found: {targets_dir}")
    tf_files = sorted([p for p in targets_dir.iterdir() if p.is_file()])

    tf_targets = {}
    for p in tf_files:
        # Ignore hidden files
        if p.name.startswith("."):
            continue
        tf = p.stem
        tf_targets[tf] = read_gene_list(p)
    if not tf_targets:
        raise RuntimeError(f"No TF target files found in: {targets_dir}")
    return tf_targets


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR; returns q-values in original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(order, start=1):
        val = pvals[idx] * m / rank
        prev = min(prev, val)
        q[idx] = prev
    return q


def run_enrichment(gene_set, tf_targets, background=None):
    gene_set = set([g for g in gene_set if g])
    if background is None:
        background = set().union(*tf_targets.values()) | gene_set
    background = set([g for g in background if g])

    M = len(background)
    K = len(gene_set & background)

    rows = []
    for tf, targets in tf_targets.items():
        targets_bg = targets & background

        a = len((gene_set & background) & targets_bg)
        if a == 0:
            continue
        b = K - a
        c = len(targets_bg) - a
        d = M - (a + b + c)

        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        overlap_frac = a / max(K, 1)

        rows.append((tf, a, len(targets_bg), K, M, odds, p, overlap_frac))

    if not rows:
        raise RuntimeError("No overlaps found. Likely gene naming mismatch between gene list and TF targets.")

    res = pd.DataFrame(rows, columns=[
        "TF", "overlap_a", "tf_targets_in_bg", "gene_set_in_bg", "background_M",
        "odds_ratio", "p_value", "overlap_frac"
    ])

    q = bh_fdr(res["p_value"].tolist())
    res["fdr_bh"] = q
    res = res.sort_values(["fdr_bh", "overlap_a"], ascending=[True, False]).reset_index(drop=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", default=None, help="Gene list .txt (one per line).")
    ap.add_argument("--ranked", default=None, help="ranked_genes_detour.csv with columns gene,beta_detour,p_detour,...")
    ap.add_argument("--label", default="custom", help="Label for output filename (e.g., pos/neg).")
    ap.add_argument("--top", type=int, default=200, help="If using --ranked: how many genes to use.")
    ap.add_argument("--direction", choices=["pos", "neg"], default="pos",
                    help="If using --ranked: pos uses highest beta_detour; neg uses lowest.")
    ap.add_argument("--background", default=None, help="Optional background gene list .txt.")
    ap.add_argument("--targets_dir", default="tf_targets", help="Directory of TF target files.")
    ap.add_argument("--out", default=None, help="Output CSV; default tf_enrichment_<label>.csv")
    args = ap.parse_args()

    targets_dir = resolve_path(args.targets_dir)
    if not targets_dir.exists():
        raise FileNotFoundError(f"targets_dir not found: {targets_dir} (also tried scripts/{args.targets_dir})")
    tf_targets = load_tf_targets(targets_dir)

    background = None
    if args.background:
        bg_path = resolve_path(args.background)
        print(f"Reading background genes from: {bg_path}")
        background = read_gene_list(bg_path)

    if args.ranked:
        ranked_path = resolve_path(args.ranked)
        print(f"Reading ranked genes from: {ranked_path}")
        df = pd.read_csv(ranked_path)
        if "gene" not in df.columns or "beta_detour" not in df.columns:
            raise ValueError("ranked file must contain columns: gene, beta_detour")

        df = df.dropna(subset=["gene", "beta_detour"])
        df["gene"] = df["gene"].astype(str)

        if args.direction == "pos":
            genes = df.sort_values("beta_detour", ascending=False).head(args.top)["gene"].tolist()
            label = args.label if args.label != "custom" else "pos"
        else:
            genes = df.sort_values("beta_detour", ascending=True).head(args.top)["gene"].tolist()
            label = args.label if args.label != "custom" else "neg"

        gene_set = set(genes)

    elif args.genes:
        genes_path = resolve_path(args.genes)
        print(f"Reading genes from: {genes_path}")
        gene_set = read_gene_list(genes_path)
        label = args.label
    else:
        raise ValueError("Provide either --genes <file.txt> or --ranked <ranked_genes_detour.csv>")

    out = args.out if args.out else f"tf_enrichment_{label}.csv"

    res = run_enrichment(gene_set, tf_targets, background=background)
    res.to_csv(out, index=False)
    print(f"Wrote: {out}  (n={len(res)})")


if __name__ == "__main__":
    main()
