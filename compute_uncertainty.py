#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-feature uncertainty across multiple CSV runs.

Non-subjective (per feature):
  - Aleatoric(f)  = mean_over_runs( Var_run(values_f) )
  - Epistemic(f)  = Var_over_runs( Mean_run(values_f) )
  - Total(f)      = Aleatoric(f) + Epistemic(f)

Subjective Logic (per feature, over runs; now feature-dependent):
  - Build shared histogram bins per feature across runs
  - p_r = per-run histogram probabilities, p̄ = mean_r p_r
  - Ĥ = H(p̄)/log K   (normalized entropy)
  - JS = mean_r JS(p_r || p̄)  (Jensen–Shannon disagreement; optional)
  - N_eff = avg non-NaN sample count per feature (or fixed)
  - c_f = N_eff * (1 - Ĥ) * exp(-γ * JS)
  - α = 1 + c_f * p̄,  u = K / sum(α)
  - Also reports entropy of SL expected probabilities for diagnostics

Outputs:
  - feature_uncertainty_summary.csv : per-feature with aleatoric, epistemic, total, subjective_u, etc.
  - subjective_logic_profiles/: per-feature JSON with full SL diagnostics
  - (optional) per-class CSV if --per_class_label is given
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import math
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from scipy.spatial.distance import jensenshannon

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy (nats) of a probability vector."""
    if p is None:
        return np.nan
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

def load_runs(input_dir: Path, pattern: str) -> tuple[list[pd.DataFrame], list[str]]:
    files = sorted([str(p) for p in input_dir.glob(pattern)])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} matching pattern '{pattern}'")
    dfs = [pd.read_csv(f) for f in files]
    return dfs, files

def intersect_numeric_features(dfs: list[pd.DataFrame], exclude_cols: set[str]) -> list[str]:
    num_sets = []
    for df in dfs:
        numeric = df.select_dtypes(include=[np.number]).columns
        numeric = [c for c in numeric if c not in exclude_cols]
        num_sets.append(set(numeric))
    return sorted(set.intersection(*num_sets))

def per_feature_uncertainties(dfs: list[pd.DataFrame], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        means, vars_ = [], []
        for df in dfs:
            col = df[feat].to_numpy()
            if col.size < 2 or np.all(np.isnan(col)):
                means.append(np.nan)
                vars_.append(np.nan)
            else:
                means.append(np.nanmean(col))
                vars_.append(np.nanvar(col, ddof=1))
        aleatoric = np.nanmean(vars_) if len(vars_) else np.nan
        epistemic = np.nanvar(means, ddof=1) if len(means) >= 2 else np.nan
        total = (aleatoric if not math.isnan(aleatoric) else 0.0) + (epistemic if not math.isnan(epistemic) else 0.0)
        rows.append({
            "feature": feat,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total
        })
    return pd.DataFrame(rows)

# ---- Subjective Logic with strength (NEW) ----

def _hist_probs(values, K, edges=None):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None, None
    if edges is None:
        counts, edges = np.histogram(v, bins=K)
    else:
        counts, _ = np.histogram(v, bins=edges)
    p = counts.astype(float)
    s = p.sum()
    p = p / s if s > 0 else np.ones(K)/K
    return p, edges

def compute_subjective_logic_with_strength(
    dfs: list[pd.DataFrame],
    feature_cols: list[str],
    bins: int,
    outdir: Path,
    use_disagreement: bool = True,
    gamma: float = 4.0,
    neff_mode: str = "samples",   # "samples" or "fixed"
    neff_cap: float | None = None
) -> pd.DataFrame:
    """
    c_f = N_eff * (1 - Ĥ) * exp(-gamma * JS), α = 1 + c_f * p̄, u = K / Σα.
    JS is mean Jensen–Shannon divergence of per-run histograms to the mean.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    multi_run = len(dfs) >= 2
    rows = []
    for feat in feature_cols:
        pooled = pd.concat([df[feat] for df in dfs], axis=0, ignore_index=True).to_numpy()
        p_pool, edges = _hist_probs(pooled, bins, edges=None)
        if edges is None:
            rows.append({"feature": feat, "subjective_u": np.nan, "subjective_expected_entropy": np.nan})
            continue

        probs, nonnan_counts = [], []
        for df in dfs:
            v = df[feat].to_numpy()
            p_r, _ = _hist_probs(v, bins, edges=edges)
            if p_r is None:
                continue
            probs.append(p_r)
            nonnan_counts.append(np.isfinite(v).sum())
        if not probs:
            rows.append({"feature": feat, "subjective_u": np.nan, "subjective_expected_entropy": np.nan})
            continue

        probs = np.stack(probs, axis=0)
        p_bar = probs.mean(axis=0)

        Hhat = entropy(p_bar) / np.log(len(p_bar))

        JS = 0.0
        if use_disagreement and multi_run:
            JS = float(np.mean([jensenshannon(p, p_bar)**2 for p in probs]))

        if neff_mode == "samples":
            N_eff = float(np.mean(nonnan_counts))
        else:
            N_eff = neff_cap if neff_cap is not None else 100.0
        if neff_cap is not None:
            N_eff = min(N_eff, float(neff_cap))

        c_f = N_eff * (1.0 - Hhat) * np.exp(-gamma * JS)

        alpha = 1.0 + c_f * p_bar
        S = alpha.sum()
        K = len(p_bar)
        u = K / S

        a = np.ones(K) / K
        b = (alpha - 1.0) / S
        p_exp = b + a * u
        ent_exp = entropy(p_exp)

        rows.append({
            "feature": feat,
            "subjective_u": float(u),
            "subjective_expected_entropy": float(ent_exp),
            "SL_strength_c": float(c_f),
            "Hhat": float(Hhat),
            "JS": float(JS),
            "N_eff": float(N_eff)
        })

        with open(outdir / f"{feat}_subjective_logic.json", "w") as f:
            json.dump({
                "feature": feat,
                "bins": bins,
                "bin_edges": [float(x) for x in edges],
                "p_bar": p_bar.tolist(),
                "N_eff": N_eff,
                "Hhat": Hhat,
                "JS": JS,
                "gamma": gamma,
                "c_f": c_f,
                "alpha": alpha.tolist(),
                "u": u,
                "expected_entropy": ent_exp
            }, f, indent=2)

    return pd.DataFrame(rows)

# ---- Optional: per-class uncertainties ----

def per_class_uncertainties(
    dfs: list[pd.DataFrame],
    feature_cols: list[str],
    label_col: str,
    bins: int,
    outdir: Path
) -> pd.DataFrame:
    classes = sorted(
        set(pd.concat([df[label_col] for df in dfs], ignore_index=True).dropna().unique().tolist())
    )
    rows = []
    outdir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        dfs_c = [df[df[label_col] == cls] for df in dfs]
        base_df = per_feature_uncertainties(dfs_c, feature_cols)
        sl_df = compute_subjective_logic_with_strength(
            dfs_c, feature_cols, bins=bins, outdir=outdir / f"class_{cls}",
            use_disagreement=True, gamma=4.0, neff_mode="samples", neff_cap=None
        )
        merged = base_df.merge(sl_df, on="feature", how="left")
        merged["class"] = cls
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["class", "feature"])

def main():
    ap = argparse.ArgumentParser(description="Compute per-feature uncertainties from multiple CSV runs.")
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing CSV files.")
    ap.add_argument("--pattern", type=str, default="train*.csv", help="Glob pattern for CSV files (default: *.csv).")
    ap.add_argument("--exclude_cols", type=str, default="label,image_path",
                    help="Comma-separated columns to exclude from feature set (default: label,image_path).")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins for Subjective Logic (default: 10).")
    ap.add_argument("--per_class_label", type=str, default=None,
                    help="If provided, compute per-class uncertainties using this label column.")
    ap.add_argument("--output", type=str, default="feature_uncertainty_summary.csv",
                    help="Output CSV for per-feature uncertainties.")
    ap.add_argument("--output_dir", type=str, default="subjective_logic_profiles",
                    help="Directory to write per-feature subjective logic JSON profiles.")
    ap.add_argument("--per_class_output", type=str, default="feature_uncertainty_per_class.csv",
                    help="Output CSV for per-class uncertainties (used only if --per_class_label is set).")
    args = ap.parse_args()
    
    t = time.time()

    input_dir = Path(args.input_dir)
    exclude_cols = set([c.strip() for c in args.exclude_cols.split(",") if c.strip()])

    dfs, files = load_runs(input_dir, args.pattern)
    print(f"Loaded {len(dfs)} CSV runs:")
    for f in files:
        print("  -", f)

    feature_cols = intersect_numeric_features(dfs, exclude_cols)
    if not feature_cols:
        raise RuntimeError("No overlapping numeric features found across runs after exclusions.")
    print(f"Using {len(feature_cols)} numeric features.")

    base_df = per_feature_uncertainties(dfs, feature_cols)

    sl_dir = Path(args.output_dir)
    sl_df = compute_subjective_logic_with_strength(
        dfs, feature_cols, bins=args.bins, outdir=sl_dir,
        use_disagreement=True,  # set False if single-run or if you want to ignore disagreement
        gamma=4.0,
        neff_mode="samples",
        neff_cap=None
    )

    summary = base_df.merge(sl_df, on="feature", how="left")
    summary = summary.sort_values("total", ascending=False)
    summary.to_csv(args.output, index=False)
    print(f"Saved per-feature summary to: {args.output}")

    if args.per_class_label is not None:
        label_col = args.per_class_label
        for i, df in enumerate(dfs):
            if label_col not in df.columns:
                raise KeyError(f"Label column '{label_col}' not found in run #{i} ({files[i]}).")
        per_class_df = per_class_uncertainties(
            dfs, feature_cols, label_col=label_col, bins=args.bins, outdir=sl_dir / "per_class_profiles"
        )
        per_class_df = per_class_df.sort_values(["class", "total"], ascending=[True, False])
        per_class_df.to_csv(args.per_class_output, index=False)
        print(f"Saved per-class summary to: {args.per_class_output}")
    print(time.time() - t)

if __name__ == "__main__":
    main()
