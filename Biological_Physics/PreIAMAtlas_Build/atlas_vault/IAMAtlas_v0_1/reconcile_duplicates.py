#!/usr/bin/env python3
"""
reconcile_duplicates.py
========================
Resolves the duplicate-named cell types surfaced in the IAMAtlas v0.1 REBUILD
distinctness tests. Operates on the iamatlas_v0_1_output_REBUILD folder.

What it does:

(A) Immune class: 12 a-prefix duplicate pairs (aCD8Tmem vs CD8Tmem etc.) are
    confirmed atlas-source naming variants of the same biological cell type
    (corr=1.000, mean|diff|<0.005 on tested CpGs). These are MERGED:
    coverage-weighted pooling, plain-name kept, a-prefix column dropped.

(B) Cycling/Secretory/Terminal classes: thin lowercase duplicates
    (lung/hepatocyte/mammary/brain/heart/skeletal) whose better-covered
    capitalized twins exist are DROPPED. Standalone thin lowercase cells
    with no twin (small_intestine, neuron) are KEPT (flagged low-confidence).

Output: writes <class>_per_celltype.RECONCILED.csv next to each input.
Backup is automatic — original files are not modified.

The reconciliation report is written to RECONCILIATION_REPORT.json.

Run BEFORE merge_iamatlas_v0_1.py.
"""

import argparse, csv, json
from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Reconciliation policy (the decisions, in one place)
# -----------------------------------------------------------------------------

# Immune: 12 confirmed a-prefix duplicates - MERGE (coverage-weighted pool,
# keep plain name, drop 'a' variant).
IMMUNE_MERGES = [
    ("aBaso",     "Baso"),
    ("aBmem",     "Bmem"),
    ("aBnv",      "Bnv"),
    ("aCD4Tmem",  "CD4Tmem"),
    ("aCD4Tnv",   "CD4Tnv"),
    ("aCD8Tmem",  "CD8Tmem"),
    ("aCD8Tnv",   "CD8Tnv"),
    ("aEos",      "Eos"),
    ("aMono",     "Mono"),
    ("aNK",       "NK"),
    ("aNeu",      "Neu"),
    ("aTreg",     "Treg"),
]

# Cycling: thin lowercase duplicates with capitalized well-covered twins - DROP
# Standalone thin lowercase (no twin) - KEEP (low-confidence)
CYCLING_DROP = ["lung"]   # has Lung_cells n=6105; lung is n=252 duplicate
CYCLING_KEEP_THIN = ["small_intestine"]  # no twin, real cell type, just thin

# Secretory: same pattern
SECRETORY_DROP = ["hepatocyte", "mammary"]  # have Hepatocytes/Breast twins

# Terminal: thin lowercase duplicates - DROP
TERMINAL_DROP = ["brain", "heart", "skeletal"]  # have Cortical_neurons/Glia/CM/Left_atrium etc twins
TERMINAL_KEEP_THIN = ["neuron"]  # has NeuMa but biologically distinct, keep

# Progenitor / stromal / stem classes - no reconciliation needed
# (sparse cells are flagged low-confidence in the merge but kept)


def reconcile_immune(in_path, out_path, report):
    """Merge a-prefix duplicates into plain-name columns. Coverage-weighted pool."""
    import pandas as pd
    df = pd.read_csv(in_path)
    print(f"\n[immune] loaded {len(df):,} CpGs, {sum(c.endswith('_mean') for c in df.columns)} cell columns")

    merges_done = []
    for a_name, plain_name in IMMUNE_MERGES:
        a_mean, a_sd = f"{a_name}_mean", f"{a_name}_sd"
        p_mean, p_sd = f"{plain_name}_mean", f"{plain_name}_sd"

        if a_mean not in df.columns:
            print(f"  [skip] {a_name} not in file")
            continue
        if p_mean not in df.columns:
            print(f"  [skip] {plain_name} not in file (would expect a-twin to also be absent)")
            continue

        # Coverage-weighted merge: if both present, weight by inverse variance
        # (precision = 1/sd^2). If only one present, take that one.
        a_m = df[a_mean].to_numpy(dtype=float)
        a_s = df[a_sd].to_numpy(dtype=float)
        p_m = df[p_mean].to_numpy(dtype=float)
        p_s = df[p_sd].to_numpy(dtype=float)

        a_ok = ~np.isnan(a_m)
        p_ok = ~np.isnan(p_m)
        both = a_ok & p_ok
        only_a = a_ok & ~p_ok
        only_p = p_ok & ~a_ok

        # Inverse-variance pooled mean and sd
        # Guard against sd==0 (shouldn't happen but be safe)
        a_var = np.where(a_s > 0, a_s**2, 1e-6)
        p_var = np.where(p_s > 0, p_s**2, 1e-6)
        wsum_both = 1.0/a_var + 1.0/p_var
        pooled_mean = np.where(both, (a_m/a_var + p_m/p_var) / wsum_both, np.nan)
        pooled_sd   = np.where(both, np.sqrt(1.0/wsum_both), np.nan)

        new_m = np.full(len(df), np.nan)
        new_s = np.full(len(df), np.nan)
        new_m[only_p] = p_m[only_p]; new_s[only_p] = p_s[only_p]
        new_m[only_a] = a_m[only_a]; new_s[only_a] = a_s[only_a]
        new_m[both]   = pooled_mean[both]
        new_s[both]   = pooled_sd[both]

        df[p_mean] = new_m
        df[p_sd]   = new_s
        df = df.drop(columns=[a_mean, a_sd])

        n_a_only, n_p_only, n_both = int(only_a.sum()), int(only_p.sum()), int(both.sum())
        merges_done.append({
            "merged_into": plain_name, "from": a_name,
            "n_only_a": n_a_only, "n_only_plain": n_p_only, "n_both_pooled": n_both,
            "final_n": int((~np.isnan(new_m)).sum()),
        })
        print(f"  [merge] {a_name} -> {plain_name}: only_a={n_a_only}, only_plain={n_p_only}, both_pooled={n_both}, final_n={n_a_only+n_p_only+n_both}")

    df.to_csv(out_path, index=False)
    report["immune"] = {
        "action": "merged a-prefix duplicates into plain-name columns (coverage-weighted)",
        "merges_performed": merges_done,
        "rows_out": len(df),
        "cells_out": sum(c.endswith('_mean') for c in df.columns),
    }
    print(f"  wrote {out_path} ({sum(c.endswith('_mean') for c in df.columns)} cells)")


def drop_columns(in_path, out_path, drop_cells, keep_thin_note, class_name, report):
    """Drop named cell columns (and their _sd partners)."""
    import pandas as pd
    df = pd.read_csv(in_path)
    n_cells_before = sum(c.endswith('_mean') for c in df.columns)
    print(f"\n[{class_name}] loaded {len(df):,} CpGs, {n_cells_before} cell columns")

    dropped = []
    for cell in drop_cells:
        cm, cs = f"{cell}_mean", f"{cell}_sd"
        if cm in df.columns:
            df = df.drop(columns=[cm, cs])
            dropped.append(cell)
            print(f"  [drop] {cell}")
        else:
            print(f"  [skip] {cell} not in file")

    df.to_csv(out_path, index=False)
    report[class_name] = {
        "action": "dropped thin lowercase duplicates (better-covered capitalized twins exist)",
        "dropped": dropped,
        "kept_thin_low_confidence": keep_thin_note,
        "rows_out": len(df),
        "cells_out": sum(c.endswith('_mean') for c in df.columns),
    }
    print(f"  wrote {out_path} ({sum(c.endswith('_mean') for c in df.columns)} cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="iamatlas_v0_1_output_REBUILD",
                    help="Folder with per-class outputs from the rebuild")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    report = {
        "input_dir": str(in_dir),
        "policy_summary": {
            "immune": "MERGE 12 a-prefix pairs (Salas activated-naming variants) into plain-name columns via coverage-weighted inverse-variance pooling",
            "cycling": f"DROP {CYCLING_DROP} (thin duplicates), KEEP {CYCLING_KEEP_THIN} (thin no-twin, low-confidence)",
            "secretory": f"DROP {SECRETORY_DROP} (thin duplicates)",
            "terminal": f"DROP {TERMINAL_DROP} (thin duplicates), KEEP {TERMINAL_KEEP_THIN} (thin no-twin, low-confidence)",
            "progenitor/stromal/stem_pluri/stem_adult": "no reconciliation (sparse cells kept as low-confidence per two-tier design)",
        },
    }

    # Immune: merge
    f_in  = in_dir / "iamatlas_v0_1_immune_per_celltype.csv"
    f_out = in_dir / "iamatlas_v0_1_immune_per_celltype.RECONCILED.csv"
    reconcile_immune(f_in, f_out, report)

    # Cycling / Secretory / Terminal: drop thin duplicates
    drop_columns(
        in_dir / "iamatlas_v0_1_cycling_per_celltype.csv",
        in_dir / "iamatlas_v0_1_cycling_per_celltype.RECONCILED.csv",
        CYCLING_DROP, CYCLING_KEEP_THIN, "cycling", report)

    drop_columns(
        in_dir / "iamatlas_v0_1_secretory_per_celltype.csv",
        in_dir / "iamatlas_v0_1_secretory_per_celltype.RECONCILED.csv",
        SECRETORY_DROP, [], "secretory", report)

    drop_columns(
        in_dir / "iamatlas_v0_1_terminal_per_celltype.csv",
        in_dir / "iamatlas_v0_1_terminal_per_celltype.RECONCILED.csv",
        TERMINAL_DROP, TERMINAL_KEEP_THIN, "terminal", report)

    # For progenitor/stromal/stems: just copy (no reconciliation) so the merge
    # step has a consistent .RECONCILED.csv filename pattern to glob on.
    import shutil
    for cls in ["progenitor", "stromal", "stem_pluri", "stem_adult"]:
        src = in_dir / f"iamatlas_v0_1_{cls}_per_celltype.csv"
        dst = in_dir / f"iamatlas_v0_1_{cls}_per_celltype.RECONCILED.csv"
        if src.exists():
            shutil.copy(src, dst)
            print(f"\n[{cls}] copied (no reconciliation needed)")
            report[cls] = {"action": "no reconciliation; copied as-is for merge step uniformity"}

    # Write report
    rep_path = in_dir / "RECONCILIATION_REPORT.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReconciliation report: {rep_path}")
    print("\nNow run: python merge_iamatlas_v0_1_REBUILD.py")


if __name__ == "__main__":
    main()
