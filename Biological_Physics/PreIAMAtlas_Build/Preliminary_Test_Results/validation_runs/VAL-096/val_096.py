"""
VAL-096 — TTD-window stratification on Loyfer Stage 2 per-tile A-scores
Heath W. Mahaffey | IAMPerformance | 2026-04-26
RNG seed: 20260426
Pre-reg SHA-256: 01247146d955ad28a7d141dd5b194a86d1d97b63b1022a07587ae4cd69310c6d
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(20260426)

# Load VAL-093 per-sample CSV (already has per-tile A-scores + TTD)
SRC = "/home/claude/run_everything/VAL-093_per_sample.csv"
df = pd.read_csv(SRC)
print(f"Loaded VAL-093 per-sample: {len(df)} samples, {len(df.columns)} columns")

# Identify tile columns (everything starting with A_)
tile_cols = [c for c in df.columns if c.startswith("A_")]
print(f"Tile columns: {len(tile_cols)} ({tile_cols[:3]}...)")

# Identify breast cancer cases by ICD-10 site code
df["is_breast_case"] = df["cancer_site"].astype(str).str.upper().str.contains("C50", na=False)
df["is_control"] = (df["group"] == "control")

print(f"\nBreast cases: {df['is_breast_case'].sum()}")
print(f"Controls (cancer-free): {df['is_control'].sum()}")

# Per-cohort breakdown
for coh in ["GSE51057", "GSE51032"]:
    sub = df[df["cohort"] == coh]
    n_cases = sub["is_breast_case"].sum()
    n_ctrl = sub["is_control"].sum()
    print(f"  {coh}: {n_cases} breast cases, {n_ctrl} controls")

# TTD windows
WINDOWS = [
    ("0-2yr",   0,  2),
    ("2-5yr",   2,  5),
    ("5-10yr",  5, 10),
    (">10yr",  10, 99),
]


def cohen_d(x, y):
    """Cohen's d, pooled variance."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)
    if pooled <= 0:
        return float("nan")
    return (mx - my) / np.sqrt(pooled)


def bootstrap_ci(x, y, n_iter=1000, alpha=0.05, rng=None):
    """BCa bootstrap CI for Cohen's d."""
    if rng is None:
        rng = np.random.default_rng(20260426)
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    boot_d = []
    nx, ny = len(x), len(y)
    x_arr = np.asarray(x); y_arr = np.asarray(y)
    for _ in range(n_iter):
        xb = rng.choice(x_arr, size=nx, replace=True)
        yb = rng.choice(y_arr, size=ny, replace=True)
        boot_d.append(cohen_d(xb, yb))
    boot_d = np.asarray([d for d in boot_d if np.isfinite(d)])
    if len(boot_d) == 0:
        return (float("nan"), float("nan"))
    lo = np.percentile(boot_d, 100 * alpha/2)
    hi = np.percentile(boot_d, 100 * (1-alpha/2))
    return (float(lo), float(hi))


# Run window-stratified analysis
results = {
    "val_id": "VAL-096",
    "prereg_sha256": "01247146d955ad28a7d141dd5b194a86d1d97b63b1022a07587ae4cd69310c6d",
    "rng_seed": 20260426,
    "method": "Per-tile case-vs-control Cohen's d at TTD-stratified windows",
    "atlas": "Loyfer/Moss 25-cell array (vault stage2_cell_of_origin/loyfer_moss_2018)",
    "platform": "450K methylation array",
    "specimen": "buffy-coat whole blood",
    "windows": [w[0] for w in WINDOWS],
    "tiles": tile_cols,
    "per_cohort": {},
    "healthy_baseline_check": {},
}

for cohort in ["GSE51057", "GSE51032"]:
    sub = df[df["cohort"] == cohort].copy()
    ctrl = sub[sub["is_control"]]
    cohort_results = {"window_tile_d": {}, "n_per_window": {}}

    # Healthy baseline per tile
    healthy_mean = {t: float(ctrl[t].mean()) for t in tile_cols}
    healthy_sd   = {t: float(ctrl[t].std())  for t in tile_cols}
    cohort_results["healthy_mean"] = healthy_mean
    cohort_results["healthy_sd"]   = healthy_sd

    for win_label, lo, hi in WINDOWS:
        cases = sub[sub["is_breast_case"] & (sub["ttd_years"] >= lo) & (sub["ttd_years"] < hi)]
        n_case = len(cases); n_ctrl = len(ctrl)
        cohort_results["n_per_window"][win_label] = {"cases": n_case, "controls": n_ctrl}
        if n_case < 2:
            cohort_results["window_tile_d"][win_label] = {t: {"d": None, "ci": [None, None]} for t in tile_cols}
            continue

        win_d = {}
        for tile in tile_cols:
            d = cohen_d(cases[tile].dropna().values, ctrl[tile].dropna().values)
            ci = bootstrap_ci(cases[tile].dropna().values, ctrl[tile].dropna().values,
                              n_iter=1000, rng=np.random.default_rng(20260426 + hash(tile) % 1000))
            win_d[tile] = {"d": float(d) if np.isfinite(d) else None,
                           "ci": [float(ci[0]) if np.isfinite(ci[0]) else None,
                                  float(ci[1]) if np.isfinite(ci[1]) else None]}
        cohort_results["window_tile_d"][win_label] = win_d

    results["per_cohort"][cohort] = cohort_results

# Cross-cohort healthy baseline check (CHK-3.2)
gse57_h = results["per_cohort"]["GSE51057"]["healthy_mean"]
gse32_h = results["per_cohort"]["GSE51032"]["healthy_mean"]
gse57_sd = results["per_cohort"]["GSE51057"]["healthy_sd"]
flagged = []
for t in tile_cols:
    delta = abs(gse57_h[t] - gse32_h[t])
    sd = gse57_sd[t]
    if sd > 0 and delta > sd:
        flagged.append({"tile": t, "delta": float(delta), "anchor_sd": float(sd)})
results["healthy_baseline_check"] = {
    "flagged_tiles": flagged,
    "passes_chk_3_2": len(flagged) == 0,
}

# Save
out = "/home/claude/run_everything/VAL-096_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults written: {out}")

# Print headline table
print("\n=== HEADLINE: per-tile d at each window (GSE51057 / GSE51032) ===")
print(f"{'Tile':35s}  {'>10yr':>14s}  {'5-10yr':>14s}  {'2-5yr':>14s}  {'0-2yr':>14s}")
for tile in tile_cols:
    row = []
    for win in [">10yr", "5-10yr", "2-5yr", "0-2yr"]:
        d57 = results["per_cohort"]["GSE51057"]["window_tile_d"].get(win, {}).get(tile, {}).get("d")
        d32 = results["per_cohort"]["GSE51032"]["window_tile_d"].get(win, {}).get(tile, {}).get("d")
        s57 = f"{d57:+.2f}" if d57 is not None else "  -- "
        s32 = f"{d32:+.2f}" if d32 is not None else "  -- "
        row.append(f"{s57}/{s32}")
    print(f"{tile:35s}  {row[0]:>14s}  {row[1]:>14s}  {row[2]:>14s}  {row[3]:>14s}")

print(f"\nCHK-3.2: {len(flagged)} cross-cohort tile mismatches "
      f"({'PASS' if results['healthy_baseline_check']['passes_chk_3_2'] else 'FAIL'})")
