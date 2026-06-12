"""
VAL-095 — UniLIFE 19-cell Stage 3 head-to-head vs Salas 450K 6-cell, dual cohort, all 4 TTD windows
Heath W. Mahaffey | IAMPerformance | 2026-04-26
RNG seed: 20260426
Pre-reg SHA-256: 5f74259d5341268ee7cdaf68322962a275dd19e4158b398f09562f6aaa44bace

Method:
- RPC (Robust Partial Correlation) deconvolution per Guo 2025 / EpiDISH protocol:
  for each sample, solve f = argmin || Y - X @ f ||² s.t. f >= 0, sum(f) ~ 1
  where Y = sample's β values at panel CpGs, X = reference matrix (cell types × CpGs).
  Final fractions normalized to sum to 1.
- Run TWO panels: UniLIFE 19-cell + Salas 450K 6-cell
- Cohen's d on each cell-type fraction at each TTD window
"""

import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import nnls

RNG_SEED = 20260426

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return float("nan")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)
    if pooled <= 0: return float("nan")
    return (mx - my) / np.sqrt(pooled)


def bootstrap_ci(x, y, n_iter=1000, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(RNG_SEED)
    if len(x) < 2 or len(y) < 2: return (float("nan"), float("nan"))
    boot_d = []
    nx, ny = len(x), len(y)
    x_arr, y_arr = np.asarray(x), np.asarray(y)
    for _ in range(n_iter):
        xb = rng.choice(x_arr, size=nx, replace=True)
        yb = rng.choice(y_arr, size=ny, replace=True)
        boot_d.append(cohen_d(xb, yb))
    boot_d = np.asarray([d for d in boot_d if np.isfinite(d)])
    if len(boot_d) == 0: return (float("nan"), float("nan"))
    return (float(np.percentile(boot_d, 100*alpha/2)),
            float(np.percentile(boot_d, 100*(1-alpha/2))))


def deconvolve_rpc(sample_betas, ref_matrix):
    """RPC-style NNLS deconvolution. Returns normalized fractions per cell type."""
    common = ref_matrix.index.intersection(sample_betas.index)
    if len(common) < 50:  # too few markers
        return np.full(ref_matrix.shape[1], np.nan)
    X = ref_matrix.loc[common].values.astype(float)  # (n_cpg, n_celltypes)
    y = sample_betas.loc[common].astype(float).values  # (n_cpg,)
    # Drop CpGs where either X or y has NaN
    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < 50:
        return np.full(ref_matrix.shape[1], np.nan)
    X = X[valid_mask]
    y = y[valid_mask]
    # NNLS: min ||X f - y||² s.t. f >= 0
    try:
        f, _ = nnls(X, y, maxiter=500)
    except Exception:
        return np.full(ref_matrix.shape[1], np.nan)
    if f.sum() <= 0:
        return np.full(ref_matrix.shape[1], np.nan)
    f = f / f.sum()
    return f


# ── Load atlases ──────────────────────────────────────────────────────────
unilife = pd.read_csv("/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv", index_col=0)
salas = pd.read_csv("/home/claude/iam_repo/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv", index_col=0)
print(f"UniLIFE: {unilife.shape[0]} CpGs × {unilife.shape[1]} cell types: {list(unilife.columns)}")
print(f"Salas 450K: {salas.shape[0]} CpGs × {salas.shape[1]} cell types: {list(salas.columns)}")

# ── Load extracted betas ──────────────────────────────────────────────────
b57 = pd.read_csv("/home/claude/run_everything/GSE51057_betas_union.csv", index_col=0)
b32 = pd.read_csv("/home/claude/run_everything/GSE51032_betas_union.csv", index_col=0)
meta = pd.read_csv("/home/claude/run_everything/cohort_metadata.csv")
print(f"GSE51057 betas: {b57.shape}")
print(f"GSE51032 betas: {b32.shape}")

# Coverage check
unilife_cov_57 = len(b57.index.intersection(unilife.index)) / len(unilife)
salas_cov_57 = len(b57.index.intersection(salas.index)) / len(salas)
print(f"\nCoverage: UniLIFE {unilife_cov_57*100:.1f}% / Salas {salas_cov_57*100:.1f}% in GSE51057")

# CHK-3.1 β distribution
def chk_3_1(b, label):
    sample_col = b.columns[0]
    betas = b[sample_col].dropna().astype(float)
    n_extreme = ((betas < 0.1) | (betas > 0.9)).sum() / len(betas)
    n_mid = ((betas >= 0.4) & (betas <= 0.6)).sum() / len(betas)
    return {"label": label, "extreme_pct": float(n_extreme*100),
            "middle_pct": float(n_mid*100), "median": float(betas.median()),
            "passes_chk_3_1": bool(n_extreme > 0.30 and n_mid < 0.10)}

chk31 = [chk_3_1(b57, "GSE51057"), chk_3_1(b32, "GSE51032")]
print(f"CHK-3.1: GSE51057 PASS={chk31[0]['passes_chk_3_1']}, GSE51032 PASS={chk31[1]['passes_chk_3_1']}")

# ── Per-sample deconvolution: BOTH panels ────────────────────────────────
print("\nRunning deconvolution (this takes ~30s for 1174 samples × 2 panels)...")
records = []
n_done = 0
for cohort_label, betas in [("GSE51057", b57), ("GSE51032", b32)]:
    for sample_id in betas.columns:
        rec = {"sample_id": sample_id, "cohort": cohort_label}
        sample_betas = betas[sample_id]
        # UniLIFE
        f_unilife = deconvolve_rpc(sample_betas, unilife)
        for i, ct in enumerate(unilife.columns):
            rec[f"f_UniLIFE_{ct}"] = float(f_unilife[i]) if np.isfinite(f_unilife[i]) else None
        # Salas
        f_salas = deconvolve_rpc(sample_betas, salas)
        for i, ct in enumerate(salas.columns):
            rec[f"f_Salas_{ct}"] = float(f_salas[i]) if np.isfinite(f_salas[i]) else None
        records.append(rec)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}/{1174} samples done")

per_sample = pd.DataFrame(records)
per_sample = per_sample.merge(meta, on=["sample_id", "cohort"], how="left")
per_sample["is_breast_case"] = per_sample["cancer_site"].astype(str).str.upper().str.contains("C50", na=False)
per_sample["is_control"] = (per_sample["group"] == "control")
per_sample.to_csv("/home/claude/run_everything/VAL-095_per_sample.csv", index=False)
print(f"\nPer-sample shape: {per_sample.shape}")

# ── Window-stratified d for each cell type, both panels ──────────────────
WINDOWS = [("0-2yr", 0, 2), ("2-5yr", 2, 5), ("5-10yr", 5, 10), (">10yr", 10, 99)]
results = {
    "val_id": "VAL-095",
    "prereg_sha256": "5f74259d5341268ee7cdaf68322962a275dd19e4158b398f09562f6aaa44bace",
    "rng_seed": RNG_SEED,
    "method": "RPC-style NNLS deconvolution (sum-to-1 normalized) on UniLIFE 19-cell + Salas 450K 6-cell, head-to-head",
    "atlas_unilife": "Guo 2025 UniLIFE 1906 CpGs × 19 cell types",
    "atlas_salas": "Salas 450K legacy 350 CpGs × 6 cell types",
    "platform": "450K (both cohorts)",
    "specimen": "buffy-coat whole blood",
    "chk_3_1": chk31,
    "coverage": {
        "GSE51057": {"unilife_pct": float(unilife_cov_57*100), "salas_pct": float(salas_cov_57*100)},
    },
    "per_cohort": {},
    "head_to_head": {},
}

unilife_cells = list(unilife.columns)
salas_cells = list(salas.columns)

for cohort in ["GSE51057", "GSE51032"]:
    sub = per_sample[per_sample["cohort"] == cohort].copy()
    ctrl = sub[sub["is_control"]]
    coh = {"window_celltype_d_unilife": {}, "window_celltype_d_salas": {},
           "n_per_window": {},
           "healthy_mean_unilife": {ct: float(ctrl[f"f_UniLIFE_{ct}"].mean()) for ct in unilife_cells},
           "healthy_sd_unilife":   {ct: float(ctrl[f"f_UniLIFE_{ct}"].std())  for ct in unilife_cells},
           "healthy_mean_salas":   {ct: float(ctrl[f"f_Salas_{ct}"].mean())   for ct in salas_cells},
           "healthy_sd_salas":     {ct: float(ctrl[f"f_Salas_{ct}"].std())    for ct in salas_cells}}

    for win_label, lo, hi in WINDOWS:
        cases = sub[sub["is_breast_case"] & (sub["ttd_years"] >= lo) & (sub["ttd_years"] < hi)]
        coh["n_per_window"][win_label] = {"cases": int(len(cases)), "controls": int(len(ctrl))}
        if len(cases) < 2:
            coh["window_celltype_d_unilife"][win_label] = {ct: None for ct in unilife_cells}
            coh["window_celltype_d_salas"][win_label] = {ct: None for ct in salas_cells}
            continue
        # UniLIFE per-cell-type d
        u_d = {}
        for ct in unilife_cells:
            col = f"f_UniLIFE_{ct}"
            d = cohen_d(cases[col].dropna().values, ctrl[col].dropna().values)
            u_d[ct] = float(d) if np.isfinite(d) else None
        coh["window_celltype_d_unilife"][win_label] = u_d
        # Salas per-cell-type d
        s_d = {}
        for ct in salas_cells:
            col = f"f_Salas_{ct}"
            d = cohen_d(cases[col].dropna().values, ctrl[col].dropna().values)
            s_d[ct] = float(d) if np.isfinite(d) else None
        coh["window_celltype_d_salas"][win_label] = s_d
    results["per_cohort"][cohort] = coh

# ── Head-to-head: aggregate UniLIFE adult-specific subtypes vs Salas equivalents ──
def agg_unilife_for_salas(unilife_d_dict):
    """Sum UniLIFE subtypes that map to each Salas cell type."""
    return {
        "B": (unilife_d_dict.get("B"), [unilife_d_dict.get("aBmem"), unilife_d_dict.get("aBnv")]),
        "CD4T": (unilife_d_dict.get("CD4T"), [unilife_d_dict.get("aCD4Tnv"), unilife_d_dict.get("aCD4Tmem"), unilife_d_dict.get("aTreg")]),
        "CD8T": (unilife_d_dict.get("CD8T"), [unilife_d_dict.get("aCD8Tnv"), unilife_d_dict.get("aCD8Tmem")]),
        "Mono": (unilife_d_dict.get("Mono"), [unilife_d_dict.get("aMono")]),
        "NK": (unilife_d_dict.get("NK"), [unilife_d_dict.get("aNK")]),
        "Neu": (unilife_d_dict.get("Gran"), [unilife_d_dict.get("aNeu"), unilife_d_dict.get("aEos"), unilife_d_dict.get("aBaso")]),
    }

print("\n=== HEAD-TO-HEAD: UniLIFE pan-lifespan vs Salas 450K, per cohort, per window ===")
for cohort in ["GSE51057", "GSE51032"]:
    print(f"\n{cohort}:")
    print(f"{'Cell type':10s}  {'>10yr':>20s}  {'5-10yr':>20s}  {'2-5yr':>20s}  {'0-2yr':>20s}")
    print(f"{'(panel)':10s}  {'Salas / UniLIFE':>20s}  {'Salas / UniLIFE':>20s}  {'Salas / UniLIFE':>20s}  {'Salas / UniLIFE':>20s}")
    for ct in ["B", "CD4T", "CD8T", "NK", "Mono", "Neu"]:
        row = []
        for win in [">10yr", "5-10yr", "2-5yr", "0-2yr"]:
            sd = results["per_cohort"][cohort]["window_celltype_d_salas"].get(win, {}).get(ct)
            # UniLIFE pan-lifespan equivalent
            ud_map = {"B":"B","CD4T":"CD4T","CD8T":"CD8T","NK":"NK","Mono":"Mono","Neu":"Gran"}
            ud = results["per_cohort"][cohort]["window_celltype_d_unilife"].get(win, {}).get(ud_map[ct])
            ss = f"{sd:+.2f}" if sd is not None else "  -- "
            us = f"{ud:+.2f}" if ud is not None else "  -- "
            row.append(f"{ss}/{us}")
        print(f"{ct:10s}  {row[0]:>20s}  {row[1]:>20s}  {row[2]:>20s}  {row[3]:>20s}")

print("\n=== UniLIFE adult-specific subtypes (12 subtypes), per cohort at >10yr / 0-2yr ===")
for cohort in ["GSE51057", "GSE51032"]:
    print(f"\n{cohort}:")
    adult = [c for c in unilife_cells if c.startswith("a")]
    for ct in adult:
        d_far = results["per_cohort"][cohort]["window_celltype_d_unilife"].get(">10yr", {}).get(ct)
        d_near = results["per_cohort"][cohort]["window_celltype_d_unilife"].get("0-2yr", {}).get(ct)
        s_far = f"{d_far:+.2f}" if d_far is not None else "  -- "
        s_near = f"{d_near:+.2f}" if d_near is not None else "  -- "
        print(f"  {ct:10s}: >10yr={s_far}  0-2yr={s_near}")

with open("/home/claude/run_everything/VAL-095_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults: /home/claude/run_everything/VAL-095_results.json")
