"""
VAL-094 — EpiSCORE breast Stage 2, all 4 TTD windows, dual cohort
Heath W. Mahaffey | IAMPerformance | 2026-04-26
RNG seed: 20260426
Pre-reg SHA-256: 501fafad68fa93635a18f43687104756f006ea89ed301de80ac469514ae15626
"""

import json
import math
import numpy as np
import pandas as pd
import rdata
import os

RNG_SEED = 20260426
H_MIN_SECRETORY = 0.8433
N_TOP_PER_CELL = 80


def H_vec(arr):
    a = np.asarray(arr, dtype=float)
    out = np.zeros_like(a)
    mask = (a > 0) & (a < 1)
    out[mask] = -a[mask]*np.log2(a[mask]) - (1-a[mask])*np.log2(1-a[mask])
    return out


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return float("nan")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy) / (nx+ny-2)
    if pooled <= 0: return float("nan")
    return (mx - my) / np.sqrt(pooled)


def bootstrap_ci(x, y, n_iter=1000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
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
    lo = np.percentile(boot_d, 100*alpha/2)
    hi = np.percentile(boot_d, 100*(1-alpha/2))
    return (float(lo), float(hi))


breast_ref = pd.read_csv("/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_zhu_teschendorff_2022/BreastRef__mrefBreast_m.csv", index_col=0)
cell_types = [c for c in breast_ref.columns if c != "weight"]
print(f"EpiSCORE BreastRef: {breast_ref.shape[0]} markers × {len(cell_types)} cell types: {cell_types}")

parsed = rdata.read_rda("/home/claude/atlases/episcore/EpiSCORE-master/data/probeInfo450k.rda")
obj = parsed["probeInfo450k.lv"]
probe_ids = np.asarray(obj["probeID"])
entrez_ids = np.asarray(obj["EID"])

def to_int_safe(x):
    try: return int(x)
    except: return None

entrez_to_cpgs = {}
for cpg, eid in zip(probe_ids, entrez_ids):
    e = to_int_safe(eid)
    if e is not None:
        entrez_to_cpgs.setdefault(e, []).append(str(cpg))
print(f"Bridge: {len(entrez_to_cpgs)} unique Entrez IDs → CpGs")

cell_to_cpgs = {}
for ct in cell_types:
    others = [c for c in cell_types if c != ct]
    spec = (breast_ref[ct] - breast_ref[others].mean(axis=1)).abs()
    top_entrez = spec.nlargest(N_TOP_PER_CELL).index.tolist()
    cpgs = []
    for e in top_entrez:
        cpgs.extend(entrez_to_cpgs.get(int(e), []))
    cell_to_cpgs[ct] = cpgs
    print(f"  {ct}: top-{N_TOP_PER_CELL} Entrez → {len(set(cpgs))} unique CpGs")

b57 = pd.read_csv("/home/claude/run_everything/GSE51057_betas_union.csv", index_col=0)
b32 = pd.read_csv("/home/claude/run_everything/GSE51032_betas_union.csv", index_col=0)
meta = pd.read_csv("/home/claude/run_everything/cohort_metadata.csv")

def chk_3_1(b, label):
    sample_col = b.columns[0]
    betas = b[sample_col].dropna().astype(float)
    n_extreme = ((betas < 0.1) | (betas > 0.9)).sum() / len(betas)
    n_mid = ((betas >= 0.4) & (betas <= 0.6)).sum() / len(betas)
    return {"label": label, "extreme_pct": float(n_extreme*100),
            "middle_pct": float(n_mid*100), "median": float(betas.median()),
            "passes_chk_3_1": bool(n_extreme > 0.30 and n_mid < 0.10)}

chk31 = [chk_3_1(b57, "GSE51057_sample0"), chk_3_1(b32, "GSE51032_sample0")]
print(f"\nCHK-3.1: GSE51057 extreme={chk31[0]['extreme_pct']:.1f}%, mid={chk31[0]['middle_pct']:.1f}% — {'PASS' if chk31[0]['passes_chk_3_1'] else 'FAIL'}")
print(f"CHK-3.1: GSE51032 extreme={chk31[1]['extreme_pct']:.1f}%, mid={chk31[1]['middle_pct']:.1f}% — {'PASS' if chk31[1]['passes_chk_3_1'] else 'FAIL'}")

def per_sample_A(betas, sample_col, cpg_list, h_min=H_MIN_SECRETORY):
    avail = [c for c in cpg_list if c in betas.index]
    if not avail: return float("nan")
    vals = betas[sample_col].loc[avail].dropna().astype(float).values
    if len(vals) == 0: return float("nan")
    h_vals = H_vec(vals)
    valid = (h_vals > 0) & np.isfinite(h_vals)
    if not valid.any(): return float("nan")
    return float(np.mean(h_vals[valid] / h_min))

print("\nComputing per-sample EpiSCORE A-scores ...")
records = []
for cohort_label, betas in [("GSE51057", b57), ("GSE51032", b32)]:
    for sample_id in betas.columns:
        rec = {"sample_id": sample_id, "cohort": cohort_label}
        for ct in cell_types:
            rec[f"A_EpiSCORE_{ct}"] = per_sample_A(betas, sample_id, cell_to_cpgs[ct])
        records.append(rec)

per_sample = pd.DataFrame(records)
per_sample = per_sample.merge(meta, on=["sample_id", "cohort"], how="left")
print(f"Per-sample shape: {per_sample.shape}")
per_sample.to_csv("/home/claude/run_everything/VAL-094_per_sample.csv", index=False)

WINDOWS = [("0-2yr", 0, 2), ("2-5yr", 2, 5), ("5-10yr", 5, 10), (">10yr", 10, 99)]
per_sample["is_breast_case"] = per_sample["cancer_site"].astype(str).str.upper().str.contains("C50", na=False)
per_sample["is_control"] = (per_sample["group"] == "control")

results = {
    "val_id": "VAL-094",
    "prereg_sha256": "501fafad68fa93635a18f43687104756f006ea89ed301de80ac469514ae15626",
    "rng_seed": RNG_SEED,
    "atlas": "EpiSCORE BreastRef mref (DNAm-derived 8-cell-type breast reference, 'weight' col dropped)",
    "bridge": "probeInfo450k Entrez ID -> 450K CpG mapping",
    "cell_types_scored": cell_types,
    "h_min_anchor": H_MIN_SECRETORY,
    "h_min_anchor_class": "secretory",
    "n_top_per_cell": N_TOP_PER_CELL,
    "chk_3_1_beta_distribution": chk31,
    "per_cohort": {},
    "healthy_baseline_check": {},
}

for cohort in ["GSE51057", "GSE51032"]:
    sub = per_sample[per_sample["cohort"] == cohort].copy()
    ctrl = sub[sub["is_control"]]
    coh = {"window_cell_d": {}, "n_per_window": {}, "healthy_mean": {}, "healthy_sd": {}}
    for ct in cell_types:
        col = f"A_EpiSCORE_{ct}"
        coh["healthy_mean"][ct] = float(ctrl[col].mean())
        coh["healthy_sd"][ct]   = float(ctrl[col].std())
    for win_label, lo, hi in WINDOWS:
        cases = sub[sub["is_breast_case"] & (sub["ttd_years"] >= lo) & (sub["ttd_years"] < hi)]
        coh["n_per_window"][win_label] = {"cases": int(len(cases)), "controls": int(len(ctrl))}
        if len(cases) < 2:
            coh["window_cell_d"][win_label] = {ct: {"d": None, "ci": [None, None]} for ct in cell_types}
            continue
        win_d = {}
        for ct in cell_types:
            col = f"A_EpiSCORE_{ct}"
            d = cohen_d(cases[col].dropna().values, ctrl[col].dropna().values)
            ci = bootstrap_ci(cases[col].dropna().values, ctrl[col].dropna().values,
                              n_iter=1000, rng=np.random.default_rng(RNG_SEED + hash(ct + win_label) % 1000))
            win_d[ct] = {"d": float(d) if np.isfinite(d) else None,
                         "ci": [float(ci[0]) if np.isfinite(ci[0]) else None,
                                float(ci[1]) if np.isfinite(ci[1]) else None]}
        coh["window_cell_d"][win_label] = win_d
    results["per_cohort"][cohort] = coh

gse57_h = results["per_cohort"]["GSE51057"]["healthy_mean"]
gse32_h = results["per_cohort"]["GSE51032"]["healthy_mean"]
gse57_sd = results["per_cohort"]["GSE51057"]["healthy_sd"]
flagged = []
for ct in cell_types:
    delta = abs(gse57_h[ct] - gse32_h[ct])
    sd = gse57_sd[ct]
    if sd > 0 and delta > sd:
        flagged.append({"cell_type": ct, "delta": float(delta), "anchor_sd": float(sd)})
results["healthy_baseline_check"] = {
    "flagged_cell_types": flagged,
    "passes_chk_3_2": len(flagged) == 0,
}

with open("/home/claude/run_everything/VAL-094_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== VAL-094 HEADLINE: per-cell-type d at each window (GSE51057 / GSE51032) ===")
print(f"{'Cell type':12s}  {'>10yr':>14s}  {'5-10yr':>14s}  {'2-5yr':>14s}  {'0-2yr':>14s}")
for ct in cell_types:
    row = []
    for win in [">10yr", "5-10yr", "2-5yr", "0-2yr"]:
        d57 = results["per_cohort"]["GSE51057"]["window_cell_d"].get(win, {}).get(ct, {}).get("d")
        d32 = results["per_cohort"]["GSE51032"]["window_cell_d"].get(win, {}).get(ct, {}).get("d")
        s57 = f"{d57:+.2f}" if d57 is not None else "  -- "
        s32 = f"{d32:+.2f}" if d32 is not None else "  -- "
        row.append(f"{s57}/{s32}")
    print(f"{ct:12s}  {row[0]:>14s}  {row[1]:>14s}  {row[2]:>14s}  {row[3]:>14s}")

print(f"\nCHK-3.2: {len(flagged)} cross-cohort cell-type baseline mismatches ({'PASS' if results['healthy_baseline_check']['passes_chk_3_2'] else 'FAIL'})")
print(f"Results: /home/claude/run_everything/VAL-094_results.json")
