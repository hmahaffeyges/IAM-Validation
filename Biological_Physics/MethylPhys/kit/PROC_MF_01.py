#!/usr/bin/env python3
# INSTRUMENT-TEST: compares a CANDIDATE detector (covariance-weighted matched filter) against the chain's own
# solver (block NNLS) on spiked specimens. Cannot go through run_sample because the candidate is not in the
# chain. Produces detection limits and bars only - no A-score, tier or report for any specimen.
"""PROC-MF-01: does a covariance-weighted matched filter lower the minimum detectable fraction of a foreign cell
in blood? Every construction is the one fixed in doors/PROC_MF_01_PREREG.md before this file was written.

  null        48 healthy arrays, 12 per laboratory, mapped (stage1_noob_450K), same draw as today's floor measurement
  N           residual covariance after the blood-only fit, LEAVE-ONE-LABORATORY-OUT, Ledoit-Wolf shrinkage
  template    t_c = mu_c - (blood background at those loci); r = specimen - blood-only reconstruction
  filter      f = (t N^-1 r) / (t N^-1 t),  sigma^2 = 1 / (t N^-1 t)
  NNLS        the repaired block solve (today's chain), fraction read directly
  threshold   per detector per cell: the 47th of 48 ordered null readings (<= 1 false positive in 48)
  limit       smallest spiked fraction at which >= 90 % of spikes exceed the threshold
  spikes      Breast, Colon_epithelial_cells, Cortical_neurons, Prostate x {0.005, 0.01, 0.02, 0.05}
              (a) into REAL healthy arrays: mapped array x (1-f) + atlas profile x f       <- decides B1
              (b) constructed blood background, 10 seeds                                   <- reported only
Controls fixed in advance: diagonal-only N (B4); N estimated on the scored laboratory (B5).
"""
import json
import lzma
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.covariance import LedoitWolf

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH + "/Synthetic_Patient_Generator")
sys.path.insert(0, CH)
import synthetic_patient_generator as SPG  # noqa: E402
import cpg_conductor as C  # noqa: E402

CELLS = ["Breast", "Colon_epithelial_cells", "Cortical_neurons", "Prostate"]
FRACS = [0.005, 0.01, 0.02, 0.05]
BLOOD = {'Neutrophils_reinius', 'CD4_T-cells', 'CD8_T-cells', 'CD14_monocytes', 'CD19_B-cells', 'CD56_NK-cells',
         'Eosinophils_reinius', 'GMP', 'HSC', 'MPP', 'MEP', 'L-MPP', 'CMP'}

dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
dec = dec_mod.WaltherIAMDeconvolver("atlas_work/IAMAtlasREBUILD.csv",
                                    celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
cols = dec.celltype_solve_columns
fam = dec.celltype_families
mu = SPG._cell_means("atlas_work/IAMAtlasREBUILD.csv")
block = [b for b in dec._block if b in mu.index]
X = pd.DataFrame({c: (mu.loc[block, fam[c]].astype(float).mean(axis=1) if c in fam else mu.loc[block, c].astype(float))
                  for c in cols}).dropna()
markers = [c for c in dec.celltype_ref if c in X.index]
Xm = X.loc[markers]
names = list(Xm.columns)


def isblood(k):
    return k in BLOOD or (k in fam and all(m in BLOOD for m in fam[k]))


bcols = [c for c in names if isblood(c)]
Xb = Xm[bcols]
print(f"block markers {len(markers):,} | solve columns {len(names)} | blood columns {len(bcols)}", flush=True)

# ---------------------------------------------------------------- the 48 healthy arrays (same draw as this morning)
sets = {"GSE87571": pd.read_parquet("stage1_betas_GSE87571_FULL.parquet")}
for g in ("GSE42861", "GSE111629", "GSE125105"):
    sets[g] = pickle.load(lzma.open(f"iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_{g}.pkl.xz", "rb"))
rng = np.random.default_rng(3)
arrays = []
for lab, d in sets.items():
    d.index = d.index.map(str)
    for gsm in rng.choice(d.columns, 12, replace=False):
        b = pd.Series(C.stage_1s_scale_map(d[gsm].dropna().to_dict(), "stage1_noob_450K")[0])
        arrays.append((lab, gsm, b.reindex(markers)))
labs = sorted(sets)
print(f"healthy arrays: {len(arrays)} | markers present per array: median {int(np.median([a[2].notna().sum() for a in arrays]))}", flush=True)

# a common marker set present on every array, so N is one matrix
present = np.all([a[2].notna().values for a in arrays], axis=0)
M = [m for m, p in zip(markers, present) if p]
Xm = Xm.loc[M]; Xb = Xb.loc[M]
A = Xm.values; Ab = Xb.values
arrays = [(lab, gsm, b.loc[M].values) for lab, gsm, b in arrays]
print(f"common markers: {len(M):,}", flush=True)


def blood_fit(v):
    f, _ = nnls(Ab, v)
    return f, Ab @ f


def nnls_full(v):
    f, _ = nnls(A, v)
    f = f / max(f.sum(), 1e-9)
    return dict(zip(names, f))


# residuals of the null after the blood-only fit
R = np.array([v - blood_fit(v)[1] for _, _, v in arrays])           # 48 x M
bg = np.array([blood_fit(v)[1] for _, _, v in arrays])               # blood reconstruction per array

# ---------------------------------------------------------------- covariance, leave-one-laboratory-out
def cov_loo(lab_out, diagonal=False, no_leave_out=False):
    idx = [i for i, (lab, _, _) in enumerate(arrays) if (lab != lab_out) or no_leave_out]
    Rr = R[idx]
    if diagonal:
        return np.diag(Rr.var(axis=0, ddof=1) + 1e-8), None
    lw = LedoitWolf().fit(Rr)
    return lw.covariance_, float(lw.shrinkage_)


COVS = {}
for mode in ("full", "diag", "same"):
    for lab in labs:
        Ncov, shr = cov_loo(lab, diagonal=(mode == "diag"), no_leave_out=(mode == "same"))
        COVS[(mode, lab)] = (np.linalg.pinv(Ncov, rcond=1e-10, hermitian=True), shr)
print("covariances built; Ledoit-Wolf shrinkage (full, per held-out lab):",
      {lab: round(COVS[('full', lab)][1], 3) for lab in labs}, flush=True)


def mf(v, cell, Ninv):
    """matched-filter amplitude and sigma for one cell on one specimen"""
    f_b, recon = blood_fit(v)
    r = v - recon
    t = Xm[cell].values - recon / max(f_b.sum(), 1e-9)   # template: cell profile minus the specimen's own blood background
    tN = t @ Ninv
    denom = float(tN @ t)
    return float(tN @ r) / denom, float(1.0 / np.sqrt(denom))


# ---------------------------------------------------------------- null readings, both detectors, all modes
null = {"nnls": {c: [] for c in CELLS}, "mf_full": {c: [] for c in CELLS}, "mf_diag": {c: [] for c in CELLS},
        "mf_same": {c: [] for c in CELLS}, "mf_full_z": {c: [] for c in CELLS}}
for lab, gsm, v in arrays:
    fr = nnls_full(v)
    for c in CELLS:
        null["nnls"][c].append(fr.get(c, 0.0))
        for mode in ("full", "diag", "same"):
            f, s = mf(v, c, COVS[(mode, lab)][0])
            null[f"mf_{mode}"][c].append(f)
            if mode == "full":
                null["mf_full_z"][c].append(f / s)
print("null readings done", flush=True)


def threshold(vals):
    return float(np.sort(vals)[len(vals) - 2])      # 47th of 48: <= 1 false positive


THR = {det: {c: threshold(null[det][c]) for c in CELLS} for det in ("nnls", "mf_full", "mf_diag", "mf_same")}

# ---------------------------------------------------------------- spikes into REAL arrays (decides B1)
spk = []
for lab, gsm, v in arrays:
    for c in CELLS:
        prof = Xm[c].values
        for f0 in FRACS:
            vs = v * (1 - f0) + prof * f0
            fr = nnls_full(vs)
            row = {"lab": lab, "gsm": gsm, "cell": c, "f": f0, "bg": "real", "nnls": fr.get(c, 0.0)}
            for mode in ("full", "diag", "same"):
                fm, s = mf(vs, c, COVS[(mode, lab)][0])
                row[f"mf_{mode}"] = fm
                if mode == "full":
                    row["mf_full_sigma"] = s
            spk.append(row)
print(f"real-array spikes scored: {len(spk)}", flush=True)

# ---------------------------------------------------------------- spikes into constructed blood (reported only)
con = []
rs = np.random.default_rng(7)
for seed in range(10):
    w = rs.dirichlet([8, 3, 2, 1, 1, 1.5, 0.5])
    base = dict(zip(["Neutrophils_reinius", "CD4_T-cells", "CD8_T-cells", "CD19_B-cells", "CD56_NK-cells", "CD14_monocytes", "GMP"], w))
    for c in CELLS:
        for f0 in FRACS:
            mix = {k: v * (1 - f0) for k, v in base.items()}; mix[c] = f0
            beta, _ = SPG.compose_cells(mix, noise_sigma=0.044, seed=1000 + seed, atlas_csv="atlas_work/IAMAtlasREBUILD.csv")
            vs = beta.reindex(M).values
            if np.isnan(vs).any():
                vs = np.where(np.isnan(vs), Xm[c].values * f0 + Ab.mean(axis=1) * (1 - f0), vs)
            fr = nnls_full(vs)
            fm, s = mf(vs, c, COVS[("full", "GSE87571")][0])
            con.append({"seed": seed, "cell": c, "f": f0, "bg": "constructed", "nnls": fr.get(c, 0.0), "mf_full": fm, "mf_full_sigma": s})
print("constructed spikes scored", flush=True)

# ---------------------------------------------------------------- detection limits
S = pd.DataFrame(spk)
def limit(det, cell, frame=S):
    for f0 in FRACS:
        sub = frame[(frame.cell == cell) & (frame.f == f0)]
        if (sub[det] > THR[det][cell]).mean() >= 0.90:
            return f0
    return None

LIM = {det: {c: limit(det, c) for c in CELLS} for det in ("nnls", "mf_full", "mf_diag", "mf_same")}
DETR = {det: {c: {str(f0): float((S[(S.cell == c) & (S.f == f0)][det] > THR[det][c]).mean()) for f0 in FRACS} for c in CELLS}
        for det in ("nnls", "mf_full", "mf_diag", "mf_same")}

print("\nDETECTION LIMIT (smallest fraction detected in >= 90 % of real-array spikes at <= 1 FP in 48)")
print(f"{'cell':<26}{'NNLS':>8}{'MF full':>9}{'MF diag':>9}{'MF same-lab':>12}")
for c in CELLS:
    print(f"{c:<26}" + "".join(f"{(LIM[d][c] if LIM[d][c] is not None else '>5%'):>{w}}" for d, w in
                                 (("nnls", 8), ("mf_full", 9), ("mf_diag", 9), ("mf_same", 12))))
print("\ndetection rate by fraction, NNLS vs MF full:")
for c in CELLS:
    print(f"  {c:<26}" + "  ".join(f"{f0:.3f}: {DETR['nnls'][c][str(f0)]:.2f}/{DETR['mf_full'][c][str(f0)]:.2f}" for f0 in FRACS))

# ---------------------------------------------------------------- bars
def lower(a, b):   # is limit a lower than limit b (None = not reached = worst)
    a = a if a is not None else 1.0; b = b if b is not None else 1.0
    return a < b

b1_wins = sum(lower(LIM["mf_full"][c], LIM["nnls"][c]) for c in CELLS)
b1_ties = sum((LIM["mf_full"][c] == LIM["nnls"][c]) for c in CELLS)
B1 = b1_wins >= 3
z = np.concatenate([null["mf_full_z"][c] for c in CELLS])
tail = float((np.abs(z) > 2).mean()); B2 = 0.02 <= tail <= 0.10
bias = {}
for f0 in (0.02, 0.05):
    sub = S[S.f == f0]; bias[str(f0)] = float((sub["mf_full"] - sub["f"]).median())
B3 = all(abs(v) <= 0.005 for v in bias.values())
# B4: diagonal limit strictly between (or equal to NNLS) - fails if diag == full on every cell where full beat nnls
b4_cells = [c for c in CELLS if lower(LIM["mf_full"][c], LIM["nnls"][c])]
B4 = any(lower(LIM["mf_full"][c], LIM["mf_diag"][c]) for c in b4_cells) if b4_cells else None
# B5: same-lab limit not more than 20 % better than leave-one-out
def num(x): return x if x is not None else 1.0
B5 = all(num(LIM["mf_same"][c]) >= 0.8 * num(LIM["mf_full"][c]) for c in CELLS)
# B6: blood composition unchanged (the filter's background fit IS the blood-only NNLS; compare to the full NNLS blood fractions)
d6 = []
for lab, gsm, v in arrays:
    full = nnls_full(v); fb, _ = blood_fit(v); fb = fb / max(fb.sum(), 1e-9); fbd = dict(zip(bcols, fb))
    d6.append(max(abs(full.get(k, 0) - fbd.get(k, 0)) for k in ("CD4_T-cells", "CD8_T-cells", "CD19_B-cells", "CD56_NK-cells", "CD14_monocytes")))
B6 = float(np.median(d6)) < 0.005

print(f"\nB1 MF limit lower than NNLS on {b1_wins} of 4 cells ({b1_ties} tie)  ->  {'MET' if B1 else 'FAILED (bar: >= 3)'}")
print(f"B2 null |z|>2 fraction {tail:.3f}  ->  {'MET' if B2 else 'FAILED (bar: 0.02-0.10)'}")
print(f"B3 median bias at 2% {bias['0.02']:+.4f}, 5% {bias['0.05']:+.4f}  ->  {'MET' if B3 else 'FAILED (bar: |bias| <= 0.005)'}")
print(f"B4 structured covariance beats diagonal on {sum(lower(LIM['mf_full'][c], LIM['mf_diag'][c]) for c in CELLS)} cells  ->  "
      f"{'MET' if B4 else ('NOT ASSESSABLE (MF never beat NNLS)' if B4 is None else 'FAILED - diagonal weights do all the work')}")
print(f"B5 same-lab covariance no more than 20 % better  ->  {'MET' if B5 else 'FAILED - covariance memorises its arrays'}")
print(f"B6 blood composition change median {np.median(d6):.4f}  ->  {'MET' if B6 else 'FAILED (bar: < 0.005)'}")

json.dump({"limits": LIM, "thresholds": THR, "detection_rate": DETR, "bars": {"B1": B1, "B1_wins": b1_wins, "B2": B2, "B2_tail": tail, "B3": B3, "B3_bias": bias,
           "B4": B4, "B5": B5, "B6": B6, "B6_median": float(np.median(d6))}, "n_markers": len(M), "n_null": len(arrays),
           "shrinkage": {lab: COVS[("full", lab)][1] for lab in labs}, "spikes_real": spk, "spikes_constructed": con},
          open("handoff/mf01_results.json", "w"), indent=1)
json.dump({det: {c: [float(x) for x in null[det][c]] for c in CELLS} for det in null}, open("handoff/mf01_null.json", "w"), indent=1)
print("\nwrote handoff/mf01_results.json, handoff/mf01_null.json")
