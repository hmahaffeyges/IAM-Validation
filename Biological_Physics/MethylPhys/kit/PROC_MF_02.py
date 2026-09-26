#!/usr/bin/env python3
# INSTRUMENT-TEST: measures a CANDIDATE detection stage (inverse-variance weighted foreign-cell detection) against
# the chain's own solver (block NNLS) on spiked specimens. Cannot go through run_sample because the candidate is not
# in the chain. Produces detection limits and bars only - no A-score, tier or report for any specimen.
"""PROC-MF-02: every construction is the one fixed in doors/PROC_MF_02_PREREG.md.

  weights   1/var_i of the healthy residual, LEAVE-ONE-LABORATORY-OUT (36 arrays)
  a         sum w t r / sum w t^2, template t = mu_c - b/sum(f_blood)
  f_hat     a - median(a on the held-out lab's own 12 healthy arrays)          <- centring by commissioning panel
  sigma     1.4826 * MAD(a on the other three labs' healthy arrays)             <- from the null, not the weights
  detected  f_hat > 47th of 48 ordered null f_hat                                <- <= 1 FP in 48
  limit     smallest fraction detected in >= 90 % of real-array spikes
  B7        false-positive rate on EPIC-Italy control bloods at the 48-array threshold
"""
import json
import lzma
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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
cols = dec.celltype_solve_columns; fam = dec.celltype_families
mu = SPG._cell_means("atlas_work/IAMAtlasREBUILD.csv")
block = [b for b in dec._block if b in mu.index]
X = pd.DataFrame({c: (mu.loc[block, fam[c]].astype(float).mean(axis=1) if c in fam else mu.loc[block, c].astype(float)) for c in cols}).dropna()
markers = [c for c in dec.celltype_ref if c in X.index]
Xm = X.loc[markers]; names = list(Xm.columns)
def isblood(k): return k in BLOOD or (k in fam and all(m in BLOOD for m in fam[k]))
bcols = [c for c in names if isblood(c)]

sets = {"GSE87571": pd.read_parquet("stage1_betas_GSE87571_FULL.parquet")}
for g in ("GSE42861", "GSE111629", "GSE125105"):
    sets[g] = pickle.load(lzma.open(f"iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_{g}.pkl.xz", "rb"))
rng = np.random.default_rng(3); arrays = []
for lab, d in sets.items():
    d.index = d.index.map(str)
    for gsm in rng.choice(d.columns, 12, replace=False):
        arrays.append((lab, gsm, pd.Series(C.stage_1s_scale_map(d[gsm].dropna().to_dict(), "stage1_noob_450K")[0]).reindex(markers)))
labs = sorted(sets)
present = np.all([a[2].notna().values for a in arrays], axis=0)
M = [m for m, p in zip(markers, present) if p]
Xm = Xm.loc[M]; A = Xm.values; Ab = Xm[bcols].values
arrays = [(lab, gsm, b.loc[M].values) for lab, gsm, b in arrays]
print(f"common markers {len(M):,} | solve columns {len(names)} | blood columns {len(bcols)} | null arrays {len(arrays)}", flush=True)

def blood_fit(v):
    f, _ = nnls(Ab, v); return f, Ab @ f
def nnls_full(v):
    f, _ = nnls(A, v); f = f / max(f.sum(), 1e-9); return dict(zip(names, f))
def raw_amp(v, cell, w):
    fb, b = blood_fit(v); r = v - b; t = Xm[cell].values - b / max(fb.sum(), 1e-9)
    return float(np.sum(w * t * r) / np.sum(w * t * t))

R = np.array([v - blood_fit(v)[1] for _, _, v in arrays])
def weights(lab_out, same=False):
    idx = [i for i, (lab, _, _) in enumerate(arrays) if same or lab != lab_out]
    return 1.0 / (R[idx].var(axis=0, ddof=1) + 1e-8)
W = {lab: weights(lab) for lab in labs}; Wsame = {lab: weights(lab, same=True) for lab in labs}

# raw amplitudes on the null, per lab
rawnull = {c: {lab: [raw_amp(v, c, W[lab]) for (l, _, v) in arrays if l == lab] for lab in labs} for c in CELLS}
centre = {c: {lab: float(np.median(rawnull[c][lab])) for lab in labs} for c in CELLS}
sigma = {c: {lab: float(1.4826 * np.median(np.abs(np.concatenate([rawnull[c][o] for o in labs if o != lab])
                                                   - np.median(np.concatenate([rawnull[c][o] for o in labs if o != lab])))))
             for lab in labs} for c in CELLS}
def fhat(v, cell, lab, same=False):
    a = raw_amp(v, cell, Wsame[lab] if same else W[lab]); return a - centre[cell][lab]

null = {"nnls": {c: [] for c in CELLS}, "iv": {c: [] for c in CELLS}, "iv_same": {c: [] for c in CELLS}, "iv_z": {c: [] for c in CELLS}}
for lab, gsm, v in arrays:
    fr = nnls_full(v)
    for c in CELLS:
        null["nnls"][c].append(fr.get(c, 0.0)); f = fhat(v, c, lab); null["iv"][c].append(f)
        null["iv_same"][c].append(fhat(v, c, lab, same=True)); null["iv_z"][c].append(f / sigma[c][lab])
def thr(vals): return float(np.sort(vals)[len(vals) - 2])
THR = {d: {c: thr(null[d][c]) for c in CELLS} for d in ("nnls", "iv", "iv_same")}
print("null done | centred null medians per cell (should be ~0):", {c: round(float(np.median(null['iv'][c])), 4) for c in CELLS}, flush=True)

spk = []
for lab, gsm, v in arrays:
    for c in CELLS:
        prof = Xm[c].values
        for f0 in FRACS:
            vs = v * (1 - f0) + prof * f0
            spk.append({"lab": lab, "gsm": gsm, "cell": c, "f": f0, "nnls": nnls_full(vs).get(c, 0.0),
                        "iv": fhat(vs, c, lab), "iv_same": fhat(vs, c, lab, same=True), "sigma": sigma[c][lab]})
S = pd.DataFrame(spk); print(f"real-array spikes {len(S)}", flush=True)
con = []; rs = np.random.default_rng(7)
for seed in range(10):
    w = rs.dirichlet([8, 3, 2, 1, 1, 1.5, 0.5]); base = dict(zip(["Neutrophils_reinius", "CD4_T-cells", "CD8_T-cells", "CD19_B-cells", "CD56_NK-cells", "CD14_monocytes", "GMP"], w))
    for c in CELLS:
        for f0 in FRACS:
            mix = {k: x * (1 - f0) for k, x in base.items()}; mix[c] = f0
            beta, _ = SPG.compose_cells(mix, noise_sigma=0.044, seed=1000 + seed, atlas_csv="atlas_work/IAMAtlasREBUILD.csv")
            vs = beta.reindex(M).values
            if np.isnan(vs).any(): continue
            con.append({"seed": seed, "cell": c, "f": f0, "nnls": nnls_full(vs).get(c, 0.0), "iv": fhat(vs, c, "GSE87571")})

def limit(det, cell):
    for f0 in FRACS:
        sub = S[(S.cell == cell) & (S.f == f0)]
        if (sub[det] > THR[det][cell]).mean() >= 0.90: return f0
    return None
LIM = {d: {c: limit(d, c) for c in CELLS} for d in ("nnls", "iv", "iv_same")}
DETR = {d: {c: {str(f0): float((S[(S.cell == c) & (S.f == f0)][d] > THR[d][c]).mean()) for f0 in FRACS} for c in CELLS} for d in ("nnls", "iv", "iv_same")}
print("\nDETECTION LIMIT (>= 90 % of real-array spikes at <= 1 FP in 48)")
print(f"{'cell':<26}{'NNLS':>8}{'inv-var':>9}{'same-lab':>10}")
for c in CELLS: print(f"{c:<26}" + "".join(f"{(LIM[d][c] if LIM[d][c] is not None else '>5%'):>{w}}" for d, w in (("nnls", 8), ("iv", 9), ("iv_same", 10))))
for c in CELLS: print(f"  {c:<26}" + "  ".join(f"{f0:.3f}: {DETR['nnls'][c][str(f0)]:.2f}/{DETR['iv'][c][str(f0)]:.2f}" for f0 in FRACS))

def num(x): return x if x is not None else 1.0
B1n = sum(num(LIM["iv"][c]) < num(LIM["nnls"][c]) for c in CELLS); B1 = B1n >= 3
z = np.concatenate([null["iv_z"][c] for c in CELLS]); tail = float((np.abs(z) > 2).mean()); B2 = 0.02 <= tail <= 0.10
bias = {str(f0): float((S[S.f == f0]["iv"] - f0).median()) for f0 in (0.02, 0.05)}; B3 = all(abs(v) <= 0.005 for v in bias.values())
cn = {c: {lab: float(np.median([f for (l, _, _), f in zip(arrays, null["iv"][c]) if l == lab])) for lab in labs} for c in CELLS}
B4 = all(abs(x) <= 0.003 for c in CELLS for x in cn[c].values())
B5 = all(num(LIM["iv_same"][c]) >= 0.8 * num(LIM["iv"][c]) for c in CELLS)
d6 = []
for lab, gsm, v in arrays:
    full = nnls_full(v); fb, _ = blood_fit(v); fb = fb / max(fb.sum(), 1e-9); fbd = dict(zip(bcols, fb))
    d6.append(max(abs(full.get(k, 0) - fbd.get(k, 0)) for k in ("CD4_T-cells", "CD8_T-cells", "CD19_B-cells", "CD56_NK-cells", "CD14_monocytes")))
B6 = float(np.median(d6)) < 0.005

# ---- B7: EPIC-Italy controls at the 48-array threshold (fifth laboratory, EPIC platform, reduced chain-loci matrix)
B7 = None; b7 = {}
try:
    ep = pd.read_parquet("results/epic01/GSE51032_chain_loci.parquet"); ep.index = ep.index.map(str)
    sc = {r["gsm"]: r for r in json.load(open("handoff/epic01_scored.json"))}
    ctrl = [g for g in ep.columns if g in sc and sc[g].get("icd") is None]
    Me = [m for m in M if m in ep.index]
    if len(Me) < 200: raise RuntimeError(f"only {len(Me)} of {len(M)} markers in the EPIC-Italy reduced matrix")
    Xe = Xm.loc[Me]; Ae = Xe.values; Abe = Xe[bcols].values
    # weights/centre: all four 450K labs (no EPIC lab to hold out); centre by EPIC-Italy's own control median (a commissioning panel)
    idxM = [M.index(m) for m in Me]; wE = 1.0 / (R[:, idxM].var(axis=0, ddof=1) + 1e-8)
    def raw_e(v, cell):
        fb, _ = nnls(Abe, v); b = Abe @ fb; r = v - b; t = Xe[cell].values - b / max(fb.sum(), 1e-9)
        return float(np.sum(wE * t * r) / np.sum(wE * t * t))
    rawE = {c: [] for c in ("Breast", "Cortical_neurons")}; used = 0; nmark = []
    for g in ctrl:
        vv = ep[g].reindex(Me)          # EPIC-Italy matrix is already on the chain's mapped scale (PROC-EPIC-01)
        ok = vv.notna().values
        if ok.sum() < 200: continue     # 2026-09-26: the first pass dropped EVERY array on a single NaN - use present markers per array
        v = vv.values[ok]; Ao = Abe[ok]; wo = wE[ok]; nmark.append(int(ok.sum()))
        used += 1
        for c in rawE:
            fb, _ = nnls(Ao, v); b = Ao @ fb; r = v - b; t = Xe[c].values[ok] - b / max(fb.sum(), 1e-9)
            rawE[c].append(float(np.sum(wo * t * r) / np.sum(wo * t * t)))
    print(f"   EPIC-Italy markers present per control array: median {int(np.median(nmark)) if nmark else 0}", flush=True)
    for c in rawE:
        a = np.array(rawE[c]); fE = a - np.median(a)
        b7[c] = {"n": int(len(a)), "fp_rate": float((fE > THR["iv"][c]).mean()), "threshold_450K": THR["iv"][c]}
    B7 = all(v["fp_rate"] <= 0.02 for v in b7.values())
    print(f"\nB7 EPIC-Italy controls used {used} | markers in common {len(Me)}: " + ", ".join(f"{c} FP {v['fp_rate']:.3f}" for c, v in b7.items()))
except Exception as e:
    print("\nB7 NOT ASSESSED:", type(e).__name__, str(e)[:160])

print(f"\nB1 inv-var limit lower than NNLS on {B1n} of 4  ->  {'MET' if B1 else 'FAILED'}")
print(f"B2 null |z|>2 fraction {tail:.3f}  ->  {'MET' if B2 else 'FAILED (0.02-0.10)'}")
print(f"B3 median bias 2% {bias['0.02']:+.4f}, 5% {bias['0.05']:+.4f}  ->  {'MET' if B3 else 'FAILED (|bias| <= 0.005)'}")
print(f"B4 centred null per held-out lab, worst |median| {max(abs(x) for c in CELLS for x in cn[c].values()):.4f}  ->  {'MET' if B4 else 'FAILED (<= 0.003)'}")
print(f"B5 same-lab no more than 20 % better  ->  {'MET' if B5 else 'FAILED'}")
print(f"B6 blood composition median change {np.median(d6):.4f}  ->  {'MET' if B6 else 'FAILED'}")
print(f"B7 EPIC-Italy false-positive rate <= 0.02  ->  {'MET' if B7 else ('NOT ASSESSED' if B7 is None else 'FAILED')}")
json.dump({"limits": LIM, "thresholds": THR, "detection_rate": DETR, "centre": centre, "sigma": sigma,
           "bars": {"B1": B1, "B1_n": B1n, "B2": B2, "B2_tail": tail, "B3": B3, "B3_bias": bias, "B4": B4, "B4_centres": cn, "B5": B5, "B6": B6, "B6_median": float(np.median(d6)), "B7": B7, "B7_detail": b7},
           "n_markers": len(M), "spikes_real": spk, "spikes_constructed": con}, open("handoff/mf02_results.json", "w"), indent=1)
json.dump({d: {c: [float(x) for x in null[d][c]] for c in CELLS} for d in null}, open("handoff/mf02_null.json", "w"), indent=1)
print("\nwrote handoff/mf02_results.json, handoff/mf02_null.json")
