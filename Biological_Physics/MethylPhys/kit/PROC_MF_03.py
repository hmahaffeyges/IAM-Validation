#!/usr/bin/env python3
# INSTRUMENT-TEST: measures a CANDIDATE detection stage (inverse-variance foreign-cell detection with a per-laboratory
# threshold) against the chain's own solver on spiked specimens from five laboratories. Cannot go through run_sample
# because the candidate is not in the chain. Produces detection limits and bars only - no A-score, tier or report.
"""PROC-MF-03: every construction is the one fixed in doors/PROC_MF_03_PREREG.md.

  weights     1/var_i of the healthy residual, leave-one-laboratory-out on the four 450K labs (36 arrays);
              for EPIC-Italy, all 48 450K arrays (no EPIC lab to hold out - EPIC never contributes to weights)
  centre      each laboratory's OWN null median of the raw amplitude
  line        each laboratory's OWN 1 - 1/n quantile of f_hat on its healthy panel (n = 12 on 450K, 424 on EPIC)
  sigma       450K: MAD of the other three labs' null; EPIC: MAD of its own 424 controls (reported)
  limit       smallest fraction with >= 90 % of spikes above the laboratory's line
"""
import json
import lzma
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH + "/Synthetic_Patient_Generator"); sys.path.insert(0, CH)
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

# ---- 450K null, same draw as MF-01/02
sets = {"GSE87571": pd.read_parquet("stage1_betas_GSE87571_FULL.parquet")}
for g in ("GSE42861", "GSE111629", "GSE125105"):
    sets[g] = pickle.load(lzma.open(f"iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_{g}.pkl.xz", "rb"))
rng = np.random.default_rng(3); arrays = []
for lab, d in sets.items():
    d.index = d.index.map(str)
    for gsm in rng.choice(d.columns, 12, replace=False):
        arrays.append((lab, gsm, pd.Series(C.stage_1s_scale_map(d[gsm].dropna().to_dict(), "stage1_noob_450K")[0]).reindex(markers)))
labs450 = sorted(sets)
present = np.all([a[2].notna().values for a in arrays], axis=0)
M = [m for m, p in zip(markers, present) if p]
Xm = Xm.loc[M]; A = Xm.values; Ab = Xm[bcols].values
arrays = [(lab, gsm, b.loc[M].values) for lab, gsm, b in arrays]
print(f"450K: {len(M):,} common markers | {len(arrays)} null arrays", flush=True)

# ---- EPIC-Italy at full marker resolution (PROC_MF_03_extract.py) - already on the chain's mapped scale? NO: raw series
# matrix is the authors' normalised betas; PROC-EPIC-01 used the same source through the chain's scale map for EPIC.
ep = pd.read_parquet("results/mf03/GSE51032_deconvolver_markers.parquet"); ep.index = ep.index.map(str)
sc = {r["gsm"]: r for r in json.load(open("handoff/epic01_scored.json"))}
ctrl = [g for g in ep.columns if g in sc and sc[g].get("icd") is None]
# the scale map PROC-EPIC-01 applied to this series (epic01_score.py line 114): GSE51032 is a 450K array - "EPIC-Italy"
# is the cohort's name, not the platform. A fifth LABORATORY on the same platform.
pipe = "GSE51032_450K"
print(f"EPIC-Italy: {ep.shape[0]:,} loci x {ep.shape[1]} arrays | controls {len(ctrl)} | scale map used by PROC-EPIC-01: {pipe}", flush=True)
Me = [m for m in M if m in ep.index]
cover = len(Me) / len(M)
print(f"EPIC-Italy carries {len(Me):,} of the detector's {len(M):,} markers ({cover:.1%})", flush=True)
ASSESS_EPIC = cover >= 0.90

def blood_fit(Abx, v):
    f, _ = nnls(Abx, v); return f, Abx @ f
def nnls_full(v):
    f, _ = nnls(A, v); f = f / max(f.sum(), 1e-9); return dict(zip(names, f))
def raw_amp(Xx, Abx, v, cell, w):
    fb, b = blood_fit(Abx, v); r = v - b; t = Xx[cell].values - b / max(fb.sum(), 1e-9)
    return float(np.sum(w * t * r) / np.sum(w * t * t))

R = np.array([v - blood_fit(Ab, v)[1] for _, _, v in arrays])
def weights(lab_out=None, same=False):
    idx = [i for i, (lab, _, _) in enumerate(arrays) if same or lab_out is None or lab != lab_out]
    return 1.0 / (R[idx].var(axis=0, ddof=1) + 1e-8)
W = {lab: weights(lab) for lab in labs450}; Wsame = {lab: weights(lab, same=True) for lab in labs450}; Wall = weights(None)

# ---- 450K: raw null per lab, centre, line, sigma
rawnull = {c: {lab: [raw_amp(Xm, Ab, v, c, W[lab]) for (l, _, v) in arrays if l == lab] for lab in labs450} for c in CELLS}
centre = {c: {lab: float(np.median(rawnull[c][lab])) for lab in labs450} for c in CELLS}
def q_line(vals): vals = np.sort(np.asarray(vals)); n = len(vals); return float(np.quantile(vals, 1 - 1.0 / n))
line = {c: {lab: q_line(np.array(rawnull[c][lab]) - centre[c][lab]) for lab in labs450} for c in CELLS}
def _pooled_sigma(c, lab):
    a = np.concatenate([rawnull[c][o] for o in labs450 if o != lab]); return float(1.4826 * np.median(np.abs(a - np.median(a))))
def _centred_sigma(c, lab):
    a = np.concatenate([np.array(rawnull[c][o]) - centre[c][o] for o in labs450 if o != lab]); return float(1.4826 * np.median(np.abs(a)))
# MF-02 SEALED definition (pooled): the bar. My first MF-03 draft used the per-lab-centred form, which is smaller and gave 0.135;
# both are reported, the pre-registered one decides.
sigma = {c: {lab: _pooled_sigma(c, lab) for lab in labs450} for c in CELLS}
sigma_centred = {c: {lab: _centred_sigma(c, lab) for lab in labs450} for c in CELLS}
def fhat(v, c, lab, same=False): return raw_amp(Xm, Ab, v, c, Wsame[lab] if same else W[lab]) - centre[c][lab]

null = {"nnls": {c: [] for c in CELLS}, "iv": {c: [] for c in CELLS}, "iv_z": {c: [] for c in CELLS}, "iv_z_centred": {c: [] for c in CELLS}}
for lab, gsm, v in arrays:
    fr = nnls_full(v)
    for c in CELLS:
        null["nnls"][c].append(fr.get(c, 0.0)); f = fhat(v, c, lab); null["iv"][c].append(f); null["iv_z"][c].append(f / sigma[c][lab]); null["iv_z_centred"][c].append(f / sigma_centred[c][lab])
# NNLS line: per lab too, same rule, so the comparison is like for like
nnls_line = {c: {lab: q_line([null["nnls"][c][i] for i, (l, _, _) in enumerate(arrays) if l == lab]) for lab in labs450} for c in CELLS}

spk = []
for lab, gsm, v in arrays:
    for c in CELLS:
        prof = Xm[c].values
        for f0 in FRACS:
            vs = v * (1 - f0) + prof * f0
            spk.append({"platform": "450K", "lab": lab, "gsm": gsm, "cell": c, "f": f0, "nnls": nnls_full(vs).get(c, 0.0),
                        "nnls_det": nnls_full(vs).get(c, 0.0) > nnls_line[c][lab],
                        "iv": fhat(vs, c, lab), "iv_det": fhat(vs, c, lab) > line[c][lab],
                        "iv_same_det": fhat(vs, c, lab, same=True) > line[c][lab]})
S = pd.DataFrame(spk); print(f"450K spikes {len(S)}", flush=True)

def limit(frame, col):
    for f0 in FRACS:
        sub = frame[frame.f == f0]
        if len(sub) and sub[col].mean() >= 0.90: return f0
    return None
LIM = {"nnls": {c: limit(S[S.cell == c], "nnls_det") for c in CELLS}, "iv": {c: limit(S[S.cell == c], "iv_det") for c in CELLS},
       "iv_same": {c: limit(S[S.cell == c], "iv_same_det") for c in CELLS}}
DETR = {d: {c: {str(f0): float(S[(S.cell == c) & (S.f == f0)][col].mean()) for f0 in FRACS} for c in CELLS} for d, col in (("nnls", "nnls_det"), ("iv", "iv_det"))}

# ---- EPIC-Italy on its own line
E = {}; espk = []; eline = {}; ecentre = {}; esigma = {}; b8 = {}
if ASSESS_EPIC:
    idx = [M.index(m) for m in Me]; Xe = Xm.loc[Me]; Abe = Xe[bcols].values; wE = Wall[idx]
    vals = {}
    for g in ctrl:
        vv = ep[g].reindex(Me)
        if vv.notna().sum() < 1000: continue      # 2026-09-26: a >10 % NaN filter kept 142 of 424; the per-array mask handles the rest
        v = vv.values.copy(); ok = ~np.isnan(v)
        vals[g] = (v, ok)
    # scale-map EPIC arrays as PROC-EPIC-01 did (author-normalised -> chain scale), on the present markers
    mapped = {}
    for g, (v, ok) in vals.items():
        d = {m: float(x) for m, x in zip(Me, v) if not np.isnan(x)}
        mm = C.stage_1s_scale_map(d, pipe)[0] if pipe else d
        mapped[g] = np.array([mm.get(m, np.nan) for m in Me])
    def raw_e(v, c):
        ok = ~np.isnan(v); fb, _ = nnls(Abe[ok], v[ok]); b = Abe[ok] @ fb; r = v[ok] - b
        t = Xe[c].values[ok] - b / max(fb.sum(), 1e-9); w = wE[ok]
        return float(np.sum(w * t * r) / np.sum(w * t * t))
    rawE = {c: np.array([raw_e(mapped[g], c) for g in mapped]) for c in CELLS}
    for c in CELLS:
        ecentre[c] = float(np.median(rawE[c])); fe = rawE[c] - ecentre[c]
        eline[c] = q_line(fe); esigma[c] = float(1.4826 * np.median(np.abs(fe)))
        thr450 = float(np.median([line[c][lab] for lab in labs450]))
        b8[c] = {"fp_at_450K_median_line": float((fe > thr450).mean()), "epic_line": eline[c], "median_450K_line": thr450, "n": int(len(fe))}
    E["null"] = {c: [float(x) for x in rawE[c] - ecentre[c]] for c in CELLS}
    print(f"EPIC-Italy controls used {len(mapped)} | own lines: " + ", ".join(f"{c} {eline[c]:.4f}" for c in CELLS), flush=True)
    gl = list(mapped)
    for g in gl:
        v = mapped[g]
        for c in CELLS:
            prof = Xe[c].values
            for f0 in FRACS:
                vs = v * (1 - f0) + prof * f0
                f = raw_e(vs, c) - ecentre[c]
                espk.append({"platform": "EPIC", "lab": "GSE51032", "gsm": g, "cell": c, "f": f0, "iv": f, "iv_det": f > eline[c]})
    SE = pd.DataFrame(espk)
    LIM["epic_iv"] = {c: limit(SE[SE.cell == c], "iv_det") for c in CELLS}
    DETR["epic_iv"] = {c: {str(f0): float(SE[(SE.cell == c) & (SE.f == f0)]["iv_det"].mean()) for f0 in FRACS} for c in CELLS}
    print(f"EPIC spikes {len(SE)}", flush=True)

def num(x): return x if x is not None else 1.0
print("\nDETECTION LIMIT on each laboratory's OWN line (>= 90 % detected)")
print(f"{'cell':<26}{'450K NNLS':>10}{'450K inv-var':>13}{'EPIC inv-var':>13}")
for c in CELLS:
    e = LIM.get("epic_iv", {}).get(c, "n/a")
    print(f"{c:<26}{str(LIM['nnls'][c] or '>5%'):>10}{str(LIM['iv'][c] or '>5%'):>13}{str(e if e is not None else '>5%'):>13}")

B1n = sum(num(LIM["iv"][c]) < num(LIM["nnls"][c]) for c in CELLS); B1 = B1n >= 3
z = np.concatenate([null["iv_z"][c] for c in CELLS]); tail = float((np.abs(z) > 2).mean()); B2 = 0.02 <= tail <= 0.10
tail_centred = float((np.abs(np.concatenate([null["iv_z_centred"][c] for c in CELLS])) > 2).mean())
bias450 = {str(f0): float((S[S.f == f0]["iv"] - f0).median()) for f0 in (0.02, 0.05)}
biasE = {str(f0): float((pd.DataFrame(espk)[pd.DataFrame(espk).f == f0]["iv"] - f0).median()) for f0 in (0.02, 0.05)} if espk else {}
B3 = all(abs(v) <= 0.005 for v in list(bias450.values()) + list(biasE.values()))
cn = {c: {lab: float(np.median([f for (l, _, _), f in zip(arrays, null["iv"][c]) if l == lab])) for lab in labs450} for c in CELLS}
if E: 
    for c in CELLS: cn[c]["GSE51032"] = float(np.median(E["null"][c]))
B4 = all(abs(x) <= 0.003 for c in CELLS for x in cn[c].values())
B5 = all(num(LIM["iv_same"][c]) >= 0.8 * num(LIM["iv"][c]) for c in CELLS)
d6 = []
for lab, gsm, v in arrays:
    full = nnls_full(v); fb, _ = blood_fit(Ab, v); fb = fb / max(fb.sum(), 1e-9); fbd = dict(zip(bcols, fb))
    d6.append(max(abs(full.get(k, 0) - fbd.get(k, 0)) for k in ("CD4_T-cells", "CD8_T-cells", "CD19_B-cells", "CD56_NK-cells", "CD14_monocytes")))
B6 = float(np.median(d6)) < 0.005
B7 = (all(num(LIM["epic_iv"][c]) <= 0.02 for c in CELLS)) if ASSESS_EPIC else None

print(f"\nB1 450K limit lower than NNLS on {B1n} of 4  ->  {'MET' if B1 else 'FAILED'}")
print(f"B2 null |z|>2 {tail:.3f} (MF-02 pooled sigma; per-lab-centred variant {tail_centred:.3f})  ->  {'MET' if B2 else 'FAILED'}")
print(f"B3 bias 450K {bias450} EPIC {biasE}  ->  {'MET' if B3 else 'FAILED'}")
print(f"B4 centred null, worst |median| {max(abs(x) for c in CELLS for x in cn[c].values()):.4f}  ->  {'MET' if B4 else 'FAILED'}")
print(f"B5 same-lab weights  ->  {'MET' if B5 else 'FAILED'}")
print(f"B6 blood composition median change {np.median(d6):.4f}  ->  {'MET' if B6 else 'FAILED'}")
print(f"B7 EPIC-Italy limit <= 2 % on its own line, all four cells  ->  {'MET' if B7 else ('NOT ASSESSED (marker coverage %.1f%% < 90%%)' % (100*cover) if B7 is None else 'FAILED')}")
if b8: print("B8 (recorded) 450K median line applied to EPIC -> FP:", {c: round(v["fp_at_450K_median_line"], 3) for c, v in b8.items()})

def _py(o):
    import numpy as _np
    if isinstance(o, dict): return {k: _py(v) for k, v in o.items()}
    if isinstance(o, list): return [_py(v) for v in o]
    if isinstance(o, (_np.bool_,)): return bool(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.integer,)): return int(o)
    return o
json.dump(_py({"limits": LIM, "detection_rate": DETR, "lines_450K": line, "nnls_lines_450K": nnls_line, "centre_450K": centre, "sigma_450K": sigma,
           "epic": {"line": eline, "centre": ecentre, "sigma": esigma, "n_controls": len(E["null"][CELLS[0]]) if E else 0, "marker_coverage": cover, "pipeline": pipe},
           "bars": {"B1": B1, "B1_n": B1n, "B2": B2, "B2_tail": tail, "B2_tail_centred_variant": tail_centred, "B3": B3, "B3_bias_450K": bias450, "B3_bias_EPIC": biasE, "B4": B4, "B4_centres": cn,
                    "B5": B5, "B6": B6, "B6_median": float(np.median(d6)), "B7": B7, "B8_recorded": b8},
           "spikes_450K": spk, "spikes_EPIC": espk}), open("handoff/mf03_results.json", "w"), indent=1)
json.dump({"450K": {d: {c: [float(x) for x in null[d][c]] for c in CELLS} for d in null}, "EPIC": E.get("null", {})}, open("handoff/mf03_null.json", "w"), indent=1)
print("\nwrote handoff/mf03_results.json, handoff/mf03_null.json")
