#!/usr/bin/env python3
# INSTRUMENT-TEST: measures two CANDIDATE changes to the per-cell scorer - (1) identity loci re-zeroed so a cell's own
# atlas profile reads A = 1.000, (2) the dilution-line inversion per present cell - against constructed truth and 48
# healthy arrays. Neither change is in the chain; run_sample cannot exercise them. Produces bar results only, no
# tier, no report for any specimen.
"""PROC-UNMIX-01, steps as fixed in doors/PROC_UNMIX_01_PREREG.md before this file was written.

  (1) RE-ZERO   for each cell, start from its v1_0 identity loci (|mu - H_min_beta| <= 0.05) and trim from the heavier
                side until mean(mu over the loci) is within 1e-4 of H_min_beta, so H(mean)/H_min = 1.000 by construction.
                Written to a SCRATCH file (results/unmix01/iamatlas_percell_identity_loci_v1_1_CANDIDATE.json); the
                chain's file is not touched here.
  (2) UNMIX     own_beta_i = (specimen_beta_i - sum_{c != cell, present} f_c mu_{c,i}) / f_cell, clipped to [0,1],
                over the cell's identity loci; A = H(mean own_beta) / H_min[class]. Linear mixture inversion, the
                form in FRACTION_AND_A.md; no other form is tried.
Bars B1-B6 as pre-registered. B6 clarified before running (dated in the pre-registration): the inversion at f = 1 must
be the identity (< 1e-6 against no-unmix on the SAME loci); the re-zero moves the f = 1 reading to 1.000 on purpose.
"""
import json, math, os, sys
import numpy as np, pandas as pd

W = os.getcwd(); CH = os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/chain")
sys.path.insert(0, CH + "/Synthetic_Patient_Generator"); sys.path.insert(0, CH)
import synthetic_patient_generator as SPG, cpg_conductor as C  # noqa: E402
OUT = os.path.join(W, "results/unmix01"); os.makedirs(OUT, exist_ok=True)
ATLAS = os.path.join(W, "atlas_work/IAMAtlasREBUILD.csv")
BLOOD = ["Neutrophils_reinius", "CD4_T-cells", "CD8_T-cells", "CD56_NK-cells", "CD14_monocytes"]
MAJORITY = ["Neutrophils_reinius", "CD14_monocytes"]
MIXES = {"typical": {"Neutrophils_reinius": 0.58, "CD4_T-cells": 0.13, "CD8_T-cells": 0.07, "CD56_NK-cells": 0.06, "CD14_monocytes": 0.08, "CD19_B-cells": 0.05, "Eosinophils_reinius": 0.03},
         "neut_heavy": {"Neutrophils_reinius": 0.76, "CD4_T-cells": 0.09, "CD8_T-cells": 0.03, "CD56_NK-cells": 0.05, "CD14_monocytes": 0.04, "CD19_B-cells": 0.02, "Eosinophils_reinius": 0.01},
         "lymph_heavy": {"Neutrophils_reinius": 0.40, "CD4_T-cells": 0.22, "CD8_T-cells": 0.14, "CD56_NK-cells": 0.10, "CD14_monocytes": 0.09, "CD19_B-cells": 0.05}}


def H(b):
    b = min(max(float(b), 1e-12), 1 - 1e-12); return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def load():
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    hmin = {k: float(v["H_min"]) for k, v in ident.items() if isinstance(v, dict) and "H_min" in v}
    hbeta = {k: float(v["H_min_beta"]) for k, v in ident.items() if isinstance(v, dict) and "H_min_beta" in v}
    pci = json.load(open(C._find("iamatlas_percell_identity_loci_v1_1.json")))
    cells = pci.get("cells") or pci.get("entries") or {k: v for k, v in pci.items() if not k.startswith("_")}
    mu = SPG._cell_means(ATLAS); mu.index = mu.index.map(str)
    return hmin, hbeta, cells, mu


def rezero(cells, mu, hbeta):
    """Trim each cell's identity loci from the heavier side until the cell's own mean sits at H_min_beta."""
    out = {}; rows = []
    for cell, e in cells.items():
        loci = [str(x) for x in e["loci"]]; cl = e["class"]; tgt = hbeta[cl]
        if cell not in mu.columns: continue
        m = mu[cell].reindex(loci).dropna(); m = m.sort_values()
        vals = m.values.astype(float); idx = list(m.index)
        lo, hi = 0, len(vals)
        while hi - lo > 100 and abs(vals[lo:hi].mean() - tgt) > 1e-4:
            if vals[lo:hi].mean() > tgt: hi -= 1
            else: lo += 1
        kept = idx[lo:hi]; mean_after = float(vals[lo:hi].mean())
        out[cell] = {"class": cl, "H_min": e["H_min"], "H_min_beta": tgt, "n_loci": len(kept), "loci": kept,
                     "own_mean_beta": mean_after, "own_A": H(mean_after) / e["H_min"], "n_loci_v1_0": len(loci)}
        rows.append((cell, len(loci), len(kept), H(float(np.mean(mu[cell].reindex(loci).dropna()))) / e["H_min"], out[cell]["own_A"]))
    return out, rows


def score(beta, cell, loci, hmin_cell, frac, others, mu, unmix):
    """A on identity loci; with unmix, remove the other present cells' expectation first."""
    idx = [l for l in loci if l in beta.index]
    if len(idx) < 100: return float("nan")
    b = beta.loc[idx].astype(float).values
    if unmix:
        if frac is None or frac <= 0: return float("nan")
        bg = np.zeros(len(idx))
        for c, f in others.items():
            if c == cell or f <= 0 or c not in mu.columns: continue
            bg += f * mu[c].reindex(idx).fillna(0.5).values
        b = np.clip((b - bg) / frac, 0.0, 1.0)
    return H(float(np.mean(b))) / hmin_cell


def deconv(dec, beta):
    r = dec.deconvolve(beta.to_dict()); return dict(r.celltype_fractions)


def main():
    hmin, hbeta, cells_v10, mu = load()
    v11, rows = rezero(cells_v10, mu, hbeta)
    json.dump({"_meta": {"built": "2026-09-27", "from": "iamatlas_percell_identity_loci_v1_1.json", "rule": "trim from the heavier side until own mean beta is within 1e-4 of the class H_min_beta (own A = 1.000)", "status": "CANDIDATE - PROC-UNMIX-01"}, "cells": v11},
              open(os.path.join(OUT, "iamatlas_percell_identity_loci_v1_1_CANDIDATE.json"), "w"))
    own = np.array([r[4] for r in rows]); before = np.array([r[3] for r in rows])
    print(f"(1) RE-ZERO: {len(rows)} cells | own A before: {before.min():.4f}-{before.max():.4f} median {np.median(before):.4f} | after: {own.min():.4f}-{own.max():.4f} | loci kept median {np.median([r[2] for r in rows]):.0f} of {np.median([r[1] for r in rows]):.0f}", flush=True)
    dec_mod = C._load_module("legacy_iam_deconvolver", C._find("legacy_iam_deconvolver.py"))
    dec = dec_mod.legacyIAMDeconvolver(ATLAS, celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
    res = {"rezero": {r[0]: {"own_A_v1_0": r[3], "own_A_v1_1": r[4], "n_v1_0": r[1], "n_v1_1": r[2]} for r in rows}}

    def read(beta, tag):
        fr = deconv(dec, beta); present = {c: f for c, f in fr.items() if f >= 0.02}
        out = {}
        for cell in present:
            if cell not in v11: continue
            e = v11[cell]; e0 = cells_v10[cell]
            out[cell] = {"f": present[cell],
                         "A_v1_0": score(beta, cell, [str(x) for x in e0["loci"]], e0["H_min"], None, None, mu, False),
                         "A_v1_1": score(beta, cell, e["loci"], e["H_min"], None, None, mu, False),
                         "A_unmix": score(beta, cell, e["loci"], e["H_min"], present[cell], present, mu, True)}
        return out

    # ---- constructed mixes (B1-B3)
    mixes = {}
    for name, mix in MIXES.items():
        beta, _ = SPG.compose_cells(mix, noise_sigma=0.044, seed=11, atlas_csv=ATLAS); beta.index = beta.index.map(str)
        mixes[name] = read(beta, name)
        print(f"\n{name}:"); [print(f"   {c:<22} f {v['f']:.3f}  v1_0 {v['A_v1_0']:.4f}  v1_1 {v['A_v1_1']:.4f}  unmix {v['A_unmix']:.4f}") for c, v in mixes[name].items()]
    res["mixes"] = mixes
    allA = [v["A_unmix"] for m in mixes.values() for v in m.values() if not math.isnan(v["A_unmix"])]
    b1 = all(0.95 <= a < 1.04 for a in allA); b2 = max(abs(a - 1.0) for a in allA)
    maj = [abs(v["A_unmix"] - v["A_v1_0"]) for m in mixes.values() for c, v in m.items() if c in MAJORITY and not math.isnan(v["A_unmix"])]
    b3 = max(maj) if maj else float("nan")
    print(f"\nB1 every present cell NORMAL after (1)+(2): {'MET' if b1 else 'FAILED'}  ({min(allA):.4f}-{max(allA):.4f})")
    print(f"B2 max |A - 1.00| = {b2:.4f}  ->  {'MET' if b2 <= 0.015 else 'FAILED (bar 0.015)'}")
    print(f"B3 majority cells moved at most {b3:.4f}  ->  {'MET' if b3 < 0.005 else 'FAILED (bar 0.005)'}   [note: B3 compares unmix on v1_1 to v1_0 - includes the re-zero shift]")
    # ---- B4 planted departure: displace CD4's profile at its identity loci so its own A = 1.06
    cell = "CD4_T-cells"; e = v11[cell]; loci = [l for l in e["loci"] if l in mu.index]
    prof = mu[cell].reindex(loci).astype(float).values
    lo_t, hi_t = 0.0, 1.0
    for _ in range(60):
        t = (lo_t + hi_t) / 2; a = H(float(np.mean(prof + t * (0.5 - prof)))) / e["H_min"]
        lo_t, hi_t = (t, hi_t) if a < 1.06 else (lo_t, t)
    mix = dict(MIXES["typical"]); beta, _ = SPG.compose_cells(mix, noise_sigma=0.044, seed=12, atlas_csv=ATLAS); beta.index = beta.index.map(str)
    # replace the CD4 contribution at its identity loci by the displaced profile
    beta.loc[loci] = beta.loc[loci].values + mix[cell] * (t * (0.5 - prof))
    planted = read(beta, "planted"); res["planted"] = {"t": t, "cells": planted}
    a4 = planted.get(cell, {}).get("A_unmix", float("nan")); others_ok = all(0.95 <= v["A_unmix"] < 1.04 for c, v in planted.items() if c != cell and not math.isnan(v["A_unmix"]))
    print(f"\nB4 planted CD4 own A = 1.06 -> read {a4:.4f} unmixed (v1_1 no unmix {planted.get(cell,{}).get('A_v1_1',float('nan')):.4f}); others NORMAL: {others_ok}  ->  {'MET' if abs(a4-1.06)<=0.015 and others_ok else 'FAILED'}")
    # ---- B5 48 healthy arrays, mapped
    pub = json.load(open(os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/kit/results/PROC_BAND_01_arrays.json")))["arrays"]
    import glob, lzma, pickle
    spread = {c: {"before": [], "after": []} for c in BLOOD}; n = 0
    for p in sorted(glob.glob(os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_*.pkl.xz"))):
        df = pickle.load(lzma.open(p, "rb")); df.index = df.index.map(str)
        for gsm in list(df.columns)[:12]:
            raw = df[gsm].dropna(); mapped, _ = C.stage_1s_scale_map(raw.to_dict(), "stage1_noob_450K")
            beta = pd.Series(mapped); beta.index = beta.index.map(str)
            r = read(beta, gsm); n += 1
            for c in BLOOD:
                if c in r and not math.isnan(r[c]["A_unmix"]): spread[c]["before"].append(r[c]["A_v1_0"]); spread[c]["after"].append(r[c]["A_unmix"])
    b5 = {}
    for c in BLOOD:
        b, a = np.array(spread[c]["before"]), np.array(spread[c]["after"])
        if len(a) < 10: b5[c] = None; continue
        b5[c] = {"n": len(a), "p90p10_before": float(np.percentile(b, 90) - np.percentile(b, 10)), "p90p10_after": float(np.percentile(a, 90) - np.percentile(a, 10)), "median_after": float(np.median(a))}
        print(f"B5 {c:<22} n {len(a):3d}  p90-p10 before {b5[c]['p90p10_before']:.4f}  after {b5[c]['p90p10_after']:.4f}  median after {b5[c]['median_after']:.4f}")
    ok5 = sum(1 for c in BLOOD if b5.get(c) and b5[c]["p90p10_after"] <= b5[c]["p90p10_before"])
    print(f"B5 spread not wider on {ok5} of 5 blood cells ({n} arrays)  ->  {'MET' if ok5 >= 4 else 'FAILED (bar 4 of 5)'}")
    res["b5"] = b5
    # ---- B6 pure cell: inversion is the identity at f = 1 (same loci, unmix vs none)
    worst = 0.0
    for cell in ["CD4_T-cells", "Neutrophils_reinius", "Breast", "Colon_epithelial_cells", "Cortical_neurons"]:
        if cell not in v11 or cell not in mu.columns: continue
        beta = mu[cell].dropna(); e = v11[cell]
        a0 = score(beta, cell, e["loci"], e["H_min"], None, None, mu, False); a1 = score(beta, cell, e["loci"], e["H_min"], 1.0, {cell: 1.0}, mu, True)
        worst = max(worst, abs(a1 - a0))
    print(f"B6 f = 1: |A_unmix - A| max {worst:.2e}  ->  {'MET' if worst < 1e-6 else 'FAILED'}   (re-zero moves the f=1 reading to 1.000 by design: {min(own):.4f}-{max(own):.4f})")
    res["bars"] = {"B1": b1, "B2": b2, "B3": b3, "B4": {"read": a4, "others_normal": others_ok}, "B5_ok": ok5, "B6": worst}
    json.dump(res, open(os.path.join(W, "handoff/unmix01_results.json"), "w"), indent=1, default=float)
    print("\nwrote handoff/unmix01_results.json")


if __name__ == "__main__":
    main()
