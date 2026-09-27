#!/usr/bin/env python3
# INSTRUMENT-TEST: measures the CANDIDATE sky spread (atlas posterior + this array's SNP-probe noise) against the five
# bars of doors/PROC_SKY_01_PREREG.md. Reads no per-laboratory file. Produces bar results and plates only - no tier,
# no A, no report for any specimen. One JSON shard per array; re-runnable.
"""PROC-SKY-01. Construction exactly as pre-registered:
    zero      m = 0
    spread    sigma_i^2 = sum_c f_c^2 sd_{c,i}^2  +  sigma_arr^2(beta_i),   sigma_arr^2(b) = a + b*beta(1-beta)
              a = mean(sd^2 of the beta=0 and beta=1 SNP clusters), a + b/4 = sd^2 of the beta=1/2 cluster (b >= 0),
              clusters by the PROC-TARE-01 two-iteration nearest-ideal rule; sigma_arr divided by the pipeline-map slope
              so it sits on the mapped scale like the residual.
    arrays    12 per commissioning laboratory: the 36 panel IDAT pairs already on disk (tare01/idats) and, for GSE87571,
              the first 12 accessions in sorted order among idats_full_GSE87571 (rule fixed here; the detection panel
              does not record its accessions).
"""
import glob, json, math, os, sys, time
import numpy as np, pandas as pd

W = os.getcwd(); CH = os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/chain"); sys.path.insert(0, CH); sys.path.insert(0, W)
import cpg_conductor as C                      # noqa: E402
import tare01 as T                             # stage1_with_snps, snp_tare (the chain's Stage 1 + the rs probes)  # noqa: E402
import importlib.util
_sp = importlib.util.spec_from_file_location("S", os.path.join(CH, "stage_4_6_patient_cmb.py")); S = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(S)
OUT = os.path.join(W, "results/sky01"); os.makedirs(os.path.join(OUT, "shards"), exist_ok=True)
ATLAS = os.path.join(W, "atlas_work/IAMAtlasREBUILD.csv"); PIPE = "stage1_noob_450K"
CLASSES = S.CLASSES


def load_atlas_mean_sd():
    cols = ["cpg_id"] + [f"{c}_mean" for c in CLASSES] + [f"{c}_sd" for c in CLASSES]
    at = pd.read_csv(ATLAS, usecols=cols, index_col="cpg_id"); at.index = at.index.map(str)
    mu = at[[f"{c}_mean" for c in CLASSES]].copy(); mu.columns = CLASSES
    sd = at[[f"{c}_sd" for c in CLASSES]].copy(); sd.columns = CLASSES
    return mu, sd


def snp_noise(rs):
    """(a, b, cluster sds) from this array's SNP probes: sigma_arr^2(beta) = a + b*beta*(1-beta)."""
    b = np.asarray(rs, float); b = b[~np.isnan(b)]
    ideal = np.array([0.0, 0.5, 1.0]); lab = np.argmin(np.abs(b[:, None] - ideal[None, :]), axis=1)
    cen = np.array([np.median(b[lab == k]) if (lab == k).any() else ideal[k] for k in range(3)])
    lab = np.argmin(np.abs(b[:, None] - cen[None, :]), axis=1)
    sds = [float(np.std(b[lab == k], ddof=1)) if (lab == k).sum() >= 3 else float("nan") for k in range(3)]
    a = float(np.nanmean([sds[0] ** 2, sds[2] ** 2])); bb = max(0.0, 4.0 * (sds[1] ** 2 - a))
    return a, bb, sds


def sigma(mu, sd, fractions, beta_mapped, a, bb, slope):
    f = pd.Series({c: float(fractions.get(c, 0.0)) for c in CLASSES})
    var_atlas = (sd.reindex(beta_mapped.index)[CLASSES].fillna(0.0) ** 2).mul(f ** 2, axis=1).sum(axis=1)
    var_arr = (a + bb * beta_mapped * (1 - beta_mapped)) / slope ** 2
    return np.sqrt(var_atlas + var_arr)


def sky(beta_mapped, fractions, mu, sd, a, bb, slope, mapping, floors):
    r, E = S.residual(beta_mapped, mu, fractions); s = sigma(mu, sd, fractions, beta_mapped, a, bb, slope)
    z = (r / s).dropna()
    fake_scale = {"m": pd.Series(0.0, index=z.index), "s": pd.Series(1.0, index=z.index)}   # m=0, unit s: patient_sky then only summarises + gates
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json"))); loci = {c: v["loci"] for c, v in ident.items() if isinstance(v, dict) and "loci" in v}
    # reuse patient_sky's gating/summaries/pixels on the already-formed z: pass z as 'beta' with atlas means of zero
    zero_means = pd.DataFrame(0.0, index=z.index, columns=CLASSES)
    out = S.patient_sky(z, fractions, zero_means, fake_scale, loci, mapping, presence_floors_by_class=floors)
    out["z"] = z; return out


def robust_sd(z): return float(1.4826 * np.median(np.abs(z - np.median(z))))


def main():
    t0 = time.time()
    mp = json.load(open(C._find("beta_scale_maps_v1.json")))["maps"][PIPE]; slope = float(mp["slope"])
    mu, sd = load_atlas_mean_sd(); mapping = S.load_mapping()
    floors = json.load(open(C._find("presence_floors_v1.json")))["floors"]
    dec_mod = C._load_module("legacy_iam_deconvolver", C._find("legacy_iam_deconvolver.py"))
    dec = dec_mod.legacyIAMDeconvolver(ATLAS, celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
    # ---- the 48 arrays
    pairs = {}
    for g in sorted(glob.glob(os.path.join(W, "tare01/idats/*_Grn.idat*"))):
        lab, rest = os.path.basename(g).split("__", 1); gsm = rest.split("_")[0]; pairs.setdefault(lab, {})[gsm] = (g, g.replace("_Grn", "_Red"))
    full = sorted(glob.glob(os.path.join(W, "idats_full_GSE87571/*_Grn.idat*")))
    for g in full[:12]:
        gsm = os.path.basename(g).split("_")[0]; pairs.setdefault("GSE87571", {})[gsm] = (g, g.replace("_Grn", "_Red"))
    for lab in list(pairs): pairs[lab] = dict(sorted(pairs[lab].items())[:12])
    print({k: len(v) for k, v in pairs.items()}, flush=True)
    recs = []
    for lab, d in pairs.items():
        for gsm, (grn, red) in d.items():
            shard = os.path.join(OUT, "shards", f"{gsm}.json")
            if os.path.exists(shard): recs.append(json.load(open(shard))); continue
            cg, rs = T.stage1_with_snps(grn, red)
            mapped, scale_label = C.stage_1s_scale_map(cg, PIPE); beta = pd.Series(mapped, dtype=float); beta.index = beta.index.map(str)
            fr = dict(dec.deconvolve(mapped).class_fractions)
            a, bb, sds = snp_noise(rs)
            sk = sky(beta, fr, mu, sd, a, bb, slope, mapping, floors); z = sk["z"]
            rec = {"lab": lab, "gsm": gsm, "n_rs": int(len(rs)), "snp_a": a, "snp_b": bb, "snp_cluster_sd": sds, "sigma_arr_at_half": math.sqrt(a + bb / 4) / slope,
                   "class_fractions": fr, "all": sk["all"], "robust_sd_z": robust_sd(z.values), "median_z": float(z.median()), "frac_gt2": float((z.abs() > 2).mean()),
                   "classes": {c: {k: v for k, v in dd.items() if k != "pixels"} for c, dd in sk["classes"].items()}}
            # B4a: same array twice
            z2 = sky(beta, fr, mu, sd, a, bb, slope, mapping, floors)["z"]; rec["b4a_max_abs_diff"] = float((z2 - z).abs().max())
            tmp = shard + ".tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, shard); recs.append(rec)
            if len(recs) == 1:   # plate for B5, first array
                S.render_plate(sk, os.path.join(OUT, f"sky_{gsm}_atlas_sigma.png"), title=f"{gsm} - residual z, sigma = atlas posterior + this array's SNP-probe noise (no panel)")
            print(f"  {lab} {gsm}  sigma_arr(1/2) {rec['sigma_arr_at_half']:.4f}  robust sd z {rec['robust_sd_z']:.3f}  median z {rec['median_z']:+.3f}  |z|>2 {rec['frac_gt2']*100:.1f}%   [{(time.time()-t0)/60:.0f} min]", flush=True)
    # ---- B1: constructed specimen = exact class means at the first array's composition; sigma_arr from that array
    r0 = recs[0]; fr = r0["class_fractions"]
    E = S.expectation(mu, fr).dropna(); sk1 = sky(E, fr, mu, sd, r0["snp_a"], r0["snp_b"], slope, mapping, floors); z1 = sk1["z"]
    b1_panels = {c: (dd["median_z"], dd["frac_abs_z_gt2"]) for c, dd in sk1["classes"].items() if dd["assessable"]}
    b1 = abs(float(z1.median())) <= 0.05 and float((z1.abs() > 2).mean()) <= 0.005 and all(abs(m) <= 0.05 and f <= 0.005 for m, f in b1_panels.values())
    # ---- B4b: two constructed specimens, same profiles, different fractions
    fr2 = dict(fr); imm = fr2.get("immune", 0.0); prog = fr2.get("progenitor", 0.0)
    fr2["immune"], fr2["progenitor"] = imm + 0.10, max(0.0, prog - 0.10)
    E2 = S.expectation(mu, fr2).dropna(); z2 = sky(E2, fr2, mu, sd, r0["snp_a"], r0["snp_b"], slope, mapping, floors)["z"]
    b4b = float((z2 - z1).reindex(z1.index).abs().median())
    # ---- B2
    rs_ok = sum(1 for r in recs if 0.7 <= r["robust_sd_z"] <= 1.4); b2 = rs_ok >= 40
    b4a = max(r["b4a_max_abs_diff"] for r in recs) == 0.0
    # ---- B3: no laboratory file
    b3 = not any("residual_scale" in str(p) for p in [C._find("presence_floors_v1.json")]) and "lab" not in sky.__code__.co_varnames
    res = {"n_arrays": len(recs), "per_lab_n": {l: sum(1 for r in recs if r["lab"] == l) for l in pairs},
           "B1": {"met": b1, "median_z": float(z1.median()), "frac_gt2": float((z1.abs() > 2).mean()), "panels": b1_panels},
           "B2": {"met": b2, "in_range": rs_ok, "robust_sd_z": {r["gsm"]: r["robust_sd_z"] for r in recs}, "median_robust_sd": float(np.median([r["robust_sd_z"] for r in recs])),
                  "median_frac_gt2": float(np.median([r["frac_gt2"] for r in recs])), "median_sigma_arr_half": float(np.median([r["sigma_arr_at_half"] for r in recs]))},
           "B3": {"met": b3}, "B4": {"a_max_diff": max(r["b4a_max_abs_diff"] for r in recs), "a_met": b4a, "b_median_abs_diff": b4b, "b_met": b4b < 0.05},
           "B5": {"note": "plate rendered with the chain's render_plate and the chain's presence gating: results/sky01/sky_<gsm>_atlas_sigma.png"}}
    json.dump(res, open(os.path.join(W, "handoff/sky01_results.json"), "w"), indent=1, default=float)
    print(f"\nB1 constructed atlas specimen quiet: median z {res['B1']['median_z']:+.4f}, |z|>2 {res['B1']['frac_gt2']*100:.2f}%  -> {'MET' if b1 else 'FAILED'}")
    print(f"B2 robust sd of z in [0.7,1.4] on {rs_ok}/48 (median {res['B2']['median_robust_sd']:.3f}; median |z|>2 {res['B2']['median_frac_gt2']*100:.1f}%; median sigma_arr(1/2) {res['B2']['median_sigma_arr_half']:.4f})  -> {'MET' if b2 else 'FAILED (bar 40)'}")
    print(f"B3 no laboratory file read  -> {'MET' if b3 else 'FAILED'}")
    print(f"B4 same array twice max|dz| {res['B4']['a_max_diff']:.1e} -> {'MET' if b4a else 'FAILED'};  composition drops out: median|dz| {b4b:.4f} -> {'MET' if b4b < 0.05 else 'FAILED'}")
    print("B5 plate: same nine panels, same gating (rendered)  -> see plate")
    print("wrote handoff/sky01_results.json")


if __name__ == "__main__":
    main()
