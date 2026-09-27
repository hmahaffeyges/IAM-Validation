#!/usr/bin/env python3
# INSTRUMENT-TEST: measures a CANDIDATE per-array tare (the array's own SNP probes) against the chain's laboratory
# zero on the commissioned panels. Cannot go through run_sample because the tare is not in the chain. Produces
# per-array tare parameters and bars only - no A-score, tier or report for any specimen.
"""PROC-TARE-01: every construction is the one fixed in doors/PROC_TARE_01_PREREG.md before this file was written.

The array reads 65 SNP probes (450K) whose true beta is 0, 0.5 or 1 by genotype. On each array:
    cluster   nearest ideal after one pass, re-assigned once (two iterations, fixed)
    T_offset  median over the 0.5 cluster of (beta - 0.5)
    T_scale   median(1 cluster) - median(0 cluster)
    tare      beta' = 0.5 + (beta - 0.5 - T_offset) / T_scale   (linear: centre, then gain)
A on the identity surface (H(mean beta over the class's identity loci) / H_min, immune class) is computed per array
with and without the tare, mapped through the commissioned pipeline map, with NO laboratory zero. Age term c(age)
from the commissioned decade curve where an age is published.

Panels: the 48-array null (12 per lab: GSE87571, GSE42861, GSE111629, GSE125105) whose raw IDATs are on disk, plus
every GSE87571 array with raw IDATs (for B5 age/sex and B6 chip term).
"""
import glob
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
import cpg_conductor as C  # noqa: E402

OUT = "results/tare01"; os.makedirs(OUT, exist_ok=True)
ZLAB = {"GSE87571": -0.0117, "GSE42861": 0.0084, "GSE111629": -0.0673, "GSE125105": -0.0346}
NORMAL = (0.95, 1.04)


def snp_tare(rs_beta):
    """Two-iteration nearest-ideal clustering; returns (T_offset, T_scale, n0, n05, n1)."""
    b = np.asarray(rs_beta, float); b = b[~np.isnan(b)]
    ideal = np.array([0.0, 0.5, 1.0]); lab = np.argmin(np.abs(b[:, None] - ideal[None, :]), axis=1)
    for _ in range(1):  # one re-assignment, fixed
        cen = np.array([np.median(b[lab == k]) if (lab == k).any() else ideal[k] for k in range(3)])
        lab = np.argmin(np.abs(b[:, None] - cen[None, :]), axis=1)
    m0, m05, m1 = [float(np.median(b[lab == k])) if (lab == k).any() else float("nan") for k in range(3)]
    return m05 - 0.5, m1 - m0, int((lab == 0).sum()), int((lab == 1).sum()), int((lab == 2).sum())


def apply_tare(beta, t_off, t_scale):
    return {k: min(max(0.5 + (v - 0.5 - t_off) / t_scale, 0.0), 1.0) for k, v in beta.items()}


def stage1_with_snps(grn, red, array_type="450k"):
    """The chain's Stage 1 (noob), returning betas AND the rs probes methylprep reports beside them."""
    import methylprep
    wd = tempfile.mkdtemp(prefix="tare_"); bc = "arr"; pos = "R01C01"
    try:
        for src, dst in ((grn, f"{bc}_{pos}_Grn.idat"), (red, f"{bc}_{pos}_Red.idat")):
            with (gzip.open(src, "rb") if src.endswith(".gz") else open(src, "rb")) as fi, open(os.path.join(wd, dst), "wb") as fo:
                shutil.copyfileobj(fi, fo)
        with open(os.path.join(wd, "samplesheet.csv"), "w") as fh:
            fh.write("Sample_Name,Sentrix_ID,Sentrix_Position\n" + f"{bc},{bc},{pos}\n")
        methylprep.run_pipeline(wd, array_type=array_type, betas=True, export=True, sample_sheet_filepath=os.path.join(wd, "samplesheet.csv"), poobah=False)
        p = glob.glob(os.path.join(wd, "**", "*_processed.csv"), recursive=True)[0]
        df = pd.read_csv(p, index_col=0, usecols=["IlmnID", "beta_value"] if "IlmnID" in open(p).readline() else None)
        col = "beta_value" if "beta_value" in df.columns else df.columns[-2]
        rs = df.loc[[i for i in df.index if str(i).startswith("rs")], col].astype(float)
        cg = df.loc[[i for i in df.index if str(i).startswith("cg")], col].astype(float).dropna()
        return cg.to_dict(), rs
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def identity_A(beta_mapped, loci, hmin):
    v = [beta_mapped[x] for x in loci if x in beta_mapped]
    if len(v) < 100: return None
    b = min(max(float(np.mean(v)), 1e-12), 1 - 1e-12)
    return (-b * math.log2(b) - (1 - b) * math.log2(1 - b)) / hmin


def main():
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))["immune"]
    loci = set(map(str, ident["band"] if isinstance(ident["band"], list) else ident.get("loci", [])))
    if not loci:   # the loci list lives under whichever key is the list of cg ids
        loci = set(map(str, next(v for v in ident.values() if isinstance(v, list) and v and str(v[0]).startswith("cg"))))
    hmin = float(ident["H_min"])
    print(f"immune identity loci: {len(loci):,} | H_min {hmin:.4f}", flush=True)
    curve = {int(k): v for k, v in json.load(open(C._find("reference_age_curve_v1.json")))["curve"].items()}
    ages = {r["gsm"]: r for r in json.load(open("iamrepo/Biological_Physics/MethylPhys/kit/results/PROC_BAND_01_arrays.json"))["arrays"]}
    # arrays with raw IDATs on disk, by laboratory
    pairs = {}
    if not os.environ.get("TARE_PANEL_ONLY"):
        for g in glob.glob("idats_full_GSE87571/*_Grn.idat*"):
            gsm = os.path.basename(g).split("_")[0]; pairs.setdefault("GSE87571", {})[gsm] = (g, g.replace("_Grn", "_Red"))
    for g in glob.glob("tare01/idats/*_Grn.idat*"):
        lab, rest = os.path.basename(g).split("__", 1); gsm = rest.split("_")[0]
        pairs.setdefault(lab, {})[gsm] = (g, g.replace("_Grn", "_Red"))
    print({k: len(v) for k, v in pairs.items()}, flush=True)
    rows = []
    for lab, d in pairs.items():
        pipe = "stage1_noob_450K"
        for i, (gsm, (g, r)) in enumerate(sorted(d.items())):
            if not os.path.exists(r): continue
            try:
                beta, rs = stage1_with_snps(g, r)
            except Exception as e:
                print("  ", gsm, type(e).__name__, str(e)[:80], flush=True); continue
            t_off, t_scale, n0, n05, n1 = snp_tare(rs.values)
            m_raw, _ = C.stage_1s_scale_map(beta, pipe); m_tar, _ = C.stage_1s_scale_map(apply_tare(beta, t_off, t_scale), pipe)
            a_raw = identity_A(m_raw, loci, hmin); a_tar = identity_A(m_tar, loci, hmin)
            age = (ages.get(gsm) or {}).get("age"); sex = (ages.get(gsm) or {}).get("sex")
            c_age = curve[min(curve, key=lambda k: abs(k - (age // 10 * 10)))] if age is not None else None
            chip = re.search(r"(\d{10,})_R\d\dC\d\d", g); chip = chip.group(1) if chip else None
            rows.append({"lab": lab, "gsm": gsm, "chip": chip, "age": age, "sex": sex, "T_offset": t_off, "T_scale": t_scale, "n_rs": n0 + n05 + n1,
                         "A_raw": a_raw, "A_tare": a_tar, "c_age": c_age})
            if i % 25 == 0 or os.environ.get("TARE_PANEL_ONLY"): print(f"  {lab} {i}/{len(d)}  T_off {t_off:+.4f} T_scale {t_scale:.4f}  A raw {a_raw} tare {a_tar}", flush=True)
    df = pd.DataFrame(rows)
    if os.environ.get("TARE_PANEL_ONLY"):
        # panel arm re-run 2026-09-27: the first run's panel rows came from a pre-fetch pair list (1+12+0); merge with the 732 GSE87571 rows
        prev = pd.read_parquet(f"{OUT}/PROC_TARE_01_per_array.parquet"); prev = prev[prev.lab == "GSE87571"]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(f"{OUT}/PROC_TARE_01_per_array.parquet"); print("arrays:", len(df), flush=True)
    res = {"n": len(df), "labs": {}}
    for lab, g in df.groupby("lab"):
        g = g.dropna(subset=["A_raw", "A_tare"]); ga = g.dropna(subset=["c_age"])
        z_meas = float(np.median(ga["A_raw"] - ga["c_age"]) - 1.0) if len(ga) else None
        res["labs"][lab] = {"n": len(g), "median_A_raw": float(g["A_raw"].median()), "median_A_tare": float(g["A_tare"].median()),
                            "sd_raw": float(g["A_raw"].std(ddof=1)), "sd_tare": float(g["A_tare"].std(ddof=1)),
                            "z_lab_commissioned": ZLAB[lab], "z_lab_measured_here": z_meas,
                            "median_T_offset": float(g["T_offset"].median()), "median_T_scale": float(g["T_scale"].median())}
    L = res["labs"]
    b1 = all(L[l]["z_lab_measured_here"] is not None and abs(L[l]["z_lab_measured_here"] - ZLAB[l]) <= 0.005 for l in L)
    off_raw = {l: L[l]["median_A_raw"] - 1.0 for l in L}; off_tar = {l: L[l]["median_A_tare"] - 1.0 for l in L}
    toward = all(abs(off_tar[l]) <= abs(off_raw[l]) for l in L)
    worst = max(off_raw, key=lambda l: abs(off_raw[l])); shrink = 1 - abs(off_tar[worst]) / max(abs(off_raw[worst]), 1e-9)
    b2 = toward and shrink >= 0.5
    b3 = all(NORMAL[0] <= L[l]["median_A_tare"] <= NORMAL[1] for l in L)
    b4 = sum(L[l]["sd_tare"] <= L[l]["sd_raw"] for l in L) >= 3
    u = df[(df.lab == "GSE87571")].dropna(subset=["age"])
    r_age = float(np.corrcoef(u["age"], u["T_offset"])[0, 1]) if len(u) > 10 else None
    r_sex = float(np.corrcoef((u["sex"] == "M").astype(float), u["T_offset"])[0, 1]) if len(u) > 10 and u["sex"].notna().any() else None
    b5 = (r_age is not None and abs(r_age) < 0.15) and (r_sex is None or abs(r_sex) < 0.15)
    ch = df[df.lab == "GSE87571"].dropna(subset=["chip", "A_raw", "A_tare"]).groupby("chip").agg(n=("gsm", "size"), raw=("A_raw", "median"), tar=("A_tare", "median"))
    ch = ch[ch.n >= 3]; chip_raw = float(ch["raw"].std(ddof=1)) if len(ch) > 3 else None; chip_tar = float(ch["tar"].std(ddof=1)) if len(ch) > 3 else None
    b6 = (chip_raw is not None) and (chip_tar <= 0.8 * chip_raw)
    res.update({"bars": {"B1": b1, "B2": b2, "B2_worst_lab": worst, "B2_shrink": shrink, "B3": b3, "B4": b4, "B5": b5, "B5_r_age": r_age, "B5_r_sex": r_sex,
                         "B6": b6, "B6_chip_sd_raw": chip_raw, "B6_chip_sd_tare": chip_tar, "B6_n_chips": int(len(ch)), "B7": None}})
    print("\n=== PROC-TARE-01 ===")
    for l in L: print(f"  {l:<10} n={L[l]['n']:>3}  A raw {L[l]['median_A_raw']:.4f} -> tared {L[l]['median_A_tare']:.4f} | sd {L[l]['sd_raw']:.4f} -> {L[l]['sd_tare']:.4f} | T_off {L[l]['median_T_offset']:+.4f} T_scale {L[l]['median_T_scale']:.4f} | z_lab commissioned {ZLAB[l]:+.4f} measured {L[l]['z_lab_measured_here']}")
    print(f"B1 reproduces the zeros (+-0.005)      -> {'MET' if b1 else 'FAILED'}")
    print(f"B2 tare moves toward 1.0, worst ({worst}) shrinks {100*shrink:.0f}%  -> {'MET' if b2 else 'FAILED (bar: toward, >=50%)'}")
    print(f"B3 all four medians inside NORMAL after tare -> {'MET' if b3 else 'FAILED'}")
    print(f"B4 spread tightens or holds in >=3 of 4  -> {'MET' if b4 else 'FAILED'}")
    print(f"B5 tare independent of age/sex r_age {r_age} r_sex {r_sex} -> {'MET' if b5 else 'FAILED'}")
    print(f"B6 chip term: sd of chip medians {chip_raw} -> {chip_tar} over {len(ch)} chips -> {'MET' if b6 else 'FAILED (bar: >=20% reduction)'}")
    print("B7 instrument-unchanged on the 11 commissioning arrays -> NOT ASSESSED in this run (recorded, not passed)")
    json.dump(res, open("handoff/tare01_results.json", "w"), indent=1); print("wrote handoff/tare01_results.json")


if __name__ == "__main__":
    main()
