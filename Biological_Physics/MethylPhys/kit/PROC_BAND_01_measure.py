#!/usr/bin/env python3
"""PROC-BAND-01 step 1: A_mapped for both identity surfaces on 318 healthy arrays, through the chain's path.

Writes handoff/band01_arrays.json — one record per array:
    gse, gsm, age, sex, class_fractions, joint_fraction, immune A_mapped/A_abs, joint A_mapped

WHAT IS AND IS NOT THE CHAIN'S OWN CODE. The scale map (stage_1s_scale_map) and the identity gauge
(stage_b_identity) are called exactly as run_full calls them, so the number measured here is the number a
reading reports. The one departure: composition comes from calling the deconvolver ONCE and reusing it for
318 arrays, where stage_a_cells constructs it per call - constructing it per array would re-read the 605 MB
atlas 318 times. Same class, same atlas, same class map, same deconvolve() call, so the fractions are
identical; what is skipped is stage_a_cells' per-cell A scoring and its 500-draw bootstrap, which the band
does not use. Stated so nobody has to infer it.

Ages come from the GEO series-matrix headers and are written out with the readings, because the band cannot be
reproduced without them.
"""
import gzip
import json
import os
import re
import sys
import time
import urllib.request

REF = "iamrepo/Biological_Physics/MethylPhys/reference_data"
CH = "iamrepo/Biological_Physics/MethylPhys/chain"
COHORTS = ["GSE87571", "GSE42861", "GSE111629", "GSE125105"]
sys.path.insert(0, CH)


def fetch_ages(gse):
    """GSM -> (age, sex) from the series-matrix header. Streamed and stopped at the table, so this is a few MB."""
    out = os.path.join("geo_hdr", gse + "_hdr.txt")
    os.makedirs("geo_hdr", exist_ok=True)
    if not os.path.exists(out):
        url = ("https://ftp.ncbi.nlm.nih.gov/geo/series/%snnn/%s/matrix/%s_series_matrix.txt.gz"
               % (gse[:-3], gse, gse))
        with urllib.request.urlopen(url, timeout=300) as r, open(out + ".gz", "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                if os.path.getsize(out + ".gz") > 60 << 20:      # the header is far smaller than this
                    break
        txt = []
        with gzip.open(out + ".gz", "rt", errors="replace") as f:
            for line in f:
                if line.startswith("!series_matrix_table_begin"):
                    break
                txt.append(line)
        open(out, "w", encoding="utf-8").write("".join(txt))
        os.remove(out + ".gz")
    hdr = open(out, encoding="utf-8", errors="replace").read()
    gsms, ages, sexes = [], {}, {}
    for line in hdr.split("\n"):
        if line.startswith("!Sample_geo_accession"):
            gsms = re.findall(r'"(GSM\d+)"', line)
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = re.findall(r'"([^"]*)"', line)
            if not gsms or len(vals) != len(gsms):
                continue
            key = vals[0].split(":")[0].strip().lower()
            if re.search(r"\bage\b", key):
                for g, v in zip(gsms, vals):
                    m = re.search(r"([\d.]+)", v.split(":")[-1])
                    if m:
                        ages[g] = float(m.group(1))
            elif re.search(r"sex|gender", key):
                for g, v in zip(gsms, vals):
                    s = v.split(":")[-1].strip().lower()
                    sexes[g] = "F" if s.startswith(("f", "wom")) else ("M" if s.startswith(("m", "man")) else None)
    return ages, sexes


def main():
    import pickle
    import lzma
    import cpg_conductor as C

    atlas_xz = os.path.join(os.path.dirname(CH), "atlas", "IAMAtlasREBUILD.csv.xz")
    atlas = os.path.join("atlas_work", "IAMAtlasREBUILD.csv")
    if not os.path.exists(atlas):
        os.makedirs("atlas_work", exist_ok=True)
        import shutil
        with lzma.open(atlas_xz, "rb") as f, open(atlas, "wb") as g:
            shutil.copyfileobj(f, g, 1 << 24)
        print("atlas decompressed: %.0f MB" % (os.path.getsize(atlas) / 1e6), flush=True)

    # The commissioned laboratory zeros, taken from the band file's own cohort records (z_lab_full_cohort,
    # computed on the full cohorts by PROC-SWITCH-01). Not refitted here - B7. lab_zero.py computes a zero
    # from a 40-array panel, which is how these were made; recomputing them on the 80-array published panels
    # would be a different number and would put a refitted reference under the band.
    import ast as _ast
    band = json.load(open(C._find("identity_band_v3.json")))
    coh = band["_meta"]["cohorts"]
    if isinstance(coh, str):
        coh = _ast.literal_eval(coh)
    zeros = {}
    for key, rec in coh.items():
        zeros[key.split("_")[0]] = rec.get("z_lab_full_cohort", rec.get("z_lab"))
    missing = [g for g in COHORTS if zeros.get(g) is None]
    assert not missing, "no commissioned laboratory zero for %s - refusing to invent one" % missing
    print("laboratory zeros (commissioned):", {k: zeros[k] for k in COHORTS}, flush=True)

    dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
    dec = dec_mod.WaltherIAMDeconvolver(atlas, celltype_class_map=str(
        C._find("IAMAtlasREBUILD_celltype_to_class.json")))
    print("deconvolver built once", flush=True)

    done = {}
    if os.path.exists("handoff/band01_arrays.json"):
        done = {r["gsm"]: r for r in json.load(open("handoff/band01_arrays.json"))["arrays"]}
        print("resuming:", len(done), "arrays already scored", flush=True)
    rows = list(done.values())
    meta = {"procedure": "PROC-BAND-01", "built": time.strftime("%Y-%m-%d %H:%M"),
            "betas": "reference_data/stage1_betas_<GSE>.pkl.xz (published Stage 1 noob output)",
            "path": "stage_1s_scale_map -> deconvolver.deconvolve -> stage_b_identity, as run_full calls them",
            "pipeline": "stage1_noob_450K", "ages": "GEO series-matrix headers"}
    for gse in COHORTS:
        ages, sexes = fetch_ages(gse)
        with lzma.open(os.path.join(REF, "stage1_betas_%s.pkl.xz" % gse), "rb") as f:
            df = pickle.load(f)
        cols = [c for c in df.columns if c not in done]
        print("%s: %d arrays x %d loci | ages for %d | %d to score"
              % (gse, df.shape[1], df.shape[0], len(ages), len(cols)), flush=True)
        t0 = time.time()
        for n, gsm in enumerate(cols, 1):
            beta = df[gsm].dropna().to_dict()
            beta_rm, scale_label = C.stage_1s_scale_map(beta, "stage1_noob_450K")
            fr = dict(dec.deconvolve(beta).class_fractions)
            lzv = zeros.get(gse)
            if isinstance(lzv, dict):
                lzv = lzv.get("z_lab", lzv.get("zero"))
            bi = C.stage_b_identity(beta_rm, {"class_fractions": fr}, ages.get(gsm), scale_label, lab_zero=lzv)
            imm, jnt = bi.get("immune", {}), bi.get("haematopoietic_progenitor", {})
            rows.append({"gse": gse, "gsm": gsm, "age": ages.get(gsm), "sex": sexes.get(gsm),
                         "lab_zero": lzv, "scale": scale_label,
                         "fr_immune": round(fr.get("immune", 0.0), 5),
                         "fr_progenitor": round(fr.get("progenitor", 0.0), 5),
                         "fr_stem_adult": round(fr.get("stem_adult", 0.0), 5),
                         "immune": {k: imm.get(k) for k in ("present", "fraction", "A_mapped", "A_abs",
                                                            "age_reference_c", "n_loci", "placement")},
                         "joint": {k: jnt.get(k) for k in ("present", "fraction", "A_mapped", "n_loci")}})
            if n % 20 == 0 or n == len(cols):
                json.dump({"_meta": meta, "arrays": rows}, open("handoff/band01_arrays.json", "w"))
                print("  [%d/%d] %s  %.1fs/array" % (n, len(cols), gsm, (time.time() - t0) / n), flush=True)
        json.dump({"_meta": meta, "arrays": rows}, open("handoff/band01_arrays.json", "w"))
        print("%s DONE in %.0fs" % (gse, time.time() - t0), flush=True)
    print("TOTAL arrays scored:", len(rows), flush=True)


if __name__ == "__main__":
    os.makedirs("handoff", exist_ok=True)
    main()
