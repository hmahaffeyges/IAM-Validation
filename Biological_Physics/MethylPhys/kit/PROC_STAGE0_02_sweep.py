#!/usr/bin/env python3
"""Decode the Stage 0 QC inputs for every GSE87571 array and record what each gate measures.

Written for PROC-STAGE0-02: the retrospective intake run. Caches per array so it resumes.
"""
import gzip, json, os, re, sys, glob, time, hashlib
import numpy as np
sys.path.insert(0, "iamrepo/Biological_Physics/MethylPhys/chain")
import stage_0_intake as S0
from stage_0_1_qc_handoff import decode_qc_inputs

CACHE = "results/stage0_retro/qc_metrics.json"
os.makedirs("results/stage0_retro", exist_ok=True)
done = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

gsms = sexes = ages = None
with gzip.open("geo/GSE87571_series_matrix.txt.gz", "rt", errors="replace") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            gsms = re.findall(r'"([^"]+)"', line)
        elif line.startswith("!Sample_characteristics_ch1"):
            v = re.findall(r'"([^"]*)"', line)
            if v and re.search(r"(sex|gender)", v[0], re.I):
                sexes = [x.split(":")[-1].strip() for x in v]
            elif v and re.search(r"age", v[0], re.I):
                ages = [re.sub(r"[^0-9.]", "", x) or None for x in v]
        elif line.startswith("!series_matrix_table_begin"):
            break
meta = {g: {"sex": (sexes[i] if sexes else None), "age": (ages[i] if ages else None)} for i, g in enumerate(gsms)}

pat = re.compile(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)_(Grn|Red)\.idat\.gz$")
pairs = {}
for p in glob.glob("idats_gse87571/*.idat.gz"):
    m = pat.search(os.path.basename(p))
    if m:
        pairs.setdefault((m.group(1), m.group(2), m.group(3)), {})[m.group(4)] = p
todo = [k for k in sorted(pairs) if k[0] not in done]
print(f"arrays: {len(pairs)} | cached: {len(done)} | to decode: {len(todo)}", flush=True)

t0 = time.time()
for i, (gsm, bar, pos) in enumerate(todo, 1):
    ch = pairs[(gsm, bar, pos)]
    try:
        q = decode_qc_inputs(ch["Grn"], ch["Red"], "HM450K")
        cs = q["control_summary"]
        dp = np.asarray(S0.compute_detection_p(q["probe_intensities"],
                                               q["neg_control_stats"]["mu_bg"],
                                               q["neg_control_stats"]["sigma_bg"]))
        det = float((dp < S0.DETECTION_P_THRESHOLD).mean())
        bead = S0.validate_bead_count(q["bead_counts"])
        sx = q["sex_intensities"]
        done[gsm] = {
            "chip": bar, "pos": pos, "declared_sex": meta.get(gsm, {}).get("sex"),
            "declared_age": meta.get(gsm, {}).get("age"),
            "bs_efficiency": cs["bisulfite_conversion_I_median"], "bs_pairs": cs["bs_pairs_used"],
            "hyb_high": cs["hyb_high_median"], "hyb_low": cs["hyb_low_median"],
            "ext_ratio": (cs["extension_meth_median"] / cs["extension_unmeth_median"]
                          if cs["extension_unmeth_median"] else None),
            "detection_pass_fraction": round(det, 5),
            "bead_pass_fraction": bead["pct_probes_bead_count_ge_3"],
            "log2_x": sx["log2_x_median"], "log2_y": sx["log2_y_median"],
            "predicted_sex": S0.predict_sex(sx["log2_x_median"], sx["log2_y_median"]),
            "mu_bg": round(q["neg_control_stats"]["mu_bg"], 1),
            "sigma_bg": round(q["neg_control_stats"]["sigma_bg"], 1),
        }
    except Exception as e:
        done[gsm] = {"error": f"{type(e).__name__}: {e}"[:200], "chip": bar, "pos": pos}
    if i % 25 == 0 or i == len(todo):
        tmp = CACHE + ".tmp"
        json.dump(done, open(tmp, "w"))
        os.replace(tmp, CACHE)
        el = time.time() - t0
        print(f"  {i}/{len(todo)} decoded, {el/i:.1f}s each, ~{(len(todo)-i)*el/i/60:.0f} min left", flush=True)
print(f"DONE {len(done)} arrays", flush=True)
