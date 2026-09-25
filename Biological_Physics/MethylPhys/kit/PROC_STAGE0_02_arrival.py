#!/usr/bin/env python3
"""PROC-STAGE0-02: Stage 0 intake, steps 0.1-0.9, over every GSE87571 IDAT pair.

Manifest entries are built from the cohort's own metadata: accession and Sentrix barcode/position from the
file name, declared age and sex from the series matrix. Nothing is invented - a field the cohort does not
publish is left absent, which is exactly what the manifest gate is there to catch.
"""
import gzip, hashlib, json, os, re, sys, glob, collections, time
sys.path.insert(0, "Biological_Physics/MethylPhys/chain")
import stage_0_intake as S0

IDAT = "idats_gse87571"
OUT = "results/stage0_retro"
os.makedirs(OUT, exist_ok=True)

# --- the cohort's metadata
gsms, ages, sexes = None, None, None
with gzip.open("geo/GSE87571_series_matrix.txt.gz", "rt", errors="replace") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            gsms = re.findall(r'"([^"]+)"', line)
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = re.findall(r'"([^"]*)"', line)
            if vals and re.search(r"age", vals[0], re.I):
                ages = [re.sub(r"[^0-9.]", "", v) or None for v in vals]
            elif vals and re.search(r"(sex|gender)", vals[0], re.I):
                sexes = [v.split(":")[-1].strip() for v in vals]
        elif line.startswith("!series_matrix_table_begin"):
            break
meta = {g: {"age": (ages[i] if ages else None), "sex": (sexes[i] if sexes else None)}
        for i, g in enumerate(gsms or [])}
print(f"metadata: {len(meta)} samples, ages {sum(1 for v in meta.values() if v['age'])}, "
      f"sex {sum(1 for v in meta.values() if v['sex'])}", flush=True)

# --- the IDAT pairs
pat = re.compile(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)_(Grn|Red)\.idat\.gz$")
pairs = {}
for p in glob.glob(os.path.join(IDAT, "*.idat.gz")):
    m = pat.search(os.path.basename(p))
    if m:
        pairs.setdefault((m.group(1), m.group(2), m.group(3)), {})[m.group(4)] = p
print(f"IDAT pairs found: {sum(1 for v in pairs.values() if len(v) == 2)} complete, "
      f"{sum(1 for v in pairs.values() if len(v) != 2)} incomplete", flush=True)

# --- the sealed set: the analysis's own rule
cal = json.load(open("results/percell/stage1_betamean_GSE87571.json"))
chip_of = {g: b for (g, b, _pos) in pairs}
by_chip = collections.Counter(chip_of[g] for g in cal if g in chip_of)
sealed_chips = {c for c, n in by_chip.items() if n >= 9}
sealed = {g for g in cal if chip_of.get(g) in sealed_chips}
print(f"sealed set: {len(sealed)} arrays across {len(sealed_chips)} chips", flush=True)

rows, t0 = [], time.time()
for i, ((gsm, bar, pos), ch) in enumerate(sorted(pairs.items()), 1):
    md = meta.get(gsm, {})
    # exactly the field names REQUIRED_MANIFEST_FIELDS asks for - the first run used my own names and every
    # array came back QUARANTINE_MANIFEST_INVALID, which is the gate working, on the wrong input
    rec = {"accession": gsm, "sentrix_id": f"{bar}_{pos}", "sentrix_barcode": bar, "sentrix_position": pos,
           # SOP §12: the engine never sees a cleartext identifier, and VALID_ARRAY_TYPES is ("HM450K",
           # "EPIC_v1", "EPIC_v2") - the gate rejected "450k" and a bare accession, correctly
           "patient_id": hashlib.sha256(gsm.encode()).hexdigest()[:32],
           "array_type": "HM450K", "intake_date": "2026-09-23",
           "substrate": "whole_blood", "specimen": "whole_blood",
           "declared_sex": (md.get("sex") or "").strip() or None,
           "declared_chronological_age": float(md["age"]) if md.get("age") else None}
    grn, red = ch.get("Grn"), ch.get("Red")
    r = {"gsm": gsm, "chip": bar, "pos": pos, "in_sealed_set": gsm in sealed}
    try:
        a = S0.step_0_1_idat_arrival(rec, grn, red, os.path.join(OUT, "intake_log.jsonl"))
        r["step_0_1"] = a.get("status")
        b = S0.step_0_2_manifest_creation(a, None, os.path.join(OUT, "manifests"), "2026-09-23")
        r["step_0_2"] = b.get("status")
        c = S0.step_0_3_integrity_hash(b, grn, red, os.path.join(OUT, "integrity_log.jsonl"))
        r["step_0_3"] = c.get("status"); r["sha_grn"] = (c.get("sha256_grn") or "")[:12]
        d = S0.step_0_4_control_probe_validation(c, grn, red, None)
        r["step_0_4"] = d.get("ctrl_qc")
        e = S0.step_0_5_detection_pvalue_qc(d if isinstance(d, dict) and "status" in d else c, None, None)
        r["step_0_5"] = e.get("detection_qc")
        f6 = S0.step_0_6_bead_count_qc(c, None)
        r["step_0_6"] = f6.get("bead_qc")
        f7b = S0.step_0_7b_platform_coverage(c, None)
        r["step_0_7b"] = f7b.get("coverage_status") or f7b.get("status")
        g = S0.step_0_9_decision_gate(c, os.path.join(OUT, "verdict_log.jsonl"))
        r["verdict"] = g.get("verdict") or g.get("status")
        r["hard_fail"] = g.get("hard_fail")
        r["deferred"] = g.get("deferred")
    except Exception as ex:
        r["verdict"] = "ERROR"; r["error"] = f"{type(ex).__name__}: {ex}"[:200]
    rows.append(r)
    if i % 100 == 0:
        print(f"  {i}/{len(pairs)} ... {time.time()-t0:.0f}s", flush=True)

json.dump(rows, open("handoff/stage0_retro.json", "w"), indent=1)
v = collections.Counter(str(r.get("verdict")) for r in rows)
assert not (len(v) == 1 and "QUARANTINE_MANIFEST" in next(iter(v))), (
    "every array returned the same manifest quarantine - the manifest entries are wrong, not the arrays")
print("\nverdicts, all arrays:", dict(v))
sv = collections.Counter(str(r.get("verdict")) for r in rows if r["in_sealed_set"])
print("verdicts, the 268 sealed arrays:", dict(sv))
errs = [r for r in rows if r.get("verdict") == "ERROR"][:3]
for e in errs:
    print("  example error:", e["gsm"], e.get("error"))
