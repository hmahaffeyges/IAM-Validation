#!/usr/bin/env python3
"""PROC-STAGE0-02: run every gate on the cached per-array QC metrics and answer the question.

The question fixed in the pre-registration: how many of the arrays behind the sealed chip result would
Stage 0 have quarantined?
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, "Biological_Physics/MethylPhys/chain")
import stage_0_intake as S0

qc = json.load(open("results/stage0_retro/qc_metrics.json"))
arrival = {r["gsm"]: r for r in json.load(open("handoff/stage0_retro.json"))}
cal = json.load(open("results/percell/stage1_betamean_GSE87571.json"))

chip_of = {g: v["chip"] for g, v in qc.items()}
by_chip = collections.Counter(chip_of[g] for g in cal if g in chip_of)
sealed_chips = {c for c, n in by_chip.items() if n >= 9}
sealed = [g for g in cal if chip_of.get(g) in sealed_chips]

rows = {}
for g, m in qc.items():
    if "error" in m:
        rows[g] = {"verdict": "ERROR", "cause": m["error"][:80], "in_sealed": g in sealed}
        continue
    rec = {"array_type": "HM450K", "array_type_detected": "HM450K", "declared_sex": m.get("declared_sex"),
           "status": arrival.get(g, {}).get("step_0_1", "STAGED"), "flags": []}
    # 0.4 control probes
    cs = {"bisulfite_conversion_I_median": m["bs_efficiency"],
          "bisulfite_conversion_II_median": (1.0 - m["bs_efficiency"]) if m["bs_efficiency"] is not None else None,
          "hyb_high_median": m["hyb_high"], "hyb_low_median": m["hyb_low"],
          "extension_meth_median": m["ext_ratio"], "extension_unmeth_median": 1.0}
    rec.update({k: v for k, v in S0.validate_control_probes(cs).items() if k != "metrics"})
    # 0.5 detection, 0.6 bead, 0.7 call rate - from the measured fractions
    det = m["detection_pass_fraction"]
    rec["detection_qc"] = ("PASS" if det >= S0.DETECTION_PASS_FRACTION else
                           "DETECTION_BORDERLINE" if det >= S0.DETECTION_BORDERLINE_FRACTION else
                           "FAIL_LOW_DETECTION")
    bead = m["bead_pass_fraction"]
    rec["bead_qc"] = "PASS" if bead >= S0.BEAD_PASS_FRACTION else "WARN_LOW_BEAD_COUNT"
    cr = det * bead
    rec["call_rate"] = round(cr, 5)
    rec["call_rate_status"] = ("PASS" if cr >= S0.CALL_RATE_PASS else
                              "CALL_RATE_BORDERLINE" if cr >= S0.CALL_RATE_BORDERLINE else "CALL_RATE_FAIL")
    # 0.7b coverage is a property of the calibrated betas; every array here calibrated on the 450K reference
    rec["hm450_coverage_gate"] = "PASS" if g in cal else "DEFERRED_PENDING_STAGE1_DECODER"
    # 0.8 sex
    sv = S0.validate_sex(m["log2_x"], m["log2_y"], m.get("declared_sex"))
    rec.update(sv)
    rec["integrity_status"] = "INTEGRITY_OK"
    out = S0.step_0_9_decision_gate(dict(rec), None)
    rows[g] = {"verdict": out.get("stage0_verdict"), "hard": out.get("stage0_hard_fail") or [],
               "borderline": out.get("stage0_borderline") or [], "deferred": out.get("stage0_deferred_qc") or [],
               "in_sealed": g in sealed, "predicted_sex": sv["predicted_sex"],
               "declared_sex": m.get("declared_sex"), "bs": m["bs_efficiency"],
               "detection": det, "bead": bead, "call_rate": rec["call_rate"]}

def summary(keys, label):
    v = collections.Counter(rows[g]["verdict"] for g in keys)
    print(f"\n{label} (n={len(keys)}): {dict(v)}")
    causes = collections.Counter()
    for g in keys:
        for h in rows[g].get("hard", []):
            causes[h] += 1
    if causes: print("   hard-failure causes:", dict(causes))
    return v, causes

all_v, all_c = summary(list(rows), "ALL ARRAYS")
seal_v, seal_c = summary([g for g in rows if rows[g]["in_sealed"]], "THE SEALED CHIP SET")

ok = [g for g in rows if rows[g]["verdict"] and rows[g]["verdict"] != "ERROR"]
sexes = [(rows[g]["declared_sex"], rows[g]["predicted_sex"]) for g in ok if rows[g].get("declared_sex")]
conc = sum(1 for d, p in sexes if (d or "").strip().upper()[:1] == p)
print(f"\nsex concordance: {conc}/{len(sexes)} = {conc/len(sexes):.4f}")
print("discordant by declared sex:", dict(collections.Counter(d for d, p in sexes if (d or '').strip().upper()[:1] != p)))

bs = np.array([rows[g]["bs"] for g in ok if rows[g].get("bs") is not None])
det = np.array([rows[g]["detection"] for g in ok])
cr = np.array([rows[g]["call_rate"] for g in ok])
bd = np.array([rows[g]["bead"] for g in ok])
for name, arr, thr in (("bisulfite efficiency", bs, S0.BS_CONVERSION_MIN),
                       ("detection pass fraction", det, S0.DETECTION_PASS_FRACTION),
                       ("call rate", cr, S0.CALL_RATE_PASS),
                       ("bead pass fraction", bd, S0.BEAD_PASS_FRACTION)):
    print(f"{name:<26} median {np.median(arr):.4f}  p05 {np.percentile(arr,5):.4f}  p95 {np.percentile(arr,95):.4f}  "
          f"min {arr.min():.4f}  threshold {thr}  below it: {int((arr<thr).sum())}/{len(arr)}")

json.dump({"_meta": {"procedure": "PROC-STAGE0-02", "built": "2026-09-23", "n_arrays": len(rows),
                     "sealed_chips": len(sealed_chips), "n_sealed": len(sealed),
                     "note": "sealed set reconstructed by the analysis rule (chips with >= 9 calibrated arrays)"},
           "verdicts_all": dict(all_v), "verdicts_sealed": dict(seal_v),
           "hard_causes_all": dict(all_c), "hard_causes_sealed": dict(seal_c),
           "sex_concordance": {"n": len(sexes), "concordant": conc},
           "distributions": {k: {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
                                 "p95": float(np.percentile(v, 95)), "min": float(v.min()),
                                 "below_threshold": int((v < t).sum())}
                             for k, v, t in (("bs_efficiency", bs, S0.BS_CONVERSION_MIN),
                                             ("detection", det, S0.DETECTION_PASS_FRACTION),
                                             ("call_rate", cr, S0.CALL_RATE_PASS),
                                             ("bead", bd, S0.BEAD_PASS_FRACTION))},
           "per_array": rows}, open("handoff/stage0_seal.json", "w"), indent=1)
print("\nwrote handoff/stage0_seal.json")
