#!/usr/bin/env python3
"""run_sample.py - one command, IDAT pair (or a beta table) to a MethylPhys report.

This exists because the Run tab used to advertise a command that did not: `build_methylphys.py` only ever
took a pre-computed bundle. Written 2026-09-22.

    # an Illumina IDAT pair (needs methylprep for Stage 1)
    python3 run_sample.py --grn SAMPLE_Grn.idat.gz --red SAMPLE_Red.idat.gz --age 58 --lab MYLAB --out report.html

    # or a beta table you have already calibrated: a two-column CSV, cpg_id,beta (no pipeline map is applied
    # unless you name one with --pipeline, and without a map the class gauge is NOT REPORTABLE by design)
    python3 run_sample.py --betas mysample.csv --age 58 --lab MYLAB --out report.html

A laboratory the chain has not commissioned has no zero and no sky scale, so the report prints
NOT REPORTABLE with the reason rather than a number. To commission one: 40 healthy arrays of that
laboratory through Stage 1, then MethylPhys/chain/lab_zero.py - the procedure is in the RUNBOOK.
"""
import os, sys, argparse, pickle, json

def _bio_root(start=None):
    """The directory that CONTAINS MethylPhys/ - i.e. Biological_Physics.

    Derived by ascending from this file until a directory holding 'MethylPhys' is found, rather than by counting
    dirname() calls. The 2026-09-22 move (CPG_Engine -> MethylPhys/chain, Testing_and_Code -> Record) changed the
    depth of every script by one; a counted chain resolved to MethylPhys and silently stopped finding Record/.
    """
    import os as _os
    d=_os.path.dirname(_os.path.abspath(start or __file__))
    for _ in range(8):
        if _os.path.isdir(_os.path.join(d,"MethylPhys")): return d
        if _os.path.basename(d)=="MethylPhys": return _os.path.dirname(d)
        nd=_os.path.dirname(d)
        if nd==d: break
        d=nd
    return _os.path.dirname(_os.path.dirname(_os.path.abspath(start or __file__)))

HERE=os.path.dirname(os.path.abspath(__file__)); ENG=os.path.dirname(HERE); BIO=_bio_root()
for d in (ENG, HERE, os.path.join(ENG,"Walther_iam_deconvolver")): sys.path.insert(0,d)

def _atlas():
    csv=os.path.join(BIO,"MethylPhys/atlas","IAMAtlasREBUILD.csv"); xz=csv+".xz"
    if not os.path.exists(csv):
        if not os.path.exists(xz): sys.exit(f"atlas not found: {xz}")
        import lzma, shutil
        print(f"decompressing the atlas once -> {csv} (605 MB)", flush=True)
        with lzma.open(xz,"rb") as f, open(csv,"wb") as g: shutil.copyfileobj(f,g,1<<24)
    return csv

SUBSTRATE_TOKENS = ("whole_blood", "plasma_cfDNA", "tissue", "unknown")


def substrate_token(text):
    """Map a laboratory's own specimen description to the controlled token the report renders.

    The custody record keeps what the laboratory wrote; the report gets a token. `--specimen "colorectal
    tumour tissue"` crashed the report's vocabulary guard on 2026-09-23 - correctly, because a condition name
    has no place in a reading - and the fix is not to forbid the description but to keep free text out of
    report prose.
    """
    t = (text or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not t:
        return "unknown"
    if t in SUBSTRATE_TOKENS:
        return t
    if "blood" in t or "buffy" in t:
        return "whole_blood"
    if "cfdna" in t or "plasma" in t or "cell_free" in t:
        return "plasma_cfDNA"
    if any(k in t for k in ("tissue", "biopsy", "tumour", "tumor", "adenoma", "carcinoma", "colon",
                            "breast", "lung", "resection")):
        return "tissue"
    return "unknown"


def main():
    ap=argparse.ArgumentParser(description="IDAT pair or beta table -> MethylPhys report")
    ap.add_argument("--grn"); ap.add_argument("--red"); ap.add_argument("--betas")
    ap.add_argument("--age", type=float, help="declared age in years; without it the age term cannot be removed. Float because cohorts publish decimal ages (GEO carries 72.0 and 54.5 as readily as 72)")
    ap.add_argument("--lab", help="laboratory / pipeline identity, e.g. GSE87571 for a commissioned one")
    ap.add_argument("--pipeline", default="stage1_noob_450K", help="pipeline map name from beta_scale_maps_v1.json")
    ap.add_argument("--specimen", default="whole blood")
    ap.add_argument("--lab-zero", type=float, default=None, help="this laboratory's measured zero; omit to read UNSET")
    ap.add_argument("--out", default="methylphys_report.html"); ap.add_argument("--id", default=None)
    ap.add_argument("--bundle", help="also write the full bundle as JSON - every stage's output, for a test harness or an integration that needs more than the report")
    ap.add_argument("--sex", default=None, help="declared sex (F/M); Stage 0.8 compares it with the array")
    ap.add_argument("--patient-id", default=None, help="already-hashed identifier; a cleartext one is hashed here")
    ap.add_argument("--array-type", default="HM450K", choices=("HM450K", "EPIC_v1", "EPIC_v2"))
    ap.add_argument("--intake-log", default=None, help="append intake and integrity records here")
    ap.add_argument("--manifest-dir", default=None, help="write the immutable per-sample manifest here")
    ap.add_argument("--no-intake", action="store_true", help="skip Stage 0 (recorded in the report as skipped)")
    a=ap.parse_args()
    if not (a.betas or (a.grn and a.red)): ap.error("give either --betas or both --grn and --red")

    intake = None
    if a.grn and a.red and not a.no_intake:
        # Stage 0 runs BEFORE calibration, and a quarantine stops the run: the gauge never sees a specimen
        # the chain of custody rejected. SOP sections 11-19.
        import hashlib, re as _re
        try:
            import stage_0_intake as S0
        except ImportError:
            sys.path.insert(0, os.path.join(BIO, "MethylPhys/chain")); import stage_0_intake as S0
        m = _re.search(r"(\d{9,12})[_-](R0\dC0\d)", os.path.basename(a.grn))
        sentrix = f"{m.group(1)}_{m.group(2)}" if m else os.path.basename(a.grn).split("_Grn")[0]
        pid = a.patient_id or (a.id or os.path.basename(a.grn).split("_")[0])
        if len(pid.replace("_", "").replace("-", "")) < 16 or not pid.replace("_", "").replace("-", "").isalnum():
            pid = hashlib.sha256(pid.encode()).hexdigest()[:32]   # SOP 12: the engine never sees a cleartext id
        nsnps, hdr = S0.read_idat_nsnps(a.grn)
        detected = S0._infer_platform(nsnps) if nsnps else None
        if detected and a.array_type and a.array_type != detected and "--array-type" in " ".join(sys.argv):
            print(f"  declared {a.array_type}, header says {detected} - Stage 0.1 will gate on the mismatch",
                  flush=True)
        declared = a.array_type if ("--array-type" in " ".join(sys.argv) or not detected) else detected
        print(f"  array type: {declared}" + (f" (header: {detected}, {nsnps:,} addresses)" if nsnps else
                                             f" (header unreadable: {hdr})"), flush=True)
        entry = {"sentrix_id": sentrix, "array_type": declared, "patient_id": pid,
                 "intake_date": __import__("datetime").date.today().isoformat(),
                 "substrate": a.specimen.replace(" ", "_"), "declared_sex": a.sex,
                 "declared_chronological_age": float(a.age) if a.age is not None else None}
        print("Stage 0: intake gates (SOP 11-19)", flush=True)

        def _stopped(r):
            """A quarantine ends intake. Without this the next step overwrites the status and the verdict
            names a downstream symptom instead of the cause (seen 2026-09-23: an array-type mismatch was
            reported as a detection failure because step 0.2 had already written MANIFEST_COMPLETE)."""
            return str(r.get("status") or "").startswith("QUARANTINE")

        rec = S0.step_0_1_idat_arrival(entry, a.grn, a.red, a.intake_log)
        if not _stopped(rec):
            rec = S0.step_0_2_manifest_creation(rec, None, a.manifest_dir, entry["intake_date"])
        if not _stopped(rec):
            rec = S0.step_0_3_integrity_hash(rec, a.grn, a.red, a.intake_log)
        if _stopped(rec):
            print(f"  {rec['status']}  flags: {rec.get('flags')}", flush=True)
            rec = S0.step_0_9_decision_gate(rec, a.intake_log)
            print(f"Stage 0 verdict: {rec.get('stage0_verdict')}  "
                  f"hard failures: {rec.get('stage0_hard_fail')}", flush=True)
            print("\nQUARANTINE - the chain stops here and nothing is scored. "
                  "Stage 0 rejected the specimen before calibration.", flush=True)
            sys.exit(2)
        try:
            from stage_0_1_qc_handoff import decode_qc_inputs
            q = decode_qc_inputs(a.grn, a.red, rec.get("array_type_detected") or declared)
            rec = S0.step_0_4_control_probe_validation(rec, a.grn, a.red, q["control_summary"])
            rec = S0.step_0_5_detection_pvalue_qc(rec, q["probe_intensities"], q["neg_control_stats"])
            rec = S0.step_0_6_bead_count_qc(rec, q["bead_counts"])
            import numpy as _np
            dp = _np.asarray(S0.compute_detection_p(q["probe_intensities"], q["neg_control_stats"]["mu_bg"],
                                                    q["neg_control_stats"]["sigma_bg"]))
            rec = S0.step_0_7_call_rate(rec, dp < S0.DETECTION_P_THRESHOLD,
                                        _np.asarray(q["bead_counts"]) >= S0.BEAD_COUNT_MIN)
            rec = S0.step_0_8_sex_check(rec, q["sex_intensities"])
        except ImportError as e:
            # the module is missing, not the data: the gates report DEFERRED and say why
            print(f"  QC hand-off unavailable ({e}); those gates report DEFERRED", flush=True)
        except Exception as e:
            # the decoder reached the file and failed on it. That is a property of the specimen's files, so it
            # is a quarantine, not a deferral - a corrupt IDAT must not reach Stage 1 with its QC "deferred".
            # (2026-09-23: one array in a 732-pair local copy was truncated at the gzip level while passing the
            # 1 MB size floor, and the earlier wiring would have let it through with six gates unmeasured.)
            rec["status"] = "QUARANTINE_CORRUPT_IDAT"
            rec.setdefault("flags", []).append(f"IDAT_DECODE_FAILED:{type(e).__name__}")
            print(f"  QUARANTINE_CORRUPT_IDAT - the decoder failed on this pair: {type(e).__name__}: {e}",
                  flush=True)
            rec = S0.step_0_9_decision_gate(rec, a.intake_log)
            print(f"Stage 0 verdict: {rec.get('stage0_verdict')}  hard failures: {rec.get('stage0_hard_fail')}",
                  flush=True)
            print("\nQUARANTINE - the chain stops here and nothing is scored.", flush=True)
            sys.exit(2)
        intake = rec

    if a.betas:
        import pandas as pd
        d=pd.read_csv(a.betas, index_col=0).iloc[:,0].dropna(); beta=d.to_dict()
        sid=a.id or os.path.basename(a.betas).split(".")[0]
    else:
        try:
            from stage_1_idat_calibration import calibrate_idat_to_beta
        except ImportError:
            for c in (os.path.join(BIO,"MethylPhys/chain"), os.path.join(BIO,"MethylPhys","Reproduction_Kit")):
                sys.path.insert(0,c)
            from stage_1_idat_calibration import calibrate_idat_to_beta
        print("Stage 1: calibrating the IDAT pair (methylprep noob) - about 25 s", flush=True)
        b,meta=calibrate_idat_to_beta(a.grn,a.red)
        b=b.iloc[:,0] if hasattr(b,"columns") else b
        beta=b.dropna().to_dict(); sid=a.id or os.path.basename(a.grn).split("_")[0]

    if intake is not None:
        import stage_0_intake as S0
        ref = None
        try:
            import json as _json, glob as _glob
            f = _glob.glob(os.path.join(BIO, "MethylPhys/chain/**/iamatlas_gauge_identity_loci_v1_0.json"),
                           recursive=True)
            if f:
                loci = set()
                for cls, d in _json.load(open(f[0])).items():
                    if isinstance(d, dict) and d.get("loci"):
                        loci.update(d["loci"])
                ref = len(loci & set(beta)) / max(len(loci), 1)
        except Exception:
            ref = None
        intake = S0.step_0_7b_platform_coverage(intake, ref)
        intake = S0.step_0_9_decision_gate(intake, a.intake_log)
        v = intake.get("stage0_verdict")
        print(f"Stage 0 verdict: {v}"
              + (f"  hard failures: {intake.get('stage0_hard_fail')}" if intake.get("stage0_hard_fail") else "")
              + (f"  deferred: {intake.get('stage0_deferred_qc')}" if intake.get("stage0_deferred_qc") else ""), flush=True)
        if v == "QUARANTINE":
            print("\nQUARANTINE - the chain stops here and nothing is scored. "
                  "Stage 0 rejected the specimen; see the flags above.", flush=True)
            sys.exit(2)

    import cpg_conductor as C, build_methylphys as B
    cfg={"age":a.age,"pipeline":a.pipeline,"lab_zero":a.lab_zero,"substrate":substrate_token(a.specimen), "substrate_as_declared":a.specimen,
         "intake": intake, "intake_skipped": bool(a.no_intake)}
    if a.lab: cfg["lab"]=a.lab
    print(f"running the chain on {len(beta):,} CpGs", flush=True)
    o=C.run_full(beta, _atlas(), cfg=cfg)
    o["intake"] = intake          # the report prints the Stage 0 record, or NOT RUN when there is none
    o["intake_skipped"] = bool(a.no_intake)
    r=B.build(o, a.out, sid)
    if a.bundle:
        import json as _json
        _json.dump(o, open(a.bundle, "w"), default=str)
        print(f"bundle: {a.bundle}", flush=True)
    imm=o["classes"].get("immune",{})
    print(f"\n{sid}: immune A'' {imm.get('A_abs')}  placement {imm.get('placement')}  tier {imm.get('tier')}")
    print(f"refusals: {len(r['refusals'])}" + (f" -> {r['refusals'][:3]}" if r["refusals"] else ""))
    print(f"report: {r['out']}  ({r['bytes']//1000} KB)")

if __name__=="__main__": main()
