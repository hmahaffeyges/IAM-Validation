#!/usr/bin/env python3
"""run_sample.py - one command: IDAT pair (or a beta table, or single-molecule reads) to a MethylPhys report.

Default engine v3 (development build, neutrophils, EPIC v1): Stage 0 intake -> Stage 1 IDAT calibration -> conductor_v3
(platform check, Stage A composition, Stage M Met-A, Stage MC C-score, Stage T same-run tare) -> report_v3, plus Stage Q IAM-A
when single-molecule input is given.

    # an Illumina EPIC v1 IDAT pair (needs methylprep for Stage 1); --sex and --age are required by Stage 0
    python3 run_sample.py --grn S_Grn.idat.gz --red S_Red.idat.gz --specimen "whole blood" --sex F --age 52 --id S001 --out S001.html

    # pass 2 of a batch: the same, with the same-run healthy references from pass 1
    python3 run_sample.py ... --slide-ref-table refs.csv        # columns A,f_neu,N; >= 20 rows -> noise-corrected tare
    python3 run_sample.py ... --slide-ref-A 0.951,0.957,0.962   # plain A values -> median tare

    # a beta table already calibrated by this chain's Stage 1: two-column CSV cpg_id,beta (Stage 0 does not run)
    python3 run_sample.py --betas S.csv --id S001 --out S001.html

    # IAM-A from a wgbstools .pat file (pipeline loyfer_pat_v1), or from a per-site table with its pipeline named
    python3 run_sample.py --pat S.pat.gz --id S001 --out S001.html
    python3 run_sample.py --site-table S_sites.csv --seq-pipeline loyfer_pat_v1 --id S001 --out S001.html

--engine legacy runs the class-floor conductor kept for the record; --lab, --lab-zero and --pipeline apply to it only.
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
for d in (ENG, HERE, os.path.join(ENG,"legacy_iam_deconvolver")): sys.path.insert(0,d)

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


def _assign_run_id(ledger_path):
    """RUN-YYYYMMDD-NN, sequential within the day, read from the ledger this run is about to append to.

    A run is not a test: it makes no claim and passes no bar, so it is not a VAL and not a PROC. It is one
    execution of the chain on one specimen, and it needs an identifier so a reading can be pointed at a year
    from now. Assigned here, at the moment of the run, and written into the bundle, the ledger row and the
    report header (author's question, 2026-09-25).
    """
    import datetime
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    n = 0
    try:
        with open(ledger_path, encoding="utf-8") as f:
            for line in f:
                if '"run_id": "RUN-' + day in line or '"run_id":"RUN-' + day in line:
                    n += 1
    except OSError:
        pass
    return "RUN-%s-%02d" % (day, n + 1)


def _covariates(a):
    """Phenotype and covariates for this run: --covariates JSON first, then --covariate key=value on top."""
    import json as _json
    cov = {}
    if getattr(a, "covariates", None):
        with open(a.covariates, encoding="utf-8") as f:
            loaded = _json.load(f)
        if not isinstance(loaded, dict):
            raise SystemExit("--covariates must contain a JSON object of key/value pairs")
        cov.update({str(k): loaded[k] for k in loaded})
    for kv in getattr(a, "covariate", None) or []:
        if "=" not in kv:
            raise SystemExit(f"--covariate expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        cov[k.strip()] = v.strip()
    return cov


def _sha12(path):
    import hashlib as _h
    try:
        h = _h.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except OSError:
        return None


def _versions(chain_dir):
    """Every input that can change a reading, with a short hash - so runs months apart are poolable.

    A matrix row that does not say which atlas and which band produced it cannot be pooled with one that
    used different files, and nothing in the reading itself reveals the difference.
    """
    import glob as _glob
    import json as _json
    import platform as _plat
    import subprocess as _sub
    import time as _time
    out = {"run_timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
           "python": _plat.python_version()}
    try:
        import methylprep as _mp
        out["methylprep"] = getattr(_mp, "__version__", "unknown")
    except Exception:
        out["methylprep"] = "not importable"
    try:
        out["chain_commit"] = _sub.run(["git", "-C", chain_dir, "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
        out["chain_dirty"] = bool(_sub.run(["git", "-C", chain_dir, "status", "--porcelain"],
                                           capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        out["chain_commit"] = "unknown"
    files = {}
    for pat in ("**/iamatlas_gauge_identity_loci_v1_0.json", "**/identity_band_v3.json",
                "**/beta_scale_maps_v1.json", "**/reference_age_curve_v1.json", "**/tier_breakpoints.json",
                "**/percell_reference_v0_3.json", "../atlas/IAMAtlasREBUILD.csv",
                "../atlas/IAMAtlasREBUILD.csv.xz", "../atlas/IAMAtlasREBUILD_celltype_to_class.json",
                "cpg_conductor.py", "iamatlas_a_scoring.py", "stage_0_intake.py",
                "stage_0_1_qc_handoff.py", "stage_1_idat_calibration.py",
                "MethylPhys_Interface/build_methylphys.py"):
        for p in _glob.glob(os.path.join(chain_dir, pat), recursive=True):
            if "RETIRED" in p:
                continue
            files[os.path.relpath(p, chain_dir)] = {"sha256_12": _sha12(p), "bytes": os.path.getsize(p)}
    out["inputs"] = files
    return out


def _class_z(o, chain_dir=None):
    """Put the z the chain already computed onto each class record, with the reference it was measured against.

    Stage 5 computes z per class in departure["top_axis_contributions"] - patient A, the age-matched mean and
    the sigma it used - but that list only carries the axes it reported, and a matrix wants the number on the
    class row it belongs to. identity_band_v3.json is a POOLED percentile band (p10/p50/p90 plus per-decade),
    not a per-class mean and sigma, so nothing here re-derives a band: the sigma is the one Stage 5 used, and
    the per-locus sky statistics are carried across as they are.
    """
    dep = o.get("departure") or {}
    by_class = {}
    for ax in (dep.get("top_axis_contributions") or []):
        if isinstance(ax, dict) and ax.get("class"):
            by_class[ax["class"]] = ax
    sky = ((o.get("patient_sky") or {}).get("classes") or {})
    for cls, rec in (o.get("classes") or {}).items():
        if not isinstance(rec, dict):
            continue
        ax = by_class.get(cls)
        if ax:
            rec["z"] = ax.get("z")
            rec["age_matched_mean"] = ax.get("age_matched_mean")
            rec["band_sigma"] = ax.get("sigma")
            rec["band_widths_from_line"] = ax.get("band_widths_from_line")
        sk = sky.get(cls)
        if isinstance(sk, dict):
            rec["sky_median_z"] = sk.get("median_z")
            rec["sky_frac_abs_z_gt2"] = sk.get("frac_abs_z_gt2")
            rec["sky_n_loci"] = sk.get("n")


def _ledger_row(o, sample_id, out_path):
    """One flat row per run: everything a disease matrix needs from this specimen, without opening the bundle."""
    intake = o.get("intake") or {}
    dep = o.get("departure") or {}
    row = {"sample_id": sample_id, "report": os.path.basename(out_path),
        "run_id": o.get("run_id"),
           "run_timestamp_utc": (o.get("versions") or {}).get("run_timestamp_utc"),
           "chain_commit": (o.get("versions") or {}).get("chain_commit"),
           "sample_run_id": intake.get("sample_run_id"), "sentrix_id": intake.get("sentrix_id"),
           "array_type": intake.get("array_type"), "substrate": o.get("context", {}).get("substrate"),
           "declared_age": intake.get("declared_chronological_age"),
           "declared_sex": intake.get("declared_sex"), "predicted_sex": intake.get("predicted_sex"),
           "stage0_verdict": intake.get("stage0_verdict"),
           "detection_pct": intake.get("pct_probes_detected_p_le_01"),
           "call_rate": intake.get("call_rate"), "lab": (o.get("patient_sky") or {}).get("lab"),
           "lab_zero": o.get("lab_zero"), "scale": o.get("scale"),
           "second_opinion_agreement": (o.get("second_opinion") or {}).get("agreement"),
           "cellular_age_reportable": (o.get("cellular_age") or {}).get("reportable")}
    # Stage 2c, so a cross-sample matrix can ask which runs showed trace material without opening
    # a bundle (2026-09-25)
    for cls, rec in sorted((o.get("classes") or {}).items()):
        row[f"A_abs.{cls}"] = rec.get("A_abs")
        row[f"sky_median_z.{cls}"] = rec.get("sky_median_z")
        row[f"sky_frac_abs_z_gt2.{cls}"] = rec.get("sky_frac_abs_z_gt2")
        row[f"tier.{cls}"] = rec.get("tier")
    for cls, pct in sorted(((o.get("composition") or {}).get("class") or {}).items()):
        row[f"pct.{cls}"] = pct
    for cell, rec in sorted((o.get("cells_all") or {}).items()):
        if isinstance(rec, dict):
            row[f"cellA.{cell}"] = rec.get("A")
            row[f"cellCov.{cell}"] = rec.get("coverage")
    for k, v in ((o.get("intake") or {}).get("covariates") or {}).items():
        row[f"cov.{k}"] = v
    return row


def _plate_dependency_check():
    """The specimen's own sky plate needs matplotlib. Without it the report silently showed only reference
    figures for months (found 2026-09-25), so the run says it out loud."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        print("  WARNING: matplotlib is not installed in this environment, so this specimen's own sky plate "
              "cannot be drawn. The report will say so on the Sky tab. Install it with "
              "`pip install matplotlib` and re-run to get the plate.", flush=True)
        return False


def main():
    ap=argparse.ArgumentParser(description="IDAT pair or beta table -> MethylPhys report")
    ap.add_argument("--grn"); ap.add_argument("--red"); ap.add_argument("--betas")
    ap.add_argument("--age", type=float, help="declared age in years; without it the age term cannot be removed. Float because cohorts publish decimal ages (GEO carries 72.0 and 54.5 as readily as 72)")
    ap.add_argument("--lab", help="(legacy engine only) laboratory / pipeline identity, e.g. GSE87571 for a commissioned one")
    ap.add_argument("--pipeline", default="stage1_noob_450K", help="(legacy engine only) pipeline map name from beta_scale_maps_v1.json")
    ap.add_argument("--specimen", default="whole blood")
    ap.add_argument("--lab-zero", type=float, default=None, help="(legacy engine only) this laboratory's measured zero; omit to read UNSET")
    ap.add_argument("--out", default="methylphys_report.html"); ap.add_argument("--id", default=None)
    ap.add_argument("--bundle", help="also write the full bundle as JSON - every stage's output, for a test harness or an integration that needs more than the report")
    ap.add_argument("--covariate", action="append", default=[], metavar="KEY=VALUE",
                    help="a phenotype or covariate to record with this run, repeatable - e.g. --covariate diagnosis=case --covariate stage=II --covariate cohort=EPIC_Italy. Kept in the custody record and the bundle; never printed in report prose")
    ap.add_argument("--covariates", help="a JSON object of covariates, merged with any --covariate flags")
    ap.add_argument("--no-bundle", action="store_true", help="do not write the machine-readable bundle beside the report (it is written by default: a run that leaves only HTML cannot be pooled later)")
    ap.add_argument("--ledger", default=None, help="append one flat row per run to this JSONL evidence ledger; defaults to evidence_ledger.jsonl beside the report")
    ap.add_argument("--sex", default=None, help="declared sex (F/M); Stage 0.8 compares it with the array")
    ap.add_argument("--patient-id", default=None, help="already-hashed identifier; a cleartext one is hashed here")
    ap.add_argument("--array-type", default=None, choices=("HM450K", "EPIC_v1", "EPIC_v2"), help="declared array type; omit to take it from the IDAT header (Stage 0.1). v3 reads EPIC_v1 only")
    ap.add_argument("--intake-log", default=None, help="append intake and integrity records here")
    ap.add_argument("--manifest-dir", default=None, help="write the immutable per-sample manifest here")
    ap.add_argument("--no-intake", action="store_true", help="skip Stage 0 (recorded in the report as skipped)")
    ap.add_argument("--engine", default="v3", choices=("v3", "legacy"), help="v3 = neutrophil chain (Met-A, C-score, tare; development build). legacy = the class-floor conductor kept for the record")
    ap.add_argument("--slide-ref-A", "--ref-A", dest="slide_ref_A", default=None, help="comma-separated untared Met-A of >= 3 same-run healthy reference arrays (same slide, else same batch) - v3 Stage T median tare, whole blood and isolated neutrophils")
    ap.add_argument("--slide-ref-table", "--ref-table", dest="slide_ref_table", default=None, help="CSV of same-run healthy references with columns A,f_neu,N (optional id): untared Met-A, neutrophil fraction and noise index from pass 1. >= 20 rows -> noise-corrected tare; 3-19 -> median tare")
    ap.add_argument("--pat", default=None, help="v3 Stage Q: a wgbstools .pat / .pat.gz file; read with the loyfer_pat_v1 extractor (stage_q_iam_a.pat_site_table)")
    ap.add_argument("--pat-max-bytes", type=int, default=None, help="read only the first N bytes of --pat (the neutrophil position P was measured on the first 60000000 bytes of each file)")
    ap.add_argument("--site-table", default=None, help="v3 Stage Q: a per-site CSV with columns pos,opp_A,err_A,opp_B,err_B; needs --seq-pipeline")
    ap.add_argument("--seq-pipeline", default=None, help="the read-level pipeline that produced --site-table (required with it); IAM-A is refused unless a position P was frozen for it")
    ap.add_argument("--seq-cell", default="neutrophils", help="cell type of the sequenced specimen (Stage Q)")
    a=ap.parse_args()
    seq = bool(a.pat or a.site_table)
    if not (a.betas or (a.grn and a.red) or seq): ap.error("give --betas, both --grn and --red, or --pat / --site-table")
    if a.pat and a.site_table: ap.error("give --pat or --site-table, not both")
    if a.site_table and not a.seq_pipeline: ap.error("--site-table needs --seq-pipeline: the pipeline that produced the table")
    if seq and a.engine != "v3": ap.error("--pat / --site-table run in the v3 engine only")
    if a.slide_ref_A and a.slide_ref_table: ap.error("give --slide-ref-A or --slide-ref-table, not both")

    intake = None; bead_ok = None
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
        if detected and a.array_type and a.array_type != detected:
            print(f"  declared {a.array_type}, header says {detected} - Stage 0.1 will gate on the mismatch",
                  flush=True)
        declared = a.array_type or detected   # neither -> the manifest lacks array_type and Stage 0.1 quarantines
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
        if _stopped(rec) or (rec.get("integrity_status") not in (None, "INTEGRITY_OK")):
            # a quarantine at 0.1-0.2, or the same bytes already taken in against this intake log (0.3): stop before calibration
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
            # the module is missing, not the data: the gates report DEFERRED and say why; a deferred detection or call rate
            # quarantines at the gate below, before calibration
            print(f"  QC hand-off unavailable ({e}); those gates report DEFERRED", flush=True)
            q = None
            rec["ctrl_qc"] = "DEFERRED_PENDING_STAGE1_DECODER"; rec.setdefault("flags", []).append("CTRL_QC_DEFERRED:no_control_intensities")
            rec = S0.step_0_5_detection_pvalue_qc(rec); rec = S0.step_0_6_bead_count_qc(rec)
            rec = S0.step_0_7_call_rate(rec); rec = S0.step_0_8_sex_check(rec)
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
        # every hard intake failure stops the run here, before calibration (0.7b's 450K coverage needs the calibrated
        # beta and is gated again after Stage 1; on EPIC it is NA_EPIC)
        rec = S0.step_0_7b_platform_coverage(rec, None)
        rec = S0.step_0_9_decision_gate(rec, a.intake_log)
        if rec.get("stage0_verdict") == "QUARANTINE":
            print(f"Stage 0 verdict: QUARANTINE  hard failures: {rec.get('stage0_hard_fail')}", flush=True)
            print("\nQUARANTINE - the chain stops here and nothing is scored. "
                  "Stage 0 rejected the specimen before calibration.", flush=True)
            sys.exit(2)
        if q is not None and q.get("probe_ids") is not None:   # the extracted bead mask, kept for the Stage-1 call rate below
            import pandas as _pdb, numpy as _npb
            bead_ok = _pdb.Series(_npb.asarray(q["bead_counts"]) >= S0.BEAD_COUNT_MIN, index=_pdb.Index(q["probe_ids"]).astype(str))
            bead_ok = bead_ok[~bead_ok.index.duplicated()]
        intake = rec

    stage1_meta=None
    if a.betas:
        import pandas as pd
        d=pd.read_csv(a.betas, index_col=0).iloc[:,0].dropna(); beta=d.to_dict()
        sid=a.id or os.path.basename(a.betas).split(".")[0]
    elif a.grn and a.red:
        try:
            from stage_1_idat_calibration import calibrate_idat_to_beta
        except ImportError:
            for c in (os.path.join(BIO,"MethylPhys/chain"), os.path.join(BIO,"MethylPhys","Reproduction_Kit")):
                sys.path.insert(0,c)
            from stage_1_idat_calibration import calibrate_idat_to_beta
        print("Stage 1: calibrating the IDAT pair (methylprep noob) - about 25 s", flush=True)
        b,meta=calibrate_idat_to_beta(a.grn,a.red,return_mask=True)
        b=b.iloc[:,0] if hasattr(b,"columns") else b
        beta=b.dropna().to_dict(); sid=a.id or os.path.basename(a.grn).split("_")[0]
        stage1_meta=meta
        det=meta.get("detection") or {}
        if det.get("detection_available"):
            print(f"Stage 1: {det['n_detected']:,} of {det['n_probes']:,} probes detected ({det['pct_detected']*100:.2f} %); {det['n_masked']:,} at background removed", flush=True)
        if intake is not None:
            # Stage 1's own numbers are recorded beside the Stage 0 record, never over it: the gate above was decided on the
            # hand-off values (0.4 controls per matched pair, 0.5 detection p <= 0.01 against the negative-control background,
            # 0.6 bead counts, 0.7 call rate on both). Stage 1 adds: control medians, poobah (p <= 0.05) detection, and a call rate
            # on poobah AND the extracted bead mask. Recorded, not gated.
            import stage_0_intake as S0, numpy as _np
            cv = S0.validate_control_probes(meta.get("controls") or {})
            s1 = {"ctrl_qc": cv["ctrl_qc"], "ctrl_metrics": cv["metrics"], "ctrl_flags": cv["flags"], "detection_statistic": "poobah p <= 0.05"}
            intake["controls"] = meta.get("controls")
            dm = meta.get("_detected_mask")
            if det.get("detection_available") and dm is not None:
                v = S0.validate_detection_p(_np.where(dm.to_numpy(bool), 0.0, 1.0))
                s1.update(detection_qc=v["detection_qc"], pct_probes_detected=v["pct_probes_detected_p_le_01"], n_probes=int(len(dm)))
                if bead_ok is not None:
                    common = dm.index.intersection(bead_ok.index)
                    s1["n_probes_bead_aligned"] = int(len(common))
                    if len(common) >= 0.5 * len(dm):
                        cr = S0.validate_call_rate(int((dm.loc[common].to_numpy(bool) & bead_ok.loc[common].to_numpy(bool)).sum()), int(len(common)))
                        s1.update(call_rate=cr["call_rate"], call_rate_status=cr["call_rate_status"])
                    else:
                        s1["call_rate_note"] = "bead mask and Stage-1 probes did not align (< 50 % shared probe names): call rate not computed"
                else:
                    s1["call_rate_note"] = "no extracted bead mask: call rate not computed"
            else:
                s1["detection_qc"] = "NOT_AVAILABLE (no poobah column from Stage 1)"
            intake["stage1_qc"] = s1
            for k in ("detection_qc", "call_rate_status", "ctrl_qc"):
                if str(s1.get(k, "")).startswith(("FAIL", "CALL_RATE_FAIL")):
                    intake.setdefault("flags", []).append(f"STAGE1_{k.upper()}:{s1[k]} (recorded, not gated)")
    else:
        beta = None; sid = a.id or os.path.basename(a.pat or a.site_table).split(".")[0]

    if intake is not None:
        import stage_0_intake as S0
        ref = None
        try:
            import json as _json, glob as _glob
            f = _glob.glob(os.path.join(BIO, "MethylPhys/chain/**/iamatlas_gauge_identity_loci_v1_0.json"),
                           recursive=True)
            if a.engine == "v3":   # v3: coverage of the neutrophil identity sites the reading uses
                fl = _json.load(open(os.path.join(BIO, "MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json")))
                st = set(fl["platforms"].get("EPIC", {}).get("neutrophils", {}).get("sites", []))
                ref = len(st & set(beta)) / max(len(st), 1); f = []
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

    if a.engine == "v3":
        import pandas as _pd, json as _json, datetime as _dt, conductor_v3 as C3, report_v3 as R3
        refs = [float(x) for x in a.slide_ref_A.split(",") if x.strip()] if a.slide_ref_A else None
        if a.slide_ref_table:
            rt = _pd.read_csv(a.slide_ref_table)
            if "A" not in rt.columns: sys.exit(f"--slide-ref-table {a.slide_ref_table}: needs a column A (and f_neu, N for the noise-corrected tare)")
            refs = [{k: (None if _pd.isna(r.get(k)) else r.get(k)) for k in ("A", "f_neu", "N", "id", "gsm") if k in rt.columns} for r in rt.to_dict("records")]
        it = intake or {}
        array_type = it.get("array_type_detected") or it.get("array_type") or a.array_type   # header first, then declared
        if beta is not None:
            o = C3.run_neutrophil(_pd.Series(beta, dtype="float64"), specimen=a.specimen, ref_A=refs, array_type=array_type, sample_id=sid)
        else:
            o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "neutrophils only", "note": "sequencing input only: no array reading"}
        if seq:   # Stage Q - IAM-A from single-molecule reads
            import stage_q_iam_a as Q
            if a.pat:
                if a.seq_pipeline and a.seq_pipeline != Q.PAT_PIPELINE:
                    sys.exit(f"--pat is read by the {Q.PAT_PIPELINE} extractor; --seq-pipeline {a.seq_pipeline} does not apply")
                print(f"Stage Q: extracting per-site copy-error counts from {os.path.basename(a.pat)} ({Q.PAT_PIPELINE})", flush=True)
                T = Q.pat_site_table(a.pat, max_bytes=a.pat_max_bytes); pipe = Q.PAT_PIPELINE
                src = {"pat": os.path.abspath(a.pat), "max_bytes": a.pat_max_bytes, **{k: T.attrs.get(k) for k in ("n_lines", "n_qualifying_lines", "n_molecules")}}
            else:
                T = _pd.read_csv(a.site_table); pipe = a.seq_pipeline; src = {"site_table": os.path.abspath(a.site_table)}
            o["iam_a"] = Q.read(T, cell=a.seq_cell, pipeline=pipe); o["iam_a"]["input"] = src
        cov = _covariates(a)
        o["intake"] = intake; o["intake_skipped"] = bool(a.no_intake); o["sample_id"] = sid; o["covariates"] = cov
        if intake is not None: intake.setdefault("covariates", {}).update(cov)
        if stage1_meta: o["stage1"] = {k: stage1_meta.get(k) for k in ("detection", "n_cpgs", "pipeline")}
        led = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "evidence_ledger.jsonl")
        o["run_id"] = _assign_run_id(led)
        r = R3.build(o, a.out, sid)
        m = o.get("met_a") or {}; t = o.get("tare") or {}; q_ = o.get("iam_a") or {}
        print(f"\n{sid}: neutrophil Met-A {m.get('A')} ({m.get('state', m.get('reason', o.get('refusal')))}) | C {(o.get('met_a_cscore') or {}).get('C')} | "
              f"tare {t.get('A_rel')}" + (f" | IAM-A {q_.get('A')} ({q_.get('state', q_.get('refusal'))})" if q_ else ""))
        if not a.no_bundle:
            bundle_path = a.bundle or (os.path.splitext(a.out)[0] + "_bundle.json")
            _json.dump(o, open(bundle_path, "w"), default=str)
            row = {"run_id": o["run_id"], "engine": "v3", "sample_id": sid, "utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                   "report": os.path.abspath(a.out), "bundle": os.path.abspath(bundle_path), "specimen": a.specimen,
                   "platform": o.get("platform"), "array_type": o.get("array_type"), "refusal": o.get("refusal"),
                   "floors_version": o.get("floors_version"), "reference_version": o.get("reference_version"),
                   "stage0_verdict": it.get("stage0_verdict"), "call_rate_status": it.get("call_rate_status"),
                   "f_neu": m.get("fraction"), "A": m.get("A"), "state": m.get("state", m.get("reason")), "n_sites": m.get("n_sites"),
                   "shift_per_1pct_loss": m.get("shift_per_1pct_loss"), "past_entropy_ceiling": m.get("past_entropy_ceiling"),
                   "C": (o.get("met_a_cscore") or {}).get("C"), "A_rel": t.get("A_rel"), "tare": t.get("state", t.get("reason")),
                   "n_refs": t.get("n_refs"), "tare_method": t.get("method"), "noise_index": m.get("noise_index"),
                   "detection_limit_pct_loss": t.get("detection_limit_pct_loss"),
                   "iam_a": q_.get("A"), "iam_a_pipeline": q_.get("pipeline"), "covariates": cov}
            with open(led, "a", encoding="utf-8") as fh: fh.write(_json.dumps(row, default=str) + "\n")
            print(f"report: {r['out']} | bundle: {bundle_path} | evidence ledger: {led} (one row appended)")
        else:
            print(f"report: {r['out']} (no bundle, no ledger row: --no-bundle)")
        return

    import cpg_conductor as C, build_methylphys as B
    cfg={"age":a.age,"pipeline":a.pipeline,"lab_zero":a.lab_zero,"substrate":substrate_token(a.specimen), "substrate_as_declared":a.specimen,
         "intake": intake, "intake_skipped": bool(a.no_intake)}
    if a.lab: cfg["lab"]=a.lab
    if stage1_meta and stage1_meta.get("snp_noise"): cfg["snp_noise"]=stage1_meta["snp_noise"]
    print(f"running the chain on {len(beta):,} CpGs", flush=True)
    o=C.run_full(beta, _atlas(), cfg=cfg)
    o["intake"] = intake          # the report prints the Stage 0 record, or NOT RUN when there is none
    o["intake_skipped"] = bool(a.no_intake)
    # 2026-09-27: a run's intake is VERIFIED only when Stage 1 handed Stage 0 the array's own detection numbers and the gate ran.
    o["intake_verified"] = bool(intake is not None and (intake.get("call_rate_status") or "").startswith(("PASS","CALL_RATE_BORDERLINE")))
    o["intake_note"] = (None if o["intake_verified"] else
                        ("betas-only input: the array's detection and control probes were not available, so intake was not verified" if a.betas else
                         "intake skipped (--no-intake)" if a.no_intake else "intake did not verify this array"))
    if stage1_meta: o["stage1"] = {k: stage1_meta.get(k) for k in ("detection","controls","snp_noise","n_cpgs","pipeline")}
    # Everything a disease matrix will need from this specimen, captured at the moment of the run: the
    # phenotype it was declared with, and every input version that could change the reading. A run that
    # records neither cannot be pooled with one from another month (2026-09-23).
    cov = _covariates(a)
    # --no-intake leaves this None, and a run without a custody record must still record its covariates
    intake = intake if isinstance(intake, dict) else {}
    intake.setdefault("covariates", {}).update(cov)
    _plate_dependency_check()
    o["intake"] = intake
    _led = a.ledger or (os.path.splitext(a.out)[0].rsplit("/", 1)[0] + "/evidence_ledger.jsonl")
    o["run_id"] = _assign_run_id(_led)
    o.setdefault("context", {})["report_path"] = os.path.abspath(a.out); o["context"]["sample_id"] = sid   # so the report prints its own filing plan (2026-09-26)
    o["versions"] = _versions(os.path.dirname(os.path.abspath(__file__)) + "/..")
    _class_z(o, os.path.dirname(os.path.abspath(__file__)) + "/..")
    if cov:
        print(f"covariates recorded: {cov}", flush=True)
    r=B.build(o, a.out, sid)
    # The bundle is written by default: a run that leaves only a 6 MB HTML cannot be pooled later.
    bundle_path = a.bundle or (os.path.splitext(a.out)[0] + "_bundle.json")
    if not a.no_bundle:
        import json as _json
        _json.dump(o, open(bundle_path, "w"), default=str)
        print(f"bundle: {bundle_path}", flush=True)
        led = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "evidence_ledger.jsonl")
        with open(led, "a", encoding="utf-8") as f:
            f.write(_json.dumps(_ledger_row(o, sid, a.out), default=str) + "\n")
        print(f"evidence ledger: {led} (one row appended)", flush=True)
    imm=o["classes"].get("immune",{})
    print(f"\n{sid}: immune A'' {imm.get('A_abs')}  placement {imm.get('placement')}  tier {imm.get('tier')}")
    print(f"refusals: {len(r['refusals'])}" + (f" -> {r['refusals'][:3]}" if r["refusals"] else ""))
    print(f"report: {r['out']}  ({r['bytes']//1000} KB)")

if __name__=="__main__": main()
