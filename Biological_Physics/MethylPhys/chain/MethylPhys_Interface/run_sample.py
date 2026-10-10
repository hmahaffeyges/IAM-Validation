#!/usr/bin/env python3
"""run_sample.py - one command: IDAT pair (or a beta table, or single-molecule reads) to a MethylPhys report.

Chain v3 (DEVELOPMENT - not commissioned; neutrophils, EPIC v1) - the only engine: Stage 0 intake -> Stage 1 IDAT calibration -> conductor_v3
(platform check, Stage A composition, Stage M Met-A, Stage MC C-score, Stage T same-run tare) -> report_v3, plus Stage Q IAM-A
when single-molecule input is given.

    # an Illumina EPIC v1 IDAT pair (needs methylprep for Stage 1); --sex and --age are optional (recorded when given)
    python3 run_sample.py --grn S_Grn.idat.gz --red S_Red.idat.gz --specimen "whole blood" --id S001 --out S001.html

    # specimens: whole blood, isolated / sorted / purified neutrophils. Any other specimen is refused at intake with a report.
    # identifiers: the bundle and the ledger carry the sha256 hash of --id (and of --patient-id); the printed report keeps the typed id.
    # development flags (DEVELOPMENT - not commissioned; written under bundle["development"], never part of the reading):
    #   --dev-direction --dev-trace --dev-foreign --dev-brightness --dev-nilc --dev-atlas-e --dev-percell-b --dev-sky
    #   --dev-selftare-ii is a no-op alias (2026-10-04): self-tare II is Stage T step 1 and always runs; the flag only copies its record
    #   to bundle["development"]["selftare_ii"], so recorded commands still run
    #   (--dev-nilc/--dev-atlas-e/--dev-percell-b/--dev-sky need --atlas-v2 <IAMAtlas_v2.parquet>; --dev-sky needs healpy)
    #   --dev-epic-v2 (an EPIC v2 IDAT pair through SeSAMe; needs --sesame-rscript <Rscript of an env with sesame>)

    # pass 2 of a batch: the same, with the same-run healthy references from pass 1
    python3 run_sample.py ... --slide-ref-table refs.csv        # column A (optional id): same-run healthy references; >= 3 rows -> median tare
    python3 run_sample.py ... --slide-ref-A 0.951,0.957,0.962   # plain A values -> median tare

    # a beta table already calibrated by this chain's Stage 1: two-column CSV cpg_id,beta (Stage 0 does not run)
    python3 run_sample.py --betas S.csv --id S001 --out S001.html

    # IAM-A from a wgbstools .pat file (pipeline loyfer_pat_v1), or from a per-site table with its pipeline named
    python3 run_sample.py --pat S.pat.gz --id S001 --out S001.html
    python3 run_sample.py --site-table S_sites.csv --seq-pipeline loyfer_pat_v1 --id S001 --out S001.html

The class-floor engine (v2) was retired on 2026-10-03 and is archived privately; chain v3 is the only engine.
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
for d in (ENG, HERE): sys.path.insert(0,d)

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
    for pat in ("Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json", "Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv",
                "Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json", "Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json",
                "Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json", "Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json",
                "Runtime Matrices/IAM_A_Positions/iama_positions_v2.json", "Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json",
                "Runtime Matrices/Intake/intake_thresholds_v1.json",
                "conductor_v3.py", "stage_m_met_a.py", "stage_q_iam_a.py", "stage_q0_intake.py", "stage_0_intake.py", "stage_0_1_qc_handoff.py",
                "stage_1_idat_calibration.py", "MethylPhys_Interface/report_v3.py", "MethylPhys_Interface/run_sample.py"):
        for p in _glob.glob(os.path.join(chain_dir, pat), recursive=True):
            files[os.path.relpath(p, chain_dir)] = {"sha256_12": _sha12(p), "bytes": os.path.getsize(p)}
    out["inputs"] = files
    return out


def _hash_id(x):
    """sha256 (first 32 hex) of a typed identifier; an identifier that is already a hash (>= 16 alphanumerics) is kept (SOP 12)."""
    import hashlib as _h
    x = str(x); y = x.replace("_", "").replace("-", "")
    return x if (len(y) >= 16 and y.isalnum() and all(c in "0123456789abcdef" for c in y.lower())) else _h.sha256(x.encode()).hexdigest()[:32]


def _redact(obj, typed, hid):
    """Every whole-token occurrence of the typed id in strings of obj replaced by its hash (decision B: no typed id in bundle or ledger)."""
    import re as _re
    if not typed or typed == hid: return obj
    rx = _re.compile(r"(?<![A-Za-z0-9])" + _re.escape(str(typed)) + r"(?![A-Za-z0-9])")
    def f(v):
        if isinstance(v, str): return rx.sub(hid, v)
        if isinstance(v, dict): return {f(k) if isinstance(k, str) else k: f(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [f(x) for x in v]
        return v
    return f(obj)


def _dev_run(a, o, beta, intake):
    """Development flags (doors/DEV_FLAGS_01.md): each writes bundle['development'][stage], labelled DEVELOPMENT - not commissioned.
    None of them changes the reading, the gauge or the tare."""
    flags = [k for k in ("selftare_ii", "direction", "trace", "foreign", "brightness", "nilc", "atlas_e", "percell_b", "sky") if getattr(a, "dev_" + k, False)]
    if a.dev_epic_v2: flags.append("epic_v2")
    if not flags: return
    import pandas as _pd, dev_stages as DV, traceback as _tb
    dev = o.setdefault("development", {"label": DV.DEV_LABEL, "flags": flags})
    if "epic_v2" in flags:
        if not (a.grn and a.red): dev["epic_v2"] = {"label": DV.DEV_LABEL, "status": "NOT_RUN", "reason": "needs an IDAT pair"}
        elif not a.sesame_rscript: dev["epic_v2"] = {"label": DV.DEV_LABEL, "status": "NOT_RUN", "reason": "--sesame-rscript not given"}
        else:
            try:
                import conductor_v3 as C3
                bv, meta = DV.epicv2_calibrate(a.grn, a.red, a.sesame_rscript)
                rd = C3.run_neutrophil(bv, specimen=a.specimen, ref_A=None, array_type=None)
                dev["epic_v2"] = {"label": DV.DEV_LABEL, "status": "OK", "calibrator": meta, "reading_cross_version": {k: rd.get(k) for k in ("composition", "met_a", "tare")},
                                  "note": "EPIC v2 betas (SeSAMe) read against the EPIC v1 floor and profiles at the shared sites; no EPIC v2 neutrophil floor exists; no gauge"}
                if beta is None: beta = bv
            except Exception as e:
                dev["epic_v2"] = {"label": DV.DEV_LABEL, "status": "ERROR", "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    if beta is None:
        for k in flags:
            dev.setdefault(k, {"label": DV.DEV_LABEL, "status": "NOT_RUN", "reason": "no beta vector on this specimen"})
        return
    b = _pd.Series(beta, dtype="float64") if not isinstance(beta, _pd.Series) else beta
    b.index = b.index.astype(str); comp = o.get("composition") or {}; fr = comp.get("fractions"); sp = a.specimen
    st1 = (o.get("tare") or {}).get("selftare_ii") or {}
    calls = {"selftare_ii": lambda: DV._rec("selftare_ii", status=st1.get("status", "NOT_RUN"), A_selftared=(o.get("met_a") or {}).get("A"),
                                            **({k: v for k, v in st1.items() if k not in ("step", "status", "note")} if st1 else {"reason": "Stage T did not run on this specimen"}),
                                            note="no-op alias: self-tare II is Stage T step 1 since 2026-10-04 and already ran (tare.selftare_ii); "
                                                 "A_selftared repeats met_a.A; nothing is recomputed"),
             "direction": lambda: DV.direction_record(b, sp, comp),
             "trace": lambda: DV.trace_cell(b), "foreign": lambda: DV.foreign_cell(b), "brightness": lambda: DV.brightness(b, sp, comp),
             "nilc": lambda: DV.nilc_e(b, a.atlas_v2), "atlas_e": lambda: DV.atlas_e(b, a.atlas_v2), "percell_b": lambda: DV.percell_b(b, fr, a.atlas_v2),
             "sky": lambda: DV.sky(b, fr, a.atlas_v2)}
    for k in flags:
        if k == "epic_v2": continue
        try: dev[k] = calls[k]()
        except Exception as e: dev[k] = {"label": DV.DEV_LABEL, "status": "ERROR", "reason": f"{type(e).__name__}: {str(e)[:300]}", "tb": _tb.format_exc()[-600:]}


def main():
    ap=argparse.ArgumentParser(description="IDAT pair or beta table -> MethylPhys report")
    ap.add_argument("--grn"); ap.add_argument("--red"); ap.add_argument("--betas")
    ap.add_argument("--age", type=float, help="declared age in years (optional; recorded when given, read by no v3 stage). Float: records carry 72.0 and 54.5 as readily as 72")
    ap.add_argument("--specimen", default="whole blood")
    ap.add_argument("--out", default="methylphys_report.html"); ap.add_argument("--id", default=None)
    ap.add_argument("--bundle", help="also write the full bundle as JSON - every stage's output, for a test harness or an integration that needs more than the report")
    ap.add_argument("--covariate", action="append", default=[], metavar="KEY=VALUE",
                    help="a phenotype or covariate to record with this run, repeatable - e.g. --covariate diagnosis=case --covariate stage=II --covariate cohort=EPIC_Italy. Kept in the custody record and the bundle; never printed in report prose")
    ap.add_argument("--covariates", help="a JSON object of covariates, merged with any --covariate flags")
    ap.add_argument("--no-bundle", action="store_true", help="do not write the machine-readable bundle beside the report (it is written by default: a run that leaves only HTML cannot be pooled later)")
    ap.add_argument("--ledger", default=None, help="append one flat row per run to this JSONL evidence ledger; defaults to evidence_ledger.jsonl beside the report")
    ap.add_argument("--sex", default=None, help="declared sex (F/M), optional; when given Stage 0.8 compares it with the array, otherwise the array's sex is recorded")
    ap.add_argument("--patient-id", default=None, help="already-hashed identifier; a cleartext one is hashed here")
    ap.add_argument("--array-type", default=None, choices=("HM450K", "EPIC_v1", "EPIC_v2"), help="declared array type; omit to take it from the IDAT header (Stage 0.1). v3 reads EPIC_v1 only")
    ap.add_argument("--intake-log", default=None, help="append intake and integrity records here")
    ap.add_argument("--manifest-dir", default=None, help="write the immutable per-sample manifest here")
    ap.add_argument("--no-intake", action="store_true", help="skip Stage 0 (recorded in the report as skipped)")
    ap.add_argument("--engine", default="v3", choices=("v3",), help="v3 is the only engine (the class-floor engine was retired 2026-10-03); the flag is kept so recorded v3 commands still run")
    ap.add_argument("--slide-ref-A", "--ref-A", dest="slide_ref_A", default=None, help="comma-separated Met-A before the median tare (met_a.A, self-tared since 2026-10-04) of >= 3 same-run healthy reference arrays (same slide, else same batch) - v3 Stage T median tare, whole blood and isolated neutrophils")
    ap.add_argument("--slide-ref-table", "--ref-table", dest="slide_ref_table", default=None, help="CSV of same-run healthy references with column A (optional id): Met-A from pass 1 (met_a.A, self-tared since 2026-10-04; before the median tare). >= 3 rows -> median tare (nothing is fitted)")
    ap.add_argument("--pat", default=None, help="v3 Stage Q: a wgbstools .pat / .pat.gz file; read with the loyfer_pat_v1 extractor (stage_q_iam_a.pat_site_table)")
    ap.add_argument("--pat-max-bytes", type=int, default=None, help="development: read only the first N bytes of --pat; IAM-A is then refused (P is measured on whole files) unless --dev-allow-partial")
    ap.add_argument("--alignment-qc", default=None, help="Stage Q0: JSON from the alignment step with conversion_rate and/or duplicate_fraction")
    ap.add_argument("--iama-ref-table", default=None, help="CSV of same-run healthy references for IAM-A (column A, optional id, pipeline): same laboratory, kit and pipeline; >= 3 rows -> IAM-A median tare")
    ap.add_argument("--dev-allow-partial", action="store_true", help="development only: read IAM-A on a --pat-max-bytes cut (e.g. to reproduce position v1)")
    ap.add_argument("--site-table", default=None, help="v3 Stage Q: a per-site CSV with columns pos,opp_A,err_A,opp_B,err_B; needs --seq-pipeline")
    ap.add_argument("--seq-pipeline", default=None, help="the read-level pipeline that produced --site-table (required with it); IAM-A is refused unless a position P was frozen for it")
    ap.add_argument("--seq-cell", default="neutrophils", help="cell type of the sequenced specimen (Stage Q)")
    ap.add_argument("--prior-betas", default=None, help="stage 12b (difference map, DEV-TOOLKIT-ADDED-01): the Stage 1 beta vector of an earlier draw of the same person (as written by --save-betas); needs --prior-bundle")
    ap.add_argument("--prior-bundle", default=None, help="stage 12b: the bundle of that earlier draw; the difference is refused unless both draws carry the same identifier hash, array type and pipeline")
    ap.add_argument("--save-betas", default=None, help="also write this specimen's Stage 1 beta vector (after the detection mask) to this path as a two-column file cpg_id,beta (.parquet or .csv); recorded in the bundle")
    ap.add_argument("--dev-selftare-ii", action="store_true", help="no-op alias, kept so recorded commands still run: self-tare II (DEV-SELFTARE-02) is Stage T step 1 since 2026-10-04 "
                    "and always runs (bundle tare.selftare_ii); the flag only copies that record to bundle['development']['selftare_ii'] and changes nothing else")
    for f, h in (("direction", "DEV-DIRECTION-02 signed move at the identity sites"),
                 ("trace", "DEV-TOOLKIT-ADDED-02 3b trace cell (isolated neutrophils)"), ("foreign", "DEV-TOOLKIT-ADDED-02 3c foreign cell (whole blood)"),
                 ("brightness", "DEV-TOOLKIT-ADDED-02 11b interval on Met-A"), ("nilc", "stage 4 NILC-e (needs --atlas-v2)"), ("atlas-e", "stage 3 atlas_e (needs --atlas-v2)"),
                 ("percell-b", "stage 5 B cells, development floor (needs --atlas-v2)"), ("sky", "stages 11-12 sky with the block-shuffle null (needs --atlas-v2, healpy)"),
                 ("epic-v2", "DEV-EPIC-V2-01: read an EPIC v2 IDAT pair through SeSAMe (needs --sesame-rscript)")):
        ap.add_argument(f"--dev-{f}", action="store_true", help=f"DEVELOPMENT - not commissioned: {h}; written under bundle['development'], not part of the reading")
    ap.add_argument("--atlas-v2", default=os.environ.get("CPG_ATLAS_V2_PARQUET"), help="atlas v2 parquet for the atlas-based development flags (not stored in the repository)")
    ap.add_argument("--sesame-rscript", default=os.environ.get("METHYLPHYS_SESAME_RSCRIPT"), help="Rscript of an environment with Bioconductor sesame (--dev-epic-v2)")
    a=ap.parse_args()
    seq = bool(a.pat or a.site_table)
    if not (a.betas or (a.grn and a.red) or seq): ap.error("give --betas, both --grn and --red, or --pat / --site-table")
    if a.pat and a.site_table: ap.error("give --pat or --site-table, not both")
    if a.site_table and not a.seq_pipeline: ap.error("--site-table needs --seq-pipeline: the pipeline that produced the table")
    if a.slide_ref_A and a.slide_ref_table: ap.error("give --slide-ref-A or --slide-ref-table, not both")

    intake = None; bead_ok = None; platform_stop = None; specimen_stop = None
    try:
        import stage_0_intake as _S0spec
    except ImportError:
        sys.path.insert(0, os.path.join(BIO, "MethylPhys/chain")); import stage_0_intake as _S0spec
    if not seq or a.betas or (a.grn and a.red):
        specimen_stop = _S0spec.specimen_refusal(a.specimen)   # author decision L (2026-10-04): refused at intake, before anything is read
        if specimen_stop:
            print("  SPECIMEN_REFUSED - " + specimen_stop, flush=True)
            intake = {"status": "SPECIMEN_REFUSED", "stage0_verdict": "REFUSED_SPECIMEN", "substrate": a.specimen,
                      "flags": [f"SPECIMEN_REFUSED:{_S0spec.normalise_specimen(a.specimen) or 'not stated'}"], "declared_sex": a.sex,
                      "declared_chronological_age": float(a.age) if a.age is not None else None}
    if a.grn and a.red and not a.no_intake and not specimen_stop:
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
        if (rec.get("array_type_detected") or declared) == "EPIC_v2":
            # unhandled platform (2026-10-03, DEV-BASE-CHAIN-01): an EPIC v2 array is refused here, before the QC decode, instead of being
            # judged by EPIC v1 rules (sex mismatch) or crashing in Stage 1. v3 reads EPIC v1 only.
            platform_stop = ("EPIC v2 array (IDAT header): chain v3 reads EPIC v1 arrays only - no frozen neutrophil floor for EPIC v2, and Stage 1 "
                             "(methylprep 1.7.1) does not calibrate EPIC v2. Intake stopped before the QC decode; nothing is scored.")
            rec["status"] = "PLATFORM_REFUSED"; rec.setdefault("flags", []).append("PLATFORM_REFUSED:EPIC_v2"); rec["stage0_verdict"] = "REFUSED_PLATFORM"
            print("  PLATFORM_REFUSED - " + platform_stop, flush=True)
            intake = rec
        if platform_stop is None:
            try:
                from stage_0_1_qc_handoff import decode_qc_inputs, _manifest as _qc_manifest
            except ImportError:
                decode_qc_inputs = None
            if decode_qc_inputs is not None:
                # the machine, not the array: the array-type manifest must load before the IDATs are judged (2026-10-03). A failure here
                # is an environment failure with its own status and exit code 3, never QUARANTINE_CORRUPT_IDAT.
                try:
                    _qc_manifest(rec.get("array_type_detected") or declared)
                except Exception as e:
                    rec["status"] = "ENVIRONMENT_MISSING_MANIFEST"
                    rec.setdefault("flags", []).append(f"ENVIRONMENT_FAILURE:{type(e).__name__}")
                    print(f"  ENVIRONMENT_MISSING_MANIFEST - the methylprep manifest for {rec.get('array_type_detected') or declared} could not be "
                          f"loaded on this machine ({type(e).__name__}: {str(e)[:200]}). The array is not judged; fix the environment and re-run.", flush=True)
                    sys.exit(3)
            try:
                if decode_qc_inputs is None: raise ImportError("stage_0_1_qc_handoff not importable")
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
    if specimen_stop:
        beta = None; sid = a.id or os.path.basename(a.grn or a.betas or "specimen").split("_")[0].split(".")[0]
    elif a.betas:
        import pandas as pd
        d=pd.read_csv(a.betas, index_col=0).iloc[:,0].dropna(); beta=d.to_dict()
        sid=a.id or os.path.basename(a.betas).split(".")[0]
    elif a.grn and a.red and platform_stop:
        beta = None; sid = a.id or os.path.basename(a.grn).split("_")[0]
    elif a.grn and a.red:
        try:
            from stage_1_idat_calibration import calibrate_idat_to_beta
        except ImportError:
            for c in (os.path.join(BIO,"MethylPhys/chain"), os.path.join(BIO,"MethylPhys","Reproduction_Kit")):
                sys.path.insert(0,c)
            from stage_1_idat_calibration import calibrate_idat_to_beta
        print("Stage 1: calibrating the IDAT pair (methylprep noob) - about 25 s", flush=True)
        sid=a.id or os.path.basename(a.grn).split("_")[0]
        try:
            b,meta=calibrate_idat_to_beta(a.grn,a.red,return_mask=True)
        except ValueError as e:
            if "Unknown array type" not in str(e): raise
            b = meta = None   # an array type methylprep cannot calibrate (EPIC v2 read without intake): refused, not a crash
            platform_stop = f"array type not calibrated by Stage 1 ({e}): chain v3 reads EPIC v1 arrays only; nothing is scored."
            print("  PLATFORM_REFUSED - " + platform_stop, flush=True)
        except (EOFError, OSError) as e:
            # the IDAT bytes could not be read (truncated or corrupt file): the specimen's files, not the chain - a named stop, not a traceback
            print(f"  QUARANTINE_CORRUPT_IDAT - Stage 1 could not read the IDAT pair: {type(e).__name__}: {e}", flush=True)
            print("\nQUARANTINE - the chain stops here and nothing is scored.", flush=True)
            sys.exit(2)
        if b is None:
            beta = None
        else:
            b=b.iloc[:,0] if hasattr(b,"columns") else b
            beta=b.dropna().to_dict()
            stage1_meta=meta
        det=(meta or {}).get("detection") or {}
        if det.get("detection_available"):
            print(f"Stage 1: {det['n_detected']:,} of {det['n_probes']:,} probes detected ({det['pct_detected']*100:.2f} %); {det['n_masked']:,} at background removed", flush=True)
        if intake is not None and meta is not None:
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

    if intake is not None and not platform_stop and not specimen_stop:
        import stage_0_intake as S0
        ref = None
        try:
            import json as _json
            # coverage of the neutrophil identity sites the v3 reading uses
            fl = _json.load(open(os.path.join(BIO, "MethylPhys/chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json")))
            st = set(fl["platforms"].get("EPIC", {}).get("neutrophils", {}).get("sites", []))
            ref = len(st & set(beta)) / max(len(st), 1)
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

    import pandas as _pd, json as _json, datetime as _dt, conductor_v3 as C3, report_v3 as R3
    refs = [float(x) for x in a.slide_ref_A.split(",") if x.strip()] if a.slide_ref_A else None
    if a.slide_ref_table:
        rt = _pd.read_csv(a.slide_ref_table)
        if "A" not in rt.columns: sys.exit(f"--slide-ref-table {a.slide_ref_table}: needs a column A")
        refs = [{k: (None if _pd.isna(r.get(k)) else r.get(k)) for k in ("A", "f_neu", "N", "id", "gsm") if k in rt.columns} for r in rt.to_dict("records")]
    it = intake or {}
    array_type = it.get("array_type_detected") or it.get("array_type") or a.array_type   # header first, then declared
    if specimen_stop:
        o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "neutrophils only", "array_type": array_type, "refusal": specimen_stop,
             "refusal_code": "SPECIMEN_REFUSED"}
    elif platform_stop:
        o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "neutrophils only", "platform": "EPIC_v2" if "EPIC v2" in platform_stop else None,
             "array_type": array_type, "refusal": platform_stop}
    elif beta is not None:
        o = C3.run_neutrophil(_pd.Series(beta, dtype="float64"), specimen=a.specimen, ref_A=refs, array_type=array_type, sample_id=sid)
    else:
        o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "neutrophils only", "note": "sequencing input only: no array reading"}
    if seq:   # Stage Q - IAM-A from single-molecule reads
        import stage_q_iam_a as Q
        if a.pat:
            if a.seq_pipeline and a.seq_pipeline != Q.PAT_PIPELINE:
                sys.exit(f"--pat is read by the {Q.PAT_PIPELINE} extractor; --seq-pipeline {a.seq_pipeline} does not apply")
            import stage_q0_intake as Q0
            print(f"Stage Q0: intake checks on {os.path.basename(a.pat)}", flush=True)
            o["iam_a_intake"] = Q0.intake(a.pat, specimen=a.specimen, cell=a.seq_cell, alignment_qc=a.alignment_qc)
            print(f"Stage Q: extracting per-site copy-error counts from {os.path.basename(a.pat)} ({Q.PAT_PIPELINE})", flush=True)
            T = Q.pat_site_table(a.pat, max_bytes=a.pat_max_bytes); pipe = Q.PAT_PIPELINE
            src = {"pat": os.path.abspath(a.pat), "max_bytes": a.pat_max_bytes, **{k: T.attrs.get(k) for k in ("n_lines", "n_qualifying_lines", "n_molecules")}}
        else:
            T = _pd.read_csv(a.site_table); pipe = a.seq_pipeline; src = {"site_table": os.path.abspath(a.site_table)}
        qi = o.get("iam_a_intake") or {}
        if qi.get("refusal_code"):
            o["iam_a"] = {"stage": "Q", "reading": "IAM-A", "cell": a.seq_cell, "A": None, "refusal_code": qi["refusal_code"],
                          "refusal": f"Stage Q0 stopped the file: {qi['refusal']}", "input": src}
        else:
            o["iam_a"] = Q.read(T, cell=a.seq_cell, pipeline=pipe, allow_partial=bool(getattr(a, "dev_allow_partial", False))); o["iam_a"]["input"] = src
            if getattr(a, "iama_ref_table", None):
                _rt = _pd.read_csv(a.iama_ref_table)
                if "A" not in _rt.columns: sys.exit(f"--iama-ref-table {a.iama_ref_table}: needs a column A")
                _rr = [{k: (None if _pd.isna(r.get(k)) else r.get(k)) for k in ("A", "id", "pipeline") if k in _rt.columns} for r in _rt.to_dict("records")]
                o["iam_a"]["tare"] = Q.tare(o["iam_a"], _rr, sample_id=a.id)
            if not a.pat: o["iam_a"]["intake"] = "Stage Q0 not run: a site table carries no molecules to check"
    # what stage 12b needs from every draw (recorded on every run so a later draw can be compared): the identifier hash and the pipeline
    if a.betas: o["pipeline"] = "chain Stage 1 (beta table)"
    if a.betas and a.patient_id: o["patient_hash"] = a.patient_id if (len(a.patient_id) >= 16 and a.patient_id.isalnum()) else __import__("hashlib").sha256(a.patient_id.encode()).hexdigest()[:32]
    if a.prior_betas:   # stage 12b - difference map (per-address difference of two draws of one person; the sky drawing is not built)
        import serial_mode as SMd
        def _ctx(b, it, pipe):
            return {"context": {"patient_hash": (it or {}).get("patient_id") or b.get("patient_hash"), "array_type": b.get("array_type"), "pipeline": pipe}}
        pb = _json.load(open(a.prior_bundle)) if a.prior_bundle and os.path.exists(a.prior_bundle) else {}
        now_pipe = (stage1_meta or {}).get("pipeline") or o.get("pipeline")
        prior_pipe = (pb.get("stage1") or {}).get("pipeline") or pb.get("pipeline")
        ok_, why = SMd.check_same_person(_ctx(o, intake, now_pipe), _ctx(pb, pb.get("intake"), prior_pipe)) if pb else (False, "serial mode refused: --prior-bundle not given or not found")
        if not ok_ or beta is None:
            o["difference_map"] = {"stage": "12b", "status": "REFUSED", "reason": (why.replace("patient", "person identifier") if not ok_ else "no beta vector on this draw")}
        else:
            bp = _pd.read_parquet(a.prior_betas).iloc[:, 0] if a.prior_betas.endswith(".parquet") else _pd.read_csv(a.prior_betas, index_col=0).iloc[:, 0]
            bp.index = bp.index.astype(str); bn = _pd.Series(beta, dtype="float64"); bn.index = bn.index.astype(str)
            _d, summ = SMd.delta_sky(bn, bp.astype("float64"))
            o["difference_map"] = {"stage": "12b", "status": "OK", "same_person": why.replace("patient", "person identifier"), "prior_run_id": pb.get("run_id"), **summ,
                                   "note": "per-address beta difference on the addresses both draws measured; no expectation, no sigma; the difference drawn as a sky is not built"}
    _dev_run(a, o, beta, intake)
    if a.save_betas and beta is not None:   # operator option: the calibrated vector this reading was made from, for re-reading without re-calibrating
        _bs = _pd.Series(beta, dtype="float32").rename("beta"); _bs.index.name = "cpg_id"
        os.makedirs(os.path.dirname(os.path.abspath(a.save_betas)), exist_ok=True)
        (_bs.to_frame().to_parquet(a.save_betas) if a.save_betas.endswith(".parquet") else _bs.to_frame().to_csv(a.save_betas))
        o["betas_saved"] = os.path.abspath(a.save_betas)
    o["command"] = [os.path.basename(sys.argv[0])] + sys.argv[1:]   # for the report's 'run it yourself' section
    cov = _covariates(a)
    o["intake"] = intake; o["intake_skipped"] = bool(a.no_intake); o["sample_id"] = sid; o["covariates"] = cov
    if intake is not None: intake.setdefault("covariates", {}).update(cov)
    if stage1_meta: o["stage1"] = {k: stage1_meta.get(k) for k in ("detection", "n_cpgs", "pipeline")}
    led = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "evidence_ledger.jsonl")
    o["run_id"] = _assign_run_id(led)
    o["versions"] = _versions(ENG)   # every frozen input and chain module, with a short hash, so runs months apart are poolable
    typed = sid; hid = _hash_id(sid)
    r = R3.build(o, a.out, typed)    # the printed report keeps the operator's typed id (author decision B, 2026-10-04)
    o = _redact(o, typed, hid); o["sample_id"] = hid; o["sample_id_hashed"] = True   # bundle and ledger carry the hash only
    m = o.get("met_a") or {}; t = o.get("tare") or {}; q_ = o.get("iam_a") or {}
    print(f"\n{sid}: neutrophil Met-A {m.get('A')} ({m.get('state', m.get('reason', o.get('refusal')))}) | C {(o.get('met_a_cscore') or {}).get('C')} | "
          f"tare {t.get('A_rel')}" + (f" | IAM-A {q_.get('A')} ({q_.get('state', q_.get('refusal'))})" if q_ else ""))
    if not a.no_bundle:
        bundle_path = a.bundle or (os.path.splitext(a.out)[0] + "_bundle.json")
        _json.dump(o, open(bundle_path, "w"), default=str)
        row = {"run_id": o["run_id"], "engine": "v3", "sample_id": hid, "utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
               "report": _redact(os.path.abspath(a.out), typed, hid), "bundle": _redact(os.path.abspath(bundle_path), typed, hid), "specimen": a.specimen,
               "label": ("Met-A COMMISSIONED 2026-10-09 (neutrophils, EPIC v1); other stages DEVELOPMENT" if (o.get("met_a") or {}).get("commissioning") else "DEVELOPMENT - not commissioned"), "refusal_code": o.get("refusal_code"),
               "platform": o.get("platform"), "array_type": o.get("array_type"), "refusal": o.get("refusal"),
               "floors_version": o.get("floors_version"), "reference_version": o.get("reference_version"),
               "stage0_verdict": it.get("stage0_verdict"), "call_rate_status": it.get("call_rate_status"),
               "f_neu": m.get("fraction"), "A": m.get("A"), "state": m.get("state", m.get("reason")), "n_sites": m.get("n_sites"),
               "shift_per_1pct_loss": m.get("shift_per_1pct_loss"), "past_entropy_ceiling": m.get("past_entropy_ceiling"),
               "C": (o.get("met_a_cscore") or {}).get("C"), "A_rel": t.get("A_rel"), "tare": t.get("state", t.get("reason")),
               "n_refs": t.get("n_refs"), "tare_method": t.get("method"), "noise_index": m.get("noise_index"),
               "detection_limit_pct_loss": t.get("detection_limit_pct_loss"),
               "iam_a": q_.get("A"), "iam_a_pipeline": q_.get("pipeline"), "covariates": cov}
        row = _redact(row, typed, hid)
        with open(led, "a", encoding="utf-8") as fh: fh.write(_json.dumps(row, default=str) + "\n")
        print(f"report: {r['out']} | bundle: {bundle_path} | evidence ledger: {led} (one row appended)")
    else:
        print(f"report: {r['out']} (no bundle, no ledger row: --no-bundle)")

if __name__=="__main__": main()
