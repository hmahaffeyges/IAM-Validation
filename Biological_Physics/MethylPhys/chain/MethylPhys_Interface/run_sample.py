#!/usr/bin/env python3
"""run_sample.py - one command: IDAT pair (or a beta table, or single-molecule reads) to a MethylPhys report.

Chain v4. Commissioned: Met-A on isolated neutrophils (2026-10-09), the C-Score on whole blood (2026-10-15), IAM-Atlas v2.1 (2026-10-16), the
IAM-Deconvolver's shares of neutrophils, monocytes, CD4 T and CD8 T cells (2026-10-17), IAM-A on healthy blood CD8 T cells by whole-genome
bisulfite (2026-10-17) and Group-A (2026-10-20, chain/stage_groupa.py). A reading outside an instrument's commissioned scope is printed and
labeled preliminary. The order: Stage 0 intake -> Stage 1 IDAT calibration -> conductor_v3 (platform check, Stage T step 1, Stage A, Stage M,
the noise gate, Stage T step 2, Stage MC) -> Stage D21 cell shares and the sky (stage_v4) -> report_v4, plus Stage Q2 IAM-A when sequencing
input is given. Every stage, gate and function is explained in the Chain Decoder, manual/OM_v4_decoder.md.

    # an Illumina EPIC v1 IDAT pair (needs methylprep for Stage 1); --sex and --age are optional (recorded when given)
    python3 run_sample.py --grn S_Grn.idat.gz --red S_Red.idat.gz --specimen "whole blood" --id S001 --out S001.html

    # specimens: whole blood, buffy coat (whole-blood path, preliminary), isolated / sorted / purified neutrophils, constructed DNA mixtures of
    # blood cells (test material). Any other specimen is refused at intake with the reason.
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
    # 2026-10-20 (Chain Decoder finding F2): the list above predates chain v4. Also every chain module this run actually loaded, every file
    # in Runtime Matrices, the lab-spread file the Deconvolver reads, and IAM-Atlas v2.1 by its manifest (which holds each part's hash; the
    # 408 MB of parts are checked by size, not re-hashed on every run).
    import sys as _sys
    cd = os.path.realpath(chain_dir)
    for m in list(_sys.modules.values()):
        f = os.path.realpath(getattr(m, "__file__", "") or "")
        if f.startswith(cd + os.sep) and f.endswith(".py"): files.setdefault(os.path.relpath(f, cd), {"sha256_12": _sha12(f), "bytes": os.path.getsize(f)})
    for p in sorted(_glob.glob(os.path.join(chain_dir, "Runtime Matrices", "**", "*"), recursive=True)):
        if os.path.isfile(p): files.setdefault(os.path.relpath(p, chain_dir), {"sha256_12": _sha12(p), "bytes": os.path.getsize(p)})
    for rel in (os.path.join("..", "doors", "data", "DEV_ATLAS_V21_01", "transfer_v21_final.json"), os.path.join("..", "atlas", "v2.1", "compact_manifest.json")):
        p = os.path.join(chain_dir, rel)
        if os.path.exists(p): files[os.path.normpath(rel)] = {"sha256_12": _sha12(p), "bytes": os.path.getsize(p)}
    ad = os.path.join(chain_dir, "..", "atlas", "v2.1")
    if os.path.isdir(ad):
        parts = sorted(x for x in os.listdir(ad) if x.endswith(".parquet"))
        files[os.path.normpath(os.path.join("..", "atlas", "v2.1")) + "/*.parquet"] = {"parts": len(parts), "bytes": sum(os.path.getsize(os.path.join(ad, x)) for x in parts),
                                                                                   "hashes": "in compact_manifest.json"}
    out["inputs"] = files
    return out


def _why(rec, check, outcome, measured, limit, source, means, todo, provisional=False):
    """Every stop explained (author rule 2026-10-20; Chain Decoder finding F4): the same five parts Stage 0's gates print - what was measured,
    the limit, where the limit comes from, what it means, what to do - printed and kept in the record (stage0_explained)."""
    x = dict(check=check, outcome=outcome, measured=measured, limit=limit, limit_source=source, provisional=provisional, means=means, what_to_do=todo)
    if isinstance(rec, dict): rec.setdefault("stage0_explained", []).append(x)
    print(f"\n  WHY: {check} - {outcome}\n    measured on this sample: {measured}\n    limit: {limit}\n    limit comes from: {source}"
          f"{' (PROVISIONAL)' if provisional else ''}\n    what it means: {means}\n    what to do: {todo}", flush=True)
    return x


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


def _metaa_atlas(a, b):
    """--dev-metaa-atlas (DEV-METAA-ATLAS-01): Met-A against IAM-Atlas v2.1 on the values Stage T step 1 maps (dev_stages.selftare_map, the map the
    commissioned Met-A reads), then the same-run tare when >= 3 references are given. Development; the reading, gauge and tare are untouched."""
    import dev_stages as DV, stage_metaa_atlas as SA
    bs, info = DV.selftare_map(b)
    if bs is None: return {"label": DV.DEV_LABEL, "status": "NOT_RUN", "reason": "Stage T step 1 could not run: " + str(info.get("reason"))}
    maps = info.get("maps") or {}
    if not all(maps.get(d) for d in ("I", "II")):
        return {"label": DV.DEV_LABEL, "status": "NOT_RUN", "reason": "Stage T step 1 mapped " + ", ".join(d for d in ("I", "II") if maps.get(d)) + " only; both designs are needed", "selftare": maps}
    r = SA.read(bs, a.dev_metaa_atlas, specimen=a.specimen)
    rec = {"label": DV.DEV_LABEL, "status": "OK" if r.get("A") is not None else "NOT_READ", **r, "selftare": {d: maps[d]["slope"] for d in ("I", "II")},
           "values": "Stage 1, then Stage T step 1 (self-tare II)"}
    if a.metaa_atlas_ref_A:
        rec["tare"] = SA.tare(r.get("A"), [float(x) for x in a.metaa_atlas_ref_A.split(",") if x.strip()])
    return rec


def _build(a):
    """--build: the chain builds a reference and records how; the report goes to --bundle (JSON), the reference to --build-out."""
    import numpy as _np, pandas as _pd, stage_info_sphere as IS
    if a.build == "info-sphere":
        if not (a.build_input and a.build_out): print("--build info-sphere needs --build-input and --build-out"); return 2
        B = _pd.read_parquet(a.build_input) if a.build_input.endswith(".parquet") else _pd.read_csv(a.build_input, index_col=0)
        pos = {k: i for i, k in enumerate(B.index.astype(str))}; G = None
        if a.build_groups:
            G = {n: [pos[str(x)] for x in v if str(x) in pos] for n, v in json.load(open(a.build_groups)).items()}
        print(f"Build: information sphere from {os.path.basename(a.build_input)} ({B.shape[0]:,} sites x {B.shape[1]} cell types)", flush=True)
        L, rep = IS.build(B.values, G, n_random=a.build_random)
        _np.savez_compressed(a.build_out, sites=_np.array(B.index.astype(str)), cells=_np.array(B.columns.astype(str)), **L)
        rep.update(input=os.path.basename(a.build_input), groups_file=os.path.basename(a.build_groups) if a.build_groups else None, output=os.path.basename(a.build_out),
                   command=[os.path.basename(sys.argv[0])] + sys.argv[1:], label="DEVELOPMENT - not commissioned (DEV-SPHERE-01)")
        if a.bundle: json.dump(rep, open(a.bundle, "w"), indent=1, default=str)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("command",)}, default=str)[:1500]); return 0
    import reference_globes as RG
    if not a.build_out: print(f"--build {a.build} needs --build-out"); return 2
    if a.build in ("explorer-purified", "explorer-noise"):
        if a.build == "explorer-noise" and not a.build_input: print("--build explorer-noise needs --build-input (the noise maps folder)"); return 2
        spec = RG.purified() if a.build == "explorer-purified" else RG.noise(a.build_input); json.dump(spec, open(a.build_out, "w"))
        print(f"Build: {spec['id']} globe spec, {sum(len(g['layers']) for g in spec['groups'])} layers -> {a.build_out}"); return 0
    if a.build == "atlas-sphere":
        RG.atlas_sphere(a.build_out, a.build_input or RG.ATLAS, RG.manifests()); return 0
    if a.build == "atlas-masks":
        n, bad = RG.atlas_masks(a.build_out); return 1 if bad else 0
    return 2


def _genome_globe(a, cnt, sid):
    """--dev-genome-globe (DEV-IAMA-GLOBE-01, development, not commissioned): the whole genome from the --pat file's molecules, cut into regions
    of --genome-region CpGs on the autosomes (stage_q2_globe), each drawn on every pixel of its span: the sample against its own predicted held
    error (its loose share, the self-reference line), and, with --prior-pat, against an earlier reading. Layers: methylation level, held error
    and the change. Written under bundle development.genome_globe; with --explorer-spec, the globe for the site's Globe Explorer."""
    import gzip, stage_q2_globe as S, dev_stages as DV
    if not a.pat: return {"label": DV.DEV_LABEL, "status": "NOT_RUN", "reason": "needs --pat (the regions are counted from molecules)"}
    S.REGION = a.genome_region; op = (lambda f: gzip.open(f, "rt")) if a.pat.endswith(".gz") else (lambda f: open(f))
    print(f"Development: whole-genome globe, regions of {a.genome_region} CpGs", flush=True)
    R = S.count_pat_regions(op(a.pat)); groups = []; rec = {"label": DV.DEV_LABEL, "status": "OK", "region_cpgs": a.genome_region, "loose_share": cnt.get("loose_share")}
    if cnt.get("loose_share") is not None:
        sp, r1 = S.self_globe(R, cnt["loose_share"], "self", f"{sid} against its own prediction"); groups += sp["groups"]; rec["self"] = r1
    if a.prior_pat:
        Rb = S.count_pat_regions(op(a.prior_pat)); sp, r2 = S.pair_globe(R, Rb, "prior", f"{sid} against {a.prior_name}", a.prior_name)
        groups += sp["groups"]; rec["against_prior"] = dict(r2, prior=os.path.basename(a.prior_pat))
    if a.explorer_spec and groups:
        import globe_viewer as GV
        json.dump(GV.dedupe(dict(id=sid, name=f"{sid}: the whole genome, molecule by molecule (development)", groups=groups,
                                 about=S.ABOUT.replace("__R__", str(a.genome_region)).replace("__WHAT__", "The sample against its own prediction" + (f", and against {a.prior_name}." if a.prior_pat else ".")))),
                  open(a.explorer_spec, "w")); rec["explorer_spec"] = os.path.basename(a.explorer_spec)
    return rec


def _dev_run(a, o, beta, intake):
    """Development flags (doors/DEV_FLAGS_01.md): each writes bundle['development'][stage], labelled DEVELOPMENT - not commissioned.
    None of them changes the reading, the gauge or the tare."""
    flags = [k for k in ("selftare_ii", "direction", "trace", "foreign", "brightness", "nilc", "atlas_e", "percell_b", "sky") if getattr(a, "dev_" + k, False)]
    if a.dev_epic_v2: flags.append("epic_v2")
    if getattr(a, "dev_metaa_atlas", None): flags.append("metaa_atlas")
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
             "sky": lambda: DV.sky(b, fr, a.atlas_v2), "metaa_atlas": lambda: _metaa_atlas(a, b)}
    for k in flags:
        if k == "epic_v2": continue
        try: dev[k] = calls[k]()
        except Exception as e: dev[k] = {"label": DV.DEV_LABEL, "status": "ERROR", "reason": f"{type(e).__name__}: {str(e)[:300]}", "tb": _tb.format_exc()[-600:]}


def main():
    ap=argparse.ArgumentParser(description="IDAT pair or beta table -> MethylPhys report")
    ap.add_argument("--grn"); ap.add_argument("--red"); ap.add_argument("--betas")
    ap.add_argument("--age", type=float, help="declared age in years (optional; recorded when given, read by no stage). Float: records carry 72.0 and 54.5 as readily as 72")
    ap.add_argument("--specimen", default="whole blood")
    ap.add_argument("--banner", default=None, help="a statement printed at the top of the report and both PDFs, e.g. that a known change was planted into a real array for an example (2026-10-18)")
    ap.add_argument("--out", default="methylphys_report.html"); ap.add_argument("--id", default=None)
    ap.add_argument("--bundle", help="also write the full bundle as JSON - every stage's output, for a test harness or an integration that needs more than the report")
    ap.add_argument("--covariate", action="append", default=[], metavar="KEY=VALUE",
                    help="a phenotype or covariate to record with this run, repeatable - e.g. --covariate diagnosis=case --covariate stage=II --covariate cohort=EPIC_Italy. Kept in the custody record and the bundle; never printed in report prose")
    ap.add_argument("--covariates", help="a JSON object of covariates, merged with any --covariate flags")
    ap.add_argument("--no-bundle", action="store_true", help="do not write the machine-readable bundle beside the report (it is written by default: a run that leaves only HTML cannot be pooled later)")
    ap.add_argument("--ledger", default=None, help="append one flat row per run to this JSONL evidence ledger; defaults to evidence_ledger.jsonl beside the report")
    ap.add_argument("--sex", default=None, help="declared sex (F/M), optional; when given Stage 0.8 compares it with the array, otherwise the array's sex is recorded")
    ap.add_argument("--patient-id", default=None, help="already-hashed identifier; a cleartext one is hashed here")
    ap.add_argument("--array-type", default=None, choices=("HM450K", "EPIC_v1", "EPIC_v2"), help="declared array type; omit to take it from the IDAT header (Stage 0.1). The chain reads EPIC_v1 only")
    ap.add_argument("--intake-log", default=None, help="append intake and integrity records here")
    ap.add_argument("--manifest-dir", default=None, help="write the immutable per-sample manifest here")
    ap.add_argument("--no-intake", action="store_true", help="skip Stage 0 (recorded in the report as skipped)")
    ap.add_argument("--engine", default="v3", choices=("v3",), help="kept so recorded commands still run: v3 names the conductor (conductor_v3.py), which chain v4 runs; the class-floor engine was retired 2026-10-03")
    ap.add_argument("--slide-ref-A", "--ref-A", dest="slide_ref_A", default=None, help="comma-separated Met-A before the median tare (met_a.A, self-tared since 2026-10-04) of >= 3 same-run healthy reference arrays (same slide, else same batch) - v3 Stage T median tare, whole blood and isolated neutrophils")
    ap.add_argument("--slide-ref-table", "--ref-table", dest="slide_ref_table", default=None, help="CSV of same-run healthy references with columns A and C (optional id): Met-A and the C-score from pass 1 (met_a.A, self-tared since 2026-10-04; before the median tare). >= 3 rows -> median tare (nothing is fitted)")
    ap.add_argument("--bam", default=None, help="chain v4 Stage Q2: an aligned whole-genome bisulfite BAM (bwa-meth or Bismark), counted by the chain (needs --ref-fasta)")
    ap.add_argument("--ref-fasta", default=None, help="the reference FASTA the --bam was aligned to (indexed)")
    ap.add_argument("--assembly", default="hg19", choices=("hg19", "hg38"), help="assembly of --bam")
    ap.add_argument("--workers", type=int, default=4, help="processes for counting --bam")
    ap.add_argument("--seq-chemistry", default="whole-genome bisulfite", help="chemistry of the sequencing input (IAM-A is commissioned on whole-genome bisulfite)")
    ap.add_argument("--dev-iama-position", action="store_true", help="development: read --pat with the retired position reading (Stage Q) instead of the commissioned self-reference (Stage Q2)")
    ap.add_argument("--geo-pat", default=None, metavar="GSM", help="fetch this GEO sample's .pat file and read it as --pat; the chain takes the file on the build it "
                    "reads (hg19), never another build (stage_q0_intake.geo_pat; Decoder: genome builds)")
    ap.add_argument("--prior-geo-pat", default=None, metavar="GSM", help="the same for --prior-pat")
    ap.add_argument("--geo-dir", default=None, help="where --geo-pat files are written (default: a temporary folder)")
    ap.add_argument("--pat", default=None, help="chain v4 Stage Q2 (IAM-A against the sample's own reference); with --dev-iama-position the retired v3 Stage Q: a wgbstools .pat / .pat.gz file; read with the loyfer_pat_v1 extractor (stage_q_iam_a.pat_site_table)")
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
    ap.add_argument("--build", default=None, choices=("info-sphere", "explorer-purified", "explorer-noise", "atlas-sphere", "atlas-masks"),
                    help="build a reference or a reference globe instead of reading a sample (2026-10-22: the chain builds its own references and draws every "
                         "globe): info-sphere = the information sphere's layout from --build-input (development, DEV-SPHERE-01); explorer-purified / "
                         "explorer-noise = the Globe Explorer's reference globes (--build-input: the noise maps folder); atlas-sphere / atlas-masks = the "
                         "atlas globe's textures and slider pictures (reference_globes.py)")
    ap.add_argument("--build-input", default=None, help="with --build info-sphere: a matrix of methylation levels, sites x cell types (.parquet or .csv, site id first)")
    ap.add_argument("--build-groups", default=None, help="with --build info-sphere: JSON {name: [site ids]} of site groups to test for landing together "
                    "('pool': the sites random groups are drawn from)")
    ap.add_argument("--build-random", type=int, default=500, help="with --build info-sphere: random groups per named group for its line (default 500)")
    ap.add_argument("--build-out", default=None, help="with --build: where the built reference is written (.npz)")
    ap.add_argument("--explorer-spec", default=None, help="also write the report globe's layers as a Globe Explorer spec (JSON) to this path, for the site's "
                    "Globe Explorer (website/art/make_explorer.py sample), so every sample globe on the site is the one this command drew")
    ap.add_argument("--dev-genome-globe", action="store_true", help="DEVELOPMENT - not commissioned (DEV-IAMA-GLOBE-01): with --pat, the whole-genome globe "
                    "(methylation level, held error and the change against the sample's own prediction; with --prior-pat, against an earlier reading too)")
    ap.add_argument("--prior-pat", default=None, help="with --dev-genome-globe: a .pat file of an earlier reading of the same person or specimen")
    ap.add_argument("--prior-name", default="the earlier reading", help="with --prior-pat: what the other file is, as the globe names it")
    ap.add_argument("--genome-region", type=int, default=1000, help="with --dev-genome-globe: CpGs per region (default 1000)")
    ap.add_argument("--dev-metaa-atlas", default=None, metavar="CELL",
                    help="DEVELOPMENT - not commissioned (DEV-METAA-ATLAS-01, 2026-10-22): Met-A of an isolated CELL (an IAM-Atlas v2.1 cell name, e.g. monocytes) "
                         "against that cell in IAM-Atlas v2.1, read on the self-tared values (Stage T step 1, the same map the commissioned Met-A reads); "
                         "written under bundle['development']['metaa_atlas'], not part of the reading")
    ap.add_argument("--metaa-atlas-ref-A", default=None, help="with --dev-metaa-atlas: comma-separated development.metaa_atlas.A of >= 3 healthy samples of the same "
                    "cell run the same way (never this sample); gives A_rel and its state (stage_metaa_atlas.tare)")
    ap.add_argument("--atlas-v2", default=os.environ.get("CPG_ATLAS_V2_PARQUET"), help="atlas v2 parquet for the atlas-based development flags (not stored in the repository)")
    ap.add_argument("--sesame-rscript", default=os.environ.get("METHYLPHYS_SESAME_RSCRIPT"), help="Rscript of an environment with Bioconductor sesame (--dev-epic-v2)")
    a=ap.parse_args()
    if a.geo_pat or a.prior_geo_pat:   # 2026-10-22 (lesson C34): the chain fetches sequencing files from GEO itself, always on the build it reads
        import stage_q0_intake as _Q0g, tempfile as _tf
        _gd = a.geo_dir or _tf.mkdtemp(prefix="geo_pat_"); os.makedirs(_gd, exist_ok=True); _geo = {}
        for _attr, _g in (("pat", a.geo_pat), ("prior_pat", a.prior_geo_pat)):
            if not _g: continue
            if getattr(a, _attr): ap.error(f"give --{_attr.replace('_', '-')} or its --geo form, not both")
            print(f"Fetching {_g}'s {_Q0g.GEO_BUILD} .pat file from GEO", flush=True); _r = _Q0g.geo_pat(_g, _gd); setattr(a, _attr, _r["path"]); _geo[_attr] = dict(_r, gsm=_g)
            print(f"  {_r['name']} ({_r['size']:,} bytes; also on GEO: {', '.join(_r['other_files']) or 'nothing else'})", flush=True)
        a._geo_fetch = _geo
    seq = bool(a.pat or a.site_table or a.bam)
    if a.build: sys.exit(_build(a))
    if not (a.betas or (a.grn and a.red) or seq): ap.error("give --betas, both --grn and --red, or --pat / --bam / --site-table")
    if sum(bool(x) for x in (a.pat, a.site_table, a.bam)) > 1: ap.error("give one of --pat, --bam or --site-table")
    if a.out and os.path.isdir(a.out): ap.error(f"--out {a.out} is a folder; give the report's file name, e.g. --out {os.path.join(a.out, 'report.html')} (the bundle and PDFs are written beside it)")
    if a.bam and not a.ref_fasta: ap.error("--bam needs --ref-fasta: the reference it was aligned to")
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
            _why(intake, "specimen (before Stage 0)", "refused", f"specimen given: {a.specimen!r}",
                 "whole blood, buffy coat, isolated / sorted / purified neutrophils, or a constructed DNA mixture of blood cells",
                 "stage_0_intake.ACCEPTED_SPECIMENS (fixed rule: the specimens the chain has a healthy reference for)",
                 "the chain has no healthy reference for this specimen, so any number it printed would have nothing to be read against",
                 "check the spelling of --specimen; for another specimen a healthy reference has to be built and commissioned first")
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
            for _x in rec.get("stage0_explained") or []:
                print(f"\n  WHY: {_x['check']} - {_x['outcome']}\n    measured on this array: {_x['measured']}\n    limit: {_x['limit']}\n"
                      f"    limit comes from: {_x['limit_source']}{' (PROVISIONAL)' if _x['provisional'] else ''}\n    what it means: {_x['means']}\n    what to do: {_x['what_to_do']}", flush=True)
            if not rec.get("stage0_explained"):
                _why(rec, "intake (Steps 0.1 to 0.3)", "stopped", f"status {rec.get('status')}; flags {rec.get('flags')}", "the rule named in the flag",
                     "stage_0_intake (fixed rules: both files present, a complete manifest, bytes not already taken in)",
                     "the files or their record failed a custody rule before anything was measured", "read the flag; correct the files or the intake log and re-run")
            print("\nQUARANTINE - the chain stops here and nothing is scored. "
                  "Stage 0 rejected the specimen before calibration.", flush=True)
            sys.exit(2)
        if (rec.get("array_type_detected") or declared) == "EPIC_v2":
            # unhandled platform (2026-10-03, DEV-BASE-CHAIN-01): an EPIC v2 array is refused here, before the QC decode, instead of being
            # judged by EPIC v1 rules (sex mismatch) or crashing in Stage 1. v3 reads EPIC v1 only.
            platform_stop = ("EPIC v2 array (IDAT header): the chain reads EPIC v1 arrays only - its healthy references are frozen for EPIC v1, and Stage 1 "
                             "(methylprep 1.7.1) does not calibrate EPIC v2. Intake stopped before the QC decode; nothing is scored.")
            rec["status"] = "PLATFORM_REFUSED"; rec.setdefault("flags", []).append("PLATFORM_REFUSED:EPIC_v2"); rec["stage0_verdict"] = "REFUSED_PLATFORM"
            print("  PLATFORM_REFUSED - " + platform_stop, flush=True)
            _why(rec, "platform (Step 0.1, IDAT header)", "refused", "EPIC v2 (from the IDAT header)", "EPIC v1",
                 "the chain's frozen references (Runtime Matrices) are built on EPIC v1 probes, and Stage 1's calibrator (methylprep 1.7.1) reads EPIC v1",
                 "an EPIC v2 array has different probes, so its values cannot be read against EPIC v1 references",
                 "read an EPIC v1 array; EPIC v2 needs its own references built and commissioned (development: dev_stages.epicv2_calibrate)")
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
                    _why(rec, "environment (before the QC decode)", "stopped, exit code 3 (the machine, not the array)", f"{type(e).__name__}: {str(e)[:160]}",
                         "the array manifest loads", "stage_0_1_qc_handoff._manifest (methylprep's manifest file for the array type)",
                         "this machine is missing the file that says which probe is which; the array was not examined",
                         "install the chain's environment (pip install -r Biological_Physics/MethylPhys/chain/requirements.txt) with network access once, so methylprep can fetch its manifest, and re-run")
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
                _why(rec, "IDAT decode (QC hand-off)", "stopped", f"{type(e).__name__}: {str(e)[:160]}", "both IDAT files decode to their end",
                     "fixed rule: a file the decoder cannot read has no trustworthy intensities",
                     "the files are truncated or damaged (seen: a gzip cut short that still passed the 1 MB size check)",
                     "download the pair again and compare its checksum with the source's; re-run")
                print("\nQUARANTINE - the chain stops here and nothing is scored.", flush=True)
                sys.exit(2)
            # every hard intake failure stops the run here, before calibration (0.7b's 450K coverage needs the calibrated
            # beta and is gated again after Stage 1; on EPIC it is NA_EPIC)
            rec = S0.step_0_7b_platform_coverage(rec, None)
            rec = S0.step_0_9_decision_gate(rec, a.intake_log)
            if rec.get("stage0_verdict") == "QUARANTINE":
                print(f"Stage 0 verdict: QUARANTINE  hard failures: {rec.get('stage0_hard_fail')}", flush=True)
                for _x in rec.get("stage0_explained") or []:   # every stop explained (author rule 2026-10-20)
                    print(f"\n  WHY: {_x['check']} - {_x['outcome']}\n    measured on this array: {_x['measured']}\n    limit: {_x['limit']}\n"
                          f"    limit comes from: {_x['limit_source']}{' (PROVISIONAL)' if _x['provisional'] else ''}\n    what it means: {_x['means']}\n    what to do: {_x['what_to_do']}", flush=True)
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
            platform_stop = f"array type not calibrated by Stage 1 ({e}): the chain reads EPIC v1 arrays only; nothing is scored."
            print("  PLATFORM_REFUSED - " + platform_stop, flush=True)
            _why(intake, "platform (Stage 1)", "refused", str(e)[:160], "an EPIC v1 array", "Stage 1's calibrator and the chain's EPIC v1 references",
                 "Stage 1 could not calibrate this array type, so there are no values to read", "read an EPIC v1 array")
        except (EOFError, OSError) as e:
            # the IDAT bytes could not be read (truncated or corrupt file): the specimen's files, not the chain - a named stop, not a traceback
            print(f"  QUARANTINE_CORRUPT_IDAT - Stage 1 could not read the IDAT pair: {type(e).__name__}: {e}", flush=True)
            _why(intake, "Stage 1 calibration (reading the IDAT pair)", "stopped", f"{type(e).__name__}: {str(e)[:160]}", "both IDAT files read to their end",
                 "fixed rule: a file the calibrator cannot read has no trustworthy values", "the files are truncated or damaged",
                 "download the pair again and compare its checksum with the source's; re-run")
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
            # 2026-10-21 (Chain Decoder finding F16): Stage 1's two "bisulfite" medians are of two different control sets (all BS Conversion I
            # probes in green, all BS Conversion II in red), not a matched-pair conversion efficiency, so they are recorded and not graded here;
            # the conversion efficiency is Stage 0's, from each matched pair in its own channel (stage_0_1_qc_handoff).
            _c1 = {k: v for k, v in (meta.get("controls") or {}).items() if not k.startswith("bisulfite_conversion_")}
            _gradable = any(k in _c1 for k in ("hyb_high_median", "hyb_low_median", "extension_meth_median", "extension_unmeth_median"))
            cv = S0.validate_control_probes(_c1) if _gradable else {"ctrl_qc": "recorded, not graded (Stage 1 records control medians that no Stage 0 rule reads)", "metrics": _c1, "flags": []}
            s1 = {"ctrl_qc": cv["ctrl_qc"], "ctrl_metrics": cv["metrics"], "ctrl_flags": cv["flags"], "detection_statistic": "poobah p <= 0.05",
                  "bisulfite_note": "Stage 1's bisulfite medians (two different control sets) are recorded in intake.controls and not graded; the conversion efficiency is Stage 0's"}
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
            for _x in intake.get("stage0_explained") or []:   # every stop explained (author rule 2026-10-20; Decoder finding F4)
                print(f"\n  WHY: {_x['check']} - {_x['outcome']}\n    measured on this array: {_x['measured']}\n    limit: {_x['limit']}\n"
                      f"    limit comes from: {_x['limit_source']}{' (PROVISIONAL)' if _x['provisional'] else ''}\n    what it means: {_x['means']}\n    what to do: {_x['what_to_do']}", flush=True)
            if not intake.get("stage0_explained"):
                _why(intake, "Stage 0 after calibration (Step 0.7b)", "stopped", f"hard failures {intake.get('stage0_hard_fail')}", "the rule named in the failure",
                     "stage_0_intake", "the calibrated array failed an intake rule", "read the named failure; re-run on a complete EPIC v1 array")
            print("\nQUARANTINE - the chain stops here and nothing is scored. "
                  "Stage 0 rejected the specimen; see the reasons above.", flush=True)
            sys.exit(2)

    import pandas as _pd, json as _json, datetime as _dt, conductor_v3 as C3, report_v4 as R3
    refs = [float(x) for x in a.slide_ref_A.split(",") if x.strip()] if a.slide_ref_A else None
    if a.slide_ref_table:
        rt = _pd.read_csv(a.slide_ref_table)
        if "A" not in rt.columns: sys.exit(f"--slide-ref-table {a.slide_ref_table}: needs a column A (and C for the C-Score tare)")
        refs = [{k: (None if _pd.isna(r.get(k)) else r.get(k)) for k in ("A", "f_neu", "N", "C", "id", "gsm") if k in rt.columns} for r in rt.to_dict("records")]   # C carried (2026-10-18, LESSONS C26): the C-Score tare reads it
    it = intake or {}
    array_type = it.get("array_type_detected") or it.get("array_type") or a.array_type   # header first, then declared
    if specimen_stop:
        o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "Met-A and the C-Score read the neutrophils; the cell shares cover six blood cells", "array_type": array_type, "refusal": specimen_stop,
             "refusal_code": "SPECIMEN_REFUSED"}
    elif platform_stop:
        o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "Met-A and the C-Score read the neutrophils; the cell shares cover six blood cells", "platform": "EPIC_v2" if "EPIC v2" in platform_stop else None,
             "array_type": array_type, "refusal": platform_stop}
    elif beta is not None and _S0spec.atlas_cell_of(a.specimen):   # 2026-10-22: an isolated IAM-Atlas v2.1 cell other than neutrophils
        o = {"build": C3.BUILD, "specimen": a.specimen, "array_type": array_type, "atlas_cell": _S0spec.atlas_cell_of(a.specimen),
             "scope": "an isolated " + _S0spec.atlas_cell_of(a.specimen) + " specimen: read by Met-A against IAM-Atlas v2.1 (development.metaa_atlas, not commissioned); "
                      "the commissioned Met-A and C-Score read neutrophils and whole blood and are not applied; the cell shares cover six blood cells"}
        if not a.dev_metaa_atlas: a.dev_metaa_atlas = o["atlas_cell"]
    elif beta is not None:
        o = C3.run_neutrophil(_pd.Series(beta, dtype="float64"), specimen=a.specimen, ref_A=refs, array_type=array_type, sample_id=sid)
    else:
        o = {"build": C3.BUILD, "specimen": a.specimen, "scope": "Met-A and the C-Score read the neutrophils; the cell shares cover six blood cells", "note": "sequencing input only: no array reading"}
    _z = o.pop("_cscore_z", None)
    if beta is not None and not o.get("refusal"):   # chain v4: the cell shares (D21) and the C-Score drawn on the sphere
        import stage_v4 as V4
        print("Stage D21: cell shares against IAM-Atlas v2.1", flush=True)
        o["cell_shares"] = V4.stage_d21(_pd.Series(beta, dtype="float64"))
        _m, _c, _t = o.get("met_a") or {}, o.get("met_a_cscore") or {}, o.get("tare") or {}
        _rd = "    ".join(x for x in (("" if str(_m.get("state", "")).startswith("withheld") else f"Met-A {_t['A_rel']:.3f} (tared)" if _t.get("A_rel") is not None else (f"Met-A {_m['A']:.3f}" if _m.get("A") is not None else "")),
                                     (f"C-score {_c['C_rel']:.2f} (tared)" if _c.get("C_rel") is not None else (f"C-score {_c['C']:.2f}" if _c.get("C") is not None else ""))) if x)
        o["banner"] = a.banner
        o["cscore_sky"] = V4.cscore_sky(_z, os.path.splitext(os.path.abspath(a.out))[0] + "_cscore_sky.png", title=f"{sid}: C-Score departures on the sphere", readings=_rd)
        if _z is not None:   # 2026-10-18: the healthy neutrophil of IAM-Atlas v2.1 at the same sites, same pixels, drawn beside the sample's sky
            import report_figs_v4 as RF
            try: o["atlas_sky"] = RF.atlas_sky_png(os.path.splitext(os.path.abspath(a.out))[0] + "_atlas_sky.png", list(_z.index), "neutrophils",
                                                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1"), V4.nest_pix2ang,
                                                    title="Healthy neutrophils, IAM-Atlas v2.1: the same sites on the same sphere")
            except Exception as e: o["atlas_sky"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
            try: o["changes_sky"] = RF.changes_sky_png(os.path.splitext(os.path.abspath(a.out))[0] + "_changes_sky.png", _z, "neutrophils",   # the overlay (2026-10-18)
                                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1"), V4.nest_pix2ang, readings=_rd)
            except Exception as e: o["changes_sky"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
        try:   # the turning globe in the HTML report (2026-10-21, step 13d): sample, its healthy reference, the difference, Met-A's departures
            import report_globe_v4 as RG
            _g = RG.build(_pd.Series(beta, dtype="float64"), a.specimen, {r["cell"]: r["share"] for r in (o.get("cell_shares") or {}).get("cells") or []}, z=_z,
                           fractions=(o.get("composition") or {}).get("fractions"),
                           sex=(intake or {}).get("declared_sex") or (intake or {}).get("predicted_sex"), out_html=os.path.splitext(os.path.abspath(a.out))[0] + "_globe.html")
            _fr = _g.pop("fragment", ""); _spec = _g.pop("spec", None); o["_globe_prose"] = _g.pop("prose", "")
            if a.explorer_spec and _spec:   # 2026-10-22: the Globe Explorer takes this file as written; it never redraws a sample itself (rule 23b-f)
                json.dump(_spec, open(a.explorer_spec, "w")); _g["explorer_spec"] = os.path.basename(a.explorer_spec)
            if _g.get("html"): _fr = _fr.replace("<h2>The sample on the globe, against its healthy reference</h2>", "<h2>The sample on the globe, against its healthy reference</h2>"
                                                 f"<p><a href='{os.path.basename(_g['html'])}'>Open the globe on its own full page</a> (the same layers, larger, on a dark background).</p>", 1)
            o["_globe_html"] = _fr; o["globe"] = _g
        except Exception as e: o["globe"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    if seq and (a.pat or a.bam) and not a.dev_iama_position:   # chain v4 Stage Q2 - IAM-A against the sample's own reference (commissioned 2026-10-17)
        import stage_v4 as V4, stage_q2_selfref as _Q2
        cells = set(_Q2.shape_reference()["cells"]); src = {"pat": os.path.abspath(a.pat)} if a.pat else {"bam": os.path.abspath(a.bam), "ref_fasta": a.ref_fasta, "assembly": a.assembly}
        if a.pat:
            import stage_q0_intake as Q0
            print(f"Stage Q0: intake checks on {os.path.basename(a.pat)}", flush=True)
            o["iam_a_intake"] = Q0.intake(a.pat, specimen=a.specimen, cell=a.seq_cell, alignment_qc=a.alignment_qc, selfref_cells=cells)
        else:   # 2026-10-20 (Chain Decoder finding F3): say that Stage Q0 did not run and why, instead of leaving no intake record
            o["iam_a_intake"] = {"stage": "Q0", "status": "not run", "reason": "Stage Q0's file checks are written for the .pat format; on an aligned BAM, "
                                 "Stage Q2's counter records the bisulfite conversion failure (non-CpG calls) and the substitution rate (A and T), and the "
                                 "sample-shape check refuses a cell type with no healthy reference"}
            print("Stage Q0: not run on a BAM (its checks are for .pat files); Stage Q2 records conversion and substitution rates", flush=True)
        qi = o.get("iam_a_intake") or {}
        if qi.get("refusal_code"):
            o["iam_a"] = {"stage": "Q2", "reading": "IAM-A", "cell": a.seq_cell, "A": None, "refusal_code": qi["refusal_code"], "refusal": f"Stage Q0 stopped the file: {qi['refusal']}", "input": src}
        else:
            print(f"Stage Q2: counting {os.path.basename(a.pat or a.bam)} (held error and loose share, one pass)", flush=True)
            _cnt = V4.count_sequencing(pat=a.pat, bam=a.bam, ref_fasta=a.ref_fasta, workers=a.workers, assembly=a.assembly)
            o["iam_a"] = V4.stage_q2(_cnt, a.seq_cell, chemistry=a.seq_chemistry, source=src)
            if a.dev_genome_globe:   # 2026-10-22: the whole-genome globe, drawn by the one command (development)
                o.setdefault("development", {})["genome_globe"] = _genome_globe(a, _cnt, sid)
    elif seq:   # Stage Q (v3, retired position reading): --site-table, or --pat with --dev-iama-position
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
    o["command"] = [os.path.basename(sys.argv[0])] + sys.argv[1:]
    if getattr(a, "_geo_fetch", None): o["geo_fetch"] = a._geo_fetch   # which file the chain took from GEO, its build and checksum   # for the report's 'run it yourself' section
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
    print(f"\n{sid}: Met-A (neutrophils) {m.get('A')} ({m.get('state', m.get('reason', o.get('refusal')))}) | C {(o.get('met_a_cscore') or {}).get('C')} | "
          f"tare {t.get('A_rel')}" + (f" | IAM-A {q_.get('A')} ({q_.get('state', q_.get('refusal'))})" if q_ else ""))
    if not a.no_bundle:
        bundle_path = a.bundle or (os.path.splitext(a.out)[0] + "_bundle.json")
        _json.dump(o, open(bundle_path, "w"), default=str)
        row = {"run_id": o["run_id"], "engine": "v4 (conductor_v3)", "sample_id": hid, "utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
               "report": _redact(os.path.abspath(a.out), typed, hid), "bundle": _redact(os.path.abspath(bundle_path), typed, hid), "specimen": a.specimen,
               "label": C3.BUILD, "refusal_code": o.get("refusal_code"),
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
