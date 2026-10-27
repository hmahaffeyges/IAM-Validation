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


def _id_collides(sid):
    """2026-10-24 (provisional reference test 1: the id 'megakaryocytes' erased the cell's name in the bundle): True when the typed sample id
    is, ignoring case, the name of an IAM-Atlas v2.1 cell type or of a provisional column, the words the bundle must keep."""
    try:
        import pyarrow.parquet as _pq
        AT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1")
        names = {c.lower() for k in (1, 2, 3) for c in _pq.ParquetFile(os.path.join(AT, f"mu_part{k}.parquet")).schema.names}
        PD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Runtime Matrices", "Provisional_References")
        names |= {f.split("_v")[0].lower() for f in (os.listdir(PD) if os.path.isdir(PD) else []) if f.endswith(".parquet")}
        return str(sid).strip().lower() in names
    except Exception:
        return False

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
    """--dev-metaa-atlas (DEV-METAA-ATLAS-01): Met-A against IAM-Atlas v2.1 on the values Stage T step 1 maps (stage_t_selftare.selftare_map, the map the
    commissioned Met-A reads), then the same-run tare when >= 3 references are given. Development; the reading, gauge and tare are untouched."""
    import dev_stages as DV, stage_metaa_atlas as SA
    if getattr(a, "_seqv", None):   # 2026-10-22 (step 40): sequencing values; the self-tare maps the two array designs, so it does not apply
        r = SA.read(b, a.dev_metaa_atlas, specimen=a.specimen, counts=getattr(a, "_seqc", None))
        rec = {"label": DV.DEV_LABEL, "status": "OK" if r.get("A") is not None else "NOT_READ", **r, "values": "sequencing values (stage_seq_values); no self-tare (arrays only)"}
        if a.metaa_atlas_ref_A: rec["tare"] = SA.tare(r.get("A"), [float(x) for x in a.metaa_atlas_ref_A.split(",") if x.strip()])
        return rec
    if getattr(a, "dev_metaa_curve", False):   # 2026-10-25 (plan 14.14): the atlas as the ruler, on Stage 1's values and the array's own curve
        r = SA.read_on_curve(b, a.dev_metaa_atlas)
        return {"label": DV.DEV_LABEL, "status": "OK" if r.get("A") is not None else "NOT_READ", **r, "values": "Stage 1 (no self-tare: the array's own curve is the ruler's map)"}
    import stage_t_selftare as _ST; bs, info = _ST.selftare_map(b)   # Stage T step 1 (F23: stage_t_selftare)
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


_ISV_ABOUT = '<h2>What you are looking at</h2>\n<p>Every point of light is one site on the human genome, one of 191,458 that IAM-Atlas v2.1 measured in all 79 of its healthy cell types and that vary between them. The address globes place each site by <i>where it sits on the DNA</i>. This sphere places each site by <i>how it behaves</i>: its methylation in the 79 cell types. Sites that behave alike land near each other, wherever they sit in the genome.</p>\n<h2>Nobody placed these sites. The math did.</h2>\n<p>No person and no program chose where any site, cell or tissue goes. Each site\'s position is computed from its 79 measured levels alone, in four fixed steps, and the same steps are applied to every site:</p>\n<ol>\n<li><b>Each site becomes a point on a sphere.</b> A site\'s level in a cell type is a probability b (methylated or not). Taking the square roots of b and 1&minus;b in every cell type gives a vector whose length is always exactly one, because b + (1&minus;b) = 1. Every site therefore lies on the surface of a sphere. The sphere is not a drawing choice; it is where probabilities naturally live.</li>\n<li><b>The site\'s own level is taken out.</b> Without this step, a site\'s average level would decide where it lands, and mostly-methylated sites would simply crowd together. With it, a site is placed only by <i>how it departs from its own level</i> across the cells: which cells it is higher in, and which lower.</li>\n<li><b>The three directions along which sites differ most are kept.</b> They hold 39 % of the differences between sites; the rest are not drawn.</li>\n<li><b>The positions are fixed.</b> Every sample is later read on this same sphere.</li>\n</ol>\n<p><b>Only then are the colors added.</b> Each site is labeled with the cell type where it departs most from its own level, and the 79 cell types are grouped into 15 families by tissue and lineage. That labeling is ours; the positions are not. The labels were laid on top of positions already computed, and nothing in the computation knew which cells are related.</p>\n<p><b>So what the picture shows is a finding, not a design.</b> The blood lineages ended up side by side: bone-marrow progenitors, B and NK cells, T cells and myeloid cells. Gut, pancreas, lung and airway, kidney, brain and muscle each hold their own region. The methylation alone put cells that share a developmental origin, and so share regulatory programs, next to each other. A further pattern the geometry produced on its own: each cell\'s methylated sites and its unmethylated sites gather on <i>opposite</i> sides of the sphere (for T cells, each side tightly at 0.88 and 0.86 on a scale where 1 is all in one place), because taking each site\'s level out turns "higher in this cell" and "lower in this cell" into opposite directions.</p>\n<h2>How to read it</h2>\n<ul>\n<li><b>Territories</b> (the colored glow) are where one family\'s identity sites are over-represented: its share of the sites there divided by its share of all identity sites, so a large family does not color the sphere by size alone. The <b>dark lanes</b> are where two families are nearly equal. Point at the sphere to read whose territory you are over and by how much.</li>\n<li><b>Colored points</b> are sites that stand apart in one cell type clearly, by at least 0.10 more than in any other: 41,241 of the 191,458. The rest carry no clear single-cell identity and are drawn as faint dust when "Every site" is on.</li>\n<li><b>The dark sky</b> (45 % of the sphere) holds no sites. It is masked, as cosmologists mask our galaxy on the microwave sky, rather than stretched to fill the sphere, because stretching would falsify the distances.</li>\n</ul>\n<h2>Why this geometry</h2>\n<p>The angle between two sites on this sphere is the Fisher&ndash;Rao distance of information geometry. For nearby states it is, up to a constant, the relative entropy between them: the same quantity IAM measures as disorder against a physical floor. The instruments and this sphere measure in one currency. The square-root geometry itself is decades-old mathematics, used for distances between probability distributions and between populations. What is new here is treating it as a sky: a masked spherical map of a whole methylome, read with the microwave-sky tools, with one individual compared against itself.</p>\n<h2>The two spheres</h2>\n<table><tr><th></th><th>Address globe</th><th>Information sphere</th></tr>\n<tr><td>A site\'s place is set by</td><td>where it sits on the DNA</td><td>how it behaves across the 79 cell types</td></tr>\n<tr><td>Neighbors are</td><td>sites next to each other on a chromosome</td><td>sites that act alike, wherever they are</td></tr>\n<tr><td>It answers</td><td>where on the genome a change is</td><td>what kind of change it is, and in which cell\'s program</td></tr>\n<tr><td>It cannot tell</td><td>that distant sites belong to one program</td><td>where on a chromosome a change is</td></tr></table>\n<p>A loss along one stretch of chromosome 2 is one patch on the address globe and scattered points here. A cell\'s whole program failing at sites spread over every chromosome is scattered dots on the address globe and one territory here. The two are kept together because each sees what the other cannot.</p>\n<h2>Reading one sample</h2>\n<p>A sample is read against its own expected healthy self: the atlas mixed at the cell shares the IAM-Deconvolver found in that sample, with no other person\'s sample involved. Each site\'s change is measured in that site\'s healthy width and in the sample\'s own spread, exactly as the address globe reads it. A territory has <b>changed</b> when its mean change stands beyond the line pure chance crosses less than once in 20 samples, judged against the sample\'s own other territories. The number of changed territories is fixed by the measurement; the slider only fades everything else to dark so the change stands alone. The sample\'s distance from its own expected healthy self is given as a Fisher&ndash;Rao angle.</p>\n<h2>Status</h2>\n<p>Development, not commissioned. On the atlas\'s real healthy cells, 11 of 12 groups of cell-marking sites land together beyond what their methylation level alone would give, and look-alike groups with the same levels do not (0 of 100). The sample reading is next tested on planted changes, then on healthy bloods, which should stay dark.</p>\n'   # the information sphere picture's explanation (DEV-SPHERE-01)

CONTROL_SITES = ("cg05575921", "cg03636183", "cg16867657")   # 2026-10-23: AHRR and F2RL3 (lowered in smokers' blood), ELOVL2 (rises with age)
CONTROL_SITES_ABOUT = {"cg05575921": "AHRR: lower in current smokers' blood", "cg03636183": "F2RL3: lower in current smokers' blood",
                       "cg16867657": "ELOVL2: rises with age in blood"}

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
        L, rep = IS.build(B.values, G, n_random=a.build_random, construction=a.build_construction or "v1")
        _np.savez_compressed(a.build_out, sites=_np.array(B.index.astype(str)), cells=_np.array(B.columns.astype(str)), **L)
        rep.update(input=os.path.basename(a.build_input), groups_file=os.path.basename(a.build_groups) if a.build_groups else None, output=os.path.basename(a.build_out),
                   command=[os.path.basename(sys.argv[0])] + sys.argv[1:], label="DEVELOPMENT - not commissioned (DEV-SPHERE-01)")
        if a.bundle: json.dump(rep, open(a.bundle, "w"), indent=1, default=str)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("command",)}, default=str)[:1500]); return 0
    if a.build == "provisional":   # 2026-10-24 (plan 40.18, development): a provisional reference column, kept apart from IAM-Atlas v2.1
        if not (a.build_samples and a.build_cell): print("--build provisional needs --build-samples CSV and --build-cell NAME"); return 2
        import stage_provisional as _PV; m = _PV.build(a.build_cell, a.build_samples, os.path.join(os.path.dirname(a.build_out or ".") or ".", "_pool_work"), a.build_out)
        print(json.dumps(m, indent=1)); return 0
    if a.build == "info-globe":   # 2026-10-23 (DEV-SPHERE-01, development): the information sphere as a picture, from IAM-Atlas v2.1
        if not a.build_out: print("--build info-globe needs --build-out PAGE.html"); return 2
        import pyarrow.parquet as _pq, info_sphere_viewer as _ISV
        AT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1"); parts, cells = [], []
        for n in (1, 2, 3):   # read part by part as stored (uint16, 65535 = no value), so memory stays near the atlas's own size
            t = _pq.read_table(os.path.join(AT, f"mu_part{n}.parquet")); cells += t.column_names
            parts.append(_np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names]).astype("uint16")); del t
        U16 = _np.hstack(parts); del parts; ok = (U16 != 65535).all(1); B = U16[ok].astype("float32") / 10000.0; del U16
        keep = B.std(1, ddof=1) > 0.08; B = B[keep]   # the pool of DEV-SPHERE-01 tests 8 to 11 (pandas spread, n - 1): measured in all 79 cell types, spread above 0.08
        print(f"Build: information sphere picture from IAM-Atlas v2.1 ({B.shape[0]:,} sites x {B.shape[1]} cell types; construction v5)", flush=True)
        L, rep = IS.build(B, None, n_random=0, construction="v5")
        cell, sgn, size, margin, cf = IS.roles(B, cells)
        import hashlib as _hl
        _sid = _np.array(_pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist(), dtype=object)[ok][keep]
        IS.save_layout(IS.LAYOUT, _sid, L["X"], cell, sgn, size, margin, cells, cf, _hl.sha256(open(os.path.join(AT, "sites.parquet"), "rb").read()).hexdigest())
        r = _ISV.write(a.build_out, L["X"], cell, sgn, size, margin, cells, cf, IS.FAMILIES, "The information sphere", a.build_about or _ISV_ABOUT)
        X3 = L["X"].astype(float); fo = cf[cell]; clr = margin >= _ISV.CLEAR
        def _R(m): return round(float(_np.linalg.norm(X3[m].mean(0))), 2) if m.sum() >= 20 else None
        rep.update(margin_median=round(float(_np.median(margin)), 3), sites_clear=int(clr.sum()), clear_margin=_ISV.CLEAR,
                   concentration_by_family={f[0]: dict(all=_R(fo == k), methylated_there=_R((fo == k) & (sgn > 0)), unmethylated_there=_R((fo == k) & (sgn < 0)),
                                                        clear_methylated_there=_R(clr & (fo == k) & (sgn > 0)), clear_unmethylated_there=_R(clr & (fo == k) & (sgn < 0)))
                                            for k, f in enumerate(IS.FAMILIES)})
        rep.update(picture=os.path.basename(a.build_out), page_bytes=r["bytes"], sites_by_family={f[0]: int((cf[cell] == k).sum()) for k, f in enumerate(IS.FAMILIES)},
                   command=[os.path.basename(sys.argv[0])] + sys.argv[1:], label="DEVELOPMENT - not commissioned (DEV-SPHERE-01)")
        if a.bundle: json.dump(rep, open(a.bundle, "w"), indent=1, default=str)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("command",)}, default=str)[:1800]); return 0
    if a.build == "constructed-draw":   # TESTS ONLY (2026-10-22): a constructed draw with a known answer, built by the chain from spent values
        if not (a.build_input and a.build_out and a.build_mix is not None): print("--build constructed-draw needs --build-input (blood values), --build-mix TISSUE:SHARE,... ('' for none) and --build-out DIR"); return 2
        import stage_seq_values as _SVc
        bl = _pd.read_csv(a.build_input, index_col=0).iloc[:, 0]; add = {x.rsplit(":", 1)[0].strip(): float(x.rsplit(":", 1)[1]) for x in a.build_mix.split(",") if x.strip()}
        db, dp = (int(x) for x in (a.build_depth or "30,15").split(","))
        _pp = (a.dev_plant.split(":")[0], float(a.dev_plant.split(":")[1])) if a.dev_plant else None   # with --build: planted before the reads are counted (40.11)
        rec = _SVc.construct_draw(bl, add, db, dp, a.build_out, plant=_pp); rec["input"] = os.path.basename(a.build_input); rec["command"] = [os.path.basename(sys.argv[0])] + sys.argv[1:]
        json.dump(rec, open(os.path.join(a.build_out, "truth.json"), "w"), indent=1); print("Build: constructed draw", json.dumps({k: v for k, v in rec.items() if k != "command"})[:600]); return 0
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
    ap.add_argument("--dev-seq-values", default=None, type=int, metavar="MIN_CALLS",
                    help="DEVELOPMENT (to-do step 40): with --pat, also make a value at each IAM-Atlas v2.1 site from the molecules (methylated calls over "
                         "calls, sites with at least MIN_CALLS calls; chain/stage_seq_values.py), so the IAM-Deconvolver, Met-A for each cell, Met-A against "
                         "the atlas and the globe read the same sequencing run IAM-A reads. The commissioned array readings (Met-A on neutrophils, the C-Score) "
                         "are not applied to sequencing values; every reading is flagged")
    ap.add_argument("--dev-thin", default=None, type=float, metavar="FRACTION",
                    help="TESTS ONLY (step 40): read a seeded thinned copy of --pat, each molecule kept with this probability (stage_seq_values.thin_pat), "
                         "so a deep sample stands in for a shallow one; IAM-A and the sequencing values both read the thinned copy; recorded and bannered")
    ap.add_argument("--cell-panel", default="blood", metavar="PANEL",
                    help="the cells the IAM-Deconvolver splits the sample into: 'blood' (the six blood cells it is commissioned on, the default), 'atlas' (all of "
                         "IAM-Atlas v2.1's cell types, development: every share flagged), or a comma-separated list of atlas cell names (development); Met-A for "
                         "each cell reads every cell the panel finds (2026-10-22)")
    ap.add_argument("--build-construction", default=None, choices=("v1", "v5"), help="--build info-sphere: v1 (default) or v5, the level taken out (DEV-SPHERE-01, development)")
    ap.add_argument("--build-mix", default=None, metavar="TISSUE:SHARE,...", help="--build constructed-draw: atlas tissues added to the plasma half and their shares (tests only)")
    ap.add_argument("--build-depth", default=None, metavar="BUFFY,PLASMA", help="--build constructed-draw: reads per site in each half (default 30,15)")
    ap.add_argument("--dev-count-only", action="store_true",
                    help="DEVELOPMENT (to-do 14): with --pat, record Stage Q2's count (held error and opportunities) even where Stage Q0 refuses an IAM-A reading, flagged; for tests on samples without a healthy reference (cell lines, mouse)")
    ap.add_argument("--dev-count-genome", default="hg19", choices=("hg19", "mm10"), help="--dev-count-only: the genome the .pat was made on (recorded)")
    ap.add_argument("--dev-tiles", default="table", choices=["table", "dictionary"], help="DEVELOPMENT (plan 14.9): Stage Q2's tiles from the commissioned table (default) or from the genome dictionary's chromosome ranges")
    ap.add_argument("--dev-count-tiles", default=None, metavar="OUT.csv[.gz]", help="DEVELOPMENT (DEV-ERASE-01 graded maintenance test, 2026-10-25): with --dev-count-only, also write the held error tile by tile (2,000 consecutive CpGs per tile, two independent halves of the molecules)")
    ap.add_argument("--dev-count-sites", default=None, metavar="CELL", help="DEVELOPMENT (DEV-METAA-HEALTHY-01 step 3 version 2): with --dev-count-only, also count the held error only at Met-A's commissioned identity sites of CELL (e.g. neutrophils), by Stage Q2's own rules (stage_q2_selfref.count_pat_sites)")
    ap.add_argument("--dev-metaa-specific", default=None, metavar="DELTA[,DELTA]",
                    help="DEVELOPMENT (39.9): also read Met-A for each cell on sites held by that cell and differing by at least DELTA from every other cell of the panel")
    ap.add_argument("--dev-seq-beta", default=None, metavar="FILE.beta[.gz]",
                    help="DEVELOPMENT (40.10): read a .beta file of per-CpG counts (hg19 CpG numbering) as values at IAM-Atlas v2.1's sites; "
                         "every reader but IAM-A (no molecules); with --dev-seq-values N, sites with fewer than N reads have no value; "
                         "or CALL.bw,COV.bw, a GRCh38 bigWig pair of per-CpG levels and coverage (2026-10-24)")
    ap.add_argument("--dev-metaa-allsites", action="store_true",
                    help="DEVELOPMENT (2026-10-22): also read Met-A for each cell over every atlas site, each weighted by the shift a loss of that cell would make "
                         "(stage_metaa_atlas.read_in_mix_allsites), recorded beside the identity-site reading")
    ap.add_argument("--dev-territories", action="store_true", default=True, help="DEVELOPMENT (plan 48): disorder at cancer's own territories (islands unmarked in every cell type; loosely held domains), against the atlas in the sample's mix. On by default since 2026-10-23 (author: 'wire in the territories score')")
    ap.add_argument("--dev-ruler", action="store_true", help="DEVELOPMENT (plan 0.2): put the array on IAM-Atlas v2.1's scale (stage_t_selftare.ruler_map) before the whole-sample Met-A and the territories. Off until the many-level ruler passes (the two-anchor line overcorrects: healthy 1.113 untared, 0.938 on it)")
    ap.add_argument("--no-territories", dest="dev_territories", action="store_false", help="skip the territory readings")
    ap.add_argument("--dev-plant-territory", default=None, metavar="islands|domains:FRACTION:SIZE", help="TESTS ONLY: a constructed gain at islands or loss in domains, planted by the chain")
    ap.add_argument("--dev-plant", default=None, metavar="CELL:LOSS[:all|runs]",
                    help="TESTS ONLY (2026-10-22, author: tests run through the chain): plant a CONSTRUCTED loss of CELL's pattern (an IAM-Atlas v2.1 cell) in this "
                         "array's values after Stage 1, before every reading, by stage_metaa_atlas.plant; in whole blood the loss is scaled by CELL's share as the "
                         "IAM-Deconvolver reads it on the unplanted values. Every report and bundle says PLANTED; the bundle records the plant (key 'planted')")
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
    ap.add_argument("--seq-cell", default=None, help="cell type of the sequenced specimen (Stage Q); default: the stated specimen's IAM-Atlas v2.1 cell when it names one (2026-10-25: 'sorted b cells' was read as neutrophils and refused), else neutrophils")
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
    ap.add_argument("--build", default=None, choices=("info-sphere", "info-globe", "provisional", "explorer-purified", "explorer-noise", "atlas-sphere", "atlas-masks", "constructed-draw"),
                    help="build a reference or a reference globe instead of reading a sample (2026-10-22: the chain builds its own references and draws every "
                         "globe): info-sphere = the information sphere's layout from --build-input (development, DEV-SPHERE-01); explorer-purified / "
                         "explorer-noise = the Globe Explorer's reference globes (--build-input: the noise maps folder); atlas-sphere / atlas-masks = the "
                         "atlas globe's textures and slider pictures (reference_globes.py)")
    ap.add_argument("--build-about", default=None, help="with --build info-globe: the explanation beside the picture (default: the one in run_sample.py)")
    ap.add_argument("--build-input", default=None, help="with --build info-sphere: a matrix of methylation levels, sites x cell types (.parquet or .csv, site id first)")
    ap.add_argument("--build-groups", default=None, help="with --build info-sphere: JSON {name: [site ids]} of site groups to test for landing together "
                    "('pool': the sites random groups are drawn from)")
    ap.add_argument("--build-random", type=int, default=500, help="with --build info-sphere: random groups per named group for its line (default 500)")
    ap.add_argument("--build-out", default=None, help="with --build: where the built reference is written (.npz)")
    ap.add_argument("--dev-provisional", default=None, metavar="CELL[,CELL]", help="DEVELOPMENT (plan 40.18): with the full atlas panel, a second deconvolution with provisional columns (stage_provisional) beside IAM-Atlas v2.1, recorded as cell_shares_provisional and flagged; the atlas's own shares are unchanged")
    ap.add_argument("--build-samples", default=None, help="--build provisional: the samples CSV (gsm, file) of per-CpG read files to pool")
    ap.add_argument("--build-cell", default=None, help="--build provisional: the cell's name")
    ap.add_argument("--explorer-spec", default=None, help="also write the report globe's layers as a Globe Explorer spec (JSON) to this path, for the site's "
                    "Globe Explorer (website/art/make_explorer.py sample), so every sample globe on the site is the one this command drew")
    ap.add_argument("--dev-genome-globe", action="store_true", help="DEVELOPMENT - not commissioned (DEV-IAMA-GLOBE-01): with --pat, the whole-genome globe "
                    "(methylation level, held error and the change against the sample's own prediction; with --prior-pat, against an earlier reading too)")
    ap.add_argument("--prior-pat", default=None, help="with --dev-genome-globe: a .pat file of an earlier reading of the same person or specimen")
    ap.add_argument("--prior-name", default="the earlier reading", help="with --prior-pat: what the other file is, as the globe names it")
    ap.add_argument("--genome-region", type=int, default=1000, help="with --dev-genome-globe: CpGs per region (default 1000)")
    ap.add_argument("--dev-info-sphere", action="store_true", default=True, help="DEVELOPMENT - not commissioned (DEV-SPHERE-01): the sample's change on the information sphere (page OUT_info_sphere.html; needs the layout from --build info-globe). On by default since 2026-10-23 (author: 'it still gets created by the chain and we learn each time it does')")
    ap.add_argument("--no-info-sphere", dest="dev_info_sphere", action="store_false", help="skip the information sphere reading")
    ap.add_argument("--dev-metaa-curve", action="store_true", help="DEVELOPMENT (plan 14.14, 2026-10-25): with --dev-metaa-atlas CELL, read Met-A with the atlas as the ruler on the array's own response curve (stage_metaa_atlas.read_on_curve) instead of the same-run tare")
    ap.add_argument("--dev-metaa-atlas", default=None, metavar="CELL",
                    help="DEVELOPMENT - not commissioned (DEV-METAA-ATLAS-01, 2026-10-22): Met-A of an isolated CELL (an IAM-Atlas v2.1 cell name, e.g. monocytes) "
                         "against that cell in IAM-Atlas v2.1, read on the self-tared values (Stage T step 1, the same map the commissioned Met-A reads); "
                         "written under bundle['development']['metaa_atlas'], not part of the reading")
    ap.add_argument("--metaa-atlas-ref-A", default=None, help="with --dev-metaa-atlas: comma-separated development.metaa_atlas.A of >= 3 healthy samples of the same "
                    "cell run the same way (never this sample); gives A_rel and its state (stage_metaa_atlas.tare)")
    ap.add_argument("--atlas-v2", default=os.environ.get("CPG_ATLAS_V2_PARQUET"), help="atlas v2 parquet for the atlas-based development flags (not stored in the repository)")
    ap.add_argument("--sesame-rscript", default=os.environ.get("METHYLPHYS_SESAME_RSCRIPT"), help="Rscript of an environment with Bioconductor sesame (--dev-epic-v2)")
    a=ap.parse_args()
    if a.dev_tiles != "table":   # 2026-10-25 (plan 14.9, development)
        import stage_q2_selfref as _Q2tile; _Q2tile.TILE_SOURCE = a.dev_tiles
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
    a._seqv = None; a._thin = None; a._seqc = None
    if a.dev_thin is not None:   # 2026-10-22 (step 40): a deep sample thinned by the chain itself
        if not a.pat: ap.error("--dev-thin needs --pat (or --geo-pat)")
        import stage_seq_values as _SVt, tempfile as _tft
        _tp, a._thin = _SVt.thin_pat(a.pat, a.dev_thin, os.path.join(_tft.mkdtemp(), "thinned_" + os.path.basename(a.pat).replace(".gz", "") + ".gz"))
        a._thin["source"] = os.path.basename(a.pat); a.pat = _tp
        a.banner = (a.banner + " | " if a.banner else "") + f"THINNED (constructed for a test): {a.dev_thin:g} of the molecules kept ({a._thin['molecules_kept']:,} of {a._thin['molecules_in']:,})"
        print(a.banner, flush=True)
    if a.dev_seq_beta:   # 2026-10-22 (40.10): a .beta file of per-CpG counts; values at the atlas sites, no molecules (IAM-A not read)
        if a.betas or a.grn or a.pat: ap.error("--dev-seq-beta makes the values itself; do not also give --betas, IDATs or --pat")
        import stage_seq_values as _SV, tempfile as _tf
        if a.dev_seq_beta.count(",") == 1 and all(x.strip().lower().endswith((".bw", ".bigwig")) for x in a.dev_seq_beta.split(",")):   # 2026-10-24: CALL.bw,COV.bw (GRCh38)
            _vb, _vc, a._seqv = _SV.values_from_bigwig(*[x.strip() for x in a.dev_seq_beta.split(",")], a.dev_seq_values or 1)
        else:
            _vb, _vc, a._seqv = _SV.values_from_beta(a.dev_seq_beta, a.dev_seq_values or 1)
        a._seqc = _vc
        a.betas = os.path.join(_tf.mkdtemp(), "sequencing_values.csv"); _vb.rename("beta").to_frame().to_csv(a.betas)
        print(f"Sequencing values from {os.path.basename(a.dev_seq_beta)}: {a._seqv['sites_with_value']:,} of {a._seqv['atlas_sites']:,} sites; median {a._seqv['calls_per_site_median']:.0f} calls", flush=True)
    elif a.dev_seq_values is not None:   # 2026-10-22 (step 40): the same molecules give a value at each atlas site
        if not a.pat: ap.error("--dev-seq-values needs --pat (or --geo-pat)")
        if a.betas or a.grn: ap.error("--dev-seq-values makes the values from --pat; do not also give --betas or IDATs")
        import stage_seq_values as _SV, tempfile as _tf
        print(f"Sequencing values: counting {os.path.basename(a.pat)} at IAM-Atlas v2.1's sites (at least {a.dev_seq_values} calls)", flush=True)
        _vb, _vc, a._seqv = _SV.values_from_pat(a.pat, a.dev_seq_values); a._seqc = _vc   # read counts per site, for Met-A's k-of-n expectation (40.3)
        a.betas = os.path.join(_tf.mkdtemp(), "sequencing_values.csv"); _vb.rename("beta").to_frame().to_csv(a.betas)
        print(f"  {a._seqv['sites_with_value']:,} of {a._seqv['atlas_sites']:,} sites with a value; median {a._seqv['calls_per_site_median']:.0f} calls per site", flush=True)
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
        if v == "QUARANTINE" and list(intake.get("stage0_hard_fail") or []) == ["hm450_coverage"]:
            # 2026-10-23 (author: read it and flag it; the EPIC-Italy smoke run): on a 450K array the coverage gate protects the commissioned
            # Met-A and C-Score, whose EPIC references a 450K chip only half covers; those readers refuse 450K themselves (conductor_v3,
            # PLATFORM_REFUSED), so the array goes on to the atlas readers, flagged, each with its own coverage. The gate's FAIL stays in the record.
            intake["stage0_verdict"] = "ADVANCE_FLAGGED_450K"; intake["stage0_hard_fail"] = []
            intake.setdefault("flags", []).append(f"HM450_COVERAGE_BELOW_COMMISSIONED_REFERENCE:{intake.get('hm450_reference_coverage')} (the commissioned readers refuse 450K; atlas readers only, flagged)")
            print(f"Stage 0: 450K coverage of the commissioned references {intake.get('hm450_reference_coverage')} (below {S0.HM450_COVERAGE_MIN:g}): the commissioned "
                  "Met-A and C-Score will refuse this array; the atlas readers read it, flagged", flush=True); v = intake["stage0_verdict"]
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
    planted_rec = None
    def _panel():   # --cell-panel (2026-10-22): the six blood cells (commissioned), every atlas cell, or a named list (development)
        import stage_v4 as _V4c
        p = (a.cell_panel or "blood").strip()
        if p == "blood": return None
        if p == "atlas": return _V4c.atlas_panel()
        cells = [c.strip() for c in p.split(",") if c.strip()]; have = set(_V4c.atlas_panel()); bad = [c for c in cells if c not in have]
        if bad: sys.exit(f"--cell-panel: not IAM-Atlas v2.1 cell names: {bad}")
        return cells
    if a.dev_plant_territory and beta is not None:   # 2026-10-23 (plan 48): TESTS ONLY, a constructed change at one territory
        import stage_metaa_atlas as _SAt
        _tn, _tf, _ts = a.dev_plant_territory.split(":"); _bt, _nt = _SAt.plant_territory(_pd.Series(beta, dtype="float64"), _tn, float(_tf), float(_ts)); beta = _bt.to_dict()
        a.banner = (a.banner + " | " if a.banner else "") + f"PLANTED (constructed for a test): {_tn} {'gain' if _tn == 'islands' else 'loss'} of {float(_ts):.0%} at {float(_tf):.0%} of the set, {_nt} sites"
        print(f"PLANTED: {a.banner}", flush=True)
    if a.dev_plant and beta is not None:   # 2026-10-22: a constructed loss planted by the chain itself, so a test is the one command end to end
        import stage_metaa_atlas as _SAp, stage_v4 as _V4p
        _pc, _, _pl = a.dev_plant.partition(":"); _pl, _, _scope = _pl.partition(":"); _pl = float(_pl or 0.01); _scope = _scope or "identity"; _b0 = _pd.Series(beta, dtype="float64")
        _mix = str(a.specimen).lower() in ("whole blood", "blood", "peripheral blood", "venous blood", "buffy coat")
        _sh = {r["cell"]: r["share"] for r in (_V4p.stage_d21(_b0, _panel()).get("cells") or [])} if _mix else {}
        _f = float(_sh.get(_pc, 0.0)) if _mix else 1.0
        if _mix and _pc not in _sh: sys.exit(f"--dev-plant {a.dev_plant}: '{_pc}' is not among the cells the IAM-Deconvolver read ({sorted(_sh)})")
        _bp = {"all": _SAp.plant_all, "runs": _SAp.plant_runs}.get(_scope, _SAp.plant)(_b0, _pc, _pl * _f); _n = int((_bp != _b0).sum()); beta = _bp.to_dict()
        planted_rec = dict(cell=_pc, loss=_pl, share=round(_f, 4), applied=round(_pl * _f, 6), sites_moved=_n, scope=_scope,
                           rule="x + loss (0.5 - mu) at " + {"all": "every atlas site the cell has a value (stage_metaa_atlas.plant_all)", "runs": "10 runs of 30 neighboring identity sites in genome order (stage_metaa_atlas.plant_runs)"}.get(_scope, "the cell's identity sites (stage_metaa_atlas.plant)"))
        a.banner = (a.banner + " | " if a.banner else "") + f"PLANTED (constructed for a test): a {_pl:.0%} loss of the {_pc} pattern" + (f" at its share {_f:.4f}" if _mix else "") + f", {_n} sites"
        print(f"PLANTED: {a.banner}", flush=True)
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
    elif beta is not None and _S0spec.is_plasma(a.specimen):   # 2026-10-22 (steps 39-40): plasma cell-free DNA, read on the full atlas panel, flagged
        o = {"build": C3.BUILD, "specimen": a.specimen, "array_type": array_type if not a._seqv else "sequencing values (development)",
             "scope": "plasma cell-free DNA (development): the IAM-Deconvolver splits it on IAM-Atlas v2.1's full panel and Met-A reads each source it "
                      "finds against the atlas in this sample's own shares; the commissioned Met-A and C-Score read whole blood and are not applied"}
        if a._seqv: o["sequencing_values"] = a._seqv
        if (a.cell_panel or "blood") == "blood": a.cell_panel = "atlas"
    elif beta is not None and _S0spec.is_tissue(a.specimen):   # 2026-10-23: solid tissue, tumor, or a specimen not named (blind runs), on the full atlas, flagged
        o = {"build": C3.BUILD, "specimen": a.specimen, "array_type": array_type if not a._seqv else "sequencing values (development)",
             "scope": "tissue or an unnamed specimen (development): the IAM-Deconvolver splits it on IAM-Atlas v2.1's full panel, so the reading says what "
                      "the sample is made of, and Met-A reads each cell it finds against the atlas in this sample's own shares; the commissioned Met-A and "
                      "C-Score read whole blood and are not applied"}
        if a._seqv: o["sequencing_values"] = a._seqv
        if (a.cell_panel or "blood") == "blood": a.cell_panel = "atlas"
    elif beta is not None and a._seqv:   # 2026-10-22 (step 40): sequencing values; the commissioned array readings are not applied
        o = {"build": C3.BUILD, "specimen": a.specimen, "array_type": "sequencing values (development)", "sequencing_values": a._seqv,
             "scope": "values made from sequencing molecules at IAM-Atlas v2.1's sites (development): the cell shares, Met-A for each cell and the globe read "
                      "them; the commissioned Met-A and C-Score read EPIC v1 arrays and are not applied"}
        if _S0spec.atlas_cell_of(a.specimen) and not a.dev_metaa_atlas: a.dev_metaa_atlas = _S0spec.atlas_cell_of(a.specimen)
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
    if getattr(a, "_thin", None): o["thinned"] = a._thin; o["banner"] = a.banner   # 2026-10-22: a thinned test copy is always recorded
    _450 = (beta is not None and o.get("refusal_code") == "PLATFORM_REFUSED" and (o.get("platform") == "450K" or str(array_type) == "HM450K"))
    if _450:   # 2026-10-22 (step 27; author: read it and flag it): no 450K neutrophil reference, so the commissioned Met-A and C-Score stay refused,
        # but the atlas readers work by site name and 450K's sites are on IAM-Atlas v2.1, so they read, flagged, with their own coverage
        o["atlas_readers_on_450k"] = ("450K array: the commissioned Met-A and C-Score are not read (no 450K healthy neutrophil reference); the IAM-Deconvolver, "
                                      "Met-A for each cell and the globe read the 450K sites on IAM-Atlas v2.1 (development, not commissioned on 450K)")
        print("450K array: atlas readers run, flagged (" + o["atlas_readers_on_450k"][:60] + "...)", flush=True)
    if beta is not None and (not o.get("refusal") or _450):   # chain v4: the cell shares (D21) and the C-Score drawn on the sphere
        import stage_v4 as V4
        print("Stage D21: cell shares against IAM-Atlas v2.1", flush=True)
        o["cell_shares"] = V4.stage_d21(_pd.Series(beta, dtype="float64"), _panel(), sequencing=bool(a._seqv))
        if a.dev_provisional and _panel():   # 2026-10-24 (plan 40.18): read and flag, beside the atlas's own shares
            print("Stage D21: a second deconvolution with provisional columns (" + a.dev_provisional + "), flagged", flush=True)
            o["cell_shares_provisional"] = V4.stage_d21_provisional(_pd.Series(beta, dtype="float64"), _panel(), [c.strip() for c in a.dev_provisional.split(",") if c.strip()], sequencing=bool(a._seqv))
        _atlas_spec = (str(a.specimen).lower() in ("whole blood", "buffy coat") or _S0spec.is_plasma(a.specimen) or _S0spec.is_tissue(a.specimen) or _S0spec.atlas_cell_of(a.specimen))
        if _atlas_spec and not (o["cell_shares"] or {}).get("cells"):   # 2026-10-25: say why the atlas readings did not run, instead of leaving them out
            _withheld_why = "not read: the cell shares were withheld (" + str((o["cell_shares"] or {}).get("withheld") or (o["cell_shares"] or {}).get("reason") or "no shares") + ")"
            for _k in ("met_a_each_cell", "met_a_whole", "territories", "cscore_each_cell"): o[_k] = {"flag": "development", "reason": _withheld_why}
        if (o["cell_shares"] or {}).get("cells") and (str(a.specimen).lower() in ("whole blood", "buffy coat") or _S0spec.is_plasma(a.specimen) or _S0spec.is_tissue(a.specimen) or _S0spec.atlas_cell_of(a.specimen)):   # 2026-10-22: Met-A for each cell, read and flagged (development); 2026-10-25: isolated atlas cells too (the cord-blood B cells read no Met-A)
            try:
                import stage_metaa_atlas as SA, stage_t_selftare as _ST
                _bs, _inf = (None, None) if a._seqv else _ST.selftare_map(_pd.Series(beta, dtype="float64"))   # the self-tare maps array designs; not sequencing
                o["met_a_each_cell"] = SA.read_in_mix(_bs if _bs is not None else _pd.Series(beta, dtype="float64"),
                                                      {r["cell"]: r["share"] for r in o["cell_shares"]["cells"]}, counts=getattr(a, "_seqc", None))
                o["met_a_each_cell"]["values"] = "Stage 1, then Stage T step 1 (self-tare II)" if _bs is not None else "Stage 1 (Stage T step 1 could not run)"
                _vals = _bs if _bs is not None else _pd.Series(beta, dtype="float64"); _shr = {r["cell"]: r["share"] for r in (o.get("cell_shares") or {}).get("cells") or [] if r.get("share")}
                a._cell_dep = {r["cell"]: SA.departure_in_mix(_vals, _shr, r["cell"]) for r in o["met_a_each_cell"].get("cells", []) if r.get("A") is not None}   # for the globe (2026-10-22)
                try: o["cscore_each_cell"] = SA.cscore_limits(SA.cscore_in_mix(_vals, _shr, a._cell_dep), o.get("array_type"))   # 2026-10-23 (41.2, 41.3); 2026-10-27: each reading's healthy limit
                except Exception as e: o["cscore_each_cell"] = {"flag": "development", "reason": f"{type(e).__name__}: {e}"[:200]}
                if a.dev_metaa_allsites: o["met_a_each_cell_allsites"] = SA.read_in_mix_allsites(_vals, _shr)
                _rv, o["atlas_ruler"] = (None, {"reason": "sequencing values are not mapped" if a._seqv else "the atlas ruler is off (--dev-ruler turns it on; development)"}) if (a._seqv or not a.dev_ruler) else _ST.ruler_map(_pd.Series(beta, dtype="float64"))   # 2026-10-23: the atlas ruler
                _vr = _rv if _rv is not None else _vals
                try: o["met_a_whole"] = SA.read_whole(_vr, _shr, counts=getattr(a, "_seqc", None)); o["met_a_whole"]["state"] = "on the atlas ruler" if _rv is not None else "untared"   # 2026-10-23: Met-A for the whole sample over every atlas site (development, flagged; on by default)
                except Exception as e: o["met_a_whole"] = {"flag": "development", "reason": f"{type(e).__name__}: {e}"[:200]}
                if a.dev_territories:   # 2026-10-23 (plan 48): disorder at cancer's own territories, against the atlas in this sample's mix
                    try: o["territories"] = SA.territory_reading(_vr, _shr, counts=getattr(a, "_seqc", None)); o["territories"]["state"] = o["met_a_whole"].get("state")
                    except Exception as e: o["territories"] = {"flag": "development", "reason": f"{type(e).__name__}: {e}"[:200]}
                if a.dev_metaa_specific:   # 2026-10-22 (39.9): the same reading on sites held by each cell and not by the others
                    o["met_a_each_cell_specific"] = {d.strip(): SA.read_in_mix(_vals, _shr, counts=getattr(a, "_seqc", None), specific=float(d)) for d in a.dev_metaa_specific.split(",") if d.strip()}
            except Exception as e: o["met_a_each_cell"] = {"flag": "development", "reason": f"{type(e).__name__}: {e}"[:200]}
        _m, _c, _t = o.get("met_a") or {}, o.get("met_a_cscore") or {}, o.get("tare") or {}
        _rd = "    ".join(x for x in (("" if str(_m.get("state", "")).startswith("withheld") else f"Met-A {_t['A_rel']:.3f} (tared)" if _t.get("A_rel") is not None else (f"Met-A {_m['A']:.3f}" if _m.get("A") is not None else "")),
                                     (f"C-score {_c['C_rel']:.2f} (tared)" if _c.get("C_rel") is not None else (f"C-score {_c['C']:.2f}" if _c.get("C") is not None else ""))) if x)
        o["banner"] = a.banner
        if planted_rec: o["planted"] = planted_rec
        o["cscore_sky"] = V4.cscore_sky(_z, os.path.splitext(os.path.abspath(a.out))[0] + "_cscore_sky.png", title=f"{sid}: C-Score departures on the sphere", readings=_rd)
        if _z is not None:   # 2026-10-18: the healthy neutrophil of IAM-Atlas v2.1 at the same sites, same pixels, drawn beside the sample's sky
            import report_figs_v4 as RF
            try: o["atlas_sky"] = RF.atlas_sky_png(os.path.splitext(os.path.abspath(a.out))[0] + "_atlas_sky.png", list(_z.index), "neutrophils",
                                                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1"), V4.nest_pix2ang,
                                                    title="Healthy neutrophils, IAM-Atlas v2.1, at Met-A's 6,000 neutrophil sites (the C-Score's sites)")
            except Exception as e: o["atlas_sky"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
            try: o["changes_sky"] = RF.changes_sky_png(os.path.splitext(os.path.abspath(a.out))[0] + "_changes_sky.png", _z, "neutrophils",   # the overlay (2026-10-18)
                                                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1"), V4.nest_pix2ang, readings=_rd)
            except Exception as e: o["changes_sky"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
        try:   # the turning globe in the HTML report (2026-10-21, step 13d): sample, its healthy reference, the difference, Met-A's departures
            import report_globe_v4 as RG
            _g = RG.build(_pd.Series(beta, dtype="float64"), a.specimen, {r["cell"]: r["share"] for r in (o.get("cell_shares") or {}).get("cells") or []}, z=_z,
                           fractions=(o.get("composition") or {}).get("fractions"),
                           sex=(intake or {}).get("declared_sex") or (intake or {}).get("predicted_sex"), out_html=os.path.splitext(os.path.abspath(a.out))[0] + "_globe.html",
                           cell_departures=getattr(a, "_cell_dep", None))
            _fr = _g.pop("fragment", ""); _spec = _g.pop("spec", None); o["_globe_prose"] = _g.pop("prose", ""); _sites = _g.pop("_sites", None)
            if a.dev_info_sphere and _sites is None: o["info_sphere"] = {"reason": "the globe was not drawn, so there is no site-by-site change to place: " + str((o.get("globe") or {}).get("reason") or "")[:160]}   # 2026-10-23: never skip silently
            if a.dev_info_sphere and _sites is not None:   # 2026-10-23 (DEV-SPHERE-01, plan 35.2): the same change, on the information sphere
                try:
                    import stage_info_sphere as IS, info_sphere_viewer as ISV
                    _lay = IS.load_layout(); _id, _zg, _sm, _rf = _sites
                    _r, _x, _lit, _sg = IS.sample_reading(_pd.Series(_zg, index=_id), _lay, IS.FAMILIES, _pd.Series(_sm, index=_id), _pd.Series(_rf, index=_id))
                    _ch = [g for g in _r["territories"] if g["changed"]]
                    _fix = (f"Changed: {_r['territories_changed']} of {_r['territories_read']} territories (each family's clearly marking sites, methylated or unmethylated "
                            f"there), beyond the line {_r['line']}. This count is fixed; the slider never changes it.")
                    if _ch: _fix += " " + "; ".join(f"{g['family']}, {'methylated' if g['methylated_there'] else 'unmethylated'} there: {'up' if g['z'] > 0 else 'down'} ({g['z']:+.1f})" for g in _ch) + "."
                    if "fisher_rao_rms_rad" in _r: _fix += f" Distance from its own expected healthy self: {_r['fisher_rao_rms_rad']:.4f} rad (Fisher-Rao, root mean square over {_r['sites_measured']:,} sites)."
                    _ip = os.path.splitext(os.path.abspath(a.out))[0] + "_info_sphere.html"
                    ISV.write(_ip, _lay["X"].astype(float), _lay["cell"].astype(int), _lay["sgn"].astype(int), _lay["size"].astype(float), _lay["margin"].astype(float),
                              [str(c) for c in _lay["cells"]], _lay["cell_family"].astype(int), IS.FAMILIES, "This sample on the information sphere", _ISV_ABOUT,
                              sample=dict(x=_x, lit=_lit, sign=_sg, fixed=_fix))
                    _r["html"] = os.path.basename(_ip); _r["label"] = "DEVELOPMENT - not commissioned (DEV-SPHERE-01)"; o["info_sphere"] = _r
                except Exception as e: o["info_sphere"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
            if a.explorer_spec and _spec:   # 2026-10-22: the Globe Explorer takes this file as written; it never redraws a sample itself (rule 23b-f)
                json.dump(_spec, open(a.explorer_spec, "w")); _g["explorer_spec"] = os.path.basename(a.explorer_spec)
            if _g.get("html"): _fr = _fr.replace("<h2>The sample on the globe, against its healthy reference</h2>", "<h2>The sample on the globe, against its healthy reference</h2>"
                                                 f"<p><a href='{os.path.basename(_g['html'])}'>Open the globe on its own full page</a> (the same layers, larger, on a dark background).</p>", 1)
            o["_globe_html"] = _fr; o["globe"] = _g
        except Exception as e: o["globe"] = {"drawn": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    if seq and (a.pat or a.bam) and not a.dev_iama_position:   # chain v4 Stage Q2 - IAM-A against the sample's own reference (commissioned 2026-10-17)
        import stage_v4 as V4, stage_q2_selfref as _Q2
        if a.seq_cell is None:   # 2026-10-25: the stated specimen's atlas cell (the cord-blood B cells were refused as "neutrophils" before)
            import stage_0_intake as _S0; a.seq_cell = _S0.atlas_cell_of(a.specimen) or "neutrophils"; print(f"Stage Q2: the sequenced cell is {a.seq_cell} (from the specimen)", flush=True)
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
        if a.dev_count_only and a.pat:   # 2026-10-22 (to-do 14; read and flag): Stage Q2's count recorded even where intake refuses a reading
            print("Stage Q2: counting only (development, --dev-count-only): held error and its opportunities, flagged", flush=True)
            _c0 = V4.count_sequencing(pat=a.pat)
            o["iam_a_count_only"] = dict(stage="Q2 count (development)", flag="DEVELOPMENT, NOT COMMISSIONED: Stage Q2's count only; no IAM-A reading",
                                         genome=a.dev_count_genome, intake_decision=qi.get("decision"), intake_refusal=qi.get("refusal"),
                                         count={k: _c0.get(k) for k in ("held_err", "held_opp", "loose_share", "tiles", "gain_err", "gain_opp", "mol_n", "mol_low_share", "mol_mid_share", "coupling", "coupling_err_share", "coupling_pair_share", "coupling_pairs")},
                                         note=(None if a.dev_count_genome == "hg19" else "the loose share uses hg19 tiles and has no meaning on " + a.dev_count_genome + "; the held error does not depend on the genome"))
            if a.dev_count_tiles:   # 2026-10-25 (DEV-ERASE-01 graded maintenance test): the held error tile by tile, two independent halves
                import gzip as _gzt, stage_q2_selfref as _Q2t
                with (_gzt.open if str(a.pat).endswith(".gz") else open)(a.pat, "rt") as _fh: _T = _Q2t.count_pat_tile_errors(_fh)
                with (_gzt.open if str(a.dev_count_tiles).endswith(".gz") else open)(a.dev_count_tiles, "wt") as _fo:
                    _fo.write("chrom,tile,iso_a,opp_a,iso_b,opp_b\n"); [_fo.write(f"{c},{j},{v[0]},{v[1]},{v[2]},{v[3]}\n") for (c, j), v in sorted(_T.items())]
                o["iam_a_count_only"]["count_tiles"] = dict(file=os.path.basename(a.dev_count_tiles), tiles=len(_T), tile_cpgs=_Q2t.TILE, halves="even and odd lines of the .pat (no shared molecule)")
                print(f"Stage Q2 held error by tile: {len(_T)} tiles written to {a.dev_count_tiles}", flush=True)
            if a.dev_count_sites:   # 2026-10-24 (step 3 version 2): the same count at Met-A's identity sites only
                import gzip as _gz, pyarrow.parquet as _pqs, stage_q2_selfref as _Q2s
                _MF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"][a.dev_count_sites]
                _K = _pqs.read_table(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).to_pandas()
                _K = _K[_K.probe.isin(set(_MF["sites"]))]; _idx = _K.loyfer_cpg_index.dropna().astype(int).tolist()
                with (_gz.open if str(a.pat).endswith(".gz") else open)(a.pat, "rt") as _fh: _cs = _Q2s.count_pat_sites(_fh, _idx)
                o["iam_a_count_only"]["count_sites"] = dict(cell=a.dev_count_sites, sites="Met-A's commissioned identity sites (metA_floors_v1_3.json, EPIC)", n_probes=len(_MF["sites"]), n_with_cpg_index=len(_idx), **_cs)
                # 2026-10-24 (step 3 version 3): every call at those sites by molecule class, the sites split by the atlas's level in the cell (>= 0.5 held)
                _AT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "atlas", "v2.1"); _ids = _pqs.read_table(os.path.join(_AT, "sites.parquet")).column("cpg").to_pylist()
                _u = next(_pqs.read_table(os.path.join(_AT, f"mu_part{_n}.parquet"), columns=[a.dev_count_sites]).column(0).to_numpy() for _n in (1, 2, 3) if a.dev_count_sites in _pqs.ParquetFile(os.path.join(_AT, f"mu_part{_n}.parquet")).schema.names)
                _lv = dict(zip(_ids, _u)); _grp = {int(r.loyfer_cpg_index): ("held" if _lv.get(r.probe, 65535) / 1e4 >= 0.5 else "absent") for r in _K.dropna(subset=["loyfer_cpg_index"]).itertuples() if _lv.get(r.probe, 65535) != 65535}
                with (_gz.open if str(a.pat).endswith(".gz") else open)(a.pat, "rt") as _fh: o["iam_a_count_only"]["count_site_classes"] = _Q2s.count_pat_site_classes(_fh, _grp)
                print("Stage Q2 molecule classes at the identity sites:", o["iam_a_count_only"]["count_site_classes"], flush=True)
                print(f"Stage Q2 at {a.dev_count_sites} identity sites: {_cs}", flush=True)
        _mixed = _S0spec.is_plasma(a.specimen) or _S0spec.is_tissue(a.specimen) or str(a.specimen).lower() in ("whole blood", "buffy coat")
        _only_q07 = [c["step"] for c in (qi.get("checks") or []) if c.get("result") == "REFUSED"] == ["Q0.7 specimen"]
        if qi.get("refusal_code") and _mixed and _only_q07:   # 2026-10-23 (read and flag): a mixed specimen has no single cell's healthy shape, so
            # IAM-A is read on its cell-independent part, the self-reference line (the held error against the one predicted from the sample's own
            # loose share), with the shape check not applied, and flagged; every other intake rule still stops the file
            print("Stage Q2: a mixed specimen: IAM-A read on the self-reference line, flagged (no healthy shape reference for a mix)", flush=True)
            _cnt = V4.count_sequencing(pat=a.pat, bam=a.bam, ref_fasta=a.ref_fasta, workers=a.workers, assembly=a.assembly)
            _B = _Q2.band(1); _r = _Q2.reading(_cnt) if _cnt.get("held_opp") and _cnt.get("loose_share") is not None else None
            o["iam_a"] = {"stage": "Q2", "reading": "IAM-A", "cell": "mixed specimen (" + str(a.specimen) + ")", "input": src, "A": None,
                          "count": {k: _cnt.get(k) for k in ("loose_share", "held_err", "held_opp", "tiles", "tile_p10", "gain_err", "gain_opp", "mol_n", "mol_low_share", "mol_mid_share", "coupling", "coupling_err_share", "coupling_pair_share", "coupling_pairs")}, "band": {"lo": _B["lo"], "hi": _B["hi"]},
                          "development": "DEVELOPMENT - read and flagged: a mixed specimen has no healthy shape reference, so the sample-shape check is not applied; "
                                         "IAM-A is commissioned on healthy blood CD8 T cells"}
            if _r is not None:
                _A = float(_r["A"]); o["iam_a"].update(A=round(_A, 4), predicted=round(_r["predicted"], 6), held_err=round(_r["held_err"], 6), loose_share=round(_r["loose_share"], 4),
                                                       state="Normal" if _B["lo"] <= _A <= _B["hi"] else ("above Normal" if _A > _B["hi"] else "below Normal"), shape_check="not applied (mixed specimen)")
                o["iam_a"]["counting"] = V4.counting_error(_cnt, _A, _B)
            else: o["iam_a"]["refusal"] = "no qualifying molecules or tiles in this file: IAM-A not read"
        elif qi.get("refusal_code"):
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
    if beta is not None:   # 2026-10-23 (DEV-REPLICATE-01 R2, R3): the calibrated value at a few sites known from the literature, recorded so a replay
        # can check the chain's intake and calibration against a known effect; nothing is scored from them and no reading uses them
        o["control_sites"] = {"flag": "recorded only: calibrated values at known control sites, not a reading",
                              "values": {c: (round(float(beta[c]), 4) if c in beta and beta[c] == beta[c] else None) for c in CONTROL_SITES},
                              "sites": CONTROL_SITES_ABOUT}
    if intake is not None: intake.setdefault("covariates", {}).update(cov)
    if stage1_meta: o["stage1"] = {k: stage1_meta.get(k) for k in ("detection", "n_cpgs", "pipeline")}
    led = a.ledger or os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "evidence_ledger.jsonl")
    o["run_id"] = _assign_run_id(led)
    o["versions"] = _versions(ENG)   # every frozen input and chain module, with a short hash, so runs months apart are poolable
    typed = sid; hid = _hash_id(sid)
    _rtyped = typed
    if _id_collides(typed):   # 2026-10-24: the redaction below replaces every whole-word match of the id; an id that is a cell name would erase that name
        _rtyped = None; o.setdefault("safeguards", []).append("the sample id equals a cell name the chain uses; it is hashed in the id field only, and the text is not searched for it")
    r = R3.build(o, a.out, typed)    # the printed report keeps the operator's typed id (author decision B, 2026-10-04)
    _gf = o.pop("geo_fetch", None)   # 2026-10-22: a public GEO accession is the file's source address, not a person's id; kept as fetched so it can be re-fetched and checked
    o = _redact(o, _rtyped, hid); o["sample_id"] = hid; o["sample_id_hashed"] = True   # bundle and ledger carry the hash only
    if _gf: o["geo_fetch"] = _gf
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
