#!/usr/bin/env python3
"""release_check_v3.py - the release check for chain v3. Exit 0 only when every check PASSES; a check that cannot run is a FAIL,
never a pass and never silently skipped.

    python3 release_check_v3.py            all checks (needs methylprep 1.7.1 and its manifest cache for the IDAT runs)
    python3 release_check_v3.py --json OUT also write the result table as JSON (kit/results/release_check.json by default)

Checks
  F1  frozen inputs: every file in FROZEN_INPUTS_v3.json hashes to its recorded sha256, and agrees with the independent record of
      the PROC-REPL-V3-01 run (doors/data/DEV_REPL_V3_01_run/COMMANDS.md) wherever that record names the file
  S1  static: no Python file under Biological_Physics/ or docs/ imports a module of the retired class-floor engine, and every module
      a file loads by path (importlib spec_from_file_location / _find / _load_module) exists in the tree
  S2  static: no tracked file name or text file in the repository carries the old private name of the deconvolver
  S3  chain_sequence.json matches the code (build_chain_sequence.py --check)
  S4  every toolkit module imports (chain/TOOLKIT.md); build-time scripts are byte-compiled
  E1  v3 end to end from a bundled EPIC v1 IDAT pair (TEST_DATA/idats/GSM8772491, a colon array declared as whole blood for this check),
      with no age and no sex given (optional since 2026-10-04): Stage 0 PROCEED, sex check NOT_DECLARED, Stage 1 runs, the platform check
      refuses the incomplete vector (fewer than 700,000 detected probes), report and bundle written
  E2  v3 end to end from a bundled 450K IDAT pair (GSM2333901): Stage 0 and Stage 1 run, Stage 0.7b quarantines the array (too few
      of the EPIC neutrophil sites), the run exits 2 and writes nothing
  E3  a constructed whole-blood specimen through run_sample.py --betas: the bundled EPIC array calibrated without the detection mask,
      its 963 composition markers and 6,000 neutrophil sites replaced by a known mixture of the frozen purified-group profiles
      (NEU 0.60): Stage A must recover the fractions, Met-A must equal 1 (the specimen IS its composition-matched expectation), and
      Stage T must tare against three references to A_rel = 1. Since 2026-10-04 the specimen's self-tare II fixed sites are also set to
      the reference scale (ref_value of Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json; see E3)
  E4  Stage Q through run_sample.py --site-table: a constructed per-site table whose copy error equals the healthy position reads
      IAM-A = 1 (Normal); the same table under another pipeline is refused; the .pat extractor reads a constructed .pat file
  E5  stage 12b (wired 2026-10-03): two constructed draws of one person through run_sample.py --prior-betas give the per-address difference;
      a draw with another identifier hash is refused
  E6  specimen rule (author decision L, 2026-10-04): the bundled IDAT pair as "tissue" and the constructed beta table as "PBMC" and "cell line"
      are refused at intake with a report naming the specimen and no reading; Stage 1 does not run on the refused IDAT pair
  E7  identifiers hashed (decision B): after E3, neither the bundle nor the ledger carries the typed id CONSTRUCTED_NEU60; the report title does
  E8  noise-site coverage (decision A): the E3 specimen with 10,000 noise sites removed (below 90 %) gets A as a number, no gauge state, no
      gauge drawn, and the plain explanation in the report
  E9  IAM-A C-score (decision C, DEVELOPMENT): constructed independent copy errors give C within 1 +/- 4 sqrt(2/blocks); errors clustered in one
      block of ten give C above that; the E4 report carries the C-score line
  E10 development flags (decision N): the E3 specimen with every development flag gives the same reading (A, state, A_rel, C) as without
      flags; every development block is labelled DEVELOPMENT - not commissioned (OK, or NOT_RUN with its reason where the atlas is not
      present); the report has the development section
  M1  the operations manual builds (manual/build_manual_v3.py)
"""
import ast, gzip, hashlib, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(MP))
RS = os.path.join(HERE, "MethylPhys_Interface", "run_sample.py"); IDAT = os.path.join(HERE, "TEST_DATA", "idats")
PY = sys.executable
# serial_mode is NOT retired: it is toolkit stage 12b (chain/serial_mode.py, chain/TOOLKIT.md) - removed from this list 2026-10-03
RETIRED_MODULES = {"cpg_conductor", "cpg_gauge_engine", "cpg_tiers", "cpg_gauge", "disease_matching", "propagate", "build_methylphys",
                   "build_chain_inventory", "build_percell_reference", "iamatlas_a_scoring", "synthetic_patient_generator",
                   "stage_5_second_chain", "stage_1_calibration", "val_finding", "preflight", "idat_parse",
                   "idat_decoder_pure", "build_run_index", "file_run", "cpg_kit", "lab_zero", "om_data", "om_lib", "om_part3",
                   "build_operations_manual", "sop_stage_links", "sop_repoint", "build_sop_mirror"}
OLD_NAME = "".join(map(chr, (119, 97, 108, 116, 104, 101, 114)))   # the retired private name, spelt so this file does not carry it
R = []


def rec(cid, ok, detail):
    R.append({"check": cid, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {cid:<4} {detail}", flush=True)
    return ok


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def tracked():
    r = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    if r.returncode: raise RuntimeError("git ls-files failed: " + r.stderr.strip()[:200])
    return [f for f in r.stdout.split("\n") if f]


def run(cmd, cwd=None, timeout=900):
    r = subprocess.run(cmd, cwd=cwd or os.path.dirname(RS), capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONPATH": HERE + os.pathsep + os.environ.get("PYTHONPATH", "")})
    return r.returncode, r.stdout + r.stderr


def F1():
    fz = json.load(open(os.path.join(HERE, "FROZEN_INPUTS_v3.json")))["files"]
    bad = [f for f, h in fz.items() if not os.path.exists(os.path.join(HERE, f)) or sha(os.path.join(HERE, f)) != h]
    rec("F1", not bad, f"{len(fz) - len(bad)} of {len(fz)} frozen inputs match FROZEN_INPUTS_v3.json" + (f"; DIFFER: {bad}" if bad else ""))
    cmd = os.path.join(MP, "doors", "data", "DEV_REPL_V3_01_run", "COMMANDS.md")
    if not os.path.exists(cmd): return rec("F1b", False, "independent record COMMANDS.md not found")
    run_rec = dict(re.findall(r"#\s+(\S+\.(?:json|csv))\s+([0-9a-f]{64})", open(cmd, encoding="utf-8").read()))
    both = {os.path.basename(f): h for f, h in fz.items() if os.path.basename(f) in run_rec}
    sup = json.load(open(os.path.join(HERE, "FROZEN_INPUTS_v3.json"))).get("superseded", {})
    canon = lambda x: hashlib.sha256(json.dumps({k: v for k, v in x.items() if k != "_meta"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    diff, meta_only = [], []
    for b, h in both.items():
        if run_rec[b] == h: continue
        f = next((f for f in fz if os.path.basename(f) == b), None); s_ = sup.get(f, {})
        # a file whose only change since the run is an added _meta block: the run record must match the pre-change bytes and the data keys must be unchanged
        if s_.get("sha256_before") == run_rec[b] and canon(json.load(open(os.path.join(HERE, f)))) == s_.get("data_sha256_canonical"): meta_only.append(b)
        else: diff.append(b)
    rec("F1b", bool(both) and not diff, f"{len(both) - len(diff)} of {len(both)} agree with the PROC-REPL-V3-01 run record"
        + (f" ({len(meta_only)} by data keys: _meta added since the run: {meta_only})" if meta_only else "") + (f"; DIFFER: {diff}" if diff else ""))


def S1(files):
    hits, missing, n = [], [], 0
    names = {os.path.basename(f) for f in files}
    for f in files:
        if not f.endswith(".py") or not f.startswith(("Biological_Physics/", "docs/")): continue
        try: src = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read(); t = ast.parse(src)
        except (OSError, SyntaxError): continue
        n += 1
        for node in ast.walk(t):
            mods = []
            if isinstance(node, ast.Import): mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level: mods = [node.module.split(".")[0]]
            hits += [f"{f}: import {m}" for m in mods if m in RETIRED_MODULES]
            if isinstance(node, ast.Call):
                fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if fn in ("spec_from_file_location", "_find", "_load_module"):
                    for a in ast.walk(node):
                        if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.endswith(".py"):
                            b = os.path.basename(a.value)
                            if b[:-3] in RETIRED_MODULES: hits.append(f"{f}: loads {b}")
                            elif b not in names: missing.append(f"{f}: loads {b} (not in the tree)")
    rec("S1", not hits and not missing, f"{n} Python files scanned; retired imports: {len(hits)}; path-loaded modules missing: {len(missing)}"
        + (f" -> {(hits + missing)[:6]}" if hits or missing else ""))


def S2(files):
    pat = re.compile(OLD_NAME, re.I); bad = [f for f in files if pat.search(f)]
    for f in files:
        p = os.path.join(ROOT, f)
        try:
            if os.path.getsize(p) > 50_000_000: continue
            b = open(p, "rb").read()
        except OSError: continue
        if b"\0" in b[:4096]: continue
        if pat.search(b.decode("utf-8", "replace")): bad.append(f)
    rec("S2", not bad, f"{len(files)} tracked files scanned; files carrying the retired name: {len(bad)}" + (f" -> {bad[:8]}" if bad else ""))


def S3():
    code, out = run([PY, "build_chain_sequence.py", "--check"], cwd=HERE)
    rec("S3", code == 0, out.strip().splitlines()[-1] if out.strip() else "no output")


def S4():
    """Each toolkit module named in TOOLKIT.md imports in a fresh interpreter; a build-time script (build_* / generate_*, which does its
    work at module level) is byte-compiled instead of run."""
    mods = sorted(set(re.findall(r"`(chain/[^`]+?\.py)`", open(os.path.join(HERE, "TOOLKIT.md"), encoding="utf-8").read())))
    bad, n_compiled = [], 0
    for m in mods:
        p = os.path.join(MP, m)
        if os.path.basename(p).startswith(("build_", "generate_")):
            r = subprocess.run([PY, "-m", "py_compile", p], capture_output=True, text=True); n_compiled += 1
        else:
            code = ("import importlib.util as u, sys; sys.path[:0] = [%r, %r]; s = u.spec_from_file_location('m', %r); "
                    "x = u.module_from_spec(s); sys.modules['m'] = x; s.loader.exec_module(x)") % (HERE, os.path.dirname(p), p)
            r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=300)
        if r.returncode: bad.append(f"{m}: {r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else r.returncode}")
    rec("S4", bool(mods) and not bad, f"{len(mods) - len(bad)} of {len(mods)} toolkit modules import ({n_compiled} build-time scripts byte-compiled)"
        + (f"; FAILED: {bad}" if bad else ""))


def _bundle(out_html):
    p = os.path.splitext(out_html)[0] + "_bundle.json"
    return json.load(open(p)) if os.path.exists(p) else None


def E1(tmp):
    g, r_ = (os.path.join(IDAT, "GSM8772491_%s.idat.gz" % c) for c in ("Grn", "Red"))
    out = os.path.join(tmp, "E1.html")
    code, log = run([PY, RS, "--grn", g, "--red", r_, "--specimen", "whole blood", "--id", "GSM8772491",
                     "--array-type", "EPIC_v1", "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    b = _bundle(out) or {}; it = b.get("intake") or {}
    ok = (code == 0 and os.path.exists(out) and it.get("stage0_verdict") in ("PROCEED", "PROCEED_WITH_PENALTY") and it.get("sex_check") == "NOT_DECLARED"
          and (b.get("stage1") or {}).get("n_cpgs") and "EPIC v1 arrays only" in str(b.get("refusal")) and b.get("array_type") == "EPIC_v1")
    rec("E1", ok, f"exit {code}; Stage 0 {it.get('stage0_verdict')} (sex check {it.get('sex_check')}); Stage 1 {(b.get('stage1') or {}).get('n_cpgs')} CpGs; "
        f"refusal: {str(b.get('refusal'))[:90]}" + ("" if ok else " | log tail: " + " / ".join(log.strip().splitlines()[-3:])[:300]))


def E2(tmp):
    g, r_ = (os.path.join(IDAT, "GSM2333901_%s.idat.gz" % c) for c in ("Grn", "Red"))
    out = os.path.join(tmp, "E2.html")
    code, log = run([PY, RS, "--grn", g, "--red", r_, "--specimen", "whole blood", "--sex", "M", "--age", "58", "--id", "GSM2333901",
                     "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    # a 450K array carries too few of the EPIC neutrophil sites: Stage 0.7b (platform coverage, after calibration) quarantines it,
    # the run exits 2 and nothing is scored or written
    ok = code == 2 and "hm450_coverage" in log and "QUARANTINE" in log and not os.path.exists(out) and "array detected: 450k" in log
    verdict = next((l.strip() for l in log.splitlines() if l.startswith("Stage 0 verdict")), "no Stage 0 verdict")
    rec("E2", ok, f"exit {code}; {verdict[:110]}; report written: {os.path.exists(out)}"
        + ("" if ok else " | log tail: " + " / ".join(log.strip().splitlines()[-3:])[:300]))


def E3(tmp):
    import numpy as np, pandas as pd
    sys.path[:0] = [HERE]
    from stage_1_idat_calibration import calibrate_idat_to_beta
    B = json.load(open(os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "blood_composition_EPIC_v1.json")))
    b, _meta = calibrate_idat_to_beta(os.path.join(IDAT, "GSM8772491_Grn.idat.gz"), os.path.join(IDAT, "GSM8772491_Red.idat.gz"),
                                      mask_detection=False, verbose=False)
    b = (b.iloc[:, 0] if hasattr(b, "columns") else b).dropna(); b.index = b.index.astype(str)
    f = dict(zip(B["groups"], [0.0] * len(B["groups"]))); f["NEU"] = 0.60
    rest = [g for g in B["groups"] if g != "NEU"]
    for k, g in enumerate(rest): f[g] = 0.40 * (k + 1) / sum(range(1, len(rest) + 1))
    M = pd.DataFrame(B["mu_markers"], index=B["markers"]); P = pd.DataFrame(B["profiles_at_neutrophil_sites"], index=B["neutrophil_sites"])
    # Fixed sites on the reference scale (author, 2026-10-04): Stage T step 1, self-tare II, takes its anchors from the array's own fixed
    # sites, so a specimen whose substituted sites are on the reference scale must have its fixed sites on the same scale, or step 1
    # rescales sites that are already referenced (the colon array's own fixed sites gave Met-A 0.0159). Source: ref_value of
    # Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json, the mean of its six reference arrays (GSM2998021, 023, 030, 057, 116,
    # 143): the same six GSE110554 purified neutrophil arrays as metA_floors_v1_3.json and neutrophil_reference_v1_1.json. Every fixed
    # site of the self-tare II sets that the specimen carries is set; bars and expected values are unchanged.
    ST = json.load(open(os.path.join(HERE, "Runtime Matrices", "Development", "dev_selftare_typeII_EPIC_v1.json")))
    fx = pd.Series(ST["ref_value"], dtype="float64"); fx = fx[fx.index.isin(b.index)]
    b.loc[fx.index] = fx.values
    b = b.reindex(b.index.union(M.index).union(P.index))
    b.loc[M.index] = (M[B["groups"]] @ pd.Series(f)[B["groups"]]).values
    b.loc[P.index] = (P[B["groups"]] @ pd.Series(f)[B["groups"]]).values
    csv = os.path.join(tmp, "E3_constructed.csv"); b.rename("beta").to_frame().to_csv(csv, index_label="cpg_id")
    out = os.path.join(tmp, "E3.html")
    code, log = run([PY, RS, "--betas", csv, "--specimen", "whole blood", "--id", "CONSTRUCTED_NEU60", "--slide-ref-A", "1.0,1.0,1.0",
                     "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    o = _bundle(out) or {}; fr = (o.get("composition") or {}).get("fractions") or {}; m = o.get("met_a") or {}; t = o.get("tare") or {}
    err = max((abs(fr.get(g, -1) - v) for g, v in f.items()), default=1)
    ok = (code == 0 and len(b) > 700_000 and err < 1e-3 and m.get("A") is not None and abs(m["A"] - 1) < 1e-3
          and t.get("A_rel") is not None and abs(t["A_rel"] - 1) < 1e-3)
    rec("E3", ok, f"exit {code}; {len(b):,} probes; composition max |error| {err:.2e}; NEU {fr.get('NEU')}; Met-A {m.get('A')}; "
        f"A_rel {t.get('A_rel')} ({t.get('method') or t.get('state')}); noise index {m.get('noise_index')}"
        + ("" if ok else " | " + " / ".join(log.strip().splitlines()[-3:])[:300]))


def E4(tmp):
    import numpy as np, pandas as pd
    sys.path[:0] = [HERE]; import stage_q_iam_a as _Q
    pos = json.load(open(_Q.POS))   # the position file Stage Q reads (v2 since 2026-10-08), not a fixed copy
    P, e0 = pos["cells"]["neutrophils"]["P"], pos["eps0"]; pipe = pos["cells"]["neutrophils"]["pipeline"]
    H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e)); target = P * H(e0)
    lo, hi = 1e-6, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if H(mid) < target else (lo, mid)
    eps = (lo + hi) / 2; n = 2000; opp = 100
    k = int(round(eps * opp * n)); per = np.full(n, k // n); per[: k % n] += 1      # k isolated errors per half, spread over the sites
    T = pd.DataFrame({"pos": np.arange(n), "opp_A": opp, "err_A": per, "opp_B": opp, "err_B": per})
    st = os.path.join(tmp, "E4_sites.csv"); T.to_csv(st, index=False)
    out = os.path.join(tmp, "E4.html")
    code, log = run([PY, RS, "--site-table", st, "--seq-pipeline", pipe, "--id", "CONSTRUCTED_SEQ", "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    q = (_bundle(out) or {}).get("iam_a") or {}
    out2 = os.path.join(tmp, "E4b.html")
    code2, _ = run([PY, RS, "--site-table", st, "--seq-pipeline", "another_pipeline", "--id", "CONSTRUCTED_SEQ_B", "--out", out2, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    q2 = (_bundle(out2) or {}).get("iam_a") or {}
    pat = os.path.join(tmp, "E4.pat.gz")
    with gzip.open(pat, "wt") as fh:
        for i in range(300): fh.write("chr1\t%d\t%s\t1\n" % (1000 + i, "CCCTCCCCCC" if i % 10 == 0 else "CCCCCCCCCC"))
    sys.path[:0] = [HERE]; import stage_q_iam_a as Q
    PT = Q.pat_site_table(pat)
    ok = (code == 0 and q.get("A") is not None and abs(q["A"] - 1) < 5e-3 and q.get("state") == "Normal" and code2 == 0
          and q2.get("A") is None and "refusal" in q2 and len(PT) > 0 and int(PT[["err_A", "err_B"]].sum().sum()) == 30)
    rec("E4", ok, f"IAM-A {q.get('A')} ({q.get('state')}) at eps {q.get('eps')}; other pipeline: {str(q2.get('refusal'))[:70]}; "
        f".pat extractor: {len(PT)} sites, {int(PT[['err_A', 'err_B']].sum().sum())} isolated errors (30 constructed)")


def E5(tmp):
    """Stage 12b (wired 2026-10-03, DEV-TOOLKIT-ADDED-01): two draws of one constructed person through run_sample.py --prior-betas; the
    difference is computed for the same identifier hash and refused for a different one."""
    import numpy as np, pandas as pd
    B = json.load(open(os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "blood_composition_EPIC_v1.json")))
    idx = [f"cgX{i:07d}" for i in range(720000)] + B["markers"] + B["neutrophil_sites"]
    rng = np.random.default_rng(5); b1 = pd.Series(rng.uniform(0.05, 0.95, len(idx)), index=idx); b1 = b1[~b1.index.duplicated()]
    b2 = (b1 + rng.normal(0, 0.01, len(b1))).clip(0.001, 0.999)
    c1, c2 = os.path.join(tmp, "E5_d1.csv"), os.path.join(tmp, "E5_d2.csv")
    b1.rename("beta").to_frame().to_csv(c1, index_label="cpg_id"); b2.rename("beta").to_frame().to_csv(c2, index_label="cpg_id")
    L = ["--ledger", os.path.join(tmp, "ledger.jsonl"), "--array-type", "EPIC_v1"]
    o1, o2, o3 = (os.path.join(tmp, f"E5_{k}.html") for k in ("d1", "d2", "d3"))
    k1, _ = run([PY, RS, "--betas", c1, "--id", "E5_D1", "--patient-id", "constructed_person_A", "--out", o1, "--save-betas", os.path.join(tmp, "E5_d1.parquet")] + L)
    k2, log2 = run([PY, RS, "--betas", c2, "--id", "E5_D2", "--patient-id", "constructed_person_A", "--out", o2, "--prior-betas", os.path.join(tmp, "E5_d1.parquet"),
                    "--prior-bundle", os.path.splitext(o1)[0] + "_bundle.json"] + L)
    k3, _ = run([PY, RS, "--betas", c2, "--id", "E5_D3", "--patient-id", "constructed_person_B", "--out", o3, "--prior-betas", os.path.join(tmp, "E5_d1.parquet"),
                 "--prior-bundle", os.path.splitext(o1)[0] + "_bundle.json"] + L)
    d2 = (_bundle(o2) or {}).get("difference_map") or {}; d3 = (_bundle(o3) or {}).get("difference_map") or {}
    ok = (k1 == 0 and k2 == 0 and k3 == 0 and d2.get("status") == "OK" and abs(d2.get("median_dbeta", 1)) < 2e-3 and 0.005 < d2.get("mean_abs_dbeta", 0) < 0.012
          and d3.get("status") == "REFUSED" and "sec-difference-map" in open(o2).read())
    rec("E5", ok, f"same person: {d2.get('status')} ({d2.get('n_addresses')} addresses, mean |d beta| {d2.get('mean_abs_dbeta')}); other person: {d3.get('status')}"
        + ("" if ok else " | " + " / ".join(log2.strip().splitlines()[-3:])[:300]))


def E6(tmp):
    g, r_ = (os.path.join(IDAT, "GSM8772491_%s.idat.gz" % c) for c in ("Grn", "Red"))
    res = []
    o1 = os.path.join(tmp, "E6_tissue.html")
    k1, log1 = run([PY, RS, "--grn", g, "--red", r_, "--specimen", "tissue", "--id", "E6_TISSUE", "--out", o1, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    b1 = _bundle(o1) or {}; res.append(("tissue IDAT", k1 == 0 and b1.get("refusal_code") == "SPECIMEN_REFUSED" and "'tissue'" in str(b1.get("refusal")) and not b1.get("met_a")
                                         and "Stage 1: calibrating" not in log1 and "not run" in open(o1).read()))
    csv = os.path.join(tmp, "E3_constructed.csv")
    for sp in ("PBMC", "cell line"):
        o = os.path.join(tmp, f"E6_{sp.replace(' ', '_')}.html")
        k, _ = run([PY, RS, "--betas", csv, "--specimen", sp, "--id", "E6_" + sp.replace(" ", "_"), "--out", o, "--ledger", os.path.join(tmp, "ledger.jsonl")])
        b = _bundle(o) or {}; res.append((sp, k == 0 and b.get("refusal_code") == "SPECIMEN_REFUSED" and not b.get("met_a") and "SPECIMEN_REFUSED" in open(o).read()))
    rec("E6", all(x[1] for x in res), "; ".join(f"{n}: {'refused with a report' if ok else 'NOT refused as specified'}" for n, ok in res))


def E7(tmp):
    out = os.path.join(tmp, "E3.html"); bp = os.path.splitext(out)[0] + "_bundle.json"; led = os.path.join(tmp, "ledger.jsonl")
    bt = open(bp).read() if os.path.exists(bp) else ""; lt = open(led).read() if os.path.exists(led) else ""; ht = open(out).read() if os.path.exists(out) else ""
    ok = bool(bt) and "CONSTRUCTED_NEU60" not in bt and "CONSTRUCTED_NEU60" not in lt and "Cellular Performance Gauge - CONSTRUCTED_NEU60" in ht
    rec("E7", ok, f"typed id in bundle: {'CONSTRUCTED_NEU60' in bt}; in ledger: {'CONSTRUCTED_NEU60' in lt}; in report title: {'Cellular Performance Gauge - CONSTRUCTED_NEU60' in ht}; "
        f"bundle sample_id {json.loads(bt).get('sample_id') if bt else None}")


def E8(tmp):
    import pandas as pd
    b = pd.read_csv(os.path.join(tmp, "E3_constructed.csv"), index_col=0).iloc[:, 0]
    ns = json.load(open(os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "noise_sites_EPIC_v1.json")))["sites"]
    b = b.drop([x for x in ns[:10000] if x in b.index]); csv = os.path.join(tmp, "E8.csv"); b.rename("beta").to_frame().to_csv(csv, index_label="cpg_id")
    out = os.path.join(tmp, "E8.html")
    code, _ = run([PY, RS, "--betas", csv, "--specimen", "whole blood", "--id", "E8", "--slide-ref-A", "1.0,1.0,1.0", "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    m = (_bundle(out) or {}).get("met_a") or {}; h = open(out).read() if os.path.exists(out) else ""; sec = h.split("id='sec-met-a'")[-1].split("<h2")[0]
    ok = code == 0 and m.get("A") is not None and str(m.get("state", "")).startswith("withheld: only") and "<svg" not in sec and "noise sites were measured on this array" in h
    rec("E8", ok, f"A {m.get('A')}; noise sites {m.get('noise_sites_measured')} of {m.get('noise_sites_total')}; state withheld: {str(m.get('state', '')).startswith('withheld')}; gauge drawn: {'<svg' in sec}")


def E9(tmp):
    import numpy as np, pandas as pd
    sys.path[:0] = [HERE]; import stage_q_iam_a as Q
    rng = np.random.default_rng(9); n, opp, eps = 50000, 20, 0.035
    k = rng.binomial(opp, eps, size=(2, n)); T = pd.DataFrame({"pos": [f"chr1:{i}" for i in range(n)], "opp_A": opp, "err_A": k[0], "opp_B": opp, "err_B": k[1]})
    a = Q.cscore(T); lim = 4 * np.sqrt(2 / a["n_blocks"])
    rate = np.full(n, eps); blk = (np.arange(n) // Q.CSCORE_BLOCK_SITES) % 10 == 0; rate[blk] = 3 * eps
    k2 = rng.binomial(opp, np.vstack([rate, rate])); T2 = T.assign(err_A=k2[0], err_B=k2[1]); b = Q.cscore(T2)
    e4 = open(os.path.join(tmp, "E4.html")).read() if os.path.exists(os.path.join(tmp, "E4.html")) else ""
    ok = abs(a["C"] - 1) <= lim and b["C"] > 1 + lim and "sec-iam-a-cscore" in e4
    rec("E9", ok, f"independent C {a['C']} ({a['n_blocks']} blocks, limit 1 +/- {lim:.3f}); clustered C {b['C']}; E4 report carries the C-score line: {'sec-iam-a-cscore' in e4}")


def E10(tmp):
    csv = os.path.join(tmp, "E3_constructed.csv"); L = ["--specimen", "whole blood", "--slide-ref-A", "1.0,1.0,1.0", "--ledger", os.path.join(tmp, "ledger.jsonl")]
    o0, o1 = os.path.join(tmp, "E10_off.html"), os.path.join(tmp, "E10_on.html")
    run([PY, RS, "--betas", csv, "--id", "E10", "--out", o0] + L)
    fl = ["--dev-selftare-ii", "--dev-direction", "--dev-trace", "--dev-foreign", "--dev-brightness", "--dev-nilc", "--dev-atlas-e", "--dev-percell-b", "--dev-sky"]
    code, log = run([PY, RS, "--betas", csv, "--id", "E10", "--out", o1] + L + fl)
    a, b = _bundle(o0) or {}, _bundle(o1) or {}
    key = lambda o: ((o.get("met_a") or {}).get("A"), (o.get("met_a") or {}).get("state"), (o.get("tare") or {}).get("A_rel"), (o.get("met_a_cscore") or {}).get("C"))
    dev = b.get("development") or {}; blocks = {k: v for k, v in dev.items() if isinstance(v, dict)}
    ok = (code == 0 and key(a) == key(b) and len(blocks) == len(fl) and all(v.get("label") == "DEVELOPMENT - not commissioned" for v in blocks.values())
          and "sec-development" in open(o1).read() and all(v.get("status") in ("OK", "NOT_RUN") for v in blocks.values()))
    rec("E10", ok, f"reading identical: {key(a) == key(b)}; blocks: " + ", ".join(f"{k} {v.get('status')}" for k, v in blocks.items())
        + ("" if ok else " | " + " / ".join(log.strip().splitlines()[-3:])[:300]))


def E11(tmp):
    """Stage Q0 (IAM-A intake, 2026-10-08, development): a constructed whole-genome .pat proceeds; each negative control stops with its named
    reason - one chromosome only, a build shift, a truncated gzip, a malformed line, a whole-blood specimen. Constructed index ranges are
    used so the check does not depend on the hg19 dictionary file."""
    import gzip
    sys.path[:0] = [HERE]; import stage_q0_intake as Q0
    rp = os.path.join(tmp, "E11_ranges.json"); R = {f"chr{i}": [i * 1000000 + 1, i * 1000000 + 900000] for i in range(1, 23)}
    json.dump({"ranges": R}, open(rp, "w")); keep = Q0.RANGES; Q0.RANGES = rp
    def mk(p, chroms=range(1, 23), shift=0, cut=False, bad=False):
        with gzip.open(p, "wt") as f:
            for c in chroms:
                for j in range(50): f.write(f"chr{c}\t{c * 1000000 + 10 + j * 7 + shift}\tCCCCCCTC\t2\n")
            if bad: f.write("chr1\tx\tCCC\t1\n")
        if cut: b = open(p, "rb").read(); open(p, "wb").write(b[:len(b) - 30])
    want = {"good": ({}, None), "one_chrom": ({"chroms": [1]}, "QUARANTINE_NOT_WHOLE_GENOME"), "wrong_build": ({"shift": 950000}, "GENOME_BUILD_MISMATCH"),
            "truncated": ({"cut": True}, "QUARANTINE_UNREADABLE_PAT"), "malformed": ({"bad": True}, "QUARANTINE_UNREADABLE_PAT")}
    got = {}
    try:
        for k, (kw, code) in want.items():
            p = os.path.join(tmp, f"E11_{k}.pat.gz"); mk(p, **kw); got[k] = Q0.intake(p, "blood granulocytes").get("refusal_code")
        got["whole_blood"] = Q0.intake(os.path.join(tmp, "E11_good.pat.gz"), "whole blood").get("refusal_code")
    finally:
        Q0.RANGES = keep
    ok = all(got[k] == c for k, (_, c) in want.items()) and got["whole_blood"] == "SPECIMEN_REFUSED"
    rec("E11", ok, "; ".join(f"{k}: {v or 'proceeds'}" for k, v in got.items()))


def M1(tmp):
    out = os.path.join(tmp, "manual.pdf")
    code, log = run([PY, os.path.join(MP, "manual", "build_manual_v3.py"), out], cwd=os.path.join(MP, "manual"))
    rec("M1", code == 0 and os.path.exists(out) and os.path.getsize(out) > 2000, log.strip().splitlines()[-1][:160] if log.strip() else "no output")


def main():
    t0 = time.time()
    print(f"release_check_v3 @ {subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}  python {PY}")
    files = tracked()
    tmp = tempfile.mkdtemp(prefix="rc_v3_")
    for name, fn, args in (("F1", F1, ()), ("S1", S1, (files,)), ("S2", S2, (files,)), ("S3", S3, ()), ("S4", S4, ()),
                           ("E1", E1, (tmp,)), ("E2", E2, (tmp,)), ("E3", E3, (tmp,)), ("E4", E4, (tmp,)), ("E5", E5, (tmp,)), ("E6", E6, (tmp,)), ("E7", E7, (tmp,)), ("E8", E8, (tmp,)), ("E9", E9, (tmp,)), ("E10", E10, (tmp,)), ("E11", E11, (tmp,)), ("M1", M1, (tmp,))):
        try: fn(*args)
        except Exception as e: rec(name, False, f"could not run: {type(e).__name__}: {str(e)[:200]}")
    n_fail = sum(r["result"] != "PASS" for r in R)
    verdict = "PASS" if not n_fail else f"FAIL ({n_fail} of {len(R)} checks)"
    print(f"RELEASE CHECK v3: {verdict}  ({len(R)} checks, {time.time() - t0:.0f} s; outputs in {tmp})")
    if True:   # the result table is always written
        jp = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else os.path.join(MP, "kit", "results", "release_check.json")
        json.dump({"check": "release_check_v3", "verdict": verdict, "checks": R, "seconds": round(time.time() - t0)}, open(jp, "w"), indent=1)
    sys.exit(0 if not n_fail else 1)


if __name__ == "__main__":
    main()
