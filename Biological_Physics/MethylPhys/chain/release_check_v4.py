#!/usr/bin/env python3
"""release_check_v4.py - the release check for chain v4, the first commissioned chain. Exit 0 only when every check PASSES; a check that
cannot run is a FAIL.  One command after a fresh clone:  python3 Biological_Physics/MethylPhys/chain/release_check_v4.py

Runs every v3 check (release_check_v3.py: intake, calibration, Met-A, C-Score, tare, refusals, identifiers, development flags, manual),
with F1 reading FROZEN_INPUTS_v4.json and S2 leaving verbatim run transcripts (doors/data/*/RECOVERED_CELLS/) out, plus:
  F2  the atlas v2.1 compact copy: every table hashes to compact_manifest.json
  V1  cell shares: a planted mixture of the six blood profiles through run_sample.py --betas; each commissioned cell's share within 0.03
      of what was planted, B and NK printed as development, the report carries the shares section
  V2  IAM-A: the 13 committed grand-test counts (doors/data/DEV_IAMA_GRAND_01/out) through Stage Q2 give the committed readings exactly
      (same samples read, same A to 1e-9, same refusals); a constructed .pat through run_sample.py reaches Stage Q2 and is refused with a reason
  V3  the C-Score sky: a planted map with runs of departing neighbors is drawn, and the NESTED layout keeps neighbors together (median
      distance between consecutive sites under a tenth of the median distance between random sites)
  V4  the v4 report: the front sections (at a glance, shares, C-Score with its sky) and every v3 section id are present
  V8  buffy coat: read on the whole-blood path, labeled as its own specimen, preliminary on Met-A and the C-Score (2026-10-18)
  V5  the same-run reference table reaches the C-Score tare (C_rel = C / median of three references) and both report PDFs are written (C26)
  V6  a planted clustered departure gives a full report: C_rel above the band, the CSCORE_CLUSTERED flag, both PDFs (C27)
  V7  IAM-A from an aligned BAM through the one command (--bam; pysam pinned): exit 0, report written, molecules counted (C28)
  V9  every Stage 0 stop is explained (author rule 2026-10-20): for each way Stage 0 can stop an array, a constructed record is stopped AND
      carries an explanation with what was measured, the limit, where the limit comes from, what it means and what to do; and an array too
      thin to read gives Met-A, the tare and the C-Score each a reason with the measured count and what is required"""
import os, sys, json, gzip, glob, hashlib, re, time, subprocess, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
import release_check_v3 as R3
MP = R3.MP; RS = R3.RS; PY = R3.PY; R = R3.R; rec = R3.rec

def F1():
    fz = json.load(open(os.path.join(HERE, "FROZEN_INPUTS_v4.json")))["files"]
    bad = [f for f, h in fz.items() if not os.path.exists(os.path.join(HERE, f)) or R3.sha(os.path.join(HERE, f)) != h]
    rec("F1", not bad, f"{len(fz) - len(bad)} of {len(fz)} frozen inputs match FROZEN_INPUTS_v4.json" + (f"; DIFFER: {bad}" if bad else ""))

def F1b():   # the v3 run-record agreement (F1b of release_check_v3); its own F1 row (the v3 list) is not repeated
    orig = R3.rec; R3.rec = lambda cid, ok, d: orig(cid, ok, d) if cid != "F1" else ok
    try: R3.F1()
    finally: R3.rec = orig

def S2(files): return R3.S2([f for f in files if "/RECOVERED_CELLS/" not in f])

def F2():
    A = os.path.join(MP, "atlas", "v2.1"); M = json.load(open(os.path.join(A, "compact_manifest.json")))
    T = {f: v for f, v in M.items() if f.endswith(".parquet") and isinstance(v, dict)}
    bad = [f for f, v in T.items() if not os.path.exists(os.path.join(A, f)) or R3.sha(os.path.join(A, f)) != v["sha256"]]
    rec("F2", bool(T) and not bad, f"{len(T) - len(bad)} of {len(T)} atlas v2.1 tables match compact_manifest.json" + (f"; DIFFER: {bad}" if bad else ""))

def V1(tmp):
    import stage_deconv_v21 as D21
    Rf = D21.reference(D21.BLOOD6, spread="mean_only"); mu = Rf["mu"]; f = np.array([0.55, 0.10, 0.07, 0.14, 0.10, 0.04])
    x = mu.fillna(0.5).values.astype(float) @ f; x = np.clip(x + np.random.default_rng(11).normal(0, 0.01, len(x)), 0.001, 0.999)
    p = os.path.join(tmp, "V1_betas.csv"); pd.DataFrame({"cpg_id": mu.index, "beta": x}).to_csv(p, index=False)
    out = os.path.join(tmp, "V1.html"); code, log = R3.run([PY, RS, "--betas", p, "--specimen", "whole blood", "--id", "PLANTED_MIX", "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    sh = (R3._bundle(out) or {}).get("cell_shares") or {}; got = {r["cell"]: r for r in sh.get("cells", [])}; truth = dict(zip(D21.BLOOD6, f))
    com = D21.DECONV_V21_COMMISSIONING["cells"]; err = {c: round(float(abs(got[c]["share"] - truth[c])), 4) for c in com if c in got}
    ok = (code == 0 and len(err) == len(com) and max(err.values()) <= 0.03 and all(got[c]["status"] == "development" for c in D21.BLOOD6 if c not in com)
          and "id='v4-shares'" in (open(out).read() if os.path.exists(out) else ""))
    rec("V1", ok, f"exit {code}; commissioned-cell errors {err}" + ("" if code == 0 else " | " + log[-200:]))

def V2(tmp):
    import stage_v4 as V4, csv
    D = os.path.join(MP, "doors", "data", "DEV_IAMA_GRAND_01"); G = json.load(open(os.path.join(D, "grand_scored.json")))
    S = {r["runs"]: r for r in csv.DictReader(open(os.path.join(D, "samples.csv")))}; mism = []; n_read = 0
    want = {r["sample"]: r for r in G["rows"]}
    for f in sorted(glob.glob(os.path.join(D, "out", "SRR*.json"))):
        o = json.load(open(f)); s = S[o["run"]]; q = V4.stage_q2(o, s["shape_type"]); w = want.get(s["gsm"])
        if w is None: mism.append(f"{s['gsm']} not in grand_scored"); continue
        if bool(q.get("A") is not None) != bool(w["read"]): mism.append(f"{s['gsm']} read {q.get('A') is not None} vs {w['read']}")
        elif w["read"]:
            n_read += 1
            if abs(q["A"] - round(w["A"], 4)) > 1e-9: mism.append(f"{s['gsm']} A {q['A']} vs {w['A']}")
    pat = os.path.join(tmp, "V2.pat.gz")
    with gzip.open(pat, "wt") as fh:
        for c in range(1, 23):
            for i in range(40): fh.write("chr%d\t%d\t%s\t1\n" % (c, 10_000 * c + i * 7, "CCCCCCCC"))
    out = os.path.join(tmp, "V2.html"); code, log = R3.run([PY, RS, "--pat", pat, "--seq-cell", "cd8 t cells", "--specimen", "sorted cd8 t cells", "--id", "CONSTRUCTED_PAT", "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    q = (R3._bundle(out) or {}).get("iam_a") or {}
    ok = not mism and n_read == G["read"] and code == 0 and q.get("stage") == "Q2" and q.get("A") is None and bool(q.get("refusal"))
    rec("V2", ok, f"{n_read} of {G['read']} grand-test readings reproduced exactly, {len(mism)} differ {mism[:3]}; constructed .pat: stage {q.get('stage')}, "
        f"refused: {str(q.get('refusal'))[:90]}" + ("" if code == 0 else " | exit %d %s" % (code, log[-200:])))

def V3(tmp):
    import stage_v4 as V4
    rng = np.random.default_rng(5); z = pd.Series(rng.normal(0, 1, 6000))
    for s0 in rng.choice(5900, 20, replace=False): z.iloc[s0:s0 + 40] += 3
    k = V4.cscore_sky(z, os.path.join(tmp, "V3_sky.png"))
    nside = k.get("nside", 16); npix = 12 * nside * nside; pix = (np.arange(6000) * npix) // 6000
    th, ph = V4.nest_pix2ang(nside, pix); v = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    d_next = np.median(np.arccos(np.clip((v[:-1] * v[1:]).sum(1), -1, 1))); j = rng.permutation(6000)
    d_rand = np.median(np.arccos(np.clip((v * v[j]).sum(1), -1, 1)))
    rec("V3", bool(k.get("drawn")) and d_next < 0.1 * d_rand, f"drawn {k.get('drawn')} (NSIDE {nside}, NESTED); median distance consecutive {d_next:.4f} rad vs random {d_rand:.3f} rad")

def V4_(tmp):
    h = open(os.path.join(tmp, "V1.html")).read() if os.path.exists(os.path.join(tmp, "V1.html")) else ""
    need = ["v4-glance", "v4-shares", "v4-cscore"] + list(R3.__dict__.get("REPORT_SECTIONS_V3", ())) + ["sec-met-a", "sec-cscore", "sec-red-flags", "sec-safeguards", "sec-integrity", "sec-run-yourself"]
    miss = [s for s in need if f"id='{s}'" not in h]
    rec("V4", bool(h) and not miss and "data:image/png;base64" in h, f"{len(need) - len(miss)} of {len(need)} sections present" + (f"; missing {miss}" if miss else "") + f"; sky image embedded: {'data:image/png;base64' in h}")

def V5(tmp):
    """The same-run tare reaches the C-Score through the one-command run (LESSONS C26, 2026-10-18): a reference table with columns A and C,
    three rows, must give C_rel = C / median(C of the references), and the report must carry the PDFs."""
    p = os.path.join(tmp, "V1_betas.csv"); t = os.path.join(tmp, "V5_refs.csv"); open(t, "w").write("id,A,C\nr1,1.00,1.10\nr2,1.01,1.20\nr3,0.99,1.15\n")
    out = os.path.join(tmp, "V5.html"); code, log = R3.run([PY, RS, "--betas", p, "--specimen", "whole blood", "--id", "PLANTED_MIX_TARED", "--ref-table", t, "--out", out,
                                                         "--ledger", os.path.join(tmp, "ledger.jsonl")])
    c = (R3._bundle(out) or {}).get("met_a_cscore") or {}; want = round(c["C"] / 1.15, 4) if c.get("C") is not None else None
    pdf = all(os.path.exists(os.path.splitext(out)[0] + f"_{v}.pdf") for v in ("researcher", "plain"))
    rec("V5", code == 0 and want is not None and c.get("C_rel") == want and c.get("C_n_refs") == 3 and pdf,
        f"C {c.get('C')} | C_rel {c.get('C_rel')} (want {want}) from {c.get('C_n_refs')} same-run references | both PDFs written: {pdf}")

def V6(tmp):
    """A clustered departure produces a full report (LESSONS C27, 2026-10-18): clustered loss planted by the C-Score's own positive-control rule
    (10 % of its sites in 20 contiguous stretches, 10 % toward 0.5) into the V1 sample, read against the V5 references, must give a report and
    both PDFs, a C_rel above the band and the CSCORE_CLUSTERED flag. Until 2026-10-18 the flag's level was missing from the report's order and
    every above-band reading crashed the report."""
    src = os.path.join(tmp, "V1_betas.csv"); p = os.path.join(tmp, "V6_betas.csv"); t = os.path.join(tmp, "V5_refs.csv")
    PL = os.path.join(MP, "doors", "data", "REPORT_EXAMPLES_V4", "plant_clustered.py")
    R3.run([PY, PL, src, p, HERE]); out = os.path.join(tmp, "V6.html")
    code, log = R3.run([PY, RS, "--betas", p, "--specimen", "whole blood", "--id", "PLANTED_CLUSTERED", "--ref-table", t, "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    c = (R3._bundle(out) or {}).get("met_a_cscore") or {}; h = open(out).read() if os.path.exists(out) else ""
    pdf = all(os.path.exists(os.path.splitext(out)[0] + f"_{v}.pdf") for v in ("researcher", "plain"))
    rec("V6", code == 0 and (c.get("C_rel") or 0) > 1.152 and "CSCORE_CLUSTERED" in h and pdf,
        f"exit {code}; C_rel {c.get('C_rel')} ({c.get('C_call')}); flag in report: {'CSCORE_CLUSTERED' in h}; both PDFs: {pdf}")

def V7(tmp):
    """IAM-A from an aligned BAM through the one command (LESSONS C28, 2026-10-18): a small planted Bismark-style BAM (every autosome present;
    methylated CpG runs with planted isolated unwritten sites) and its FASTA, read with --bam, must exit 0, write the report and count
    qualifying molecules. Until 2026-10-18 pysam, which --bam needs, was missing from requirements.txt and no check used this route."""
    import pysam
    rng = np.random.default_rng(7); D = os.path.join(tmp, "V7"); os.makedirs(D, exist_ok=True)
    seq = list(rng.choice(list("AT"), 400_000))
    for i in range(1000, 399_000, 25): seq[i], seq[i + 1] = "C", "G"
    ref = "".join(seq); fa = os.path.join(D, "ref.fa")
    with open(fa, "w") as f:
        f.write(">chr1\n" + ref + "\n")
        for c in range(2, 23): f.write(f">chr{c}\n" + "A" * 2000 + "\n")
    pysam.faidx(fa)
    hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(ref)}] + [{"SN": f"chr{c}", "LN": 2000} for c in range(2, 23)]}
    bam = os.path.join(D, "planted.bam"); starts = np.sort(rng.integers(1000, len(ref) - 300, 20_000))
    with pysam.AlignmentFile(bam, "wb", header=hdr) as B:
        for k, st in enumerate(starts):
            st = int(st); L = 150; q = ref[st:st + L]
            xm = ["Z" if q[j] == "C" and j + 1 < L and q[j + 1] == "G" else "." for j in range(L)]
            for j in range(L):
                if xm[j] == "Z" and rng.random() < 0.04: xm[j] = "z"
            a = pysam.AlignedSegment(); a.query_name = f"r{k}"; a.query_sequence = q; a.flag = 0; a.reference_id = 0; a.reference_start = st
            a.mapping_quality = 40; a.cigarstring = f"{L}M"; a.query_qualities = pysam.qualitystring_to_array("I" * L)
            a.set_tags([("XM", "".join(xm)), ("XG", "CT"), ("XR", "CT")]); B.write(a)
    pysam.index(bam); out = os.path.join(tmp, "V7.html")
    code, log = R3.run([PY, RS, "--bam", bam, "--ref-fasta", fa, "--assembly", "hg38", "--workers", "2", "--seq-cell", "cd8 t cells", "--specimen", "sorted cells",
                        "--id", "PLANTED_BAM", "--out", out, "--ledger", os.path.join(tmp, "ledger.jsonl")])
    q = (R3._bundle(out) or {}).get("iam_a") or {}; n = (q.get("count") or {}).get("held_opp")
    rec("V7", code == 0 and os.path.exists(out) and bool(n), f"exit {code}; report written: {os.path.exists(out)}; held-error opportunities counted: {n}; "
        f"A {q.get('A')} ({q.get('state') or q.get('refusal', '')})"[:300])

def V8(tmp):
    """Buffy coat (2026-10-18): accepted at intake, read on the whole-blood path, labeled as its own specimen. Met-A and the C-Score carry no
    commissioned label on it; the report prints the preliminary line for both; both PDFs are written."""
    import html as _h, re as _re
    p = os.path.join(tmp, "V1_betas.csv"); t = os.path.join(tmp, "V5_refs.csv")
    if not os.path.exists(t): open(t, "w").write("id,A,C\nr1,1.00,1.10\nr2,1.01,1.20\nr3,0.99,1.15\n")
    out = os.path.join(tmp, "V8.html"); code, log = R3.run([PY, RS, "--betas", p, "--specimen", "buffy coat", "--id", "PLANTED_BUFFY", "--ref-table", t, "--out", out,
                                                         "--ledger", os.path.join(tmp, "ledger.jsonl")])
    b = R3._bundle(out) or {}; m = b.get("met_a") or {}; c = b.get("met_a_cscore") or {}
    txt = _h.unescape(_re.sub(r"<[^>]+>", " ", open(out, encoding="utf-8").read())) if os.path.exists(out) else ""
    lines = ("preliminary until Met-A is officially commissioned on buffy coat" in txt or "Met-A is not yet officially commissioned on buffy coat" in txt,   # withheld form
             "preliminary until the C-Score is officially commissioned on buffy coat" in txt)
    pdf = all(os.path.exists(os.path.splitext(out)[0] + f"_{v}.pdf") for v in ("researcher", "plain"))
    ok = code == 0 and m.get("specimen") == "buffy coat" and not m.get("commissioning") and not c.get("commissioning") and all(lines) and pdf
    rec("V8", ok, f"exit {code} | specimen {m.get('specimen')} | Met-A commissioned label: {bool(m.get('commissioning'))} | C-Score: {bool(c.get('commissioning'))} | preliminary lines {lines} | PDFs {pdf}")

def V9(tmp):
    import stage_0_intake as S0
    ok_rec = dict(status="IDAT_OK", integrity_status="INTEGRITY_OK", ctrl_qc="PASS", ctrl_qc_metrics={}, detection_qc="PASS", call_rate_status="CALL_RATE_PASS",
                  call_rate=0.99, hm450_coverage_gate="NA_EPIC", sex_check="MATCH", bead_qc="PASS")
    hyb = S0.validate_control_probes(dict(bisulfite_conversion_I_median=3e4, bisulfite_conversion_II_median=1e3, hyb_high_median=7300., hyb_low_median=21170., extension_meth_median=1e4, extension_unmeth_median=1e4))
    ext = S0.validate_control_probes(dict(bisulfite_conversion_I_median=3e4, bisulfite_conversion_II_median=1e3, hyb_high_median=2e4, hyb_low_median=5e3, extension_meth_median=1e4, extension_unmeth_median=1e3))
    cases = {"intake files": dict(status="QUARANTINE_MISSING_CHANNEL", flags=["MISSING_CHANNEL:red"]), "integrity": dict(integrity_status="RE_TRANSMISSION_DETECTED"),
             "hybridization controls": dict(ctrl_qc=hyb["ctrl_qc"], ctrl_qc_metrics=hyb["metrics"]), "extension controls": dict(ctrl_qc=ext["ctrl_qc"], ctrl_qc_metrics=ext["metrics"]),
             "detection": dict(detection_qc="FAIL_LOW_DETECTION", pct_probes_detected_p_le_01=0.80), "detection not measured": dict(detection_qc="DEFERRED_PENDING_STAGE1_DECODER"),
             "call rate": dict(call_rate_status="CALL_RATE_FAIL", call_rate=0.85), "450K coverage": dict(hm450_coverage_gate="FAIL", hm450_reference_coverage=0.5),
             "sex": dict(sex_check="MISMATCH", declared_sex="F", predicted_sex="M")}
    bad = []
    for name, chg in cases.items():
        r = dict(ok_rec, **chg); S0.step_0_9_decision_gate(r); X = r.get("stage0_explained") or []
        full = [x for x in X if x.get("outcome", "").startswith("stopped") and all(str(x.get(k) or "").strip() for k in ("measured", "limit", "limit_source", "means", "what_to_do"))]
        if r.get("stage0_verdict") != "QUARANTINE" or not full: bad.append(f"{name}: verdict {r.get('stage0_verdict')}, {len(full)} full explanation(s)")
    r = dict(ok_rec); S0.step_0_9_decision_gate(r); clean = r.get("stage0_verdict") == "PROCEED" and not r.get("stage0_explained")
    import conductor_v3 as C3_, re as _re   # downstream: a thin EPIC-sized vector; Met-A, the tare and the C-Score each say why, with the measured count
    _ix = [f"cg{i:08d}" for i in range(860000)]; _o = C3_.run_neutrophil(pd.Series(np.random.default_rng(1).uniform(0.05, 0.95, len(_ix)), index=_ix), specimen="whole blood")
    for k in ("met_a", "tare", "met_a_cscore"):
        why = str((_o.get(k) or {}).get("reason") or "")
        if not _re.search(r"\d+ of \d+", why) or "required" not in why: bad.append(f"{k} gives no measured reason: {why[:80]!r}")
    rec("V9", not bad and clean and hyb["ctrl_qc"].startswith("FAIL") and ext["ctrl_qc"].startswith("FAIL"),
        f"{len(cases)} kinds of Stage 0 stop and 3 downstream stops (Met-A, tare, C-Score) checked; {len(bad)} unexplained; a clean array proceeds with nothing to explain: {clean}" + (f"; MISSING: {bad}" if bad else ""))

def main():
    t0 = time.time(); files = R3.tracked(); tmp = tempfile.mkdtemp(prefix="rc_v4_")
    print(f"release_check_v4 @ {subprocess.run(['git', '-C', R3.ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}  python {PY}")
    steps = [("F1", F1, ()), ("F1b", F1b, ()), ("F2", F2, ()), ("S1", R3.S1, (files,)), ("S2", S2, (files,)), ("S3", R3.S3, ()), ("S4", R3.S4, ())] + \
            [(n, getattr(R3, n), (tmp,)) for n in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11")] + \
            [("V1", V1, (tmp,)), ("V2", V2, (tmp,)), ("V3", V3, (tmp,)), ("V4", V4_, (tmp,)), ("V5", V5, (tmp,)), ("V8", V8, (tmp,)), ("V6", V6, (tmp,)), ("V7", V7, (tmp,)), ("V9", V9, (tmp,)), ("D1", R3.D1, ()), ("D2", R3.D2, ()), ("M1", R3.M1, (tmp,))]
    for name, fn, args in steps:
        try: fn(*args)
        except Exception as e: rec(name, False, f"could not run: {type(e).__name__}: {str(e)[:200]}")
    n_fail = sum(r["result"] != "PASS" for r in R); verdict = "PASS" if not n_fail else f"FAIL ({n_fail} of {len(R)} checks)"
    print(f"RELEASE CHECK v4: {verdict}  ({len(R)} checks, {time.time() - t0:.0f} s; outputs in {tmp})")
    jp = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else os.path.join(MP, "kit", "results", "release_check_v4.json")
    json.dump({"check": "release_check_v4", "verdict": verdict, "checks": R, "seconds": round(time.time() - t0)}, open(jp, "w"), indent=1)
    sys.exit(0 if not n_fail else 1)

if __name__ == "__main__":
    main()
