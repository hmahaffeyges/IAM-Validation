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
      thin to read gives Met-A, the tare and the C-Score each a reason with the measured count and what is required
  V13 the globe as a browser draws it: the one command on a real healthy blood and its planted copy, each globe page opened in Chromium,
      the slider moved as a reader would; at the right end a healthy blood lights under 1 % of the globe, the plant lights Met-A's layer more
  V12 the report's globe: the atlas's healthy neutrophil against itself shows no difference; blood weights rescale to 1; the script parses
  V10 Group-A in the chain (stage_groupa.py, commissioned 2026-10-20): on the committed readings of DEV-GROUPA-COMMISSION-01 it gives the sealed
      bars 2 and 3 to every printed digit; its tare is word for word the sealed scorer's; groups in other specimens are labeled preliminary
  V11 Every stop of the one command explained (Chain Decoder finding F4, 2026-10-20): a refused specimen run through run_sample.py prints and keeps
      the five-part explanation and the report shows it; and in run_sample.py every exit with code 2 or 3 is preceded, in its block, by an
      explanation (_why, or Stage 0's stage0_explained)"""
import os, sys, json, gzip, glob, hashlib, re, time, subprocess, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "MethylPhys_Interface")]
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
    hyb = S0.validate_control_probes(dict(bisulfite_conversion_I_median=3e4, bisulfite_conversion_II_median=1e3, hyb_high_median=9000., hyb_low_median=7000., hyb_medium_median=8000., extension_meth_median=1e4, extension_unmeth_median=1e4))
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
    swp = S0.validate_control_probes(dict(bisulfite_conversion_I_median=0.97, bisulfite_conversion_II_median=0.03, hyb_high_median=6845., hyb_low_median=20040.,
                                          hyb_medium_median=12834., extension_meth_median=1e4, extension_unmeth_median=1e4))   # the swapped GSE184269 slides
    swap_ok = "HYB_FAIL" not in swp["flags"] and str(swp["metrics"].get("hyb_order", "")).startswith("swapped")
    if not swap_ok: bad.append(f"a High/Low label swap (spread intact, medium between) is stopped or not recorded: {swp['ctrl_qc']}")
    rec("V9", not bad and clean and hyb["ctrl_qc"].startswith("FAIL") and ext["ctrl_qc"].startswith("FAIL"),
        f"{len(cases)} kinds of Stage 0 stop and 3 downstream stops (Met-A, tare, C-Score) checked; {len(bad)} unexplained; a clean array proceeds with nothing to explain: {clean}; a High/Low swap proceeds, recorded: {swap_ok}" + (f"; MISSING: {bad}" if bad else ""))

def V10(tmp):
    import stage_groupa as GA, ast as _ast
    G = os.path.join(HERE, "..", "doors", "data", "GROUPA_COMMISSIONING"); S = json.load(open(os.path.join(G, "scored_groupa_01.json")))
    I = pd.read_csv(os.path.join(G, "scored_groupa_01_rows_idat.csv"))[["gsm", "A", "C", "slide", "sex"]]
    V = pd.read_csv(os.path.join(G, "scored_groupa_01_rows_values.csv"))[["gsm", "A", "C", "slide", "planted", "half"]]
    b2 = GA.compare(I, "sex", "female", "male"); b3 = GA.compare(V, "half", "planted", "unplanted", healthy=~V.planted.astype(bool))
    K = ("n1", "n2", "mean1", "mean2", "difference", "rel_difference", "se", "t"); same = all(b2[k] == S["B2"][k] for k in K) and all(b3[k] == S["B3"][k] for k in K)
    def body(path):
        src = open(path).read(); f = next(n for n in _ast.walk(_ast.parse(src)) if isinstance(n, _ast.FunctionDef) and n.name == "tare")
        return "\n".join(l.strip() for l in _ast.get_source_segment(src, f).split("\n") if l.strip() and not l.strip().startswith("#"))
    tare_same = body(os.path.join(HERE, "stage_groupa.py")) == body(os.path.join(G, "score_groupa.py"))
    pre = "preliminary" in str(GA.compare(I, "sex", "female", "male", specimen="buffy coat").get("commissioning", ""))
    rec("V10", same and tare_same and pre, f"sealed bars reproduced: {same} (B2 t {b2['t']}, B3 t {b3['t']}); tare identical to the sealed scorer: {tare_same}; other specimens labeled preliminary: {pre}")

def V11(tmp):
    import re as _re
    b = os.path.join(tmp, "v11_b.csv"); pd.DataFrame({"beta": [0.5, 0.6]}, index=["cg00000029", "cg00000108"]).to_csv(b)
    out = os.path.join(tmp, "v11", "r.html"); os.makedirs(os.path.dirname(out), exist_ok=True)
    rs = os.path.join(HERE, "MethylPhys_Interface", "run_sample.py")
    r = subprocess.run([sys.executable, rs, "--betas", b, "--specimen", "saliva", "--id", "v11", "--out", out], capture_output=True, text=True)
    bp = out[:-5] + "_bundle.json"; bun = json.load(open(bp)) if os.path.exists(bp) else {}   # the bundle is named after the report file
    ex = (bun.get("intake") or {}).get("stage0_explained") or []
    parts = bool(ex) and all(ex[0].get(k) for k in ("measured", "limit", "limit_source", "means", "what_to_do"))
    shown = os.path.exists(out) and "sec-why-stopped" in open(out, encoding="utf-8").read() and "WHY: specimen" in r.stdout
    import ast as _ast
    src = open(rs, encoding="utf-8").read(); bare = []
    def _is_exit(st):
        c = st.value if isinstance(st, _ast.Expr) else None
        return (isinstance(c, _ast.Call) and _ast.unparse(c.func) == "sys.exit" and c.args and isinstance(c.args[0], _ast.Constant) and c.args[0].value in (2, 3))
    for node in _ast.walk(_ast.parse(src)):   # every statement list (a block); an exit's explanation must come before it IN THE SAME BLOCK
        for field in ("body", "orelse", "finalbody"):
            B = getattr(node, field, None)
            if not isinstance(B, list): continue
            for i, st in enumerate(B):
                if _is_exit(st):
                    before = "\n".join(_ast.get_source_segment(src, x) or "" for x in B[:i])
                    if "_why(" not in before and "stage0_explained" not in before: bare.append(st.lineno)
    import stage_v4 as _V4, stage_q0_intake as _Q0
    K = ("measured", "limit", "limit_source", "means", "what_to_do")
    d21 = _V4.stage_d21(pd.Series([0.5] * 10, index=[f"cg{i:08d}" for i in range(10)])).get("explained") or {}
    q2a = _V4.stage_q2({"held_opp": 0, "loose_share": None}, "cd8 t cells").get("explained") or {}
    q2b = _V4.stage_q2({"held_opp": 100, "loose_share": 0.3, "held_err": 0.05, "tiles": 10, "tile_p10": 0.01}, "cd8 t cells").get("explained") or {}
    bad = os.path.join(tmp, "v11_bad.pat"); open(bad, "w").write("chr1\tnot-a-number\n")
    q0 = _Q0.intake(bad, specimen="purified", cell="cd8 t cells", selfref_cells={"cd8 t cells"}).get("explained") or {}
    stages = {n: all(x.get(k) for k in K) for n, x in (("D21 withheld", d21), ("Q2 no molecules", q2a), ("Q2 shape", q2b), ("Q0 unreadable", q0))}
    rec("V11", r.returncode == 0 and parts and shown and not bare and all(stages.values()),
        f"refused specimen: exit {r.returncode}, five parts kept {parts}, printed and in the report {shown}; exits 2/3 without an explanation: {bare or 'none'}; "
        f"stage stops with all five parts: {stages}")

def V12(tmp):
    """The report's globe (report_globe_v4, 2026-10-21): the atlas's own healthy neutrophil, laid on the globe as if it were an isolated
    neutrophil sample, must show no difference anywhere; a whole-blood reference must weigh the Stage D21 shares to 1; the layout must match
    the atlas site list it was frozen on; and, where node is installed, the page's script must parse."""
    import report_globe_v4 as RG, pyarrow.parquet as pq, shutil, subprocess as sp
    ids = pq.read_table(os.path.join(RG.ATLAS, "sites.parquet")).column("cpg").to_pylist(); mu = RG._mu(["neutrophils"])["neutrophils"]
    r = RG.build(pd.Series(mu, index=ids), "isolated neutrophils", None, os.path.join(tmp, "globe.html"))
    # sex (2026-10-21): the X and Y list is the one frozen from Stage 1's pinned manifests, every change count is also given on the X and Y,
    # and a planted change on the X alone is counted there and nowhere else
    import json as _j; XY = _j.load(open(RG.SEX_SITES)); sexok = XY["x_sites"] == len(XY["X"]) > 10000 and XY["sites_without_chromosome"] == 0 and "blocks_changed_on_XY" in r
    xs = [i for i in ids if i in set(XY["X"])][:3000]; m2 = pd.Series(mu, index=ids).copy(); m2.loc[xs] = np.clip(m2.loc[xs] + 0.4, 0, 1)
    r2 = RG.build(m2, "isolated neutrophils", None); sexok = sexok and r2["blocks_changed"] > 0 and r2["blocks_changed_on_XY"] == r2["blocks_changed"]
    zero = sexok and r.get("drawn") and r["median_abs_difference"] == 0 and r["share_above"]["0.05"] == 0 and r.get("sites_changed") == 0   # and no site changed
    name, w = RG.reference("whole blood", {"neutrophils": 0.5, "monocytes": 0.1, "cd4 t cells": 0.2})
    weights = abs(sum(w.values()) - 1) < 1e-9 and abs(w["neutrophils"] - 0.625) < 1e-9
    js = "not checked (node not installed)"
    if shutil.which("node"):
        f = os.path.join(tmp, "globe.js"); open(f, "w").write(re.search(r"<script>(.*?)</script>", r["fragment"], re.S).group(1))
        js = "parses" if sp.run(["node", "--check", f], capture_output=True).returncode == 0 else "SYNTAX ERROR"
    rec("V12", bool(zero and weights and js != "SYNTAX ERROR"), f"healthy atlas neutrophil against itself: drawn {r.get('drawn')}, median difference {r.get('median_abs_difference')}, "
        f"share above 0.05 {(r.get('share_above') or {}).get('0.05')}, sites changed {r.get('sites_changed')}; X and Y counted apart: {sexok} (planted on the X: {r2['blocks_changed']} changed, {r2['blocks_changed_on_XY']} on the X and Y); blood weights sum to 1 and rescale: {weights}; page script: {js}")

def V13(tmp):
    """The globe as a reader's browser draws it (2026-10-21, lesson C32: "if the chain didn't do it, it didn't happen"). The one command runs on
    a real healthy whole blood (GSE235717 GSM7508429, its IDATs from GEO) and on the same array with the C-Score's positive-control plant
    (doors/data/REPORT_EXAMPLES_V4/plant_clustered.py); each run's own globe page is opened in Chromium (playwright), the layer chosen and the
    slider moved as a reader would, and the canvas screenshotted at left, middle and right (written to GLOBE_SHOTS or tmp/globe_shots), for
    every group and every layer (2026-10-21). Bars: every layer has a slider; a change layer starts on a visible map, a healthy blood is dark
    at its right end, and the Met-A layer's key says disorder; a map layer's slider removes most of the map; the plant lights the Met-A layer
    more than the healthy blood does. Without a browser (a laptop): not checked; on GitHub (GITHUB_ACTIONS) a missing browser is a failure."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        need = bool(os.environ.get("GITHUB_ACTIONS")); rec("V13", not need, "not checked (playwright not installed)" + ("; required on GitHub" if need else "")); return
    import urllib.request
    from PIL import Image
    shots = os.environ.get("GLOBE_SHOTS") or os.path.join(tmp, "globe_shots"); os.makedirs(shots, exist_ok=True)
    g = "GSM7508429"; u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"; idat = {}
    for f in sorted(set(re.findall(rf"{g}_[^\"]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))):
        q = os.path.join(tmp, f); urllib.request.urlretrieve(u + f, q); idat["Grn" if "_Grn" in f else "Red"] = q
    RS = os.path.join(HERE, "MethylPhys_Interface", "run_sample.py"); PL = os.path.join(HERE, "..", "doors", "data", "REPORT_EXAMPLES_V4", "plant_clustered.py")
    hb = os.path.join(tmp, "h_betas.csv"); pb = os.path.join(tmp, "p_betas.csv")
    r1 = subprocess.run([sys.executable, RS, "--grn", idat["Grn"], "--red", idat["Red"], "--specimen", "whole blood", "--id", g, "--save-betas", hb, "--out", os.path.join(tmp, "h.html")], capture_output=True, text=True)
    subprocess.run([sys.executable, PL, hb, pb, HERE], capture_output=True, text=True, check=True)
    r2 = subprocess.run([sys.executable, RS, "--betas", pb, "--specimen", "whole blood", "--id", g + "_PLANTED", "--banner", "PLANTED: a constructed example, not a person.",
                         "--out", os.path.join(tmp, "p.html")], capture_output=True, text=True)
    if r1.returncode or r2.returncode or not all(os.path.exists(os.path.join(tmp, x)) for x in ("h_globe.html", "p_globe.html")):
        rec("V13", False, f"the one command did not write both globe pages (exit {r1.returncode}, {r2.returncode}): {(r1.stderr or r2.stderr)[-200:]}"); return
    def lit(png):   # share of the globe's disk that is a change color (saturated red or blue)
        a = np.asarray(Image.open(png).convert("RGB")).astype(int); r, gg, b = a[..., 0], a[..., 1], a[..., 2]
        disk = (a.sum(-1) > 0) | True; n = a.shape[0]; yy, xx = np.mgrid[0:n, 0:a.shape[1]]
        disk = (yy - n / 2) ** 2 + (xx - a.shape[1] / 2) ** 2 < (min(n, a.shape[1]) / 2 - 2) ** 2
        col = ((r > 150) & (gg < 110) & (b < 110)) | ((b > 150) & (r < 110))
        return float(col[disk].mean())
    def dark(png):   # share of the disk that is near black (the slider's "gone" color)
        a = np.asarray(Image.open(png).convert("RGB")).astype(int); n = a.shape[0]; yy, xx = np.mgrid[0:n, 0:a.shape[1]]
        disk = (yy - n / 2) ** 2 + (xx - a.shape[1] / 2) ** 2 < (min(n, a.shape[1]) / 2 - 2) ** 2; return float((a.sum(-1) < 75)[disk].mean())
    out, bad = {}, []
    with sync_playwright() as P:
        br = P.chromium.launch(); pg = br.new_page(viewport={"width": 1100, "height": 760})
        js_opts = lambda c: pg.evaluate(f"Array.from(document.querySelector('{c}').options).map(o => o.textContent)")
        js_pick = lambda c, k: pg.evaluate(f"(() => {{ const e = document.querySelector('{c}'); e.selectedIndex = {k}; e.dispatchEvent(new Event('change')); }})()")
        for tag in ("h", "p"):
            pg.goto("file://" + os.path.join(tmp, f"{tag}_globe.html")); pg.wait_for_timeout(2500); pg.click(".g-spin")
            for gi, gname in enumerate(js_opts(".g-group")):   # 2026-10-21 (author: "go through each one"): every group and every layer
                js_pick(".g-group", gi); pg.wait_for_timeout(800)
                for i, name in enumerate(js_opts(".g-layer")):
                    js_pick(".g-layer", i); pg.wait_for_timeout(2500)
                    if not pg.locator(".g-slide").is_visible(): bad.append(f"{tag} '{name}': no slider"); continue
                    mx = int(pg.eval_on_selector(".g-range", "e => e.max")); key = pg.inner_text(".g-key")
                    for t in (0, mx // 2, mx):
                        pg.eval_on_selector(".g-range", f"e => {{ e.value = {t}; e.dispatchEvent(new Event('input')); }}"); pg.wait_for_timeout(1500)
                        f = os.path.join(shots, f"{tag}_g{gi}_layer{i}_slider{t}.png"); pg.locator(".g-canvas").screenshot(path=f)
                        out[(tag, gi, i, t)] = (name, lit(f), dark(f), mx, key)
        br.close()
    for (tag, gi, i, t), (name, l, d, mx, key) in out.items():
        if t != 0: continue
        dr = out[(tag, gi, i, mx)][2]
        if name.startswith("the change"):
            if d > 0.5: bad.append(f"{tag} '{name}': left end is not a visible map (dark {d:.2f})")
            if tag == "h" and out[(tag, gi, i, mx)][1] >= 0.01: bad.append(f"{tag} '{name}': healthy blood not dark at the right end (lit {out[(tag, gi, i, mx)][1]:.4f})")
            if "Met-A" in name and "disorder" not in key: bad.append(f"{tag} '{name}': key does not say disorder")
        elif mx and dr - d < 0.2: bad.append(f"{tag} '{name}': the slider removes too little (dark {d:.2f} -> {dr:.2f})")
    H = [v[1] for k, v in out.items() if k[0] == "h" and k[3] == v[3] and v[0].startswith("the change")]
    Pm = [v[1] for k, v in out.items() if k[0] == "p" and k[3] == v[3] and "Met-A" in v[0] and v[0].startswith("the change")]
    Hm = [v[1] for k, v in out.items() if k[0] == "h" and k[3] == v[3] and "Met-A" in v[0] and v[0].startswith("the change")]
    if not (Pm and Hm and Pm[0] > Hm[0]): bad.append(f"the plant does not light the Met-A layer more than the healthy blood ({Pm} vs {Hm})")
    json.dump({f"{k[0]}_g{k[1]}_layer{k[2]}_slider{k[3]}": {"layer": v[0], "lit": round(v[1], 5), "dark": round(v[2], 4)} for k, v in out.items()}, open(os.path.join(shots, "lit.json"), "w"), indent=1)
    rec("V13", not bad, f"in Chromium, every group and layer of both globe pages ({len({k[:3] for k in out})} layers, {len(out)} frames): "
        + ("all pass" if not bad else "FAILED: " + "; ".join(bad[:6])) + f"; Met-A layer right end healthy {Hm[0] if Hm else None} vs planted {Pm[0] if Pm else None}; screenshots in {shots}")

def main():
    t0 = time.time(); files = R3.tracked(); tmp = tempfile.mkdtemp(prefix="rc_v4_")
    print(f"release_check_v4 @ {subprocess.run(['git', '-C', R3.ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}  python {PY}")
    steps = [("F1", F1, ()), ("F1b", F1b, ()), ("F2", F2, ()), ("S1", R3.S1, (files,)), ("S2", S2, (files,)), ("S3", R3.S3, ()), ("S4", R3.S4, ())] + \
            [(n, getattr(R3, n), (tmp,)) for n in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11")] + \
            [("V1", V1, (tmp,)), ("V2", V2, (tmp,)), ("V3", V3, (tmp,)), ("V4", V4_, (tmp,)), ("V5", V5, (tmp,)), ("V8", V8, (tmp,)), ("V6", V6, (tmp,)), ("V7", V7, (tmp,)), ("V9", V9, (tmp,)), ("V10", V10, (tmp,)), ("V11", V11, (tmp,)), ("V12", V12, (tmp,)), ("V13", V13, (tmp,)), ("D1", R3.D1, ()), ("D2", R3.D2, ()), ("M1", R3.M1, (tmp,))]
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
