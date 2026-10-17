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
  V4  the v4 report: the front sections (at a glance, shares, C-Score with its sky) and every v3 section id are present"""
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

def main():
    t0 = time.time(); files = R3.tracked(); tmp = tempfile.mkdtemp(prefix="rc_v4_")
    print(f"release_check_v4 @ {subprocess.run(['git', '-C', R3.ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}  python {PY}")
    steps = [("F1", F1, ()), ("F1b", F1b, ()), ("F2", F2, ()), ("S1", R3.S1, (files,)), ("S2", S2, (files,)), ("S3", R3.S3, ()), ("S4", R3.S4, ())] + \
            [(n, getattr(R3, n), (tmp,)) for n in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11")] + \
            [("V1", V1, (tmp,)), ("V2", V2, (tmp,)), ("V3", V3, (tmp,)), ("V4", V4_, (tmp,)), ("D1", R3.D1, ()), ("D2", R3.D2, ()), ("M1", R3.M1, (tmp,))]
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
