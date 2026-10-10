"""DEV-FINGERPRINT-01, planted test of the SEALED scorer (2026-10-10, before any LNCaP run or array is read). Healthy prostate (PrEC) data only.
Fake 'LNCaP' runs: PrEC runs with a known copy-error rise planted (every methylated call turned to T with probability DELTA, seeded per run;
small enough for Stage Q to read, so Arm A decides). Fake 'LNCaP' arrays: the two PrEC arrays with the prostate identity sites moved toward
beta 0.5 until Met-A_rel hits a target. Two cases: PLANTED (target = curve B at the planted IAM-A_rel + 0.10; expected FINGERPRINT) and
ONLINE (target = curve B; expected no fingerprint). Arm B must read no fingerprint in both (planted loss is scattered, not run loss).
The fake cancer is built from libraries that are also in the healthy reference: a test of the code path, not of the statistics.
score_fingerprint_01.py runs byte-identical (sha256 checked) in a scratch copy of its folder, so nothing in the committed folder is written.
Usage: plant_fingerprint_01.py WORKDIR(with pat/ and betas/ of PrEC) SCRATCH"""
import os, sys, gzip, json, math, shutil, hashlib, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
SRC = os.path.join(HERE, "../DEV_FINGERPRINT_01"); W0, SCR = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(MP, "chain")); import stage_q_iam_a as Q
DELTA = 0.04
FAKE = {"SRR4238609": ("SRR4238616", 1), "SRR4238610": ("SRR4238617", 2), "SRR4238611": ("SRR4238616", 3), "SRR4238612": ("SRR4238617", 4), "SRR4238613": ("SRR4238614", 5)}
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
def Hb(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
def plant(src, dst, seed):
    rg = np.random.default_rng(seed); C, T = ord("C"), ord("T")
    with gzip.open(src, "rt") as fi, gzip.open(dst, "wt", compresslevel=3) as fo:
        buf = []
        def flush():
            pats = [b[2] for b in buf]; s = np.frombuffer("".join(pats).encode(), dtype=np.uint8).copy()
            hit = (s == C) & (rg.random(s.size) < DELTA); s[hit] = T; out = s.tobytes().decode(); k = 0
            for b, p in zip(buf, pats): fo.write(f"{b[0]}\t{b[1]}\t{out[k:k + len(p)]}\t1\n"); k += len(p)
            buf.clear()
        for l in fi:
            c, st, p, n = l.rstrip("\n").split("\t")
            for _ in range(int(n)): buf.append((c, st, p))
            if len(buf) >= 2_000_000: flush()
        if buf: flush()
def workdir(case, arrays):
    w = os.path.join(SCR, "w_" + case); os.makedirs(os.path.join(w, "pat"), exist_ok=True); os.makedirs(os.path.join(w, "betas"), exist_ok=True)
    for r in ("SRR4238614", "SRR4238615", "SRR4238616", "SRR4238617"):
        d = os.path.join(w, "pat", r + ".pat.gz"); os.path.lexists(d) or os.symlink(os.path.realpath(os.path.join(W0, "pat", r + ".pat.gz")), d)
    for r in FAKE:
        d = os.path.join(w, "pat", r + ".pat.gz"); os.path.lexists(d) or os.symlink(os.path.join(SCR, "planted", r + ".pat.gz"), d)
    for g, b in arrays.items(): pd.DataFrame({"beta": b}).to_parquet(os.path.join(w, "betas", g + ".parquet"))
    return w
def scorer_copy():
    t = os.path.join(SCR, "tree"); d = os.path.join(t, "doors/data/DEV_FINGERPRINT_01"); os.makedirs(os.path.join(t, "doors/data"), exist_ok=True)
    for name, target in (("chain", os.path.join(MP, "chain")), ("doors/data/DEV_RUNLOSS_01", os.path.join(MP, "doors/data/DEV_RUNLOSS_01"))):
        p = os.path.join(t, name); os.path.lexists(p) or os.symlink(target, p)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("score_fingerprint_01.py", "prostate_identity_sites_v1.json", "insilico_prec_SRR4238614.csv"): shutil.copy(os.path.join(SRC, f), d)
    a, b = (hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (os.path.join(SRC, "score_fingerprint_01.py"), os.path.join(d, "score_fingerprint_01.py")))
    assert a == b; return os.path.join(d, "score_fingerprint_01.py"), a
if __name__ == "__main__":
    os.makedirs(os.path.join(SCR, "planted"), exist_ok=True)
    for r, (src, seed) in FAKE.items():
        dst = os.path.join(SCR, "planted", r + ".pat.gz")
        if not os.path.exists(dst): plant(os.path.join(W0, "pat", src + ".pat.gz"), dst, seed); print("planted", r, "from", src, flush=True)
    # IAM-A_rel and curve B exactly as the scorer computes them (the scorer prints both; its printed values are the ones recorded)
    def ee(paths):
        e = o = 0.0
        for p in paths: T = Q.pat_site_table(p); e += T.err_A.sum() + T.err_B.sum(); o += T.opp_A.sum() + T.opp_B.sum()
        return e / o
    eP = ee([os.path.join(W0, "pat", r + ".pat.gz") for r in ("SRR4238614", "SRR4238615", "SRR4238616", "SRR4238617")])
    eL = ee([os.path.join(SCR, "planted", r + ".pat.gz") for r in FAKE]); IA = H(eL) / H(eP)
    RS = pd.read_csv(os.path.join(SRC, "insilico_prec_SRR4238614.csv"))
    XS = np.array([1.00, 1.02, 1.05, 1.10, 1.16]); YB = np.array([1.000, 1.023, 1.056, 1.115, 1.188]); XQ = np.interp(XS, RS.IAMA_rel_simple, RS.IAMA_rel)
    SLOPE = np.polyfit(XQ, YB, 1)[0]; cB = float(np.interp(IA, XQ, YB, right=YB[-1] + SLOPE * (IA - XQ[-1])))
    print(f"planted IAM-A_rel {IA:.4f} (eps PrEC {eP:.5f}, planted {eL:.5f}) | curve B {cB:.4f}", flush=True)
    S = json.load(open(os.path.join(SRC, "prostate_identity_sites_v1.json")))["sites"]
    B = {g: pd.read_parquet(os.path.join(W0, "betas", g + ".parquet")).beta for g in ("GSM2309172", "GSM2309173")}
    for g in B: B[g].index = B[g].index.astype(str)
    common = pd.concat([B[g].reindex(S) for g in B], axis=1).dropna().index; ref = np.mean([Hb(B[g][common]).mean() for g in B])
    def shifted(t):
        out = {}
        for fake, g in (("GSM2309170", "GSM2309172"), ("GSM2309171", "GSM2309173")):
            b = B[g].copy(); b[common] = b[common] + t * (0.5 - b[common]); out[fake] = b
        return out
    def metA(t): o = shifted(t); return np.mean([Hb(o[g][common]).mean() for g in o]) / ref
    sc, sha = scorer_copy(); res = {}
    for case, target in (("PLANTED", cB + 0.10), ("ONLINE", cB)):
        lo, hi = 0.0, 1.0
        for _ in range(60): mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if metA(mid) < target else (lo, mid)
        arr = dict(shifted(lo)); arr.update({g: B[g] for g in B}); w = workdir(case, arr)
        out = subprocess.run([sys.executable, sc, w, "score"], capture_output=True, text=True); tail = [l for l in out.stdout.split("\n") if l.startswith(("IAM-A_rel", "ARM"))]
        res[case] = dict(target_MetA_rel=round(target, 4), t=round(lo, 5), scorer=tail, rc=out.returncode, err=out.stderr[-400:] if out.returncode else "")
        print(case, json.dumps(res[case], indent=1), flush=True)
    ok = (res["PLANTED"]["rc"] == 0 and "-> FINGERPRINT |" in res["PLANTED"]["scorer"][-1] and "ARM A" in res["PLANTED"]["scorer"][-1]
          and res["ONLINE"]["rc"] == 0 and "-> no fingerprint |" in res["ONLINE"]["scorer"][-1]
          and all(r["scorer"][-1].rstrip().endswith("-> no fingerprint") for r in res.values()))
    json.dump(dict(delta=DELTA, IAMA_rel=IA, curveB=cB, scorer_sha256=sha, cases=res, all_as_expected=bool(ok)), open(os.path.join(HERE, "plant_fingerprint_01_output.json"), "w"), indent=1)
    print("ALL AS EXPECTED" if ok else "NOT AS EXPECTED")
