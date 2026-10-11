"""DEV-FINGERPRINT-02, planted test of score_fingerprint_02.py (written 2026-10-10 before it runs and before any cancer file is read). Normal data only.
Pair breast_mcf10a (6 normal MCF 10A files). The scorer runs byte-identical (sha256 checked) inside a scratch git repo that mirrors the paths it uses
(chain, DEV_RUNLOSS_01 and run9_fp2 linked; DEV_FINGERPRINT_02 copied), so nothing in the committed folder is written.
Fake cancer files: 3 normal files with copy error planted (each methylated call turned to T with probability DELTA, seeded); fake cancer array: the
normal array with the identity sites moved toward beta 0.5 by a factor chosen to hit a Met-A_rel target. Cases:
  NULL     fake cancer = 3 normal files unplanted, array unchanged             -> expect no fingerprint on both arms
  PLANTED  planted files, Met-A_rel target = curve B(IAM-A_rel) + 0.10          -> expect Arm A decides, FINGERPRINT
  ONLINE   planted files, Met-A_rel target = curve B(IAM-A_rel)                 -> expect no fingerprint
Arm B must read no fingerprint in all three (planted loss is scattered). The fake cancer comes from libraries that are also in the normal reference:
a test of the code path, not of the statistics. Usage: plant_fingerprint_02.py PATDIR BETADIR SCRATCH"""
import os, sys, gzip, json, math, shutil, hashlib, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../..")); PATD, BETD, SCR = sys.argv[1:4]
DELTA = 0.04; PAIR = "breast_mcf10a"; NC = "MCF 10A"; AN, AC = "ENCSR329HJR", "ENCSR889ZLL"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def Hb(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
def plant(src, dst, seed):
    rg = np.random.default_rng(seed); C, T = ord("C"), ord("T"); buf = []
    def flush(fo):
        pats = [b[2] for b in buf]; s = np.frombuffer("".join(pats).encode(), np.uint8).copy()
        s[(s == C) & (rg.random(s.size) < DELTA)] = T; out = s.tobytes().decode(); k = 0
        for b, p in zip(buf, pats): fo.write(f"{b[0]}\t{b[1]}\t{out[k:k + len(p)]}\t1\n"); k += len(p)
        buf.clear()
    with gzip.open(src, "rt") as fi, gzip.open(dst, "wt", compresslevel=3) as fo:
        for l in fi:
            q = l.rstrip("\n").split("\t"); buf.extend([q] * int(q[3]))      # one line per molecule, each planted independently
            if len(buf) >= 200000: flush(fo)
        if buf: flush(fo)
def run(case):
    root = os.path.join(SCR, case); shutil.rmtree(root, ignore_errors=True)
    M = os.path.join(root, "Biological_Physics/MethylPhys"); os.makedirs(os.path.join(M, "doors/data"), exist_ok=True); os.makedirs(os.path.join(M, "boxruns"), exist_ok=True)
    os.symlink(os.path.join(MP, "chain"), os.path.join(M, "chain")); os.symlink(os.path.join(MP, "doors/data/DEV_RUNLOSS_01"), os.path.join(M, "doors/data/DEV_RUNLOSS_01"))
    D = os.path.join(M, "doors/data/DEV_FINGERPRINT_02"); shutil.copytree(HERE, D, ignore=shutil.ignore_patterns("sealed_*", "scored_*", "plant_*_output.json"))
    R9 = os.path.join(M, "boxruns/run9_fp2"); os.makedirs(R9); nr = pd.read_csv(os.path.join(MP, "boxruns/run9_fp2/normal_rrbs_files.csv")); nr.to_csv(os.path.join(R9, "normal_rrbs_files.csv"), index=False)
    nf = sorted(nr[nr.cell == NC].file); src = nf[:3]; fake = [f"FAKE{i}" for i in range(3)]
    pd.DataFrame(dict(file=fake, cell="MCF-7")).to_csv(os.path.join(R9, "cancer_rrbs_files.csv"), index=False)
    P = os.path.join(root, "pat"); B = os.path.join(root, "betas"); os.makedirs(P); os.makedirs(B)
    for f in nf: os.symlink(os.path.join(PATD, f + ".pat.gz"), os.path.join(P, f + ".pat.gz"))
    for i, (s, k) in enumerate(zip(src, fake)):
        if case == "NULL": shutil.copy(os.path.join(PATD, s + ".pat.gz"), os.path.join(P, k + ".pat.gz"))
        else: plant(os.path.join(PATD, s + ".pat.gz"), os.path.join(P, k + ".pat.gz"), 100 + i)
    shutil.copy(os.path.join(BETD, AN + ".parquet"), os.path.join(B, AN + ".parquet"))
    sc = os.path.join(D, "score_fingerprint_02.py"); assert sha(sc) == sha(os.path.join(HERE, "score_fingerprint_02.py"))
    g = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)
    g("init", "-q"); r = subprocess.run([sys.executable, sc, "seal", PAIR, P, B], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    S = json.load(open(os.path.join(D, f"sealed_{PAIR}.json")))
    # fake cancer array: normal array with identity sites moved toward 0.5 to hit the target Met-A_rel
    sys.path.insert(0, os.path.join(MP, "chain")); import stage_q_iam_a as Q
    e = sum(float(Q.pat_site_table(os.path.join(P, k + ".pat.gz")).pipe(lambda T: T.err_A.sum() + T.err_B.sum())) for k in fake) / \
        sum(float(Q.pat_site_table(os.path.join(P, k + ".pat.gz")).pipe(lambda T: T.opp_A.sum() + T.opp_B.sum())) for k in fake)
    H = lambda x: -(x * math.log2(x) + (1 - x) * math.log2(1 - x)); IA = H(e) / H(S["eps_normal"])
    XQ, YB = np.array(S["curveB_XQ"]), np.array(S["curveB_YB"]); cB = float(np.interp(IA, XQ, YB, right=YB[-1] + S["slope"] * (IA - XQ[-1])))
    b = pd.read_parquet(os.path.join(B, AN + ".parquet")).beta; b.index = b.index.astype(str); sites = json.load(open(os.path.join(D, "identity_sites_breast_basal_epithelium.json")))["sites"]
    target = {"NULL": None, "PLANTED": cB + 0.10, "ONLINE": cB}[case]; bc = b.copy()
    if target is not None:
        s_ = [x for x in sites if x in b.index and pd.notna(b[x])]; h0 = float(Hb(b[s_]).mean())
        lo, hi = 0.0, 1.0
        for _ in range(60):
            t = (lo + hi) / 2; v = b[s_] + t * (0.5 - b[s_])
            if float(Hb(v).mean()) / h0 < target: lo = t
            else: hi = t
        bc.loc[s_] = b[s_] + lo * (0.5 - b[s_])
    pd.DataFrame({"beta": bc}).to_parquet(os.path.join(B, AC + ".parquet"))
    g("add", "-A"); g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seal")
    r = subprocess.run([sys.executable, sc, "score", PAIR, P, B], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]
    o = json.load(open(os.path.join(D, f"scored_{PAIR}.json"))); o["planted_IAMA_rel_from_files"] = IA; o["target_MetA_rel"] = target; return o
if __name__ == "__main__":
    out = {c: run(c) for c in ("NULL", "PLANTED", "ONLINE")}
    exp = {"NULL": "no fingerprint", "PLANTED": "FINGERPRINT", "ONLINE": "no fingerprint"}
    for c, o in out.items():
        o["expected"] = exp[c]; o["armB_ok"] = o["armB"] == "no fingerprint"; o["PASS"] = (o["VERDICT"] == exp[c]) and o["armB_ok"]
        print(f"{c:8s} IAM-A_rel {o['IAMA_rel']:.4f} share {o['share_read']:.3f} arm {o['deciding_arm']} | Met-A_rel {o['MetA_rel']:.4f} curve B {o['curveB']:.4f} z {o['z']:.2f}"
              f" -> {o['VERDICT']} (expected {exp[c]}) | arm B {o['armB']} | {'PASS' if o['PASS'] else 'FAIL'}")
    json.dump(out, open(os.path.join(HERE, "plant_fingerprint_02_output.json"), "w"), indent=1, default=float)
    print("PLANTED TEST", "PASSED" if all(o["PASS"] for o in out.values()) else "FAILED")
