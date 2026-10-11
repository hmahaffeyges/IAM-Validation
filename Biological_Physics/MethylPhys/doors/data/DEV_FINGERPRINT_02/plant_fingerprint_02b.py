"""DEV-FINGERPRINT-02, planted test of score_fingerprint_02b.py (Arm B decides; donor bar). Derived from plant_fingerprint_02.py.
Cases on one pair's normal files: NULL (unplanted, expect no fingerprint); RUN05 (5 % of molecules lose every methylated call, expect FINGERPRINT);
RUN01 (1 %, under the donor bar, expect no fingerprint); SCAT (scattered loss 0.04, a copy-error rise, not run loss: expect no fingerprint).
Arrays: the fake cancer array is the first normal array unchanged (Arm A is reported only).
Original docstring of plant_fingerprint_02.py follows.
 (written 2026-10-10 before it runs and before any cancer file is read). Normal data only.
Any pair (argument; default breast_mcf10a, the first run). The scorer runs byte-identical (sha256 checked) inside a scratch git repo that mirrors the paths it uses
(chain, DEV_RUNLOSS_01 and run9_fp2 linked; DEV_FINGERPRINT_02 copied), so nothing in the committed folder is written.
Fake cancer files: 3 normal files with copy error planted (each methylated call turned to T with probability DELTA, seeded); fake cancer array: the
normal array with the identity sites moved toward beta 0.5 by a factor chosen to hit a Met-A_rel target. Cases:
  NULL     fake cancer = 3 normal files unplanted, array unchanged             -> expect no fingerprint on both arms
  PLANTED  planted files, Met-A_rel target = curve B(IAM-A_rel) + 0.10          -> expect Arm A decides, FINGERPRINT
  ONLINE   planted files, Met-A_rel target = curve B(IAM-A_rel)                 -> expect no fingerprint
Arm B must read no fingerprint in all three (planted loss is scattered). The fake cancer comes from libraries that are also in the normal reference:
a test of the code path, not of the statistics. Usage: plant_fingerprint_02.py PATDIR BETADIR SCRATCH [PAIR]
Fake cancer files: up to 3 normal files (fewer if the pair has fewer); the fake cancer array takes the pair's first cancer array name."""
import os, sys, gzip, json, math, shutil, hashlib, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../..")); PATD, BETD, SCR = sys.argv[1:4]
DELTA = 0.04; PAIR = sys.argv[4] if len(sys.argv) > 4 else "breast_mcf10a"
SCORER = os.environ.get("SCORER", "score_fingerprint_02b.py")   # scorer file under test (2026-10-11: T47D pairs use score_fingerprint_02b2.py)
_src = open(os.path.join(HERE, SCORER)).read(); PAIRS = eval(_src[_src.index("PAIRS = {") + 8: _src.index("STEP, PAIR")].strip())
NC, _CC, SC, _AN, _AC = PAIRS[PAIR]; AN, AC = _AN[0], _AC[0]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def Hb(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
def plant(src, dst, seed, mode, x):
    """mode 'run': each molecule loses every C with probability x; 'scat': each C turned to T with probability x."""
    rg = np.random.default_rng(seed); C, T = ord("C"), ord("T"); buf = []
    def flush(fo):
        pats = [b[2] for b in buf]; L = np.array([len(p) for p in pats]); s_ = np.frombuffer("".join(pats).encode(), np.uint8).copy()
        if mode == "run": hit = np.repeat(rg.random(len(pats)) < x, L) & (s_ == C)
        else: hit = (s_ == C) & (rg.random(s_.size) < x)
        s_[hit] = T; out = s_.tobytes().decode(); k = 0
        for b, p in zip(buf, pats): fo.write(f"{b[0]}\t{b[1]}\t{out[k:k + len(p)]}\t1\n"); k += len(p)
        buf.clear()
    with gzip.open(src, "rt") as fi, gzip.open(dst, "wt", compresslevel=3) as fo:
        for l in fi:
            q = l.rstrip("\n").split("\t"); buf.extend([q] * int(q[3]))
            if len(buf) >= 200000: flush(fo)
        if buf: flush(fo)
CASES = {"NULL": (None, 0), "RUN05": ("run", 0.05), "RUN01": ("run", 0.01), "SCAT": ("scat", 0.04)}
EXP = {"NULL": "no fingerprint", "RUN05": "FINGERPRINT", "RUN01": "no fingerprint", "SCAT": "no fingerprint"}
def run(case):
    root = os.path.join(SCR, case); shutil.rmtree(root, ignore_errors=True)
    M = os.path.join(root, "Biological_Physics/MethylPhys"); os.makedirs(os.path.join(M, "doors/data"), exist_ok=True); os.makedirs(os.path.join(M, "boxruns"), exist_ok=True)
    os.symlink(os.path.join(MP, "chain"), os.path.join(M, "chain")); os.symlink(os.path.join(MP, "doors/data/DEV_RUNLOSS_01"), os.path.join(M, "doors/data/DEV_RUNLOSS_01"))
    D = os.path.join(M, "doors/data/DEV_FINGERPRINT_02"); shutil.copytree(HERE, D, ignore=shutil.ignore_patterns("sealed_*", "scored_*", "plant_*_output.json", "donor_null"))
    R9 = os.path.join(M, "boxruns/run9_fp2"); os.makedirs(R9); nr = pd.read_csv(os.path.join(MP, "boxruns/run9_fp2/normal_rrbs_files.csv")); nr.to_csv(os.path.join(R9, "normal_rrbs_files.csv"), index=False)
    nf = sorted(nr[nr.cell == NC].file); src = nf[:3]; fake = [f"FAKE{i}" for i in range(len(src))]
    pd.DataFrame(dict(file=fake, cell=_CC)).to_csv(os.path.join(R9, "cancer_rrbs_files.csv"), index=False)
    P = os.path.join(root, "pat"); B = os.path.join(root, "betas"); os.makedirs(P); os.makedirs(B)
    for f in nf: os.symlink(os.path.join(PATD, f + ".pat.gz"), os.path.join(P, f + ".pat.gz"))
    for i, (s, k) in enumerate(zip(src, fake)):
        mode, x = CASES[case]
        if mode is None: shutil.copy(os.path.join(PATD, s + ".pat.gz"), os.path.join(P, k + ".pat.gz"))
        else: plant(os.path.join(PATD, s + ".pat.gz"), os.path.join(P, k + ".pat.gz"), 100 + i, mode, x)
    for a_ in _AN: shutil.copy(os.path.join(BETD, a_ + ".parquet"), os.path.join(B, a_ + ".parquet"))
    sc = os.path.join(D, SCORER); assert sha(sc) == sha(os.path.join(HERE, SCORER))
    g = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)
    g("init", "-q"); r = subprocess.run([sys.executable, sc, "seal", PAIR, P, B], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    S = json.load(open(os.path.join(D, f"sealed_{PAIR}.json")))
    bc = pd.read_parquet(os.path.join(B, AN + ".parquet")).beta   # Arm A reported only: the fake cancer array is the first normal array unchanged
    target = None
    pd.DataFrame({"beta": bc}).to_parquet(os.path.join(B, AC + ".parquet"))
    g("add", "-A"); g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seal")
    r = subprocess.run([sys.executable, sc, "score", PAIR, P, B], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]
    o = json.load(open(os.path.join(D, f"scored_{PAIR}.json"))); o["case"] = case; return o
if __name__ == "__main__":
    out = {c: run(c) for c in CASES}
    for c, o in out.items():
        o["expected"] = EXP[c]; o["PASS"] = (o["VERDICT"] == "FINGERPRINT") == (EXP[c] == "FINGERPRINT")
        print(f"{c:6s} Arm B excess per file {[round(v, 4) for v in o['cancer_excess'].values()]} bar {o['armB_bar']:.4f} -> {o['VERDICT']} (expected {EXP[c]}) | "
              f"Arm A (reported) z {o['z']:.2f} | {'PASS' if o['PASS'] else 'FAIL'}")
    json.dump(out, open(os.path.join(HERE, f"plant02b_{PAIR}_output.json"), "w"), indent=1, default=float)
    print("PLANTED TEST", "PASSED" if all(o["PASS"] for o in out.values()) else "FAILED")
