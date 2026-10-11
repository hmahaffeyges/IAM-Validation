"""DEV-FINGERPRINT-02 donor null (2026-10-10, normal data only; written after the bronchial planted test failed on power, before any cancer file).
The bronchial normal comes from two donors, each with RRBS and an array (ENCODE): donor A ENCDO106AAA = RRBS ENCSR000DFF (ENCFF000MME, ENCFF000MMO) +
array ENCSR000ACY; donor B ENCDO099AAA = RRBS ENCSR000DFV (ENCFF000MJV, ENCFF000MJW) + array ENCSR000ACA. The sealed scorer, byte-identical, runs in a
scratch git repo with donor A as the normal and donor B in the cancer slot, and the other way round. Two healthy donors should read no fingerprint.
NOT a sealed test: both arrays' identity-site entropies were already printed by normal_array_check.py. It measures, before any cancer file is read,
whether donor-to-donor variation lies on curve B (then a single-donor reference is sound) or off it (then single-donor pairs can call a false
fingerprint, and their error term must carry the donor-to-donor spread). Usage: donor_null_02.py PATDIR BETADIR SCRATCH"""
import os, sys, json, shutil, hashlib, subprocess, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../..")); PATD, BETD, SCR = sys.argv[1:4]
DON = {"A": (["ENCFF000MME", "ENCFF000MMO"], "ENCSR000ACY"), "B": (["ENCFF000MJV", "ENCFF000MJW"], "ENCSR000ACA")}
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def run(ref, test):
    root = os.path.join(SCR, f"null_{ref}{test}"); shutil.rmtree(root, ignore_errors=True)
    M = os.path.join(root, "Biological_Physics/MethylPhys"); os.makedirs(os.path.join(M, "doors/data")); os.makedirs(os.path.join(M, "boxruns/run9_fp2"))
    os.symlink(os.path.join(MP, "chain"), os.path.join(M, "chain")); os.symlink(os.path.join(MP, "doors/data/DEV_RUNLOSS_01"), os.path.join(M, "doors/data/DEV_RUNLOSS_01"))
    D = os.path.join(M, "doors/data/DEV_FINGERPRINT_02"); shutil.copytree(HERE, D, ignore=shutil.ignore_patterns("sealed_*", "scored_*", "plant_*", "normal_*", "donor_null"))
    sc = os.path.join(D, "score_fingerprint_02.py"); s = open(sc).read()
    a = s[s.index("PAIRS = {"): s.index("STEP, PAIR")]
    s2 = s.replace(a, 'PAIRS = {"donor_null": ("REF", "TEST", "lung_bronchus_epithelium", ["%s"], ["%s"])}\n\n' % (DON[ref][1], DON[test][1]))
    open(sc, "w").write(s2)     # only the pair table differs; the scoring code is the sealed scorer's (checked below)
    assert s2.replace(s2[s2.index("PAIRS = {"): s2.index("STEP, PAIR")], "") == s.replace(a, ""), "scorer code changed"
    R9 = os.path.join(M, "boxruns/run9_fp2")
    pd.DataFrame(dict(file=DON[ref][0], cell="REF", experiment="x")).to_csv(os.path.join(R9, "normal_rrbs_files.csv"), index=False)
    pd.DataFrame(dict(file=DON[test][0], cell="TEST", experiment="x")).to_csv(os.path.join(R9, "cancer_rrbs_files.csv"), index=False)
    os.makedirs(os.path.join(D, "normal_REF")); shutil.copy(os.path.join(HERE, "donor_null", {"A": "insilico_donorA_ENCFF000MME.csv", "B": "insilico_donorB_ENCFF000MJW.csv"}[ref]), os.path.join(D, "normal_REF"))   # the reference donor's own planted-loss response
    P = os.path.join(root, "pat"); B = os.path.join(root, "betas"); os.makedirs(P); os.makedirs(B)
    for f in DON[ref][0] + DON[test][0]: os.symlink(os.path.join(PATD, f + ".pat.gz"), os.path.join(P, f + ".pat.gz"))
    for x in (ref, test): os.symlink(os.path.join(BETD, DON[x][1] + ".parquet"), os.path.join(B, DON[x][1] + ".parquet"))
    g = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True); g("init", "-q")
    r = subprocess.run([sys.executable, sc, "seal", "donor_null", P, B], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
    g("add", "-A"); g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seal")
    r = subprocess.run([sys.executable, sc, "score", "donor_null", P, B], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
    return json.load(open(os.path.join(D, "scored_donor_null.json")))
out = {}
for ref, test in (("A", "B"), ("B", "A")):
    o = run(ref, test); out[f"ref {ref}, test {test}"] = o
    print(f"ref {ref} test {test}: IAM-A_rel {o['IAMA_rel']:.4f} share {o['share_read']:.3f} | Met-A_rel {o['MetA_rel']:.4f} curve B {o['curveB']:.4f} "
          f"excess {o['MetA_rel'] - o['curveB']:+.4f} se {o['se']:.4f} z {o['z']:.2f} | arm A {o['armA']} | arm B {o['armB']} (bar {o['armB_bar']:.4f}) -> {o['VERDICT']}")
json.dump(out, open(os.path.join(HERE, "donor_null_02_output.json"), "w"), indent=1, default=float)
