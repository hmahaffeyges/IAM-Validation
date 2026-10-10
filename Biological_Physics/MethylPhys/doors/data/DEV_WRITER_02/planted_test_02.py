"""DEV-WRITER-02 planted test of context_eps.one() and score_writer_02.py's fit (synthetic only; no cell file). Synthetic molecules of 10 CpGs,
each CpG given a random NNCGNN context; methylated molecules carry isolated T errors at eps_c = K x bracket_c, unmethylated molecules carry
isolated C at a context-independent rate. Expected: copy-error slope 1, control slope 0, k = K. Usage: python3 planted_test_02.py"""
import os, sys, types, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); Z = pd.read_csv(os.path.join(HERE, "enzyme_D_256.csv")).set_index("ctx")
rg = np.random.default_rng(7); K = 2.0; NCPG = 200000; tmp = os.path.join(HERE, "_planted"); os.makedirs(tmp, exist_ok=True)
ctxs = rg.choice(Z.index.values, NCPG); M = dict(enumerate(ctxs)); br = Z.pred1.reindex(ctxs).values
with open(os.path.join(tmp, "SYN.pat.txt"), "w") as f:
    for _ in range(400000):
        s = int(rg.integers(1, NCPG - 11)); idx = np.arange(s, s + 10)
        if rg.random() < 0.6:
            pat = np.where(rg.random(10) < K * br[idx], "T", "C")
            if (pat == "T").sum() > 2: continue                       # keep >= 80 % C
        else:
            pat = np.where(rg.random(10) < 0.01, "C", "T")
            if (pat == "C").sum() > 2: continue
        f.write(f"chr1\t{s}\t{''.join(pat)}\t1\n")
sys.argv = ["x", "/dev/null", tmp, "/dev/null", "/dev/null"]
src = open(os.path.join(HERE, "context_eps.py")).read().split('if __name__ == "__main__":')[0].replace("WIN = [l.split() for l in open(WINB)]", "WIN = []")
sys.modules.setdefault("pysam", types.ModuleType("pysam"))   # one() does not use pysam; the box run does
mod = types.ModuleType("ce"); exec(src, mod.__dict__)
R = pd.DataFrame(mod.one(("SYN", M))); R = R[R.ctx != "NA"]; R["cls"] = [min(c, Z.loc[c, "rc"]) for c in R.ctx]
G = R.groupby("cls")[["meth_err", "meth_opp", "ctrl_err", "ctrl_opp"]].sum(); x = np.log(Z.pred1.reindex(G.index))
out = {}
for ch in ("meth", "ctrl"):
    e = G[f"{ch}_err"] / G[f"{ch}_opp"]; ok = G[f"{ch}_err"] > 0; out[ch] = np.polyfit(x[ok], np.log(e[ok]), 1)[0]
e = G.meth_err / G.meth_opp; ok = G.meth_err > 0; k = float(np.exp(np.mean(np.log(e[ok]) - x[ok])))
# isolated-only counting and the >=80 % rule bias eps slightly low; the test is on the slopes
print(f"classes {len(G)} | copy-error slope {out['meth']:.3f} (expect 1) | control slope {out['ctrl']:.3f} (expect 0) | k {k:.2f} (planted {K})")
ok_ = 0.9 < out["meth"] < 1.1 and abs(out["ctrl"]) < 0.1
print("PLANTED TEST", "PASSED" if ok_ else "FAILED")
import shutil; shutil.rmtree(tmp)
