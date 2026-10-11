"""DEV-FINGERPRINT-02 'why' diagnostic (descriptive, 2026-10-11; sealed verdicts are not re-scored). Arm B reads only LOSS: molecules of normally
methylated territory (healthy beta >= 0.8, >= 6 calls) that read <= 20 % methylated. This adds its mirror, GAIN: molecules of normally unmethylated
territory (healthy beta <= 0.2, >= 6 calls) that read >= 80 % methylated. Same loader and rules as runloss_01.py. Healthy beta is pooled from the
pair's normal files; each normal file is read against the OTHER normal files (leave-one-out), each cancer file against all normal files.
Usage: python3 both_directions.py PATDIR PAIR [PAIR ...]   -> why/both_directions_<pair>.csv"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TOP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TOP, "../DEV_RUNLOSS_01")); import runloss_01 as R
def read2(f, bv):
    called = f["C"] | f["T"]; calls = np.bincount(f["mol"], weights=called, minlength=f["N"]); mc = np.bincount(f["mol"], weights=f["C"], minlength=f["N"])
    tb = np.bincount(f["mol"], weights=np.nan_to_num(bv[f["pos"]]) * called, minlength=f["N"])
    with np.errstate(invalid="ignore", divide="ignore"): mb = tb / calls; frac = np.where(calls > 0, mc / np.maximum(calls, 1), 0)
    ok = calls >= 6; M = ok & (mb >= 0.8); U = ok & (mb <= 0.2); n = f["n"]
    wM, wU = n * M, n * U
    return dict(mol_M=int(wM.sum()), loss_M=float((wM * (frac <= 0.2)).sum() / max(wM.sum(), 1)),
                mol_U=int(wU.sum()), gain_U=float((wU * (frac >= 0.8)).sum() / max(wU.sum(), 1)),
                beta_U=float((wU * mc).sum() / max((wU * calls).sum(), 1)), beta_M=float((wM * mc).sum() / max((wM * calls).sum(), 1)))
if __name__ == "__main__":
    PATD = sys.argv[1]
    for P in sys.argv[2:]:
        S = json.load(open(os.path.join(TOP, f"sealed_{P}.json"))); nf = [r["file"] for r in S["normal_files"]]; cf = S["cancer_files_listed"]
        have = [x for x in cf if os.path.exists(os.path.join(PATD, x + ".pat.gz"))]
        NL = {x: R.load(os.path.join(PATD, x + ".pat.gz")) for x in nf}; rows = []
        for x in nf:
            bv = R.beta_v([NL[y] for y in nf if y != x]); rows.append(dict(pair=P, side="normal (leave-one-out)", file=x, **read2(NL[x], bv)))
        bv = R.beta_v(list(NL.values()))
        for x in have: rows.append(dict(pair=P, side="cancer", file=x, **read2(R.load(os.path.join(PATD, x + ".pat.gz")), bv)))
        T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, f"both_directions_{P}.csv"), index=False)
        print(T.round(4).to_string(index=False), flush=True)
