"""DEV-ATLAS-V21-01 culture check, step 1b (written 2026-10-12, before any GSE179847 or GSE82234 array is read). Reference data, not a test.
The step-1 mask (culture_drift_01_cpgs.csv) is unchanged; this measures whether its drift replicates in other laboratories.
Columbia, GSE179847 (EPIC v1): the 92 control arrays (Normal, 21 % oxygen, untreated; culture_gse179847_controls.csv). Lines with >= 8 arrays
(HC1, HC2 breast fibroblasts; HC3 foreskin fibroblasts). Within each, Spearman rho of beta with population doublings at CpGs measured in >= 8
arrays; Columbia drift = |rho| >= 0.5 with the same sign in >= 2 of those lines (step 1's cut; all three are fibroblasts, so this set is
fibroblast drift and cannot meet step 1's two-cell-type clause by itself).
Aachen, GSE82234 (450K): 3 HUVEC donors, passage 4 and a late passage (13-20). Fewer than 8 arrays per donor, so no trend under the rule;
descriptive only: per donor, late minus early beta.
Reported (descriptive): Columbia drift count; overlap with the step-1 set and with the batch-free core (culture_drift_01_batchfree_cpgs.csv);
among batch-free core CpGs measured in Columbia, the share that is Columbia drift with the SAME sign, against the share of all measured CpGs that
are Columbia drift (the chance level); Aachen: among batch-free core CpGs, the share where all three donors move in the core's direction, and
the same share at all CpGs.
Usage: python3 culture_drift_02.py COLUMBIA_BETADIR AACHEN_BETADIR   |   python3 culture_drift_02.py --planted"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); CUT = 0.5; MIN = 8
def rho(B, x):
    ok = B.notna().sum(1) >= MIN; R = B[ok].rank(axis=1); xr = pd.Series(rankdata(x), index=B.columns)
    Rc = R.sub(R.mean(1), axis=0); xc = xr - xr.mean()
    return Rc.mul(xc, axis=1).sum(1) / np.sqrt((Rc ** 2).sum(1) * (xc ** 2).sum())
def drift(rhos):
    R = pd.concat(rhos, axis=1); big = R.abs() >= CUT; pos = (R >= CUT).sum(1); neg = (R <= -CUT).sum(1)
    return pd.Series(np.where(pos >= 2, 1, np.where(neg >= 2, -1, 0)), index=R.index)   # +1/-1 drift, 0 not
def replication(Dc, core):
    m = core[core.cpg.isin(Dc.index)]; same = (Dc.loc[m.cpg].values == m.sign.values) & (m.sign.values != 0)
    return len(m), float(same.mean()), float((Dc != 0).mean())
if __name__ == "__main__":
    if sys.argv[1] == "--planted":   # 3 lines of 20 arrays, 20,000 CpGs; 1,000 planted drifting (+), 1,000 (-), the rest noise
        rg = np.random.default_rng(20261012); n = 20000; S = np.zeros(n); S[:1000] = 1; S[1000:2000] = -1; rhos = {}
        for ln in range(3):
            pd_ = np.sort(rg.uniform(0, 70, 20)); B = 0.5 + 0.004 * np.outer(S, pd_) + rg.normal(0, 0.05, (n, 20))
            rhos[ln] = rho(pd.DataFrame(B, index=[f"c{i}" for i in range(n)]), pd_)
        Dc = drift(rhos); core = pd.DataFrame({"cpg": [f"c{i}" for i in range(2000)] + [f"c{i}" for i in range(2000, 3000)], "sign": [1] * 1000 + [-1] * 1000 + [1] * 1000})
        tp = int((Dc.iloc[:1000] == 1).sum() + (Dc.iloc[1000:2000] == -1).sum()); fp = int((Dc.iloc[2000:] != 0).sum())
        k, share, chance = replication(Dc, core)
        ok = tp >= 1900 and fp <= 0.01 * 18000 and abs(share - 2000 * (tp / 2000) / 3000) < 0.02
        out = f"planted: found {tp}/2000, false {fp}/18000, core share same-sign {share:.3f} (expect {tp / 3000:.3f}), chance {chance:.3f} -> {'PASSED' if ok else 'FAILED'}"
        open(os.path.join(HERE, "culture_drift_02_planted_output.txt"), "w").write(out + "\n"); print(out); sys.exit(0 if ok else 1)
    L = pd.read_csv(os.path.join(HERE, "culture_gse179847_controls.csv")); core = pd.read_csv(os.path.join(HERE, "culture_drift_01_batchfree_cpgs.csv"))
    S1 = set(pd.read_csv(os.path.join(HERE, "culture_drift_01_cpgs.csv")).cpg); rhos = {}
    for ln, g in L.groupby("cell_line"):
        if len(g) < MIN: continue
        B = pd.concat({x: pd.read_parquet(os.path.join(sys.argv[1], f"{x}.parquet")).beta for x in g.gsm}, axis=1)
        rhos[ln] = rho(B, g.set_index("gsm").population_doublings.loc[B.columns].values)
    Dc = drift(rhos).dropna(); k, share, chance = replication(Dc, core); dset = set(Dc.index[Dc != 0])
    out = [f"Columbia lines used: {sorted(rhos)}; CpGs measured {len(Dc)}; Columbia drift {len(dset)} ({100 * len(dset) / len(Dc):.2f} %)",
           f"overlap with step-1 set: {len(dset & S1)} ({100 * len(dset & S1) / max(len(dset), 1):.1f} % of Columbia drift)",
           f"batch-free core CpGs measured in Columbia: {k}; Columbia drift with the same sign: {100 * share:.1f} % (chance level, share of all measured CpGs that drift: {100 * chance:.1f} %)"]
    A = pd.read_csv(os.path.join(HERE, "culture_gse82234.csv")); A["p"] = A.passage.str.extract(r"(\d+)")[0].astype(int); dl = {}
    for dn, g in A.groupby("donor"):
        e, l = g.sort_values("p").gsm.iloc[0], g.sort_values("p").gsm.iloc[-1]
        dl[dn] = pd.read_parquet(os.path.join(sys.argv[2], f"{l}.parquet")).beta - pd.read_parquet(os.path.join(sys.argv[2], f"{e}.parquet")).beta
    Dl = pd.concat(dl, axis=1).dropna(); c = core[core.cpg.isin(Dl.index)].set_index("cpg").sign
    allc = (np.sign(Dl.loc[c.index]).eq(c, axis=0)).all(1).mean(); base = ((np.sign(Dl) > 0).all(1) | (np.sign(Dl) < 0).all(1)).mean() / 2
    out.append(f"Aachen (descriptive): core CpGs on 450K {len(c)}; all 3 donors move in the core's direction at {100 * allc:.1f} % (all-3-agree in one given direction at all CpGs: {100 * base:.1f} %)")
    open(os.path.join(HERE, "culture_drift_02_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
