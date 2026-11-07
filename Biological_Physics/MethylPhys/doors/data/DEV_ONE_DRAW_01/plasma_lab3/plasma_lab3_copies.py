"""DEV-ONE-DRAW-01, plasma from a new laboratory (GSE186007, a linear-amplification bisulfite method), development after the second reading:
are a plasma's calls independent molecules, or copies of fewer molecules? At the library-floor sites every healthy cell of the IAM Sequencing
Atlas holds unmarked (chain/Runtime Matrices/Seq_Atlas/floor_sites_v1.parquet, side 'unmarked'), a methylated call is a false call. From
independent molecules the false calls at a site are binomial; from copies, each false molecule arrives as several calls at once. The
dispersion index D = sum (m - n p)^2 / sum n p (1 - p), over sites with n >= 2 calls (p the pooled false-call rate), is near 1 for independent
molecules and near the number of copies per molecule for copies. Counts only; nothing fitted.
Usage: python3 plasma_lab3_copies.py synthetic OUT.txt
       python3 plasma_lab3_copies.py one SAMPLE.lbeta GSM OUT.json          (one plasma, from its converted table)
       python3 plasma_lab3_copies.py score JSON_DIR READINGS_CSV OUT.txt      (D against each healthy plasma's tared whole-sample Met-A)"""
import sys, os, json, glob
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
FLOOR = os.path.join(H, "..", "..", "..", "..", "chain", "Runtime Matrices", "Seq_Atlas", "floor_sites_v1.parquet")

def dispersion(m, n, nmin=2):
    k = n >= nmin; m, n = m[k].astype(float), n[k].astype(float)
    p = m.sum() / n.sum() if n.sum() else np.nan
    if not p or not np.isfinite(p) or p >= 1: return dict(sites=int(k.sum()), calls=int(n.sum()), rate=None, D=None)
    D = float(((m - n * p) ** 2).sum() / (n * p * (1 - p)).sum())
    multi = float((m >= 2).sum() / max((m >= 1).sum(), 1))
    return dict(sites=int(k.sum()), calls=int(n.sum()), rate=round(float(p), 6), D=round(D, 4), share_of_sites_with_a_false_call_holding_two_or_more=round(multi, 4))

def synthetic(out):
    rng = np.random.default_rng(20261106); S = 46925; L = []; ok = True
    for lam in (2, 15):
        for c in (1, 2, 4, 8):
            ps = rng.uniform(0, 0.02, S) + 0.006                    # each site's own small level plus the conversion floor
            k = rng.poisson(lam / c, S)                             # original molecules (calls = molecules x copies)
            mm = rng.binomial(k, ps); m = mm * c; n = k * c          # each molecule's call copied c times
            r = dispersion(m, n); D = r["D"]
            good = (D < 1.3) if c == 1 else (abs(D - c) <= 0.25 * c)
            ok &= bool(good); L.append(f"calls per site about {lam}, copies {c}: D {D} (bar: {'below 1.3' if c == 1 else f'within 25 % of {c}'}) {'met' if good else 'NOT met'}")
    L.append("test: " + ("met" if ok else "not met")); open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L)); return ok

def one(lbeta, gsm, out):
    import pandas as pd
    a = np.fromfile(lbeta, np.uint16).reshape(-1, 2).astype(np.int64)
    F = pd.read_parquet(FLOOR); u = F[F.side == "unmarked"].cpg.to_numpy(np.int64) - 1; u = u[(u >= 0) & (u < len(a))]
    r = dispersion(a[u, 0], a[u, 1]); r["gsm"] = gsm
    r["by_calls"] = {f"{lo}-{hi}": dispersion(a[u, 0][(a[u, 1] >= lo) & (a[u, 1] <= hi)], a[u, 1][(a[u, 1] >= lo) & (a[u, 1] <= hi)]) for lo, hi in ((2, 3), (4, 7), (8, 15), (16, 10 ** 6))}
    json.dump(r, open(out, "w"), indent=1); print(json.dumps({k: r[k] for k in ("gsm", "sites", "calls", "rate", "D")}))

def score(jdir, readings, out):
    import pandas as pd
    from scipy.stats import spearmanr
    J = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(jdir, "**", "*_copies.json"), recursive=True)])
    R = pd.read_csv(readings); X = R[R.group == "HEA"].merge(J[["gsm", "D", "rate", "sites", "calls"]], on="gsm")
    t = X["Met-A whole sample, tared"].astype(float); dist = (t - 1).abs(); L = [f"healthy plasmas with D: {X.D.notna().sum()} of {int((R.group == 'HEA').sum())}"]
    L.append(f"D: median {X.D.median():.3f} (range {X.D.min():.3f} to {X.D.max():.3f}); false-call rate median {X.rate.median():.5f}")
    for nm, y in (("tared whole-sample Met-A", t), ("its distance from 1", dist)):
        r = spearmanr(X.D, y); L.append(f"Spearman, D with {nm}: {r.statistic:+.3f} (p {r.pvalue:.2g})")
    L.append("rows (gsm, D, false-call rate, tared whole-sample Met-A):"); L += [f"  {g} {d} {p} {a:.4f}" for g, d, p, a in zip(X.gsm, X.D, X.rate, t)]
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L[:5]))

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "synthetic": sys.exit(0 if synthetic(a[1]) else 1)
    if a[0] == "one": one(a[1], a[2], a[3])
    if a[0] == "score": score(a[1], a[2], a[3])
