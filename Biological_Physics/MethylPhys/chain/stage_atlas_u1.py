"""Atlas v2.1 use test U1 (reference profiles), scored ONCE by the sealed and amended bar (chain Stage A21), written 2026-10-16 before any
held-out array is read. The instrument's job: a healthy cell from any laboratory reads inside the atlas's range for that cell.
 1. For each atlas cell named in heldout_cell_map_v21.json: the 90 % posterior predictive interval at every site, from the fit's 100 saved draws
    of the cell's mean and donor spread, plus the measured laboratory spread (Stage A21 predictive_interval, kind 'array').
 2. For each held-out array (atlas_v21_heldout_samples.csv, roles 'held-out'), its Stage 1 calibrated values: coverage = share of its measured
    sites inside its cell's interval (Stage A21 u1_coverage).
 3. Per cell and platform family (450K, EPIC), all laboratories pooled: Stage A21 u1_verdict (>= 95 % of samples with coverage 0.80-0.97).
    Arrays whose cell is not in v2.1, and arrays with no calibrated file that passed Stage 1 and the identity check, are listed, not scored.
Usage: python score_u1_v21.py <blocks dir> <shards dir> <out dir> [workers]   |   functions are reused by planted_u1_v21.py"""
# Chain copy (2026-10-16) of doors/data/DEV_ATLAS_V21_01/score_u1_v21.py: the logic is unchanged; only where it finds its data folder.
import os, sys, json, glob, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../..")); H = os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01")   # moved into the chain 2026-10-16 (rule 23b-e); data stay in the test folder
sys.path.insert(0, HERE); import stage_atlas_commission as A21; import stage_q2_selfref as Q2
def family(p): return "EPIC" if "EPIC" in str(p) else "450K"
def _block(args):
    f, cells, lab = args; Z = np.load(f, allow_pickle=True); names = list(Z["cells"]); out = {}
    for c in cells:
        i = names.index(c); lo, hi = A21.predictive_interval(Z["draws"][:, i, :].astype(float), Z["sd_draws"][:, i, :].astype(float), kind="array", lab=lab)
        out[c] = pd.DataFrame(dict(lo=lo, hi=hi), index=list(Z["cpg"]))
    return out
def intervals(blocks, cells, workers=8, lab=None):
    lab = A21.lab_sd() if lab is None else lab; F = sorted(glob.glob(os.path.join(blocks, "block_*_draws.npz")))
    with Pool(workers) as p: parts = p.map(_block, [(f, cells, lab) for f in F])
    return {c: pd.concat([x[c] for x in parts]) for c in cells}, lab
def coverage(iv, shard):
    y = pd.read_parquet(shard).iloc[:, 0]; j = iv.index.intersection(y.index)
    return A21.u1_coverage(iv.loc[j, "lo"].values, iv.loc[j, "hi"].values, y.loc[j].values.astype(float)), int(len(j))
def score(samples, cmap, IV, shards):
    rows = []
    for _, s in samples.iterrows():
        c = cmap.get(s.cell_type); base = s.get("base", s["sample"]); sh = os.path.join(shards, f"{base}.parquet") if c and isinstance(base, str) else None
        if c is None: rows.append(dict(sample=s["sample"], series=s.series, label=s.cell_type, cell=None, family=family(s.platform), coverage=None, sites=0, status="listed: cell not in v2.1")); continue
        if sh is None or not os.path.exists(sh): rows.append(dict(sample=s["sample"], series=s.series, label=s.cell_type, cell=c, family=family(s.platform), coverage=None, sites=0, status="listed: withheld by Stage 1")); continue
        cv, n = coverage(IV[c], sh); rows.append(dict(sample=s["sample"], series=s.series, label=s.cell_type, cell=c, family=family(s.platform), coverage=cv, sites=n, status="scored"))
    T = pd.DataFrame(rows); V = {}
    for (c, fm), g in T[T.cell.notna()].groupby(["cell", "family"]):
        V[f"{c} | {fm}"] = dict(A21.u1_verdict(dict(zip(g["sample"], g.coverage.astype(float))), withheld=list(g[g.status != "scored"]["sample"])),
                               laboratories=sorted(set(g.series)), median_coverage=float(g.coverage.dropna().median()) if g.coverage.notna().any() else None)
    return T, V
if __name__ == "__main__":
    blocks, shards, out = sys.argv[1:4]; w = int(sys.argv[4]) if len(sys.argv) > 4 else 8; os.makedirs(out, exist_ok=True)
    S = pd.read_csv(os.path.join(H, "atlas_v21_heldout_samples.csv")); S = S[S.role.str.startswith("held-out")]
    # each sample's calibrated array file, as recorded by the identity check (2026-10-14): base = shard name; status ok = passed Stage 1 and identity
    I = pd.read_csv(os.path.join(H, "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")][["sample", "series", "base"]]
    S = S.merge(I, on=["sample", "series"], how="left")   # run 1 looked for <sample>.parquet and found none; nothing was scored
    cmap = json.load(open(os.path.join(H, "heldout_cell_map_v21.json")))["map"]
    IV, lab = intervals(blocks, sorted({v for v in cmap.values() if v}), w)
    T, V = score(S, cmap, IV, shards)
    Q2.write_csv(T, os.path.join(out, "u1_samples_v21.csv"), manifest=["ATL-005"])
    Q2.write_json(dict(test="atlas v2.1 U1 reference profiles", lab_sd=lab, share_bar=A21.SHARE, coverage_bar=[0.80, 0.97], verdicts=V,
                       scored=int((T.status == "scored").sum()), listed=int((T.status != "scored").sum())), os.path.join(out, "u1_scored_v21.json"), manifest=["ATL-005"])
    for k, v in V.items(): print(f"{k:<44} {v['verdict']:<13} n {v['n']:<4} share {v['share']} median {v['median_coverage']}")
