"""DEV-DIRECTION-AUDIT-01 condition 3: is the IAM-A C-score's failure on RRBS (healthy C ~ 15 vs derived null 1) caused by mixing territories with
different copy error? Healthy files only. Per file: per-site opportunities and isolated errors (Stage Q's rule, territory_eps) from (a) all qualifying
molecules, (b) held molecules only (healthy beta >= 0.8, reference = the pair's other healthy files); sites in genomic order (chr, CpG index), blocks of
1,000 sites, C = sum (k_b - eps o_b)^2 / sum eps (1 - eps) o_b (stage_q_iam_a.cscore's formula, its block size and minimum 10 blocks).
Usage: python3 cscore_by_territory.py PATDIR OUT.csv   (healthy files and references from the DEV_FINGERPRINT_02 seals)"""
import sys, os, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import territory_eps as TE
FP = os.path.join(HERE, "../DEV_FINGERPRINT_02"); PATD, OUT = sys.argv[1], sys.argv[2]
def sites(f, bv, which):
    b, mol, n = f["b"], f["mol"], f["d"].n.values; N = len(n); C = b == 67; called = C | (b == 84)
    calls = np.bincount(mol, called, N); nc = np.bincount(mol, C, N); q = (calls >= 6) & (nc >= 0.8 * calls)
    bvx = np.zeros(len(b)); ok = called & (f["pos"] < len(bv)); bvx[ok] = np.nan_to_num(bv[f["pos"][ok]])
    mb = np.bincount(mol, bvx, N) / np.maximum(np.bincount(mol, ok, N), 1); sel = q & (mb >= 0.8) if which == "held" else q
    ci = np.flatnonzero(called); m = mol[ci]; v = C[ci]
    first = np.r_[True, m[1:] != m[:-1]]; last = np.r_[m[1:] != m[:-1], True]; inner = ~first & ~last & sel[m]
    err = inner & ~v & np.r_[False, v[:-1]] & np.r_[v[1:], False]; w = n[m]
    chrc = f["d"].chr.values[m]; ck = np.array([int(c) if c.isdigit() else {"X": 23, "Y": 24, "M": 25}.get(c, 99) for c in pd.Series(chrc).str.replace("chr", "", regex=False)])
    key = ck.astype(np.int64) * 10**9 + f["pos"][ci]
    T = pd.DataFrame(dict(k=key[inner], o=w[inner], e=(w * err)[inner])).groupby("k", sort=True).sum()
    return T.o.values.astype(float), T.e.values.astype(float)
def C_of(o, k, B=1000):
    nb = len(o) // B
    if nb < 10: return None, nb
    ob = o[:nb * B].reshape(nb, -1).sum(1); kb = k[:nb * B].reshape(nb, -1).sum(1); e = kb.sum() / ob.sum()
    return float(((kb - e * ob) ** 2).sum() / (e * (1 - e) * ob).sum()), nb
rows = []
for sp in sorted(glob.glob(os.path.join(FP, "sealed_*.json"))):
    S = json.load(open(sp))
    if "gain" in os.path.basename(sp) or "normal_files" not in S: continue
    nf = [r["file"] for r in S["normal_files"]]
    if len(nf) < 2: continue
    F = {x: TE.load(os.path.join(PATD, x + ".pat.gz")) for x in nf}
    for x in nf:
        bv = TE.beta_ref([F[y] for y in nf if y != x])
        for which in ("all", "held"):
            o, k = sites(F[x], bv, which); c, nb = C_of(o, k)
            rows.append(dict(pair=S["pair"], file=x, territory=which, C=c, n_blocks=nb, se_null=float(np.sqrt(2 / nb)) if nb else None))
            print(rows[-1], flush=True)
pd.DataFrame(rows).drop_duplicates(["file", "territory"]).to_csv(OUT, index=False)
