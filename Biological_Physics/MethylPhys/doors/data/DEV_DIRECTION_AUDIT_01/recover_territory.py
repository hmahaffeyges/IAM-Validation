"""DEV-DIRECTION-AUDIT-01 condition 2, recovery test of territory_eps.read (healthy molecules only; 2026-10-11). On one healthy file, a share r of the
normally unmethylated molecules (>= 6 calls, healthy beta <= 0.2, reference = another healthy file) receives the exact call pattern (C/T in call order)
of a randomly drawn qualifying HELD molecule with the same number of calls. The expected gained copy error is then known exactly: the errors and
opportunities the copied patterns carry (Stage Q's rule), weighted by the target line's count, added to the natural gained territory. PASS: the reader's
gained eps equals the expected value to 1e-12, and the held eps is unchanged to 1e-12. r = 0.01 / 0.05. Usage: python3 recover_territory.py PAT REF_PAT OUT.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import territory_eps as TE
f = TE.load(sys.argv[1]); bv = TE.beta_ref([TE.load(sys.argv[2])]); base = TE.read(f, bv)
b, mol, n = f["b"], f["mol"], f["d"].n.values; N = len(n); C = b == 67; called = C | (b == 84)
calls = np.bincount(mol, called, N).astype(np.int64); nc = np.bincount(mol, C, N); q = (calls >= 6) & (nc >= 0.8 * calls)
bvx = np.zeros(len(b)); ok = called & (f["pos"] < len(bv)); bvx[ok] = np.nan_to_num(bv[f["pos"][ok]])
mb = np.bincount(mol, bvx, N) / np.maximum(np.bincount(mol, ok, N), 1)
held = np.flatnonzero(q & (mb >= 0.8)); U = np.flatnonzero((calls >= 6) & (mb <= 0.2) & ~q)
ci = np.flatnonzero(called); starts = np.searchsorted(mol[ci], np.arange(N)); seqs = lambda i: ci[starts[i]:starts[i] + calls[i]]
def pattern_stats(v):
    inner = np.zeros(len(v), bool); inner[1:-1] = True; e = inner & ~v & np.r_[False, v[:-1]] & np.r_[v[1:], False]; return int(inner.sum()), int(e.sum())
bylen = {}
for i in held: bylen.setdefault(int(calls[i]), []).append(i)
out = {"base": base}
for r, seed in ((0.01, 1), (0.05, 2)):
    g = np.random.default_rng(seed); nb = b.copy(); po = pe = 0.0; k = 0
    for i in U[g.random(len(U)) < r]:
        donors = bylen.get(int(calls[i]))
        if not donors: continue
        dv = C[seqs(donors[g.integers(len(donors))])]; nb[seqs(i)] = np.where(dv, 67, 84); o, e = pattern_stats(dv); po += o * n[i]; pe += e * n[i]; k += 1
    x = TE.read(dict(f, b=nb), bv)
    exp_g = (base["gained"]["errors"] + pe) / (base["gained"]["opportunities"] + po)
    res = dict(planted_lines=k, gained_eps=x["gained"]["eps"], gained_expected=exp_g, held_eps=x["held"]["eps"], held_base=base["held"]["eps"],
               planted_pattern_eps=pe / po, all_eps=x["all"]["eps"], all_base=base["all"]["eps"],
               PASS=bool(abs(x["gained"]["eps"] - exp_g) < 1e-12 and abs(x["held"]["eps"] - base["held"]["eps"]) < 1e-12))
    out[f"r{r}"] = res; print(r, {a: (round(v, 6) if isinstance(v, float) else v) for a, v in res.items()}, flush=True)
out["PASS"] = all(out[k]["PASS"] for k in ("r0.01", "r0.05")); json.dump(out, open(sys.argv[3], "w"), indent=1); print("RECOVERY TEST", "PASSED" if out["PASS"] else "FAILED")
