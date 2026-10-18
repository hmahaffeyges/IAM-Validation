"""LESSONS C23 check (2026-10-18): how far the counter's correction, as it was before the fix (response curve weighted by the length mix of
EVERY molecule), moves the temperature ratio when the law is exactly true. Planted genome: 90 % methylated stretches (1 % and 3 % CpG density)
and 10 % unmethylated islands (10 %), 150-letter molecules. 'right' weights by molecules of the methylated stretches only; 'old' by all.
Both curves exact (binomial). Written through the chain."""
import os, sys, numpy as np
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "chain")))
import stage_temperature as TL
rng = np.random.default_rng(23); L = 150
def calls_hist(cg, n_mol):
    k = rng.binomial(L - 1, cg, n_mol); return {int(a): int(b) for a, b in zip(*np.unique(k, return_counts=True)) if a >= 6}
meth = {}; isl = calls_hist(0.10, 25_000)
for cg, m in ((0.01, 175_000), (0.03, 50_000)):
    for a, b in calls_hist(cg, m).items(): meth[a] = meth.get(a, 0) + b
allm = dict(meth); [allm.__setitem__(a, allm.get(a, 0) + b) for a, b in isl.items()]
def g_old(N, p):   # the pre-fix curve: every molecule of length n weighted by its chance to qualify
    iso = opp = 0.0
    for n, w in N.items():
        kmax = int(np.floor(n - TL.Q2.QUAL_MIN_METH * n + 1e-9)); k = np.arange(kmax + 1)
        pk = np.array([comb(n, int(x)) for x in k], float) * p ** k * (1 - p) ** (n - k)
        iso += w * (n - 2) * np.sum(pk * k * (n - k) * (n - k - 1)) / (n * (n - 1) * (n - 2)); opp += w * pk.sum() * (n - 2)
    return iso / opp
G = np.linspace(0.005, 0.10, 191); g_right = np.array([g_old(meth, p) for p in G]); g_all = np.array([g_old(allm, p) for p in G])
lo = lambda p: np.log((1 - p) / p); law = TL.predicted_ratio(10, 16); out = dict(law_ratio=round(law, 4), B1_window=[1.0159, 1.0265], rows=[])
for pw in (0.03, 0.045, 0.06, 0.08):
    pc = 1 / (1 + np.exp(law * lo(pw))); rc, rw = (float(np.interp(p, G, g_right)) for p in (pc, pw))
    ic, iw = (float(np.interp(r, g_all, G)) for r in (rc, rw)); tc, tw = (float(np.interp(r, g_right, G)) for r in (rc, rw))
    out["rows"].append(dict(p_warm=pw, right_mix=round(lo(tc) / lo(tw), 4), all_molecule_mix=round(lo(ic) / lo(iw), 4)))
TL.Q2.write_json(out, os.path.join(HERE, "counter_bias_check.json"), manifest=["PHY-057"]); print(out)
