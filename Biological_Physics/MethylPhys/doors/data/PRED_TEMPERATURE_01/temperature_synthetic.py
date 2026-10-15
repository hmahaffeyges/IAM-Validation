"""PRED-TEMPERATURE-01 synthetic (2026-10-15, before any data is searched). Molecules are planted at two temperatures with the firm-rate
mistakes the sealed law sets (temperature_law.json), counted with the committed PROC-CHANNEL-01 counter
(development/island_eraser_2026-10-15/island_eraser.py count(), unchanged), and the recovered within-species log-odds ratio is compared
with the sealed value. Repeated at several molecule counts to find how many molecules it takes to tell the two temperatures apart.
Run 2 (after run 1, NOT MET, showed a systematic low bias from the analytic inversion eps(1 - eps)^2): counts are converted back
to an error rate through the counter's own response curve, measured on planted molecules (500,000 per point, its own seed), not a formula.
MET if, at the largest count, every repeat recovers the sealed ratio within 1 percent and the two temperatures are always told apart."""
# NO CHAIN: plants molecules and counts them with the committed PROC-CHANNEL-01 counter; no chain stage reads temperature
import os, sys, json, math, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", "..", "..", "..", "..")
_f = os.path.join(ROOT, "development", "island_eraser_2026-10-15", "island_eraser.py"); ns = {"__file__": _f}
exec(open(_f).read().split("if sys.argv[1] ==")[0], ns); count = ns["count"]
T = json.load(open(os.path.join(H, "temperature_law.json")))
pairs = [("18 C", "28 C", 291.15, 301.15), ("torpor 5 C vs active 37 C", None, 278.15, 310.15)]
def planted(e, n, rng, L=10):
    for i in range(n):
        calls = "".join("T" if rng.random() < e else "C" for _ in range(L)); yield f"chr1\t{i}\t{calls}\t1"
def logodds(x): return math.log((1 - x) / x)
crng = np.random.default_rng(2026)
grid = np.linspace(0.005, 0.06, 23)
resp = np.array([count(planted(g, 500000, crng))["loss"] for g in grid])          # what the counter reports for each planted error rate
def inv_curve(q): return float(np.interp(q, resp, grid))                          # response is increasing; checked below
assert np.all(np.diff(resp) > 0), "counter response not increasing"
rng = np.random.default_rng(16); out = []
e37 = T["basis_37C"]["firm"]
for name, _, Ta, Tb in pairs:
    ea = 1 / (1 + math.exp(logodds(e37) * 310.15 / Ta)); eb = 1 / (1 + math.exp(logodds(e37) * 310.15 / Tb)); sealed = Tb / Ta
    for n in (2000, 20000, 200000):
        rs = []; apart = 0
        for rep in range(5):
            ca = count(planted(ea, n, rng)); cb = count(planted(eb, n, rng))
            # counts are converted to an error rate through the counter's measured response curve
            inv = inv_curve
            xa, xb = inv(ca["loss"]), inv(cb["loss"]); rs.append(logodds(xa) / logodds(xb)); apart += xa < xb
        out.append(dict(pair=name, molecules=n, sealed=round(sealed, 4), recovered_median=round(float(np.median(rs)), 4),
                        worst_miss_pct=round(100 * max(abs(r / sealed - 1) for r in rs), 2), told_apart=f"{apart} of 5"))
for o in out: print(o)
big = [o for o in out if o["molecules"] == 200000]
ok = all(o["worst_miss_pct"] <= 1 and o["told_apart"] == "5 of 5" for o in big)
L = [f"{o['pair']}, {o['molecules']} molecules per temperature: sealed {o['sealed']}, recovered {o['recovered_median']}, worst miss {o['worst_miss_pct']} percent, told apart {o['told_apart']}" for o in out]
L.append(f"SYNTHETIC: {'MET' if ok else 'NOT MET'}")
open(os.path.join(H, "temperature_synthetic.txt"), "w").write("\n".join(L) + "\n"); print(L[-1])
