"""The three PHY-050 middle-number tests rescored by the agreed bar method (2026-11-12; exploration, not a verdict; the sealed verdicts stand).
Pull = (observed / predicted - 1) / combined relative uncertainty (metrology guide, coverage k = 2): statistical (each test's own interval or
a 4,000-draw resampling of its median), the meter (2 % systematic, the counter synthetic at the real clustering; per-call counter 2.5 % on
its 7.2 % low reading), the natural spread for a median of n (7.5 % x 1.25 / sqrt n, scatter_split.py) where the test sets samples against
one prediction, and, in the second column, the coin's own uncertainty (5 % on the ATP free energy, the published in-vivo spread, carried to
1/(1+M)). Within 2 passes; 2-3 watch; beyond 3 a miss. Writes rescore_middle.csv and rescore_middle_summary.txt."""
import os, json, glob, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
M = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
e0 = 1 / (1 + M); g = np.random.default_rng(20261112); coin = 0.05 * M / (1 + M); meter = 0.02
def pull(d, parts): s = float(np.sqrt(sum(p * p for p in parts))); return round(d / s, 2), round(d / np.hypot(s, coin), 2)
def med(r): r = np.array(r); bs = [np.median(g.choice(r, len(r))) for _ in range(4000)]; return float(np.median(r)), float(np.std(bs)) / float(np.median(r))
rows = []
s1 = (0.05165 - 0.04930) / 2 / 1.96 / 0.05048; o1 = 0.05048 / (1 - 0.072); d1 = o1 / e0 - 1   # PHY_050_PERCALL_OUTCOME.md; synthetic per-call bias
rows.append(("test 1: per-call firm rate, 153 Loyfer", "not met", round(o1, 5), round(e0, 5), round(d1, 4), *pull(d1, [s1, 0.025])))
J = [json.load(open(f))["iam_a_count_only"]["count"] for f in sorted(glob.glob(os.path.join(H, "lone_loss_jhu_out", "*.json")))]
m2, s2 = med([c["held_err_length"] / (e0 + 0.0284 * c["loose_share"]) for c in J]); n2 = len(J)
rows.append((f"test 2: lone-loss rate, {n2} tumor-adjacent tissues (JHU)", "not met", round(m2, 4), 1.0, round(m2 - 1, 4), *pull(m2 - 1, [s2, meter, 0.075 * 1.25 / np.sqrt(n2)])))
L = []
for f in sorted(glob.glob(os.path.join(H, "lone_loss_loyfer_out", "*.json"))):
    c = json.load(open(f)); c = c[0] if isinstance(c, list) else c
    if c.get("held_err_length") is not None: L.append(c["held_err_length"] / (e0 + 0.0284 * c["loose_share"]))
m3, s3 = med(L)
rows.append((f"test 3: lone-loss rate, {len(L)} unseen Loyfer samples", "not met (by one sample)", round(m3, 4), 1.0, round(m3 - 1, 4), *pull(m3 - 1, [s3, meter])))
R = pd.DataFrame(rows, columns=["test", "sealed verdict", "observed", "predicted", "off", "pull, coin exact", "pull, coin uncertainty"])
R.to_csv(os.path.join(H, "rescore_middle.csv"), index=False)
t = ("PHY-050 rescored by the agreed bar method (exploration; the sealed verdicts stand; a flip needs a fresh sealed test)\n" + R.to_string(index=False) +
     "\nReading: test 3 (the lone-loss rate) agrees with 1/(1+M) within about half a standard deviation; test 1 (the per-call rate) and test 2\n(tumor-adjacent tissue) remain misses. Losses cluster (coupling 1.41), so the per-call rate counts stretches lost together; the lone-loss rate\nis the quantity the law's held error is set against.\n")
open(os.path.join(H, "rescore_middle_summary.txt"), "w").write(t); print(t)
