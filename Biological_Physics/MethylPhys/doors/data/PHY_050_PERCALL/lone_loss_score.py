"""Scores PHY-050's second test (PHY_050_LONE_LOSS_PREREG.md), written before five of the six counts were read (the canary, row 0, was
opened to check the job). Each count is the one command's bundle (run_sample.py --pat --dev-count-only). Prediction per sample:
0.04558 + 0.0284 x loose share. Bars: 1. at least 5 of 6 within 10 % of their own prediction; 2. median observed/predicted 0.97 to 1.03.
Reported beside: molecule-end damage (held error 1-2 calls from the nearer end minus 4-6 calls in).
Usage: python3 lone_loss_score.py <folder of bundles> <summary.txt>"""
import sys, os, glob, json, statistics
P0, S = 0.04558, 0.0284
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "GSM*.json"))):
    c = json.load(open(f))["iam_a_count_only"]["count"]
    d = {int(k): v[0] / v[1] for k, v in (c.get("held_by_distance") or {}).items() if v[1]}
    near = [d[k] for k in (1, 2) if k in d]; far = [d[k] for k in (4, 5, 6) if k in d]
    pred = P0 + S * c["loose_share"]; obs = c["held_err_length"]
    rows.append((os.path.basename(f)[:-5], c["loose_share"], obs, pred, obs / pred, (sum(near) / len(near) - sum(far) / len(far)) if near and far else None))
L = ["PHY-050 second test: lone-loss rate against 0.04558 + 0.0284 x loose share, Johns Hopkins tissue samples labeled normal (GSE86340)",
     "sample loose observed predicted ratio end_damage"]
L += [f"{g} {l:.4f} {o:.5f} {p:.5f} {r:.4f} {('%.4f' % e) if e is not None else 'NA'}" for g, l, o, p, r, e in rows]
within = sum(abs(r[4] - 1) <= 0.10 for r in rows); med = statistics.median(r[4] for r in rows)
b1 = len(rows) == 6 and within >= 5; b2 = 0.97 <= med <= 1.03
L += [f"samples {len(rows)}", f"bar 1 within 10 %: {within} of {len(rows)}: {'met' if b1 else 'not met'}", f"bar 2 median ratio {med:.4f}: {'met' if b2 else 'not met'}",
      f"verdict: {'MET' if b1 and b2 else 'NOT MET'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
