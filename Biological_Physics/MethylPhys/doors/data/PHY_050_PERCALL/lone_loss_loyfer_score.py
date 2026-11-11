"""Scores PHY-050's third test (doors/PHY_050_LONE_LOSS_LOYFER_PREREG.md), written before any of its samples was counted. Each count is the one
command's bundle (run_sample.py --geo-pat --dev-count-only), as written by lone_loss_loyfer_count.py. Prediction per sample: 0.04558 + 0.0284 x
loose share (the second test's, unchanged). Bars: 1. at least 83 % of the samples read within 10 % of their own prediction; 2. the median of
observed over predicted between 0.97 and 1.03. A sample whose count did not come back is listed as not read and does not enter the bars.
Reported beside: molecule-end damage (held error 1-2 calls from the nearer end minus 4-6 calls in), and the sample's cell and tissue.
Usage: python3 lone_loss_loyfer_score.py <folder of counts> <summary.txt>"""
import sys, os, glob, json, csv, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); P0, S = 0.04558, 0.0284
LIST = list(csv.DictReader(open(os.path.join(HERE, "lone_loss_loyfer_unseen_samples.csv"))))
rows, missing = [], []
for r in LIST:
    f = os.path.join(sys.argv[1], r["sample"] + ".json")
    if not os.path.exists(f): missing.append(r["sample"]); continue
    c = json.load(open(f)); c = c.get("iam_a_count_only", {}).get("count", c)
    d = {int(k): v[0] / v[1] for k, v in (c.get("held_by_distance") or {}).items() if v[1]}
    near = [d[k] for k in (1, 2) if k in d]; far = [d[k] for k in (4, 5, 6) if k in d]
    pred = P0 + S * c["loose_share"]; obs = c["held_err_length"]
    rows.append((r["sample"], r["tissue"] + " " + r["cell"], c["loose_share"], obs, pred, obs / pred, (sum(near) / len(near) - sum(far) / len(far)) if near and far else None))
L = ["PHY-050 third test: lone-loss rate against 0.04558 + 0.0284 x loose share, healthy sorted Loyfer samples never counted before (GSE186458)",
     "sample cell loose observed predicted ratio end_damage"]
L += [f"{g} {c.replace(' ', '_')} {l:.4f} {o:.5f} {p:.5f} {q:.4f} {('%.4f' % e) if e is not None else 'NA'}" for g, c, l, o, p, q, e in rows]
n = len(rows); within = sum(abs(r[5] - 1) <= 0.10 for r in rows); med = statistics.median(r[5] for r in rows) if rows else float("nan")
b1 = n > 0 and within >= 0.83 * n; b2 = 0.97 <= med <= 1.03
L += [f"samples read {n} of {len(LIST)}; not read: {' '.join(missing) or 'none'}", f"bar 1 within 10 %: {within} of {n} ({100 * within / max(n, 1):.1f} %; 83 % needed): {'met' if b1 else 'not met'}",
      f"bar 2 median ratio {med:.4f}: {'met' if b2 else 'not met'}", f"verdict: {'MET' if b1 and b2 else 'NOT MET'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
