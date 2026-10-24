"""DEV-ALL-INFO-01 pass 1 scorer (written and committed before any count exists). It only summarizes what run_sample.py's Stage Q2 count
recorded for each sample (held error, its opportunities, loose share, tiles) by group from all_samples.csv, with each group's median and
the leukaemias against the cord-blood controls (two-sided Mann-Whitney p).  Usage: python3 pass1_score.py FOLDER [OUT.txt]"""
import sys, os, glob, json, csv, numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "pass1_summary.txt")
S = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "all_samples.csv")))}; C = {}
for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
    try: B = json.load(open(f))
    except Exception: continue
    q = B.get("count") or (B.get("iam_a_count_only") or {}).get("count") or (B.get("iam_a") or {}).get("count")
    g = next((k for k in S if k in os.path.basename(f) or k == B.get("sample_id")), None)
    if q and g and q.get("held_opp"): C[g] = q
L = [f"DEV-ALL-INFO-01 pass 1: Stage Q2 counts at 10 million read pairs, {len(C)} of {len(S)} samples"]
G = {}
for g, r in sorted(S.items()):
    q = C.get(g); G.setdefault(r["genotype"], []).append(q)
    L.append(f"{g} {r['title']:22} {r['genotype']:18} " + (f"held error {q['held_err']:.5f} (opp {q['held_opp']}) | loose share {q.get('loose_share')}" if q else "no count"))
for k, v in G.items():
    v = [q for q in v if q]
    if v: L.append(f"{k:18} n {len(v):2} | held error median {np.median([q['held_err'] for q in v]):.5f} | loose share median {np.median([q['loose_share'] for q in v if q.get('loose_share') is not None]):.4f}")
a = [q for q in G.get("leukaemia", []) if q]; b = [q for q in G.get("cord-blood B cells", []) if q]
if len(a) >= 3 and len(b) >= 2:
    for f in ("held_err", "loose_share"):
        x = [q[f] for q in a if q.get(f) is not None]; y = [q[f] for q in b if q.get(f) is not None]
        L.append(f"leukaemia against cord-blood B cells, {f}: {np.median(x):.5f} against {np.median(y):.5f} | Mann-Whitney p {stats.mannwhitneyu(x, y).pvalue:.2g}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
