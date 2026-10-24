"""DEV-ERASE-01 maintenance knockout scorer (written and committed before either count exists; plan 14.10). It only summarizes what
run_sample.py's Stage Q2 count recorded for the Reh control (WT) and the UHRF1 knockout (KO): held error, its opportunities, loose share and
tiles, and the knockout's held error and loose share as multiples of the control's.  Usage: python3 uhrf1_score.py FOLDER [OUT.txt]"""
import sys, os, glob, json
d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "uhrf1_summary.txt")
G = {"GSM4914798": "control (Reh-NT)", "GSM4914799": "UHRF1 knockout"}; C = {}
for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
    try: B = json.load(open(f))
    except Exception: continue
    q = B.get("count") or (B.get("iam_a_count_only") or {}).get("count") or (B.get("iam_a") or {}).get("count")
    g = next((k for k in G if k in os.path.basename(f) or k == B.get("gsm") or k == B.get("sample_id")), None)
    if q and g and q.get("held_opp"): C[g] = q
L = ["DEV-ERASE-01 maintenance knockout (GSE116229 Reh pair), Stage Q2 counts"]
for g, n in G.items():
    q = C.get(g); L.append(f"{n:18} " + (f"held error {q['held_err']:.5f} (opportunities {q['held_opp']}) | loose share {q.get('loose_share')} (tiles {q.get('tiles')})" if q else "no count found"))
if len(C) == 2:
    a, b = C["GSM4914798"], C["GSM4914799"]
    L.append(f"knockout against control: held error x {b['held_err'] / a['held_err']:.3f} | loose share x {(b['loose_share'] or 0) / max(a['loose_share'] or 1e-9, 1e-9):.3f}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
