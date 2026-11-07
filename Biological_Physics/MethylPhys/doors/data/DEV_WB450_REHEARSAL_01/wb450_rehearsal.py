"""DEV-WB450-REHEARSAL-01 (development, spent data only): how whole-sample Met-A behaves on healthy 450K whole bloods tared on their own
run, before the plan 60.6 test on GSE85210 is written. Reads the pass-1 bundles of the series run on GSE67393's test half (spent; DEV-CSCORE-LIMIT-01)
read by the one command with --full-chain --cell-panel atlas --dev-metaa-allsites. Each blood's whole-sample Met-A against the median of the
run's other bloods (leave-one-out), and against the chain's own healthy prediction. Usage: python3 wb450_rehearsal.py BUNDLE_DIR OUT.txt"""
import sys, os, glob, json, pandas as pd
src, out = sys.argv[1:3]; rows = []
for f in glob.glob(os.path.join(src, "pass1-*", "GSM*.json")):
    if f.endswith(("_values.json", "_planted.json")): continue
    B = json.load(open(f)); w = B.get("met_a_whole") or {}
    rows.append(dict(gsm=os.path.basename(f)[:10], A=pd.to_numeric(w.get("A"), errors="coerce"), A_pred=pd.to_numeric((w.get("healthy_prediction") or {}).get("A_pred"), errors="coerce")))
D = pd.DataFrame(rows).drop_duplicates("gsm").reset_index(drop=True); D["against prediction"] = D.A / D.A_pred
D["tared on the run"] = [D.A[i] / D.A[D.index != i].median() for i in D.index]; L = [f"bloods read: {len(D)}"]
for c in ("A", "against prediction", "tared on the run"):
    v = D[c].dropna(); x = v / v.median() if c == "A" else v
    L.append(f"{c}: median {v.median():.4f}, standard deviation {v.std():.4f}, range {v.min():.3f} to {v.max():.3f}, share within 0.95 to 1.05{' of the median' if c == 'A' else ''} {((x >= 0.95) & (x <= 1.05)).mean():.3f}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
