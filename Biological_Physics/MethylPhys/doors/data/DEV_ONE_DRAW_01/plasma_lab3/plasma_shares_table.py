"""The IAM-Deconvolver's cell shares of plasmas read by the one command (2026-11-08; development, nothing scored): every bundle under LAND
(as written by plasma_lab3_read.sh), its cells found and shares, written to OUT/plasma_shares.csv (one row per plasma and cell) and
OUT/plasma_shares_summary.txt (median share of each cell over the plasmas, and how many plasmas hold it at 1 % or more). Run on GitHub by
score_run.yml, since each run's files are too large to bring to a laptop. Usage: plasma_shares_table.py LAND OUT"""
import os, sys, json, glob
import pandas as pd
land, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); rows = []
for f in glob.glob(os.path.join(land, "**", "*.json"), recursive=True):
    try: B = json.load(open(f))
    except Exception: continue
    cf = (B.get("cells_found") or {}).get("cells") if isinstance(B, dict) else None
    if not cf: continue
    sid = B.get("sample_id") or B.get("id") or os.path.basename(f).split("_")[0].split(".")[0]
    for c in cf: rows.append(dict(sample=sid, cell=c.get("cell"), share=c.get("share"), ruler=str((B.get("sequencing_ruler") or {}).get("deconvolver"))))
X = pd.DataFrame(rows).drop_duplicates(["sample", "cell"]); X.to_csv(os.path.join(out, "plasma_shares.csv"), index=False)
n = X["sample"].nunique(); W = X.pivot_table(index="sample", columns="cell", values="share", aggfunc="first").fillna(0)
S = pd.DataFrame({"median_share": W.median(), "plasmas_at_1pct": (W >= 0.01).sum()}).sort_values("median_share", ascending=False)
open(os.path.join(out, "plasma_shares_summary.txt"), "w").write(f"{n} plasmas; Deconvolver: {sorted(X.ruler.unique())}\n" + S.round(4).to_string() + "\n")
print(open(os.path.join(out, "plasma_shares_summary.txt")).read()[:3000])
