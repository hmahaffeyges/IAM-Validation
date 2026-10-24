"""DEV-FIRES-01, added after the first reading (disclosed): the same Group-A comparison expressed as an effect size (Cohen's d on the tared
readings, pooled SD), so it can be set beside the earlier engine's effect sizes window by window. Women only (--sex F semantics of
fires_score.py). Readings: Met-A for the whole sample, its islands part, Met-A neutrophils, the domains territory.
Usage: python3 fires_effect.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); OUT_EFFECT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], "fires_effect_sexF.txt")
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null", "--sex", "F"]
exec(open(os.path.join(H, "fires_score.py")).read().split("T = [f\"DEV-FIRES-01")[0])
L = ["DEV-FIRES-01 effect sizes, women only, Group-A tare (cancer-free on the same chip); Cohen's d on the tared readings"]
for site in ("breast", "colorectal"):
    for wn in ["all", "0-2 y", "2-5 y", "5-10 y", "over 10 y"]:
        keep = (X.group == "cancer-free") | ((X.group == site) & ((win == wn) if wn != "all" else True)); cells = []
        for c in ("Met-A whole", "Met-A whole, islands", "Met-A neutrophils", "territory domains"):
            R = X[keep].assign(A=pd.to_numeric(X[keep][c], errors="coerce")).reset_index()[["gsm", "A", "slide", "group"]]; R["C"] = np.nan
            r = GA.tare(R, healthy.reindex(R.gsm).reset_index(drop=True).fillna(False).astype(bool))
            x, y = r[r.group == site].A_rel.dropna(), r[r.group == "cancer-free"].A_rel.dropna()
            sd = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2)); cells.append(f"{c} d {(x.mean() - y.mean()) / sd:+.2f}")
        L.append(f"{site:10} {wn:9} n {int(((X.group == site) & keep).sum()):3} | " + " | ".join(cells))
open(OUT_EFFECT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
