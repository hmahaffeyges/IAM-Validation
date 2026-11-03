"""Healthy pass 6 (the common floor) against pass 5 (no floor), by each input's floor (diagnostic, added after pass 6 was scored; nothing
scored). From each pass 6 bundle's floor records and the two passes' readings tables. Usage: python3 pass6_floor_diag.py BUNDLE_DIR OUT.txt"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); R5 = pd.read_csv(os.path.join(H, "healthy_fullchain5_readings.csv")); rows = []
for f in os.listdir(sys.argv[1]):
    if not f.endswith("_bundle.json"): continue
    B = json.load(open(os.path.join(sys.argv[1], f))); i = ((((B.get("territories") or {}).get("sets") or {}).get("islands") or {}).get("counted") or {}); m = (B.get("development") or {}).get("metaa_atlas") or {}
    fi, fm = i.get("floor") or {}, m.get("floor") or {}
    rows.append(dict(gsm=f[:-12], islands6=i.get("A"), fi=(fi.get("sample") or np.nan) - (fi.get("reference") or np.nan), metaa6=m.get("A"), fm=(fm.get("sample") or np.nan) - np.mean(fm.get("donors") or [np.nan])))
D = pd.DataFrame(rows).set_index("gsm")
for k, rd in (("islands", "territory islands"), ("metaa", "Met-A sorted cell")): D[k + "5"] = R5[R5.reading == rd].drop_duplicates("gsm").set_index("gsm").A.reindex(D.index)
L = []
for k, x in (("islands", "fi"), ("metaa", "fm")):
    g = D[[k + "5", k + "6", x]].dropna(); s5 = np.polyfit(g[x], g[k + "5"], 1)[0]; sc = np.polyfit(g[x], g[k + "6"] - g[k + "5"], 1)[0]
    L.append(f"{k}: {len(g)} inputs; median pass 5 {g[k + '5'].median():.4f}, pass 6 {g[k + '6'].median():.4f}; against the floor difference (sample minus reference): "
             f"pass 5 rho {spearmanr(g[x], g[k + '5']).correlation:+.3f}, pass 6 rho {spearmanr(g[x], g[k + '6']).correlation:+.3f}; slope of the reading {s5:+.2f}, of the correction {sc:+.2f} (share of the correction the data carry {-s5 / sc:.3f})")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
