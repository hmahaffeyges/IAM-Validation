"""The information globe of the IAM Sequencing Atlas beside IAM-Atlas v2.1's, from the build reports of run_sample.py --build info-globe
(globe/seq_globe.json, globe/v21_globe.json). Nothing scored. Usage: python3 globe_compare.py OUT.csv OUT.txt"""
import os, sys, json, pandas as pd
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "globe"); G = {n: json.load(open(os.path.join(H, n + "_globe.json"))) for n in ("seq", "v21")}
D = pd.DataFrame({f"{n}_{k}": {f: (v if k == "sites" else (g["concentration_by_family"].get(f) or {}).get(k)) for f, v in g["sites_by_family"].items()}
                  for n, g in G.items() for k in ("sites", "all", "clear_methylated_there")}); D.index.name = "family"; D.to_csv(sys.argv[1])
L = [f"{g['atlas']}: {g['sites']} sites placed, {g['sites_clear']} clear, margin median {g['margin_median']}; family concentration median {D[n + '_all'].median():.2f}"
     for n, g in G.items()]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
