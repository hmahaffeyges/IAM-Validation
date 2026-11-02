"""BLUEPRINT's healthy libraries at the islands (diagnostic, added after the outside-laboratory test was scored; nothing scored): from each
one-command bundle, the islands' level change against the sample's own mix of the IAM Sequencing Atlas, the islands reading, the median calls
per site and the sites with a value. Usage: python3 blueprint_islands_offset.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, json, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "blueprint_healthy_test_samples.csv")); rows = []
for g in S.gsm:
    f = os.path.join(sys.argv[1], g + "_bundle.json")
    if not os.path.exists(f): continue
    B = json.load(open(f)); i = ((B.get("territories") or {}).get("sets") or {}).get("islands") or {}; v = B.get("sequencing_values") or {}
    rows.append(dict(gsm=g, islands_level_change=i.get("level_change"), islands_A=i.get("A"), calls_per_site_median=v.get("calls_per_site_median"), sites_with_value=v.get("sites_with_value")))
D = pd.DataFrame(rows); D.to_csv(sys.argv[2], index=False)
L = [f"{len(D)} libraries; islands level change median {D.islands_level_change.median():.5f} (below the atlas in {int((D.islands_level_change < 0).sum())});",
     f"islands reading against the level change: rho {spearmanr(D.islands_level_change, D.islands_A, nan_policy='omit').correlation:+.3f}; median calls per site {D.calls_per_site_median.median():g}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
