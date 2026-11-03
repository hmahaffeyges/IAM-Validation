"""plasma_seq_02: each plasma's domains and whole-sample Met-A against every cell's share and the sites read (diagnostic, added after the
30 were scored; nothing scored). From each one-command bundle. Reported: for every cell with a median share of 0.01 or more, the Spearman
rank correlation of its share with the domains reading, largest first; and the domains against the sites with a value and the cells found.
Usage: python3 plasma_shares_diag.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, pandas as pd
from scipy.stats import spearmanr
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "SRR*_bundle.json"), recursive=True)):
    B = json.load(open(f)); s = (B.get("territories") or {}).get("sets") or {}; sh = {x["cell"]: x["share"] for x in ((B.get("cell_shares") or {}).get("cells") or [])}
    rows.append(dict(run=os.path.basename(f).split("_")[0], domains=(s.get("domains") or {}).get("A"), whole=(B.get("met_a_whole") or {}).get("A"),
                     sites=(B.get("sequencing_values") or {}).get("sites_with_value"), cells_found=sum(v >= 0.01 for v in sh.values()), **{"share|" + k: v for k, v in sh.items()}))
D = pd.DataFrame(rows).fillna({c: 0.0 for c in pd.DataFrame(rows).columns if c.startswith("share|")}); D.to_csv(sys.argv[2], index=False)
C = [c for c in D.columns if c.startswith("share|") and D[c].median() >= 0.01]
R = sorted(((c[6:], D[c].median(), spearmanr(D[c], D.domains).correlation) for c in C), key=lambda x: -abs(x[2]))
L = [f"{len(D)} plasmas; domains median {D.domains.median():.4f}; whole-sample Met-A median {D.whole.median():.4f}",
     f"domains against the sites with a value: rho {spearmanr(D.sites, D.domains).correlation:+.3f}; against the cells found (share 0.01 or more): rho {spearmanr(D.cells_found, D.domains).correlation:+.3f}"]
L += [f"{c}: median share {m:.3f}; domains rho {r:+.3f}" for c, m, r in R]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
