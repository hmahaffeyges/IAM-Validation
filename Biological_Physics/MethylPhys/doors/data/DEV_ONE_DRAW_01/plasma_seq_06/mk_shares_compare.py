"""plasma_seq_06 against plasma_seq_02: the 30 tuning plasmas' cell shares with and without the provisional megakaryocyte column in the
Deconvolver's reference (development, diagnosis only; added after both were read; nothing scored). From the committed share diagnostics
(plasma_seq_02/plasma_shares_diag.csv, plasma_seq_06/plasma_shares_diag_06.csv). Usage: python3 mk_shares_compare.py OUT.txt"""
import os, sys, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__))
a = pd.read_csv(os.path.join(H, "..", "plasma_seq_02", "plasma_shares_diag.csv")).set_index("run"); b = pd.read_csv(os.path.join(H, "plasma_shares_diag_06.csv")).set_index("run").reindex(a.index)
ep, mk = "share|erythrocyte progenitors", "share|megakaryocytes"
L = [f"{len(a)} plasmas; erythrocyte progenitors median {a[ep].median():.4f} without megakaryocytes, {b[ep].median():.4f} with; megakaryocytes median {b[mk].median():.4f} (5 to 95 % {b[mk].quantile(.05):.4f} to {b[mk].quantile(.95):.4f})",
     f"the erythrocyte progenitor drop against the megakaryocyte share: rho {spearmanr(a[ep] - b[ep], b[mk]).correlation:+.3f}; the megakaryocyte share against plasma_seq_02's domains: rho {spearmanr(b[mk], a.domains).correlation:+.3f}"]
for c in ("granulocytes", "monocytes", "colon macrophages", "nk cells", "hepatocyte"): L.append(f"{c}: {a['share|' + c].median():.4f} without, {b['share|' + c].median():.4f} with")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
