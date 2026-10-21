"""DEV-MAPPABILITY-01: the masked re-read scored (exploration, not sealed). Per array, each reading with the poor-mapping probes removed
against the same array as read: Met-A, the C-Score, the neutrophil share, and the six Stage D21 shares. Usage: python3 masked_reread_01_score.py
ROWS_DIR. Writes masked_reread_01_summary.txt here."""
import os, sys, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(sys.argv[1], "masked_reread_01_rows.csv")); T = T[T.case != "ERROR"]
a = T[T.case == "as read"].set_index("gsm"); m = T[T.case == "poor-mapping probes removed"].set_index("gsm"); g = a.index.intersection(m.index)
L = [f"arrays {len(g)}; probes removed per array: median {int((a.n_probes - m.n_probes).loc[g].median())}; Met-A identity sites measured: {int(a.sites.loc[g].median())} -> {int(m.sites.loc[g].median())}"]
for c, name in [("A", "Met-A"), ("C", "C-Score"), ("f_neu", "Stage A neutrophil share")] + [(k, "Stage D21 " + k[6:]) for k in a.columns if k.startswith("share_")]:
    d = (m[c] - a[c]).loc[g].dropna()
    if len(d): L.append(f"{name}: mean change {d.mean():+.4f}, largest {d.abs().max():.4f} (as read: median {a[c].loc[g].median():.4f})")
open(os.path.join(H, "masked_reread_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
