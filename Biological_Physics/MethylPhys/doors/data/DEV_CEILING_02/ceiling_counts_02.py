"""DEV-CEILING-02: per-pair direction counts quoted in the note, computed from the committed rows file (written after the run, adds
no method - it only counts what ceiling_position_02.py already wrote).

# NO CHAIN: counting over ceiling_position_02_rows.csv.
"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(HERE, "ceiling_position_02_rows.csv"))
L = []
for c, g in T.groupby("cancer", sort=False):
    L.append(f"{c}: methylated arm moved toward the ceiling in {int((g.u_m > 0).sum())} of {len(g)}; u_m quartiles "
             f"{g.u_m.quantile(.25):+.3f} to {g.u_m.quantile(.75):+.3f}; largest {g.u_m.max():+.3f}; unmethylated arm toward it in {int((g.u_u > 0).sum())} of {len(g)}")
L.append(f"all four cancers: methylated arm toward the ceiling in {int((T.u_m > 0).sum())} of {len(T)}; past the ceiling: {int(T.past_ceiling.sum())}")
open(os.path.join(HERE, "ceiling_counts_02_output.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
