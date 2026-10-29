"""DEV-METAA-ATLAS-CURVE-01, Met-A on sequencing with the atlas's source term: synthetic (written before any sequencing sample is read with it).
For three atlas columns, a planted healthy sequencing sample: at each identity site, n reads (n from 4 to 30, seed 20261028) drawn from the
column's level in sequencing units (sequencing_term_v1); read by stage_metaa_atlas.read with its counts. Bars: (1) the healthy plant reads
within 0.03 of 1 in every column; (2) the same plant read against the column on the array scale (the previous rule) reads below 0.9, the
gap the 17 sequencing inputs showed; (3) a plant that lost 5 % of its pattern (every level moved 5 % toward 0.5) reads above its healthy
plant in every column.
Run 2 (amended after run 1, disclosed): run 1 read one plant per column and missed bar 1 in b cells (1.0339); over 30 plants the column
reads mean 1.0004, sd 0.0209, so a single plant carries counting scatter of about 0.02. Bar 1 now tests bias, as intended: the mean of 30
plants within 0.01 of 1 in every column; bars 2 and 3 unchanged (read on the first plant). Usage: python3 seq_synthetic.py OUT.txt"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "chain"))
import stage_metaa_atlas as SA
rng = np.random.default_rng(20261028); L = []; ok = []
for cell in ("b cells", "naive cd4 t cells", "cd8 t cells"):
    S = SA.identity_sites(cell); mu = SA._col("mu", cell).reindex(S).dropna()
    ps = (mu * SA.SEQ_TERM["b"] + SA.SEQ_TERM["a"]).clip(0, 1); n = pd.Series(rng.integers(4, 31, len(ps)), index=ps.index)
    def plant(p): return pd.Series(rng.binomial(n.to_numpy(), p.to_numpy()) / n.to_numpy(), index=p.index)
    H30 = [SA.read(plant(ps), cell, "sorted " + cell, counts=n)["A"] for _ in range(30)]; h = H30[0]   # run 2: 30 plants
    t0 = dict(SA.SEQ_TERM); SA.SEQ_TERM.update(a=0.0, b=1.0); old = SA.read(plant(ps), cell, "sorted " + cell, counts=n)["A"]; SA.SEQ_TERM.update(t0)
    lost = SA.read(plant(ps + 0.05 * (0.5 - ps)), cell, "sorted " + cell, counts=n)["A"]
    ok += [abs(float(np.mean(H30)) - 1) <= 0.01, old < 0.9, lost > h]
    L.append(f"{cell}: healthy, mean of 30 {np.mean(H30):.4f} (sd {np.std(H30):.4f}) | array-scale column (previous rule) {old:.4f} | 5 % loss {lost:.4f} | sites {len(ps)}")
L.append(f"synthetic (all bars, three columns): {'met' if all(ok) else 'not met'} {ok}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
