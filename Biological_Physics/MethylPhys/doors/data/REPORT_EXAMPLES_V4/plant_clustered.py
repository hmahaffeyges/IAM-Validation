"""A planted clustered departure for the example report (2026-10-18), by the C-Score's own positive-control rule (CSCORE_COMMISSIONING_PLAN
step 2): F = 10 % of the C-Score's sites, as contiguous stretches in its own site order (conductor ref() sites_ordered), each moved 10 % of
the way toward beta = 0.5. 20 stretches, placed with a fixed seed. Labeled PLANTED wherever it is shown: it is a real healthy array with a
known change put in, not a patient sample. Usage: python3 plant_clustered.py IN_BETAS.csv OUT_BETAS.csv CHAIN_DIR"""
import sys, numpy as np, pandas as pd
src, dst, chain = sys.argv[1:4]; sys.path.insert(0, chain)
import conductor_v3 as C3
S = list(C3.ref()["sites_ordered"]); F, LOSS, K = 0.10, 0.10, 20
b = pd.read_csv(src, index_col=0).iloc[:, 0].astype(float)
have = [s for s in S if s in b.index and np.isfinite(b[s])]; n = int(round(F * len(have))); L = n // K
rng = np.random.default_rng(20261018); starts = np.sort(rng.choice(len(have) - L, K, replace=False))
pick = sorted({have[i] for s0 in starts for i in range(s0, s0 + L)})
b.loc[pick] = b.loc[pick] + LOSS * (0.5 - b.loc[pick])
b.to_frame("beta").rename_axis("cpg_id").to_csv(dst); print(f"planted: {len(pick)} of {len(have)} C-Score sites in {K} stretches, loss {LOSS} toward 0.5")
