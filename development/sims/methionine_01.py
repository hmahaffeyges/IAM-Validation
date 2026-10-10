"""DEV-METHIONINE-01 step 1 (2026-10-10, before any PRJDB12471 read is downloaded): can IAM-A see methionine depletion in the Yokogami 2022
glioma-initiating-cell series (3 lines, control vs methionine-free medium 48-72 h, RRBS)? Calculation unchanged (common.restore_ratio_A: restore
rate ∝ S/(K_m+S)), mapped to Stage Q's measured response on RRBS molecules (stand-in: wild-type mouse liver RRBS, insilico_wt_SRR3111471.csv).
Copy error is only written when DNA is copied: in 48-72 h only a share g of cells copy their DNA (proliferation "markedly decreased", the paper),
and each copied duplex carries one new strand, so the renewed share of molecules is g/2.
Grid: SAM in culture 20/40/60 µM; SAM fall 2x/4x/10x; g 0.25/0.5/1.0.
Detection per line (control vs depleted, one library each): z = (IAM-A_rel - 1) / sd_pair; sd_pair from library-to-library spread
0.02 / 0.04 (same-lab library repeats: mouse runs 3 %, whole-blood TruSeq repeats 4-6 %). Usage: python3 methionine_01.py RESPONSE.csv"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import restore_ratio_A, EPS0
R = pd.read_csv(sys.argv[1]); fold_axis = (R.eps_simple / R.eps_simple.iloc[0]).values
rows = []
for S0 in (20, 40, 60):
    for f in (2, 4, 10):
        for g in (0.25, 0.5, 1.0):
            A, e = restore_ratio_A(S0, f, g / 2); q = float(np.interp(e / EPS0, fold_axis, R.IAMA_rel))
            rows.append(dict(SAM_uM=S0, SAM_fall=f, share_copied=g, true_IAMA=round(A, 4), StageQ_IAMA=round(q, 4),
                             z_sd002=round((q - 1) / 0.02, 1), z_sd004=round((q - 1) / 0.04, 1)))
T = pd.DataFrame(rows); print(T.to_string(index=False))
print("cells with Stage Q IAM-A_rel >= 1.08 (3 of 3 lines detectable at sd 0.02, z >= 4 each... >= 2 sd at sd 0.04):", int((T.StageQ_IAMA >= 1.08).sum()), "of", len(T))
