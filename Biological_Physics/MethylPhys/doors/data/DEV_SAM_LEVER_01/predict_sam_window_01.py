"""DEV-SAM-LEVER-01: the IAM-A window Stage Q must read, from the calculated prediction (development/sims/common.restore_ratio_A,
unchanged) mapped through Stage Q's measured response on the wild-type molecules (insilico_loss_02.py on an aligned wild-type .pat).
The calculation gives a fold rise in true copy error, f = ε/ε0, for each (liver SAM, renewed share). insilico_loss_02 plants a scattered loss δ:
true ε = ε_v + δ(1 − ε_v), so its true fold is eps_simple/eps_v, and IAMA_rel is what Stage Q reads. Window = measured IAMA_rel at the
table's smallest and largest f; middle cell (60 µM, 60 % renewed) also given. Usage: predict_sam_window_01.py RESPONSE.csv"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../../../../../development/sims"))
from common import restore_ratio_A, EPS0
from sam_lever_01 import FOLD
R = pd.read_csv(sys.argv[1]); R["fold"] = R.eps_simple / R.eps.iloc[0]
ok = R[R.share_read >= 0.70]
def read_at(f): return float(np.interp(f, R.fold, R.IAMA_rel)) if f <= ok.fold.max() else None
rows = []
for S0 in (30, 60, 90):
    for ren in (0.3, 0.6, 1.0):
        A, e = restore_ratio_A(S0, FOLD, ren); f = e / EPS0; rows.append(dict(SAM_uM=S0, renewed=ren, IAMA_true=round(A, 4), fold=round(f, 4), IAMA_StageQ=read_at(f)))
T = pd.DataFrame(rows); print(f"wild-type eps_v {R.eps.iloc[0]:.5f}; readable (>= 70 % of molecules) to true fold {ok.fold.max():.3f}")
print(T.to_string(index=False))
mid = T[(T.SAM_uM == 60) & (T.renewed == 0.6)].IAMA_StageQ.iloc[0]
print(f"WINDOW (Stage Q, knockout vehicle / wild type): {T.IAMA_StageQ.min():.4f} - {T.IAMA_StageQ.max():.4f}; middle cell {mid:.4f}")
