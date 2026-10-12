"""DEV-COMPOSITION-TRUTH-04 feasibility of bar 7 on a clinical truth set (2026-10-11; no array read). The measured RMSE against a clinical
differential combines the chain's own error m and the count's error t: RMSE ~ sqrt(m^2 + t^2) (independent errors). Bar 7 is RMSE <= 0.02,
so it can be met only if m <= sqrt(0.02^2 - t^2). The chain's m from the records: 0.019 on the reference laboratory's own constructed EPIC
mixtures (COMMISSIONING_NOTE bar 7, in-sample); 0.061-0.138 for other methods on out-of-laboratory mixtures (DEV-NILC-01, not the chain's
commissioned path). t is not known for GSE167202's analyser; it is tabulated over a range. The count is automated (one-decimal percentages and an
immature-granulocyte %, gse167202_cbc.csv), so cell-sampling error is small and t is set by method bias and scatter. Also written: the share of
samples whose immature-granulocyte % exceeds 2 (cells a methylation neutrophil reading may count but 'segmented neutrophils' does not).
Writes feasibility_bar7_output.txt. Usage: python3 feasibility_bar7.py"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = pd.read_csv(os.path.join(HERE, "gse167202_cbc.csv"))
out = ["truth error t | largest chain error m that can still meet RMSE 0.02 | RMSE if m = 0.019 (in-sample value)"]
for t in (0.005, 0.010, 0.015, 0.020):
    mmax = np.sqrt(max(0.0, 0.02 ** 2 - t ** 2)); out.append(f"{t:.3f} | {mmax:.4f} | {np.sqrt(0.019 ** 2 + t ** 2):.4f}")
ig = C.immature_granulocytes_percent.dropna()
out.append(f"immature granulocytes %: given for {len(ig)}; above 2 % in {int((ig > 2).sum())} ({(ig > 2).mean():.1%}); above 5 % in {int((ig > 5).sum())}")
open(os.path.join(HERE, "feasibility_bar7_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
