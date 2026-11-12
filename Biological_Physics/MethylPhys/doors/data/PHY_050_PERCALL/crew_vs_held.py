"""Restore crew and copying against each cell type's held error (PHY-079; exploration on spent data; nothing sealed). The October test
DEV-POSITION-EXPR-01 (sealed eraser-to-writer form: not met) reported, after its run, a writer-alone lead (rho +0.36 for 1/(DNMT1+UHRF1)).
Here the same Human Protein Atlas values (position_expr_01_points.csv: W = DNMT1 + UHRF1, MKI67) are set against today's per-type held error
over the M-based prediction (scatter_split.csv, the lone-loss counter) and the census lifespan (lifespan_vs_held.csv). Spearman; a
two-predictor rank model scored leave-one-out. Writes crew_vs_held.csv and crew_vs_held_summary.txt."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(H, "..", "DEV_POSITION_EXPR_01", "position_expr_01_points.csv"))
S = pd.read_csv(os.path.join(H, "scatter_split.csv")).set_index("cell")["median"]; LF = pd.read_csv(os.path.join(H, "lifespan_vs_held.csv")).set_index("cell").lifespan_days
rows = []
for _, r in X.iterrows():
    cs = [c.strip() for c in str(r.loyfer).split(";") if c.strip() in S.index]
    if cs:
        lf = [LF[c] for c in cs if c in LF.index]
        rows.append(dict(hpa=r.hpa, cells="; ".join(cs), held_over_prediction=float(np.median([S[c] for c in cs])), crew_DNMT1_UHRF1=float(r.W), MKI67=float(r.MKI67), lifespan_days=float(np.median(lf)) if lf else np.nan))
Z = pd.DataFrame(rows).sort_values("held_over_prediction"); Z.to_csv(os.path.join(H, "crew_vs_held.csv"), index=False)
a = spearmanr(Z.crew_DNMT1_UHRF1, Z.held_over_prediction); b = spearmanr(Z.MKI67, Z.held_over_prediction)
zz = Z.dropna(subset=["lifespan_days"]); c = spearmanr(np.log(zz.lifespan_days), zz.held_over_prediction)
rk = lambda v: pd.Series(v).rank().to_numpy(); y = rk(zz.held_over_prediction); A = np.column_stack([np.ones(len(zz)), rk(zz.crew_DNMT1_UHRF1), rk(np.log(zz.lifespan_days))])
loo = [A[i] @ np.linalg.lstsq(A[np.arange(len(zz)) != i], y[np.arange(len(zz)) != i], rcond=None)[0] for i in range(len(zz))]
t = (f"Crew and copying against held error (exploration). {len(Z)} cell types: restore crew (DNMT1 + UHRF1 RNA) rho {a.statistic:+.2f} (p {a.pvalue:.3f}); "
     f"proliferation (MKI67) rho {b.statistic:+.2f} (p {b.pvalue:.3f}). With a census lifespan, n={len(zz)}: lifespan rho {c.statistic:+.2f} (p {c.pvalue:.3f}); "
     f"crew plus lifespan, leave-one-out rho {spearmanr(loo, zz.held_over_prediction).statistic:+.2f}.\nMore restore crew, lower held error; crew is made mostly in copying cells, so crew and copying are partly one signal.\n\n" + Z.to_string(index=False) + "\n")
open(os.path.join(H, "crew_vs_held_summary.txt"), "w").write(t); print(t[:600])
