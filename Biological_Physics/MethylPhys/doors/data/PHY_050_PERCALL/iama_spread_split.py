"""How much of IAM-A's healthy spread is between cell types? (exploration on spent counts; nothing sealed; 2026-11-13)
IAM-A = held_err / eps_pred(L), eps_pred(L) = 0.040265 + 0.034625 L (IAM-A's baseline, calibrated: selfref_line_v1.json), on the 153 Loyfer
healthy samples it was built on (DEV_CHAIN_Q2_01/percall_out; neurons set aside, their marks outside CpGs are not counted). One-way random
effects on log IAM-A: between-type and within-type standard deviations. If each cell type had its own baseline, only the within-type spread
would remain. Writes iama_spread_split_summary.txt."""
import os, json, glob, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = []
for f in glob.glob(os.path.join(H, "..", "DEV_CHAIN_Q2_01", "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("held_err") is not None: R.append(dict(gsm=x["sample"], cell=str(x["cell"]).lower(), e=x["held_err"], L=x["loose_share"]))
X = pd.DataFrame(R).drop_duplicates("gsm"); X = X[~X.cell.str.contains("neuron")]
X["A"] = X.e / (0.040265 + 0.034625 * X.L); X["lA"] = np.log(X.A)
g = X.groupby("cell").lA; k = X.cell.nunique(); n = len(X)
within = float(np.sqrt(((X.lA - g.transform("mean")) ** 2).sum() / (n - k))); tot = float(X.lA.std(ddof=1))
ms_b = float((g.count() * (g.mean() - X.lA.mean()) ** 2).sum() / (k - 1)); n0 = (n - (g.count() ** 2).sum() / n) / (k - 1)
between = float(np.sqrt(max(0, (ms_b - within ** 2) / n0)))
t = (f"IAM-A on its baseline's own healthy samples: {n} samples, {k} cell types; median {X.A.median():.3f}, middle 95 % {np.percentile(X.A, 2.5):.3f}-{np.percentile(X.A, 97.5):.3f}\n"
     f"log standard deviation: total {tot:.4f}; between cell types {between:.4f} ({100 * between ** 2 / tot ** 2:.0f} % of the variance); within a cell type {within:.4f}\n"
     f"+/-2 sd: one baseline for all types {np.exp(-2 * tot):.3f}-{np.exp(2 * tot):.3f}; a baseline per type {np.exp(-2 * within):.3f}-{np.exp(2 * within):.3f}\n"
     "Caveats: 1-3 samples per type, one laboratory (cross-laboratory spread is not in 'within'); exploration, nothing sealed.\n")
open(os.path.join(H, "iama_spread_split_summary.txt"), "w").write(t); print(t)
