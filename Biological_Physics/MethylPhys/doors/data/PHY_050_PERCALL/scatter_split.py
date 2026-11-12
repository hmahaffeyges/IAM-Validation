"""The healthy held error's scatter, split three ways (PHY-079; exploration on spent counts; nothing sealed).
Each of the 153 Loyfer samples' length-corrected lone-loss rate (held_err_length, DEV_CHAIN_Q2_01/percall_out; neurons set aside) over
the M-based prediction 1/(1+M) + 0.0284 x loose share, on a log scale. Meter: the sample-to-sample scatter of the counter synthetic's
paired replicates at the real clustering (DEV_CHAIN_Q2_01/counter_synthetic.csv) plus each sample's counting noise. Within a cell type,
across samples: meter plus the samples' own differences. Between cell types: the spread of cell-type means beyond what the within-type
scatter alone would give. Writes scatter_split.csv and scatter_split_summary.txt."""
import os, json, glob, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(H, "..", "DEV_CHAIN_Q2_01")
M = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
rows = []
for f in sorted(glob.glob(os.path.join(Q, "percall_out", "*.json"))):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("held_err_length") is not None:
            rows.append(dict(gsm=x["sample"], cell=x["cell"], obs=x["held_err_length"], loose=x["loose_share"], opp=x.get("held_opp")))
P = pd.DataFrame(rows).drop_duplicates("gsm"); P = P[~P.cell.str.contains("euron")].copy()
P["ratio"] = P.obs / (1 / (1 + M) + 0.0284 * P.loose); P["lr"] = np.log(P.ratio)
S = pd.read_csv(os.path.join(Q, "counter_synthetic.csv")); S = S[S.case == "clustered (next loss x1.5)"]
pr = S.groupby("loose_planted").length.agg(lambda v: np.log(v.iloc[0] / v.iloc[1]) if len(v) == 2 else np.nan).dropna()
meter = float(np.sqrt((pr ** 2).mean() / 2)); count = float(np.median(np.sqrt((1 - P.obs) / (P.obs * P.opp.clip(lower=1)))))
mul = P.groupby("cell").filter(lambda x: len(x) >= 2); k = mul.cell.nunique(); nb = len(mul) / k
within = float(np.sqrt(mul.groupby("cell").lr.var(ddof=1).mean())); between = float(mul.groupby("cell").lr.mean().std(ddof=1))
between_true = float(np.sqrt(max(between ** 2 - within ** 2 / nb, 0))); samples = float(np.sqrt(max(within ** 2 - meter ** 2 - count ** 2, 0)))
T = mul.groupby("cell").ratio.agg(["count", "median", "min", "max"]).round(3).sort_values("median"); T.to_csv(os.path.join(H, "scatter_split.csv"))
t = (f"Healthy held-error scatter, split (exploration). {len(P)} samples, {P.cell.nunique()} cell types ({k} with 2 or more samples). "
     f"Standard deviations on a log scale (about percent):\n  all samples {100*P.lr.std(ddof=1):.1f} %\n  meter: synthetic replicates {100*meter:.1f} %, counting {100*count:.2f} %\n"
     f"  within a cell type, beyond the meter: {100*samples:.1f} %\n  between cell types (beyond within-type scatter): {100*between_true:.1f} %\n\n"
     "Cell types, observed over the M-based prediction:\n" + T.to_string() + "\n")
open(os.path.join(H, "scatter_split_summary.txt"), "w").write(t); print(t[:700])
