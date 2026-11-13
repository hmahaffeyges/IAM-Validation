"""Each cell type's net delivery: effective coins per restore out of the 2-coin SAM price (PHY-083; exploration on spent counts; predictions
written before computing, in the record). Per type: median zero-loose held error (lone-loss reading minus 0.0284 x loose share, per sample;
DEV_CHAIN_Q2_01/percall_out, neurons out) and median measured clustering (coupling); the candidate's lone-loss reading at that clustering
is interpolated on the committed forward-model grid (DEV_CHAIN_Q2_01/forward_out, independent and follow-on runs); the per-site loss the
type's reading implies gives the work per restore in coins (M from the canon). Then Spearman against restore crew and proliferation RNA
(crew_vs_held.csv) and census lifespan (lifespan_vs_held.csv). Writes celltype_net.csv and celltype_net_summary.txt."""
import os, json, glob, math, numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(H, "..", "DEV_CHAIN_Q2_01")
M = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
E = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["eps0_candidate"]["value"])
rows = []
for f in glob.glob(os.path.join(Q, "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("coupling") is not None: rows.append(dict(cell=x["cell"], held=x["held_err_length"], loose=x["loose_share"], coupling=x["coupling"]))
P = pd.DataFrame(rows); P = P[~P.cell.str.contains("neuron")]; P["held0"] = P.held - 0.0284 * P.loose
T = P.groupby("cell").agg(n=("held0", "size"), held0=("held0", "median"), coupling=("coupling", "median")).reset_index()
F = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(Q, "forward_out", "*.csv"))]); F = F[F["mode"].isin(["ind", "seq"])].groupby(["mode", "param"]).mean(numeric_only=True).sort_values("coupling")
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
def coins(h, cp):
    e = E * h / np.interp(cp, F.coupling, F.held_err_length); return brentq(lambda w: odds(w) - (1 / e - 1), 1, 300) / M
T["coins"] = [coins(h, c) for h, c in zip(T.held0, T.coupling)]; T["net_pct"] = 100 * T.coins / 2
T["outside_clustering_grid"] = (T.coupling > F.coupling.max()) | (T.coupling < F.coupling.min())
T = T.sort_values("net_pct"); T.round(5).to_csv(os.path.join(H, "celltype_net.csv"), index=False); Ti = T.set_index("cell")
C = pd.read_csv(os.path.join(H, "crew_vs_held.csv")); X = []
for _, r in C.iterrows():
    cs = [c.strip() for c in str(r.cells).split(";") if c.strip() in Ti.index]
    if cs: X.append(dict(net=Ti.loc[cs, "net_pct"].median(), crew=r.crew_DNMT1_UHRF1, mki=r.MKI67))
X = pd.DataFrame(X); Ls = pd.read_csv(os.path.join(H, "lifespan_vs_held.csv")); Ls = Ls[Ls.cell.isin(Ti.index)]
L = [f"Cell-type net delivery (exploration): {len(T)} types; net {T.net_pct.min():.1f}-{T.net_pct.max():.1f} % of the 2-coin price (median {T.net_pct.median():.1f} %)",
     "above 100 % (more than the price buys): " + "; ".join(f"{c} {v:.1f} % (n {n})" for c, v, n in zip(T.cell, T.net_pct, T.n) if v > 100)]
for k, lab in (("crew", "restore crew (DNMT1 + UHRF1 RNA)"), ("mki", "proliferation (MKI67 RNA)")):
    s = spearmanr(X.net, X[k]); L.append(f"net vs {lab}: rho {s.statistic:+.2f}, p {s.pvalue:.3f}, n {len(X)}")
s = spearmanr(Ti.loc[Ls.cell, "net_pct"].values, Ls.lifespan_days.values); L.append(f"net vs census lifespan: rho {s.statistic:+.2f}, p {s.pvalue:.3f}, n {len(Ls)}")
open(os.path.join(H, "celltype_net_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
