"""PHY-049 open sea (2026-10-15): does counting the own-read loose share the way the arrays do (open-sea CpGs only) tighten the line?
One script builds BOTH own-read versions from the same samples with the same steps, so the only difference is the loose-share rule:
 every CpG  = own_reads_genome.csv (+ run2 for the ten GEO refused once), loose_0.7000
 open sea   = own_reads_opensea.csv (own_reads_opensea.py), loose_0.7000
Per cell: median loose share over its samples; held error weighted by opportunities. Held error is scale-matched to PROC-CHANNEL-01 by the
constant of the committed own_reads_genome_cells.csv (gerr_s / gerr = 0.74128, identical for all 56 cells). Line: Theil-Sen, as
loose_end_line.py. GATE: the held error is counted unchanged in the open-sea run, so per sample it must equal the every-CpG run's (relative
difference < 1e-9) on every sample both read; otherwise the comparison stops. Writes opensea_line.json and opensea_line_cells.csv."""
import os, json, numpy as np, pandas as pd
from scipy.stats import theilslopes, linregress
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "..", "Biological_Physics", "MethylPhys", "doors", "data")
SCALE = 0.7412824486
G = pd.read_csv(os.path.join(H, "own_reads_genome.csv")); G2 = pd.read_csv(os.path.join(H, "run2", "own_reads_genome_run2.csv"))
cell = dict(zip(G["sample"], G["cell"])); G2["cell"] = G2["sample"].map(cell)
EV = pd.concat([G, G2]).dropna(subset=["held_err"]).drop_duplicates("sample", keep="last")
OS = pd.read_csv(os.path.join(H, "own_reads_opensea.csv")); OS = OS[OS["loose_0.7000"].notna()].copy(); OS["cell"] = OS["sample"].map(cell)
both = EV.merge(OS, on="sample", suffixes=("_ev", "_os"))
gate = float((both.held_err_os / both.held_err_ev - 1).abs().max())
assert gate < 1e-9, f"held error differs between runs by {gate}: the open-sea run did not count the same molecules"
EV = EV[EV["sample"].isin(both["sample"])]; OS = OS[OS["sample"].isin(both["sample"])]
A = pd.read_csv(os.path.join(R, "DEV_ARCH_01", "arch_vs_held.csv")).rename(columns={"Unnamed: 0": "cell"})
def per_cell(D):
    g = D.groupby("cell")
    return pd.DataFrame({"L": g["loose_0.7000"].median(), "eps": g.apply(lambda x: (x.held_err * x.held_opp).sum() / x.held_opp.sum(), include_groups=False) * SCALE, "n": g.size()})
def line(x, y):
    s, i, _, _ = theilslopes(y, x); res = y - (i + s * x)
    return dict(cells=len(x), firm=round(float(i), 4), loose=round(float(i + s), 4), r=round(float(linregress(x, y).rvalue), 3),
                within_0_003=int((abs(res) <= 0.003).sum()), residual_abs_median=round(float(np.median(abs(res))), 4))
E, O = per_cell(EV), per_cell(OS)
M = A.set_index("cell")[["loose_share", "held_err", "klass"]].join(E.add_suffix("_ev"), how="inner").join(O.add_suffix("_os"), how="inner")
out = {"samples_both": int(len(both)), "held_error_gate_max_rel_diff": gate,
       "arrays": line(M.loose_share.values, M.held_err.values),
       "own_reads_every_CpG": line(M.L_ev.values, M.eps_ev.values),
       "own_reads_open_sea": line(M.L_os.values, M.eps_os.values),
       "loose_share_vs_arrays_r": {"every_CpG": round(float(linregress(M.loose_share, M.L_ev).rvalue), 3), "open_sea": round(float(linregress(M.loose_share, M.L_os).rvalue), 3)},
       "loose_share_median": {"arrays": round(float(M.loose_share.median()), 4), "every_CpG": round(float(M.L_ev.median()), 4), "open_sea": round(float(M.L_os.median()), 4)}}
M.round(6).to_csv(os.path.join(H, "opensea_line_cells.csv")); json.dump(out, open(os.path.join(H, "opensea_line.json"), "w"), indent=1); print(json.dumps(out, indent=1))
