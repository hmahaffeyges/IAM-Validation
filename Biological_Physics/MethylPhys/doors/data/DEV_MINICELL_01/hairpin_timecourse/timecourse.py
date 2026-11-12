"""Hairpin time course (PHY-034; exploration on spent files; nothing sealed). GSE116482's nine SKO files (three clones at days 15, 30, 60),
each reduced to strand-pair counts by level bin (bin, sites, MM, MU, UM, UU; the 11-11 single-pass binning that reproduced the original file).
Per file: the mark level, maintenance fidelity MM/(MM+MU+UM) over all bins and at the most fully held bin (9, the top level bin, as in HAIRPIN_GENOTYPES), and restores per loss f/(1-f).
Question: across days, does the restore/loss odds stay fixed while the level moves? Writes timecourse.csv and timecourse_summary.txt."""
import os, glob, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
LAB = {"15": (15, "C1"), "16": (30, "C1"), "17": (60, "C1"), "18": (15, "C2"), "19": (30, "C2"), "20": (60, "C2"), "21": (15, "C3"), "22": (30, "C3"), "23": (60, "C3")}
rows = []
for f in sorted(glob.glob(os.path.join(H, "GSM*.tsv"))):
    T = pd.read_csv(f, sep="\t", header=None, names=["bin", "sites", "MM", "MU", "UM", "UU"]).set_index("bin"); g = os.path.basename(f)[:-4]
    tot = T[["MM", "MU", "UM", "UU"]].sum(); n = tot.sum()
    level = (2 * tot.MM + tot.MU + tot.UM) / (2 * n)
    hi = T.loc[T.index.max(), ["MM", "MU", "UM"]]
    f_all = tot.MM / (tot.MM + tot.MU + tot.UM); f_hi = hi.MM / hi.sum()
    d, c = LAB[g[-2:]]; rows.append(dict(gsm=g, day=d, clone=c, level=round(level, 4), fidelity_all=round(f_all, 4), fidelity_most_held=round(f_hi, 4), restore_over_loss=round(f_hi / (1 - f_hi), 2)))
R = pd.DataFrame(rows).sort_values(["clone", "day"]); R.to_csv(os.path.join(H, "timecourse.csv"), index=False)
S = R.groupby("clone").agg(odds_min=("restore_over_loss", "min"), odds_max=("restore_over_loss", "max"), level_d15=("level", "first"), level_d60=("level", "last"))
S["odds_spread_pct"] = (100 * (S.odds_max - S.odds_min) / S.odds_min).round(1); S["level_change_pct"] = (100 * (S.level_d60 - S.level_d15) / S.level_d15).round(1)
t = "Hairpin time course (GSE116482 SKO clones; exploration)\n" + R.to_string(index=False) + "\n\nBy clone:\n" + S.to_string() + "\n"
open(os.path.join(H, "timecourse_summary.txt"), "w").write(t); print(t)
