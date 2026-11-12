"""Are the two strands of a held CpG lost independently? (PHY-083, the two-coin question; exploration on spent files; nothing sealed.)
GSE116482's nine SKO hairpin files (both strands of each molecule), reduced to strand-pair counts by level bin (MM, MU, UM, UU).
At the two most fully held bins: per-strand loss p = (MU + UM + 2 UU) / 2n; if the strands were lost independently, UU would be p^2 and
half-lost pairs 2p(1 - p). The ratio of observed to independent UU measures how far the pair behaves as one unit.
Confound named before writing: SKO cells have no de novo writers, so a pair lost on both strands stays lost and is copied; that alone
produces excess UU, whatever the price of a restore. Writes strand_pairs_summary.txt."""
import os, glob, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
LAB = {"15": (15, "C1"), "16": (30, "C1"), "17": (60, "C1"), "18": (15, "C2"), "19": (30, "C2"), "20": (60, "C2"), "21": (15, "C3"), "22": (30, "C3"), "23": (60, "C3")}
rows = []
for f in sorted(glob.glob(os.path.join(H, "GSM*.tsv"))):
    T = pd.read_csv(f, sep="\t", header=None, names=["bin", "sites", "MM", "MU", "UM", "UU"]).set_index("bin")
    for b in (8, 9):
        x = T.loc[b]; n = x.MM + x.MU + x.UM + x.UU; p = (x.MU + x.UM + 2 * x.UU) / (2 * n); d, c = LAB[os.path.basename(f)[-6:-4]]
        rows.append(dict(clone=c, day=d, bin=b, p_strand=round(p, 4), UU_obs=round(x.UU / n, 5), UU_if_independent=round(p * p, 5),
                         UU_ratio=round((x.UU / n) / (p * p), 2), half_lost_obs=round((x.MU + x.UM) / n, 4), half_lost_if_independent=round(2 * p * (1 - p), 4)))
G = {"GSM3669526": "WT rep1", "GSM3669527": "WT rep2", "GSM3669518": "SKO_Con rep1", "GSM3669519": "SKO_Con rep2", "GSM3669516": "PKO_Con rep1", "GSM3669517": "PKO_Con rep2"}
for g, lab in G.items():   # 2026-11-12: clones with their writers (genotypes/, the same single-pass binning as HAIRPIN_GENOTYPES)
    T = pd.read_csv(os.path.join(H, "genotypes", g + ".tsv"), sep="\t", header=None, names=["bin", "sites", "MM", "MU", "UM", "UU"]).set_index("bin")
    for b in (8, 9):
        x = T.loc[b]; n = x.MM + x.MU + x.UM + x.UU; p = (x.MU + x.UM + 2 * x.UU) / (2 * n)
        rows.append(dict(clone=lab, day=0, bin=b, p_strand=round(p, 4), UU_obs=round(x.UU / n, 5), UU_if_independent=round(p * p, 5),
                         UU_ratio=round((x.UU / n) / (p * p), 2), half_lost_obs=round((x.MU + x.UM) / n, 4), half_lost_if_independent=round(2 * p * (1 - p), 4)))
R = pd.DataFrame(rows)
t = ("Strand pairs at the most fully held bins (GSE116482 SKO hairpins; exploration)\n" + R.to_string(index=False) +
     f"\n\nmedian UU observed / UU if independent, SKO time course: bin 9 {R[(R.bin == 9) & (R.day > 0)].UU_ratio.median():.2f}, bin 8 {R[(R.bin == 8) & (R.day > 0)].UU_ratio.median():.2f}"
     f"\nwild type (writers present), bin 9: {R[(R.bin == 9) & R.clone.str.startswith('WT')].UU_ratio.min():.2f}-{R[(R.bin == 9) & R.clone.str.startswith('WT')].UU_ratio.max():.2f}\n")
open(os.path.join(H, "strand_pairs_summary.txt"), "w").write(t); print(t[-160:])
