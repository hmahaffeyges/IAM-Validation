"""Part 4, Chapter 'Fish at river temperature' (part4/p4_22b_salmonid.tex).
fig_fish_library: per-fish copy error against the library terms, for the four salmonid sets read on single molecules.
(a) Methow steelhead RBC and sperm, (b) coho smolts, (c) brook charr sperm, (d) Rimouski Atlantic salmon fin (F0, F1):
copy error against bisulfite conversion failure. (e) brook charr: copy error against duplicate fraction.
(f) holding energy E = ln((1-eps)/eps) per fish, with the healthy human range on the same statistic and the range a
fixed holding energy in joules would give at 10 C.
Every number is read from the per-fish tables in the repository or from the pre-registration text; nothing is typed in.
Copy error: eps_corr (sequencing error subtracted) for Methow, charr, Rimouski; coho as scored (eps_cc_common, no
subtraction), with the subtracted value also printed. Spearman rho printed and drawn per panel."""
import sys, re, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import _bookstyle as S
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

BP = S.REPO / "Biological_Physics"
SAL = pd.read_csv(BP / "Salmonid/PROC_SALMON_01/salmon_readings.csv")
COHO = pd.read_csv(BP / "Salmonid/DEV_COHO_CC_01/coho_cc_fish.csv")
CHR = pd.read_csv(BP / "MethylPhys/doors/data/charr_readings.csv")
RIM = pd.read_csv(BP / "MethylPhys/doors/data/rimouski_readings.csv")
PRE = (BP / "Salmonid/PROC_SALMON_01/PROC_SALMON_01_PREREG.md").read_text()

# human healthy range on the same statistic and the two temperatures, as stated in the pre-registration
m = re.search(r"E = ([\d.]+)[–-]([\d.]+) kT at ([\d.]+) K → predicted at ([\d.]+) K", PRE)
H_LO, H_HI, T_H, T_F = (float(x) for x in m.groups())
J_LO, J_HI = H_LO * T_H / T_F, H_HI * T_H / T_F          # fixed holding energy in joules, read at T_F

# brook charr: the failed download (84 qualifying molecules) is left out of every panel and of rho
CHR["failed"] = CHR.qualifying < 1000
CH = CHR[~CHR.failed]
COHO["eps"] = COHO.eps_cc_common
COHO["eps_sub"] = COHO.eps_cc_common - COHO.sub_err


def rho(x, y):
    r = spearmanr(x, y)
    return float(r.correlation), float(r.pvalue), len(x)


def sg(v):
    return f"{v:+.2f}".replace("-", "\u2212")


def lab(r):
    return f"ρ = {sg(r[0])} (n = {r[2]})"


S.apply()
fig, ax = plt.subplots(2, 3, figsize=(S.TEXTW, 4.7), gridspec_kw=dict(wspace=0.45, hspace=0.75))
ax = ax.ravel()
pct = mt.FuncFormatter(lambda v, _: f"{100*v:g}")
res = {}

# (a) Methow steelhead
a = ax[0]
for t, col, mk, name in (("RBC", S.DATA, "o", "red cells"), ("Sp", S.IAM, "s", "sperm")):
    g = SAL[SAL.tissue == t]
    a.plot(g.conv_fail, g.eps_corr, mk, color=col, ms=3.2, mew=0, alpha=0.9, label=name)
    res[f"Methow {t}"] = rho(g.eps_corr, g.conv_fail)
a.set_title("Methow steelhead, RRBS\n" + f"ρ = {sg(res['Methow RBC'][0])} red cells, {sg(res['Methow Sp'][0])} sperm", fontsize=7)
a.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)
a.set_ylim(0.0, 0.050)

# (b) coho smolts
a = ax[1]
for o, col, mk in (("hatchery", S.GR, "o"), ("wild", S.ALT, "^")):
    g = COHO[COHO.origin == o]
    a.plot(g.conv_fail, g.eps, mk, color=col, ms=3.2, mew=0, alpha=0.9, label=o)
res["coho"] = rho(COHO.eps, COHO.conv_fail)
res["coho, sequencing error subtracted"] = rho(COHO.eps_sub, COHO.conv_fail)
a.set_title("coho smolts, RRBS\n" + lab(res["coho"]), fontsize=7)
a.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)

# (c) brook charr: conversion failure
a = ax[2]
a.plot(CH.conv_fail, CH.eps_corr, "o", color=S.GOLD, ms=3.2, mew=0, alpha=0.9)
res["charr conversion"] = rho(CH.eps_corr, CH.conv_fail)
res["charr conversion, all 40"] = rho(CHR.eps_corr, CHR.conv_fail)
a.set_title("brook charr sperm, WGBS\n" + lab(res["charr conversion"]), fontsize=7)

# (d) Rimouski Atlantic salmon fin
a = ax[3]
for gen, col, mk in (("F0", S.ALT2, "o"), ("F1", S.SKY, "D")):
    g = RIM[RIM.generation == gen]
    a.plot(g.conv_fail, g.eps_corr, mk, color=col, ms=3.0, mew=0, alpha=0.9, label=gen + (" adults" if gen == "F0" else " offspring"))
    res[f"Rimouski {gen}"] = rho(g.eps_corr, g.conv_fail)
res["Rimouski all"] = rho(RIM.eps_corr, RIM.conv_fail)
a.set_xscale("log")
a.set_title("Atlantic salmon fin, WGBS\n" + f"ρ = {sg(res['Rimouski all'][0])} all; F0 {sg(res['Rimouski F0'][0])}, F1 {sg(res['Rimouski F1'][0])}", fontsize=7)
a.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)

for i in range(4):
    ax[i].set_xlabel("conversion failure (%)")
    ax[i].set_ylabel("copy error ε")
    ax[i].xaxis.set_major_formatter(pct)
ax[3].xaxis.set_minor_formatter(mt.NullFormatter())
ax[3].set_xticks([0.002, 0.005, 0.01, 0.02, 0.03])

# (e) brook charr: duplicate fraction
a = ax[4]
a.plot(CH.dup_frac, CH.eps_corr, "o", color=S.GOLD, ms=3.2, mew=0, alpha=0.9)
res["charr duplicates"] = rho(CH.eps_corr, CH.dup_frac)
res["charr duplicates, all 40"] = rho(CHR.eps_corr, CHR.dup_frac)
a.set_title("brook charr sperm, WGBS\n" + lab(res["charr duplicates"]), fontsize=7)
a.set_xlabel("duplicate fraction (%)"); a.set_ylabel("copy error ε"); a.xaxis.set_major_formatter(pct)

# (f) holding energy per fish
a = ax[5]
sets = [("steelhead red cells", SAL[SAL.tissue == "RBC"].E_kT, S.DATA),
        ("coho smolts", COHO.E_kT, S.GR),
        ("salmon fin F0", RIM[RIM.generation == "F0"].E_kT, S.ALT2),
        ("charr sperm", CH.E_kT, S.GOLD),
        ("steelhead sperm", SAL[SAL.tissue == "Sp"].E_kT, S.IAM)]
rng = np.random.default_rng(1)
a.axhspan(H_LO, H_HI, color=S.LIGHT, alpha=0.5, lw=0)
a.axhspan(J_LO, J_HI, facecolor="none", edgecolor=S.GR, hatch="////", lw=0.0, alpha=0.6)
for i, (name, e, col) in enumerate(sets):
    x = i + rng.uniform(-0.18, 0.18, len(e))
    a.plot(x, e, "o", color=col, ms=2.2, mew=0, alpha=0.8)
    a.plot([i - 0.28, i + 0.28], [np.median(e)] * 2, color="black", lw=1.0)
a.set_xticks(range(len(sets))); a.set_xticklabels([s[0] for s in sets], fontsize=5.5, rotation=35, ha="right", rotation_mode="anchor")
a.set_xlim(-0.6, len(sets) + 0.9)
a.set_ylabel(r"holding energy ($k_BT$)")
a.set_title("holding energy per fish\nmedian bar; bands at right", fontsize=7)
a.set_ylim(3.15, 4.3)
a.text(len(sets) + 0.85, (H_LO + H_HI) / 2, "human\n37 °C", fontsize=5.5, ha="right", va="center", color=S.GR)
a.text(len(sets) + 0.85, (J_LO + J_HI) / 2, "fixed\nenergy\n10 °C", fontsize=5.5, ha="right", va="center", color=S.GR,
       bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

for i, L in enumerate("abcdef"):
    S.panel_letter(ax[i], L, dx=-0.24, dy=1.16)

out = S.save(fig, "part4", "fig_fish_library")
print("human range", H_LO, H_HI, "at", T_H, "K; fixed-joule range at", T_F, "K:", round(J_LO, 3), round(J_HI, 3))
for k, v in res.items():
    print(f"{k:40s} rho {v[0]:+.3f}  p {v[1]:.4f}  n {v[2]}")
for name, e, _ in sets:
    print(f"{name.replace(chr(10), ' '):20s} median E {np.median(e):.3f}  range {e.min():.3f}-{e.max():.3f}  n {len(e)}")
e = COHO.eps_sub
print("coho sequencing-error subtracted: median eps %.4f, median E %.3f" % (e.median(), np.log((1 - e.median()) / e.median())))
