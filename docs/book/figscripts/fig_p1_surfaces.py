"""Part 1, Chapter 'The places in IAM' (p1_01_encoding_surfaces.tex), Table tab:surfaces.
fig_encoding_ladder: the encoding surfaces by characteristic size and cost per bit k_B T ln2.
fig_landauer_price: (a) the price per bit against surface temperature; (b) bits held against the holographic bound A/(4 l_P^2 ln2).
Temperatures and bit counts are computed exactly as in docs/verification/scripts/verify_encoding_ladder.py (CODATA 2018, H0 = 67.36).
Sizes of the cell, transistor and qubit surfaces are representative scales (stated below), not measurements; horizon sizes are computed."""
import numpy as np, scipy.constants as C
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _bookstyle as S
import matplotlib.pyplot as plt

hbar, c, G, k = C.hbar, C.c, C.G, C.k
Msun, Mpc, ln2 = 1.98847e30, 3.0857e22, np.log(2)
lP = np.sqrt(hbar * G / c**3)
a0 = C.physical_constants["Bohr radius"][0]

def bh(m):
    M = m * Msun; T = hbar * c**3 / (8 * np.pi * G * M * k); rs = 2 * G * M / c**2
    A = 4 * np.pi * rs**2; return T, A / (4 * lP**2) / ln2, rs, A

H0 = 67.36e3 / Mpc; T_gh = hbar * H0 / (2 * np.pi * k); R_h = c / H0; A_h = 4 * np.pi * R_h**2; N_h = A_h / (4 * lP**2) / ln2
T1, N1, rs1, A1 = bh(1.0)
Ts, Ns, rss, As = bh(4.3e6)
N_cpg = 28217448                                   # hg19 CpG index (verify_encoding_ladder.py)
# representative sizes (m): nucleus diameter 6 um; transistor gate length 20 nm; Josephson junction 200 nm
rows = [  # label, T, size, N or None, representative-size flag
    ("cell (CpG record)", 310.15, 6e-6, N_cpg, True),
    ("transistor, 300 K", 300.0, 2e-8, None, True),
    ("qubit, 15 mK", 0.015, 2e-7, None, True),
    ("black hole, 1 M$_\\odot$", T1, rs1, N1, False),
    ("black hole, Sgr A$^*$", Ts, rss, Ns, False),
    ("cosmic horizon", T_gh, R_h, N_h, False),
]
cost = lambda T: k * T * ln2
for r in rows:
    print(f"{r[0]:28s} T {r[1]:.3e} K  size {r[2]:.3e} m  cost {cost(r[1]):.3e} J" + (f"  N {r[3]:.3e}" if r[3] else ""))
print("cost span cell/cosmic:", f"{cost(310.15)/cost(T_gh):.2e}", " size span atom->Hubble radius (decades):", round(np.log10(R_h / a0), 1))

S.apply()
col = {True: S.GR, False: S.IAM}
# ---------------- Figure 1: the ladder -----------------
fig, ax = plt.subplots(figsize=(S.TEXTW, 3.3))
offs = {"cell (CpG record)": (8, 8), "transistor, 300 K": (14, -12), "qubit, 15 mK": (8, -4),
        "black hole, 1 M$_\\odot$": (8, 4), "black hole, Sgr A$^*$": (8, 4), "cosmic horizon": (-10, 8)}
for lab, T, size, N, rep in rows:
    ax.plot(size, cost(T), "o", ms=6, mfc="white" if rep else S.IAM, mec=col[rep], mew=1.2, zorder=3)
    txt = lab + ("" if N is None else f"\n{S.sci(N, 2)} bits")
    dx, dy = offs[lab]
    ax.annotate(txt, (size, cost(T)), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                ha="right" if dx < 0 else ("center" if dx == 0 else "left"), va="center")
ax.axvline(a0, color=S.LIGHT, lw=0.8, ls=":")
ax.text(a0 * 1.6, 3e-55, "atom (Bohr radius)", fontsize=7, color=S.GR, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-11, 1e28); ax.set_ylim(1e-55, 1e-18)
ax.set_xticks([1e-10, 1e-5, 1, 1e5, 1e10, 1e15, 1e20, 1e25])
ax.set_yticks([1e-50, 1e-40, 1e-30, 1e-20])
ax.set_xlabel("characteristic size of the encoding surface (m)")
ax.set_ylabel("cost per bit, $k_BT\\ln2$ (J)")
ax.set_title("The price of a bit falls 32 orders of magnitude from the cell to the cosmic horizon")
ax.plot([], [], "o", mfc="white", mec=S.GR, label="size representative (cost exact)")
ax.plot([], [], "o", mfc=S.IAM, mec=S.IAM, label="size and cost computed")
ax.legend(loc="upper right", fontsize=7)
S.save(fig, "part1", "fig_encoding_ladder")

# ---------------- Figure 2: price vs temperature; bits vs holographic bound -----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.9), gridspec_kw=dict(wspace=0.38))
TT = np.logspace(-31, 3, 200)
a1.plot(TT, cost(TT), color=S.LIGHT, lw=1.0, zorder=1)
for lab, T, size, N, rep in rows:
    a1.plot(T, cost(T), "o", ms=5, color=S.IAM, zorder=3)
lab_off = {"cell (CpG record)": (-7, 6, "right"), "transistor, 300 K": (-7, -6, "right"), "qubit, 15 mK": (6, -6, "left"),
           "black hole, 1 M$_\\odot$": (6, -6, "left"), "black hole, Sgr A$^*$": (6, -6, "left"), "cosmic horizon": (6, -6, "left")}
for lab, T, size, N, rep in rows:
    if lab == "transistor, 300 K":
        continue
    if lab == "cell (CpG record)":
        lab = "cell 310 K, transistor 300 K"
        dx, dy, ha = lab_off["cell (CpG record)"]
        a1.annotate(lab, (T, cost(T)), xytext=(dx, dy), textcoords="offset points", fontsize=7, ha=ha, va="center")
        continue
    dx, dy, ha = lab_off[lab]
    a1.annotate(lab, (T, cost(T)), xytext=(dx, dy), textcoords="offset points", fontsize=6.5 if False else 7, ha=ha, va="center")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(1e-32, 1e5); a1.set_ylim(1e-55, 1e-18)
a1.set_xticks([1e-30, 1e-20, 1e-10, 1]); a1.set_yticks([1e-50, 1e-40, 1e-30, 1e-20])
a1.set_xlabel("surface temperature $T$ (K)"); a1.set_ylabel("cost per bit, $k_BT\\ln2$ (J)")
a1.set_title("The price depends only on temperature")
S.panel_letter(a1, "a", dx=-0.2)
AA = np.logspace(-12, 55, 200)
a2.plot(AA, AA / (4 * lP**2) / ln2, color=S.LIGHT, lw=1.0, zorder=1)
a2.text(1e20, 1e20 / (4 * lP**2) / ln2 * 30, "bound $A/(4\\ell_P^2\\ln 2)$", rotation=0, fontsize=7, color=S.GR, ha="right", va="bottom")
for lab, A, N in (("black hole, 1 M$_\\odot$", A1, N1), ("Sgr A$^*$", As, Ns), ("cosmic horizon", A_h, N_h)):
    a2.plot(A, N, "o", ms=5, color=S.IAM, zorder=3)
    a2.annotate(lab, (A, N), xytext=(6, -7), textcoords="offset points", fontsize=7)
A_nuc = 4 * np.pi * (3e-6)**2
a2.plot(A_nuc, N_cpg, "o", ms=5, mfc="white", mec=S.GR, mew=1.1, zorder=3)
a2.annotate("cell: $2.8\\times10^{7}$ CpG sites\non a 6 µm nucleus", (A_nuc, N_cpg), xytext=(7, 0), textcoords="offset points", fontsize=7, va="center")
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1e-13, 1e57); a2.set_ylim(1, 1e126)
a2.set_xticks([1e-10, 1e10, 1e30, 1e50]); a2.set_yticks([1, 1e40, 1e80, 1e120])
a2.set_xlabel("surface area $A$ (m$^2$)"); a2.set_ylabel("bits held $N$")
a2.set_title("Horizons hold the most a surface can")
S.panel_letter(a2, "b", dx=-0.2)
print("bound at the nucleus area:", f"{A_nuc/(4*lP**2)/ln2:.2e}", "bits")
S.save(fig, "part1", "fig_landauer_price")
