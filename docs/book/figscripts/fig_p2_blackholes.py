"""Part 2, Chapter 'Black-hole horizons' (p2_01_blackholes.tex).
fig_smarr: (a) the bits of a horizon priced at its own temperature, N k_B T ln2, against M c^2: Schwarzschild horizons give M c^2/2
(Smarr), the cosmic horizon gives its enclosed mass-energy M_H c^2; (b) Kerr: T S/(M c^2) = sqrt(1 - chi^2)/2 and the spin share 2 Omega_H J.
fig_bh_temperature: (a) Hawking temperature against mass with T_CMB and T_GH; (b) black-body evaporation time.
Constants CODATA 2018 (scipy); H0 = 67.4 for M_eq as in the chapter."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt

hbar, c, G, k = C.hbar, C.c, C.G, C.k
Msun, Mpc, yr, ln2 = 1.98847e30, 3.0857e22, 3.15576e7, np.log(2)
lP = np.sqrt(hbar * G / c**3)
T_bh = lambda M: hbar * c**3 / (8 * np.pi * G * M * k)
S_bh = lambda M: k * 4 * np.pi * G * M**2 / (hbar * c)

S.apply()
# ---------------- Smarr -----------------
m = np.logspace(0, 11, 23)
M = m * Msun
tot = S_bh(M) / (k * ln2) * k * T_bh(M) * ln2
print("BH: max |N k T ln2/(Mc^2) - 1/2| =", np.max(np.abs(tot / (M * c**2) - 0.5)))
cos = []
for H0 in (67.4, 100.0, 1000.0, 1e4):
    H = H0 * 1e3 / Mpc; R = c / H; T = hbar * H / (2 * np.pi * k); N = 4 * np.pi * R**2 / (4 * lP**2) / ln2
    MH = 3 * H**2 / (8 * np.pi * G) * 4 / 3 * np.pi * R**3
    cos.append((MH * c**2, N * k * T * ln2))
print("cosmic: N k T ln2/(M_H c^2) =", [round(b / a, 10) for a, b in cos])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.8), gridspec_kw=dict(wspace=0.35))
a1.axhline(0.5, color=S.IAM, lw=0.8, ls="--"); a1.axhline(1.0, color=S.ALT, lw=0.8, ls=":")
a1.plot(M * c**2, tot / (M * c**2), "o", color=S.IAM, ms=3.5)
a1.plot([p[0] for p in cos], [p[1] / p[0] for p in cos], "s", color=S.ALT, ms=4)
a1.text(1e57, 0.44, "black holes, 1 to $10^{11}$ M$_\\odot$", fontsize=7, ha="center", va="top", color=S.IAM)
a1.text(1e47, 0.56, "Schwarzschild horizons: $\\frac{1}{2}$ (Smarr)", color=S.IAM, fontsize=7)
a1.text(1e71, 1.06, "cosmic horizon, four values of $H$: 1", color=S.ALT, fontsize=7, ha="right")
a1.set_xscale("log"); a1.set_xlim(1e46, 1e72); a1.set_ylim(0, 1.25)
a1.set_xticks([1e50, 1e60, 1e70])
a1.set_xlabel("rest energy $Mc^2$ (J)"); a1.set_ylabel("bits $\\times\\,k_BT\\ln 2$ / $Mc^2$")
a1.set_title("A horizon's bits cost half its rest energy")
S.panel_letter(a1, "a", dx=-0.18)
chi = np.linspace(0, 0.9999, 400)
a2.plot(chi, np.sqrt(1 - chi**2) / 2, color=S.IAM, lw=1.5)
rp = 1 + np.sqrt(1 - chi**2); spin = 2 * (chi / (rp**2 + chi**2)) * chi
a2.plot(chi, spin, color=S.GOLD, lw=1.2, ls="--")
for x in (0.5, 0.9, 0.998):
    y = np.sqrt(1 - x**2) / 2; a2.plot(x, y, "o", color=S.IAM, ms=4)
    a2.annotate(f"{y:.3f}", (x, y), xytext=(-6, 2), textcoords="offset points", fontsize=7, ha="right", va="bottom")
a2.plot(0, 0.5, "o", color=S.IAM, ms=4)
a2.text(0.02, 0.56, "$T_HS/Mc^2$ (horizon bits)", color=S.IAM, fontsize=7)
a2.text(0.42, 0.75, "$2\\Omega_HJ/Mc^2$ (spin)", color=S.GOLD, fontsize=7)
a2.set_xlim(-0.02, 1.02); a2.set_ylim(0, 1.0)
a2.set_xlabel("spin $\\chi = Jc/GM^2$"); a2.set_ylabel("share of $Mc^2$")
a2.set_title("Rotation moves the half into spin")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_smarr")

# ---------------- temperature and evaporation -----------------
T_cmb = 2.7255; H0 = 67.16e3 / Mpc; T_gh = hbar * H0 / (2 * np.pi * k)   # photon-sector H0, as the caption
M_cmb = hbar * c**3 / (8 * np.pi * G * k * T_cmb); M_eq = c**3 / (4 * G * H0)
tau = lambda M: 5120 * np.pi * G**2 * M**3 / (hbar * c**4) / yr
print(f"T_GH {T_gh:.3e} K; M_CMB {M_cmb:.2e} kg = {M_cmb/Msun:.2e} Msun; M_eq {M_eq/Msun:.3e} Msun; tau(1 Msun) {tau(Msun):.3e} yr; T(1 Msun) {T_bh(Msun):.3e} K")
mm = np.logspace(-20, 24, 300); MM = mm * Msun
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.8), gridspec_kw=dict(wspace=0.35))
for ax in (a1, a2):
    ax.axvspan(3, 6.6e10, color=S.SKY, alpha=0.25, lw=0)
a1.plot(mm, T_bh(MM), color=S.IAM, lw=1.5)
a1.axhline(T_cmb, color=S.DATA, lw=0.9, ls="--"); a1.axhline(T_gh, color=S.ALT, lw=0.9, ls="--")
a1.text(1e23, T_cmb * 4, "CMB, 2.7255 K", fontsize=7, color=S.DATA, ha="right")
a1.text(1e-19, T_gh * 6, f"cosmic horizon, ${T_gh/10**np.floor(np.log10(T_gh)):.1f}\\times10^{{{int(np.floor(np.log10(T_gh)))}}}$ K", fontsize=7, color=S.ALT)
a1.plot(M_cmb / Msun, T_cmb, "o", color=S.DATA, ms=4); a1.plot(M_eq / Msun, T_gh, "o", color=S.ALT, ms=4)
a1.annotate("balance with the CMB:\n$4.5\\times10^{22}$ kg", (M_cmb / Msun, T_cmb), xytext=(5, -5), textcoords="offset points", fontsize=7, va="top")
a1.annotate("$M_{\\rm eq}=2.3\\times10^{22}$ M$_\\odot$", (M_eq / Msun, T_gh), xytext=(-8, 26), textcoords="offset points", fontsize=7, ha="right", arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GR))
a1.text(4e5, 1e13, "known black holes", fontsize=7, color=S.IAM, ha="center", va="center")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(1e-20, 1e24); a1.set_ylim(1e-32, 1e16)
a1.set_xticks([1e-20, 1e-10, 1, 1e10, 1e20]); a1.set_yticks([1e-30, 1e-20, 1e-10, 1, 1e10])
a1.set_xlabel("mass ($M_\\odot$)"); a1.set_ylabel("Hawking temperature (K)")
a1.set_title("Black holes are hotter than the horizon")
S.panel_letter(a1, "a", dx=-0.18)
a2.plot(mm, tau(MM), color=S.IAM, lw=1.5)
a2.axhline(13.8e9, color=S.GR, lw=0.8, ls=":"); a2.text(1e3, 13.8e9 * 30, "age of the universe", fontsize=7, color=S.GR)
a2.plot(1, tau(Msun), "o", color=S.IAM, ms=4)
a2.annotate("1 M$_\\odot$: $2.1\\times10^{67}$ yr", (1, tau(Msun)), xytext=(6, -2), textcoords="offset points", fontsize=7, va="top")
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1e-20, 1e24); a2.set_ylim(1e5, 1e140)
a2.set_xticks([1e-20, 1e-10, 1, 1e10, 1e20]); a2.set_yticks([1e10, 1e50, 1e90, 1e130])
a2.set_xlabel("mass ($M_\\odot$)"); a2.set_ylabel("evaporation time (yr)")
a2.set_title("Evaporation time (black body)")
S.panel_letter(a2, "b", dx=-0.18)
S.save(fig, "part2", "fig_bh_temperature")
