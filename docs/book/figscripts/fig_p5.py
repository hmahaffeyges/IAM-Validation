"""Part 5 figures. Every curve is computed from the chapter's own formula; the status of each (derived, calculated, conjecture,
observed) is stated in the caption that carries it.
p5_01 fig_meq: M_eq(z) = c^3/(4 G H(z)), where a black hole and the cosmic horizon share a temperature (H0 67.16, Om 0.3153, Or 9.1e-5; photon-sector Level 2 values),
against the largest known black hole (~7e10 Msun).
p5_02 fig_recession: recession speed H0 D/c against proper distance; the Hubble radius; the Pleiades (136 pc = 444 ly).
p5_03 fig_satellites: Milky Way satellites with a measured velocity dispersion or upper limit (Local Volume Database, Pace et al.;
docs/verification/observations/data/lvdb_dwarf_mw.csv) against the 4 km/s floor.
p5_04 fig_eraser: F = 1 - exp(-Q_L/Q), Q_L = k_B T ln2 at 300 K, with the chapter's detector examples (form assumed, not derived).
p5_05 fig_decoherence_profiles: (a) the exponential the rate integral gives against the assumed ramp 1 - E_q(t/tau)/e;
(b) tau_IAM against temperature at m = 1e-12 kg (silica, 2200 kg m^-3), with tau_DP.
p5_06 fig_bell_decay: pointer-basis dephasing with coherence c = 1 - D, D = E_q(t/tau)/e (assumed ramp): S_max = 2 sqrt(1 + c^2) (Horodecki 1995)
and, at the settings optimal for the undecohered pair, S = sqrt2 (1 + c); photons stay at 2 sqrt2."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
c, G, hbar, k = C.c, C.G, C.hbar, C.k
Msun, Mpc, ly = 1.98847e30, 3.0857e22, 9.4607e15
# ---------------- p5_01 -----------------
Om, Orad = 0.3153, 9.1e-5; OL = 1 - Om - Orad; H0 = 67.16e3 / Mpc
Hz = lambda z: H0 * np.sqrt(Om * (1 + z)**3 + Orad * (1 + z)**4 + OL)
Meq = lambda z: c**3 / (4 * G * Hz(z)) / Msun
for zz in (0, 1, 1e6):
    print(f"M_eq(z={zz:g}) = {Meq(zz):.2e} Msun")
z = np.logspace(-2, np.log10(2e6), 300)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
ax.plot(1 + z, Meq(z), color=S.ALT, lw=1.6)
ax.axhline(7e10, color=S.IAM, lw=0.9, ls="--"); ax.text(1.1, 7e10 * 3, "largest known black hole, ~$7\\times10^{10}$ M$_\\odot$", fontsize=7, color=S.IAM)
for zz in (0, 1, 1e6):
    ax.plot(1 + zz, Meq(zz), "o", color=S.ALT, ms=4)
    ax.annotate(("$z$ = $10^6$" if zz > 1e5 else f"$z$ = {zz:g}") + f": {S.sci(Meq(zz), 2)} M$_\\odot$", (1 + zz, Meq(zz)), xytext={0: (6, -12), 1: (6, 5)}.get(zz, (-6, 6)), textcoords="offset points", fontsize=7, ha="left" if zz < 1e5 else "right")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 3e6); ax.set_ylim(1e9, 1e24)
ax.set_xlabel("$1+z$"); ax.set_ylabel("$M_{\\rm eq}$, where $T_{\\rm BH}=T_{\\rm GH}$ (M$_\\odot$)")
ax.set_title("Known black holes are lighter than $M_{\\rm eq}$ back to $z=10^6$")
S.save(fig, "part7", "fig_meq")

# ---------------- p5_02 -----------------
D = np.logspace(1, 11, 300)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
for h, ls in ((70.0, "-"), (67.36, "--")):
    DH = c / (h * 1e3 / Mpc) / ly; ax.plot(D, D / DH, color=S.IAM, lw=1.4, ls=ls, label=f"$H_0$ = {h:g}: $D_H$ = {DH/1e10:.2f}$\\times10^{{10}}$ ly")
    print(f"H0 {h}: D_H = {DH:.3e} ly")
ax.axhline(1, color=S.GR, lw=0.8, ls=":"); ax.text(3e7, 1.5, "$v_{\\rm rec}=c$", fontsize=7, color=S.GR, ha="right")
DH70 = c / (70e3 / Mpc) / ly; Dp = 136 * 3.0857e16 / ly
print(f"Pleiades {Dp:.0f} ly = {Dp/DH70:.1e} of D_H; xi (7 days) = {Dp/(7/365.25):.2e}")
ax.plot(Dp, Dp / DH70, "o", color=S.DATA, ms=4)
ax.annotate(f"Pleiades, {Dp:.0f} ly: $3\\times10^{{-8}}$ of $D_H$", (Dp, Dp / DH70), xytext=(6, -2), textcoords="offset points", fontsize=7, va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(10, 1e11); ax.set_ylim(1e-10, 10)
ax.set_xlabel("proper distance $D$ (ly)"); ax.set_ylabel("recession speed $H_0D/c$")
ax.legend(loc="center right", fontsize=7)
ax.set_title("Recession exceeds $c$ beyond the Hubble radius")
S.save(fig, "part7", "fig_recession")

# ---------------- p5_03 -----------------
d = pd.read_csv(S.REPO / "docs/verification/observations/data/lvdb_dwarf_mw.csv")
kk = d[d.vlos_sigma.notna() | d.vlos_sigma_ul.notna()].copy()
meas = kk[kk.vlos_sigma.notna()]; ul = kk[kk.vlos_sigma.isna()]
below = int(((kk.vlos_sigma < 4) | (kk.vlos_sigma.isna() & (kk.vlos_sigma_ul < 4))).sum())
print(f"{len(d)} satellites, {len(kk)} with dispersion or upper limit; {below} below 4 km/s ({100*below/len(kk):.0f} %)")
fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 2.8))
ax.axhline(4, color=S.IAM, lw=1.0, ls="--")
ax.text(0.98, 4.4, "proposed floor, 4 km s$^{-1}$", fontsize=7, color=S.IAM, ha="right", transform=ax.get_yaxis_transform())
lo = meas[meas.vlos_sigma < 4]; hi = meas[meas.vlos_sigma >= 4]
ax.errorbar(hi.M_V, hi.vlos_sigma, yerr=[hi.vlos_sigma_em.fillna(0), hi.vlos_sigma_ep.fillna(0)], fmt="o", color=S.GR, ms=3.5, lw=0.6, capsize=0)
ax.errorbar(lo.M_V, lo.vlos_sigma, yerr=[lo.vlos_sigma_em.fillna(0), lo.vlos_sigma_ep.fillna(0)], fmt="o", color=S.DATA, ms=3.5, lw=0.6, capsize=0)
ulb = ul[ul.vlos_sigma_ul < 4]; ula = ul[ul.vlos_sigma_ul >= 4]
ax.errorbar(ulb.M_V, ulb.vlos_sigma_ul, yerr=0.25 * ulb.vlos_sigma_ul, uplims=True, fmt="v", color=S.DATA, ms=3.5, lw=0.6)
ax.errorbar(ula.M_V, ula.vlos_sigma_ul, yerr=0.25 * ula.vlos_sigma_ul, uplims=True, fmt="v", color=S.GR, ms=3.5, lw=0.6)
s2 = kk[kk.name == "Segue 2"].iloc[0]
ax.annotate("Segue 2 (< 2.06)", (s2.M_V, s2.vlos_sigma_ul), xytext=(8, -10), textcoords="offset points", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.4, color=S.GR))
ax.set_yscale("log"); ax.set_ylim(0.5, 40); ax.set_yticks([1, 2, 4, 10, 20]); ax.set_yticklabels(["1", "2", "4", "10", "20"])
ax.invert_xaxis(); ax.set_xlabel("absolute magnitude $M_V$ (brighter to the right)"); ax.set_ylabel("velocity dispersion (km s$^{-1}$)")
ax.set_title(f"{below} of {len(kk)} Milky Way satellites lie below the proposed floor")
S.save(fig, "part7", "fig_satellites")

# ---------------- p5_04 -----------------
QL = k * 300 * np.log(2) / C.e
r = np.logspace(-2, 3.3, 400); F = 1 - np.exp(-1 / r)
print(f"Q_L(300 K) = {QL:.4f} eV; CCD 3 eV -> Q/Q_L {3/QL:.0f}; rod 2.5 eV -> {2.5/QL:.0f}; F(1) = {1-np.exp(-1):.3f}")
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
ax.axvspan(1e-2, 0.04, color=S.ALT, alpha=0.15, lw=0); ax.axvspan(100, r[-1], color=S.DATA, alpha=0.12, lw=0)
ax.plot(r, F, color=S.IAM, lw=1.6)
ax.plot(1, 1 - np.exp(-1), "o", color=S.IAM, ms=4); ax.annotate("$Q=Q_L$: $F$ = 0.632", (1, 1 - np.exp(-1)), xytext=(6, 4), textcoords="offset points", fontsize=7)
for nm, Q in (("CCD pixel", 3.0), ("retinal rod", 2.5)):
    v = Q / QL; ax.plot(v, 1 - np.exp(-1 / v), "s", color=S.DATA, ms=4)
ax.annotate("rod (140), CCD (170)", (155, 0.006), xytext=(-4, 22), textcoords="offset points", fontsize=7, ha="right", arrowprops=dict(arrowstyle="-", lw=0.4, color=S.GR))
ax.text(0.012, 0.42, "reversible\nmarkers", fontsize=7, color=S.ALT)
ax.text(1900, 0.42, "irreversible\ndetectors", fontsize=7, color=S.DATA, ha="right")
ax.set_xscale("log"); ax.set_xlim(1e-2, r[-1]); ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("dissipated energy over the Landauer quantum, $Q/Q_L$"); ax.set_ylabel("erasure fidelity $F$")
ax.set_title("Proposed universal curve $F=1-e^{-Q_L/Q}$")
S.save(fig, "part4", "fig_eraser")

# ---------------- p5_05 -----------------
eta = np.linspace(1e-3, 5, 600); Eq = np.exp(1 - 1 / eta)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(eta, np.exp(-eta), color=S.IAM, lw=1.6); a1.plot(eta, 1 - Eq / np.e, color=S.ALT2, lw=1.4, ls="--")
a1.text(2.6, 0.11, "rate integral: $e^{-t/\\tau}$", fontsize=7, color=S.IAM, va="bottom")
a1.text(1.6, 1 - np.exp(1 - 1 / 1.6) / np.e + 0.04, "assumed ramp: $1-E_q(t/\\tau)/e$", fontsize=7, color=S.ALT2, va="bottom")
a1.set_xlim(0, 5); a1.set_ylim(0, 1.05)
a1.set_xlabel("$t/\\tau_{\\rm IAM}$"); a1.set_ylabel("remaining coherence $C(t)$")
a1.set_title("Exponential onset or sigmoidal ramp")
S.panel_letter(a1, "a", dx=-0.16)
rho, m = 2200.0, 1e-12; R = (3 * m / (4 * np.pi * rho))**(1 / 3); EG = G * m**2 / R
TT = np.logspace(-3, 0, 200)
tI = hbar * (k * TT)**2 * np.log(2) / EG**3; tD = hbar / EG
print(f"R {R*1e6:.2f} um, E_G {EG:.2e} J; tau_IAM(10 mK) {hbar*(k*0.01)**2*np.log(2)/EG**3:.0f} s; tau_DP {tD*1e6:.1f} us")
a2.plot(TT * 1e3, tI, color=S.IAM, lw=1.6); a2.axhline(tD, color=S.GR, lw=1.2)
for Tm in (10,):
    v = hbar * (k * Tm * 1e-3)**2 * np.log(2) / EG**3; a2.plot(Tm, v, "o", color=S.IAM, ms=4)
    a2.annotate(f"{v:.0f} s", (Tm, v), xytext=(-6, 4), textcoords="offset points", fontsize=7, ha="right")
a2.text(1.2, tD * 3, f"$\\tau_{{\\rm DP}}=\\hbar/E_G$ = {tD*1e6:.1f} µs, independent of $T$", fontsize=7, color=S.GR)
a2.text(100, 3e3, "$\\tau_{\\rm IAM}\\propto T^2$", fontsize=7, color=S.IAM)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1, 1000); a2.set_ylim(1e-7, 1e7)
a2.set_xticks([1, 10, 100, 1000]); a2.set_xticklabels(["1", "10", "100", "1000"])
a2.set_xlabel("temperature (mK)"); a2.set_ylabel("coherence time (s)")
a2.set_title("$10^{-12}$ kg silica sphere")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part4", "fig_decoherence_profiles")

# ---------------- p5_06 -----------------
cc = 1 - Eq / np.e; Smax = 2 * np.sqrt(1 + cc**2); Sfix = np.sqrt(2) * (1 + cc)
from scipy.optimize import brentq
tx = brentq(lambda t: np.sqrt(2) * (2 - np.exp(1 - 1 / t) / np.e) - 2, 0.05, 5)
print(f"fixed settings: |S| = 2 at c = {np.sqrt(2)-1:.3f} (D = {2-np.sqrt(2):.3f}), t/tau = {tx:.3f}; S_max >= 2 always")
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.5))
ax.axhline(2 * np.sqrt(2), color=S.ALT, lw=1.2); ax.text(4.9, 2 * np.sqrt(2) + 0.05, "photons: $2\\sqrt{2}$ at any distance", fontsize=7, color=S.ALT, ha="right", va="bottom")
ax.axhline(2, color=S.GR, lw=0.9, ls="--"); ax.text(0.1, 1.93, "local bound $|S|=2$", fontsize=7, color=S.GR, ha="left", va="top")
ax.plot(eta, Smax, color=S.IAM, lw=1.6); ax.text(2.6, 2.3, "matter, optimal settings: $2\\sqrt{1+c^2}\\geq2$", fontsize=7, color=S.IAM)
ax.plot(eta, Sfix, color=S.IAM, lw=1.4, ls="--"); ax.text(2.6, 1.45, "matter, fixed settings: $\\sqrt{2}\\,(1+c)$", fontsize=7, color=S.IAM)
ax.plot([tx], [2], "o", color=S.IAM, ms=4); ax.annotate(f"$t/\\tau$ = {tx:.2f}", (tx, 2), xytext=(10, -13), textcoords="offset points", fontsize=7)
ax.set_xlim(0, 5); ax.set_ylim(0, 3.2)
ax.set_xlabel("$t/\\tau_{\\rm IAM}$"); ax.set_ylabel("CHSH value $|S|$")
ax.set_title("Bell violation for massive pairs under the assumed ramp")
S.save(fig, "part4", "fig_bell_decay")
