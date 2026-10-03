"""Part 3 device-physics figures (G3, TODO 5.4). Every number is computed here from physical constants (CODATA via
scipy.constants) or from the published values named below; nothing is typed onto a curve by hand.

p3_01 fig_p3_pairbreaking: black-body photon number flux (one side, into a hemisphere) with h nu > 2 Delta, against the
      radiator temperature, for Al, Ta and Nb with BCS gaps Delta = 1.764 k_B T_c (T_c = 1.2, 4.47, 9.25 K). Marked: the
      15 mK mixing-chamber plate, the 2.725 K CMB, a 4 K stage, a 50 K stage and 300 K.
p3_05 fig_p3_coherence_optimum: model p(T1) = a/T1 + b/(T1_free - T1). The minimum sits at T1*/T1_free = r/(1+r),
      r = sqrt(a/b); the curves are drawn for the illustrative ratios a/b = 1, 3 and 10 and normalised to their own minimum.
p3_06 fig_p3_switch_floors: (a) minimum energy of a thermally reliable binary switch, E = k_B T ln(1/p), in Landauer
      units k_B T ln 2, against the error probability p per operation; (b) energy per switch per node under the 1974
      constant-field rules (C and V both scale as 1/kappa, E ~ kappa^-3) and at fixed voltage (E ~ kappa^-1), kappa = sqrt 2.
Run from any directory:  python docs/book/figscripts/fig_p3_devices_g3.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as C
from scipy import integrate
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()
h, kB, c, e = C.h, C.k, C.c, C.e
LN2 = np.log(2.0)


def photon_flux_above(nu0, T):
    """Photons s^-1 m^-2 emitted by a black body at T into a hemisphere with frequency above nu0."""
    x0 = h * nu0 / (kB * T)
    if x0 > 700:
        return 0.0
    I, _ = integrate.quad(lambda x: x * x / np.expm1(x), x0, x0 + 200.0, limit=200)
    return 2 * np.pi * (kB * T / h) ** 3 / c ** 2 * I


def power_above(nu0, T):
    x0 = h * nu0 / (kB * T)
    I, _ = integrate.quad(lambda x: x ** 3 / np.expm1(x), x0, x0 + 200.0, limit=200)
    return 2 * np.pi * (kB * T) ** 4 / (h ** 3 * c ** 2) * I


# ---------------- numbers quoted in the insertion blocks ----------------
metals = {"Al": 1.2, "Ta": 4.47, "Nb": 9.25}
thr = {m: 2 * 1.764 * kB * Tc / h for m, Tc in metals.items()}           # 2 Delta / h
nuAl = thr["Al"]
TCMB = 2.725
sigT4 = C.Stefan_Boltzmann * TCMB ** 4
nums = {
    "Delta_Al_ueV": 1.764 * kB * 1.2 / e * 1e6,
    "2Delta_over_h_GHz": {m: v / 1e9 for m, v in thr.items()},
    "CMB_sigmaT4_W_m2": sigT4,
    "CMB_P_above_Al_W_m2": power_above(nuAl, TCMB),
    "CMB_fraction_above_Al": power_above(nuAl, TCMB) / sigT4,
    "flux_above_Al": {T: photon_flux_above(nuAl, T) for T in (0.015, 1.0, TCMB, 4.0, 50.0, 300.0)},
    "pairs_per_100keV_upper": 100e3 * e / (2 * 1.764 * kB * 1.2),
    "E_over_kT_p1e-15": np.log(1e15), "Landauer_units_p1e-15": np.log(1e15) / LN2,
    "Dennard_constant_field_per_node": 1 - 2 ** -1.5, "Dennard_fixed_V_per_node": 1 - 2 ** -0.5,
}
for k, v in nums.items():
    print(k, v)

# ---------------- Figure 1: pair-breaking black-body flux ----------------
fig, ax = plt.subplots(figsize=(S.TEXTW * 0.62, 2.7))
T = np.logspace(np.log10(0.3), np.log10(400), 300)
cols = {"Al": S.IAM, "Ta": S.ALT, "Nb": S.GOLD}
for m, nu0 in thr.items():
    F = np.array([photon_flux_above(nu0, t) for t in T])
    ok = F > 1e0
    ax.plot(T[ok], F[ok], color=cols[m], lw=1.2, label=f"{m}, $2\\Delta/h$ = {nu0/1e9:.0f} GHz")
marks = [(TCMB, "CMB 2.725 K"), (4.0, "4 K stage"), (50.0, "50 K stage"), (300.0, "300 K")]
for t, lab in marks:
    ax.axvline(t, color=S.LIGHT, lw=0.6, ls="--", zorder=0)
    ax.text(t * 1.06, 1e22, lab, rotation=90, fontsize=6, color=S.GR, va="top")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.3, 400); ax.set_ylim(1e1, 1e24)
ax.legend(loc="lower right")
ax.set_xlabel("temperature of the radiating surface (K)")
ax.set_ylabel(r"photons above $2\Delta/h$ (s$^{-1}$ m$^{-2}$)")
ax.set_title(r"Pair-breaking photons from a surface at temperature $T$")
S.save(fig, "part3", "fig_p3_pairbreaking")

# ---------------- Figure 2: coherence-optimum model ----------------
fig, ax = plt.subplots(figsize=(S.TEXTW * 0.62, 2.6))
u = np.linspace(0.05, 0.97, 400)                                   # T1 / T1_free
for ab, col, ylab in [(1.0, S.GR, 1.32), (3.0, S.IAM, 1.18), (10.0, S.ALT, 1.32)]:
    p = ab / u + 1.0 / (1 - u)
    ustar = np.sqrt(ab) / (1 + np.sqrt(ab))
    pmin = ab / ustar + 1.0 / (1 - ustar)
    ax.plot(u, p / pmin, color=col, lw=1.2, label=f"$a/b$ = {ab:g}: optimum at {ustar:.2f}")
    ax.plot([ustar], [1.0], "o", color=col, ms=3.5)
ax.axhline(1.0, color=S.LIGHT, lw=0.6, ls=":")
ax.set_xlim(0, 1); ax.set_ylim(0.9, 2.2)
ax.legend(loc="upper center")
ax.set_xlabel(r"$T_1/T_{1,\rm free}$")
ax.set_ylabel("two-qubit error / its minimum")
ax.set_title(r"Optimum of $a/T_1+b/(T_{1,\rm free}-T_1)$ at $r/(1+r)$, $r=\sqrt{a/b}$")
S.save(fig, "part3", "fig_p3_coherence_optimum")

# ---------------- Figure 3: switching-energy floors and Dennard scaling ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5))
pp = np.logspace(-30, np.log10(0.5), 300)
a1.plot(pp, np.log(1 / pp) / LN2, color=S.IAM)
for pv in (0.5, 1e-15, 1e-25):
    yv = np.log(1 / pv) / LN2
    a1.plot([pv], [yv], "o", color=S.IAM, ms=3)
    a1.text(pv * (3 if pv < 0.1 else 0.02), yv + (6 if pv < 0.1 else 4), f"{yv:.1f}" + (" (Landauer)" if pv == 0.5 else ""), fontsize=6)
a1.set_xscale("log")
a1.set_xlim(1.0, 1e-30); a1.set_ylim(0, 105)
a1.set_xticks([1, 1e-5, 1e-10, 1e-15, 1e-20, 1e-25, 1e-30])
a1.minorticks_off()
a1.set_xlabel("error probability per operation, $p$")
a1.set_ylabel(r"$E_{\min}/k_BT\ln 2$")
a1.set_title(r"Reliable switch: $E_{\min}=k_BT\ln(1/p)$")
S.panel_letter(a1, "a")
nodes = np.arange(0, 9)
kap = np.sqrt(2.0)
a2.plot(nodes, kap ** (-3.0 * nodes), "o-", color=S.IAM, ms=3, label=r"constant field, $\kappa^{-3}$")
a2.plot(nodes, kap ** (-1.0 * nodes), "s-", color=S.GR, ms=3, label=r"fixed voltage, $\kappa^{-1}$")
a2.set_yscale("log")
a2.set_xlabel(r"node steps ($\kappa=\sqrt{2}$ each)")
a2.set_ylabel("energy per switch (relative)")
a2.set_title(f"{100*(1-kap**-3):.1f} % per node vs {100*(1-kap**-1):.1f} %")
a2.legend(loc="lower left")
S.panel_letter(a2, "b")
fig.tight_layout()
S.save(fig, "part3", "fig_p3_switch_floors")

