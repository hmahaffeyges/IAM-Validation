"""Part V (label part:3) platform figures, proposal 2026-10-03.

fig_p3_platform_floors : (a) the Landauer cost k_B T ln 2 at the temperature each platform's record
                         actually sees; (b) thermal occupation of each platform's encoding gap against
                         the temperature of the bath it couples to.
fig_p3_chip_floor_Tj   : the chip gauge floor k_B T_j ln 2 at the chip's own junction temperature, and
                         where it sits on the gauge of one chip (E_sw = 1.92-1.97e-18 J, Chapter ch:cmos).
Every number is computed here from h, k_B and the stated gap and temperature. Nothing is fitted.
"""
import numpy as np
np.seterr(over="ignore")
import matplotlib.pyplot as plt
import _bookstyle as bs

bs.apply()
kB = 1.380649e-23
h = 6.62607015e-34
LN2 = np.log(2)


def kTln2(T):
    return kB * T * LN2


def p_two_level(f, T):
    """Equilibrium population of the upper level of a two-level record with gap h f at temperature T."""
    return 1.0 / (1.0 + np.exp(h * f / (kB * T)))


def n_bose(f, T):
    """Mean occupation of a bosonic mode (motion, radiation) at frequency f and temperature T."""
    return 1.0 / np.expm1(h * f / (kB * T))


# ---------------------------------------------------------------- figure 1
fig, (a, b) = plt.subplots(1, 2, figsize=(bs.TEXTW, 3.3))
T = np.logspace(-6, 3, 400)
a.loglog(T, kTln2(T), color=bs.GR, lw=1.0)
a.set_xlabel("temperature the record sees, T (K)")
a.set_ylabel(r"Landauer cost $k_BT\ln2$ (J per bit)")
pts = [  # (T, label, colour, marker)
    (10e-6, "atom motion in tweezers, 10 µK", bs.ALT, "s"),
    (0.47e-3, "ion motion at the Doppler limit (Yb$^+$)", bs.ALT2, "D"),
    (0.015, "transmon: mixing chamber, 15 mK", bs.IAM, "o"),
    (0.035, "transmon: the qubit itself, 35 mK", bs.IAM, "^"),
    (1.5, "hot silicon spin qubits, 1.5 K", bs.GOLD, "v"),
    (4.0, "cryogenic NV and atom arrays, 4 K", bs.DATA, "P"),
    (300.0, "room-temperature radiation and lattice", bs.DATA, "o"),
]
for Tp, lab, c, mk in pts:
    a.plot(Tp, kTln2(Tp), mk, ms=4, color=c, zorder=3, ls="none", label=lab)
a.legend(loc="upper left", fontsize=5.0, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.35)
a.axvline(348.15, color=bs.LIGHT, lw=0.6, ls=":")
a.text(348.15 * 1.3, 1e-28, "chip junction, 75 °C", rotation=90, fontsize=5.8, color=bs.GR, va="bottom")
a.set_xlim(2e-6, 3e3)
a.set_ylim(1e-30, 1e-17)
bs.panel_letter(a, "a")

T2 = np.logspace(-4, 3, 500)
curves = [
    (0.2e9, "fluxonium 0.2 GHz", bs.SKY, "-", p_two_level),
    (5e9, "transmon 5 GHz", bs.IAM, "-", p_two_level),
    (15e9, "Si spin 15 GHz", bs.GOLD, "-", p_two_level),
    (2.87e9, "NV 2.87 GHz", bs.DATA, "--", p_two_level),
    (100e9, "Rydberg line 100 GHz (photons)", bs.ALT, ":", n_bose),
    (193.4e12, "1550 nm photon", bs.GR, "-.", n_bose),
]
for f, lab, c, ls, fn in curves:
    y = fn(f, T2)
    m = y > 1e-16
    b.loglog(T2[m], y[m], ls=ls, color=c, lw=1.0, label=lab)
ops = [(0.035, 5e9, bs.IAM, p_two_level), (0.020, 0.2e9, bs.SKY, p_two_level), (0.1, 15e9, bs.GOLD, p_two_level),
       (1.5, 15e9, bs.GOLD, p_two_level), (300.0, 2.87e9, bs.DATA, p_two_level), (300.0, 100e9, bs.ALT, n_bose),
       (4.0, 100e9, bs.ALT, n_bose), (300.0, 193.4e12, bs.GR, n_bose)]
for Tp, f, c, fn in ops:
    b.plot(Tp, fn(f, Tp), "o", ms=3.2, color=c, zorder=3)
for Tp, f, fn, lab, off in [(0.035, 5e9, p_two_level, "35 mK", (-4, -8)), (0.020, 0.2e9, p_two_level, "20 mK", (-4, 4)),
                            (0.1, 15e9, p_two_level, "0.1 K", (5, -6)), (1.5, 15e9, p_two_level, "1.5 K", (-2, 5)),
                            (4.0, 100e9, n_bose, "4 K", (4, -9)), (300.0, 100e9, n_bose, "300 K", (-4, 5))]:
    b.annotate(lab, (Tp, fn(f, Tp)), xytext=off, textcoords="offset points", fontsize=5.6, color=bs.GR,
               ha="left" if off[0] > 0 else "right")
b.axhline(1e-2, color=bs.LIGHT, lw=0.6, ls="--")
b.text(1.5e-4, 1.3e-2, "1 %", fontsize=6, color=bs.GR)
b.set_xlabel("temperature of the bath the gap couples to (K)")
b.set_ylabel("thermal occupation")
b.set_ylim(1e-15, 1e3)
b.set_xlim(1e-4, 1e3)
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.4, handlelength=2.0, columnspacing=1.0)
bs.panel_letter(b, "b")
fig.tight_layout(w_pad=1.6)
bs.save(fig, "part3", "fig_p3_platform_floors")

# ---------------------------------------------------------------- figure 2
fig, (a, b) = plt.subplots(1, 2, figsize=(bs.TEXTW, 2.4))
Tc = np.linspace(273.15, 398.15, 200)
a.plot(Tc - 273.15, kTln2(Tc) * 1e21, color=bs.IAM, label=r"$k_BT_j\ln2$ (Landauer)")
a.plot(Tc - 273.15, kB * Tc * np.log(1e15) * 1e21, color=bs.ALT, ls="--", label=r"$k_BT_j\ln(1/p)$, $p=10^{-15}$")
a.set_yscale("log")
a.set_xlabel("junction temperature (°C)")
a.set_ylabel(r"floor energy ($10^{-21}$ J)")
a.legend(loc="center right", fontsize=6)
bs.panel_letter(a, "a")
for E, ls in [(1.92e-18, "-"), (1.97e-18, "--")]:
    b.plot(Tc - 273.15, kTln2(Tc) / E * 1e3, color=bs.IAM, ls=ls, lw=1.0)
for Tm in (75, 105):
    b.axvline(Tm, color=bs.LIGHT, lw=0.6, ls=":")
    b.text(Tm + 1, 1.62, f"{Tm} °C", fontsize=6, color=bs.GR)
b.set_xlabel("junction temperature (°C)")
b.set_ylabel(r"floor on the gauge ($\times10^{-3}$)")
b.text(0.03, 0.92, r"$E_{\rm sw}=1.92$ (solid) and $1.97\times10^{-18}$ J (dashed)", transform=b.transAxes, fontsize=6)
bs.panel_letter(b, "b", dx=-0.16)
fig.tight_layout(w_pad=2.4)
bs.save(fig, "part3", "fig_p3_chip_floor_Tj")
