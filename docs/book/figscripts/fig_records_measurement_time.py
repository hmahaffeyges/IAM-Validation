#!/usr/bin/env python3
"""Figures for Part 2 'Records at the quantum scale' (p2_14) and Part 5 'Duration ...' (p5_03) and 'Measurement ...' (p5_04).
Numbers: docs/verification/scripts/verify_records_measurement_time.py (same equations).
Outputs: figures/part4/fig_qd_local_cosmic, fig_qd_exponent, fig_qd_mu; figures/part7/fig_time_two_faces, fig_mp_record, fig_mp_tau_systems.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import _bookstyle as bs
from _cosmo import Om, OL, bm, H2, mu, LCDM, IAM, D, f
import scipy.constants as C
from scipy.integrate import solve_ivp

bs.apply()


def box(ax, x, y, w, h, text, fc="white", ec=bs.GR, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, color=bs.GR):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=color))


# QD Fig. 1: local framework and its cosmic extension
fig, ax = plt.subplots(figsize=(bs.TEXTW, 3.1))
ax.set_xlim(0, 1); ax.set_ylim(0, 1.12); ax.axis("off")
ax.text(0.22, 1.06, "Local records\n(decoherence, quantum Darwinism)", ha="center", va="top", fontsize=7, weight="bold")
ax.text(0.75, 1.06, "The same records summed\nover structure formation", ha="center", va="top", fontsize=7, weight="bold")
box(ax, 0.04, 0.76, 0.36, 0.11, "system S + environment E\n$|\\Psi\\rangle=\\sum_i c_i|s_i\\rangle|e_i\\rangle$")
box(ax, 0.04, 0.56, 0.36, 0.11, "einselection: pointer states survive\n$\\tau_D\\sim\\tau_R(\\lambda_{th}/\\Delta x)^2\\to0$ for macroscopic $\\Delta x$")
box(ax, 0.04, 0.36, 0.36, 0.11, "redundancy $R_\\delta$: many fragments\ncarry the same pointer record")
box(ax, 0.04, 0.16, 0.36, 0.11, "observers 1 ... N read fragments\nand agree (objectivity)")
for y in (0.76, 0.56, 0.36):
    arrow(ax, 0.22, y, 0.22, y - 0.085)
box(ax, 0.56, 0.76, 0.38, 0.11, "collapse of bound structure; halo decoherence\n$\\tau\\sim\\hbar R/GM^2=2.5\\times10^{-87}$ s: instantaneous", fc="#EAF3FA", ec=bs.IAM)
box(ax, 0.56, 0.56, 0.38, 0.11, "IAM's Law: $k_BT\\ln2$ per bit at the nearest\nencoding surface; the cosmic horizon is the largest", fc="#EAF3FA", ec=bs.IAM)
box(ax, 0.56, 0.36, 0.38, 0.11, "accumulated ledger $E(a)=\\exp(1-1/a)$;\nrate $\\dot I\\propto\\rho_mD^nfH$ with $n=7/2$", fc="#EAF3FA", ec=bs.IAM)
box(ax, 0.56, 0.16, 0.38, 0.11, "matter: $\\mu(a)<1$, $\\mu_0=-0.136$\nlight on null geodesics: $\\Sigma=1$", fc="#EAF3FA", ec=bs.IAM)
for y in (0.76, 0.56, 0.36):
    arrow(ax, 0.75, y, 0.75, y - 0.085, bs.IAM)
for y in (0.815, 0.615, 0.415, 0.215):
    arrow(ax, 0.405, y, 0.555, y, bs.LIGHT)
ax.text(0.48, 0.05, "the virial half sets how much bound energy writes ($\\beta_m=\\Omega_m/2$)", ha="center", fontsize=6.5, color=bs.GR)
bs.save(fig, "part4", "fig_qd_local_cosmic")

# QD Fig. 3, quantitative: the exponent from both directions
def grow():
    def rhs(l, y):
        a = np.exp(l); dlnH = -1.5 * Om * a**-3 / H2(a)
        return [y[1], -(2 + dlnH) * y[1] + 1.5 * Om * a**-3 / H2(a) * y[0]]
    return solve_ivp(rhs, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
S = grow()
Dn = lambda a: S.sol(np.log(a))[0] / S.sol(0)[0]
fg = lambda a: S.sol(np.log(a))[1] / S.sol(np.log(a))[0]
def slope(nn, a1, a2):
    aa = np.logspace(np.log10(a1), np.log10(a2), 60)
    g = aa**-3 * Dn(aa)**nn * fg(aa) * np.sqrt(H2(aa))
    return np.polyfit(np.log(aa), np.log(g), 1)[0]
ns = np.linspace(2, 5, 31)
pm = [slope(x, 0.01, 0.1) for x in ns]; pl = [slope(x, 0.25, 1.0) for x in ns]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(bs.TEXTW, 2.5))
a1.plot(ns, pm, color=bs.IAM, label="matter era, $a=0.01$-$0.1$")
a1.plot(ns, pl, color=bs.ALT, ls="--", label="late, $a=0.25$-$1$")
a1.axhline(-1, color=bs.LIGHT, lw=0.8); a1.axvline(3.5, color=bs.LIGHT, lw=0.8)
a1.text(3.55, -2.9, "$n=7/2$", fontsize=6.5, color=bs.GR); a1.text(4.0, -1.25, "$E=e^{1-1/a}$ needs $p=-1$", fontsize=6.5, color=bs.GR)
a1.set_xlabel("exponent $n$ in $\\dot I\\propto\\rho_mD^nfH$"); a1.set_ylabel("slope $p$ of $dS_{\\rm info}/d\\ln a\\propto a^p$")
a1.legend(loc="upper left"); bs.panel_letter(a1, "a")
nu = np.linspace(1, 3, 200)
a2.plot(nu, nu**2 - 1, color=bs.IAM)
for v, lab in ((np.sqrt(4.5), "7/2 at $\\nu=2.12$"), (np.sqrt(3.5), "5/2 at $\\nu=1.87$")):
    a2.plot([v], [v**2 - 1], "o", ms=3.5, color=bs.DATA); a2.annotate(lab, (v, v**2 - 1), xytext=(-62, 8), textcoords="offset points", fontsize=6.5)
a2.set_xlabel("peak height $\\nu=\\delta_c/\\sigma(M_{\\min})D$"); a2.set_ylabel("$n_{\\rm eff}$, one record per particle")
bs.panel_letter(a2, "b")
fig.tight_layout(); bs.save(fig, "part4", "fig_qd_exponent")

# QD Fig. 4: mu(z), 1-mu and the f sigma8 deficit
z = np.linspace(0, 3.5, 300); a = 1 / (1 + z)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(bs.TEXTW, 2.4))
b1.plot(z, mu(a), color=bs.IAM, label="$\\mu(z)$")
b1.axhline(1, color=bs.GR, ls="--", lw=0.8, label="general relativity")
b1.plot([0], [mu(1.0)], "o", ms=3.5, color=bs.IAM); b1.annotate(f"$\\mu(0)={mu(1.0):.4f}$", (0, mu(1.0)), xytext=(8, 0), textcoords="offset points", fontsize=6.5)
b1.set_xlabel("redshift $z$"); b1.set_ylabel("$\\mu$"); b1.legend(loc="lower right"); bs.panel_letter(b1, "a")
fs8 = 100 * (1 - f(IAM, a) * D(IAM, a) / (f(LCDM, a) * D(LCDM, a)))
b2.plot(z, 100 * (1 - mu(a)), color=bs.IAM, label="$1-\\mu$ (coupling)")
b2.plot(z, fs8, color=bs.DATA, ls="-.", label="$f\\sigma_8$ deficit (growth)")
for zz in (0, 0.3, 0.5, 1.0):
    aa = 1 / (1 + zz); v = 100 * (1 - f(IAM, aa) * D(IAM, aa) / (f(LCDM, aa) * D(LCDM, aa)))
    b2.plot([zz], [v], "o", ms=3, color=bs.DATA)
b2.set_xlabel("redshift $z$"); b2.set_ylabel("per cent"); b2.legend(loc="upper right"); bs.panel_letter(b2, "b")
fig.tight_layout(); bs.save(fig, "part4", "fig_qd_mu")

# Two Faces Fig. 1
fig, ax = plt.subplots(1, 3, figsize=(bs.TEXTW, 2.3))
aa = np.linspace(0.02, 2, 300)
ax[0].plot(aa, aa, color=bs.GR); ax[0].set_ylim(0, 2.3)
ax[0].text(0.08, 1.85, "a label on the manifold:\nno direction,\nno accumulation", fontsize=6.3)
ax[0].set_xlabel("scale factor $a$"); ax[0].set_ylabel("value"); ax[0].set_title("coordinate time", fontsize=7); bs.panel_letter(ax[0], "a")
ax[1].plot(aa, aa, color=bs.ALT, label="timelike worldline")
ax[1].plot(aa, 0 * aa, color=bs.GOLD, ls="--", label="null geodesic, $d\\tau=0$")
ax[1].set_ylim(-0.1, 2.3); ax[1].legend(loc="upper left")
ax[1].set_xlabel("scale factor $a$"); ax[1].set_ylabel("proper time (arb.)"); ax[1].set_title("proper time", fontsize=7); bs.panel_letter(ax[1], "b")
ax[2].plot(aa, np.exp(1 - 1 / aa), color=bs.IAM, label="matter: $E(a)=e^{1-1/a}$")
ax[2].plot(aa, 0 * aa, color=bs.GOLD, ls="--", label="light: 0")
ax[2].axhline(np.e, color=bs.LIGHT, lw=0.8); ax[2].text(0.05, np.e + 0.06, "asymptote $e$", fontsize=6.3, color=bs.GR)
ax[2].axvline(1, color=bs.LIGHT, lw=0.8); ax[2].text(1.03, 0.1, "today", fontsize=6.3, color=bs.GR)
ax[2].set_ylim(-0.1, 3.2); ax[2].legend(loc="upper left", bbox_to_anchor=(0.0, 0.82))
ax[2].set_xlabel("scale factor $a$"); ax[2].set_ylabel("accumulated record"); ax[2].set_title("accumulated record", fontsize=7); bs.panel_letter(ax[2], "c")
fig.tight_layout(); bs.save(fig, "part7", "fig_time_two_faces")

# Measurement Figs. 1-2
fig, (m1, m2) = plt.subplots(1, 2, figsize=(bs.TEXTW, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
m1.set_xlim(0, 1); m1.set_ylim(0, 1); m1.axis("off")
box(m1, 0.02, 0.28, 0.42, 0.6, "no irreversible record yet\n\nphotons in flight ($d\\tau=0$)\nsuperpositions, pairs in transit\nreversible markers\n(polarisation rotation,\nbeam splitter)\n\nerasure possible", fc="#FFF6E5", ec=bs.GOLD)
box(m1, 0.56, 0.28, 0.42, 0.6, "record written in matter\n\nabsorbed in a detector,\namplified, dissipated\n($Q\\gg k_BT\\ln2$)\n\ncost $k_BT\\ln2$ per bit,\npaid at reset or erasure", fc="#EAF3FA", ec=bs.IAM)
arrow(m1, 0.45, 0.58, 0.55, 0.58, bs.GR)
m1.text(0.5, 0.13, "no observer and no consciousness enter", ha="center", fontsize=6.5, color=bs.GR)
bs.panel_letter(m1, "a", dx=0.02)
c = C.c; t_det = 5 / c * 1e9
m2.plot([0, t_det], [0, 5], color=bs.GOLD, lw=1.2, label="photon, coherent in flight")
m2.axhline(2.0, color=bs.LIGHT, lw=0.8); m2.text(0.4, 2.1, "slits", fontsize=6.3)
m2.axvline(12.0, color=bs.GR, ls=":", lw=0.8); m2.text(12.3, 0.25, "choice of\nconfiguration", fontsize=6.3)
m2.plot([t_det], [5], "o", ms=4, color=bs.IAM); m2.annotate(f"absorbed: record written\n$t={t_det:.1f}$ ns", (t_det, 5), xytext=(-40, -50), textcoords="offset points", fontsize=6.3)
m2.set_xlim(0, 19); m2.set_ylim(0, 5.9)
m2.set_xlabel("time since emission (ns)"); m2.set_ylabel("distance along apparatus (m)"); m2.legend(loc="upper left")
bs.panel_letter(m2, "b")
fig.tight_layout(); bs.save(fig, "part4", "fig_mp_record")

# Measurement Fig. 5 / Table 1
hb, G, kB = C.hbar, C.G, C.k
rows = [("electron", 9.109e-31, 1.0e-10), ("C60", 1.2e-24, 5.0e-10), ("virus", 1e-18, 5.0e-8), ("bacterium", 1e-15, 5.0e-7),
        ("dust grain", 1e-12, 5.0e-6), ("sand grain", 1e-6, 5.0e-4), ("cat", 4.0, 0.15), ("person", 70.0, 0.30)]
m_ = np.array([r[1] for r in rows]); R_ = np.array([r[2] for r in rows]); EG = G * m_**2 / R_
tPD = hb / EG; tI = hb * (kB * 300)**2 * np.log(2) / EG**3
tP = np.sqrt(hb * G / c**5)
fig, axx = plt.subplots(figsize=(0.62 * bs.TEXTW, 2.7))
axx.loglog(m_, tI, "o-", color=bs.IAM, ms=3, label="$\\tau_{\\rm IAM}$, 300 K (capacity $k_BT/E_G$ assumed)")
axx.loglog(m_, tPD, "s--", color=bs.GR, ms=3, label="$\\tau_{\\rm PD}=\\hbar/E_G$")
axx.axhline(tP, color=bs.LIGHT, lw=0.8); axx.text(1e-4, tP * 5, "Planck time", fontsize=6.3, color=bs.GR)
axx.axhline(4.35e17, color=bs.LIGHT, lw=0.8, ls=":"); axx.text(2e-30, 4.35e17 * 5, "age of the universe", fontsize=6.3, color=bs.GR)
axx.axvspan(1e-15, 1e-10, color=bs.SKY, alpha=0.18, lw=0)
for (name, mm, rr), t in zip(rows, tI):
    off = (-14, -9) if name == "cat" else (3, 3)
    axx.annotate(name, (mm, t), xytext=off, textcoords="offset points", fontsize=5.8)
axx.set_xlabel("mass (kg)"); axx.set_ylabel("time (s)"); axx.legend(loc="upper right", fontsize=6)
bs.save(fig, "part4", "fig_mp_tau_systems")
