"""Part 2, Chapters 'Entanglement and the cost of records' (p4_21_entanglement_records.tex) and 'Electroweak symmetry breaking and the
matter sector' (p4_22_electroweak.tex). Equations and constants as docs/verification/scripts/verify_entanglement_electroweak.py
(scipy.constants CODATA; PDG 2022; Planck 2018).
fig_chsh_dephasing: (a) CHSH value against the coherence c of a Bell pair: S_max = 2 sqrt(1 + c^2) for pointer-basis dephasing
  (Horodecki criterion, computed from the density matrix), S = sqrt2 (1 + c) with the pure-state settings, and S_max = 2 sqrt2 p for
  isotropic noise with p = c; (b) S_max(t) for the exponential c = exp(-t/tau) and for the assumed profile c = 1 - E(t/tau)/e.
fig_tau_temperature: silica spheres (2200 kg/m^3). (a) tau_IAM = hbar (k_B T)^2 ln2 / E_G^3 against bath temperature for 1e-15, 1e-13 and
  1e-12 kg, with the Diosi-Penrose time hbar/E_G of each (independent of T); (b) the ratio tau_IAM/tau_PD against mass at 10, 20 and 50 mK.
fig_ew_timeline: (a) the thermal history: the events of the chapter's table at their computed times and temperatures (radiation era
  t = 0.301 g*^(-1/2) m_P/T^2 for the electroweak, QCD and nucleosynthesis points; the Friedmann integral with radiation, matter and Lambda
  for recombination, z = 30 and today); (b) E(a) = exp(1 - 1/a) from z = 40 to today.
fig_cornell: the Cornell potential in units of its crossover radius r0 = sqrt(4 alpha_s/(3 sigma)), V/(sigma r0) = -r0/r + r/r0:
  (a) the potential and its two parts; (b) the share of the virial weight r dV/dr carried by the linear (confining) part, x^2/(1+x^2),
  which goes from 0 (Coulomb-like, k = -1) to 1 (linear, k = +1).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
from scipy.integrate import quad
from scipy.optimize import brentq
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
hb, kB, G, c = C.hbar, C.k, C.G, C.c
E = lambda a: np.exp(1 - 1 / a)
# ---------------- CHSH -----------------
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1, -1]); P = [sx, sy, sz]
psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
def rho_deph(cf):
    r = np.outer(psi, psi.conj()).astype(complex); r[0, 3] *= cf; r[3, 0] *= cf; return r
rho_iso = lambda p: p * np.outer(psi, psi) + (1 - p) * np.eye(4) / 4
Tm = lambda r: np.array([[np.trace(r @ np.kron(a, b)).real for b in P] for a in P])
def Smax(r):
    ev = np.sort(np.linalg.eigvalsh(Tm(r).T @ Tm(r))); return 2 * np.sqrt(ev[-1] + ev[-2])
def Sfix(r):
    T = Tm(r); z_, x_ = np.array([0, 0, 1.]), np.array([1., 0, 0]); b = (z_ + x_) / np.sqrt(2); bp = (z_ - x_) / np.sqrt(2)
    e = lambda u, v: u @ T @ v; return e(z_, b) + e(z_, bp) + e(x_, b) - e(x_, bp)
cc = np.linspace(0, 1, 101)
Sd = np.array([Smax(rho_deph(q)) for q in cc]); Sf = np.array([Sfix(rho_deph(q)) for q in cc]); Si = np.array([Smax(rho_iso(q)) for q in cc])
assert np.allclose(Sd, 2 * np.sqrt(1 + cc**2)) and np.allclose(Sf, np.sqrt(2) * (1 + cc)) and np.allclose(Si, 2 * np.sqrt(2) * cc)
tt = np.linspace(1e-4, 3.5, 300); ce = np.exp(-tt); cr = 1 - E(tt) / np.e
print(f"fixed settings lose violation at c = {np.sqrt(2)-1:.4f}; isotropic at p = {1/np.sqrt(2):.4f}; t=1: S_exp {2*np.sqrt(1+np.exp(-2)):.4f}, S_assumed {2*np.sqrt(1+(1-E(1)/np.e)**2):.4f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
a1.plot(cc, Sd, color=S.IAM, lw=1.6, label="dephasing, best settings")
a1.plot(cc, Sf, color=S.ALT, lw=1.3, ls="--", label="dephasing, pure-state settings")
a1.plot(cc, Si, color=S.DATA, lw=1.3, ls="-.", label="isotropic noise, $p=c$")
a1.axhline(2, color=S.GR, lw=0.8); a1.axhline(2 * np.sqrt(2), color=S.GR, lw=0.6, ls=":")
a1.text(0.02, 2.03, "local bound 2", fontsize=7, color=S.GR, va="bottom"); a1.text(0.02, 2 * np.sqrt(2) - 0.03, "$2\\sqrt{2}$", fontsize=7, color=S.GR, va="top")
a1.plot(np.sqrt(2) - 1, 2, "o", color=S.ALT, ms=3.5); a1.plot(1 / np.sqrt(2), 2, "o", color=S.DATA, ms=3.5)
a1.set_xlim(0, 1); a1.set_ylim(0, 3.0)
a1.set_xlabel("coherence $c$ (or $p$)"); a1.set_ylabel("CHSH value $S$")
a1.legend(loc="lower right", fontsize=6)
a1.set_title("Dephasing keeps a violation while $c>0$")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(tt, 2 * np.sqrt(1 + ce**2), color=S.IAM, lw=1.6, label="$c=e^{-t/\\tau_{\\rm IAM}}$ (rate integral)")
a2.plot(tt, 2 * np.sqrt(1 + cr**2), color=S.ALT2, lw=1.4, ls="--", label="$c=1-E(t/\\tau_{\\rm IAM})/e$ (assumed)")
a2.axhline(2, color=S.GR, lw=0.8)
for q in (0.5, 1, 2, 3):
    a2.plot(q, 2 * np.sqrt(1 + np.exp(-q)**2), "o", color=S.IAM, ms=3); a2.plot(q, 2 * np.sqrt(1 + (1 - E(q) / np.e)**2), "o", color=S.ALT2, ms=3)
a2.set_xlim(0, 3.5); a2.set_ylim(1.9, 2.9)
a2.set_xlabel("$t/\\tau_{\\rm IAM}$"); a2.set_ylabel("$S_{\\max}$")
a2.legend(loc="upper right", fontsize=6.5)
a2.set_title("The violation fades to 2, never below")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part4", "fig_chsh_dephasing")

# ---------------- tau(T) -----------------
rho = 2200.0
EG = lambda m: G * m * m / ((3 * m / (4 * np.pi * rho))**(1 / 3))
tI = lambda m, T: hb * (kB * T)**2 * np.log(2) / EG(m)**3
tPD = lambda m: hb / EG(m)
mx = 10**brentq(lambda l: np.log(tI(10**l, .01) / tPD(10**l)), -20, -5)
print(f"tau_IAM(1e-12, 10 mK) {tI(1e-12,.01):.4g} s; 20 mK {tI(1e-12,.02):.4g}; tau_PD(1e-12) {tPD(1e-12):.4g}; (1e-15,10 mK) {tI(1e-15,.01):.3g}, {tPD(1e-15):.3g}; crossover {mx:.3g} kg")
TT = np.logspace(-3, 0, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
for m, col in ((1e-15, S.ALT), (1e-13, S.GOLD), (1e-12, S.IAM)):
    lab = f"{m:.0e}".replace("1e-", "$10^{-") + "}$ kg"
    a1.plot(TT * 1e3, tI(m, TT), color=col, lw=1.5, label=f"$\\tau_{{\\rm IAM}}$, {lab}")
    a1.axhline(tPD(m), color=col, lw=0.9, ls=":")
a1.plot(10, tI(1e-12, .01), "o", color=S.IAM, ms=4); a1.plot(20, tI(1e-12, .02), "o", color=S.IAM, ms=4)
a1.annotate(f"{tI(1e-12,.01):.0f} s at 10 mK", (10, tI(1e-12, .01)), xytext=(6, -8), textcoords="offset points", fontsize=7, color=S.IAM)
a1.text(1.1, 2e-7, "dotted: Di\u00f3si\u2013Penrose $\\hbar/E_G$", fontsize=6.5, color=S.GR, va="bottom")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(1, 1000); a1.set_ylim(1e-8, 1e36)
a1.set_xlabel("bath temperature $T$ (mK)"); a1.set_ylabel("time (s)")
a1.legend(loc="upper left", fontsize=6)
a1.set_title("$\\tau_{\\rm IAM}\\propto T^2$; Di\u00f3si\u2013Penrose is flat")
S.panel_letter(a1, "a", dx=-0.14)
mm = np.logspace(-16, -8, 200)
for T, col, ls in ((0.01, S.IAM, "-"), (0.02, S.ALT, "--"), (0.05, S.ALT2, "-.")):
    a2.plot(mm, tI(mm, T) / tPD(mm), color=col, lw=1.5, ls=ls, label=f"{T*1e3:.0f} mK")
a2.axhline(1, color=S.GR, lw=0.8)
a2.plot(mx, 1, "o", color=S.IAM, ms=4); a2.annotate(f"crossover {mx:.1e} kg".replace("e-10", "$\\times10^{-10}$"), (mx, 1), xytext=(-6, -10), textcoords="offset points", fontsize=7, ha="right", va="top")
a2.axvline(1e-12, color=S.LIGHT, lw=0.6, ls=":"); a2.text(1.15e-12, 1e18, "1 pg", fontsize=7, color=S.GR)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1e-16, 1e-8); a2.set_ylim(1e-12, 1e30)
a2.set_xlabel("mass $m$ (kg)"); a2.set_ylabel("$\\tau_{\\rm IAM}/\\tau_{\\rm PD}$")
a2.legend(loc="upper right", fontsize=6.5)
a2.set_title("Where the two channels separate")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part4", "fig_tau_temperature")

# ---------------- electroweak timeline -----------------
GeV = 1e9 * C.e; hbar_GeVs = hb / GeV
mPl = np.sqrt(hb * c / G) * c**2 / GeV
t_of_T = lambda T, gs: 0.301 * mPl / (np.sqrt(gs) * T**2) * hbar_GeVs
T0K = 2.7255; h = 0.6736; Om = 0.3153; Ogh2 = 2.4728e-5; Or = Ogh2 * (1 + 0.2271 * 3.046) / h**2; OL = 1 - Om - Or
H0 = 100 * h * 1e3 / C.parsec / 1e6; yr = 3.15576e7
age = lambda z: quad(lambda a: 1 / (a * H0 * np.sqrt(Or / a**4 + Om / a**3 + OL)), 0, 1 / (1 + z), epsabs=0, epsrel=1e-10, limit=200)[0]
K_per_GeV = GeV / kB
ev = [("electroweak crossover", t_of_T(159.5, 106.75), 159.5 * K_per_GeV),
      ("QCD confinement", t_of_T(0.15, 17.25), 0.15 * K_per_GeV),
      ("nucleosynthesis", t_of_T(1e-4, 3.36), 1e-4 * K_per_GeV),
      ("nucleosynthesis end", t_of_T(7e-5, 3.36), 7e-5 * K_per_GeV),
      ("recombination", age(1089.9), T0K * 1090.9),
      ("first haloes, $z\\approx30$", age(30), T0K * 31),
      ("today", age(0), T0K)]
for nm, t, T in ev: print(f"{nm}: t {t:.3g} s ({t/yr:.3g} yr), T {T:.4g} K")
zz = np.logspace(-3, 8.5, 260); tz = np.array([age(z) for z in zz])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.9), gridspec_kw=dict(wspace=0.36))
tl = np.logspace(-12.5, 2.5, 50); a1.plot(tl, np.sqrt(0.301 * mPl * hbar_GeVs / (np.sqrt(106.75) * tl)) * K_per_GeV, color=S.LIGHT, lw=0.8, ls=":")
a1.plot(tz, T0K * (1 + zz), color=S.GR, lw=1.0)
off = {"electroweak crossover": (6, 0), "QCD confinement": (6, 0), "nucleosynthesis": (6, 2), "nucleosynthesis end": (6, -7),
       "recombination": (-7, 0), "first haloes, $z\\approx30$": (-7, 0), "today": (-7, 0)}
for nm, t, T in ev:
    col = S.IAM if nm.startswith("electroweak") else S.DATA
    a1.plot(t, T, "o", color=col, ms=4)
    if nm != "nucleosynthesis end":
        a1.annotate(nm, (t, T), xytext=off[nm], textcoords="offset points", fontsize=6.5, color=col, ha="right" if nm in ("today", "recombination", "first haloes, $z\\approx30$") else "left", va="center")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(1e-13, 3e18); a1.set_ylim(1, 1e17)
a1.set_xlabel("cosmic time $t$ (s)"); a1.set_ylabel("temperature (K)")
a1.set_title("Thermal history")
S.panel_letter(a1, "a", dx=-0.16)
ag = np.logspace(np.log10(1 / 41), 0, 300)
a2.plot(ag, E(ag), color=S.IAM, lw=1.6)
for z_, lab in ((30, "$z=30$"), (10, "$z=10$"), (1, "$z=1$"), (0, "today")):
    a = 1 / (1 + z_); a2.plot(a, E(a), "o", color=S.IAM, ms=3.5)
    a2.annotate(f"{lab}: {S.sci(E(a)) if E(a) < 0.01 else f'{E(a):.2f}'}", (a, E(a)), xytext=(6, -2) if z_ else (-6, 5), textcoords="offset points", fontsize=7,
                va="top" if z_ else "bottom", ha="left" if z_ else "right")
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(1 / 41, 1.6); a2.set_ylim(1e-17, 10)
a2.set_xticks([1 / 31, 1 / 11, 0.5, 1]); a2.set_xticklabels(["1/31", "1/11", "1/2", "1"])
a2.set_xlabel("scale factor $a$"); a2.set_ylabel("$E(a)=\\exp(1-1/a)$")
a2.set_title("Activation $E(a)$ once structure forms")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_ew_timeline")

# ---------------- Cornell -----------------
x = np.logspace(-1.3, 1.3, 400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(x, -1 / x + x, color=S.IAM, lw=1.6, label="Cornell $V$")
a1.plot(x, -1 / x, color=S.ALT, lw=1.1, ls="--", label="Coulomb-like part ($k=-1$)")
a1.plot(x, x, color=S.DATA, lw=1.1, ls="-.", label="linear part ($k=+1$)")
a1.axhline(0, color=S.GR, lw=0.6); a1.axvline(1, color=S.LIGHT, lw=0.6, ls=":")
a1.set_xscale("log"); a1.set_xlim(0.05, 20); a1.set_ylim(-8, 8)
a1.set_xlabel("separation $r/r_0$"); a1.set_ylabel("$V/(\\sigma r_0)$")
a1.legend(loc="upper left", fontsize=6.5)
a1.set_title("Two regimes of the quark potential")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(x, x**2 / (1 + x**2), color=S.DATA, lw=1.6)
a2.axhline(0.5, color=S.LIGHT, lw=0.6, ls=":"); a2.axvline(1, color=S.LIGHT, lw=0.6, ls=":")
a2.text(17, 0.06, "short range: $2\\langle K\\rangle=-\\langle V\\rangle$ (Coulomb)", fontsize=7, color=S.ALT, ha="right")
a2.text(0.06, 0.94, "long range: $2\\langle K\\rangle=+\\langle V\\rangle$ (linear)", fontsize=7, color=S.DATA, ha="left", va="top")
a2.set_xscale("log"); a2.set_xlim(0.05, 20); a2.set_ylim(0, 1.0)
a2.set_xlabel("separation $r/r_0$"); a2.set_ylabel("linear share of $r\\,dV/dr$")
a2.set_title("Hadrons are not $1/r$ systems")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_cornell")
