"""Part 2, Chapter 'Entropy law or entropy source' (p2_03a_entropic_gravity.tex).
fig_eg_forms: (a) the extra Hubble friction on matter perturbations, in units of H, for the friction form (2 + beta_m E(a)) H and for the
Level 2 form 2 H_IAM; (b) the linear growth factor relative to LambdaCDM, D/D_LCDM - 1, for the friction form, the Level 2 form and the
effective-coupling form G_eff = mu G with the exact mu(a). Same early amplitude, beta_m = Omega_m/2 fixed, Omega_m = 0.3153.
Numbers: docs/verification/scripts/verify_entropic_gravity.py, section 5."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
import _bookstyle as S
import matplotlib.pyplot as plt

Om = 0.3153; OL = 1 - Om; b = Om / 2
Ea = lambda a: np.exp(1 - 1 / a)
E2L = lambda a: Om * a**-3 + OL
E2I = lambda a: E2L(a) + b * Ea(a)
mu = lambda a: E2L(a) / E2I(a)

def dlnH(f, a, h=1e-5):
    return 0.5 * (f(a * np.exp(h)) - f(a * np.exp(-h))) / (2 * h) / f(a)

def growth(mode):
    def rhs(l, y):
        a = np.exp(l); eL = E2L(a); kL = dlnH(E2L, a); src = 1.5 * Om * a**-3 / eL
        if mode == "LCDM":   return [y[1], -(2 + kL) * y[1] + src * y[0]]
        if mode == "L1":     return [y[1], -(2 + kL) * y[1] + src * mu(a) * y[0]]
        if mode == "L2":     return [y[1], -(kL + 2 * np.sqrt(E2I(a) / eL)) * y[1] + src * y[0]]
        if mode == "fric":   return [y[1], -(2 + b * Ea(a) + kL) * y[1] + src * y[0]]
    a0 = 1e-3
    return solve_ivp(rhs, (np.log(a0), 0), [a0, a0], dense_output=True, rtol=1e-10, atol=1e-14)

S.apply()
z = np.linspace(0, 3, 400); a = 1 / (1 + z); l = np.log(a)
gL = growth("LCDM").sol(l)[0]
D = {m: 100 * (growth(m).sol(l)[0] / gL - 1) for m in ("fric", "L2", "L1")}
for m in D:
    print(m, f"z=0: {D[m][0]:+.2f} %")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(z, b * Ea(a), color=S.IAM, lw=1.5, label=r"$(2+\beta_mE)H$: extra $\beta_mE(a)$")
a1.plot(z, 2 * (np.sqrt(E2I(a) / E2L(a)) - 1), color=S.ALT, lw=1.5, ls="--", label=r"$2H_{\rm IAM}$: extra $2(H_{\rm IAM}/H-1)$")
a1.set_xlim(0, 3); a1.set_ylim(0, 0.17)
a1.set_xlabel("redshift $z$"); a1.set_ylabel("extra friction / $H$")
a1.set_title("Friction on matter perturbations")
a1.legend(loc="upper right", handlelength=2.2)
S.panel_letter(a1, "a", dx=-0.16)

a2.axhline(0, color=S.GR, lw=0.8)
a2.plot(z, D["fric"], color=S.IAM, lw=1.5, label=f"friction form ({D['fric'][0]:+.2f} % today)")
a2.plot(z, D["L2"], color=S.ALT, lw=1.5, ls="--", label=f"Level 2, $2H_{{\\rm IAM}}$ ({D['L2'][0]:+.2f} %)")
a2.plot(z, D["L1"], color=S.ALT2, lw=1.5, ls=":", label=f"exact $\\mu(a)$, $\\mu G$ ({D['L1'][0]:+.2f} %)")
a2.set_xlim(0, 3); a2.set_ylim(-1.9, 0.2)
a2.set_xlabel("redshift $z$"); a2.set_ylabel(r"$D/D_{\Lambda{\rm CDM}}-1$ (%)")
a2.set_title("Linear growth factor")
a2.legend(loc="lower right", handlelength=2.2)
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_eg_forms")
