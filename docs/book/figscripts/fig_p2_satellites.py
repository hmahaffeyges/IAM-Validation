"""Part 2, Chapter 'Missing satellites and growth suppression from mu < 1' (p2_19_missing_satellites.tex).
fig_sat_mechanisms: (a) the matter coupling mu(z) = H^2/(H^2 + beta_m E(a) H0^2), beta_m = Omega_m/2 = 0.15765, Omega_m 0.3153;
  (b) the linear growth deficit D_IAM/D_LCDM - 1 against redshift in the three implementation forms of Appendix app:der:growth
  (i) G_eff = mu G, (ii) matter friction on the LCDM clock, (iii) whole growth equation on the matter-sector rate; same early amplitude;
  (c) the Press-Schechter abundance change at fixed mass, Delta ln n = (nu^2 - 1) eps, for eps = -0.78 % (form i), with the peak-height range
  of satellite halos today (nu = 0.24-0.35 for 1e7-1e9 Msun) and the order-of-magnitude deficit (ln 10).
Numbers: docs/verification/scripts/verify_cluster_mass_satellites.py sections A, F, G, H.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
Om, OL = 0.3153, 0.6847; bm = Om / 2
E = lambda a: np.exp(1 - 1 / a); H2 = lambda a: Om * a**-3 + OL; Hm2 = lambda a: H2(a) + bm * E(a)
mu = lambda a: H2(a) / Hm2(a); Oma = lambda a: Om * a**-3 / H2(a)
dlnH = lambda a: -1.5 * Om * a**-3 / H2(a); dlnHm = lambda a: (-3 * Om * a**-3 + bm * E(a) / a) / (2 * Hm2(a))
forms = {"LCDM": lambda l, y: [y[1], -(2 + dlnH(np.exp(l))) * y[1] + 1.5 * Oma(np.exp(l)) * y[0]],
         "i": lambda l, y: [y[1], -(2 + dlnH(np.exp(l))) * y[1] + 1.5 * Oma(np.exp(l)) * mu(np.exp(l)) * y[0]],
         "ii": lambda l, y: [y[1], -(dlnH(np.exp(l)) + 2 * np.sqrt(Hm2(np.exp(l)) / H2(np.exp(l)))) * y[1] + 1.5 * Oma(np.exp(l)) * y[0]],
         "iii": lambda l, y: [y[1], -(2 + dlnHm(np.exp(l))) * y[1] + 1.5 * Om * np.exp(-3 * l) / Hm2(np.exp(l)) * y[0]]}
Sol = {k: solve_ivp(v, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14) for k, v in forms.items()}
z = np.linspace(0, 3, 301); a = 1 / (1 + z)
dD = {k: 100 * (Sol[k].sol(np.log(a))[0] / Sol["LCDM"].sol(np.log(a))[0] - 1) for k in ("i", "ii", "iii")}
eps = -dD["i"][0] / 100
print(f"mu(0) {mu(1.0):.4f}; dD/D today i {dD['i'][0]:.2f} % ii {dD['ii'][0]:.2f} % iii {dD['iii'][0]:.2f} %")
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.45))
a1.plot(z, mu(a), color=S.IAM, lw=1.6, label="$\\mu(z)$")
a1.axhline(1, color=S.GR, lw=0.9, ls="--", label="$\\Lambda$CDM")
a1.annotate(f"$\\mu(0)={mu(1.0):.3f}$", (0, mu(1.0)), xytext=(10, 12), textcoords="offset points", fontsize=7, va="bottom",
            arrowprops=dict(arrowstyle="-", color=S.GR, lw=0.5))
a1.set_xlim(0, 3); a1.set_ylim(0.85, 1.01); a1.set_xlabel("redshift $z$"); a1.set_ylabel("matter coupling $\\mu$")
a1.legend(loc="center right", fontsize=6.5); a1.set_title("The coupling"); S.panel_letter(a1, "a", dx=-0.28)
for k, c, ls, lab in (("i", S.IAM, "-", "(i) $G_{\\rm eff}=\\mu G$"), ("ii", S.ALT, "--", "(ii) friction"), ("iii", S.GOLD, ":", "(iii) all on $H_m$")):
    a2.plot(z, dD[k], color=c, ls=ls, lw=1.4, label=f"{lab}: {dD[k][0]:.2f} %")
a2.axhline(0, color=S.GR, lw=0.6)
a2.set_xlim(0, 3); a2.set_xlabel("redshift $z$"); a2.set_ylabel("$\\Delta D/D$ (%)")
a2.set_ylim(-2.6, 0.2); a2.legend(loc="lower right", fontsize=6); a2.set_title("Growth, three forms"); S.panel_letter(a2, "b", dx=-0.28)
nu = np.linspace(0.02, 3, 400)
a3.plot(nu, np.abs(nu**2 - 1) * 100 * eps, color=S.IAM, lw=1.6, label=f"$|\\Delta\\ln n|$, $\\epsilon=-{100*eps:.2f}$ %")
a3.axhline(100 * np.log(10), color=S.DATA, lw=1.0, ls="--", label="tenfold deficit ($\\ln10$)")
a3.axvspan(0.24, 0.35, color=S.LIGHT, alpha=0.6, lw=0); a3.text(0.40, 1.5e-2, "satellite\nhalos", fontsize=6.5, color=S.GR, ha="left")
a3.set_yscale("log"); a3.set_xlim(0, 3); a3.set_ylim(1e-2, 1e3)
a3.set_xlabel("peak height $\\nu=\\delta_c/\\sigma_M$"); a3.set_ylabel("$|\\Delta\\ln n|$ (%)")
a3.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6); a3.set_title("Halo abundance"); S.panel_letter(a3, "c", dx=-0.28)
S.save(fig, "part2", "fig_sat_mechanisms")
