"""Part 2, Chapter 'What the surveys will measure' (p2_16_survey_predictions.tex).
Same equations as docs/verification/scripts/verify_obs_chapters.py sections A-D and H (via _cosmo.py): LambdaCDM background, Om 0.3153,
beta_m = Om/2, E(a) = exp(1 - 1/a), mu = H^2/(H^2 + beta_m E H0^2), Sigma = 1, linear growth with the same early amplitude.
fig_survey_ramp: (a) 1 - mu, the f sigma8 deficit and the growth deficit against z, with the 10, 50 and 90 % activation milestones of
  1 - mu(0); (b) what light sees: the E_G change, the potential change Delta Phi/Phi = Delta D/D (Sigma = 1) and the ISW source (1 - f) D / a.
fig_survey_precision: the precision each test needs. (a) Separation of the IAM mu(z) from GR as a function of the template error sigma(mu0),
  using the template-equivalent mu0 = -0.072 of verify_euclid_template.py, with Euclid's published errors on 1 + mu0 (23.3 %, 4 %, ~1 %;
  Albuquerque et al. 2025, Frusciante et al. 2025); (b) separation of the matter-sector (72.26) and photon-sector (67.16) rates by a siren
  population as a function of sigma(H0), with GW170817 alone (+12/-8, Abbott et al. 2017) and the 3 sigma requirement.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.optimize import brentq
import _bookstyle as S
import _cosmo as K
import matplotlib.pyplot as plt

S.apply()
z = np.linspace(0, 2.5, 251); a = 1 / (1 + z)
one_mu = 100 * (1 - K.mu(a)); dfs8 = K.fs8_deficit(z); dD = K.amp_deficit(z)
EG = 100 * (K.f(K.LCDM, a) / K.f(K.IAM, a) - 1)
src = lambda Sx, aa: (1 - K.f(Sx, aa)) * K.D(Sx, aa) / aa
isw = 100 * (src(K.IAM, a) / src(K.LCDM, a) - 1)
d0 = 1 - K.mu(1.0)
mil = {q: 1 / brentq(lambda x: (1 - K.mu(x)) / d0 - q, 0.05, 1) - 1 for q in (0.10, 0.50, 0.90)}
print("z=0: 1-mu %.2f %%, fs8 %.2f %%, D %.2f %%; z=0.3: fs8 %.2f %%, E_G %+.2f %%; milestones %s" % (
    one_mu[0], dfs8[0], dD[0], K.fs8_deficit(0.3), 100 * (K.f(K.LCDM, 1 / 1.3) / K.f(K.IAM, 1 / 1.3) - 1),
    {k: round(v, 2) for k, v in mil.items()}))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
a1.plot(z, one_mu, color=S.IAM, lw=1.6, label="$1-\\mu$ (the coupling)")
a1.plot(z, dfs8, color=S.DATA, lw=1.4, label="$f\\sigma_8$ deficit")
a1.plot(z, dD, color=S.ALT, lw=1.4, ls="--", label="growth deficit $|\\Delta D/D|$")
for q, zq in mil.items():
    yl = {0.10: 2.2, 0.50: 9.0, 0.90: 6.0}[q]
    a1.axvline(zq, color=S.LIGHT, lw=0.6, ls=":")
    a1.text(zq + 0.025, yl, f"{100*q:.0f} % of today", fontsize=6, color=S.GR, rotation=90, va="bottom", ha="left")
for zz in (0.0, 0.3, 1.0):
    v = K.fs8_deficit(zz); a1.plot(zz, v, "o", color=S.DATA, ms=3.5)
    a1.annotate(f"{v:.2f} %", (zz, v), xytext=(5, 3), textcoords="offset points", fontsize=7, color=S.DATA)
a1.set_xlim(0, 2.5); a1.set_ylim(0, 14.5)
a1.set_xlabel("redshift $z$"); a1.set_ylabel("per cent below $\\Lambda$CDM")
a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), handlelength=1.8)
a1.set_title("The growth-rate ramp")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(z, EG, color=S.IAM, lw=1.6, label="$E_G$")
a2.plot(z, isw, color=S.ALT2, lw=1.4, label="ISW source $(1-f)D/a$")
a2.plot(z, -dD, color=S.ALT, lw=1.4, ls="--", label="$\\Delta\\Phi/\\Phi=\\Delta D/D$ ($\\Sigma=1$)")
a2.axhline(0, color=S.GR, lw=0.8, ls="--")
e3 = 100 * (K.f(K.LCDM, 1 / 1.3) / K.f(K.IAM, 1 / 1.3) - 1); a2.plot(0.3, e3, "o", color=S.IAM, ms=3.5)
a2.annotate(f"{e3:+.1f} % at $z$ = 0.3", (0.3, e3), xytext=(6, 4), textcoords="offset points", fontsize=7, color=S.IAM)
a2.set_xlim(0, 2.5); a2.set_ylim(-1.5, 4.5)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("change from $\\Lambda$CDM (%)")
a2.legend(loc="upper right", handlelength=1.8)
a2.set_title("What light sees")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_survey_ramp")

# ---------------- precision -----------------
mu0 = K.mu(1.0) - 1; Hm, Hg = 72.26, 67.16
sg = np.linspace(0.02, 0.25, 300)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
mu0eq = 0.072   # template-equivalent mu0 over 0 < z < 2 (docs/verification/scripts/verify_euclid_template_output.txt)
sg = np.linspace(0.008, 0.25, 400)
a1.plot(sg, mu0eq / sg, color=S.IAM, lw=1.6)
for s_, nm, dx in ((0.233, "conservative cuts", -6), (0.04, "WL + GCph to $k\\approx4$/Mpc", 7), (0.01, "all probes, ~1 %", 7)):
    a1.plot(s_, mu0eq / s_, "o", color=S.IAM, ms=4.5)
    a1.annotate(f"{nm}: {mu0eq/s_:.1f}$\\sigma$", (s_, mu0eq / s_), xytext=(dx, 6), textcoords="offset points", fontsize=7,
                ha="right" if dx < 0 else "left")
for lev in (1, 2, 3):
    a1.axhline(lev, color=S.LIGHT, lw=0.6, ls=":")
a1.set_xlim(0, 0.25); a1.set_ylim(0, 8)
a1.set_xlabel("Euclid error $\\sigma(\\mu_0)$ on the template"); a1.set_ylabel("separation of the IAM $\\mu(z)$ from GR ($\\sigma$)")
a1.set_title("Growth: published Euclid errors")
S.panel_letter(a1, "a", dx=-0.14)
sh = np.linspace(0.5, 12, 300)
a2.plot(sh, (Hm - Hg) / sh, color=S.ALT, lw=1.6)
a2.axhline(3, color=S.LIGHT, lw=0.6, ls=":")
s3 = (Hm - Hg) / 3; a2.plot(s3, 3, "o", color=S.ALT, ms=4.5)
a2.annotate(f"3$\\sigma$ needs $\\sigma(H_0)\\leq${s3:.2f}\n({100*s3/Hm:.1f} % of {Hm})", (s3, 3), xytext=(8, 6), textcoords="offset points", fontsize=7)
sgw = 10.0  # GW170817 alone: 70.0 +12.0 / -8.0, mean half-width
a2.plot(sgw, (Hm - Hg) / sgw, "s", color=S.DATA, ms=4.5)
a2.annotate("GW170817 alone\n(+12/$-$8)", (sgw, (Hm - Hg) / sgw), xytext=(-4, 9), textcoords="offset points", fontsize=7, color=S.DATA, ha="center")
a2.set_xlim(0, 12); a2.set_ylim(0, 6)
a2.set_xlabel("siren $\\sigma(H_0)$ (km s$^{-1}$ Mpc$^{-1}$)"); a2.set_ylabel(f"separation of {Hm} from {Hg} ($\\sigma$)")
a2.set_title("Expansion: the two rates by sirens")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_survey_precision")
