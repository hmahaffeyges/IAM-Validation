"""Part 2 chapters ch:darkenergy (p2_11), ch:wzfuture (p2_20) and ch:surveys (p2_16): figures added in the line-for-line carriage.
Equations and numbers: docs/verification/scripts/verify_dark_energy_far_future_surveys_book.py (output beside it).
fig_wz_eos_maturity_a : (a) w_info(a) = -1 - 1/(3a) against a, with w = -1 and today's -4/3; (b) the fraction written E(a)/e against a;
                        (c) the writing rate per unit a, E/(e a^2) (peak 4/e^2 at a = 1/2), and per e-fold, E/(e a) (peak 1/e at a = 1).
fig_two_rulers_future_l2 : photon-sector H(a) and matter-sector H_m(a) = sqrt(H^2 + beta_m E H0^2) with the Level 2 photon H0 = 67.16
                        (Om 0.3153); (b) their ratio.
fig_survey_transition : (a) E(a) against z with 10/50/90 % of today's value; (b) mu(z), exact and MGCAMB form, with the mu > 1 side
                        marked; (c) normalised turn-on (1 - mu)/(1 - mu0); (d) |dmu/dz|; (e) deviations of observables from LambdaCDM
                        (same early amplitude); (f) exact minus MGCAMB form, x 100.
fig_survey_isw        : (a) ISW source H D (1 - f) (Sigma = 1) for LambdaCDM, IAM exact and the MGCAMB form; (b) ratio to LambdaCDM;
                        (c) potential (Phi+Psi)(z)/(Phi+Psi)(z=3); (d) ISW-galaxy amplitude ratio against the mean redshift of a
                        Gaussian galaxy window (sigma_z = 0.1), with DESI BGS (0.3) and LRG (0.5, 0.7) marked.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import quad
import _bookstyle as S
import _cosmo as C
import matplotlib.pyplot as plt

S.apply()
E = lambda a: np.exp(1 - 1 / a)
M = lambda s: s.replace("-", "\u2212")

# ---------------------------------------------------------------- 1. w(a), maturity in a
fig, ax = plt.subplots(1, 3, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.42))
a = np.linspace(0.2, 5, 400)
ax[0].plot(a, -1 - 1 / (3 * a), color=S.IAM, lw=1.6, label="$w_{\\rm info}=-1-1/(3a)$")
ax[0].axhline(-1, color=S.GR, lw=0.8, ls="--", label="$\\Lambda$CDM, $w=-1$")
ax[0].plot(1, -4 / 3, "o", color=S.IAM, ms=4); ax[0].annotate(M("today, -4/3"), (1, -4 / 3), xytext=(6, -10), textcoords="offset points", fontsize=7, color=S.IAM)
ax[0].set_xlim(0, 5); ax[0].set_ylim(-2.8, -0.8); ax[0].set_xlabel("scale factor $a$"); ax[0].set_ylabel("$w$")
ax[0].legend(loc="lower right"); S.panel_letter(ax[0], "a", dx=-0.3)
a2 = np.linspace(0.05, 8, 600)
ax[1].plot(a2, E(a2) / np.e, color=S.IAM, lw=1.6); ax[1].axhline(1, color=S.GR, lw=0.8, ls="--")
ax[1].text(7.9, 0.96, "saturation, $E/e\\to1$", fontsize=7, color=S.GR, ha="right", va="top")
ax[1].plot(1, 1 / np.e, "o", color=S.IAM, ms=4); ax[1].annotate("today, $1/e$", (1, 1 / np.e), xytext=(6, -10), textcoords="offset points", fontsize=7, color=S.IAM)
ax[1].set_xlim(0, 8); ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("scale factor $a$"); ax[1].set_ylabel("fraction written, $E(a)/e$")
S.panel_letter(ax[1], "b", dx=-0.2)
ax[2].plot(a2, E(a2) / (np.e * a2**2), color=S.IAM, lw=1.6, label="per unit $a$: $E/(e\\,a^2)$")
ax[2].plot(a2, E(a2) / (np.e * a2), color=S.ALT, lw=1.4, ls="--", label="per e-fold: $E/(e\\,a)$")
ax[2].plot(0.5, 4 / np.e**2, "o", color=S.IAM, ms=3.5); ax[2].plot(1, 1 / np.e, "o", color=S.ALT, ms=3.5)
ax[2].annotate("$a=1/2$", (0.5, 4 / np.e**2), xytext=(6, 0), textcoords="offset points", fontsize=7, color=S.IAM)
ax[2].annotate("$a=1$", (1, 1 / np.e), xytext=(6, 2), textcoords="offset points", fontsize=7, color=S.ALT)
ax[2].set_xlim(0, 8); ax[2].set_ylim(0, 0.62); ax[2].set_xlabel("scale factor $a$"); ax[2].set_ylabel("writing rate")
ax[2].text(2.2, 0.06, "per unit $a$: $E/(e\\,a^2)$", fontsize=6.5, color=S.IAM); ax[2].text(3.0, 0.215, "per e-fold: $E/(e\\,a)$", fontsize=6.5, color=S.ALT)
S.panel_letter(ax[2], "c", dx=-0.2)
S.save(fig, "part2", "fig_wz_eos_maturity_a")

# ---------------------------------------------------------------- 2. two rulers, Level 2 photon H0
H0, Om, OL = 67.16, 0.3153, 0.6847; bm = Om / 2
a = np.logspace(np.log10(0.3), 2, 500)
H = H0 * np.sqrt(Om * a**-3 + OL); Hm = np.sqrt(H**2 + bm * E(a) * H0**2)
fig, ax = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5), gridspec_kw=dict(wspace=0.32))
ax[0].plot(a, H, color=S.GR, lw=1.4, label="photon sector $H(a)$")
ax[0].plot(a, Hm, color=S.IAM, lw=1.6, label="matter sector $H_m(a)$")
ax[0].axhline(H0 * np.sqrt(OL), color=S.GR, lw=0.6, ls=":"); ax[0].axhline(H0 * np.sqrt(OL + bm * np.e), color=S.IAM, lw=0.6, ls=":")
ax[0].axvline(1, color=S.LIGHT, lw=0.6, ls=":")
ax[0].set_xscale("log"); ax[0].set_xlim(0.3, 100); ax[0].set_ylim(40, 160)
ax[0].text(95, H0*np.sqrt(OL)-2, "55.57", fontsize=7, color=S.GR, ha="right", va="top"); ax[0].text(95, H0*np.sqrt(OL+bm*np.e)+2, "70.86", fontsize=7, color=S.IAM, ha="right", va="bottom")
ax[0].set_xlabel("scale factor $a$"); ax[0].set_ylabel("rate (km s$^{-1}$ Mpc$^{-1}$)"); ax[0].legend(loc="upper right")
S.panel_letter(ax[0], "a", dx=-0.16)
ax[1].plot(a, Hm / H, color=S.IAM, lw=1.6); ax[1].axhline(np.sqrt(1 + bm * np.e / OL), color=S.IAM, lw=0.6, ls=":")
for aa in (1, 2, 10):
    r = np.sqrt(Om * aa**-3 + OL + bm * E(aa)) / np.sqrt(Om * aa**-3 + OL)
    ax[1].plot(aa, r, "o", color=S.IAM, ms=3.5); ax[1].annotate(f"{r:.3f}", (aa, r), xytext=(4, -10), textcoords="offset points", fontsize=7, color=S.IAM)
ax[1].text(1.0, 1.282, "limit 1.275", fontsize=7, color=S.IAM, va="bottom")
ax[1].set_xscale("log"); ax[1].set_xlim(0.3, 100); ax[1].set_ylim(1.0, 1.31)
ax[1].set_xlabel("scale factor $a$"); ax[1].set_ylabel("$H_m/H$"); S.panel_letter(ax[1], "b", dx=-0.16)
S.save(fig, "part2", "fig_two_rulers_future_l2")

# ---------------------------------------------------------------- 3. transition zone
z = np.linspace(0, 3, 601); a = 1 / (1 + z)
mu = C.mu(a); mug = C.mu_mgcamb(a); mu0 = C.mu(1.0)
fig, ax = plt.subplots(2, 3, figsize=(S.TEXTW, 4.5), gridspec_kw=dict(wspace=0.42, hspace=0.55)); ax = ax.ravel()
zz = np.linspace(0, 4, 600)
ax[0].plot(zz, E(1 / (1 + zz)), color=S.IAM, lw=1.6)
for q, zq in ((0.1, 2.30), (0.5, 0.69), (0.9, 0.11)):
    ax[0].plot(zq, q, "o", color=S.IAM, ms=3.5); ax[0].annotate(f"{int(q*100)} %, $z$={zq:.2f}", (zq, q), xytext=(6, -1) if q != 0.9 else (6, 2), textcoords="offset points", fontsize=6.5, color=S.IAM)
ax[0].axvline(1, color=S.LIGHT, lw=0.6, ls=":"); ax[0].text(1.25, 0.62, "inflection in $a$\n($z=1$)", fontsize=6.5, color=S.GR, va="bottom")
ax[0].axvspan(0.06, 1.12, color=S.SKY, alpha=0.25, lw=0)
ax[0].set_xlim(0, 4); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("redshift $z$"); ax[0].set_ylabel("$E(a)$"); S.panel_letter(ax[0], "a", dx=-0.2)
ax[1].plot(z, mu, color=S.IAM, lw=1.6, label="IAM, exact"); ax[1].plot(z, mug, color=S.ALT, lw=1.2, ls="--", label="MGCAMB form")
ax[1].axhline(1, color=S.GR, lw=0.8); ax[1].axhspan(1, 1.1, color=S.LIGHT, alpha=0.35, lw=0)
ax[1].text(2.95, 1.05, "$\\mu>1$: $f(R)$, nDGP", fontsize=6.5, color=S.GR, ha="right", va="center")
ax[1].set_xlim(0, 3); ax[1].set_ylim(0.84, 1.1); ax[1].set_xlabel("redshift $z$"); ax[1].set_ylabel("$\\mu(z)$"); ax[1].legend(loc="lower right")
S.panel_letter(ax[1], "b", dx=-0.2)
ax[2].plot(z, (1 - mu) / (1 - mu0), color=S.IAM, lw=1.6, label="IAM, exact"); ax[2].plot(z, (1 - mug) / (1 - C.mu_mgcamb(1.0)), color=S.ALT, lw=1.2, ls="--", label="MGCAMB form")
for q in (0.1, 0.5, 0.9): ax[2].axhline(q, color=S.LIGHT, lw=0.5, ls=":")
ax[2].set_xlim(0, 3); ax[2].set_ylim(0, 1.02); ax[2].set_xlabel("redshift $z$"); ax[2].set_ylabel("$(1-\\mu)/(1-\\mu_0)$"); ax[2].legend(loc="upper right")
S.panel_letter(ax[2], "c", dx=-0.2)
ax[3].plot(z, np.abs(np.gradient(mu, z)), color=S.IAM, lw=1.6, label="IAM, exact"); ax[3].plot(z, np.abs(np.gradient(mug, z)), color=S.ALT, lw=1.2, ls="--", label="MGCAMB form")
ax[3].plot(0, abs(np.gradient(mu, z)[0]), "o", color=S.IAM, ms=3.5, clip_on=False); ax[3].annotate("largest at $z=0$: 0.229", (0, 0.229), xytext=(8, -4), textcoords="offset points", fontsize=6.5, color=S.IAM)
ax[3].set_xlim(0, 3); ax[3].set_ylim(0, 0.25); ax[3].set_xlabel("redshift $z$"); ax[3].set_ylabel("$|d\\mu/dz|$"); ax[3].legend(loc="upper right", bbox_to_anchor=(1.0, 0.85))
S.panel_letter(ax[3], "d", dx=-0.2)
zs = np.linspace(0, 2, 201)
ax[4].plot(zs, 100 * (C.mu(1 / (1 + zs)) - 1), color=S.GR, lw=1.2, label="$\\mu-1$")
ax[4].plot(zs, -C.fs8_deficit(zs), color=S.IAM, lw=1.6, label="$\\Delta(f\\sigma_8)/f\\sigma_8$")
ax[4].plot(zs, -C.amp_deficit(zs), color=S.ALT, lw=1.3, ls="--", label="$\\Delta D/D=\\Delta\\Phi/\\Phi$")
ax[4].axhline(0, color=S.LIGHT, lw=0.5); ax[4].set_xlim(0, 2); ax[4].set_ylim(-14.5, 0.8)
ax[4].set_xlabel("redshift $z$"); ax[4].set_ylabel("change from $\\Lambda$CDM (%)"); ax[4].legend(loc="lower right")
S.panel_letter(ax[4], "e", dx=-0.2)
ax[5].plot(z, 100 * (mu - mug), color=S.ALT2, lw=1.5); ax[5].axhline(0, color=S.LIGHT, lw=0.5)
i = np.argmax(np.abs(mu - mug)); ax[5].plot(z[i], 100 * (mu - mug)[i], "o", color=S.ALT2, ms=3.5)
ax[5].annotate(f"{100*(mu-mug)[i]:.2f} at $z$={z[i]:.2f}", (z[i], 100 * (mu - mug)[i]), xytext=(8, 3), textcoords="offset points", fontsize=6.5, color=S.ALT2)
ax[5].set_xlim(0, 3); ax[5].set_ylim(-0.5, 3.2); ax[5].set_xlabel("redshift $z$"); ax[5].set_ylabel("$[\\mu_{\\rm exact}-\\mu_{\\rm MGCAMB}]\\times100$")
S.panel_letter(ax[5], "f", dx=-0.2)
S.save(fig, "part2", "fig_survey_transition")

# ---------------------------------------------------------------- 4. ISW
def src(Sx, aa): return np.sqrt(C.H2(aa)) * C.D(Sx, aa) * (1 - C.f(Sx, aa))
z = np.linspace(0, 2, 401); a = 1 / (1 + z)
sL, sI, sM = src(C.LCDM, a), src(C.IAM, a), src(C.MGC, a)
fig, ax = plt.subplots(2, 2, figsize=(S.TEXTW, 4.3), gridspec_kw=dict(wspace=0.32, hspace=0.5)); ax = ax.ravel()
nrm = sL.max()
ax[0].plot(z, sL / nrm, color=S.GR, lw=1.3, label="$\\Lambda$CDM"); ax[0].plot(z, sI / nrm, color=S.IAM, lw=1.6, label="IAM, exact")
ax[0].plot(z, sM / nrm, color=S.ALT, lw=1.1, ls="--", label="MGCAMB form")
ax[0].set_xlim(0, 2); ax[0].set_ylim(0, 1.1); ax[0].set_xticks([0, 0.5, 1, 1.5, 2]); ax[0].set_xlabel("redshift $z$"); ax[0].set_ylabel("ISW source $HD(1-f)$ (rel.)"); ax[0].legend(loc="upper right")
S.panel_letter(ax[0], "a", dx=-0.2)
ax[1].plot(z, sI / sL, color=S.IAM, lw=1.6, label="IAM, exact"); ax[1].plot(z, sM / sL, color=S.ALT, lw=1.1, ls="--", label="MGCAMB form")
ax[1].axhline(1, color=S.GR, lw=0.8); ax[1].set_xlim(0, 2); ax[1].set_ylim(0.99, 1.10); ax[1].set_xticks([0, 0.5, 1, 1.5, 2])
ax[1].set_xlabel("redshift $z$"); ax[1].set_ylabel("source ratio to $\\Lambda$CDM"); ax[1].legend(loc="upper left")
S.panel_letter(ax[1], "b", dx=-0.14)
z3 = np.linspace(0, 3, 301); a3 = 1 / (1 + z3)
pot = lambda Sx: (C.D(Sx, a3) / a3) / (C.D(Sx, 0.25) / 0.25)
ax[2].plot(z3, pot(C.LCDM), color=S.GR, lw=1.3, label="$\\Lambda$CDM"); ax[2].plot(z3, pot(C.IAM), color=S.IAM, lw=1.6, ls="--", label="IAM, exact")
pL5 = (C.D(C.LCDM, 1 / 1.5) * 1.5) / (C.D(C.LCDM, 0.25) * 4); pI5 = (C.D(C.IAM, 1 / 1.5) * 1.5) / (C.D(C.IAM, 0.25) * 4)
ax[2].annotate(M(f"z = 0.5: {100*(pI5/pL5-1):.2f} %"), (0.5, pI5), xytext=(12, -16), textcoords="offset points", fontsize=6.5, color=S.IAM,
               arrowprops=dict(arrowstyle="-", lw=0.5, color=S.IAM))
ax[2].set_xlim(0, 3); ax[2].set_ylim(0.7, 1.02); ax[2].set_xlabel("redshift $z$"); ax[2].set_ylabel("$(\\Phi+\\Psi)(z)/(\\Phi+\\Psi)(z{=}3)$"); ax[2].legend(loc="lower right")
S.panel_letter(ax[2], "c", dx=-0.14)
def amp(Sx, zc, sig=0.1):
    g = lambda q: np.exp(-0.5 * ((q - zc) / sig)**2) * src(Sx, 1 / (1 + q))
    return quad(g, max(0.0, zc - 5 * sig), zc + 5 * sig)[0]
zc = np.linspace(0.15, 1.8, 34)
ax[3].plot(zc, [amp(C.IAM, q) / amp(C.LCDM, q) for q in zc], color=S.IAM, lw=1.6, label="IAM, exact")
ax[3].plot(zc, [amp(C.MGC, q) / amp(C.LCDM, q) for q in zc], color=S.ALT, lw=1.1, ls="--", label="MGCAMB form")
for nm, q in (("BGS", 0.3), ("LRG", 0.5), ("LRG", 0.7)):
    r = amp(C.IAM, q) / amp(C.LCDM, q); ax[3].plot(q, r, "o", color=S.IAM, ms=3.5)
    ax[3].annotate(f"{nm} {r:.3f}", (q, r), xytext={"0.3": (-14, -14), "0.5": (-4, 8), "0.7": (4, -14)}[str(q)], textcoords="offset points", fontsize=6.5, color=S.IAM)
ax[3].axhline(1, color=S.GR, lw=0.8); ax[3].set_xlim(0, 1.9); ax[3].set_xticks([0, 0.5, 1, 1.5]); ax[3].set_ylim(0.99, 1.10)
ax[3].set_xlabel("mean redshift of galaxy sample"); ax[3].set_ylabel("$A_{\\rm ISW}^{\\rm IAM}/A_{\\rm ISW}^{\\Lambda\\rm CDM}$"); ax[3].legend(loc="upper left")
S.panel_letter(ax[3], "d", dx=-0.14)
S.save(fig, "part2", "fig_survey_isw")
