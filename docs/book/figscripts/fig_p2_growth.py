"""Part 2, Chapters 'Growth across redshift: the S8 trend' (p2_08_s8_trend.tex) and 'Dark energy or two rulers?' (p2_09_sector_tension.tex).
Linear growth with the same early amplitude (_cosmo.py; same equations as verify_s8_trend.py and verify_sector_tension.py).
fig_s8_inferred: (a) S8 a survey at z infers with LambdaCDM growth, S8_Planck x D_IAM/D_LCDM, exact coupling and the MGCAMB form of the
Level 1 chains; (b) effective growth index gamma = ln f/ln Omega_m(a), against LambdaCDM and the measured 0.633 (+0.025/-0.024).
fig_growth_signatures: (a) ISW source (1 - f) D, ratio to LambdaCDM; (b) M_lens/M_dyn = 1/mu in the linear coupling.
fig_mu_fsigma8: mu(z), the coupling deficit 1 - mu and the f sigma8 deficit on one panel (the coupling deficit is not the growth deficit).
fig_w0wa: DESI DR2 w0-wa (arXiv:2503.14738) against LambdaCDM and the two-ruler mock (TWO_RULER_DESI_TEST.md)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import _bookstyle as S, _cosmo as K
import matplotlib.pyplot as plt

S8P = 0.832
z = np.linspace(0.0, 2.0, 300); a = 1 / (1 + z)
S8e = S8P * K.D(K.IAM, a) / K.D(K.LCDM, a); S8m = S8P * K.D(K.MGC, a) / K.D(K.LCDM, a)
gL = np.log(K.f(K.LCDM, a)) / np.log(K.Oma(a)); gI = np.log(K.f(K.IAM, a)) / np.log(K.Oma(a))
for q in (0, 0.25, 0.5, 1.0, 2.0):
    aa = 1 / (1 + q)
    print(f"z {q}: S8 {S8P*K.D(K.IAM,aa)/K.D(K.LCDM,aa):.4f} (deficit {K.amp_deficit(q):.2f} %, MGCAMB form {K.amp_deficit(q, K.MGC):.2f} %)"
          f"  gamma IAM {np.log(K.f(K.IAM,aa))/np.log(K.Oma(aa)):.3f} LCDM {np.log(K.f(K.LCDM,aa))/np.log(K.Oma(aa)):.3f}")
S.apply()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.axhline(S8P, color=S.GR, lw=0.9, ls="--"); a1.text(1.97, S8P + 0.0002, "Planck $\\Lambda$CDM, 0.832", fontsize=7, ha="right", va="bottom", color=S.GR)
a1.plot(z, S8e, color=S.IAM, lw=1.6); a1.plot(z, S8m, color=S.ALT, lw=1.2, ls="-.")
a1.text(0.6, 0.8298, "exact coupling", fontsize=7, color=S.IAM, va="top")
a1.text(0.45, 0.8185, "MGCAMB form (Level 1 chains)", fontsize=7, color=S.ALT)
a1.plot(0, S8e[0], "o", color=S.IAM, ms=4); a1.annotate(f"{S8e[0]:.4f}", (0, S8e[0]), xytext=(6, -4), textcoords="offset points", fontsize=7, va="top")
a1.set_xlim(-0.03, 2); a1.set_ylim(0.816, 0.834)
a1.set_xlabel("survey redshift $z$"); a1.set_ylabel("inferred $S_8$")
a1.set_title("Lowest today, recovered by $z\\approx1$")
S.panel_letter(a1, "a", dx=-0.2)
a2.axhspan(0.633 - 0.024, 0.633 + 0.025, color=S.DATA, alpha=0.18, lw=0)
a2.axhline(0.633, color=S.DATA, lw=0.8); a2.text(1.97, 0.636, "measured, 0.633", fontsize=7, color=S.DATA, ha="right", va="bottom")
a2.plot(z, gL, color=S.GR, lw=1.2, ls="--"); a2.plot(z, gI, color=S.IAM, lw=1.6)
a2.text(1.97, gL[-1] - 0.004, "$\\Lambda$CDM", fontsize=7, color=S.GR, ha="right", va="top")
a2.text(0.05, gI[0] + 0.006, f"informational term, {gI[0]:.3f} today", fontsize=7, color=S.IAM, va="bottom")
a2.set_xlim(0, 2); a2.set_ylim(0.53, 0.67)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("growth index $\\gamma$")
a2.set_title("About 40 % of the measured $\\gamma$ shift")
S.panel_letter(a2, "b", dx=-0.2)
S.save(fig, "part2", "fig_s8_inferred")

isw = ((1 - K.f(K.IAM, a)) * K.D(K.IAM, a)) / ((1 - K.f(K.LCDM, a)) * K.D(K.LCDM, a))
for q in (0.1, 0.5, 1.0):
    aa = 1 / (1 + q); print(f"ISW source ratio z {q}: {((1-K.f(K.IAM,aa))*K.D(K.IAM,aa))/((1-K.f(K.LCDM,aa))*K.D(K.LCDM,aa)):.4f};  1/mu {1/K.mu(aa):.3f}")
print("1/mu today", round(1 / K.mu(1.0), 3))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.33))
zz = z[z >= 0.05]; aa = 1 / (1 + zz)
a1.plot(zz, ((1 - K.f(K.IAM, aa)) * K.D(K.IAM, aa)) / ((1 - K.f(K.LCDM, aa)) * K.D(K.LCDM, aa)), color=S.IAM, lw=1.6)
a1.axhline(1, color=S.GR, lw=0.8, ls="--"); a1.text(1.95, 1.001, "$\\Lambda$CDM", fontsize=7, color=S.GR, ha="right", va="bottom")
a1.set_xlim(0, 2); a1.set_ylim(0.995, 1.045)
a1.set_xlabel("redshift $z$"); a1.set_ylabel("ISW source $(1-f)D$, ratio")
a1.set_title("Potentials decay faster: ISW larger")
S.panel_letter(a1, "a", dx=-0.2)
a2.plot(z, 1 / K.mu(a), color=S.IAM, lw=1.6); a2.axhline(1, color=S.GR, lw=0.8, ls="--")
for q in (0, 0.5, 1.0):
    v = 1 / K.mu(1 / (1 + q)); a2.plot(q, v, "o", color=S.IAM, ms=4)
    a2.annotate(f"{v:.2f}", (q, v), xytext=(6, 2), textcoords="offset points", fontsize=7)
a2.set_xlim(-0.04, 2); a2.set_ylim(0.98, 1.2)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("$M_{\\rm lens}/M_{\\rm dyn}=1/\\mu$")
a2.set_title("Cluster masses, linear coupling")
S.panel_letter(a2, "b", dx=-0.2)
S.save(fig, "part2", "fig_growth_signatures")

# ---------------- required figure: mu, 1 - mu and the f sigma8 deficit -----------------
z = np.linspace(0, 2, 400); a = 1 / (1 + z)
one_mu = 100 * (1 - K.mu(a)); fs8 = K.fs8_deficit(z)
marks = (0.0, 0.3, 1.0)
for q in marks:
    print(f"z {q}: 1-mu {100*(1-K.mu(1/(1+q))):.2f} %  f sigma8 deficit {float(K.fs8_deficit(q)):.2f} %")
tr = (("BGS", 0.295), ("LRG1", 0.510), ("LRG2", 0.706), ("LRG3", 0.919), ("ELG2", 1.317), ("QSO", 1.491))
fig, ax = plt.subplots(figsize=(0.8 * S.TEXTW, 3.0))
ax.plot(z, one_mu, color=S.GR, lw=1.2, ls="--")
ax.plot(z, fs8, color=S.IAM, lw=1.8)
for q in marks:
    v = float(K.fs8_deficit(q)); ax.plot(q, v, "o", color=S.IAM, ms=4.5, zorder=4)
    ax.annotate(f"{v:.2f} %", (q, v), xytext=(7, 4), textcoords="offset points", fontsize=7, color=S.IAM)
ax.text(0.10, 13.5, "coupling deficit $1-\\mu$", fontsize=7, color=S.GR, va="center")
ax.annotate("$f\\sigma_8$ deficit\n(linear growth,\nsame early amplitude)", (0.85, float(K.fs8_deficit(0.85))), xytext=(1.25, 4.2), fontsize=7, color=S.IAM, arrowprops=dict(arrowstyle="-", lw=0.5, color=S.IAM))
for nm, q in tr:
    ax.plot([q, q], [-1.7, -0.9], color=S.DATA, lw=0.8)
ax.text(1.55, -1.3, "DESI $z$", fontsize=7, color=S.DATA, ha="left", va="center")
ax.set_xlim(-0.03, 2); ax.set_ylim(-2.2, 15)
ax.set_xlabel("redshift $z$"); ax.set_ylabel("deficit relative to $\\Lambda$CDM (%)")
axr = ax.twinx(); axr.spines["right"].set_visible(True)
axr.plot(z, K.mu(a), color=S.ALT, lw=1.0, ls=":")
axr.set_ylim(1 - 0.15 / 100 * 100 * (15 / 15) , 1.0 + 1.2 / 100)
axr.set_ylim(0.70, 1.005)
axr.set_ylabel("$\\mu(z)$ (dotted)", color=S.ALT); axr.tick_params(axis="y", colors=S.ALT)
ax.set_title("A 13.6 % weaker coupling today gives a 4.25 % lower $f\\sigma_8$")
S.save(fig, "part2", "fig_mu_fsigma8")

# ---------------- w0 wa -----------------
desi = [("DESI + CMB", -0.42, 0.21, 0.21, -1.75, 0.58, 0.58), ("DESI + CMB + Pantheon+", -0.838, 0.055, 0.055, -0.62, 0.19, 0.22),
        ("DESI + CMB + Union3", -0.667, 0.088, 0.088, -1.09, 0.27, 0.31), ("DESI + CMB + DES Y5", -0.752, 0.057, 0.057, -0.86, 0.20, 0.23)]
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.9))
ax.axvline(-1, color=S.LIGHT, lw=0.8, ls=":"); ax.axhline(0, color=S.LIGHT, lw=0.8, ls=":")
for i, (nm, w0, e0m, e0p, wa, eam, eap) in enumerate(desi):
    ax.errorbar(w0, wa, xerr=[[e0m], [e0p]], yerr=[[eam], [eap]], fmt="s", color=S.DATA, ms=4, capsize=1.5, lw=0.8)
    pos = [(-0.40, -2.35), (-0.62, -0.45), (-0.55, -1.30), (-0.58, -0.85)][i]
    ax.annotate(nm, (w0, wa), xytext=pos, textcoords="data", fontsize=7, color=S.DATA, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.DATA, shrinkA=0, shrinkB=3))
ax.plot(-1, 0, "o", color=S.GR, ms=6); ax.annotate("$\\Lambda$CDM; photon-ruler\nprediction of the term", (-1, 0), xytext=(8, 6), textcoords="offset points", fontsize=7, ha="left", va="bottom", color=S.GR)
ax.plot(-1.19, 0.38, "D", color=S.IAM, ms=5); ax.annotate("two-ruler mock", (-1.19, 0.38), xytext=(-2, 8), textcoords="offset points", fontsize=7, color=S.IAM)
ax.set_xlim(-1.45, -0.1); ax.set_ylim(-2.5, 1.0)
ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_a$")
ax.set_title("The mock lands opposite DESI's quadrant")
S.save(fig, "part2", "fig_w0wa")
