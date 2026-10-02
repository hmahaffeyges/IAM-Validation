"""Part 2, Chapters 'Lensing mass and dynamical mass' (p2_17_lensing_dynamics.tex) and 'Three cluster masses' (p2_18_three_way_clusters.tex).
Equations as docs/verification/scripts/verify_obs_chapters.py section E; chains read with _chains.py (30 % burn-in, weighted).
fig_lensdyn_forms: (a) M_lens/M_dyn in the two implementation forms: 1/mu(z) in the Level 1 effective-coupling form, 1 in the Level 2
  form, with the quasi-static f(R) range 3/4 <= M_lens/M_dyn <= 1 (1 <= mu <= 4/3, Sigma = 1) and the four test values of the chapter;
  (b) sigma8 posteriors, LambdaCDM against the informational term, Level 1 (Planck, MGCAMB) and Level 2 (modified CAMB) chains.
fig_threeway_slope: (a) the gravitational part R = 1/mu (Level 1 form), the illustrative non-thermal factor C_NT = 1 + 0.20 (1+z)^0.2
  and their product, at the four bin centres; (b) the redshift slopes of the three curves.
fig_lensdyn_test and fig_threeway_estimators: described at their blocks below (precision needed by the redshift-shape tests).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import _bookstyle as S
import _cosmo as K
import _chains as Ch
import matplotlib.pyplot as plt

S.apply()
R = lambda z: 1 / K.mu(1 / (1 + np.asarray(z, float)))
z = np.linspace(0, 2.2, 221)
# chains
post = {}
for nm, (fl, fi) in (("Level 1", (Ch.L1["Planck"][0], Ch.L1["Planck"][1])), ("Level 2", (Ch.L2["C"], Ch.L2["A"]))):
    X0, X1 = Ch.load(*fl), Ch.load(*fi)
    post[nm] = [(X.sigma8.values, X.weight.values) for X in (X0, X1)]
    m0, s0 = Ch.wmean_sd(*post[nm][0]); m1, s1 = Ch.wmean_sd(*post[nm][1])
    print(f"{nm}: sigma8 LCDM {m0:.4f} +/- {s0:.4f}; IAM {m1:.4f} +/- {s1:.4f}; shift {100*(m1/m0-1):+.2f} %")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
a1.axhspan(0.75, 1.0, color=S.ALT, alpha=0.12, lw=0)
a1.text(2.15, 0.985, "$f(R)$, quasi-static: $\\leq1$", fontsize=7, color=S.ALT, ha="right", va="top")
a1.plot(z, R(z), color=S.IAM, lw=1.6, label="Level 1 form: $1/\\mu(z)$")
a1.axhline(1, color=S.GR, lw=1.2, ls="--", label="Level 2 form: 1")
for zt in (0.2, 0.5, 1.0, 2.0):
    a1.plot(zt, R(zt), "o", color=S.IAM, ms=4)
    a1.annotate(f"{R(zt):.3f}", (zt, R(zt)), xytext=(4, 4), textcoords="offset points", fontsize=7, color=S.IAM)
a1.annotate(f"{R(0):.3f} today", (0, R(0)), xytext=(6, 0), textcoords="offset points", fontsize=7, color=S.IAM, va="center")
a1.set_xlim(0, 2.2); a1.set_ylim(0.94, 1.18)
a1.set_xlabel("cluster redshift $z$"); a1.set_ylabel("$M_{\\rm lens}/M_{\\rm dyn}$")
a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))
a1.set_title("The ratio depends on the form")
S.panel_letter(a1, "a", dx=-0.14)
bins = np.linspace(0.76, 0.85, 61)
for (nm, ls) in (("Level 1", "-"), ("Level 2", "--")):
    for (v, w), col, lab in zip(post[nm], (S.GR, S.IAM), ("$\\Lambda$CDM", "IAM")):
        h, e = np.histogram(v, bins=bins, weights=w, density=True)
        a2.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=col, lw=1.2, ls=ls, label=f"{nm}, {lab}")
a2.set_xlim(0.765, 0.845); a2.set_ylim(0, 1.55 * a2.get_ylim()[1])
a2.set_xlabel("$\\sigma_8$"); a2.set_ylabel("posterior density")
a2.legend(loc="upper left", fontsize=6, handlelength=2.2, ncol=2, columnspacing=1.0)
a2.set_title("$\\sigma_8$ in the chains")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_lensdyn_forms")

# ---------------- three-way slope -----------------
C = lambda z: 1 + 0.20 * (1 + np.asarray(z, float))**0.2
zc = np.array([0.15, 0.25, 0.40, 0.65]); z2 = np.linspace(0.05, 1.0, 191); h = 1e-4
d = lambda F, zz: (F(zz + h) - F(zz - h)) / (2 * h)
P = lambda zz: R(zz) * C(zz)
for zz in zc:
    print(f"z {zz:.3f}: R {R(zz):.3f} C_NT {C(zz):.4f} product {P(zz):.3f}; slopes {d(R,zz):+.3f} {d(C,zz):+.3f} {d(P,zz):+.3f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
for F, col, ls, lab in ((R, S.IAM, "-", "gravitational part $R=1/\\mu$"), (C, S.GOLD, "--", "non-thermal $C_{\\rm NT}$ (illustrative)"),
                        (P, S.ALT2, "-", "product $R\\times C_{\\rm NT}$")):
    a1.plot(z2, F(z2), color=col, lw=1.5, ls=ls, label=lab); a1.plot(zc, F(zc), "o", color=col, ms=3.5)
a1.set_xlim(0, 1.0); a1.set_ylim(1.0, 1.50)
a1.set_xlabel("cluster redshift $z$"); a1.set_ylabel("ratio to the true mass")
a1.legend(loc="upper right")
a1.set_title("Two effects, Level 1 form")
S.panel_letter(a1, "a", dx=-0.14)
for F, col, ls, lab in ((R, S.IAM, "-", "$dR/dz$"), (C, S.GOLD, "--", "$dC_{\\rm NT}/dz$"), (P, S.ALT2, "-", "$d(R\\,C_{\\rm NT})/dz$")):
    a2.plot(z2, d(F, z2), color=col, lw=1.5, ls=ls, label=lab)
a2.axhline(0, color=S.GR, lw=0.8)
a2.plot(0.3, d(R, 0.3), "o", color=S.IAM, ms=4)
a2.annotate(f"$-${abs(d(R,0.3)):.3f} at $z$ = 0.3", (0.3, d(R, 0.3)), xytext=(8, -2), textcoords="offset points", fontsize=7, color=S.IAM, va="top")
a2.set_xlim(0, 1.0); a2.set_ylim(-0.32, 0.08)
a2.set_xlabel("cluster redshift $z$"); a2.set_ylabel("slope per unit redshift")
a2.legend(loc="lower right")
a2.set_title("The sign of the slope is the test")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_threeway_slope")

# ---------------- p2_17: what the five-bin test needs -----------------
# fig_lensdyn_test: (a) the slope dR/dz of the Level 1 curve; (b) significance with which five redshift bins (centres 0.2, 0.5, 0.8, 1.2,
# 1.8 inside 0.1 < z < 2, equal fractional error per bin) separate the parameter-free curve from the best constant, sqrt(chi2_const - 0).
zb = np.array([0.2, 0.5, 0.8, 1.2, 1.8]); Rb = R(zb)
sig = np.linspace(0.002, 0.05, 300)
dchi = np.array([np.sum((Rb - Rb.mean())**2) / s_**2 for s_ in sig])     # best constant under equal errors is the mean
s3 = np.sqrt(np.sum((Rb - Rb.mean())**2) / 9)
print(f"five bins: R {np.round(Rb,3)}; 3 sigma separation from a constant needs sigma per bin <= {s3:.4f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
zz = np.linspace(0, 2, 201); dRz = np.array([(R(q + h) - R(max(q - h, 0))) / (q + h - max(q - h, 0)) for q in zz])
a1.plot(zz, dRz, color=S.IAM, lw=1.6); a1.axhline(0, color=S.GR, lw=0.8, ls="--")
for q in (0.0, 0.3):
    v = (R(q + h) - R(max(q - h, 0))) / (q + h - max(q - h, 0)); a1.plot(q, v, "o", color=S.IAM, ms=4)
    a1.annotate(f"$-${abs(v):.2f} at $z$ = {q:g}", (q, v), xytext=(8, -2), textcoords="offset points", fontsize=7, color=S.IAM, va="top")
a1.text(1.95, 0.012, "hydrostatic bias: slope $\\geq0$", fontsize=7, color=S.GR, ha="right", va="bottom")
a1.set_xlim(0, 2); a1.set_ylim(-0.35, 0.06)
a1.set_xlabel("cluster redshift $z$"); a1.set_ylabel("$dR/dz$ (Level 1 form)")
a1.set_title("The ratio falls with redshift")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(100 * sig, np.sqrt(dchi), color=S.IAM, lw=1.6)
for lev in (1, 2, 3): a2.axhline(lev, color=S.LIGHT, lw=0.6, ls=":")
a2.plot(100 * s3, 3, "o", color=S.IAM, ms=4.5)
a2.annotate(f"3$\\sigma$ needs {100*s3:.1f} % per bin", (100 * s3, 3), xytext=(8, 4), textcoords="offset points", fontsize=7)
a2.set_xlim(0, 5); a2.set_ylim(0, 8)
a2.set_xlabel("error on $M_{\\rm lens}/M_{\\rm dyn}$ per bin (%)"); a2.set_ylabel("curve against a constant ($\\sigma$)")
a2.set_title("Five bins in $0.1<z<2$")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_lensdyn_test")

# ---------------- p2_18: three estimators and the slope test -----------------
# fig_threeway_estimators: (a) M/M_true for the three estimators in the Level 1 form (X-ray hydrostatic = mu, SZ = X-ray through the Y-M
# calibration, lensing = 1) and in the Level 2 form (all 1); (b) significance of a non-zero slope of R x C_NT fitted as a straight line
# over the four bin centres, against the per-bin fractional error (equal errors).
zl = np.linspace(0, 1.0, 201)
slope_P = np.polyfit(zc, P(zc), 1)[0]; Sxx = np.sum((zc - zc.mean())**2)
sigs = np.linspace(0.002, 0.05, 300); sig_slope = lambda s_: s_ * P(zc).mean() / np.sqrt(Sxx)
s3b = abs(slope_P) * np.sqrt(Sxx) / (3 * P(zc).mean())
print(f"product slope over the four bins {slope_P:+.3f}; 3 sigma needs per-bin fractional error <= {s3b:.4f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(zl, K.mu(1 / (1 + zl)), color=S.DATA, lw=1.6, label="X-ray hydrostatic: $\\mu$")
a1.plot(zl, K.mu(1 / (1 + zl)), color=S.GOLD, lw=1.2, ls=(0, (4, 3)), label="SZ (calibrated on X-ray)")
a1.plot(zl, np.ones_like(zl), color=S.IAM, lw=1.6, label="weak lensing: $\\Sigma=1$")
a1.axhline(1.0, color=S.GR, lw=0.6, ls=":")
a1.text(0.98, 1.004, "Level 2 form: all three at 1", fontsize=7, color=S.GR, ha="right", va="bottom")
a1.set_xlim(0, 1.0); a1.set_ylim(0.84, 1.03)
a1.set_xlabel("cluster redshift $z$"); a1.set_ylabel("$M_{\\rm est}/M_{\\rm true}$, Level 1 form")
a1.legend(loc="lower right", fontsize=6.5)
a1.set_title("Three estimators, two potentials")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(100 * sigs, abs(slope_P) / sig_slope(sigs), color=S.ALT2, lw=1.6)
for lev in (1, 2, 3): a2.axhline(lev, color=S.LIGHT, lw=0.6, ls=":")
a2.plot(100 * s3b, 3, "o", color=S.ALT2, ms=4.5)
a2.annotate(f"3$\\sigma$ needs {100*s3b:.1f} % per bin", (100 * s3b, 3), xytext=(8, 4), textcoords="offset points", fontsize=7)
a2.set_xlim(0, 5); a2.set_ylim(0, 8)
a2.set_xlabel("error on $M_{\\rm lens}/M_{\\rm hydro}$ per bin (%)"); a2.set_ylabel("slope of $R\\times C_{\\rm NT}$ against zero ($\\sigma$)")
a2.set_title("The four bins of $0.1<z<0.8$")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_threeway_estimators")
