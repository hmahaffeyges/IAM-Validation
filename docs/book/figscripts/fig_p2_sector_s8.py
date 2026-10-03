"""Figures for Chapter 'Growth across redshift: the S8 trend' (p2_08) and 'Dark energy or two rulers?' (p2_09, p2_09b).
fig_mu_evolution: (a) mu(z) at the DESI DR1 tracer redshifts; (b) the coupling deficit 1 - mu and the f sigma8 deficit, with the DESI DR2
  phantom-crossing redshifts (arXiv:2503.14738).
fig_sector_phantom: (a) f sigma8(z), IAM (sigma8 0.7998) and LambdaCDM (0.8087), with DESI DR1 ShapeFit-only f sigma_s8 (DESI 2024 V, App. A)
  and SDSS (eBOSS DR16 Table III, 6dFGS); (b) D_V/r_d, photon ruler (LambdaCDM = IAM), with DESI DR1 ShapeFit-only D_V/r_d (App. A);
  (c) w0-wa plane: DR2 fits, LambdaCDM, the two-ruler mock (TWO_RULER_DESI_TEST.md), z_cross map.
fig_s8_trend_iam: (a) E(a) = exp(1 - 1/a); (b) the S8 a LambdaCDM analysis infers: lensing amplitude, f sigma8 at z, and a fit to the
  13 f sigma8 points above z_min (verify_s8_trend.py, section 8).
Same equations as docs/verification/scripts/verify_sector_tension.py and verify_s8_trend.py."""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import _bookstyle as S
import _cosmo as C
S.apply()
zz = np.linspace(0, 3, 400); aa = 1 / (1 + zz)
tr = {"BGS": 0.295, "LRG1": 0.510, "LRG2": 0.706, "LRG3": 0.919, "ELG2": 1.317, "QSO": 1.491}
zx = {"DESI+CMB": 0.50, "+Pantheon+": 0.35, "+Union3": 0.44, "+DES Y5": 0.41}

# ---------------- fig_mu_evolution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5))
ax1.plot(zz, C.mu(aa), color=S.IAM, label=r"$\mu(z)=H^2/(H^2+\beta_mE(a)H_0^2)$")
ax1.axhline(1, color=S.GR, ls="--", lw=0.8, label="general relativity")
for k, z in tr.items():
    ax1.plot(z, C.mu(1 / (1 + z)), "o", ms=3.5, color=S.DATA)
    ax1.annotate(k, (z, C.mu(1 / (1 + z))), xytext=(3, 4) if k == "QSO" else (3, -9), textcoords="offset points", fontsize=6)
ax1.set_xlabel("redshift $z$"); ax1.set_ylabel(r"coupling $\mu$"); ax1.set_xlim(0, 3); ax1.set_ylim(0.85, 1.015)
ax1.legend(loc="lower right"); S.panel_letter(ax1, "a")
sec = ax1.secondary_xaxis("top", functions=(lambda z: 1 / (1 + np.asarray(z, float)), lambda a: 1 / np.clip(np.asarray(a, float), 1e-3, None) - 1)); sec.set_xticks([1, 0.7, 0.5, 0.4, 0.3]); sec.set_xlabel("scale factor $a$", fontsize=7)
ax2.plot(zz, 100 * (1 - C.mu(aa)), color=S.IAM, ls="--", label=r"coupling deficit $1-\mu$")
ax2.plot(zz, C.fs8_deficit(zz), color=S.IAM, label=r"$f\sigma_8$ deficit")
for k, z in zx.items():
    ax2.axvline(z, color=S.LIGHT, lw=0.7)
ax2.text(0.52, 12.2, "DR2 crossing\nredshifts 0.35-0.50", fontsize=6, color=S.GR)
for k, z in tr.items():
    ax2.plot(z, C.fs8_deficit(z), "o", ms=3, color=S.DATA)
ax2.set_xlabel("redshift $z$"); ax2.set_ylabel(r"deficit against $\Lambda$CDM (%)"); ax2.set_xlim(0, 2); ax2.set_ylim(0, 14.5)
ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78)); S.panel_letter(ax2, "b")
fig.tight_layout(); S.save(fig, "part2", "fig_mu_evolution")

# ---------------- fig_sector_phantom
dsv = {"BGS": (0.295, 0.377174, 88.510877, 7.788174, 1314.664401), "LRG1": (0.510, 0.513635, 41.295470, 12.514437, 541.309833),
       "LRG2": (0.706, 0.483623, 28.119682, 15.675560, 762.457717), "LRG3": (0.919, 0.422164, 22.370314, 19.676985, 847.499793),
       "ELG2": (1.317, 0.376715, 13.997473, 23.861806, 2342.506886), "QSO": (1.491, 0.434858, 19.785658, 25.708520, 3013.788566)}
leg = [(0.067, 0.423, 0.055), (0.15, 0.53, 0.16), (0.38, 0.500, 0.047), (0.51, 0.455, 0.039), (0.70, 0.448, 0.043), (0.85, 0.315, 0.095), (1.48, 0.462, 0.045)]
z2 = np.linspace(0.01, 2, 200); a2 = 1 / (1 + z2)
fsI = C.f(C.IAM, a2) * C.D(C.IAM, a2) / C.D(C.IAM, 1.0) * 0.7998
fsL = C.f(C.LCDM, a2) * C.D(C.LCDM, a2) / C.D(C.LCDM, 1.0) * 0.8087
fig, ax = plt.subplots(1, 3, figsize=(S.TEXTW, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.05]})
ax[0].plot(z2, fsL, color=S.GR, ls="--", label=r"$\Lambda$CDM ($\sigma_8$ 0.8087)")
ax[0].plot(z2, fsI, color=S.IAM, label=r"informational term (0.7998)")
ax[0].errorbar([v[0] for v in leg], [v[1] for v in leg], [v[2] for v in leg], fmt="o", ms=2.5, color=S.LIGHT, mec=S.GR, elinewidth=0.6, label="SDSS, 6dFGS")
ax[0].errorbar([v[0] for v in dsv.values()], [v[1] for v in dsv.values()], [np.sqrt(v[2] * 1e-4) for v in dsv.values()], fmt="D", ms=2.8, color=S.DATA, elinewidth=0.6, label=r"DESI DR1 $f\sigma_{s8}$")
ax[0].set_xlabel("redshift $z$"); ax[0].set_ylabel(r"$f\sigma_8(z)$"); ax[0].set_ylim(0.2, 0.72); ax[0].legend(loc="upper right", fontsize=5.5); S.panel_letter(ax[0], "a")
cH = 299792.458 / 67.36
DV = lambda z: (z * (cH * quad(lambda x: 1 / np.sqrt(C.H2(1 / (1 + x))), 0, z)[0]) ** 2 * cH / np.sqrt(C.H2(1 / (1 + z)))) ** (1 / 3) / 147.09
ax[1].plot(z2, [DV(z) for z in z2], color=S.GR, label=r"photon ruler ($\Lambda$CDM $=$ IAM)")
ax[1].errorbar([v[0] for v in dsv.values()], [v[3] for v in dsv.values()], [np.sqrt(v[4] * 1e-4) for v in dsv.values()], fmt="D", ms=2.8, color=S.DATA, elinewidth=0.6, label="DESI DR1 (ShapeFit only)")
ax[1].set_xlabel("redshift $z$"); ax[1].set_ylabel(r"$D_V/r_d$"); ax[1].legend(loc="upper left", fontsize=5.5); S.panel_letter(ax[1], "b")
W0, WA = np.meshgrid(np.linspace(-1.5, 0.0, 200), np.linspace(-3, 1.2, 200))
with np.errstate(divide="ignore", invalid="ignore"):
    AX = 1 + (1 + W0) / WA; ZX = np.where((AX > 0) & (AX < 1), 1 / AX - 1, np.nan)
cs = ax[2].pcolormesh(W0, WA, np.clip(ZX, 0, 2), cmap="Blues", shading="auto", alpha=0.55, vmin=0, vmax=2)
cb = fig.colorbar(cs, ax=ax[2], pad=0.02); cb.set_label(r"$z_{\rm cross}$", fontsize=6); cb.ax.tick_params(labelsize=5)
ax[2].plot(np.linspace(-1.5, 0, 10), -1 - np.linspace(-1.5, 0, 10), color=S.GR, lw=0.6, ls="-.")
dr2 = [("DESI+CMB", -0.42, 0.21, 0.21, -1.75, 0.58, 0.58, S.ALT2), ("+Pantheon+", -0.838, 0.055, 0.055, -0.62, 0.19, 0.22, S.DATA),
       ("+Union3", -0.667, 0.088, 0.088, -1.09, 0.27, 0.31, S.GOLD), ("+DES Y5", -0.752, 0.057, 0.057, -0.86, 0.20, 0.23, S.ALT)]
for nm, w, wl, wu, a_, al, au, c in dr2:
    ax[2].errorbar(w, a_, xerr=[[wl], [wu]], yerr=[[al], [au]], fmt="o", ms=3, color=c, elinewidth=0.7, label=nm)
ax[2].plot(-1, 0, "s", ms=4, color="k", label=r"$\Lambda$CDM")
ax[2].plot(-1.19, 0.38, "*", ms=6, color=S.IAM, label="two-ruler mock")
ax[2].set_xlim(-1.5, 0); ax[2].set_ylim(-3, 1.2); ax[2].set_xlabel("$w_0$"); ax[2].set_ylabel("$w_a$")
ax[2].legend(loc="lower left", fontsize=5, handletextpad=0.3); S.panel_letter(ax[2], "c")
fig.tight_layout(); S.save(fig, "part2", "fig_sector_phantom")

# ---------------- fig_s8_trend_iam
S8P = 0.832
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4))
ax1.plot(zz, C.E(aa), color=S.IAM); ax1.axhline(0.1, color=S.LIGHT, lw=0.7); ax1.axvline(np.log(10), color=S.LIGHT, lw=0.7)
ax1.text(2.36, 0.16, "$E<0.1$\nfor $z>2.30$", fontsize=6, color=S.GR)
ax1.set_xlabel("redshift $z$"); ax1.set_ylabel(r"$E(a)=e^{1-1/a}$"); ax1.set_xlim(0, 3); ax1.set_ylim(0, 1.05); S.panel_letter(ax1, "a")
z3 = np.linspace(0, 2, 200); a3 = 1 / (1 + z3)
ax2.axhline(S8P, color=S.GR, ls="--", lw=0.8, label=r"$\Lambda$CDM (Planck 0.832)")
ax2.plot(z3, S8P * C.D(C.IAM, a3) / C.D(C.LCDM, a3), color=S.IAM, label="lensing amplitude at $z$")
ax2.plot(z3, S8P * (1 - C.fs8_deficit(z3) / 100), color=S.ALT, label=r"$f\sigma_8$ at $z$")
zs = np.array([0.067, 0.15, 0.295, 0.38, 0.51, 0.51, 0.70, 0.706, 0.85, 0.919, 1.317, 1.48, 1.491])
er = np.array([0.055, 0.16, 0.094, 0.047, 0.039, 0.064, 0.043, 0.053, 0.095, 0.047, 0.037, 0.045, 0.044])
tL = lambda z: C.f(C.LCDM, 1 / (1 + z)) * C.D(C.LCDM, 1 / (1 + z)) / C.D(C.LCDM, 1.0)
ratio = lambda z: 1 - C.fs8_deficit(z) / 100
pts = []
for zmin in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    k = zs >= zmin; t = np.array([tL(z) for z in zs[k]]); w = (t / er[k]) ** 2
    pts.append((np.average(zs[k], weights=w), S8P * (w * np.array([ratio(z) for z in zs[k]])).sum() / w.sum(), np.sqrt(C.Om / 0.3) / np.sqrt(w.sum())))
pts = np.array(pts)
ax2.errorbar(pts[:, 0], pts[:, 1], pts[:, 2], fmt="o", ms=3, color=S.DATA, elinewidth=0.6, capsize=1.5, label=r"fit to $f\sigma_8$ above $z_{\rm min}$")
ax2.set_xlabel("effective redshift"); ax2.set_ylabel("inferred $S_8$"); ax2.set_xlim(0, 2); ax2.set_ylim(0.76, 0.89)
ax2.legend(loc="upper right", fontsize=5.5); S.panel_letter(ax2, "b")
fig.tight_layout(); S.save(fig, "part2", "fig_s8_trend_iam")
