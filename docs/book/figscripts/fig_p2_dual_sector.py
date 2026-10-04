"""Figures for the dual-sector chapters (ch:dsvalidation p2_10, ch:dsnote p2_05).

Inputs: docs/verification/scripts/verify_dual_sector_chapters_data.json (written by verify_dual_sector_chapters.py, run it first)
and the chain table Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv. Equations as in _cosmo.py: LambdaCDM background (Planck 2018,
Om 0.3153), beta_m = Om/2 = 0.15765, E(a) = exp(1 - 1/a), mu(a) = H^2/(H^2 + beta_m E(a) H0^2), Sigma = 1.
Outputs (docs/book/figures/part2/): fig_dsv_three_tests, fig_dsv_hubble, fig_dsv_systematics, fig_dsv_schematic,
fig_dsnote_probes, fig_dsnote_growth.
"""
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import _bookstyle as S

S.apply()
J = json.load(open(S.REPO / "docs/verification/scripts/verify_dual_sector_chapters_data.json"))
T = pd.read_csv(S.REPO / "Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv")
row = lambda n: T[T.chain.str.startswith(n)].iloc[0]
Om, OL = 0.3153, 0.6847
bm = Om / 2
H0g = row("iam_level2_runA").H0
H0m = H0g * np.sqrt(1 + bm)
E = lambda a: np.exp(1 - 1 / a)
H2 = lambda a: Om * a**-3 + OL
mu = lambda a: H2(a) / (H2(a) + bm * E(a))

# ------------------------------------------------------------------ fig_dsv_three_tests
b = np.array(J["beta_grid"]); P = np.array(J["beta_profile"]); P315 = np.array(J["beta_profile_Om315"])
fig, ax = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5), gridspec_kw=dict(wspace=0.35))
ax[0].plot(b, P, color=S.IAM, label=r"$\Omega_m$ free (Tests A, B and C alike)")
ax[0].plot(b, P315, color=S.GR, ls="--", label=r"$\Omega_m=0.315$ fixed")
ax[0].plot([-0.30], [P.min()], "o", color=S.IAM, ms=4)
ax[0].text(-0.28, P.min() - 0.9, f"{P.min():.2f}", fontsize=6, va="center")
ax[0].axvline(bm, color=S.LIGHT, lw=0.8); ax[0].text(bm - 0.01, P.min() + 19, r"$\beta_m$", fontsize=6, color=S.GR, ha="right")
ax[0].set_xlabel(r"$\beta$ in the distances"); ax[0].set_ylabel(r"minimum $\chi^2$")
ax[0].set_ylim(P.min() - 2, P.min() + 22); ax[0].legend(loc="upper left", fontsize=6)
S.panel_letter(ax[0], "a", dx=-0.16)
H0s = np.linspace(60, 75, 151)
dchi = P - P.min()
Z = np.tile(dchi[:, None], (1, len(H0s)))
ax[1].contourf(H0s, b, Z, levels=[0, 2.30, 6.18], colors=[S.IAM, S.SKY], alpha=0.8)
ax[1].contour(H0s, b, Z, levels=[2.30, 6.18], colors=[S.IAM], linewidths=0.6)
for h, lab, col in ((67.4, "Planck 67.4", S.GR), (73.04, "SH0ES 73.04", S.DATA)):
    ax[1].axvline(h, color=col, lw=0.9, ls="--"); ax[1].text(h + 0.2, 0.24, lab, fontsize=6, color=col, rotation=90, va="top")
ax[1].axhline(bm, color=S.ALT, lw=0.9); ax[1].text(60.4, bm + 0.012, r"$\beta_m=0.15765$", fontsize=6, color=S.ALT)
ax[1].set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]"); ax[1].set_ylabel(r"$\beta$ in the distances"); ax[1].set_ylim(-0.3, 0.3)
ax[1].text(74.6, -0.27, "68 % and 95 %\n(2 parameters)", fontsize=6, ha="right", va="bottom", color="white")
S.panel_letter(ax[1], "b", dx=-0.16)
S.save(fig, "part2", "fig_dsv_three_tests")

# ------------------------------------------------------------------ fig_dsv_hubble
z = np.array(J["hd_z"]); mb = np.array(J["hd_mb"]); er = np.array(J["hd_err"]); res = np.array(J["hd_resid"]); off = J["hd_offset_H70"]
c = 299792.458
zg = np.linspace(0, 2.4, 4000)
def dl(Omm, bb, zz):
    aa = 1 / (1 + zg); H = np.sqrt(Omm * aa**-3 + 1 - Omm + bb * np.exp(1 - 1 / aa))
    dc = np.concatenate([[0], np.cumsum(0.5 * (1 / H[1:] + 1 / H[:-1]) * np.diff(zg))])
    return (1 + zz) * c / 70.0 * np.interp(zz, zg, dc)
fig, ax = plt.subplots(2, 1, figsize=(S.TEXTW * 0.72, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
ax[0].errorbar(z, mb, er, fmt="o", ms=1.2, color=S.DATA, ecolor=S.LIGHT, elinewidth=0.4, alpha=0.7, label="Pantheon+ (1588 SNe)")
zc = np.logspace(-2, np.log10(2.3), 300)
ax[0].plot(zc, 5 * np.log10(dl(0.315, 0, zc)) + off, color=S.GR, label=r"$\Lambda$CDM, $\Omega_m=0.315$, offset fitted")
ax[0].set_xscale("log"); ax[0].set_ylabel(r"$m_b^{\rm corr}$ [mag]"); ax[0].legend(loc="upper left", fontsize=6)
S.panel_letter(ax[0], "a", dx=-0.12)
ax[1].plot(z, res, "o", ms=1.0, color=S.LIGHT)
edges = np.logspace(-2, np.log10(2.3), 13); w = 1 / er**2
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (z > lo) & (z <= hi)
    if s.sum() > 2:
        m = np.sum(w[s] * res[s]) / np.sum(w[s]); e = 1 / np.sqrt(np.sum(w[s]))
        ax[1].errorbar(np.sqrt(lo * hi), m, e, fmt="s", ms=3, color=S.DATA, elinewidth=0.8)
ax[1].axhline(0, color=S.GR, lw=0.8); ax[1].set_ylim(-0.6, 0.6)
ax[1].set_xlabel("redshift $z_{\\rm CMB}$"); ax[1].set_ylabel("residual [mag]")
S.panel_letter(ax[1], "b", dx=-0.12)
S.save(fig, "part2", "fig_dsv_hubble")

# ------------------------------------------------------------------ fig_dsv_systematics
fig, ax = plt.subplots(1, 3, figsize=(S.TEXTW, 2.3), gridspec_kw=dict(wspace=0.6))
for k, (key, lab, col, dx) in enumerate((("diag", "own offset, diagonal", S.GR, -0.012), ("full", "own offset, full cov.", S.IAM, 0.0),
                                          ("common", "common offset, full cov.", S.ALT, 0.012))):
    for i, B in enumerate(J["bins"]):
        bb, lo, hi = B[key]; x = i + dx * 10
        ax[0].errorbar(x, bb, [[bb - lo], [hi - bb]], fmt="o", ms=3, color=col, elinewidth=0.8, label=lab if i == 0 else None)
ax[0].axhline(0, color=S.GR, lw=0.6, ls=":"); ax[0].axhline(bm, color=S.LIGHT, lw=0.8)
ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels([f"{B['lo']:.2f}-{B['hi']:.2f}\nN={B['n_full']}" for B in J["bins"]], fontsize=5.5)
ax[0].set_ylabel(r"best $\beta$ (68 %)"); ax[0].legend(fontsize=5, loc="upper left"); ax[0].set_ylim(-0.6, 1.5)
S.panel_letter(ax[0], "a", dx=-0.3)
oms = [0.308, 0.315, 0.322]
txt = open(S.REPO / "docs/verification/scripts/verify_dual_sector_chapters_output.txt").read().splitlines()
vals = [l for l in txt if l.strip().startswith("Om 0.3") and "|" in l]
bf = [float(l.split("beta")[1].split("|")[0]) for l in vals]; bd = [float(l.split("|")[1]) for l in vals]
ax[1].plot(oms, bf, "o-", color=S.IAM, ms=3, label="full covariance"); ax[1].plot(oms, bd, "s--", color=S.GR, ms=3, label="diagonal errors")
ax[1].set_xlabel(r"$\Omega_m$ (fixed)"); ax[1].set_ylabel(r"best $\beta$"); ax[1].legend(fontsize=5.5); ax[1].axhline(0, color=S.GR, lw=0.6, ls=":")
S.panel_letter(ax[1], "b", dx=-0.3)
ss = J["subsample"]
ax[2].errorbar([s["n"] for s in ss], [s["mean"] for s in ss], [s["sd"] for s in ss], fmt="o", ms=3, color=S.IAM, elinewidth=0.8)
ax[2].set_xscale("log"); ax[2].set_xlabel("number of SNe"); ax[2].set_ylabel(r"best $\beta$ ($\Omega_m=0.315$)")
ax[2].axhline(0, color=S.GR, lw=0.6, ls=":")
S.panel_letter(ax[2], "c", dx=-0.3)
S.save(fig, "part2", "fig_dsv_systematics")

# ------------------------------------------------------------------ fig_dsv_schematic
fig, ax = plt.subplots(figsize=(S.TEXTW, 3.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, title, lines, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc="white", ec=col, lw=1.0))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=7, fontweight="bold", color=col)
    for i, l in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.62 - 0.33 * i, l, ha="left", va="top", fontsize=6)
box(0.2, 3.1, 4.6, 2.7, "Photon sector (null worldlines)",
    [r"no proper time: $d\tau=0$, no records written", r"$\beta_\gamma<0.0052$ (95 %): acoustic scale",
     r"$\beta_\gamma/\beta_m<0.033$", r"$H_0=67.16\pm0.47$ (Level 2 chain)", r"$\Sigma=1$: lensing and $\theta_s$ unmodified"], S.GR)
box(5.2, 3.1, 4.6, 2.7, "Matter sector (timelike worldlines)",
    [r"records written: $S_{\rm info}>0$ iff $d\tau>0$", r"$\beta_m=\Omega_m/2=0.15765$, fixed",
     r"$\mu(a)<1$: $\mu_0=-0.136$, $\sigma_8$ 0.809 $\to$ 0.800", r"$H_0^{\rm matter}=67.16\sqrt{1+\beta_m}=72.26$",
     "background (distances): LCDM"], S.ALT)
box(0.2, 0.2, 4.6, 2.5, "Type Ia supernovae (Pantheon+)",
    [r"Hubble-flow shape: $\beta=-0.035$ (68 %: $-0.065$ to $0$)", r"$\beta_m$ in the distances: $\Delta\chi^2=+23.6$",
     r"$M$ free: magnitudes carry no $H_0$", r"normalisation: Cepheid ladder $73.04\pm1.04$"], S.DATA)
box(5.2, 0.2, 4.6, 2.5, "Mechanism (Chapter on the theory)",
    [r"irreversible records on timelike worldlines", r"Landauer cost on the horizon: $-dE=T_H\,d(S_{\rm geo}+S_{\rm info})$",
     r"$E(a)=\exp(1-1/a)$, production per e-fold peaks today", r"no free parameter beyond $\Lambda$CDM"], S.IAM)
S.save(fig, "part2", "fig_dsv_schematic")

# ------------------------------------------------------------------ fig_dsnote_probes
zz = np.linspace(0, 3, 400); aa = 1 / (1 + zz)
fig, ax = plt.subplots(2, 2, figsize=(S.TEXTW, 4.4), gridspec_kw=dict(wspace=0.4, hspace=0.5))
A = ax[0, 0]
A.plot(zz, H0g * np.sqrt(H2(aa)), color=S.GR, label=rf"photon sector, $H_0={H0g:.2f}$")
A.plot(zz, H0g * np.sqrt(H2(aa) + bm * E(aa)), color=S.ALT, label=rf"matter sector, $H_0={H0m:.2f}$")
A.set_xlabel("redshift $z$"); A.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]"); A.legend(fontsize=6, loc="upper left")
ins = A.inset_axes([0.58, 0.12, 0.38, 0.38]); zs_ = np.linspace(0, 0.3, 50); a_ = 1 / (1 + zs_)
ins.plot(zs_, H0g * np.sqrt(H2(a_)), color=S.GR); ins.plot(zs_, H0g * np.sqrt(H2(a_) + bm * E(a_)), color=S.ALT)
ins.tick_params(labelsize=5); ins.set_xlim(0, 0.3)
S.panel_letter(A, "a", dx=-0.17)
B = ax[0, 1]; zm = np.array([0.11, 0.69, 2.3])
B.plot(zz, mu(aa), color=S.IAM, label=r"$\mu(z)$ (matter)"); B.axhline(1, color=S.GR, lw=0.9, ls="--", label=r"$\Sigma=1$ (light)")
B.plot(zm, mu(1 / (1 + zm)), "o", color=S.IAM, ms=3)
for zv in zm:
    B.annotate(f"{mu(1/(1+zv)):.3f}", (zv, mu(1 / (1 + zv))), (zv + 0.12, mu(1 / (1 + zv)) - 0.012), fontsize=6)
B.plot([0], [mu(1.0)], "o", color=S.IAM, ms=3); B.text(0.12, mu(1.0) - 0.003, f"{mu(1.0):.3f} (z = 0)", fontsize=6, va="center")
B.set_xlabel("redshift $z$"); B.set_ylabel(r"$\mu$, $\Sigma$"); B.legend(fontsize=6, loc="lower right"); B.set_ylim(0.85, 1.01)
S.panel_letter(B, "b", dx=-0.17)
Cx = ax[1, 0]
Cx.plot(zz, 100 * E(aa), color=S.GOLD, label=r"$E(a)$ [%]")
Cx.plot(zz, 100 * (np.sqrt(1 + bm * E(aa) / H2(aa)) - 1) * 10, color=S.ALT, label=r"$10\times(H_m/H-1)$ [%]")
for zv in zm:
    a1 = 1 / (1 + zv); Cx.plot([zv], [100 * E(a1)], "o", color=S.GOLD, ms=3)
    Cx.annotate(f"{100*E(a1):.0f} %\n{100*(np.sqrt(1+bm*E(a1)/H2(a1))-1):.1f} %", (zv, 100 * E(a1)), (zv + 0.1, 100 * E(a1) + 4), fontsize=5.5)
Cx.set_xlabel("redshift $z$"); Cx.set_ylabel("per cent"); Cx.legend(fontsize=6, loc="upper right"); Cx.set_ylim(0, 105)
S.panel_letter(Cx, "c", dx=-0.17)
Dx = ax[1, 1]
Dx.plot(zz, 100 * (mu(aa) - 1), color=S.IAM)
Dx.text(0.08, 100 * (mu(1.0) - 1), f"{100*(mu(1.0)-1):.1f} % at z = 0", fontsize=6, va="center")
Dx.set_xlabel("redshift $z$"); Dx.set_ylabel(r"$\Omega_m(a;\beta_m)/\Omega_m(a;0)-1$ [%]")
S.panel_letter(Dx, "d", dx=-0.17)
S.save(fig, "part2", "fig_dsnote_probes")

# ------------------------------------------------------------------ fig_dsnote_growth
def grow(bkg_beta, use_mu):
    def h2(a): return H2(a) + (bm * E(a) if bkg_beta else 0.0)
    def dlnh(a):   # dlnH/dlna
        dE = E(a) / a      # dE/dlna
        return 0.5 * (-3 * Om * a**-3 + (bm * dE if bkg_beta else 0.0)) / h2(a)
    def rhs(l, y):
        a = np.exp(l); src = 1.5 * Om * a**-3 / h2(a) * (mu(a) if use_mu else 1.0)
        return [y[1], -(2 + dlnh(a)) * y[1] + src * y[0]]
    return solve_ivp(rhs, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
models = {"LCDM": grow(False, False), "B": grow(True, False), "C": grow(False, True), "D": grow(True, True)}
def fD(Sx, a):
    y = Sx.sol(np.log(a)); return y[1]   # f*D = dD/dlna
zg2 = np.linspace(0, 2, 200); ag2 = 1 / (1 + zg2)
fig, ax = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.85))
ref = fD(models["LCDM"], ag2)
for k, lab, col, ls in (("C", "C: perturbations only", S.IAM, "-"), ("B", "B: background only", S.GR, "--"),
                        ("D", "D: both", S.ALT, ":")):
    ax[0].plot(zg2, 100 * (fD(models[k], ag2) / ref - 1), color=col, ls=ls, label=lab)
ax[0].axhline(0, color=S.LIGHT, lw=0.8)
ax[0].set_xlabel("redshift $z$"); ax[0].set_ylabel(r"$f\sigma_8$ relative to $\Lambda$CDM [%]"); ax[0].legend(fontsize=5.5, loc="lower right")
S.panel_letter(ax[0], "a", dx=-0.17)
A2, C2 = row("iam_level2_runA"), row("iam_level2_runC")
pts = [("Planck 2018 (published)", 0.832, 0.013, 0.013, S.GR), (r"$\Lambda$CDM chain (Level 2)", C2.S8, C2.S8_sd, C2.S8_sd, S.GR),
       (r"$\beta_m$ fixed (Level 2)", A2.S8, A2.S8_sd, A2.S8_sd, S.IAM), ("KiDS-Legacy", 0.815, 0.021, 0.016, S.DATA),
       ("DES Y3 3x2pt", 0.776, 0.017, 0.017, S.DATA), ("KiDS-1000 shear", 0.759, 0.021, 0.024, S.DATA)]
for i, (lab, v, lo, hi, col) in enumerate(pts):
    ax[1].errorbar(v, i, xerr=[[lo], [hi]], fmt="o", color=col, ms=3, elinewidth=0.8)
ax[1].set_yticks(range(len(pts))); ax[1].set_yticklabels([p[0] for p in pts], fontsize=6); ax[1].invert_yaxis()
ax[1].set_xlabel(r"$S_8=\sigma_8\sqrt{\Omega_m/0.3}$")
S.panel_letter(ax[1], "b", dx=-0.62)
S.save(fig, "part2", "fig_dsnote_growth")
for k in ("B", "C", "D"):
    print(k, "fsigma8 rel. LCDM at z = 0, 0.5, 1:", [round(100 * (fD(models[k], 1/(1+zv)) / fD(models["LCDM"], 1/(1+zv)) - 1), 2) for zv in (0, 0.5, 1)])
