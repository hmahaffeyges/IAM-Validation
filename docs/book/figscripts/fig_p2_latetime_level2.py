"""Part 2, Chapters ch:latetime (p2_07_late_time_growth.tex) and ch:level2 (p2_06_dual_sector_perturbation.tex).
All curves from the model algebra; all posteriors from the chain files (30 % burn-in per file, weighted; _chains.py).
  fig_mu_profile           (a) mu(z): exact IAM form and the MGCAMB form mu = 1 + mu0 Omega_DE/Omega_L; (b) E(a) = exp(1 - 1/a)
  fig_posterior_comparison Planck + RSD 1-D posteriors, LambdaCDM (F) and mu0 fixed (D)
  fig_fsigma8              f sigma8(z), linear growth on the chain means, with the seven compiled measurements
  fig_deltachi2_final      Delta chi2 (fixed minus LambdaCDM) per data combination, with the likelihood ratio
  fig_mu0_posterior_final  posterior of mu0 with the amplitude free, Planck and Planck + RSD
  fig_l2_triangle          Level 2: H0, sigma8, Omega_m, Run A and Run C (68 % and 95 % regions)
  fig_l2_sigma8            Level 2: sigma8 posterior, Run A and Run C
  fig_l2_h0split           Level 2: H0 photon (Run A) and H0 matter = H0 sqrt(1 + beta_m), with Planck and SH0ES
  fig_l2_background        Level 2: perturbation-level runs (A, C) against the two background-level runs
Verification of every number: docs/verification/scripts/verify_late_time_level2.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import _bookstyle as S, _chains as CH
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter, gaussian_filter1d

S.apply()
Om0, bm = 0.3153, 0.15765
H2 = lambda z, om=Om0: om * (1 + z)**3 + 1 - om
mu_ex = lambda z, om=Om0: H2(z, om) / (H2(z, om) + bm * np.exp(-z))
mu_mg = lambda z, om=Om0, m0=-0.13495: 1 + m0 * (1 - om) / H2(z, om) / (1 - om)


def hist1(v, w, lo, hi, n=120, sm=2.0):
    h, e = np.histogram(v, bins=n, range=(lo, hi), weights=w, density=True)
    c = 0.5 * (e[1:] + e[:-1]); h = gaussian_filter1d(h, sm); return c, h / h.max()


def levels2(h, fr=(0.95, 0.68)):
    s = np.sort(h.ravel())[::-1]; c = np.cumsum(s) / s.sum()
    return [s[np.searchsorted(c, f)] for f in fr]


# ---------------- fig_mu_profile ----------------
z = np.linspace(0, 5, 500)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5), gridspec_kw=dict(wspace=0.32))
a1.plot(z, mu_ex(z), color=S.IAM, lw=1.6, label="exact: $H^2/(H^2+\\beta E H_0^2)$")
a1.plot(z, mu_mg(z), color=S.ALT, lw=1.3, ls="--", label="MGCAMB: $1+\\mu_0\\,\\Omega_{\\rm DE}(a)/\\Omega_\\Lambda$")
a1.axhline(1, color=S.LIGHT, lw=0.8)
a1.set_xlim(0, 5); a1.set_ylim(0.85, 1.01); a1.set_xlabel("redshift $z$"); a1.set_ylabel("$\\mu(z)$")
a1.legend(loc="lower right"); a1.set_title("Coupling for matter perturbations"); S.panel_letter(a1, "a", dx=-0.16)
aa = np.linspace(0.02, 1, 400)
a2.plot(aa, np.exp(1 - 1 / aa), color=S.IAM, lw=1.6)
for zq in (0, 1, 3):
    a2.plot(1 / (1 + zq), np.exp(-zq), "o", color=S.IAM, ms=4)
    a2.annotate(f"$z={zq}$", (1 / (1 + zq), np.exp(-zq)), xytext=(-8, 6) if zq else (-8, -12), textcoords="offset points", fontsize=7, ha="right")
a2.set_xlim(0, 1.02); a2.set_ylim(-0.02, 1.05); a2.set_xlabel("scale factor $a$"); a2.set_ylabel("$E(a)=\\exp(1-1/a)$")
a2.set_title("Activation: zero early, one today"); S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_mu_profile")

# ---------------- Level 1 chains ----------------
L1 = {k: [CH.load(*f) for f in v] for k, v in CH.L1.items()}
C, F, Fr = L1["Planck + RSD"]
pars = [("H0", "$H_0$"), ("sigma8", "$\\sigma_8$"), ("ombh2", "$\\Omega_bh^2$"), ("omch2", "$\\Omega_ch^2$"), ("ns", "$n_s$"), ("logA", "$\\ln10^{10}A_s$")]
fig, axs = plt.subplots(2, 3, figsize=(S.TEXTW, 3.4), gridspec_kw=dict(wspace=0.38, hspace=0.6))
for ax, (p, lab) in zip(axs.ravel(), pars):
    lo = min(C[p].quantile(0.001), F[p].quantile(0.001)); hi = max(C[p].quantile(0.999), F[p].quantile(0.999))
    for X, col, nm in ((C, S.GR, "$\\Lambda$CDM (F)"), (F, S.IAM, "$\\mu_0=-0.135$ fixed (D)")):
        c, h = hist1(X[p].values, X.weight.values, lo, hi); ax.plot(c, h, color=col, lw=1.3, label=nm)
    ax.set_xlabel(lab); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.locator_params(axis="x", nbins=3)
axs[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.45), ncol=2)
S.save(fig, "part2", "fig_posterior_comparison")

# ---------------- fig_fsigma8 ----------------
def growth(mu, om, zs):
    def rhs(l, y):
        a = np.exp(l); h2 = om / a**3 + 1 - om; dl = -1.5 * om / a**3 / h2
        return [y[1], -(2 + dl) * y[1] + 1.5 * om / a**3 / h2 * mu(a) * y[0]]
    s = solve_ivp(rhs, [np.log(1e-3), 0], [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-13)
    y = s.sol(-np.log1p(np.asarray(zs, float))); return y[0], y[1]
zz = np.linspace(0, 1.6, 200)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
for X, col, m0, nm in ((C, S.GR, 0.0, "$\\Lambda$CDM ($\\mu=1$), chain F"), (F, S.IAM, -0.13495, "$\\mu_0=-0.135$, chain D")):
    om = 1 - np.average(X.omegal, weights=X.weight); s8 = np.average(X.sigma8, weights=X.weight)
    mu = (lambda a, om=om, m0=m0: 1 + m0 * (1 - om) / (om / a**3 + 1 - om) / (1 - om))
    D, Dp = growth(mu, om, zz); D0, _ = growth(mu, om, [0])
    ax.plot(zz, s8 * Dp / D0[0], color=col, lw=1.5, label=nm)
dat = np.array([[0.067, 0.423, 0.055], [0.150, 0.530, 0.160], [0.380, 0.497, 0.045], [0.510, 0.459, 0.038],
                [0.700, 0.473, 0.041], [0.850, 0.315, 0.095], [1.480, 0.462, 0.045]])
ax.errorbar(dat[:, 0], dat[:, 1], dat[:, 2], fmt="o", ms=3.5, color=S.DATA, elinewidth=0.9, capsize=2, label="measurements (seven $z$)")
ax.set_xlim(0, 1.6); ax.set_ylim(0.2, 0.72); ax.set_xlabel("redshift $z$"); ax.set_ylabel("$f\\sigma_8(z)$")
ax.legend(loc="upper right", fontsize=6.5); ax.set_title("Growth is lower at every redshift")
S.save(fig, "part2", "fig_fsigma8")

# ---------------- fig_deltachi2_final ----------------
names = list(L1.keys()); d = [L1[k][1].chi2.min() - L1[k][0].chi2.min() for k in names]
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.4))
x = np.arange(len(names)); ax.bar(x, d, color=S.IAM, width=0.55)
for xi, v in zip(x, d):
    ax.annotate(f"{v:+.2f}\n$e^{{-\\Delta\\chi^2/2}}={np.exp(-v/2):.2f}$", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=6.5)
ax.set_ylim(0, 2.6); ax.set_ylabel("$\\Delta\\chi^2$ (fixed $-$ $\\Lambda$CDM)"); ax.set_title("Same parameter count: read as a likelihood ratio")
S.save(fig, "part2", "fig_deltachi2_final")

# ---------------- fig_mu0_posterior_final ----------------
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.4))
for k, ls, col in (("Planck", "--", S.GR), ("Planck + RSD", "-", S.IAM)):
    X = L1[k][2]; c, h = hist1(X.mu0.values, X.weight.values, -0.5, 0.2, n=70, sm=1.2)
    ax.plot(c, h, ls=ls, color=col, lw=1.4, label=k)
ax.axvline(-0.135, color=S.DATA, lw=1.0); ax.text(-0.14, 1.02, "prediction\n$-0.135$", fontsize=6.5, ha="right", va="bottom", color=S.DATA)
ax.axvline(0, color=S.LIGHT, lw=0.8, ls=":"); ax.axvline(0.2, color="black", lw=0.8)
ax.text(0.195, 0.05, "prior edge", rotation=90, fontsize=6.5, ha="right", va="bottom")
ax.set_xlim(-0.5, 0.22); ax.set_ylim(0, 1.25); ax.set_xlabel("$\\mu_0$"); ax.set_ylabel("posterior (peak = 1)")
ax.legend(loc="upper left"); ax.set_title("Free amplitude: rises to the prior edge")
S.save(fig, "part2", "fig_mu0_posterior_final")

# ---------------- Level 2 ----------------
A = CH.load(*CH.L2["A"]); Cc = CH.load(*CH.L2["C"])
Bb = CH.load("camb_validation/chains/iam_l2b_runA.1.txt"); Bd = CH.load("camb_validation/chains/iam_l2b_runD.1.txt")
tri = [("H0", "$H_0$"), ("sigma8", "$\\sigma_8$"), ("omegam", "$\\Omega_m$")]
rng = {p: (min(A[p].quantile(0.002), Cc[p].quantile(0.002)), max(A[p].quantile(0.998), Cc[p].quantile(0.998))) for p, _ in tri}
fig, axs = plt.subplots(3, 3, figsize=(0.75 * S.TEXTW, 0.75 * S.TEXTW), gridspec_kw=dict(wspace=0.08, hspace=0.08))
for i, (pi, li) in enumerate(tri):
    for j, (pj, lj) in enumerate(tri):
        ax = axs[i, j]
        if j > i: ax.axis("off"); continue
        for X, col, nm in ((Cc, S.GR, "$\\Lambda$CDM (Run C)"), (A, S.IAM, "informational term (Run A)")):
            if i == j:
                c, h = hist1(X[pi].values, X.weight.values, *rng[pi]); ax.plot(c, h, color=col, lw=1.3, label=nm); ax.set_yticks([])
            else:
                h, xe, ye = np.histogram2d(X[pj], X[pi], bins=50, range=[rng[pj], rng[pi]], weights=X.weight); h = gaussian_filter(h, 1.3)
                ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), h.T, levels=sorted(levels2(h)), colors=[col], linewidths=[0.8, 1.2])
        ax.set_xlim(*rng[pj])
        if i != j: ax.set_ylim(*rng[pi])
        if i == 2: ax.set_xlabel(lj); ax.locator_params(axis="x", nbins=3)
        else: ax.set_xticklabels([])
        if j == 0 and i > 0: ax.set_ylabel(li); ax.locator_params(axis="y", nbins=3)
        elif i != j: ax.set_yticklabels([])
h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper right", bbox_to_anchor=(0.92, 0.88))
S.save(fig, "part2", "fig_l2_triangle")

fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.3))
for X, col, nm in ((Cc, S.GR, "$\\Lambda$CDM (Run C)"), (A, S.IAM, "informational term (Run A)")):
    c, h = hist1(X.sigma8.values, X.weight.values, 0.775, 0.835); ax.plot(c, h, color=col, lw=1.5, label=nm)
    m = np.average(X.sigma8, weights=X.weight); ax.axvline(m, color=col, lw=0.8, ls="--")
    ax.annotate(f"{m:.4f}", (m, 1.02), ha="center", va="bottom", fontsize=6.5, color=col)
ax.set_xlim(0.775, 0.835); ax.set_ylim(0, 1.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("$\\sigma_8$"); ax.legend(loc="upper left", fontsize=6.5, ncol=2); ax.set_title("$\\sigma_8$ lower by 0.009 ($-1.1$ %)")
S.save(fig, "part2", "fig_l2_sigma8")

fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 2.4))
ax.axvspan(67.36 - 0.54, 67.36 + 0.54, color=S.LIGHT, alpha=0.45, lw=0); ax.text(67.36, 1.05, "Planck $\\Lambda$CDM\n67.36 $\\pm$ 0.54", ha="center", fontsize=6.5)
ax.axvspan(73.04 - 1.04, 73.04 + 1.04, color=S.SKY, alpha=0.35, lw=0); ax.text(73.04, 1.05, "SH0ES\n73.04 $\\pm$ 1.04", ha="center", fontsize=6.5)
for v, col, nm in ((A.H0.values, S.GR, "$H_0$ photon (Run A, sampled)"), (A.H0.values * np.sqrt(1 + bm), S.IAM, "$H_0$ matter $=H_0\\sqrt{1+\\beta_m}$")):
    c, h = hist1(v, A.weight.values, 64, 76, n=240); ax.plot(c, h, color=col, lw=1.5, label=nm)
ax.set_xlim(64.5, 76); ax.set_ylim(0, 1.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$)"); ax.legend(loc="upper center", ncol=2, fontsize=6.5)
ax.set_title("One posterior, two rates: 67.16 and 72.26")
S.save(fig, "part2", "fig_l2_h0split")

fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 2.4))
ax.axvspan(67.36 - 0.54, 67.36 + 0.54, color=S.LIGHT, alpha=0.45, lw=0); ax.axvspan(73.04 - 1.04, 73.04 + 1.04, color=S.SKY, alpha=0.35, lw=0)
for X, col, ls, nm in ((Bb, S.DATA, "-", "term in the background: Planck"), (Bd, S.DATA, ":", "term in the background: Planck + growth"),
                       (Cc, S.GR, "-", "$\\Lambda$CDM (Run C)"), (A, S.IAM, "-", "term in the perturbations (Run A)")):
    c, h = hist1(X.H0.values, X.weight.values, 59, 75, n=320); ax.plot(c, h, color=col, ls=ls, lw=1.4, label=nm)
ax.axvline(72.26, color=S.IAM, lw=1.0, ls="--"); ax.text(72.4, 0.62, "matter rate\n72.26", fontsize=6.5, color=S.IAM)
ax.set_xlim(59.5, 75); ax.set_ylim(0, 1.7); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$)"); ax.legend(loc="upper left", fontsize=6, ncol=2)
ax.axvline(66.12, color=S.DATA, lw=1.0, ls="--"); ax.text(65.95, 0.62, "expansion rate\ntoday, 66.1", fontsize=6.5, color=S.DATA, ha="right")
ax.set_title("Coded background term: sampled $H_0$ 61.5, rate today 66.1")
S.save(fig, "part2", "fig_l2_background")
