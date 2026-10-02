"""Part 2, Chapters 'Why matter and light are in different sectors' (p2_05_dual_sector_note.tex) and
'Type Ia supernovae in the dual-sector picture' (p2_10_dual_sector_validation.tex).

fig_beta_gamma: the photon coupling tested on the CMB acoustic scale. Same model and numbers as
  docs/verification/scripts/verify_beta_gamma.py: H_photon = 67.4 sqrt(Om a^-3 + Or a^-4 + OL + beta_g e^(1-1/a)),
  r_s = 144.43 Mpc fixed, theta_s = 0.0104110 +/- 0.0000031 (Planck 2018). (a) Delta chi^2 profile in beta_g >= 0;
  (b) the shift of theta_s, in units of the Planck error, up to the full matter coupling beta_m = Om/2.
fig_sn_h0_flat: Pantheon+SH0ES (Brout et al. 2022). (a) Hubble-flow residuals (z_HD > 0.01, 1590 SNe) against LambdaCDM at
  Om = 0.315 with the offset M fitted by generalised least squares on the full STAT+SYS covariance, binned in log z (inverse-variance
  weighted means of the diagonal errors), with the beta_m-on-distances curve and the best-fit beta = -0.035 curve, each with its own
  GLS offset; (b) the chi^2 profile in H0 with M free, diagonal errors, 0.01 < zCMB < 2.5 (1588 SNe), minimised over (Om, beta, M) with
  the bounds of verify_dual_sector_validation.py: flat, because H0 and M enter only as M - 5 log10 H0.
Data: downloaded once from github.com/PantheonPlusSH0ES/DataRelease into figscripts/_data/ (or set PANTHEON_DIR), as fig_p2_sn_future.py.
"""
import sys, os, pathlib, urllib.request; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()

# ---------------- fig_beta_gamma -----------------
c, H0, Om, Or = 299792.458, 67.4, 0.315, 9.24e-5; OL = 1 - Om - Or; obs, err, rs = 0.0104110, 3.1e-6, 144.43
bm = 0.3153 / 2
Hg = lambda a, b: H0 * np.sqrt(Om * a**-3 + Or * a**-4 + OL + b * np.exp(1 - 1 / a))
def theta(b):
    return rs / (quad(lambda z: 1 / Hg(1 / (1 + z), b), 0, 1090, limit=500, epsabs=0, epsrel=1e-12)[0] * c)
t0 = theta(0.0); c0 = ((t0 - obs) / err)**2
dchi = lambda b: ((theta(b) - obs) / err)**2 - c0
lim = {lev: brentq(lambda b: dchi(b) - lev, 1e-6, 0.05) for lev in (1, 4, 9)}
bg = np.linspace(0, 0.009, 91); prof = np.array([dchi(b) for b in bg])
bb = np.linspace(0, 0.17, 69); shift = np.array([(theta(b) - t0) / err for b in bb])
s_m = (theta(bm) - t0) / err; p_m = 100 * (theta(bm) / t0 - 1)
print(f"beta_g < {lim[1]:.4f} (68 %), {lim[4]:.4f} (95 %), {lim[9]:.4f} (99.7 %); ratio to beta_m {lim[4]/bm:.3f}; "
      f"beta_m on photon paths: {p_m:+.3f} % = {s_m:.1f} sigma")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.36))
a1.plot(bg, prof, color=S.DATA, lw=1.5)
for lev, ls in ((1, ":"), (4, "--"), (9, ":")):
    a1.axhline(lev, color=S.LIGHT, lw=0.7, ls=ls)
a1.axvline(lim[4], color=S.DATA, lw=0.8, ls="--")
a1.annotate(f"95 %: $\\beta_\\gamma$ < {lim[4]:.4f}", (lim[4], 4), xytext=(6, -12), textcoords="offset points", fontsize=7, color=S.DATA, va="top")
a1.set_xlim(0, 0.009); a1.set_ylim(0, 14)
a1.set_xlabel("photon coupling $\\beta_\\gamma$"); a1.set_ylabel("$\\Delta\\chi^2$ (acoustic scale)")
a1.set_title("Light is not coupled")
S.panel_letter(a1, "a", dx=-0.16)
a2.plot(bb, shift, color=S.IAM, lw=1.5)
a2.axhspan(-2, 2, color=S.LIGHT, alpha=0.45, lw=0)
a2.text(0.165, 3.5, "Planck $\\pm2\\sigma$", fontsize=7, color=S.GR, ha="right", va="bottom")
a2.plot(bm, s_m, "o", color=S.IAM, ms=5)
a2.text(0.006, 31, f"$\\beta_m=\\Omega_m/2$ on light paths:\n$\\theta_s$ {p_m:+.2f} %, {s_m:.0f}$\\sigma$", fontsize=7, color=S.IAM, ha="left", va="top")
a2.set_xlim(0, 0.17); a2.set_ylim(-3, 34)
a2.set_xlabel("coupling applied to photon paths, $\\beta$"); a2.set_ylabel("$\\Delta\\theta_s$ / $\\sigma_{\\theta_s}$")
a2.set_title("The matter coupling on light paths")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_beta_gamma")

# ---------------- fig_sn_h0_flat -----------------
DD = pathlib.Path(os.environ.get("PANTHEON_DIR", S.HERE / "_data")); DD.mkdir(parents=True, exist_ok=True)
B = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
for f, u in (("PantheonPlusSH0ES.dat", "Pantheon%2BSH0ES.dat"), ("PantheonPlusSH0ES_STAT+SYS.cov", "Pantheon%2BSH0ES_STAT%2BSYS.cov")):
    if not (DD / f).exists():
        urllib.request.urlretrieve(B + u, DD / f)
D = pd.read_csv(DD / "PantheonPlusSH0ES.dat", sep=r"\s+")
raw = np.loadtxt(DD / "PantheonPlusSH0ES_STAT+SYS.cov"); N = int(raw[0]); Cv = raw[1:].reshape(N, N)
m = (D.zHD > 0.01).values; Ci = np.linalg.inv(Cv[np.ix_(m, m)])
zz, zh, y, ed = D.zHD.values[m], D.zHEL.values[m], D.m_b_corr.values[m], D.m_b_corr_err_DIAG.values[m]
zg = np.linspace(0, zz.max() + 0.01, 4000)
def dlf(Om_, b, zq, zhq, Hz=70.0):
    a = 1 / (1 + zg); H = Hz * np.sqrt(Om_ * a**-3 + 1 - Om_ + b * np.exp(1 - 1 / a))
    dc = np.concatenate([[0], np.cumsum(0.5 * (c / H[1:] + c / H[:-1]) * np.diff(zg))]); return (1 + zhq) * np.interp(zq, zg, dc)
o = np.ones_like(y)
def gls(b):
    mod = 5 * np.log10(dlf(0.315, b, zz, zh)) + 25; r = y - mod; off = (o @ Ci @ r) / (o @ Ci @ o); d = r - off
    return mod + off, d @ Ci @ d
mod0, chi0 = gls(0.0); modm, chim = gls(bm); mode, chie = gls(-0.035)
print(f"{m.sum()} SNe; LCDM chi2 {chi0:.2f}; beta_m {chim:.2f} (dchi2 {chim-chi0:+.2f}); beta -0.035 {chie:.2f} (dchi2 {chie-chi0:+.2f})")
res = y - mod0; edges = np.logspace(np.log10(0.01), np.log10(2.3), 13)
zb, rb, eb, nb = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (zz >= lo) & (zz < hi)
    if k.sum() < 3: continue
    w = 1 / ed[k]**2; zb.append(np.exp(np.average(np.log(zz[k]), weights=w))); rb.append(np.average(res[k], weights=w)); eb.append(1 / np.sqrt(w.sum())); nb.append(k.sum())
zb, rb, eb = map(np.array, (zb, rb, eb)); print("bins:", nb)
srt = np.argsort(zz)
# (b) diagonal errors, the three-test setup
s2 = D[(D.zCMB > 0.01) & (D.zCMB < 2.5)]; z2, mb2, dm2 = s2.zCMB.values, s2.m_b_corr.values, s2.m_b_corr_err_DIAG.values
zg2 = np.linspace(0, z2.max(), 3000)
def dl2(Om_, Hz, b):
    a = 1 / (1 + zg2); H = Hz * np.sqrt(Om_ * a**-3 + 1 - Om_ + b * np.exp(1 - 1 / a))
    dc = np.concatenate([[0], np.cumsum(0.5 * (c / H[1:] + c / H[:-1]) * np.diff(zg2))]); return (1 + z2) * np.interp(z2, zg2, dc)
def chi2(p, Hz):
    Om_, b, M = p
    if not (0.2 < Om_ < 0.4 and -0.3 < b < 0.3): return 1e10
    return np.sum(((mb2 - (M + 5 * np.log10(dl2(Om_, Hz, b)) + 25)) / dm2)**2)
Hs = np.array([62, 64, 66, 67.4, 69, 70, 71.5, 73.04, 74.5]); prof2 = []
for Hz in Hs:
    r = minimize(chi2, [0.315, 0.0, -19.3 + 5 * np.log10(Hz / 70)], args=(Hz,), method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
    prof2.append((r.fun, r.x[2] - 5 * np.log10(Hz)))
prof2 = np.array(prof2); print(f"{len(z2)} SNe; chi2 range {prof2[:,0].min():.3f}-{prof2[:,0].max():.3f}; M - 5log10 H0 range {prof2[:,1].min():.4f} to {prof2[:,1].max():.4f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.36))
a1.axhline(0, color=S.GR, lw=0.8, ls="--")
a1.plot(zz[srt], (modm - mod0)[srt], color=S.IAM, lw=1.4)
a1.plot(zz[srt], (mode - mod0)[srt], color=S.DATA, lw=1.0, ls="-.")
a1.errorbar(zb, rb, eb, fmt="o", color=S.DATA, ms=3.5, lw=0.8, capsize=1.5)
a1.text(0.012, 0.155, f"$\\beta_m$ on distances: $\\Delta\\chi^2$ = {chim-chi0:+.1f}", fontsize=7, color=S.IAM)
a1.text(0.012, 0.125, f"best $\\beta$ = $-$0.035: $\\Delta\\chi^2$ = $-${abs(chie-chi0):.1f}", fontsize=7, color=S.DATA)
a1.set_xscale("log"); a1.set_xlim(0.01, 2.4); a1.set_ylim(-0.2, 0.18)
a1.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 2]); a1.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1", "2"])
a1.set_xlabel("redshift $z_{\\rm HD}$"); a1.set_ylabel("residual from $\\Lambda$CDM (mag)")
a1.set_title("Hubble-flow shape: $\\Lambda$CDM geometry")
S.panel_letter(a1, "a", dx=-0.17)
a2.plot(Hs, prof2[:, 0], "o-", color=S.DATA, ms=3.5, lw=1.0)
for hv, nm, col in ((67.4, "Planck 67.4", S.GR), (73.04, "SH0ES 73.04", S.IAM)):
    a2.axvline(hv, color=col, lw=0.8, ls=":"); a2.text(hv + 0.2, prof2[:, 0].mean() + 1.6, nm, fontsize=7, color=col, rotation=90, va="bottom")
a2.set_ylim(prof2[:, 0].mean() - 4, prof2[:, 0].mean() + 6); a2.set_xlim(61, 75.5)
a2.text(61.5, prof2[:, 0].mean() - 3.2, f"$M-5\\log_{{10}}H_0$ = $-${abs(prof2[:,1].mean()):.3f} at every $H_0$", fontsize=7, color=S.GR)
a2.set_xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$), $M$ free"); a2.set_ylabel("minimum $\\chi^2$ (diagonal errors)")
a2.set_title("The magnitudes do not choose $H_0$")
S.panel_letter(a2, "b", dx=-0.17)
S.save(fig, "part2", "fig_sn_h0_flat")
