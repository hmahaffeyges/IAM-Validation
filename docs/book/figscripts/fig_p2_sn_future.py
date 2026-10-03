"""Part 2, Chapters 'Type Ia supernovae in the dual-sector picture' (p2_10_dual_sector_validation.tex) and 'The equation of state of the record'
(p2_11_dark_energy.tex).
fig_sn_beta: Pantheon+SH0ES (Brout et al. 2022) Hubble flow z_HD > 0.01 (1590 SNe), full STAT+SYS covariance, M marginalised analytically,
Omega_m = 0.315: chi^2 profile in beta when beta E(a) is put into the distances (method of docs/verification/scripts/verify_dual_sector_validation.py).
The data are downloaded once from github.com/PantheonPlusSH0ES/DataRelease into figscripts/_data/ (or set PANTHEON_DIR).
fig_two_rulers_future: H(a) photon and H_m(a) matter into the future (H0 = 67.4, Om = 0.315, OL = 0.685, beta_m = 0.3153/2), with their ratio.
fig_maturity: the fraction of the eventual record written, E(a)/e, against cosmic time, and the writing rate per Gyr (verify_wz_far_future.py)."""
import sys, os, pathlib, urllib.request; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.integrate import quad
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
# ---------------- supernovae -----------------
DD = pathlib.Path(os.environ.get("PANTHEON_DIR", S.HERE / "_data"))
DD.mkdir(parents=True, exist_ok=True)
B = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
for f, u in (("PantheonPlusSH0ES.dat", "Pantheon%2BSH0ES.dat"), ("PantheonPlusSH0ES_STAT+SYS.cov", "Pantheon%2BSH0ES_STAT%2BSYS.cov")):
    if not (DD / f).exists():
        urllib.request.urlretrieve(B + u, DD / f)
D = pd.read_csv(DD / "PantheonPlusSH0ES.dat", sep=r"\s+"); c = 299792.458
raw = np.loadtxt(DD / "PantheonPlusSH0ES_STAT+SYS.cov"); N = int(raw[0]); Cv = raw[1:].reshape(N, N)
m = (D.zHD > 0.01).values; Ci = np.linalg.inv(Cv[np.ix_(m, m)]); zz, zh, y = D.zHD.values[m], D.zHEL.values[m], D.m_b_corr.values[m]
zg = np.linspace(0, zz.max() + 0.01, 4000)
def dl(Om, b):
    a = 1 / (1 + zg); H = 70.0 * np.sqrt(Om * a**-3 + 1 - Om + b * np.exp(1 - 1 / a))
    dc = np.concatenate([[0], np.cumsum(0.5 * (c / H[1:] + c / H[:-1]) * np.diff(zg))]); return (1 + zh) * np.interp(zz, zg, dc)
def chi(Om, b):
    r = y - (5 * np.log10(dl(Om, b)) + 25); o = np.ones_like(r); d = r - (o @ Ci @ r) / (o @ Ci @ o); return d @ Ci @ d
bs = np.linspace(-0.3, 0.3, 601); P = np.array([chi(0.315, b) for b in bs]); i = P.argmin()
c0, c1 = chi(0.315, 0.0), chi(0.315, 0.15765); ok = bs[P <= P[i] + 1]
print(f"{m.sum()} SNe; LCDM chi2 {c0:.2f}; beta_m on distances {c1:.2f}; dchi2 {c1-c0:+.2f}; best beta {bs[i]:+.3f} (68 %: {ok.min():+.3f} to {ok.max():+.3f})")
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.7))
ax.plot(bs, P - c0, color=S.DATA, lw=1.5)
ax.axvline(0, color=S.GR, lw=0.8, ls="--"); ax.axvline(0.15765, color=S.IAM, lw=0.9)
ax.plot(0.15765, c1 - c0, "o", color=S.IAM, ms=5)
ax.annotate(f"$\\beta_m$ on supernova distances:\n$\\Delta\\chi^2$ = {c1-c0:+.1f}", (0.15765, c1 - c0), xytext=(-8, 12), textcoords="offset points", fontsize=7, ha="right", color=S.IAM)
ax.axvspan(ok.min(), ok.max(), color=S.DATA, alpha=0.15, lw=0)
ax.text(-0.075, 57, f"best {bs[i]:+.3f}\n(shaded: 68 %)", fontsize=7, color=S.DATA, va="top", ha="right")
ax.text(0.005, 52, "$\\Lambda$CDM", fontsize=7, color=S.GR)
ax.set_xlim(-0.3, 0.3); ax.set_ylim(-3, 60)
ax.set_xlabel("$\\beta$ applied to distances"); ax.set_ylabel("$\\Delta\\chi^2$ relative to $\\Lambda$CDM")
ax.set_title("Supernova geometry is $\\Lambda$CDM")
S.save(fig, "part2", "fig_sn_beta")

# ---------------- far future, two rulers -----------------
H0 = 67.4; Om = 0.315; OL = 0.685; bm = 0.3153 / 2; Gyr = 977.792 / H0
E = lambda a: np.exp(1 - 1 / a); Hh = lambda a: np.sqrt(Om * a**-3 + OL)
aa = np.logspace(np.log10(0.3), 2, 400)
Hp = H0 * Hh(aa); Hm = H0 * np.sqrt(Hh(aa)**2 + bm * E(aa))
for q in (1, 2, 10, 1e6):
    print(f"a {q:g}: H {H0*Hh(q):.2f}  H_m {H0*np.sqrt(Hh(q)**2+bm*E(q)):.2f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(aa, Hp, color=S.GR, lw=1.4); a1.plot(aa, Hm, color=S.IAM, lw=1.6)
a1.axhline(H0 * np.sqrt(OL), color=S.GR, lw=0.7, ls=":"); a1.axhline(H0 * np.sqrt(OL + bm * np.e), color=S.IAM, lw=0.7, ls=":")
a1.text(95, H0 * np.sqrt(OL) - 1.0, f"photons → {H0*np.sqrt(OL):.1f}", fontsize=7, color=S.GR, ha="right", va="top")
a1.text(95, H0 * np.sqrt(OL + bm * np.e) + 1.0, f"matter → {H0*np.sqrt(OL+bm*np.e):.1f}", fontsize=7, color=S.IAM, ha="right", va="bottom")
a1.axvline(1, color=S.LIGHT, lw=0.6, ls=":"); a1.text(1.08, 105, "today", fontsize=7, color=S.GR)
a1.set_xscale("log"); a1.set_xticks([0.3, 1, 3, 10, 30, 100]); a1.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
a1.set_ylim(45, 115); a1.set_xlabel("scale factor $a$"); a1.set_ylabel("$H$ (km s$^{-1}$ Mpc$^{-1}$)")
a1.set_title("The two rulers diverge")
S.panel_letter(a1, "a", dx=-0.16)
a2.plot(aa, Hm / Hp, color=S.IAM, lw=1.6); a2.axhline(1, color=S.GR, lw=0.8, ls="--")
for q in (1, 2, 10):
    r = np.sqrt(Hh(q)**2 + bm * E(q)) / Hh(q); a2.plot(q, r, "o", color=S.IAM, ms=4)
    a2.annotate(f"{r:.3f}", (q, r), xytext=(4, -9), textcoords="offset points", fontsize=7)
rinf = np.sqrt(OL + bm * np.e) / np.sqrt(OL); a2.axhline(rinf, color=S.IAM, lw=0.7, ls=":")
a2.text(0.32, rinf + 0.005, f"limit {rinf:.3f}", fontsize=7, color=S.IAM, va="bottom")
a2.set_xscale("log"); a2.set_xticks([0.3, 1, 3, 10, 30, 100]); a2.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
a2.set_ylim(0.99, 1.31); a2.set_xlabel("scale factor $a$"); a2.set_ylabel("$H_m/H$")
a2.set_title("Ratio of the matter and photon rates")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_two_rulers_future")

# ---------------- maturity -----------------
t = lambda a: quad(lambda x: 1 / (x * Hh(x)), 1e-8, a, limit=200)[0] * Gyr
ag = np.logspace(np.log10(0.15), 2, 220); tg = np.array([t(q) for q in ag]); t0 = t(1.0)
frac = 100 * E(ag) / np.e; rate = 100 * E(ag) / ag * Hh(ag) / np.e / Gyr     # % of the eventual record per Gyr
from scipy.optimize import minimize_scalar
apk = minimize_scalar(lambda q: -E(q) / q * Hh(q), bounds=(0.05, 5), method='bounded').x
rpk = 100 * E(apk) / apk * Hh(apk) / np.e / Gyr; tpk = t(apk)
print(f"today: {100/np.e:.1f} % written, rate {100*Hh(1)/np.e/Gyr:.3f} %/Gyr; peak {rpk:.3f} %/Gyr at a {apk:.3f} (z {1/apk-1:.2f}), t {tpk:.1f} Gyr")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(tg, frac, color=S.IAM, lw=1.6)
for fr in (0.10, 1 / np.e, 0.5, 0.75, 0.9):
    q = -1 / np.log(fr); tq = t(q); a1.plot(tq, 100 * fr, "o", color=S.IAM, ms=4)
    a1.annotate(f"{100*fr:.0f} % at {tq:.1f} Gyr" if fr != 1 / np.e else f"today: {100*fr:.1f} %, {tq:.1f} Gyr", (tq, 100 * fr),
                xytext=(7, -4), textcoords="offset points", fontsize=7, va="top")
a1.set_xlim(0, 100); a1.set_ylim(0, 105)
a1.set_xlabel("cosmic time (Gyr)"); a1.set_ylabel("record written, $E(a)/e$ (%)")
a1.set_title("Today the record is $1/e$ written")
S.panel_letter(a1, "a", dx=-0.16)
a2.plot(tg, rate, color=S.IAM, lw=1.6)
a2.plot(tpk, rpk, "o", color=S.IAM, ms=4); a2.annotate(f"peak {rpk:.2f} %/Gyr at $z$ = {1/apk-1:.2f}", (tpk, rpk), xytext=(8, 2), textcoords="offset points", fontsize=7)
r0 = 100 * Hh(1) / np.e / Gyr; a2.plot(t0, r0, "o", mfc="white", mec=S.IAM, ms=5); a2.annotate(f"today {r0:.2f} %/Gyr", (t0, r0), xytext=(8, 0), textcoords="offset points", fontsize=7, va="center")
a2.set_xlim(0, 100); a2.set_ylim(0, 3.9)
a2.set_xlabel("cosmic time (Gyr)"); a2.set_ylabel("writing rate (% per Gyr)")
a2.set_title("Written fastest when galaxies assembled")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_maturity")
