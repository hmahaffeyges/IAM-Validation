#!/usr/bin/env python3
"""Verification for Part 2, Chapters ch:latetime (p2_07_late_time_growth.tex, the MGCAMB runs, 'Level 1') and ch:level2
(p2_06_dual_sector_perturbation.tex, the modified-CAMB runs, 'Level 2'). Every number printed in those chapters is recomputed here
from the model algebra (sympy), from the chain files (30 % burn-in per file, weighted statistics, as mgcamb_validation/
CHAIN_EXTRACTION_FINAL.csv) or from the CAMB growth outputs docs/verification/chains/data/growth_on.json / growth_off.json.
Run from anywhere:  python docs/verification/scripts/verify_late_time_level2.py > docs/verification/scripts/verify_late_time_level2_output.txt
Corrections applied (docs/verification/PAPER_ERRATA.md): LG1-LG12, P1-P16; checks LATE_TIME_GROWTH_CHECK.md, DUAL_SECTOR_PERTURBATION_CHECK.md.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd, sympy as sp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[3]
OK = []
def check(name, val, ref, tol):
    good = abs(val - ref) <= tol
    OK.append(good)
    print(f"  [{'ok' if good else 'XX'}] {name}: {val:.6g} (expected {ref} +/- {tol})")

print("=" * 100); print("A. Model algebra (sympy)"); print("=" * 100)
a, b, Om, H0, E = sp.symbols("a beta Omega_m H_0 E", positive=True)
Ea = sp.exp(1 - 1 / a)
# ln E = -int_a^1 da'/a'^2 : the record density on the apparent horizon ~ 1/a^2 in matter domination, integrated from a to today
ap = sp.symbols("ap", positive=True)
lnE = -sp.integrate(1 / ap**2, (ap, a, 1))
print("  ln E(a) = -int_a^1 da'/a'^2 =", sp.simplify(lnE), "  -> E(a) = exp(1 - 1/a):", sp.simplify(sp.exp(lnE) - Ea) == 0)
print("  E(1) =", Ea.subs(a, 1), "; lim_{a->0} E =", sp.limit(Ea, a, 0, "+"), "; E(z) = e^{-z}:",
      sp.simplify(Ea.subs(a, 1 / (1 + sp.Symbol("z"))) - sp.exp(-sp.Symbol("z"))) == 0)
H2 = sp.Function("H2")(a)
Hm2 = H2 + b * Ea * H0**2
mu = H2 / Hm2
print("  mu = H^2/H_m^2 = H^2/(H^2 + beta E H0^2); at a = 1 with H(1) = H0:", sp.simplify(mu.subs(a, 1).subs(H2.subs(a, 1), H0**2)))
print("  H_m/H at a = 1:", sp.simplify(sp.sqrt(Hm2 / H2).subs(a, 1).subs(H2.subs(a, 1), H0**2)))
# MGCAMB DES form mu = 1 + mu0 Omega_DE(a)/Omega_L ; Omega_DE(1) = Omega_L
mu0 = sp.Symbol("mu0"); OL = 1 - Om
OmDE = OL / (Om * a**-3 + OL)
print("  MGCAMB: mu(a=1) - 1 =", sp.simplify((1 + mu0 * OmDE / OL).subs(a, 1) - 1), "; printed form without /Omega_L gives mu(1)-1 =",
      sp.simplify((mu0 * OmDE).subs(a, 1)), "(not mu0: LG1)")
print("  exact mu0 = 1/(1+beta) - 1 =", sp.simplify(1 / (1 + b) - 1))
# background-level test: H^2 = H0^2[Om a^-3 + Or a^-4 + OL + beta E]; flatness at a=1 then requires Om + Or + OL + beta = 1
print("  background form at a = 1: H^2/H0^2 = Omega_m + Omega_r + Omega_L + beta -> closure needs Omega_L reduced by beta")

print("=" * 100); print("B. Numbers from the coupling"); print("=" * 100)
Omv = 0.3153; bm = Omv / 2
check("beta_m = Omega_m/2", bm, 0.15765, 5e-6)
check("mu(0) = 1/(1+beta_m)", 1 / (1 + bm), 0.8638, 5e-5)
check("1 - mu(0) (coupling deficit, %)", 100 * (1 - 1 / (1 + bm)), 13.62, 0.005)
check("sqrt(1+beta_m)", np.sqrt(1 + bm), 1.0759, 5e-5)
check("mu(0) with Omega_m = 0.315 (rounded)", 1 / (1 + 0.315 / 2), 0.8640, 5e-4)
check("exact mu0 = -beta/(1+beta)", -bm / (1 + bm), -0.13618, 5e-6)
b135 = 0.13495 / (1 - 0.13495)
check("beta giving the run value mu0 = -0.13495", b135, 0.1560, 5e-5)
check("difference of mu0 values", -0.13495 - (-bm / (1 + bm)), 0.0012, 1e-4)
check("as a fraction of sigma(mu0) = 0.125 (free run E)", (0.13618 - 0.13495) / 0.125, 0.01, 0.002)
Hz2 = lambda z, om=Omv: om * (1 + z)**3 + 1 - om
mu_ex = lambda z, om=Omv, bb=bm: Hz2(z, om) / (Hz2(z, om) + bb * np.exp(-z))
mu_mg = lambda z, om=Omv, m0=-0.13495: 1 + m0 * ((1 - om) / Hz2(z, om)) / (1 - om)
print("  Table (z, a, E = e^-z, mu exact, H_m/H), Omega_m = 0.3153 [P3-corrected]:")
for z in (0, 0.2, 0.5, 1, 2, 3, 5):
    print(f"    z {z:3}: a {1/(1+z):.3f}  E {np.exp(-z):.4f}  mu {mu_ex(z):.4f}  H_m/H {np.sqrt(1/mu_ex(z)):.4f}  MGCAMB mu {mu_mg(z):.4f}")
check("mu(z=0.2)", mu_ex(0.2), 0.905, 5e-4); check("mu(z=0.5)", mu_ex(0.5), 0.948, 5e-4)
check("H_m/H z=0.2", np.sqrt(1 / mu_ex(0.2)), 1.051, 5e-4); check("H_m/H z=0.5", np.sqrt(1 / mu_ex(0.5)), 1.027, 5e-4)
zz = np.linspace(0, 6, 60001); gap = (mu_ex(zz) - mu_mg(zz)) / mu_ex(zz)
i = np.argmax(gap)
check("max (mu_exact - mu_MGCAMB)/mu_exact (%)", 100 * gap[i], 2.8, 0.05); print(f"    at z = {zz[i]:.2f}")
check("gap at z = 1 (%)", 100 * (mu_ex(1) - mu_mg(1)) / mu_ex(1), 2.5, 0.05)
print(f"    gap at z = 0.5: {100*(mu_ex(0.5)-mu_mg(0.5))/mu_ex(0.5):.2f} %, z = 2: {100*(mu_ex(2)-mu_mg(2))/mu_ex(2):.2f} %, "
      f"largest z with gap >= 1 %: {zz[gap >= 0.01].max():.2f}; mu values at z = 0: exact {mu_ex(0):.4f}, MGCAMB {mu_mg(0):.4f}")
print(f"    E(a) at z = 3: {np.exp(-3):.4f} ; mu(3) = {mu_ex(3):.5f} ; mu(5) = {mu_ex(5):.6f}")

print("=" * 100); print("C. Level 1 chains (MGCAMB), final files, 30 % burn-in, weighted"); print("=" * 100)
def load(files, burn=0.3):
    out = []
    for f in files:
        p = REPO / f; cols = open(p).readline().lstrip("#").split()
        X = pd.read_csv(p, sep=r"\s+", comment="#", names=cols); out.append(X.iloc[int(burn * len(X)):])
    return pd.concat(out, ignore_index=True)
def ms(X, c):
    m = np.average(X[c], weights=X.weight); return m, np.sqrt(np.average((X[c] - m)**2, weights=X.weight))
def wq(v, w, q):
    i = np.argsort(v); v, w = np.asarray(v)[i], np.asarray(w)[i]; cw = (np.cumsum(w) - 0.5 * w) / w.sum(); return np.interp(q, cw, v)
def rminus1(prog):
    last = [l for l in open(REPO / prog) if l.strip() and not l.startswith("#")][-1].split(); return float(last[3])
MG = "mgcamb_validation/chains/"
L1 = {"Planck": ("lcdm_baseline", ["iam_fixed_mu0_r2"], ["iam_float_mu0_r2"]),
      "Planck + RSD": ("planck_rsd_lcdm_baseline", ["planck_rsd_iam_fixed"], ["planck_rsd_mu0_float"]),
      "Planck + BAO": ("planck_bao_lcdm_baseline", ["planck_bao_iam_fixed"], ["planck_bao_mu0_float"]),
      "Planck + Pantheon+": ("planck_pantheon_lcdm_baseline", ["planck_pantheon_iam_fixed"], ["planck_pantheon_mu0_float"])}
def files(stem):
    n = 4 if stem.endswith("_r2") else 1
    return [MG + f"{stem}.{k}.txt" for k in range(1, n + 1)]
pars = ["H0", "sigma8", "ombh2", "omch2", "omegal", "tau", "ns", "logA"]
L1res = {}
for data, (lc, fx, fl) in L1.items():
    C, F, Fr = load(files(lc)), load(files(fx[0])), load(files(fl[0]))
    print(f"  --- {data}: samples (after burn-in) LCDM {len(C)}, fixed {len(F)}, free {len(Fr)}")
    for p in pars:
        if p in C:
            print(f"    {p:7s}  LCDM {ms(C,p)[0]:.5f} +/- {ms(C,p)[1]:.5f} | fixed {ms(F,p)[0]:.5f} +/- {ms(F,p)[1]:.5f} | free {ms(Fr,p)[0]:.5f} +/- {ms(Fr,p)[1]:.5f}")
    for X, nm in ((C, "LCDM"), (F, "fixed"), (Fr, "free")):
        if "omegal" in X:
            om = 1 - X.omegal; mo = np.average(om, weights=X.weight)
            s8 = X.sigma8 * np.sqrt(om / 0.3); print(f"    {nm}: Omega_m = 1 - Omega_L = {mo:.4f}; S8 = {np.average(s8, weights=X.weight):.4f}")
    c2c, c2f, c2r = C.chi2.min(), F.chi2.min(), Fr.chi2.min()
    d = c2f - c2c; L1res[data] = d
    m0 = Fr.mu0.values; w = Fr.weight.values
    med, q10, q05, q95 = wq(m0, w, 0.5), wq(m0, w, 0.10), wq(m0, w, 0.05), wq(m0, w, 0.95)
    pbelow = w[m0 < -0.135].sum() / w.sum(); pabove = w[m0 > 0.15].sum() / w.sum()
    mm, ss = ms(Fr, "mu0")
    print(f"    chi2_min LCDM {c2c:.3f}, fixed {c2f:.3f}, free {c2r:.3f};  Delta chi2 (fixed - LCDM) = {d:+.2f}; likelihood ratio e^(-D/2) = {np.exp(-d/2):.2f}")
    print(f"    Delta sigma8 = {ms(F,'sigma8')[0]-ms(C,'sigma8')[0]:+.4f} ({100*(ms(F,'sigma8')[0]/ms(C,'sigma8')[0]-1):+.2f} %)")
    print(f"    free mu0: mean {mm:+.3f} +/- {ss:.3f}; median {med:+.3f}; 10 % quantile {q10:+.3f}; 5 % quantile {q05:+.3f}; 95 % quantile {q95:+.3f}; P(mu0 < -0.135) = {pbelow:.2f};"
          f" P(mu0 > 0.15) = {pabove:.2f}; (mean - (-0.135))/sd = {(mm+0.135)/ss:.2f}; mean/sd = {mm/ss:.2f}")
    for stem in (lc, fx[0], fl[0]):
        print(f"    final R-1 {stem}: {rminus1(MG + stem + '.progress'):.4f}")
check("L1 Planck Delta chi2", L1res["Planck"], 0.96, 0.01); check("L1 Planck+RSD Delta chi2", L1res["Planck + RSD"], 0.56, 0.01)
check("L1 Planck+BAO Delta chi2", L1res["Planck + BAO"], 1.73, 0.01); check("L1 Planck+Pantheon+ Delta chi2", L1res["Planck + Pantheon+"], 1.58, 0.01)

print("=" * 100); print("D. Level 2 chains (CAMB 1.5.8, equations_iam_level2.f90)"); print("=" * 100)
CV = "camb_validation/chains/"
A, Cc, D = load([CV + "iam_level2_runA.1.txt"]), load([CV + "iam_level2_runC_lcdm.1.txt"]), load([CV + "iam_level2_runD.1.txt"])
p2 = [("H0", "H0"), ("sigma8", "sigma8"), ("S8", "S8"), ("ombh2", "omega_b"), ("omch2", "omega_c"), ("tau", "tau"), ("ns", "n_s"), ("logA", "ln10^10As"), ("omegam", "Omega_m")]
for c, lab in p2:
    (mc, sc), (ma, sa), (md, sd) = ms(Cc, c), ms(A, c), ms(D, c)
    print(f"    {lab:10s} C {mc:.5f} +/- {sc:.5f} | A {ma:.5f} +/- {sa:.5f} | shift A-C {(ma-mc)/sc:+.2f} sigma | D {md:.5f} +/- {sd:.5f} | D-A {(md-ma)/sa:+.2f} | D-C {(md-mc)/sc:+.2f}")
check("L2 sigma8 shift (sigma)", (ms(A,'sigma8')[0]-ms(Cc,'sigma8')[0])/ms(Cc,'sigma8')[1], -1.51, 0.01)
check("L2 S8 shift (sigma)", (ms(A,'S8')[0]-ms(Cc,'S8')[0])/ms(Cc,'S8')[1], -0.78, 0.01)
ds8 = ms(A,'sigma8')[0] - ms(Cc,'sigma8')[0]
print(f"    Delta sigma8 = {ds8:+.4f} = {100*ds8/ms(Cc,'sigma8')[0]:+.2f} %  [P13: 1.1 %, not 1.5 %]")
print(f"    lowest chi2: C {Cc.chi2.min():.2f}, A {A.chi2.min():.2f}, D {D.chi2.min():.2f}; Delta(A-C) = {A.chi2.min()-Cc.chi2.min():+.2f}; likelihood ratio {np.exp(-(A.chi2.min()-Cc.chi2.min())/2):.2f}")
cav = lambda X, c="chi2": np.average(X[c], weights=X.weight)
print(f"    chain-average chi2: C {cav(Cc):.2f}, A {cav(A):.2f}; Delta = {cav(A)-cav(Cc):+.2f}")
print(f"    chain-average chi2 minus lowest: C {cav(Cc)-Cc.chi2.min():.1f}, A {cav(A)-A.chi2.min():.1f}")
print(f"    Run D chain averages: chi2 {cav(D):.2f}, CMB {cav(D,'chi2__CMB'):.2f}, RSD {cav(D,'chi2__iam_rsd'):.2f}; at the best point RSD "
      f"{D.loc[D.chi2.idxmin(),'chi2__iam_rsd']:.2f}; minimum RSD {D.chi2__iam_rsd.min():.2f}")
for r in ("iam_level2_runA", "iam_level2_runC_lcdm", "iam_level2_runD", "iam_l2b_runA", "iam_l2b_runD"):
    print(f"    final R-1 {r}: {rminus1(CV + r + '.progress'):.4f}")
m, s = ms(A, "omegam"); print(f"    Omega_m (Run A) = {m:.4f} +/- {s:.4f}; Omega_m/2 = {m/2:.4f} +/- {s/2:.4f}; (Omega_m/2 - 0.15765)/(sd/2) = {(m/2-0.15765)/(s/2):.2f}; "
                             f"relative {100*(m/2/0.15765-1):.1f} %")
H0A, sH0A = ms(A, "H0")
check("H0(matter) = H0_A sqrt(1+beta_m)", H0A * np.sqrt(1 + bm), 72.26, 0.005)
check("its uncertainty", sH0A * np.sqrt(1 + bm), 0.50, 0.005)
check("photon vs Planck (67.36 +/- 0.54)", (67.16 - 67.36) / 0.54, -0.37, 0.005)
check("matter vs SH0ES (73.04 +/- 1.04)", (72.26 - 73.04) / 1.04, -0.75, 0.005)
check("Planck 67.4 +/- 0.5 vs SH0ES 73.04 +/- 1.04 (sigma)", (73.04 - 67.4) / np.hypot(0.5, 1.04), 4.9, 0.05)
st = np.sqrt(1.22**2 + 1.33**2 + 0.70**2)
print(f"    TRGB (Freedman 2025) 70.39 +/- {st:.2f}: below matter by {(72.26-70.39)/st:.2f} sigma, above photon by {(70.39-67.16)/st:.2f} sigma")
OmA = m
Hz = lambda z: H0A * np.sqrt(OmA * (1 + z)**3 + 1 - OmA)
Hmz = lambda z: np.sqrt(Hz(z)**2 + bm * np.exp(-z) * H0A**2)
print("    Table: Run A posterior (H0 %.3f, Omega_m %.4f), beta_m 0.15765 [P3-corrected]" % (H0A, OmA))
for z in (0, 0.5, 1, 2, 3, 5):
    print(f"      z {z}: H {Hz(z):.2f}  H_m {Hmz(z):.2f}  ratio {Hmz(z)/Hz(z):.4f}  E {np.exp(-z):.6f}  (H_m/H - 1) {100*(Hmz(z)/Hz(z)-1):.3f} %")
Bb, Bd = load([CV + "iam_l2b_runA.1.txt"]), load([CV + "iam_l2b_runD.1.txt"])
for X, nm in ((Bb, "L2b A (background, Planck)"), (Bd, "L2b D (background, Planck + growth)")):
    h, sh = ms(X, "H0"); o, so = ms(X, "omegam"); s8, ss8 = ms(X, "sigma8")
    print(f"    {nm}: H0 {h:.2f} +/- {sh:.2f}; Omega_m {o:.4f} +/- {so:.4f}; sigma8 {s8:.4f} +/- {ss8:.4f}; S8 {ms(X,'S8')[0]:.3f}; lowest chi2 {X.chi2.min():.2f}"
          f" (vs C {Cc.chi2.min():.2f}: {X.chi2.min()-Cc.chi2.min():+.2f}); CMB chi2 lowest {X.chi2__CMB.min():.2f}")
    print(f"       (67.36 - H0)/0.54 = {(67.36-h)/0.54:.1f}; in quadrature with own sd {(67.36-h)/np.hypot(0.54,sh):.1f}; vs SH0ES (73.04-H0)/hypot(1.04,sd) = {(73.04-h)/np.hypot(1.04,sh):.1f};"
          f" vs Run C: H0 {(h-ms(Cc,'H0')[0])/ms(Cc,'H0')[1]:+.1f} sigma, Omega_m {(o-ms(Cc,'omegam')[0])/ms(Cc,'omegam')[1]:+.1f} sigma")
h = ms(Bb, "H0")[0]; check("background run H0 vs Planck in Planck sigma", (67.36 - h) / 0.54, 10.9, 0.05)
# What the background chains actually coded (camb_validation/prepare_level2b.sh): dtauda's grhoa2 = 8 pi G rho a^4 (CAMB comment), and the patch adds
# 0.15765 * E(a) * a^2 * grho0, i.e. rho_extra a^4 = beta E a^2 rho0  ->  Delta H^2 = beta_m E(a) H0^2 / a^2 (not beta_m E(a) H0^2), with Omega_L unchanged.
asym = sp.Symbol("a", positive=True)
print("    background patch: Delta(8piG rho a^4) = beta E a^2 rho0 -> Delta H^2/H0^2 =", sp.simplify(b * Ea * asym**2 / asym**4), "; at a = 1:", sp.simplify((b*Ea/a**2).subs(a, 1)))
for X, nm in ((Bb, "A_b"), (Bd, "D_b")):
    h, sh = ms(X, "H0"); o, so = ms(X, "omegam")
    print(f"    {nm}: sampled H0 {h:.2f} +/- {sh:.2f} -> expansion rate today H(z=0) = H0 sqrt(1+beta_m) = {h*np.sqrt(1+bm):.2f} +/- {sh*np.sqrt(1+bm):.2f}; "
          f"matter fraction today Omega_m/(1+beta_m) = {o/(1+bm):.4f}; (67.36 - H(0))/0.54 = {(67.36-h*np.sqrt(1+bm))/0.54:.1f}; vs Run C H0: {(h*np.sqrt(1+bm)-ms(Cc,'H0')[0])/ms(Cc,'H0')[1]:+.1f} sigma")
print(f"    E(a)/a^2 at a = 0.5: {np.exp(-1)/0.25:.3f} (E alone {np.exp(-1):.3f}); max of E/a^2 at a = 0.5: {max(np.exp(1-1/x)/x**2 for x in np.linspace(0.05,1,2000)):.3f}")

print("=" * 100); print("E. Growth of the coded mechanism (CAMB built from equations_iam_level2.f90, switch on/off, Run A means)"); print("=" * 100)
on = json.load(open(REPO / "docs/verification/chains/data/growth_on.json")); off = json.load(open(REPO / "docs/verification/chains/data/growth_off.json"))
z = np.array(on["z"]); s_on, s_off = np.array(on["s8"]), np.array(off["s8"]); f_on, f_off = np.array(on["fs8"]), np.array(off["fs8"])
lna = -np.log1p(z)
from scipy.interpolate import CubicSpline
dln_on = CubicSpline(lna[::-1], np.log(s_on[::-1]))(lna, 1); dln_off = CubicSpline(lna[::-1], np.log(s_off[::-1]))(lna, 1)
for k in range(len(z)):
    print(f"    z {z[k]:4}: sigma8 on/off {s_on[k]/s_off[k]:.4f} | f = dln sigma8/dln a on {dln_on[k]:.3f} off {dln_off[k]:.3f} | CAMB fs8/s8 on {f_on[k]/s_on[k]:.3f} off {f_off[k]/s_off[k]:.3f}"
          f" | fs8 (density) on/off {dln_on[k]*s_on[k]/(dln_off[k]*s_off[k]):.3f} | CAMB fs8 on/off {f_on[k]/f_off[k]:.3f}")
check("sigma8 ratio code z=0", s_on[0] / s_off[0], 0.9880, 5e-4)
check("CAMB f / density f, switch on, z=0", (f_on[0] / s_on[0]) / dln_on[0], 1.088, 0.01)
# closed-form mu applied as G_eff = mu G, same primordial amplitude, Run A Omega_m
def growth(mu, om, zs):
    def rhs(l, y):
        a = np.exp(l); h2 = om / a**3 + 1 - om; dl = -1.5 * om / a**3 / h2
        return [y[1], -(2 + dl) * y[1] + 1.5 * om / a**3 / h2 * mu(a) * y[0]]
    s = solve_ivp(rhs, [np.log(1e-3), 0], [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-13)
    y = s.sol(-np.log1p(np.asarray(zs, float))); return y[0], y[1]
mua = lambda a, om=OmA: (om / a**3 + 1 - om) / (om / a**3 + 1 - om + bm * np.exp(1 - 1 / a))
Dg, _ = growth(lambda a: 1.0, OmA, [0, 0.5, 1]); Dm, _ = growth(mua, OmA, [0, 0.5, 1])
print("    D ratio closed-form mu (z = 0, 0.5, 1):", np.round(Dm / Dg, 4))
check("closed-form D ratio z=0", (Dm / Dg)[0], 0.9922, 5e-4)
pt = (0.315 - np.interp(0.85, z, f_off)) / 0.095
print(f"    fsigma8 data z = 0.85: 0.315 +/- 0.095; LCDM (switch off, Run A means) {np.interp(0.85, z, f_off):.3f}; pull {pt:+.2f} sigma")

print("=" * 100); print("F. f sigma8 deficit and Euclid template equivalence (as verify_euclid_template.py)"); print("=" * 100)
Om0 = 0.3153
def fD(mu, zs):
    _, y1 = growth(mu, Om0, zs); return y1
mu_i = lambda a: (Om0 / a**3 + 1 - Om0) / (Om0 / a**3 + 1 - Om0 + bm * np.exp(1 - 1 / a))
mu_t = lambda a, m0: 1 + m0 * ((1 - Om0) / (Om0 / a**3 + 1 - Om0)) / (1 - Om0)
deficit = lambda mu, zs: 1 - fD(mu, zs) / fD(lambda a: 1.0, zs)
d = 100 * deficit(mu_i, [0, 0.3, 0.5, 1.0]); print("    IAM f sigma8 deficit % (z 0, 0.3, 0.5, 1):", np.round(d, 2))
check("f sigma8 deficit z=0 (%)", d[0], 4.25, 0.01)
print("    MGCAMB-form (mu0 = -0.136) deficit %:", np.round(100 * deficit(lambda a: mu_t(a, -0.136), [0, 0.3, 0.5, 1.0]), 2))
print(f"    mu(z=1): IAM {mu_i(0.5):.3f}, template {mu_t(0.5, -0.136):.3f}")
for name, zs in (("spectroscopic 0.9-1.7", np.array([0.9, 1.1, 1.3, 1.5, 1.7])), ("photometric 0.2-2.0", np.linspace(0.2, 2.0, 10)), ("all 0-2", np.linspace(0, 2, 21))):
    dI = deficit(mu_i, zs); t = deficit(lambda a: mu_t(a, -0.1), zs)
    meq = brentq(lambda m: np.sum((deficit(lambda a: mu_t(a, m), zs) - dI) * t), -0.5, 0.0)
    print(f"    {name:24s} template-equivalent mu0 {meq:+.3f}: /0.233 = {abs(meq)/0.233:.2f}, /0.04 = {abs(meq)/0.04:.1f}, /0.01 = {abs(meq)/0.01:.0f} sigma")
print(f"    present precision: 0.136/0.125 = {0.136/0.125:.2f} sigma from GR (MGCAMB form, chain E)")
# DESI Y5 growth-only, the forecast script's assumed per-bin errors, IAM mu(z) itself (same primordial amplitude)
zd = np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330]); sd = np.array([0.020, 0.015, 0.012, 0.015, 0.020, 0.022, 0.035])
fs_g = fD(lambda a: 1.0, zd); fs_i = fD(mu_i, zd); norm = 0.8087 / fD(lambda a: 1.0, [0])[0] * fD(lambda a: 1.0, [0])[0]
_, f0 = growth(lambda a: 1.0, Om0, [0]); D0, _ = growth(lambda a: 1.0, Om0, [0])
scale = 0.8087 / D0[0]   # sigma8 today of the LambdaCDM run C, same primordial amplitude for both
chi = np.sqrt(np.sum(((fs_i - fs_g) * scale / sd)**2))
print(f"    DESI Y5 growth-only (assumed errors {sd.tolist()}): LCDM fs8 {np.round(fs_g*scale,3).tolist()}, IAM {np.round(fs_i*scale,3).tolist()}; "
      f"distinction sqrt(chi2) = {chi:.2f} sigma")

print("=" * 100); print(f"SUMMARY: {sum(OK)} of {len(OK)} checks pass"); print("=" * 100)
