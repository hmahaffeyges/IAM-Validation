#!/usr/bin/env python3
"""Verification for the dual-sector chapters of Part 2 (ch:dual p2_04, ch:dsnote p2_05, ch:dsvalidation p2_10).

Every number written into those chapters from the supernova analysis, the dual-sector algebra and the chain record is
recomputed here. Sections:
  0. sympy: H0-M degeneracy, activation function, Omega_m dilution, mu(a), matter-sector H0, S8 definition
  1. Dual-sector numbers (mu, E, H_m/H, effective matter fraction, H0 split, distances to SH0ES / TRGB / Planck)
  2. Pantheon+ in the setup of the three tests (zCMB 0.01-2.5, diagonal errors): Tests A, B, C with Nelder-Mead,
     Powell, L-BFGS-B; the H0 profile; the beta profile; the (H0, beta) grid
  3. Pantheon+ full STAT+SYS covariance (zHD > 0.01): Delta chi2 of beta_m on distances, best beta, Omega_m free,
     Omega_m variation, redshift bins, sample-size convergence (diagonal), distance-shape change of beta_m in H(z)
  4. Chain record from Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv (Delta chi2, sigma8, H0, S8, R-1, samples)
Data: public Pantheon+ release (github.com/PantheonPlusSH0ES/DataRelease), downloaded into the working directory if absent.
Writes verify_dual_sector_chapters_data.json (profiles and grids used by docs/book/figscripts/fig_p2_dual_sector.py).
numpy, scipy, pandas, sympy. Runtime about two minutes.
"""
import os, json, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, sympy as sp
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
c = 299792.458

# ---------------------------------------------------------------- 0. algebra
print("0. Algebra (sympy)")
M, H0, lam, z, a, rho, beta, Om = sp.symbols("M H0 lambda z a rho beta Omega_m", positive=True)
I = sp.Function("I")(z)                       # dimensionless comoving integral, d_L = (1+z) (c/H0) I(z)
m_model = M + 5*sp.log((1+z)*c/H0*I, 10) + 25
shift = sp.simplify(m_model.subs({M: M + 5*sp.log(lam, 10), H0: lam*H0}) - m_model)
print("   m(M + 5 log10 lambda, lambda H0) - m(M, H0) =", sp.simplify(sp.expand_log(shift, force=True)))
f = sp.Function("f")
sol = sp.dsolve(sp.Eq(f(a).diff(a), f(a)/a**2), f(a))
print("   d rho/da = rho/a^2  ->", sol)
Ea = sp.exp(1 - 1/a)
print("   E(1) =", Ea.subs(a, 1), "; d2E/da2 = 0 at a =", sp.solve(sp.diff(Ea, a, 2), a),
      "; d(E/a)/da = 0 at a =", sp.solve(sp.diff(Ea/a, a), a), "; E as a->oo:", sp.limit(Ea, a, sp.oo))
H2L = Om*a**-3 + (1 - Om)
mu_expr = H2L/(H2L + beta*Ea)
Omd = (Om*a**-3/(H2L + beta*Ea))/(Om*a**-3/H2L)
print("   Omega_m(a;beta)/Omega_m(a;0) - mu(a) =", sp.simplify(Omd - mu_expr))
print("   mu(a=1) =", sp.simplify(mu_expr.subs(a, 1)), "; 1 - mu(1) =", sp.simplify(1 - mu_expr.subs(a, 1)))
print("   H_m(a=1)/H0 =", sp.sqrt(sp.simplify((H2L + beta*Ea).subs(a, 1))))

# ---------------------------------------------------------------- 1. dual-sector numbers
print("1. Dual-sector numbers")
Om0, bm = 0.3153, 0.3153/2
Ef = lambda zz: np.exp(-zz)
H2 = lambda zz: Om0*(1+zz)**3 + 1 - Om0
muf = lambda zz: H2(zz)/(H2(zz) + bm*Ef(zz))
print(f"   beta_m = {bm:.5f}; sqrt(1+beta_m) = {np.sqrt(1+bm):.5f}; 1 - mu0 = beta_m/(1+beta_m) = {bm/(1+bm):.4f}")
for zz in (0, 0.11, 0.2, 0.5, 0.69, 1, 2, 2.3, 3, 5):
    print(f"   z {zz:4}: E {Ef(zz):.4f}  mu {muf(zz):.4f}  H_m/H - 1 {100*(np.sqrt(1+bm*Ef(zz)/H2(zz))-1):5.2f} %  "
          f"Omega_m(a;b)/Omega_m(a;0) - 1 {100*(muf(zz)-1):6.2f} %")
for zz in (3, 5):
    print(f"   1 - mu at z = {zz}: {1-muf(zz):.1e}")
H0g, H0g_sd = 67.16105463603412, 0.46685719444116597   # Level 2 Run A (CHAIN_EXTRACTION_FINAL.csv)
H0m = H0g*np.sqrt(1+bm); H0m_sd = H0g_sd*np.sqrt(1+bm)
print(f"   H0 photon {H0g:.3f} +- {H0g_sd:.3f}; H0 matter {H0m:.3f} +- {H0m_sd:.3f}")
print(f"   photon vs Planck 2018 67.36 +- 0.54: {(H0g-67.36)/0.54:+.2f} sigma (Planck error only)")
print(f"   matter vs SH0ES 73.04 +- 1.04: {(H0m-73.04)/1.04:+.2f} sigma (SH0ES error), {(H0m-73.04)/np.hypot(1.04,H0m_sd):+.2f} sigma (both)")
print(f"   matter vs TRGB 70.39 +- 1.94: {(H0m-70.39)/1.94:+.2f} sigma")
print(f"   Test A effective H0 sqrt(1+beta): 67.4*sqrt(0.7) = {67.4*np.sqrt(0.7):.2f}; 73.04*sqrt(0.7) = {73.04*np.sqrt(0.7):.2f}")
print(f"   H0 split: (73.04/67.4)^2 - 1 = {(73.04/67.4)**2-1:.4f}; (72.26/67.16)^2 - 1 = {(72.26/67.16)**2-1:.4f}")

# ---------------------------------------------------------------- 2. Pantheon+, the three tests
os.chdir(os.environ.get("PANTHEON_DIR", "."))
B = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
for fn, u in (("PantheonPlusSH0ES.dat", "Pantheon%2BSH0ES.dat"), ("PantheonPlusSH0ES_STAT+SYS.cov", "Pantheon%2BSH0ES_STAT%2BSYS.cov")):
    if not os.path.exists(fn): urllib.request.urlretrieve(B + u, fn)
D = pd.read_csv("PantheonPlusSH0ES.dat", sep=r"\s+")
print(f"2. Pantheon+ release: {len(D)} light curves, z_HD {D.zHD.min():.5f} to {D.zHD.max():.4f}")
s = D[(D.zCMB > 0.01) & (D.zCMB < 2.5)]
zs, mb, dm = s.zCMB.values, s.m_b_corr.values, s.m_b_corr_err_DIAG.values
print(f"   0.01 < zCMB < 2.5: {len(zs)} SNe, z max {zs.max():.4f}, median sigma_mb {np.median(dm):.3f} mag")
zg = np.linspace(0, 2.4, 4000)
def dc_grid(Omm, b):
    aa = 1/(1+zg); H = np.sqrt(Omm*aa**-3 + 1-Omm + b*np.exp(1-1/aa))
    return np.concatenate([[0], np.cumsum(0.5*(1/H[1:] + 1/H[:-1])*np.diff(zg))])
def dl(Omm, H0v, b, zz, zh=None):   # Mpc
    return (1+(zz if zh is None else zh))*c/H0v*np.interp(zz, zg, dc_grid(Omm, b))
def chi2(p, prior=None, walls=True):
    Omm, H0v, b, Mm = p
    if walls and not (0.2 < Omm < 0.4 and 60 < H0v < 75 and -0.3 < b < 0.3 and -20 < Mm < -18): return 1e10
    r = np.sum(((mb - (Mm + 5*np.log10(dl(Omm, H0v, b, zs)) + 25))/dm)**2)
    return r + (((H0v-prior[0])/prior[1])**2 if prior else 0)
tests = (("A Planck prior", (67.4, 0.5), [0.315, 67.4, 0, -19.3]), ("B SH0ES prior", (73.04, 1.04), [0.315, 73.04, 0, -19.3]),
         ("C no prior", None, [0.315, 70, 0, -19.3]))
OUT = {}
for nm, pr, x0 in tests:
    r = minimize(chi2, x0, args=(pr,), method="Nelder-Mead", options=dict(maxiter=5000, xatol=1e-6, fatol=1e-6))
    print(f"   Nelder-Mead {nm:15s} Om {r.x[0]:.4f} H0 {r.x[1]:.2f} beta {r.x[2]:+.4f} M {r.x[3]:.4f} chi2 {r.fun:.2f} chi2/dof {r.fun/(len(zs)-4):.4f}")
bnds = [(0.2, 0.4), (60, 75), (-0.3, 0.3), (-20, -18)]
for meth in ("Powell", "L-BFGS-B"):
    for nm, pr, x0 in tests[:2]:
        r = minimize(lambda p: chi2(p, pr, walls=False), x0, method=meth, bounds=bnds)
        print(f"   {meth:11s} {nm:15s} Om {r.x[0]:.4f} H0 {r.x[1]:.2f} beta {r.x[2]:+.4f} M {r.x[3]:.4f} chi2 {r.fun:.2f}")
# analytic offset: chi2 depends on (Om, beta) and on the single offset M - 5 log10 H0
w = 1/dm**2
def chi2_shape(Omm, b, zz=zs, y=mb, ww=w):
    r = y - 5*np.log10(dl(Omm, 70.0, b, zz))
    off = np.sum(ww*r)/np.sum(ww); return np.sum(ww*(r-off)**2), off
print("   profile in H0 (minimised over Om, beta, M):")
for H0v in (64, 67.4, 70, 73.04):
    r = minimize(lambda q: chi2([q[0], H0v, q[1], q[2]]), [0.315, 0, -19.3+5*np.log10(H0v/70)], method="Nelder-Mead",
                 options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
    print(f"   H0 {H0v:5.2f}: chi2 {r.fun:.4f}   M - 5 log10 H0 = {r.x[2]-5*np.log10(H0v):.4f}")
Omgrid = np.linspace(0.2, 0.4, 161); bgrid = np.round(np.linspace(-0.3, 0.3, 61), 4)
prof_b, prof_Om = [], []
for b in bgrid:
    vals = [chi2_shape(o, b)[0] for o in Omgrid]; i = int(np.argmin(vals)); prof_b.append(vals[i]); prof_Om.append(Omgrid[i])
prof_b = np.array(prof_b); ib = int(np.argmin(prof_b))
print(f"   beta profile (Om in [0.2, 0.4], offset free): min chi2 {prof_b[ib]:.2f} at beta {bgrid[ib]:+.2f} (Om {prof_Om[ib]:.4f});"
      f" beta 0: chi2 {prof_b[bgrid==0][0]:.2f} (Om {np.array(prof_Om)[bgrid==0][0]:.4f}); beta +0.16: {prof_b[np.isclose(bgrid,0.16)][0]:.2f}")
c315 = np.array([chi2_shape(0.315, b)[0] for b in bgrid])
print(f"   at Om 0.315: chi2(beta=0) {c315[bgrid==0][0]:.2f}; best beta {bgrid[int(np.argmin(c315))]:+.2f} chi2 {c315.min():.2f}")
OUT["beta_grid"] = bgrid.tolist(); OUT["beta_profile"] = prof_b.tolist(); OUT["beta_profile_Om"] = list(map(float, prof_Om))
OUT["beta_profile_Om315"] = c315.tolist()
H0grid = np.linspace(60, 75, 31)
OUT["H0_grid"] = H0grid.tolist(); OUT["H0_profile"] = [float(prof_b.min())]*len(H0grid)   # exact: chi2 independent of H0
# Hubble diagram residuals (diagonal setup, LCDM Om 0.315, offset fitted)
_, off = chi2_shape(0.315, 0.0)
OUT["hd_z"] = zs.tolist(); OUT["hd_mb"] = mb.tolist(); OUT["hd_err"] = dm.tolist()
OUT["hd_offset_H70"] = float(off)       # m_model = 5 log10(d_L[Mpc] at H0 = 70) + offset
OUT["hd_resid"] = (mb - (5*np.log10(dl(0.315, 70.0, 0.0, zs)) + off)).tolist()
print(f"   LCDM Om 0.315 offset (H0 = 70 units): {off:.4f}; implied M at H0 = 73.04: {off - 25 + 5*np.log10(70/73.04):.3f}")

# ---------------------------------------------------------------- 3. full covariance
raw = np.loadtxt("PantheonPlusSH0ES_STAT+SYS.cov"); N = int(raw[0]); C = raw[1:].reshape(N, N)
m = (D.zHD > 0.01).values; idx = np.where(m)[0]
zz, zh, y = D.zHD.values[m], D.zHEL.values[m], D.m_b_corr.values[m]
Csub = C[np.ix_(m, m)]; Ci = np.linalg.inv(Csub)
def chiF(Omm, b, sel=None):
    if sel is None: ci, Z, ZH, Y = Ci, zz, zh, y
    else:
        ci = np.linalg.inv(Csub[np.ix_(sel, sel)]); Z, ZH, Y = zz[sel], zh[sel], y[sel]
    r = Y - (5*np.log10(dl(Omm, 70.0, b, Z, ZH)) + 25); o = np.ones_like(r)
    d = r - (o@ci@r)/(o@ci@o); return d@ci@d
print(f"3. Full covariance, zHD > 0.01: {m.sum()} SNe")
c0, c1 = chiF(0.315, 0), chiF(0.315, bm)
print(f"   Om 0.315: LCDM chi2 {c0:.2f}; beta_m = {bm:.5f} in the distances {c1:.2f}; Delta chi2 {c1-c0:+.2f}")
bs = np.round(np.linspace(-0.3, 0.3, 121), 4); P = np.array([chiF(0.315, b) for b in bs]); i = P.argmin(); ok = bs[P <= P[i]+1]
print(f"   Om 0.315: best beta {bs[i]:+.3f}, 68 % {ok.min():+.3f} to {ok.max():+.3f}")
OUT["full_beta"] = bs.tolist(); OUT["full_dchi2"] = (P - c0).tolist()
for b in (0.0, bm):
    g = [(o, chiF(o, b)) for o in np.linspace(0.2, 0.45, 101)]; o, v = min(g, key=lambda t: t[1])
    print(f"   beta {b:.5f} fixed, Om free: Om {o:.4f} chi2 {v:.2f} (Delta vs LCDM at 0.315: {v-c0:+.2f})")
print("   Omega_m variation (best beta at fixed Om; full covariance | diagonal setup):")
for o in (0.308, 0.315, 0.322):
    Pf = np.array([chiF(o, b) for b in bs]); Pd = np.array([chi2_shape(o, b)[0] for b in bs])
    print(f"   Om {o}: beta {bs[Pf.argmin()]:+.3f} | {bs[Pd.argmin()]:+.3f}")
def best_beta(fun, lo=-0.9, hi=2.0, n=581):
    bb = np.round(np.linspace(lo, hi, n), 4); P_ = np.array([fun(b) for b in bb]); j = int(np.nanargmin(P_)); ok_ = bb[P_ <= P_[j]+1]
    return bb[j], ok_.min(), ok_.max()
# common offset: the LCDM (Om 0.315) offset of the whole Hubble-flow sample, full covariance (generalised least squares)
r0 = y - (5*np.log10(dl(0.315, 70.0, 0.0, zz, zh)) + 25); o1 = np.ones_like(r0); off_full = (o1@Ci@r0)/(o1@Ci@o1)
def chiF_fixed(Omm, b, sel):
    ci = np.linalg.inv(Csub[np.ix_(sel, sel)])
    d = y[sel] - (5*np.log10(dl(Omm, 70.0, b, zz[sel], zh[sel])) + 25) - off_full; return d@ci@d
print("   redshift bins (Om 0.315): (i) own offset per bin, diagonal | full covariance; (ii) common offset of the whole sample, full covariance")
bins = ((0.01, 0.30), (0.30, 0.70), (0.70, 2.30))
OUT["bins"] = []
for lo, hi in bins:
    seld = (zs > lo) & (zs <= hi); self_ = np.where((zz > lo) & (zz <= hi))[0]
    bd = best_beta(lambda b: chi2_shape(0.315, b, zs[seld], mb[seld], w[seld])[0])
    bf = best_beta(lambda b: chiF(0.315, b, self_))
    bc = best_beta(lambda b: chiF_fixed(0.315, b, self_), -0.6, 0.6, 481)
    print(f"   {lo:.2f}-{hi:.2f}: (i) diag N {seld.sum():4d} beta {bd[0]:+.3f} [{bd[1]:+.3f}, {bd[2]:+.3f}] | "
          f"full N {len(self_):4d} beta {bf[0]:+.3f} [{bf[1]:+.3f}, {bf[2]:+.3f}]  (ii) beta {bc[0]:+.3f} [{bc[1]:+.3f}, {bc[2]:+.3f}]")
    OUT["bins"].append(dict(lo=lo, hi=hi, n_diag=int(seld.sum()), diag=list(map(float, bd)), n_full=len(self_), full=list(map(float, bf)),
                            common=list(map(float, bc))))
rng = np.random.default_rng(20260223)
print("   sample-size convergence (diagonal setup, Om 0.315, 30 random subsamples each):")
OUT["subsample"] = []
for n in (100, 200, 400, 800, 1200, len(zs)):
    vals = []
    for k in range(30 if n < len(zs) else 1):
        sel = rng.choice(len(zs), n, replace=False)
        bb = np.round(np.linspace(-0.6, 0.6, 121), 3)
        vals.append(bb[np.argmin([chi2_shape(0.315, b, zs[sel], mb[sel], w[sel])[0] for b in bb])])
    print(f"   N {n:4d}: beta mean {np.mean(vals):+.3f}, sd {np.std(vals):.3f}")
    OUT["subsample"].append(dict(n=n, mean=float(np.mean(vals)), sd=float(np.std(vals))))
print("   shape of d_L with beta_m in H(z) (Om 0.315):")
zq = np.array([0.05, 0.1, 0.5, 1.0, 2.0])
rat = dl(0.315, 70, bm, zq)/dl(0.315, 70, 0, zq)
for zv, rv in zip(zq, rat):
    rel = rv/rat[0]
    print(f"   z {zv:4}: d_L ratio at fixed H0 {100*(rv-1):+.2f} %; relative to z = 0.05 {100*(rel-1):+.2f} % = {5*np.log10(rel):+.3f} mag")
zc = np.linspace(0.01, 2.3, 300); rc = dl(0.315, 70, bm, zc)/dl(0.315, 70, 0, zc)
OUT["shape_z"] = zc.tolist(); OUT["shape_mag"] = (5*np.log10(rc/(dl(0.315,70,bm,np.array([0.05]))/dl(0.315,70,0,np.array([0.05])))[0])).tolist()

# ---------------------------------------------------------------- 4. chain record
T = pd.read_csv(REPO/"Cosmological_Physics/mgcamb_validation"/"CHAIN_EXTRACTION_FINAL.csv")
print(f"4. Chain record: {len(T)} chains; max final R-1 {T['R-1_final(progress)'].max():.4f}; samples {T.samples.min()}-{T.samples.max()}")
g = lambda n: T[T.chain.str.startswith(n)].iloc[0]
pairs = (("Planck", "iam_fixed_mu0", "lcdm_baseline", "iam_float_mu0"), ("Planck+RSD", "planck_rsd_iam_fixed", "planck_rsd_lcdm", "planck_rsd_mu0"),
         ("Planck+BAO", "planck_bao_iam_fixed", "planck_bao_lcdm", "planck_bao_mu0"),
         ("Planck+Pantheon+", "planck_pantheon_iam_fixed", "planck_pantheon_lcdm", "planck_pantheon_mu0"),
         ("Level 2 Planck", "iam_level2_runA", "iam_level2_runC", None))
for nm, iam, lcdm, fl in pairs:
    A_, L_ = g(iam), g(lcdm)
    extra = "" if fl is None else f"; free mu0 chi2_min {g(fl).chi2_min:.2f}"
    print(f"   {nm:17s} dchi2 {A_.chi2_min-L_.chi2_min:+.2f}  sigma8 {L_.sigma8:.4f} -> {A_.sigma8:.4f} ({100*(A_.sigma8/L_.sigma8-1):+.2f} %, "
          f"{A_.sigma8-L_.sigma8:+.4f})  H0 {L_.H0:.2f} -> {A_.H0:.2f}{extra}")
A2, C2, D2 = g("iam_level2_runA"), g("iam_level2_runC"), g("iam_level2_runD")
print(f"   Level 2 Run A: Om {A2.omegam:.4f} +- {A2.omegam_sd:.4f}, Om/2 {A2.omegam/2:.4f} +- {A2.omegam_sd/2:.4f} "
      f"({(A2.omegam/2-bm)/(A2.omegam_sd/2):+.2f} sigma from {bm:.5f}); S8 {A2.S8:.4f} +- {A2.S8_sd:.4f}")
print(f"   Level 2 Run C (LCDM): S8 {C2.S8:.4f} +- {C2.S8_sd:.4f}; Om/2 {C2.omegam/2:.4f}; S8 shift {A2.S8-C2.S8:+.4f} = {(A2.S8-C2.S8)/C2.S8_sd:+.2f} sigma")
print(f"   S8 check sigma8 sqrt(Om/0.3): Run A {A2.sigma8*np.sqrt(A2.omegam/0.3):.4f}; Planck 2018 published 0.832 +- 0.013: shift {(A2.S8-0.832)/0.013:+.2f} sigma")
print(f"   Level 2 Run D (IAM, Planck+RSD): sigma8 {D2.sigma8:.4f}, H0 {D2.H0:.2f}, dchi2 vs Run A data differ (not paired)")
for n in ("iam_l2b_runA", "iam_l2b_runD"):
    r_ = g(n); print(f"   {n}: H0 {r_.H0:.2f} +- {r_.H0_sd:.2f}, sigma8 {r_.sigma8:.4f}, Om {r_.omegam:.4f}; vs Planck 67.36 +- 0.54: "
                     f"{(r_.H0-67.36)/np.hypot(r_.H0_sd,0.54):+.1f} sigma (both errors)")
print(f"   likelihood ratio exp(-0.54/2) = {np.exp(-0.54/2):.2f}")
for _, r_ in T.iterrows():
    print(f"   {r_.chain:28s} {r_.level:4s} files {r_.files} samples {r_.samples:6d} R-1 {r_['R-1_final(progress)']:.4f} chi2_min {r_.chi2_min:.2f}")
json.dump(OUT, open(HERE/"verify_dual_sector_chapters_data.json", "w"))
print("wrote verify_dual_sector_chapters_data.json")
