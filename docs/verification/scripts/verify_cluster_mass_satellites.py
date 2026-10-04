#!/usr/bin/env python3
"""Recomputes every calculated / derived number in Part 2 chapters p2_17 (lensing mass and dynamical mass), p2_18 (three cluster masses)
and p2_19 (missing satellites: growth suppression from mu < 1). Published measurements and chain posteriors are inputs, not recomputed. numpy, scipy, sympy.
Run from docs/verification/scripts/. Output: verify_cluster_mass_satellites_output.txt."""
import os, numpy as np, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
Om, OL = 0.3153, 0.6847; bm = Om/2; H0 = 67.36            # canon: Planck 2018, beta_m = Omega_m/2 = 0.15765
E = lambda a: np.exp(1-1/a); H2 = lambda a, Om=Om, OL=OL: Om*a**-3+OL
mu = lambda a, b=bm, Om=Om, OL=OL: H2(a, Om, OL)/(H2(a, Om, OL)+b*E(a))
R = lambda z, **k: 1/mu(1/(1+z), **k)
print("A. mu(z) and M_lens/M_dyn = 1/mu (canon Omega_m 0.3153, beta_m %.5f; in brackets Omega_m 0.315, Omega_L 0.685, beta_m 0.1575)" % bm)
for z in (0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
    a = 1/(1+z); p = dict(b=0.1575, Om=0.315, OL=0.685)
    print(f"   z {z:4}: a {a:.3f}  mu {mu(a):.4f} [{mu(a,**p):.4f}]  1/mu {1/mu(a):.4f} [{1/mu(a,**p):.4f}]  excess {100*(1/mu(a)-1):5.2f} %  E(a) {E(a):.4f}")
print(f"   mu(z=0) = 1/(1+beta_m) = {1/(1+bm):.5f}; mu0 = mu(0)-1 = {1/(1+bm)-1:.4f}; 1-mu(0) = {100*(1-1/(1+bm)):.2f} %")
print("B. algebra (sympy)")
a, b, Omr, OLr = sp.symbols('a beta_m Omega_m Omega_L', positive=True); Ea = sp.exp(1-1/a); Hs = Omr*a**-3+OLr
mus = Hs/(Hs+b*Ea); print("   1/mu - (1 + beta_m E/H^2) =", sp.simplify(1/mus-(1+b*Ea/Hs)))
m_, G_, A_, rho, Psi, Phi = sp.symbols('mu G a2 rhodelta Psi Phi')
sol = sp.solve([sp.Eq(Psi, m_*rho), sp.Eq(Phi+Psi, 2*rho)], [Psi, Phi])        # Poisson (x -4 pi G a^2 / k^2 dropped), Sigma = 1
print("   Level 1 form with Sigma = 1: Phi/Psi =", sp.simplify(sol[Phi]/sol[Psi]), "; (Phi+Psi)/2 per unit source =", sp.simplify((sol[Phi]+sol[Psi])/2/rho))
for z in (0, 0.3, 1.0): print(f"   slip Phi/Psi at z {z}: {2/mu(1/(1+z))-1:.4f}")
print("C. slopes")
h = 1e-5; dR = lambda z: (R(z+h)-R(z-h))/(2*h)
CNT = lambda z: 1+0.20*(1+z)**0.2
print(f"   dR/dz: z 0 {(R(h)-R(0))/h:.3f}; z 0.3 {dR(0.3):.3f}; z 0.5 {dR(0.5):.3f}; z 1 {dR(1):.3f}")
print(f"   d(R C_NT)/dz at z 0.3 = {(R(0.3+h)*CNT(0.3+h)-R(0.3-h)*CNT(0.3-h))/(2*h):.3f}")
print("D. three-way table (bin centres), C_NT = 1 + 0.20(1+z)^0.2, compiled ratios as printed (per-bin sources not traced)")
obs = [(1.28, 0.15), (1.22, 0.12), (1.25, 0.10), (1.18, 0.13)]; zc = []; tot = []
for (lo, hi), (o, s) in zip(((0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8)), obs):
    z = (lo+hi)/2; zc.append(z); tot.append(R(z)*CNT(z))
    print(f"   {lo}-{hi}: z {z:.3f}  R {R(z):.3f}  C_NT {CNT(z):.4f}  R*C_NT {R(z)*CNT(z):.3f}  dC_NT/dz {0.04*(1+z)**-0.8:+.3f}  (R*C_NT - obs)/s {(R(z)*CNT(z)-o)/s:+.2f}")
sl = np.polyfit(zc, tot, 1)[0]; print(f"   straight-line slope of R*C_NT over the four centres: {sl:.3f}")
f = lambda e: abs(sl)/(e*np.mean(tot)/np.sqrt(np.sum((np.array(zc)-np.mean(zc))**2)))-3
print(f"   fractional error per bin for a 3 sigma slope: {100*brentq(f, 1e-4, 1):.2f} %")
print(f"   printed forecasts: 0.18/0.03 = {0.18/0.03:.1f}; 0.18/0.008 = {0.18/0.008:.1f}; slope difference to +0.02..+0.04: {(0.18+0.02)/0.03:.1f}-{(0.18+0.04)/0.03:.1f} and {(0.18+0.02)/0.008:.0f}-{(0.18+0.04)/0.008:.0f} (in units of the assumed errors)")
zb = np.array([0.2, 0.5, 0.8, 1.2, 1.8]); Rb = R(zb)
g = lambda e: np.sqrt(np.sum((Rb-np.mean(Rb))**2/(e*Rb)**2))-3
print(f"   lensing/dynamics five bins (0.2,0.5,0.8,1.2,1.8) vs best constant: 3 sigma at {100*brentq(g, 1e-4, 1):.2f} % per bin")
print("E. published cluster ratios")
for nm, x, s in (("WtG 1-b", 0.688, 0.072), ("CCCP 1-b", 0.780, 0.092), ("needed by counts + CMB 1-b", 0.58, 0.04)):
    print(f"   {nm} = {x} +/- {s}: M_WL/M_Planck = 1/(1-b) = {1/x:.3f} +/- {s/x**2:.3f}")
print(f"   CMB lensing 1/(1-b) = 0.99 +/- 0.19 (input)")
print(f"   LoCuSS beta_X 0.95 +/- 0.05 -> M_WL/M_X = {1/0.95:.3f} +/- {0.05/0.95**2:.3f}; IAM Level 1 at z 0.225: mu {mu(1/1.225):.3f}, 1/mu {R(0.225):.3f}; offset {(0.95-mu(1/1.225))/0.05:+.2f} sigma in beta")
for nm, bz, s, z in (("WtG z<0.3", 0.90, 0.09, 0.225), ("WtG z>0.3", 0.71, 0.07, 0.4), ("CCCP z<0.3", 0.96, 0.09, 0.225), ("CCCP z>0.3", 0.61, 0.09, 0.4)):
    print(f"   beta_P {nm} = {bz} +/- {s}; IAM mu at z {z} = {mu(1/(1+z)):.3f}; offset {(bz-mu(1/(1+z)))/s:+.2f} sigma")
import csv; s8 = {r["chain"]: float(r["sigma8"]) for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv")))}  # unrounded chain means
print(f"   sigma8: Level 1 {s8['lcdm_baseline']:.4f} -> {s8['iam_fixed_mu0 (r2 final)']:.4f} {100*(s8['iam_fixed_mu0 (r2 final)']/s8['lcdm_baseline']-1):.2f} %; Level 2 {s8['iam_level2_runC_lcdm']:.4f} -> {s8['iam_level2_runA']:.4f} {100*(s8['iam_level2_runA']/s8['iam_level2_runC_lcdm']-1):.2f} %")
print("F. growth (same early amplitude), three forms, and Press-Schechter")
Oma = lambda a: Om*a**-3/H2(a); Hm2 = lambda a: H2(a)+bm*E(a); dlnH = lambda a: -1.5*Om*a**-3/H2(a)
dlnHm = lambda a: (-3*Om*a**-3+bm*E(a)/a)/(2*Hm2(a))
forms = {"LCDM": lambda l, y: [y[1], -(2+dlnH(np.exp(l)))*y[1]+1.5*Oma(np.exp(l))*y[0]],
         "i G_eff": lambda l, y: [y[1], -(2+dlnH(np.exp(l)))*y[1]+1.5*Oma(np.exp(l))*mu(np.exp(l))*y[0]],
         "ii friction": lambda l, y: [y[1], -(dlnH(np.exp(l))+2*np.sqrt(Hm2(np.exp(l))/H2(np.exp(l))))*y[1]+1.5*Oma(np.exp(l))*y[0]],
         "iii all on H_m": lambda l, y: [y[1], -(2+dlnHm(np.exp(l)))*y[1]+1.5*Om*np.exp(-3*l)/Hm2(np.exp(l))*y[0]]}
S = {k: solve_ivp(v, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14) for k, v in forms.items()}
D = lambda k, a: S[k].sol(np.log(a))[0]; fr = lambda k, a: S[k].sol(np.log(a))[1]/S[k].sol(np.log(a))[0]
for k in ("i G_eff", "ii friction", "iii all on H_m"): print(f"   Delta D/D today, form {k}: {100*(D(k,1)/D('LCDM',1)-1):.2f} %")
for z in (0, 0.3, 0.5, 1.0):
    a_ = 1/(1+z); print(f"   f sigma8 deficit (form i) z {z}: {100*(1-fr('i G_eff',a_)*D('i G_eff',a_)/(fr('LCDM',a_)*D('LCDM',a_))):.2f} %")
eps = 1-D('i G_eff', 1)/D('LCDM', 1)
print(f"   PS: Delta ln n = (nu^2-1) eps, eps = {100*eps:.2f} %: nu 0.5 {100*(0.25-1)*-eps:+.2f} %... nu 1 0, nu 2 {100*3*-eps:+.2f} %; ln 10 = {np.log(10):.3f}")
print("G. missing satellites: the coupling and the chain values")
print(f"   beta_m on the Level 2 posterior Omega_m 0.3166 +/- 0.0066: {0.3166/2:.4f} +/- {0.0066/2:.4f}")
print(f"   H0 matter sector 67.161 sqrt(1.15765) = {67.161*np.sqrt(1.15765):.2f}; (67.36-67.16)/0.54 = {(67.36-67.16)/0.54:.2f} sigma; (73.04-72.26)/1.04 = {(73.04-72.26)/1.04:.2f} sigma")
print(f"   sigma8 0.7998 +/- 0.0058 vs 0.802 (+0.022/-0.018): {(0.802-0.7998)/np.hypot(0.018, 0.0058):.2f} sigma")
print(f"   E(a) at z 2 = {E(1/3):.3f}; mu(z=2) = {mu(1/3):.4f}; linear power ratio today (form i) (D ratio)^2 - 1 = {100*((D('i G_eff',1)/D('LCDM',1))**2-1):.2f} %")
print("H. added: turnover of R x C_NT, hydrostatic-bias constant, Herbonnet, absolute-error test, Press-Schechter algebra, sigma_M at satellite masses")
P = lambda z: R(z)*CNT(z); dP = lambda z: (P(z+h)-P(z-h))/(2*h)
print(f"   d(R C_NT)/dz at z 0.5 {dP(0.5):+.3f}, 1.0 {dP(1.0):+.3f}; turnover (dP/dz = 0) at z = {brentq(dP, 0.5, 5):.2f}")
print(f"   b = 0.17: 1/(1-b) = {1/0.83:.3f}; Herbonnet 1-b = 0.84 +/- 0.04 +/- 0.05: 1/(1-b) = {1/0.84:.3f} +/- {np.hypot(0.04,0.05)/0.84**2:.3f}")
print(f"   1-b_hydro = (1-b_NT)(1-b_IAM) = 1 - b_NT - b_IAM + b_NT b_IAM; at z 0.3 with C_NT: b_NT = {1-1/CNT(0.3):.3f}, b_IAM = {1-mu(1/1.3):.3f}, cross term {(1-1/CNT(0.3))*(1-mu(1/1.3)):.4f}")
s3 = np.sqrt(np.sum((Rb-np.mean(Rb))**2))/3; print(f"   five bins, equal absolute error: 3 sigma at {100*s3:.2f} % per bin (as fig_lensdyn_test)")
nu_, e_, dc = sp.symbols('nu epsilon delta_c', positive=True); sig = sp.symbols('sigma_M', positive=True)
lnn = sp.log(dc/sig) - (dc/sig)**2/2                       # ln n at fixed M up to M-only terms (PS multiplicity nu exp(-nu^2/2))
d1 = sp.simplify(sp.diff(lnn.subs(sig, sig*(1+e_)), e_).subs(e_, 0).subs(dc, nu_*sig)); print("   PS: d ln n / d epsilon (sigma_M -> sigma_M(1+eps)) =", d1)
# sigma_M today, Eisenstein-Hu no-wiggle transfer, Planck 2018 (h 0.6736, Om 0.3153, Ob 0.0493, ns 0.9649), LCDM sigma8 0.811
hh, Ob, ns = 0.6736, 0.0493, 0.9649; Omh2 = Om*hh**2; fb = Ob/Om; th = 2.7255/2.7
s_ = 44.5*np.log(9.83/Omh2)/np.sqrt(1+10*(Ob*hh**2)**0.75); aG = 1-0.328*np.log(431*Omh2)*fb+0.38*np.log(22.3*Omh2)*fb**2
def T(k):
    Ge = Om*hh*(aG+(1-aG)/(1+(0.43*k*hh*s_)**4)); q = k*th**2/Ge
    L0 = np.log(2*np.e+1.8*q); C0 = 14.2+731/(1+62.5*q); return L0/(L0+C0*q**2)
k = np.logspace(-4, 4, 20000)
W = lambda x: 3*(np.sin(x)-x*np.cos(x))/x**3
trap = getattr(np, "trapezoid", None) or np.trapz
sR = lambda Rr, A: np.sqrt(trap(A*k**(3+ns)*T(k)**2*W(k*Rr)**2/(2*np.pi**2), np.log(k)))     # k in h/Mpc
A = (0.811/sR(8.0, 1.0))**2
rhob = 2.775e11*Om                                          # h^2 Msun / Mpc^3
for Mm in (1e7, 1e8, 1e9, 1e12):
    Rr = (3*Mm*hh/(4*np.pi*rhob))**(1/3); sm = sR(Rr, A); nu = 1.686/sm
    print(f"   M {Mm:.0e} Msun: R {Rr:.3f} Mpc/h, sigma_M {sm:.2f}, nu {nu:.3f}, Delta ln n = (nu^2-1) eps = {100*(nu**2-1)*(-eps):+.2f} %")
