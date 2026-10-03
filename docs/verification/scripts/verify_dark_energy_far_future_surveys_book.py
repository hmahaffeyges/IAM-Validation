#!/usr/bin/env python3
"""Book carriage check for Part 2 chapters ch:darkenergy (p2_11), ch:wzfuture (p2_20) and ch:surveys (p2_16).
Recomputes every equation step (sympy) and every number printed in those chapters. numpy, scipy, sympy.
Inputs: Planck 2018 base (H0 67.4, Om 0.315, OL 0.685) where the chapter says so; Level 2 photon sector H0 67.16 with
Om 0.3153, OL 0.6847 elsewhere; beta_m = Om/2 = 0.15765; E(a) = exp(1 - 1/a); mu = H^2/(H^2 + beta_m E H0^2); Sigma = 1.
Run: python docs/verification/scripts/verify_dark_energy_far_future_surveys_book.py > ..._output.txt"""
import numpy as np, sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar

pr = print
# ---------------------------------------------------------------- A. algebra (sympy)
a, t, H0s, rho0, Om_s, OL_s, bm_s, f_s = sp.symbols('a t H_0 rho_0 Omega_m Omega_Lambda beta_m f', positive=True)
Es = sp.exp(1 - 1/a)
pr("A. Algebra")
dlnE = sp.simplify(sp.diff(sp.log(Es), a) * a)
pr("  A1 dlnE/dlna =", dlnE)
w = sp.simplify(-1 - sp.Rational(1, 3) * dlnE)
pr("  A2 continuity rho' = -3(1+w) rho/a with rho = rho0 E(a)  ->  w_info =", w)
chk = sp.simplify(sp.diff(rho0*Es, a) + 3*(1 + w)*rho0*Es/a)
pr("     residual of continuity equation:", chk)
pr("  A3 w_info(1) =", w.subs(a, 1), "; lim a->oo w =", sp.limit(w, a, sp.oo), "; dw/da =", sp.simplify(sp.diff(w, a)))
pr("  A4 CPL: w0 =", w.subs(a, 1), ", wa = -dw/da|1 =", -sp.diff(w, a).subs(a, 1))
pr("  A5 E(0+) =", sp.limit(Es, a, 0, '+'), "; E(1) =", Es.subs(a, 1), "; E(oo) =", sp.limit(Es, a, sp.oo))
d2a = sp.simplify(sp.diff(Es, a, 2)); pr("  A6 d2E/da2 =", sp.factor(d2a), " zero at a =", sp.solve(sp.Eq(sp.factor(d2a*a**4/Es), 0), a))
x = sp.symbols('x', real=True); Ex = sp.exp(1 - sp.exp(-x))   # x = ln a
d2x = sp.simplify(sp.diff(Ex, x, 2)); pr("  A7 d2E/d(ln a)^2 =", sp.factor(d2x), " zero at ln a =", sp.solve(sp.Eq(sp.simplify(d2x/Ex*sp.exp(2*x)), 0), x))
rate_a = sp.simplify(sp.diff(Es/sp.E, a)); pr("  A8 d(E/e)/da =", rate_a, "; at a=1:", sp.nsimplify(rate_a.subs(a, 1)), "=", float(rate_a.subs(a, 1)),
   "; max at a =", sp.solve(sp.diff(rate_a, a), a), "value", sp.nsimplify(rate_a.subs(a, sp.Rational(1, 2))), "=", round(float(rate_a.subs(a, sp.Rational(1, 2))), 4))
rate_ln = sp.simplify(a*rate_a); pr("     d(E/e)/dln a =", rate_ln, "; max at a =", sp.solve(sp.diff(rate_ln, a), a), "value", rate_ln.subs(a, 1))
ff = sp.symbols('f', positive=True); pr("  A9 a(f) from exp(1-1/a) = f e:", sp.solve(sp.Eq(sp.exp(1 - 1/a), ff*sp.E), a))
pr("  A10 d(E/e)/dt|_{a=1} = d(E/e)/da * aH |_{a=1} = H0 * ", sp.simplify(rate_a.subs(a, 1) * 1))
pr("  A11 matter-sector rate today H_m^2 = H0^2 (1 + beta_m)  [Omega_m + Omega_L = 1, E(1) = 1]:", sp.simplify((Om_s + (1 - Om_s) + bm_s*Es).subs(a, 1)))
pr("  A12 mu(z=0) - 1 = -beta_m/(1+beta_m):", sp.simplify(1/(1 + bm_s) - 1))
# ISW: Pogosian-Silvestri convention k^2 (Phi+Psi) = -8 pi G a^2 Sigma rho_bar Delta; rho_bar a^2 ~ a^-1, Delta ~ D
Dfun = sp.Function('D')(a); Hf = sp.Function('H')(a)
src = sp.diff(Dfun/a, a) * a * (a*Hf)    # d/deta = a H d/dln a = a^2 H d/da
pr("  A13 d(D/a)/deta =", sp.simplify(src), " = H D (f - 1) with f = dlnD/dlna")
pr("  A14 E_G = Omega_m0 Sigma / f -> dE_G/E_G = -df/f at fixed Sigma = 1")

# ---------------------------------------------------------------- B. background, rates, maturity
pr("\nB. Background and rates")
GYR = 977.792
def bg(H0, Om, OL):
    Hh = lambda a: np.sqrt(Om*a**-3 + OL); tH = GYR/H0
    age = lambda a: quad(lambda y: 1/(y*Hh(y)), 1e-9, a, limit=400)[0]*tH
    return Hh, tH, age
for lab, H0, Om, OL in (("Planck 2018 base", 67.4, 0.315, 0.685), ("Level 2 photon sector", 67.16, 0.3153, 0.6847)):
    Hh, tH, age = bg(H0, Om, OL); bm = 0.3153/2
    pr(f"  B1 {lab}: H0 {H0}, H_inf = H0 sqrt(OL) = {H0*np.sqrt(OL):.2f}; sqrt(OL) = {np.sqrt(OL):.4f}; age today {age(1):.3f} Gyr")
    for aa in (1.0, 2.0, 10.0, 1e6):
        Hm = np.sqrt(Hh(aa)**2 + bm*np.exp(1 - 1/aa)); pr(f"     a={aa:g}: H {H0*Hh(aa):6.2f}  H_m {H0*Hm:6.2f}  ratio {Hm/Hh(aa):.4f}")
    pr(f"     asymptote H_m = H0 sqrt(OL + beta_m e) = {H0*np.sqrt(OL + bm*np.e):.2f}; ratio sqrt(1 + beta_m e/OL) = {np.sqrt(1 + bm*np.e/OL):.4f}")
    pr(f"     rate today (1/e) H0 = {100*H0/GYR/np.e:.3f} % per Gyr")
pr("  B2 maturity table, Planck 2018 base (as printed) and Level 2 background (age differences)")
HhP, tHP, ageP = bg(67.4, 0.315, 0.685); HhL, tHL, ageL = bg(67.16, 0.3153, 0.6847)
for fr in (0.01, 0.05, 0.10, 0.20, 1/np.e, 0.5, 0.75, 0.90, 0.95, 0.99):
    aa = -1/np.log(fr)
    pr(f"     {100*fr:5.1f}%  a {aa:7.3f}  z {1/aa-1:6.2f}  age {ageP(aa):5.1f}  from now {ageP(aa)-ageP(1):+5.1f}   | L2 age {ageL(aa):5.1f} from now {ageL(aa)-ageL(1):+5.1f}")
pr(f"     intervals: 36.8->50 {ageP(1/np.log(2))-ageP(1):.1f}; 50->90 {ageP(-1/np.log(0.9))-ageP(1/np.log(2)):.1f}; 90->99 {ageP(-1/np.log(0.99))-ageP(-1/np.log(0.9)):.1f} Gyr")
E = lambda a: np.exp(1 - 1/a); HP = HhP
r_t = minimize_scalar(lambda y: -E(y)/y*HP(y), bounds=(0.05, 5), method="bounded").x
pr(f"  B3 dE/dt peak a {r_t:.3f} z {1/r_t-1:.2f}, {ageP(1)-ageP(r_t):.1f} Gyr ago; rate {100*E(r_t)/r_t*HP(r_t)/np.e/tHP:.2f} %/Gyr; today {100*HP(1)/np.e/tHP:.2f}")
pr(f"     paper's 1/(e a^2) at a=0.5: {1/(np.e*0.25):.3f}; correct E/(e a^2): {E(0.5)/(np.e*0.25):.3f}; at a=1 both {1/np.e:.4f}")
pr("  B4 informational density weight (beta_m = 0.3153/2, OL 0.685)")
bm = 0.3153/2
for zz in (0, 0.5, 1, 2, 3):
    aa = 1/(1+zz); rr = bm*E(aa)/0.685; pr(f"     z={zz}: w {-1-(1+zz)/3:.3f}  rho_info/rho_L {rr:.4f}  share {rr/(1+rr):.4f}")
rr = bm*np.e/0.685; pr(f"     saturation: {rr:.3f} share {rr/(1+rr):.3f}")
pr("  B5 no finite-time singularity: H_m^2/H0^2 <= Om a^-3 + OL + beta_m e; for a >= 1 the bound is", round(0.315 + 0.685 + bm*np.e, 4),
   "-> a(t) grows at most as exp(H0 sqrt(OL+beta_m e) t) asymptotically; no a -> oo at finite t")
pr("  B6 DESI DR2 (arXiv:2503.14738) vs CPL image (-4/3, -1/3)")
for nm, w0, s0, wa, sa in (("Pantheon+", -0.838, 0.055, -0.62, 0.21), ("Union3", -0.667, 0.088, -1.09, 0.29), ("DES Y5", -0.752, 0.057, -0.86, 0.22)):
    ac = 1 - (-1 - w0)/wa
    pr(f"     {nm}: w0 off {(-4/3-w0)/s0:+.1f} sigma, wa off {(-1/3-wa)/sa:+.1f} sigma; crossing a {ac:.3f} z {1/ac-1:.2f}")
pr("  B7 CPL error of the image: w_info(a) - CPL(a)")
for aa in (0.25, 0.5, 1, 2, 5):
    pr(f"     a={aa}: w_info {-1-1/(3*aa):.4f}  CPL {-4/3 - (1/3)*(1-aa):.4f}  diff {(-1-1/(3*aa)) - (-4/3 - (1/3)*(1-aa)):+.4f}")

# ---------------------------------------------------------------- C. survey quantities
pr("\nC. Survey predictions (Om 0.3153, OL 0.6847, beta_m 0.15765)")
Om, OL = 0.3153, 0.6847; bm = Om/2
H2 = lambda a: Om*a**-3 + OL; mu = lambda a: H2(a)/(H2(a) + bm*E(a)); Oma = lambda a: Om*a**-3/H2(a)
MU0M = -0.13495; mum = lambda a: 1 + MU0M*(OL/H2(a))/OL
pr(f"  C1 beta_m {bm:.5f}; mu0 = mu(0)-1 = {mu(1.0)-1:.5f}; with beta_m 0.1575 {1/(1.1575)-1:.5f}; MGCAMB runs {MU0M}; 1/mu(0) = {1/mu(1.0):.4f}")
def grow(m):
    def r(l, y):
        aa = np.exp(l); return [y[1], -(2 - 1.5*Om*aa**-3/H2(aa))*y[1] + 1.5*Oma(aa)*m(aa)*y[0]]
    return solve_ivp(r, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
L, I, M = grow(lambda aa: 1.0), grow(mu), grow(mum)
D = lambda S, aa: S.sol(np.log(aa))[0]; fG = lambda S, aa: S.sol(np.log(aa))[1]/S.sol(np.log(aa))[0]
pr("  C2 table: z, mu, mu-1 %, dD/D %, dfs8 %, dPhi/Phi % (= dD/D), mu_MGCAMB, (mu_exact - mu_MGCAMB) x100")
for zz in (0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
    aa = 1/(1+zz); dd = 100*(D(I, aa)/D(L, aa) - 1); dfs = 100*(fG(I, aa)*D(I, aa)/(fG(L, aa)*D(L, aa)) - 1)
    pr(f"     z {zz:3}: mu {mu(aa):.3f}  {100*(mu(aa)-1):6.2f}  dD/D {dd:6.2f}  dfs8 {dfs:6.2f}  dPhi {dd:6.2f}  mgc {mum(aa):.3f}  diff {100*(mu(aa)-mum(aa)):+.2f}")
zz = np.linspace(0, 3, 30001); dif = np.array([mu(1/(1+q)) - mum(1/(1+q)) for q in zz])
rel = dif/np.array([mum(1/(1+q)) for q in zz])
pr(f"     max |exact - MGCAMB| = {100*abs(dif).max():.2f} x 1e-2 in mu at z = {zz[np.argmax(abs(dif))]:.2f}; relative {100*abs(rel).max():.2f} % at z = {zz[np.argmax(abs(rel))]:.2f}")
pr("  C3 activation milestones (fraction of 1 - mu(0))")
d0 = 1 - mu(1.0); HL = lambda aa: np.sqrt(H2(aa)); tH = GYR/67.36
age = lambda aa: quad(lambda y: 1/(y*HL(y)), 1e-9, aa, limit=400)[0]*tH; t0 = age(1)
for q in (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
    aa = brentq(lambda y: (1 - mu(y))/d0 - q, 0.05, 1); pr(f"     {100*q:4.0f}%: a {aa:.3f} z {1/aa-1:.2f} lookback {t0-age(aa):.1f} Gyr")
pr(f"     (lookback uses H0 = 67.36, Om 0.3153; age today {t0:.2f} Gyr)")
dm = np.gradient(np.array([mu(1/(1+q)) for q in zz]), zz)
pr(f"  C4 |dmu/dz| max at z = {zz[np.argmax(abs(dm))]:.3f}; value at 0: {abs(dm[0]):.3f}; at 0.5: {abs(dm[5000]):.3f}; mu rises monotonically with z on 0-3: {bool(np.all(np.diff([mu(1/(1+q)) for q in zz]) > 0))}")
dmm = np.gradient(np.array([mum(1/(1+q)) for q in zz]), zz); pr(f"     MGCAMB |dmu/dz| at 0: {abs(dmm[0]):.3f}, max at z = {zz[np.argmax(abs(dmm))]:.3f}")
for q in (0.1, 0.5, 0.9):
    aa = 1/(1 - np.log(q)); pr(f"     E(a) = {q}: a {aa:.3f} z = {1/aa-1:.2f}")
pr("  C5 ISW: source S = H D (1 - f) (Sigma = 1); ratio IAM/LCDM")
Sx = lambda S, aa: D(S, aa)*(1 - fG(S, aa))
for zq in (0.0, 0.3, 0.5, 0.7, 1.0, 1.5):
    aa = 1/(1+zq); pr(f"     z {zq}: S_IAM/S_LCDM {Sx(I, aa)/Sx(L, aa):.4f}  (MGCAMB {Sx(M, aa)/Sx(L, aa):.4f})")
def amp(S, zc, sig=0.1):
    g = lambda q: np.exp(-0.5*((q - zc)/sig)**2)*Sx(S, 1/(1+q))*np.sqrt(H2(1/(1+q)))
    return quad(g, max(0.0, zc - 5*sig), zc + 5*sig)[0]
for nm, zc in (("DESI BGS", 0.3), ("DESI LRG", 0.5), ("DESI LRG", 0.7)):
    pr(f"     {nm} z~{zc}: A_IAM/A_LCDM = {amp(I, zc)/amp(L, zc):.3f}  (MGCAMB {amp(M, zc)/amp(L, zc):.3f}); paper printed 1.092/1.115/1.165")
U = lambda S: quad(lambda q: Sx(S, 1/(1+q))*np.sqrt(H2(1/(1+q))), 0.05, 1.5)[0]
U0 = lambda S: quad(lambda q: Sx(S, 1/(1+q)), 0.05, 1.5)[0]
pr(f"     uniform weight 0.05<z<1.5: with H factor {U(I)/U(L):.3f}; source (1-f)D only {U0(I)/U0(L):.3f}")
pot = lambda S, zq: (D(S, 1/(1+zq))*(1+zq))/(D(S, 0.25)*4)
pr(f"     potential (Phi+Psi)(z)/(Phi+Psi)(z=3): z 0.5 LCDM {pot(L,0.5):.4f} IAM {pot(I,0.5):.4f} diff {100*(pot(I,0.5)/pot(L,0.5)-1):+.2f} %; z 0 diff {100*(pot(I,0)/pot(L,0)-1):+.2f} %")
pr("  C6 E_G = Om/f: change", "; ".join(f"z {zq}: {100*(fG(L,1/(1+zq))/fG(I,1/(1+zq))-1):+.2f} %" for zq in (0, 0.3, 0.5, 1.0)))
pr("  C7 sirens")
Hm = 67.16*np.sqrt(1 + bm); pr(f"     H0_m = 67.16 sqrt(1.15765) = {Hm:.2f}; separation {Hm-67.16:.2f}; 3 sigma needs sigma <= {(Hm-67.16)/3:.2f} = {100*(Hm-67.16)/3/Hm:.1f} % of {Hm:.2f}")
pr(f"     a 2 % siren measurement (sigma {0.02*Hm:.2f}) separates them at {(Hm-67.16)/(0.02*Hm):.1f} sigma; 1 %: {(Hm-67.16)/(0.01*Hm):.1f} sigma")
pr(f"     SH0ES 73.04 +- 1.04: {(73.04-Hm)/1.04:.2f} sigma from {Hm:.2f}; GW170817 70.0 (+12/-8) offsets {(Hm-70)/12:.2f}, {(70-67.16)/8:.2f} sigma")
pr("  C8 scorecard checks")
pr(f"     Planck lensing detected at 40 sigma -> amplitude error ~ {100/40:.1f} %; IAM C_phiphi change -0.08 % is {0.08/2.5:.3f} of it")
pr(f"     KiDS-1000 3x2pt sigma8 0.76 (+0.025 -0.020; Heymans 2021) vs IAM 0.800: {(0.800-0.76)/0.025:.1f} sigma (upper error)")
pr(f"     Planck lensing error / IAM change: {2.5/0.08:.0f}")
pr(f"     M_lens/M_dyn (Level 1 form) 1/mu(0) = {1/mu(1.0):.3f}")
pr("  C9 effective equation of state of the vacuum-like total (Lambda + informational term), OL 0.685, beta_m 0.3153/2")
OLp = 0.685; bmp = 0.3153/2
weff = lambda aa: -1 - (bmp*E(aa)/(OLp + bmp*E(aa)))/(3*aa)
for zq in (3, 2, 1, 0.5, 0.25, 0, -0.5, -0.9):
    aa = 1/(1+zq); pr(f"     z {zq:5}: a {aa:6.3f}  w_eff {weff(aa):.4f}")
from scipy.optimize import minimize_scalar as _ms
r = _ms(weff, bounds=(0.1, 20), method="bounded"); pr(f"     minimum w_eff {r.fun:.4f} at a {r.x:.3f} (z {1/r.x-1:.2f})")
