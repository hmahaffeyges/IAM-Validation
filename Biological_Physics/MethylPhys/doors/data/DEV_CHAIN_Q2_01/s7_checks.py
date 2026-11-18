import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
#!/usr/bin/env python3
"""SPIDER S7 checks, 2026-11-18 (private; proposals only).
IAM's Law at 310 K in long form: M, the floor (A1), two coins per restore (A2), the constant push (A3),
eps0 (A4), E_hold, phi, the IAM-A line, and where the cell and the horizon share Euler's theorem.
Every step tagged DERIVED HERE in SPIDER_S7_CELL_2026-11-18.md cites a check id below.
Run: python3 spider_s7_cell_checks.py -> spider_s7_cell_checks_output.json
Light: sympy + mpmath + one small numpy profile scan (seconds)."""
import json, math
import sympy as sp
import mpmath as mp
import numpy as np

mp.mp.dps = 30
RES = []
def check(cid, desc, ok, detail=""):
    RES.append({"id": cid, "desc": desc, "pass": bool(ok), "detail": str(detail)[:400]})
def is_zero(e):
    return sp.simplify(sp.expand(e)) == 0

# ---------------------------------------------------------------- constants (CODATA 2018 exact SI)
kB = mp.mpf('1.380649e-23'); NA = mp.mpf('6.02214076e23'); R = kB*NA
Tb = mp.mpf('310.15'); dG = mp.mpf('54000')
M = dG/(R*Tb)
def H2(e):  # binary entropy in bits
    e = mp.mpf(e); return -(e*mp.log(e, 2) + (1-e)*mp.log(1-e, 2))

# S7-01 M, Landauer bits per coin, e_bit
ebit = kB*Tb*mp.log(2)
check("S7-01", "M = dG_ATP/(R T) = 20.94 with R = k_B N_A exact; M/ln2 = 30.21; k_B T ln2 = 2.968e-21 J at 310.15 K",
      abs(R - mp.mpf('8.314462618')) < 1e-9 and abs(M - mp.mpf('20.94')) < 5e-3 and abs(M/mp.log(2) - mp.mpf('30.21')) < 5e-3
      and abs(ebit - mp.mpf('2.968e-21')) < 1e-24, f"M={mp.nstr(M,8)}, M/ln2={mp.nstr(M/mp.log(2),6)}, ebit={mp.nstr(ebit,6)}")

# S7-02 floor: a two-state site with a thermal channel (rates a both ways) and a driven channel
# (local detailed balance: forward/backward = e^A for a cycle affinity A): steady odds <= e^A
a_, b_, A_ = sp.symbols('a b A', positive=True)
odds2 = (a_ + b_*sp.exp(A_))/(a_ + b_)
gap = sp.simplify(sp.exp(A_)*(a_ + b_) - (a_ + b_*sp.exp(A_)))
lim_b = sp.limit(odds2, b_, sp.oo)
eps_min1 = 1/(1 + mp.e**M); eps_min2 = 1/(1 + mp.e**(2*M))
check("S7-02", "two-channel steady odds = (a + b e^A)/(a + b) <= e^A (gap = a(e^A - 1) >= 0), reached as b -> oo; "
      "floor 1/(1+e^M) = 8.1e-10 (1 in 1.24e9), H = 2.5e-8 bits; with the two-coin affinity 2M the floor is 1/(1+e^(2M))",
      is_zero(gap - a_*(sp.exp(A_) - 1)) and sp.simplify(lim_b - sp.exp(A_)) == 0
      and abs(1/eps_min1 - mp.mpf('1.24e9'))/mp.mpf('1.24e9') < 0.01 and abs(H2(eps_min1) - mp.mpf('2.5e-8')) < 1e-9,
      f"eps_min(M)={mp.nstr(eps_min1,4)}, H={mp.nstr(H2(eps_min1),3)} bits; eps_min(2M)={mp.nstr(eps_min2,4)}")

# S7-03 two coins per SAM: chemical potentials a(ATP) d(ADP) m(AMP) p(Pi)
a, d, m, p = sp.symbols('mu_ATP mu_ADP mu_AMP mu_Pi', real=True)
coin = a - d - p                      # -dG of ATP -> ADP + Pi (positive number in cells)
cost = a - m - 2*p                    # -dG of ATP -> AMP + 2Pi
dG_AK = 2*d - a - m                   # dG of ATP + AMP -> 2 ADP
ok3 = is_zero(cost - (2*coin + dG_AK))
band = [2 - (R*Tb*mp.log(2))/dG, 2 + (R*Tb*mp.log(2))/dG]   # |dG_AK| <= RT ln 2
check("S7-03", "cost(ATP->AMP+2Pi) = 2*coin + dG_AK (AK: ATP+AMP->2ADP); = 2 coins at AK equilibrium; within a factor 2 of K_AK: 1.967-2.033",
      ok3 and abs(band[0] - mp.mpf('1.967')) < 5e-4 and abs(band[1] - mp.mpf('2.033')) < 5e-4, f"band={[mp.nstr(q,5) for q in band]}")

# S7-04 driven first passage: D T'' + v T' = -1, T'(0) = 0 (reflecting), T(l) = 0 (absorbing), drift v toward x = l.
x, l, D, v, w = sp.symbols('x l D v w', positive=True)
Tf = sp.Function('T')
sol = sp.dsolve(sp.Eq(D*Tf(x).diff(x, 2) + v*Tf(x).diff(x), -1), Tf(x),
                ics={Tf(x).diff(x).subs(x, 0): 0, Tf(l): 0})
Td = sp.simplify(sol.rhs.subs(x, 0))
Tu = sp.simplify(sp.dsolve(sp.Eq(D*Tf(x).diff(x, 2), -1), Tf(x), ics={Tf(x).diff(x).subs(x, 0): 0, Tf(l): 0}).rhs.subs(x, 0))
Td_w = sp.simplify(Td.subs(v, w*D/l))
odds_book = (w/2)/(1 - (1 - sp.exp(-w))/w)
check("S7-04", "MFPT: T_u = l^2/(2D); T_d = (l^2/D)(w - 1 + e^-w)/w^2; T_u/T_d = (w/2)/(1 - (1 - e^-w)/w) (book eq:middle_passage)",
      is_zero(Tu - l**2/(2*D)) and is_zero(Td_w - (l**2/D)*(w - 1 + sp.exp(-w))/w**2)
      and is_zero(sp.simplify(Tu/Td_w - odds_book)), f"T_d={Td_w}")

# S7-05 series of the odds and of the held share per unit paid s(w) = ln(odds)/w
ser_small = sp.series(odds_book, w, 0, 3).removeO()
s_w = sp.log(odds_book)/w
ser_s = sp.series(s_w, w, 0, 3).removeO()
alg = (w/2)/(1 - 1/w)            # e^-w dropped
ser_large = sp.series(alg, w, sp.oo, 3)
check("S7-05", "odds(w) = 1 + w/3 + w^2/36 + ... (small w); s(w) = 1/3 - w/36 + w^2/810; "
      "large w: odds = w/2 + 1/2 + 1/(2w) + 1/(2w^2) + ... (+O(e^-w))",
      is_zero(ser_small - (1 + w/3 + w**2/36)) and is_zero(ser_s - (sp.Rational(1, 3) - w/36 + w**2/810))
      and is_zero(sp.expand(ser_large.removeO()) - (w/2 + sp.Rational(1, 2) + 1/(2*w) + 1/(2*w**2))),
      f"small={ser_small}; s={ser_s}; large={ser_large}")

# S7-06 EULER IN THE CELL: tau_u(y) = y^2/(2D) (degree 2 in distance), tau_d(y) = y/v (degree 1).
y = sp.symbols('y', positive=True)
tau_u = y**2/(2*D); tau_d = y/v
eu2 = is_zero(y*tau_u.diff(y) - 2*tau_u); eu1 = is_zero(y*tau_d.diff(y) - tau_d)
ratio_euler = sp.simplify((sp.Rational(1, 2)*l*tau_u.diff(y).subs(y, l))/(l*tau_d.diff(y).subs(y, l)))
lim_ratio = sp.limit(odds_book/w, w, sp.oo)
Mg, Sg, cg = sp.symbols('M_g s_g c', positive=True)
S_bh = Sg*Mg**2
TH = cg**2/sp.diff(S_bh, Mg)       # first law d(Mc^2) = T dS
check("S7-06", "Euler degree 2 in the cell's passage: y tau_u' = 2 tau_u, y tau_d' = tau_d; strong-push odds = (1/2) v l/D = w/2; "
      "lim odds/w = 1/2; horizon S ~ M^2 gives T_H S = M c^2/2 by the same identity",
      eu2 and eu1 and is_zero(ratio_euler - v*l/(2*D)) and lim_ratio == sp.Rational(1, 2)
      and is_zero(TH*S_bh - Mg*cg**2/2), f"ratio={ratio_euler}")

# S7-07 the numbers at W = 2 M kT
def odds_num(wv):
    wv = mp.mpf(wv); return (wv/2)/(1 - (1 - mp.e**(-wv))/wv)
W = 2*M
od = odds_num(W); eps0 = 1/(1 + od); Eh = mp.log(od); phi = Eh/M; s_paid = Eh/W
check("S7-07", "W = 2M = 41.88: g/f = 21.45 (= M + 0.51), eps0 = 0.04454, E_hold = 3.066 kT, phi = 0.1464, held per unit paid 0.0732",
      abs(W - mp.mpf('41.88')) < 5e-3 and abs(od - mp.mpf('21.45')) < 5e-3 and abs(eps0 - mp.mpf('0.04454')) < 5e-6
      and abs(Eh - mp.mpf('3.066')) < 5e-4 and abs(phi - mp.mpf('0.1464')) < 5e-5 and abs(s_paid - mp.mpf('0.0732')) < 5e-5,
      f"W={mp.nstr(W,6)}, odds={mp.nstr(od,7)}, odds-(M+1/2)={mp.nstr(od-M-0.5,4)}, eps0={mp.nstr(eps0,6)}, "
      f"E_hold={mp.nstr(Eh,6)}, phi={mp.nstr(phi,6)}, s={mp.nstr(s_paid,5)}")

# S7-08 the simpler odds g/f = M, and the A2 band carried through
eps_M = 1/(1 + M); Eh_M = mp.log(M)
eps_band = [1/(1 + odds_num(c*M)) for c in band]
check("S7-08", "g/f = M gives eps 0.04558, E_hold 3.042 kT; SAM band 1.967-2.033 coins gives eps0 0.04524-0.04386",
      abs(eps_M - mp.mpf('0.04558')) < 5e-6 and abs(Eh_M - mp.mpf('3.042')) < 5e-4
      and abs(eps_band[0] - mp.mpf('0.04524')) < 5e-6 and abs(eps_band[1] - mp.mpf('0.04386')) < 5e-6,
      f"eps(M)={mp.nstr(eps_M,6)}, band eps={[mp.nstr(e_,6) for e_ in eps_band]}")

# S7-09 A3: constant push is the best profile at fixed total work.
pp, aa = sp.symbols('p a', positive=True)
bb = (1 - aa*pp)/(1 - pp)             # keeps a p + b (1-p) = 1
J = pp/aa + (1 - pp)/bb
excess = sp.factor(sp.simplify(J - 1))   # = p (a-1)^2 / (a (1 - a p)) >= 0 when b > 0
def Td_profile(F, wv):
    xg = np.linspace(0, 1, F.size); dx = xg[1] - xg[0]
    ez = np.exp(wv*F)
    inner = np.concatenate([[0.0], np.cumsum(0.5*(ez[1:] + ez[:-1])*dx)])
    integrand = np.exp(-wv*F)*inner
    return float(np.sum(0.5*(integrand[1:] + integrand[:-1])*dx))
wv = float(W); N = 4001
F0 = np.linspace(0, 1, N); T0 = Td_profile(F0, wv)
T0_exact = float((wv - 1 + math.exp(-wv))/wv**2)
rng = np.random.default_rng(20261118); best_other = 0.0; nprof = 0
for kk in range(400):
    k = int(rng.integers(2, 40))
    wts = rng.lognormal(0, rng.uniform(0.05, 1.5), size=k)
    knots = np.concatenate([[0], np.cumsum(wts)])/wts.sum()
    F = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, k + 1), knots)
    best_other = max(best_other, T0/Td_profile(F, wv)); nprof += 1
for eps_ in [1e-3, 1e-2]:
    F = F0 + eps_*np.sin(2*np.pi*F0)/(2*np.pi)
    best_other = max(best_other, T0/Td_profile(F, wv)); nprof += 1
check("S7-09a", "A3 at strong push: ballistic T_d = int dx/v with int v fixed; Cauchy-Schwarz gives the constant push as the optimum "
      "(two-piece case: int 1/F' - 1 = p(a-1)^2/(a(1-ap)) >= 0, zero only at a = 1); numeric integrator reproduces the exact constant-push T_d",
      is_zero(excess - pp*(aa - 1)**2/(aa*(1 - aa*pp))) and abs(T0 - T0_exact)/T0_exact < 1e-5, f"excess={excess}; T0 numeric/exact-1={T0/T0_exact-1:.2e}")
# finite w: first correction T = (1/w) int dx/F' - (1/w^2)[1/(2F'(0)^2) + 1/(2F'(1)^2)] + O(1/w^3)
c9 = 0.05
F9 = F0 - c9*np.sin(2*np.pi*F0)/(2*np.pi)          # F' = 1 - c cos(2 pi x)
T9 = Td_profile(F9, wv)
T9_pred = (1/wv)/math.sqrt(1 - c9**2) - (1/wv**2)*(1/(1 - c9)**2)
# end-flattened profile: push switched off within a = 0.025 of each end, constant between (fine grid)
N2 = 200001; xg2 = np.linspace(0, 1, N2); T02 = Td_profile(xg2, wv)
fp2 = np.where((xg2 < 0.025) | (xg2 > 0.975), 1e-12, 1.0)
F2 = np.concatenate([[0], np.cumsum(0.5*(fp2[1:] + fp2[:-1]))*(xg2[1] - xg2[0])]); F2 = F2/F2[-1]
gain_flat = T02/Td_profile(F2, wv)
eps_flat = 1/(1 + float(od)*gain_flat)
check("S7-09b", "A3 as an exact ceiling FAILS at w = 41.88: the 1/w^2 term rewards a weaker push at the two ends; "
      "F' = 1 - 0.05 cos(2 pi x) gains +0.13 % in odds; switching the push off within 0.025 of each end gains +2.25 % "
      "(eps0 0.04360 instead of 0.04454); the random scan also finds profiles above constant",
      abs(T9 - T9_pred)/T9 < 2e-3 and T0/T9 > 1 and 1.02 < gain_flat < 1.03 and best_other > 1 and abs(eps_flat - 0.04360) < 2e-4,
      f"T9/T9_pred-1={T9/T9_pred-1:.2e}; cos-profile gain={T0/T9:.6f}; flat-end gain={gain_flat:.5f}; eps_flat={eps_flat:.5f}; "
      f"random scan best={best_other:.6f} over {nprof} profiles")

# S7-10 UHRF1 series read through the passage odds
def w_for(epsv):
    target = (1 - mp.mpf(epsv))/mp.mpf(epsv)
    return mp.findroot(lambda q: odds_num(q) - target, 2*target)
wc = w_for('0.03216'); wk = w_for('0.09407')
check("S7-10", "inversion: eps 0.03216 needs w = 59.2, eps 0.09407 needs w = 18.2 (held error x2.925)",
      abs(wc - mp.mpf('59.2')) < 0.05 and abs(wk - mp.mpf('18.2')) < 0.05 and abs(mp.mpf('0.09407')/mp.mpf('0.03216') - mp.mpf('2.925')) < 1e-3,
      f"w_control={mp.nstr(wc,5)}, w_KO={mp.nstr(wk,5)}")

# S7-11 derived vs measured
pull_E = (mp.mpf('3.084') - Eh)/mp.mpf('0.12')
check("S7-11", "derived E_hold 3.066 vs measured 3.084 +/- 0.12 kT (pull 0.15 on the type spread); phi 0.1464 vs 0.1473; "
      "eps0 inside the pooled healthy lone-loss range 0.04375-0.04657; H(eps0) = 0.2627 bits vs H_ref 0.2593",
      abs(pull_E) < 1 and mp.mpf('0.04375') < eps0 < mp.mpf('0.04657') and abs(H2(eps0) - mp.mpf('0.2627')) < 5e-4
      and abs(H2('0.04376') - mp.mpf('0.2593')) < 5e-4,
      f"pull={mp.nstr(pull_E,3)}, H(eps0)={mp.nstr(H2(eps0),5)}, H(0.04376)={mp.nstr(H2('0.04376'),5)}")

# S7-12 the balance and its relaxation
f_, g_, e_, e0 = sp.symbols('f g epsilon e0', positive=True)
eps_ss = sp.solve(sp.Eq((1 - e_)*f_, e_*g_), e_)[0]
step = sp.simplify((e0 + (1 - e0)*f_ - e0*g_) - eps_ss - (1 - f_ - g_)*(e0 - eps_ss))
check("S7-12", "balance eps = f/(f+g), odds = g/f; distance to steady state shrinks by (1 - f - g) per interval",
      is_zero(eps_ss - f_/(f_ + g_)) and is_zero(step))

# S7-13 temperature
def eps_T(Tv, coin=dG):
    Mv = coin/(R*mp.mpf(Tv)); return 1/(1 + odds_num(2*Mv))
r1K = eps_T(Tb + 1)/eps_T(Tb)
dT_needed = mp.findroot(lambda t: eps_T(Tb + t)/eps_T(Tb) - mp.mpf('1.39'), 130)
check("S7-13", "a nucleus 1 K warmer raises eps x1.0030; the healthy type spread x1.39 would need about +134 K",
      abs(r1K - mp.mpf('1.0030')) < 2e-4 and 120 < dT_needed < 150, f"x{mp.nstr(r1K,6)} per K; dT={mp.nstr(dT_needed,4)} K")

# S7-14 coin elasticity (arithmetic beside F026; not a new test)
el = mp.diff(lambda q: mp.log(1/(1 + odds_num(2*q))), M)*M
c_needed = mp.findroot(lambda c: eps_T(Tb, dG/c)/eps_T(Tb) - mp.mpf('1.39'), 1.4)
r_tissue = eps_T(Tb)/eps_T(Tb, mp.mpf('61700'))
check("S7-14", "d ln eps / d ln M = -0.93; through the exact passage a coin lower by x1.431 raises eps x1.39 and the tissue top 61.7 kJ/mol lowers eps x1.133 (the register's x1.418 and x1.143 are the eps ~ 1/M approximation, reproduced here)",
      abs(el + mp.mpf('0.933')) < 0.01 and abs(c_needed - mp.mpf('1.431')) < 2e-3 and abs(r_tissue - mp.mpf('1.133')) < 1e-3 and abs(mp.mpf('1.39')**1 - 1.39) < 1e-12 and abs(mp.mpf('61.7')/54 - mp.mpf('1.1426')) < 1e-3,
      f"elasticity={mp.nstr(el,4)}, coin factor={mp.nstr(c_needed,5)}, tissue={mp.nstr(r_tissue,5)}")

# S7-15 type spread as a hold difference vs half a Landauer bit (provisional ratios)
dE_approx = mp.log(mp.mpf('1.17')/mp.mpf('0.804'))
e1 = eps0*mp.mpf('0.804'); e2 = eps0*mp.mpf('1.17')
dE_exact = mp.log((1 - e1)/e1) - mp.log((1 - e2)/e2)
check("S7-15", "type spread A 0.804-1.17: ln ratio 0.375 kT (0.39 through the odds) vs half a Landauer bit ln2/2 = 0.347 kT",
      abs(dE_approx - mp.mpf('0.375')) < 1e-3 and abs(mp.log(2)/2 - mp.mpf('0.3466')) < 1e-4,
      f"approx={mp.nstr(dE_approx,4)}, odds={mp.nstr(dE_exact,4)}")

# S7-16 A8R: loss rate from the measured restore rate (hESC)
gm = mp.mpf('0.443'); es = mp.mpf('0.04454')/mp.mpf('0.8642')
fl = gm*es/(1 - es)
check("S7-16", "hESC: g = 0.443/h, eps = 0.04454/0.8642 = 0.0515 -> f = 0.0241/h; restore half-life ln2/g = 1.56 h",
      abs(es - mp.mpf('0.0515')) < 1e-4 and abs(fl - mp.mpf('0.0241')) < 2e-4 and abs(mp.log(2)/gm - mp.mpf('1.565')) < 2e-3,
      f"f={mp.nstr(fl,4)}/h, t_half={mp.nstr(mp.log(2)/gm,4)} h")

# S7-17 Euler on the price side: E(S) homogeneous of degree p => S dE/dS = p E
S_, cst, pdeg, Tbath = sp.symbols('S c p T_b', positive=True)
E_p = cst*S_**pdeg
ok17 = is_zero(S_*sp.diff(E_p, S_) - pdeg*E_p)
E_bath = Tbath*S_; E_hor = cst*sp.sqrt(S_)
check("S7-17", "Euler on E(S): S E' = p E; bath p = 1 -> (dE/dS) S = E = T_b S (both bookings equal); horizon p = 1/2 -> (dE/dS) S = E/2",
      ok17 and is_zero(S_*sp.diff(E_bath, S_) - E_bath) and is_zero(S_*sp.diff(E_hor, S_) - E_hor/2)
      and is_zero(sp.diff(E_bath, S_) - E_bath/S_))

# S7-18 overdamped Clausius virial (stationarity): <x U'> = kT; Euler -> <U> = kT/degree
xx, kk_, wv_ = sp.symbols('x k_s w', positive=True)
pl = sp.exp(-wv_*xx); Zl = sp.integrate(pl, (xx, 0, sp.oo))
U_lin_avg = sp.simplify(sp.integrate(wv_*xx*pl, (xx, 0, sp.oo))/Zl)
ph = sp.exp(-kk_*xx**2/2); Zh = sp.integrate(ph, (xx, -sp.oo, sp.oo))
xUp_h = sp.simplify(sp.integrate(xx*kk_*xx*ph, (xx, -sp.oo, sp.oo))/Zh)
U_h_avg = sp.simplify(sp.integrate(kk_*xx**2/2*ph, (xx, -sp.oo, sp.oo))/Zh)
check("S7-18", "stationary overdamped virial <x U'> = kT (kT = 1): linear push on a half-line <U> = kT (degree 1); "
      "harmonic <x U'> = kT, <U> = kT/2 (degree 2)",
      U_lin_avg == 1 and xUp_h == 1 and U_h_avg == sp.Rational(1, 2))

# S7-19 like-for-like with the horizon's held/paid
check("S7-19", "Landauer efficiency of one restore: ln2/(2M) = 0.01655 (per coin ln2/M = 0.0331)",
      abs(mp.log(2)/(2*M) - mp.mpf('0.01655')) < 1e-5, f"{mp.nstr(mp.log(2)/(2*M),5)}")

# S7-20 the paid pump never holds more than 1/3 of what it pays
ws = [mp.mpf(10)**k for k in range(-4, 4)]
svals = [mp.log(odds_num(q))/q for q in ws]
mono = all(svals[i] > svals[i + 1] for i in range(len(svals) - 1))
check("S7-20", "s(w) = ln odds/w falls monotonically from 1/3 (w -> 0) over w = 1e-4..1e3; per coin (two coins) 2/3 at zero drive",
      mono and abs(svals[0] - mp.mpf(1)/3) < 1e-4 and all(sv < mp.mpf(1)/3 for sv in svals),
      f"s(1e-4)={mp.nstr(svals[0],8)}, s(41.88)={mp.nstr(s_paid,5)}")

# S7-21 the book's literal boundary wording: drift toward the reflecting end makes the passage exponential
z = sp.Symbol('z', positive=True)
Tagainst = sp.simplify(sp.integrate(sp.exp(w*y)*sp.integrate(sp.exp(-w*z), (z, 0, y)), (y, 0, 1)))
check("S7-21", "push toward the reflecting end: passage time (e^w - 1 - w)/w^2 (exponential in w), not the linear odds",
      is_zero(Tagainst - (sp.exp(w) - 1 - w)/w**2), f"{Tagainst}")

# S7-22 methionine prediction
def ratio_drop(dW):
    return (1/(1 + odds_num(W - dW)))/eps0
r_ln = [ratio_drop(mp.log(r)) for r in (mp.mpf('3.33'), mp.mpf('5.0'))]
r_2ln = [ratio_drop(2*mp.log(r)) for r in (mp.mpf('3.33'), mp.mpf('5.0'))]
check("S7-22", "SAM/SAH down by r = 3.33-5.00: W lowered by 2 ln r gives x1.057-1.077 (the sealed numbers); ln r gives only x1.026-1.036",
      abs(r_2ln[0] - mp.mpf('1.057')) < 2e-3 and abs(r_2ln[1] - mp.mpf('1.077')) < 2e-3 and r_ln[1] < mp.mpf('1.04'),
      f"ln r: {[mp.nstr(q,5) for q in r_ln]}; 2 ln r: {[mp.nstr(q,5) for q in r_2ln]}")

# S7-23 IAM-A line as a two-compartment mixture
L_, ef, el_, rw = sp.symbols('L e_f e_l r', positive=True)
eps_mix = (ef*(1 - L_) + rw*el_*L_)/((1 - L_) + rw*L_)
d2 = sp.simplify(sp.diff(eps_mix, L_, 2))
check("S7-23", "mixture eps(L) = (e_f(1-L) + r e_l L)/((1-L) + r L): straight in L iff r = 1 (or e_l = e_f)",
      is_zero(d2.subs(rw, 1)) and not is_zero(d2.subs({rw: sp.Rational(1, 2), ef: sp.Rational(1, 25), el_: sp.Rational(1, 10)})),
      f"d2={sp.factor(d2)}")

# S7-24 forward-table offset and England's durability number
check("S7-24", "measured firm 0.04026 over the M-alone counter level 0.0359 = 1.121; England durability ln(1/eps0) = 3.111 kT",
      abs(mp.mpf('0.04026')/mp.mpf('0.0359') - mp.mpf('1.121')) < 1e-3 and abs(mp.log(1/eps0) - mp.mpf('3.1114')) < 5e-4,
      f"{mp.nstr(mp.mpf('0.04026')/mp.mpf('0.0359'),5)}, {mp.nstr(mp.log(1/eps0),6)}")

# ---------------------------------------------------------------- output
npass = sum(r_['pass'] for r_ in RES); nfail = len(RES) - npass
json.dump({"checks": RES, "passed": npass, "failed": nfail}, open("spider_s7_cell_checks_output.json", "w"), indent=1)
for r_ in RES:
    print(("PASS " if r_['pass'] else "FAIL ") + r_['id'] + "  " + (r_['detail'][:160] if r_['detail'] else ""))
print(f"passed {npass}, failed {nfail}")
