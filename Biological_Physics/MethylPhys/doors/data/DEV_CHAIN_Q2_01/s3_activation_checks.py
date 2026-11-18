import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
#!/usr/bin/env python3
"""SPIDER S3 checks: the structure-formation record rate, the activation E(a)=exp(1-1/a),
phi=1-1/a, rho_info and w_info=-1-1/(3a). Private derivation web; proposals only.
sympy + numpy/scipy, runs in seconds on one core. Output: spider_s3_activation_checks_output.json
Check ids G1..G30 are cited in the report SPIDER_S3_ACTIVATION_2026-11-18.md."""
import json, math
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import erfc
from scipy.optimize import brentq

RESULTS = []
def check(cid, desc, ok, value=None):
    RESULTS.append({"id": cid, "desc": desc, "pass": bool(ok), "value": value})
    print(("PASS " if ok else "FAIL ") + cid + ": " + desc + ("" if value is None else f"  -> {value}"))

t, a0 = sp.symbols('t a0', positive=True)
a = sp.Function('a')(t)
A, n, c, Om, G, H0, beta, s0 = sp.symbols('A n c Omega_m G H_0 beta s_0', positive=True)
H = sp.diff(a, t)/a

# ---------- G1: phi = 1 - 1/a obeys phi_dot = H/a for ANY a(t); phi = -z ----------
phi = 1 - 1/a
check("G1", "phi=1-1/a gives dphi/dt = H/a for arbitrary a(t); phi = -z",
      sp.simplify(sp.diff(phi, t) - H/a) == 0 and sp.simplify((1 - (1 + sp.Symbol('z'))) + sp.Symbol('z')) == 0)

# ---------- G2: constraint rho_dot = rho H/a integrates to exp(1-1/a) from a=1; from a->0 diverges ----------
x, eps = sp.symbols('x epsilon', positive=True)
lnE = sp.integrate(1/x**2, (x, 1, A))
div = sp.limit(sp.integrate(1/x**2, (x, eps, 1)), eps, 0)
check("G2", "d ln rho/da = 1/a^2 integrated from a=1 gives 1-1/a; from a->0 it diverges",
      sp.simplify(lnE - (1 - 1/A)) == 0 and div == sp.oo)

# ---------- G3: continuity -> w_info ----------
rho = sp.exp(1 - 1/A)
w = sp.Symbol('w')
dlnrho_dlna = sp.simplify(A*sp.diff(sp.log(rho), A))
wsol = sp.solve(sp.Eq(dlnrho_dlna, -3*(1 + w)), w)[0]
check("G3", "continuity with d ln rho/d ln a = 1/a gives w = -1 - 1/(3a); w(1) = -4/3",
      sp.simplify(wsol - (-1 - 1/(3*A))) == 0 and wsol.subs(A, 1) == sp.Rational(-4, 3))

# ---------- G4: minisuperspace action: record pressure from the a-equation, lambda cancels ----------
N = sp.Symbol('N', positive=True)
tt = sp.Symbol('t')
af = sp.Function('a')(tt); ph = sp.Function('phi')(tt); lam = sp.Function('lam')(tt)
rho0 = sp.Symbol('rho0', positive=True)
L = -N*af**3*rho0*sp.exp(ph) + lam*af**3*(sp.diff(ph, tt) - sp.diff(af, tt)/af**2)
EL_a = sp.diff(L, af) - sp.diff(sp.diff(L, sp.diff(af, tt)), tt)
EL_phi = sp.diff(L, ph) - sp.diff(sp.diff(L, sp.diff(ph, tt)), tt)
# use constraint phi' = a'/a^2 and solve phi-equation for lam'
lamdot = sp.solve(EL_phi, sp.diff(lam, tt))[0]
EL_a2 = EL_a.subs(sp.diff(lam, tt), lamdot).subs(sp.diff(ph, tt), sp.diff(af, tt)/af**2).subs(N, 1)
P_info = sp.simplify(EL_a2/(3*af**2))          # fluid convention dL/da = 3 N a^2 P
rho_info = rho0*sp.exp(ph)
Hs = sp.diff(af, tt)/af
target = -rho_info*(1 + 1/(3*af))
check("G4", "minisuperspace a-equation: P_info = -rho(1+1/(3a)), lambda drops out (book p1_02:417 confirmed)",
      sp.simplify(P_info - target) == 0 and not P_info.has(lam))

# ---------- G5: NEC ----------
check("G5", "effective record fluid: rho+P = -rho/(3a) < 0 for rho>0 (null energy condition violated by T_ab)",
      sp.simplify(rho_info + target + rho_info/(3*af)) == 0)

# ---------- G6: combined w_eff and CPL tangent ----------
E = sp.exp(1 - 1/A)
OL = 1 - Om; bm = Om/2
weff = (-OL + bm*E*(-1 - 1/(3*A)))/(OL + bm*E)
w0 = sp.simplify(weff.subs(A, 1))
dw = sp.simplify(sp.diff(weff, A).subs(A, 1))
check("G6", "w_eff(1) = 2(3-Om)/(3(Om-2)), dw/da(1) = Om^2/(3(2-Om)^2)",
      sp.simplify(w0 - 2*(3 - Om)/(3*(Om - 2))) == 0 and sp.simplify(dw - Om**2/(3*(2 - Om)**2)) == 0,
      {"w0(0.3153)": float(w0.subs(Om, 0.3153)), "wa(0.3153)": float(-dw.subs(Om, 0.3153))})

# ---------- G7: balance-law form (register X7) ----------
lnEc = c*(1 - 1/A)
check("G7", "ln E = c(1-1/a) obeys d lnE/d ln a = c - ln E (relaxation 1 per e-fold; c = target, not derived)",
      sp.simplify(A*sp.diff(lnEc, A) - (c - lnEc)) == 0)

# ---------- G8: the exponent n under each bookkeeping choice (matter era) ----------
# rate density r ~ rho_m D^n f H ; matter era rho~a^-3, D~a, f=1, H~a^-3/2, T_H~H, A_H~H^-2, V_H~H^-3
p = sp.Symbol('p')
pw = {"rho": -3, "D": 1, "H": sp.Rational(-3, 2)}
def slope(vol_total, divT, divA):
    # power of a in dS/dln a = (rate)/(H) * [V_H] * [1/T_H] * [1/A_H]
    s = pw["rho"] + n*pw["D"] + pw["H"] - pw["H"]          # rho_m D^n f H / H
    if vol_total: s += -3*pw["H"]                            # x V_H ~ H^-3
    if divT: s += -pw["H"]                                   # / T_H ~ H
    if divA: s += 2*pw["H"]                                  # / A_H ~ H^-2
    return sp.simplify(s)
table = {}
for vt in (False, True):
    for dT in (False, True):
        for dA in (False, True):
            nreal = sp.Symbol('nreal', real=True)
            nreq = sp.solve(sp.Eq(slope(vt, dT, dA).subs(n, nreal), -1), nreal)[0]
            table[f"{'volume-total' if vt else 'density'}|{'/T_H' if dT else 'no T'}|{'/A_H' if dA else 'no A'}"] = str(nreq)
book_choice = table["density|/T_H|/A_H"]
check("G8", "matter-era dS/dlna ~ a^(n-9/2) for the book's choice; n needed for S ~ -1/a in all 8 bookkeepings",
      book_choice == "7/2" and table["density|/T_H|no A"] == "1/2", table)

# ---------- G9: the book's numerical table uses a rate differing from eq:Idot by H^3 ----------
a_, Hh, rhom, Dn, f_ = sp.symbols('a H rho_m Dn f', positive=True)
R_table = (8*sp.pi*G*rhom/(3*Hh**2))*f_*Dn                 # Omega_m(a) f D^n
per_efold_table = R_table*(Hh/2)/Hh                        # R/(T_H A_H) dt per d ln a, T_H A_H = 2/H (hbar=c=k=1)
per_efold_book = rhom*Dn*f_*Hh*(Hh/2)/Hh                   # rho_m D^n f H /(T_H A_H) per d ln a
ratio = sp.simplify(per_efold_book/per_efold_table)
# matter era powers
m_table = sp.simplify(-3 + n - 2*sp.Rational(-3, 2))       # rho a^-3, D^n a^n, H^-2 -> a^3
m_book = sp.simplify(-3 + n + sp.Rational(-3, 2))
check("G9", "book Table (sec:numerics) integrand / eq:Idot integrand = H^3 x const; matter era per e-fold a^n vs a^(n-9/2)",
      sp.simplify(ratio - 3*Hh**3/(8*sp.pi*G)) == 0 and m_table == n and sp.simplify(m_book - (n - sp.Rational(9, 2))) == 0,
      {"ratio_book_over_table": str(ratio), "table_power": str(m_table), "book_power": str(m_book)})

# ---------- G10: Cai-Kim with S_info: d(Delta H^2)/dt = (G/pi) H^4 Sdot ----------
Sd, rhoS, PS = sp.symbols('Sdot rho P')
Hdot = -4*sp.pi*G*(rhoS + PS) + G/(2*sp.pi)*Hh**3*Sd       # book eq:law_F2info
rhodot = -3*Hh*(rhoS + PS)
dDeltaH2 = sp.simplify(2*Hh*Hdot - sp.Rational(8, 3)*sp.pi*G*rhodot)
check("G10", "from eq:law_F2info + continuity: d(H^2 - 8piG rho/3)/dt = (G/pi) H^4 Sdot_info (linear in Sdot)",
      sp.simplify(dDeltaH2 - G/sp.pi*Hh**4*Sd) == 0)

# ---------- G11: first law fed the n=7/2 history (matter era) -> increments ~ a^-7 (early-dominated) ----------
# dS/dln a ~ a^-1 ; H^4 ~ a^-6  -> d(DeltaH^2)/d ln a ~ a^-7
inc_power = sp.Integer(-6) + sp.Integer(-1)
aa = sp.Symbol('aa', positive=True)
DeltaH2 = sp.integrate(aa**inc_power/aa, (aa, x, 1))         # from a_min = x to 1
check("G11", "first law + S~-1/a (n=7/2): d(DeltaH^2)/dln a ~ a^-7; integral from a_min ~ a_min^-7/7 (early writing dominates, Lambda-like after)",
      inc_power == -7 and sp.limit(DeltaH2*x**7, x, 0) == sp.Rational(1, 7))

# ---------- G12: entropy history the first law needs for rho_info ~ E(a) ----------
hh = sp.Function('h')(A)                                     # H/H0
# Sdot = 8 pi^2 rho_info/(3 a H^3), rho_info = 3H0^2 beta E/(8 pi G); per e-fold divide by H; S_geo0 = pi/(G H0^2)
rho_i = 3*H0**2*beta*E/(8*sp.pi*G)
dSdlna = sp.simplify(8*sp.pi**2*rho_i/(3*A*(H0*hh)**3)/(H0*hh))
Sgeo0 = sp.pi/(G*H0**2)
check("G12", "first law needs dS_info/dln a = beta S_geo0 E(a) (H0/H)^4 / a; matter era ~ E a^5 (book eq:law_Sneed, all a)",
      sp.simplify(dSdlna - beta*Sgeo0*E/(A*hh**4)) == 0 and
      sp.simplify(sp.diff(sp.log(E*A**6/A), A)*A - (1/A + 5)) == 0)

# ---------- G13: E/e is not the entropy fraction when S ~ phi ----------
check("G13", "with S_info ~ phi = 1-1/a (p2_03 eq:th:S72), E/e = exp(-1/a) is not S/S_inf: at a=1 E/e = 0.368, S/S_inf = 0",
      abs(float(sp.exp(-1)) - 0.3679) < 1e-4 and (1 - 1/sp.Integer(1)) == 0)

# ---------- G14-G15: spherical collapse: delta_c, turnaround, Delta_vir; virialization energy ----------
th = sp.Symbol('theta', positive=True)
# R = A(1-cos th), t = B(th - sin th), A^3 = G M B^2 ; linear delta = (3/20)(6(th - sin th))^(2/3)
dlin = sp.Rational(3, 20)*(6*(th - sp.sin(th)))**sp.Rational(2, 3)
dc = dlin.subs(th, 2*sp.pi)
nl_ta = sp.Rational(9, 2)*(th - sp.sin(th))**2/(1 - sp.cos(th))**3
dvir = sp.Rational(9, 2)*(2*sp.pi)**2/(1)**3 * 8 / 2      # rho/rho_bar at t(2pi) with R = A (= R_ta/2): 9/2 (2pi)^2 * (2)^3 / 2^3 ... see below
# explicit: rho/rho_bar = 9 (th - sin th)^2 / (2 (1-cos th)^3) evaluated with th=2pi but radius frozen at R=A (1-cos th -> 1)
dvir = sp.Rational(9, 2)*(2*sp.pi)**2/1**3
check("G14", "spherical collapse: delta_c = (3/20)(12 pi)^(2/3) = 1.6865; turnaround rho/rho_bar = 9pi^2/16; Delta_vir = 18 pi^2 = 177.65 (R_vir = R_ta/2, EdS)",
      abs(float(dc) - 1.68647) < 1e-4 and sp.simplify(nl_ta.subs(th, sp.pi) - 9*sp.pi**2/16) == 0 and sp.simplify(dvir - 18*sp.pi**2) == 0,
      {"delta_c": float(dc), "Delta_vir": float(dvir), "delta_lin_ta": float(dlin.subs(th, sp.pi))})
Wta = sp.Symbol('W_ta', negative=True)
Wvir = 2*Wta; Kvir = -Wvir/2
check("G15", "collisionless virialization from rest at turnaround: W_vir = 2 W_ta, K_vir = |W_ta|; energy conserved; all released |dW| goes to K (= half of final |W|)",
      sp.simplify(Kvir + Wvir - Wta) == 0 and sp.simplify(Kvir - (Wta - Wvir)) == 0 and sp.simplify(Kvir/(-Wvir) - sp.Rational(1, 2)) == 0)

# ---------- G16-G17: Layzer-Irvine cosmic energy equation ----------
# EdS linear: |W| ~ a (W per mass ~ D^2/a), K = (2/3)|W|  ; check d(K+W)/dt + H(2K+W) = 0
k1 = sp.Symbol('k1', positive=True)
Wabs = k1*a; K = sp.Rational(2, 3)*Wabs; Wv = -Wabs
LI = sp.simplify(sp.diff(K + Wv, t) + H*(2*K + Wv))
check("G16", "Layzer-Irvine d(K+W)/dt + H(2K+W) = 0 holds for linear EdS growth (K/|W| = 2/3, |W| ~ a)", LI == 0)
fsym = sp.Symbol('f', positive=True)
KoverW = sp.Rational(2, 3)*fsym**2/Om
check("G17", "linear theory K/|W| = 2 f^2/(3 Omega_m(a)) (= 2/3 in EdS); virial 1/2 is reached only by collapse; the expansion drain H(2K+W) vanishes at 2K+W=0",
      KoverW.subs({fsym: 1, Om: 1}) == sp.Rational(2, 3))

# ---------- G18: Press-Schechter collapsed fraction: rate and early asymptote ----------
dcs, sig, Dg = sp.symbols('delta_c sigma D', positive=True)
nu = dcs/(sig*Dg)
F = sp.erfc(nu/sp.sqrt(2))
dFdlnD = sp.simplify(Dg*sp.diff(F, Dg))
check("G18", "PS (excursion-set normalised) F = erfc(nu/sqrt2): dF/d ln D = sqrt(2/pi) nu exp(-nu^2/2)",
      sp.simplify(dFdlnD - sp.sqrt(2/sp.pi)*nu*sp.exp(-nu**2/2)) == 0)
# early asymptote in matter era D = a : a^2 ln F -> -delta_c^2/(2 sigma^2)
Fasym = sp.sqrt(2/sp.pi)/nu*sp.exp(-nu**2/2)
lim = sp.limit((Dg**2*sp.log(Fasym)), Dg, 0)
check("G19", "PS early activation: a^2 ln F -> -delta_c^2/(2 sigma^2) (matter era): Gaussian in 1/a, i.e. exp(-k/a^2), not exp(-k/a)",
      sp.simplify(lim + dcs**2/(2*sig**2)) == 0)

# ---------- numerics: LCDM growth (book's parameters) ----------
OM, ORr = 0.315, 9.1e-5
OLm = 1 - OM - ORr
def h2(av): return OM*av**-3 + ORr*av**-4 + OLm
def dlnh_dlna(av): return 0.5*(-3*OM*av**-3 - 4*ORr*av**-4)/h2(av)
def Om_a(av): return OM*av**-3/h2(av)
def rhs(lna, y):
    av = math.exp(lna); D, Dp = y
    return [Dp, -(2 + dlnh_dlna(av))*Dp + 1.5*Om_a(av)*D]
ai = 1e-6
# radiation era growing-mode ~ ln a, start near matter-radiation era with Meszaros-ish IC: use D=a, D'=a at a_i small enough in matter? start deep, error decays
sol = solve_ivp(rhs, [math.log(ai), math.log(50.0)], [ai, ai], dense_output=True, rtol=1e-10, atol=1e-14)
def Dfun(av):
    y = sol.sol(math.log(av)); return y[0]
D1 = Dfun(1.0)
def Dn_(av): return Dfun(av)/D1
def fgr(av):
    y = sol.sol(math.log(av)); return y[1]/y[0]
f0 = fgr(1.0)
check("G20", "LCDM growth solved (Om=0.315, Or=9.1e-5): f(1) in 0.50-0.55, linear K/|W| today = 2f^2/(3 Om) = 0.55-0.62 (not the virial 1/2)",
      0.50 < f0 < 0.55 and 0.55 < 2*f0**2/(3*OM) < 0.62, {"f0": round(f0, 4), "K_over_W_linear_today": round(2*f0**2/(3*OM), 4)})

# ---------- G21-G23: reproduce the book's Table (sec:numerics) ----------
def I_of(Rfun, amax, amin=1e-4, npts=4000):
    grid = np.exp(np.linspace(math.log(amin), math.log(amax), npts))
    vals = np.array([Rfun(x) for x in grid])            # integrand R(a)/a da = R d ln a
    lnag = np.log(grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5*(vals[1:] + vals[:-1])*np.diff(lnag))])
    return grid, cum
def fit_ab(Rfun, lo=0.15, hi=2.0, logspace=False):
    grid, cum = I_of(Rfun, hi)
    I1 = np.interp(0.0, np.log(grid), cum)
    sel = (grid >= lo) & (grid <= hi)
    av = grid[sel]; y = cum[sel]/I1
    if logspace:
        X = np.vstack([np.ones_like(av), -1/av]).T
        coef, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
        return float(coef[0]), float(coef[1]), av, y
    from scipy.optimize import curve_fit
    pp, _ = curve_fit(lambda xx, al, be: np.exp(al - be/xx), av, y, p0=[1, 1], maxfev=20000)
    return float(pp[0]), float(pp[1]), av, y
R_pow = lambda nn: (lambda av: Om_a(av)*fgr(av)*Dn_(av)**nn)
res_pow = {}
for nn in (2.0, 2.5, 3.0, 3.5, 4.0):
    al, be, _, _ = fit_ab(R_pow(nn)); res_pow[nn] = (round(al, 3), round(be, 3))
res_pow_log = {nn: tuple(round(v, 3) for v in fit_ab(R_pow(nn), logspace=True)[:2]) for nn in (3.5,)}
book_pow = {2.0: (0.53, 0.59), 2.5: (0.66, 0.74), 3.0: (0.79, 0.88), 3.5: (0.93, 1.02), 4.0: (1.06, 1.16)}
okp = all(abs(res_pow[k][0] - v[0]) <= 0.03 and abs(res_pow[k][1] - v[1]) <= 0.03 for k, v in book_pow.items())
check("G21", "book Table power-law rows (R = Om(a) f D^n, curve_fit of exp(alpha - beta/a) to I/I(1), 0.15-2: the book script's method) reproduced within 0.03",
      okp, {**{str(k): v for k, v in res_pow.items()}, "D3.5_fit_in_log_space": res_pow_log[3.5]})
def R_PS(sstar):
    def R(av):
        D = Dn_(av); nuv = 1.686/(sstar*D)
        return Om_a(av)*math.sqrt(2/math.pi)*nuv*math.exp(-nuv**2/2)*fgr(av)
    return R
res_ps = {s: tuple(round(v, 3) for v in fit_ab(R_PS(s))[:2]) for s in (1.0, 1.2)}
res_ps_log = {s: tuple(round(v, 3) for v in fit_ab(R_PS(s), logspace=True)[:2]) for s in (1.0, 1.2)}
book_ps = {1.0: (1.06, 1.24), 1.2: (0.86, 1.02)}
okps = all(abs(res_ps[k][0] - v[0]) <= 0.04 and abs(res_ps[k][1] - v[1]) <= 0.04 for k, v in book_ps.items())
check("G22", "book Table Press-Schechter rows (sigma*=1.0, 1.2) reproduced within 0.04 (linear-space fit); the same rows fitted in log space give beta > 2",
      okps and all(v[1] > 2 for v in res_ps_log.values()),
      {"linear": {str(k): v for k, v in res_ps.items()}, "log_space": {str(k): v for k, v in res_ps_log.items()}})
# beta = 1 is reached by choosing sigma* (or n): find the crossing
s_cross = brentq(lambda s: fit_ab(R_PS(s))[1] - 1.0, 0.9, 1.6)
n_cross = brentq(lambda nn: fit_ab(R_pow(nn))[1] - 1.0, 2.5, 4.5)
# window dependence for D^3.5
win = {w: tuple(round(v, 3) for v in fit_ab(R_pow(3.5), lo=w[0], hi=w[1])[:2]) for w in ((0.15, 1.0), (0.3, 1.0), (0.5, 2.0))}
check("G23", "beta = 1 is a tuning: crossing at sigma* and n found; fitted (alpha, beta) move with the window",
      1.0 < s_cross < 1.5 and 3.0 < n_cross < 4.0, {"sigma*_beta1": round(s_cross, 3), "n_beta1": round(n_cross, 3), "D3.5_windows": {str(k): v for k, v in win.items()}})

# ---------- G24: the table's integrand is back-loaded, eq:Idot's front-loaded (full LCDM, n = 7/2) ----------
def local_power(fun, a1, a2):
    return (math.log(fun(a2)) - math.log(fun(a1)))/(math.log(a2) - math.log(a1))
tab_pe = lambda av: Om_a(av)*fgr(av)*Dn_(av)**3.5                      # per e-fold, table
book_pe = lambda av: OM*av**-3*Dn_(av)**3.5*fgr(av)*math.sqrt(h2(av))  # rho_m D^n f H per e-fold (H cancels 1/H, times H/2 const)
pt, pb = local_power(tab_pe, 0.01, 0.1), local_power(book_pe, 0.01, 0.1)
check("G24", "full LCDM, n=7/2, 0.01<=a<=0.1: table per-e-fold power ~ +3.5 (rising), eq:Idot power ~ -1 (falling): opposite histories",
      abs(pt - 3.5) < 0.2 and abs(pb + 1.0) < 0.1, {"table_power": round(pt, 3), "eqIdot_power": round(pb, 3)})

# ---------- G25: shape of the PS collapse history vs exp(1-1/a) ----------
zs = [0.5, 1.0, 2.0, 3.0]
def lnF_PS(av, sstar):
    return math.log(erfc(1.686/(sstar*Dn_(av))/math.sqrt(2))) - math.log(erfc(1.686/sstar/math.sqrt(2)))
shape = {}
for s in (1.0, 1.2, 2.0):
    shape[s] = [round(lnF_PS(1/(1 + z), s), 3) for z in zs]
target_lnE = [-z for z in zs]
# the collapsed fraction (no horizon factor) is much flatter than E for galaxy-scale sigma* ~ 1-2
ratios = {s: [round(shape[s][i]/target_lnE[i], 3) for i in range(len(zs))] for s in shape}
check("G25", "ln[F_PS(a)/F_PS(1)] / ln E at z=0.5,1,2,3 rises with z for every sigma* (Gaussian tail, 1/D^2): no sigma* gives E's 1/a shape",
      all(all(r[i+1] > r[i] for i in range(len(r)-1)) for r in ratios.values()),
      {"lnE": target_lnE, **{f"lnF sigma*={k}": v for k, v in shape.items()}, **{f"ratio sigma*={k}": v for k, v in ratios.items()}})

# ---------- G26: needed first-law record rate vs shapes (log slopes today) ----------
need = lambda av: math.exp(1 - 1/av)/(av*h2(av)**2)                # E (H0/H)^4 / a
p_need_today = local_power(need, 0.98, 1.02)
p_need_md = local_power(need, 0.02, 0.03)
# analytic slope today with Or=0: 1/a - 1 + 6*Om_a ; check numerics agree
p_need_formula = 1/1.0 - 1 + 4*1.5*(OM + 4/3*ORr)/h2(1.0)
check("G26", "rate the first law needs for E(a): d ln(dS/dlna)/d ln a = 1/a - 1 - 4 dlnh/dlna (= 1.89 today, ~ 1/a+5 in matter era)",
      abs(p_need_today - p_need_formula) < 0.01 and abs(p_need_md - (1/0.0245 + 5)) < 3.0,
      {"today": round(p_need_today, 3), "matter_era_a0.02-0.03": round(p_need_md, 2)})

# ---------- G27: three different S_info(a) in the book are not proportional ----------
# (i) p2_03 eq:th:S72: S ~ phi + C (per e-fold 1/a); (ii) p2_11 eq:de_sinfo: S = S0 E (per e-fold E/a); (iii) first law: per e-fold E h^-4 /a
pe = {"p2_03 (S~ln E)": lambda av: 1/av, "p2_11 (S=S0 E)": lambda av: math.exp(1 - 1/av)/av, "first law (eq:law_Sneed)": need}
r05 = {k: round(v(0.5)/v(1.0), 4) for k, v in pe.items()}
check("G27", "per-e-fold writing at a=0.5 relative to today differs among the book's three S_info forms",
      len(set(r05.values())) == 3, r05)

# ---------- G28: what the book's own linear-closure model gives for the far future (register X7 cross-check) ----------
grid50, cum50 = I_of(R_pow(3.5), 50.0)
I1 = np.interp(0.0, np.log(grid50), cum50)
c_lin = math.log(cum50[-1]/I1)
check("G28", "book's numerical record (D^3.5, E=I/I(1)) grows only e^c with c ~ 0.3 by a=50, not e (X7 minicosmos value 0.33)",
      0.2 < c_lin < 0.45, {"c_linear_closure": round(c_lin, 3)})

# ---------- G29: the 1/a 'surface density' argument uses D/A_H, inconsistent with n=7/2 bookkeeping ----------
# p2_03 eq:th:surfdens: D/A_H ~ a^-2 per da. With n=7/2 the book's per-da rate is a^(n-11/2) = a^-2 as well, but the
# remaining factor rho_m D^(n-1) f /(T_H a) must be constant (app_C3:160). Check it is a^(n-7/2).
rem = sp.simplify(-3 + (n - 1) - sp.Rational(-3, 2) - 1)
check("G29", "eq:th:surfdens (D/A_H ~ a^-2) equals the book's rate only if rho_m D^(n-1) f/(T_H a) ~ a^(n-7/2) is constant, i.e. it restates n=7/2",
      sp.simplify(rem - (n - sp.Rational(7, 2))) == 0)

# ---------- G30: E(z)=exp(-z) and per-e-fold writing E/a peaks at a=1 for c=1 (a=c in general) ----------
Ec = sp.exp(c*(1 - 1/A))
peak = sp.solve(sp.diff(Ec/A, A), A)
check("G30", "for E = exp(c(1-1/a)) the writing per e-fold dE/dln a peaks at a = c (book's a=1 is c=1)",
      peak == [c])

npass = sum(r["pass"] for r in RESULTS); nfail = len(RESULTS) - npass
out = {"script": "spider_s3_activation_checks.py", "checks_passed": npass, "checks_failed": nfail, "results": RESULTS}
json.dump(out, open("spider_s3_activation_checks_output.json", "w"), indent=1, default=str)
print(f"{npass} passed, {nfail} failed")
