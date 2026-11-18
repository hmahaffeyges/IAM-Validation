import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
#!/usr/bin/env python3
"""SPIDER S2 checks, 2026-11-18 (private; proposals only).
Closes the gaps declared by the long-form pass (IAM_LAW_LONG_FORM_2026-11-18.md).
Sections: CO (Collins Sec. 1.2, equation by equation), LL (Landau-Lifshitz Sec. 34 via Gaite 2013),
GR (Misner-Sharp, LTB, Schwarzschild interior, Buchdahl, TOV + Juttner gas sequence, Tolman mass),
KH (Kodama-Hayward factor with radiation), KO (Koide chance estimate under stated priors).
Run: python3 s2_checks.py -> s2_checks_output.json   (sympy, mpmath, numpy, scipy; < 2 min on a laptop core)"""
import json, math
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

RES = []
def check(cid, desc, ok, detail=""):
    RES.append({"id": cid, "desc": desc, "pass": bool(ok), "detail": str(detail)[:600]})

def zero(e):
    return sp.simplify(sp.expand(e)) == 0

# =====================================================================  CO. Collins 1978, Sec. 1.2
t = sp.symbols('t', real=True)
N = 3
ms = sp.symbols('m1:4', positive=True)
X = [[sp.Function(f'x{i}{k}')(t) for k in range(3)] for i in range(N)]
R = [sp.Matrix(X[i]) for i in range(N)]
V = [r.diff(t) for r in R]
P = [ms[i]*V[i] for i in range(N)]
I = sum(ms[i]*R[i].dot(R[i]) for i in range(N))
G = sum(P[i].dot(R[i]) for i in range(N))
T = sp.Rational(1, 2)*sum(ms[i]*V[i].dot(V[i]) for i in range(N))
# (1.2.2)-(1.2.3): G = sum p.r = (1/2) dI/dt
check("CO1", "Collins (1.2.2)-(1.2.3): G = sum p_i.r_i = (1/2) dI/dt", zero(G - I.diff(t)/2))
# (1.2.4)-(1.2.6) with (1.2.1) p_dot = f: dG/dt = 2T + sum f.r
F = [ms[i]*R[i].diff(t, 2) for i in range(N)]
check("CO2", "Collins (1.2.4)-(1.2.6): dG/dt = sum rdot.p + sum pdot.r = 2T + sum f_i.r_i (needs p = m rdot: Newtonian)",
      zero(G.diff(t) - 2*T - sum(F[i].dot(R[i]) for i in range(N))))
# (1.2.8)-(1.2.9): potential a r^n (degree n), F_ij = -grad_i(a r_ij^n) = -n a r^(n-2) (r_i - r_j)
n_, a_ = sp.symbols('n a', real=True)
x1, y1, z1, x2, y2, z2 = sp.symbols('x1 y1 z1 x2 y2 z2', real=True)
r12 = sp.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
U12 = a_*r12**n_
Fij = sp.Matrix([-sp.diff(U12, v) for v in (x1, y1, z1)])
check("CO3", "Collins (1.2.8)-(1.2.9): F_ij = -n a_ij r_ij^(n-2)(r_i - r_j); n is the DEGREE OF THE POTENTIAL (gravity n = -1), "
      "not the force exponent", all(zero(Fij[k] - (-n_*a_*r12**(n_-2)*sp.Matrix([x1-x2, y1-y2, z1-z2])[k])) for k in range(3)))
# (1.2.10) as printed: f_i = sum_j F_ij = sum_{j>i} (F_ij + F_ji). With F_ji = -F_ij the right side is 0 for every pair,
# so (1.2.10) is not an identity as printed (index slip). (1.2.11) (pairing after dotting with positions) IS correct.
Fs = sp.symbols('F0:3', real=True); Fv = sp.Matrix(Fs)
lhs_1210 = Fv                      # force on particle 1 from particle 2 in a two-body system
rhs_1210 = Fv + (-Fv)              # F_12 + F_21
r1v = sp.Matrix([x1, y1, z1]); r2v = sp.Matrix([x2, y2, z2])
check("CO4", "Collins (1.2.10) as printed is NOT an identity (RHS F_ij + F_ji = 0 by Newton III; LHS = F_ij != 0); "
      "(1.2.11) sum f_i.r_i = sum_{i<j} F_ij.(r_i - r_j) IS correct",
      (rhs_1210 == sp.zeros(3, 1)) and (lhs_1210 != sp.zeros(3, 1)) and
      zero(Fv.dot(r1v) + (-Fv).dot(r2v) - Fv.dot(r1v - r2v)))
# (1.2.12): dG/dt = 2T - nU, three bodies, general n (Euler)
xs = sp.symbols('xa0:9', real=True)
pos = [sp.Matrix(xs[3*i:3*i+3]) for i in range(3)]
aij = sp.symbols('a12 a13 a23', real=True)
pairs = [(0, 1), (0, 2), (1, 2)]
Utot = sum(aij[k]*sp.sqrt((pos[i]-pos[j]).dot(pos[i]-pos[j]))**n_ for k, (i, j) in enumerate(pairs))
vir = sum(pos[i].dot(sp.Matrix([-sp.diff(Utot, v) for v in xs[3*i:3*i+3]])) for i in range(3))
check("CO5", "Collins (1.2.12): sum f_i.r_i = -n U for pair potentials of degree n (3 bodies, general n)",
      zero(sp.simplify(vir + n_*Utot)))
# (1.2.13)-(1.2.16) on a bound Kepler orbit (n = -1), numerically over one period: (G(T0)-G(0))/T0 = 2<T> - n<U> = 0
mu_, ecc = 1.0, 0.6
def kepler(tt, y):
    rr = math.hypot(y[0], y[1]); return [y[2], y[3], -mu_*y[0]/rr**3, -mu_*y[1]/rr**3]
aK = 1.0; rp = aK*(1-ecc); vp = math.sqrt(mu_*(1+ecc)/(aK*(1-ecc)))
TK = 2*math.pi*math.sqrt(aK**3/mu_)
sol = solve_ivp(kepler, [0, TK], [rp, 0, 0, vp], rtol=1e-11, atol=1e-13, dense_output=True)
ts = np.linspace(0, TK, 200001); Y = sol.sol(ts)
Tk = 0.5*(Y[2]**2 + Y[3]**2); Uk = -mu_/np.hypot(Y[0], Y[1])
Tbar = np.trapezoid(Tk, ts)/TK; Ubar = np.trapezoid(Uk, ts)/TK
Gt = Y[0]*Y[2] + Y[1]*Y[3]
check("CO6", "Collins (1.2.14)-(1.2.16): over one Kepler period (e = 0.6) [G(T0)-G(0)]/T0 = 0 and 2<T> + <U> = 0",
      abs((Gt[-1]-Gt[0])/TK) < 1e-8 and abs(2*Tbar + Ubar) < 1e-6, f"2<T>+<U> = {2*Tbar+Ubar:.2e}")

# =====================================================================  LL. Landau-Lifshitz Sec. 34 (via Gaite 2013)
p_, mm, c_, Zk, rr_ = sp.symbols('p m c k r', positive=True)
# Gaite eq (24): circular relativistic Coulomb orbit, c p^2/sqrt(p^2+m^2c^2) = k/r ; E = c sqrt(p^2+m^2c^2) - k/r
Ecirc = c_*sp.sqrt(p_**2 + mm**2*c_**2) - c_*p_**2/sp.sqrt(p_**2 + mm**2*c_**2)
vrel = c_*p_/sp.sqrt(p_**2 + mm**2*c_**2)
check("LL1", "LL Sec. 34 particle form on the circular relativistic Coulomb orbit: E = m c^2 sqrt(1 - v^2/c^2) exactly (no average needed)",
      zero(sp.simplify(Ecirc - mm*c_**2*sp.sqrt(1 - vrel**2/c_**2))))
check("LL2", "ultrarelativistic limit p -> infinity: E -> 0 (Gaite Sec. 2.3: the bound state becomes scale invariant)",
      sp.limit(Ecirc, p_, sp.oo) == 0)
# Gaite eq (28): traceless content (massless gas) E_g = 3 P V; pressure balance P = -P_vac = B; E_vac = B V -> E = 4BV, traceless share 3/4
B_, Vv = sp.symbols('B V', positive=True)
Eg = 3*B_*Vv; Evac = B_*Vv
check("LL3", "Gaite eq (28): massless (traceless) content carries 3/4 of a trace-confined bound state, the trace part 1/4; "
      "same identity p = u/3 that gives black-body T S = (4/3) U",
      sp.simplify(Eg/(Eg + Evac)) == sp.Rational(3, 4) and sp.simplify((Eg + B_*Vv*3/3)/Eg) == sp.Rational(4, 3))

# =====================================================================  GR. compactness 1/2 in full GR
# GR1: flat FRW, areal radius R = a r: g^{ab} d_a R d_b R = 1 - H^2 R^2 = 1 - 2 G m_MS/R with m_MS = (4 pi/3) rho R^3 (Friedmann)
tt, rc, Gn, rho = sp.symbols('t r_c G rho', positive=True)
af = sp.Function('a')(tt)
Rareal = af*rc
grad2 = -(sp.diff(Rareal, tt))**2 + (sp.diff(Rareal, rc))**2/af**2     # inverse metric diag(-1, 1/a^2) (c = 1)
H = sp.diff(af, tt)/af
m_ms = sp.Rational(4, 3)*sp.pi*rho*Rareal**3
fried = {sp.diff(af, tt): af*sp.sqrt(sp.Rational(8, 3)*sp.pi*Gn*rho)}
check("GR1", "flat FRW: g^ab dR dR = 1 - H^2R^2 = 1 - 2G m_MS/R (Misner-Sharp, exact GR); trapping <=> G m_MS/(R c^2) = 1/2 <=> R = c/H",
      zero((grad2 - (1 - 2*Gn*m_ms/Rareal)).subs(fried)) and zero(grad2 - (1 - H**2*Rareal**2)))
# GR2: Misner-Sharp mass in FRW changes only by pressure work: dm/dt = -4 pi R^2 p dR/dt; dust: m conserved -> E = Rdot^2/2 - Gm/R conserved
rhof = sp.Function('rho')(tt); pf = sp.Function('p')(tt)
mfr = sp.Rational(4, 3)*sp.pi*rhof*Rareal**3
cont = {sp.diff(rhof, tt): -3*H*(rhof + pf)}
dm = sp.diff(mfr, tt).subs(cont)
Eshell = sp.Rational(1, 2)*sp.diff(Rareal, tt)**2 - Gn*mfr/Rareal
check("GR2", "FRW: dm_MS/dt = -4 pi R^2 p dR/dt (exact); for dust m_MS is constant and flat FRW has E_shell = Rdot^2/2 - G m/R = 0 exactly "
      "(K = |V| per unit mass is an exact GR statement for dust shells: Lemaitre-Tolman-Bondi energy function)",
      zero(dm + 4*sp.pi*Rareal**2*pf*sp.diff(Rareal, tt)) and
      zero(Eshell.subs(sp.diff(af, tt), af*sp.sqrt(sp.Rational(8, 3)*sp.pi*Gn*rhof))))
# GR3: Schwarzschild interior (uniform density) solves TOV; central pressure diverges at Xi = GM/(Rc^2) = 4/9 (Buchdahl)
rs_, Ms_, Rs_, rh = sp.symbols('r M R rho0', positive=True)
A1 = sp.sqrt(1 - 2*Ms_*rs_**2/Rs_**3); A0 = sp.sqrt(1 - 2*Ms_/Rs_)
pint = rh*(A1 - A0)/(3*A0 - A1)
mr = Ms_*rs_**3/Rs_**3
tov = sp.diff(pint, rs_) + (rh + pint)*(mr + 4*sp.pi*rs_**3*pint)/(rs_*(rs_ - 2*mr))
tov_num = [abs(complex(tov.subs({rh: 3*Mv/(4*sp.pi*Rv**3), Ms_: Mv, Rs_: Rv, rs_: rv}).evalf())) for (Mv, Rv, rv) in
           [(0.1, 1.0, 0.3), (0.3, 1.0, 0.7), (0.2, 0.5, 0.25), (0.42, 1.0, 0.9)]]
Xi = sp.symbols('Xi', positive=True)
pc = sp.simplify(pint.subs(rs_, 0).subs(Ms_, Xi*Rs_)/rh)
check("GR3", "Schwarzschild interior solves TOV (4 spot points, |residual| < 1e-12); p_c/rho c^2 = (1-sqrt(1-2Xi))/(3sqrt(1-2Xi)-1) "
      "diverges at Xi = 4/9 < 1/2 (Buchdahl 1959): no static star reaches compactness 1/2",
      max(tov_num) < 1e-12 and sp.solve(sp.Eq(3*sp.sqrt(1 - 2*Xi) - 1, 0), Xi) == [sp.Rational(4, 9)],
      f"max TOV residual {max(tov_num):.1e}; p_c(Xi) = {pc}")
# GR4: TOV sequence of an isentropic ideal Juttner gas (G = c = m = 1). theta = kT/mc^2.
mp.mp.dps = 40
def u_of(th):   # mean energy per particle / mc^2
    z = 1/mp.mpf(th); return mp.besselk(1, z)/mp.besselk(2, z) + 3/z
lth = np.linspace(math.log(1e-7), math.log(50.0), 1400)
uu = np.array([float(u_of(math.exp(x))) for x in lth])
cv = np.array([float(mp.diff(lambda s: u_of(mp.e**s), x)/mp.e**x) for x in lth])   # du/dtheta
ln_n = 1.5*lth + np.concatenate([[0.0], np.cumsum(0.5*((cv[1:]-1.5) + (cv[:-1]-1.5))*np.diff(lth))])  # adiabat: dln n/dtheta = c_v/theta
S_u, S_cv, S_ln = CubicSpline(lth, uu - 1), CubicSpline(lth, cv), CubicSpline(lth, ln_n)
def eos(th):
    if th <= 1e-7:
        return math.exp(1.5*math.log(th) + float(S_ln(lth[0])) - 1.5*lth[0]), 1.5*th, 1.5
    x = math.log(th); return math.exp(float(S_ln(x))), float(S_u(x)), float(S_cv(x))
check("GR4a", "Juttner gas tables: c_v -> 3/2 (cold) and -> 3 (ultrarelativistic); u - 1 -> 3 theta/2 (cold)",
      abs(cv[0] - 1.5) < 1e-5 and abs(cv[-1] - 3) < 0.02 and abs((uu[0]-1)/(1.5e-7) - 1) < 1e-5, f"cv[0]={cv[0]:.6f}, cv[-1]={cv[-1]:.4f}")

def star(thc):
    def rhs(r, y):
        m, th, nu, M0, Ep, IT = y
        th = max(th, 1e-14)
        n, um1, c_v = eos(th); p = n*th; e = n*(1 + um1)
        A = (m + 4*math.pi*r**3*p)/(r*(r - 2*m))
        dpdr = -(e + p)*A
        g = 1/math.sqrt(1 - 2*m/r)
        return [4*math.pi*r**2*e, dpdr/(n*(1 + c_v)), 2*A, 4*math.pi*r**2*n*g, 4*math.pi*r**2*e*g,
                4*math.pi*r**2*(e + 3*p)*math.exp(nu/2)*g]
    n, um1, _ = eos(thc); e_c = n*(1 + um1); r0 = 1e-6
    y0 = [4/3*math.pi*r0**3*e_c, thc, 0.0, 0.0, 0.0, 0.0]
    ev = lambda r, y: y[1] - 1e-9*thc; ev.terminal = True; ev.direction = -1
    s = solve_ivp(rhs, [r0, 1e4], y0, events=ev, rtol=1e-10, atol=1e-14, method='LSODA')
    Rr = s.t[-1]; m, th, nu, M0, Ep, IT = s.y[:, -1]
    C = math.log(1 - 2*m/Rr) - nu            # normalize lapse: e^{nu(R)} = 1 - 2M/R
    MT = IT*math.exp(C/2)
    U = Ep - M0; W = m - Ep
    return dict(theta_c=thc, M=m, R=Rr, Xi=m/Rr, M0=M0, U_over_W=U/abs(W), Eb_over_M0=(m - M0)/M0, tolman_rel_err=abs(MT - m)/m)

thcs = np.exp(np.linspace(math.log(1e-4), math.log(20.0), 70))
seq = [star(x) for x in thcs]
Ms = np.array([s_['M'] for s_ in seq])
imax = int(np.argmax(Ms))
# refine the maximum
from scipy.optimize import minimize_scalar
if 0 < imax < len(seq) - 1:
    res = minimize_scalar(lambda lx: -star(math.exp(lx))['M'], bracket=(math.log(thcs[imax-1]), math.log(thcs[imax]), math.log(thcs[imax+1])),
                          options={'xtol': 1e-6})
    smax = star(math.exp(res.x))
else:
    smax = seq[imax]
# point where binding energy vanishes (M = M0) along the sequence, if any
Eb = np.array([s_['Eb_over_M0'] for s_ in seq])
izero = [i for i in range(len(Eb)-1) if Eb[i] < 0 <= Eb[i+1]]
s_eb0 = None
if izero:
    i0 = izero[0]
    lx0 = brentq(lambda lx: star(math.exp(lx))['Eb_over_M0'], math.log(thcs[i0]), math.log(thcs[i0+1]), xtol=1e-6)
    s_eb0 = star(math.exp(lx0))
maxXi = max(s_['Xi'] for s_ in seq)
check("GR4b", "Tolman mass = TOV (ADM) mass for every model of the sequence (rel. err < 1e-5): the GR virial identity, numerically",
      max(s_['tolman_rel_err'] for s_ in seq) < 1e-5, f"max rel err {max(s_['tolman_rel_err'] for s_ in seq):.1e}")
check("GR4c", "cold limit (theta_c = 1e-4): U/|W| -> 1/2 (Newtonian virial half recovered from the GR sequence)",
      abs(seq[0]['U_over_W'] - 0.5) < 2e-3, f"U/|W| = {seq[0]['U_over_W']:.5f}, Xi = {seq[0]['Xi']:.2e}")
check("GR4d", "maximum mass (turning point, onset of radial instability) exists at Xi < 4/9 and the star is still BOUND there "
      "(E_b = (M - M0) < 0, U/|W| < 1): GR instability pre-empts the Newtonian E = 0 endpoint",
      0 < imax < len(seq) - 1 and smax['Xi'] < 4/9 and smax['Eb_over_M0'] < 0 and smax['U_over_W'] < 1,
      {k: round(v, 5) for k, v in smax.items()})
check("GR4e", "no model of the whole sequence (theta_c up to 20) reaches compactness 1/2; max Xi along the sequence < 4/9",
      maxXi < 4/9, f"max Xi = {maxXi:.4f}; E_b = 0 at {({k: round(v, 5) for k, v in s_eb0.items()} if s_eb0 else 'not reached')}")
# Newtonian-gravity estimate of the long-form spider (C4) at the same central temperature, for comparison
def KW_newton(th):
    z = mp.mpf(1)/th; Ebar = mp.besselk(1, z)/mp.besselk(2, z) + 3/z
    return float((Ebar - 1)*z/3)
seq_out = [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in s_.items()} for s_ in seq[::5]]

# =====================================================================  KH. Kodama-Hayward factor with radiation
eps_, q_ = sp.symbols('epsilon q', real=True)
check("KH1", "T_KH/T_GH = 1 - eps/2 = (1 - q)/2 with q = eps - 1 (deceleration parameter): equals 1/2 exactly at q = 0",
      zero((1 - eps_/2) - (1 - (eps_ - 1))/2))
a_s, Om, Orad, OL = sp.symbols('a Omega_m Omega_r Omega_L', positive=True)
E2 = Om*a_s**-3 + Orad*a_s**-4 + OL
eps_expr = -a_s*sp.diff(E2, a_s)/(2*E2)       # eps = -dlnH/dlna
check("KH2", "flat LCDM + radiation: eps = (3/2) Omega_m(a) + 2 Omega_r(a); T_KH/T_GH = 1 - (3/4)Omega_m(a) - Omega_r(a)",
      zero(eps_expr - (sp.Rational(3, 2)*Om*a_s**-3 + 2*Orad*a_s**-4)/E2))
# radiation density today from T_CMB = 2.7255 K
kB, hbar, cc, Gc = 1.380649e-23, 1.054571817e-34, 2.99792458e8, 6.67430e-11
Tcmb = 2.7255
rho_g = (math.pi**2/15)*(kB*Tcmb)**4/(hbar**3*cc**5)               # kg/m^3 equivalent (energy/c^2)
H100 = 100e3/3.0856775814913673e22
rho_c100 = 3*H100**2/(8*math.pi*Gc)
Og_h2 = rho_g/rho_c100
nu_fac = 7/8*(4/11)**(4/3)
h = 0.6736; Om0 = 0.3153
cases = {"photons only": Og_h2/h**2, "photons + 2 massless nu (Planck base: one 0.06 eV)": Og_h2*(1 + nu_fac*2.044)/h**2,
         "photons + N_eff = 3.044 massless": Og_h2*(1 + nu_fac*3.044)/h**2}
fac = {k: 1 - 0.75*Om0 - v for k, v in cases.items()}
check("KH3", "Omega_gamma h^2 = 2.47e-5 (T_CMB 2.7255 K); today's T_KH/T_GH with radiation 0.76343-0.76350 vs 0.763525 without: "
      "the correction is -Omega_r, at most -9.2e-5", abs(Og_h2 - 2.473e-5) < 0.005e-5 and
      all(0.76340 < v < 0.763526 for v in fac.values()), {k: round(v, 6) for k, v in fac.items()} | {"Og_h2": f"{Og_h2:.4e}"})
Or0 = cases["photons + N_eff = 3.044 massless"]; OL0 = 1 - Om0 - Or0
def ratio_KH(z):
    a = 1/(1+z); E2v = Om0*a**-3 + Or0*a**-4 + OL0
    return 1 - 0.75*Om0*a**-3/E2v - Or0*a**-4/E2v
zq0 = brentq(lambda z: ratio_KH(z) - 0.5, 0.1, 2.0)
zeq = Om0/Or0 - 1
hist = {f"z={z}": round(ratio_KH(z), 5) for z in [0, 0.5, 1, 3, 10, 100, 1000, round(zeq), 1e5]}
check("KH4", "history of T_KH/T_GH (radiation included): 1/2 exactly at the q = 0 epoch; 0.25 deep in matter era; -> 0 in radiation era",
      abs(ratio_KH(0) - fac["photons + N_eff = 3.044 massless"]) < 1e-12 and 0.6 < zq0 < 0.7 and ratio_KH(1e5) < 0.02,
      {"z(q=0)": round(zq0, 4), "z_eq": round(zeq, 1)} | hist)

# =====================================================================  KO. Koide chance estimate under stated priors
me, mmu, mtau, smtau = 0.51099895000, 105.6583755, 1776.93, 0.09
def Qf(m1, m2, m3):
    return (m1 + m2 + m3)/(math.sqrt(m1) + math.sqrt(m2) + math.sqrt(m3))**2
Qobs = Qf(me, mmu, mtau)
mtau_star = brentq(lambda x: Qf(me, mmu, x) - 2/3, 1000, 3000)
dQdm = (Qf(me, mmu, mtau + 1e-4) - Qf(me, mmu, mtau - 1e-4))/2e-4
sigQ = abs(dQdm)*smtau
zobs = (2/3 - Qobs)/sigQ
check("KO1", "inputs: Q = 0.66666446, sigma_Q = 5.08e-6 (from m_tau 1776.93 +/- 0.09 MeV), Koide m_tau* = 1776.969 MeV, pull 0.43",
      abs(Qobs - 0.66666446) < 2e-8 and abs(sigQ - 5.08e-6) < 0.05e-6 and abs(mtau_star - 1776.969) < 0.001 and abs(zobs - 0.43) < 0.02,
      f"Q={Qobs:.8f}, sigQ={sigQ:.3e}, m*={mtau_star:.4f}, z={zobs:.3f}")
# local probability P(|Q - 2/3| < sigQ) = 2 sigQ p0(2/3) and Bayes factor BF = phi(z)/(sigQ p0(2/3)) for a point hypothesis Q = 2/3
phi = math.exp(-zobs**2/2)/math.sqrt(2*math.pi)
rng = np.random.default_rng(20261118)
def dens_mc(sampler, w=0.004, nrep=8, nper=1_000_000):
    hits = 0; tot = 0
    for _ in range(nrep):
        s3 = sampler(nper); Qd = (s3**2).sum(1)/s3.sum(1)**2
        hits += int((np.abs(Qd - 2/3) < w).sum()); tot += nper
    p = hits/tot/(2*w); return p, p/math.sqrt(max(hits, 1))
priors = {}
priors["P1 Q uniform on [1/3, 1]"] = (1.5, 0.0)
priors["P2 sqrt(m) iid exponential (one-mode prior; centred on the hypothesis)"] = dens_mc(lambda k: rng.exponential(size=(k, 3)))
for L in (3.5, 6.0, 12.0):
    priors[f"P3 m iid log-uniform over {L} decades"] = dens_mc(lambda k, L=L: np.sqrt(10**(rng.uniform(0, L, size=(k, 3)))))
priors["P4 m iid uniform on [0, 1]"] = dens_mc(lambda k: np.sqrt(rng.uniform(0, 1, size=(k, 3))))
# conditional priors: m_e, m_mu fixed; only m_tau uncertain. p_Q(2/3) = p(m*)/|dQ/dm| at m*
dQdm_star = abs((Qf(me, mmu, mtau_star + 1e-4) - Qf(me, mmu, mtau_star - 1e-4))/2e-4)
lo, hi = mmu, 1e5
priors["P5 m_e, m_mu fixed; m_tau log-uniform on [m_mu, 100 GeV]"] = ((1/(mtau_star*math.log(hi/lo)))/dQdm_star, 0.0)
mu_h, sd_h = 1784.0, 3.5      # DELCO 1978: 1784 (+3, -4) MeV, symmetrized
priors["P6 m_e, m_mu fixed; m_tau ~ N(1784, 3.5) MeV (pre-1992 knowledge, DELCO 1978)"] = (
    (math.exp(-((mtau_star - mu_h)/sd_h)**2/2)/(sd_h*math.sqrt(2*math.pi)))/dQdm_star, 0.0)
mu_h2 = 1782.0   # DELCO 1978 as listed at the conference / by some reviews: 1782 (+2 or +3, -4 to -7); sources differ
priors["P6b m_e, m_mu fixed; m_tau ~ N(1782, 3.5) MeV (alternative quoting of DELCO 1978)"] = (
    (math.exp(-((mtau_star - mu_h2)/sd_h)**2/2)/(sd_h*math.sqrt(2*math.pi)))/dQdm_star, 0.0)
table = {}
for k, (p0, dp0) in priors.items():
    table[k] = {"p0(2/3)": round(p0, 3), "p0_err": round(dp0, 3), "P(|Q-2/3|<sigma)": float(f"{2*sigQ*p0:.3g}"),
                "Bayes_factor_Q=2/3_vs_prior": float(f"{phi/(sigQ*p0):.3g}")}
check("KO2", "the chance number depends on the prior: P(|Q-2/3| < sigma_Q) spans more than two decades across seven stated priors",
      max(v["P(|Q-2/3|<sigma)"] for v in table.values())/min(v["P(|Q-2/3|<sigma)"] for v in table.values()) > 100, table)
p2 = table["P2 sqrt(m) iid exponential (one-mode prior; centred on the hypothesis)"]["P(|Q-2/3|<sigma)"]
check("KO3", "reproduces the long-form D11 number (~9e-6) as the P2 case; P2 is the hypothesis' own distribution, so it is not a null",
      5e-6 < p2 < 1.4e-5, f"P2 local probability {p2:.3g}")
# look-elsewhere: count simple targets (distinct rationals p/q in (1/3, 1) with q <= qmax)
from fractions import Fraction
def n_targets(qmax):
    return len({Fraction(p, q) for q in range(1, qmax+1) for p in range(1, q+1) if Fraction(1, 3) < Fraction(p, q) < 1})
lee = {qm: n_targets(qm) for qm in (3, 6, 12)}
P1loc = table["P1 Q uniform on [1/3, 1]"]["P(|Q-2/3|<sigma)"]
from math import gcd
lee_indep = {qm: sum(1 for q in range(1, qm+1) for p in range(1, q) if gcd(p, q) == 1 and 3*p > q) for qm in (3, 6, 12)}
check("KO4", "look-elsewhere: reduced rationals p/q in (1/3, 1) with q <= 3, 6, 12 number 2, 7, 30 (two independent counts agree); "
      "P1 global = local x number of targets",
      lee == lee_indep and lee[3] == 2 and lee[6] == 7 and lee[12] == 30, {f"q<={k}": v for k, v in lee.items()} |
      {f"P1_global_q<={k}": float(f"{P1loc*v:.3g}") for k, v in lee.items()})
# the predictive (pre-registered) test: given 1978 knowledge, how probable that later precise measurements land within
# today's 1-sigma (0.09 MeV) of the Koide value
p_pred = (math.erf((mtau_star + smtau - mu_h)/(sd_h*math.sqrt(2))) - math.erf((mtau_star - smtau - mu_h)/(sd_h*math.sqrt(2))))/2
Q_1978 = Qf(me, mmu, mu_h); sQ_1978 = dQdm_star*sd_h
p_pred2 = (math.erf((mtau_star + smtau - mu_h2)/(sd_h*math.sqrt(2))) - math.erf((mtau_star - smtau - mu_h2)/(sd_h*math.sqrt(2))))/2
check("KO5", "predictive form: with 1978 knowledge (m_tau = 1784 +/- 3.5), Q was (2.1 sigma) above 2/3; chance that the true m_tau "
      "lies within +/-0.09 MeV of Koide's value is ~2e-3 (one formula, no look-elsewhere inside the test)",
      1e-3 < p_pred < 4e-3 and 1.9 < (Q_1978 - 2/3)/sQ_1978 < 2.3,
      f"p_pred(1784)={p_pred:.3g}; p_pred(1782)={p_pred2:.3g}; Q(1784)={Q_1978:.6f}; pull={(Q_1978-2/3)/sQ_1978:.2f}; "
      f"pull(1782)={(Qf(me, mmu, mu_h2)-2/3)/sQ_1978:.2f}")

# ===================================================================== output
npass = sum(r['pass'] for r in RES); nfail = len(RES) - npass
out = {"checks": RES, "passed": npass, "failed": nfail,
       "tov_sequence_every5th": seq_out, "tov_max_mass": smax, "tov_Eb_zero": s_eb0,
       "newtonian_C4_KW_at_theta_c_of_max": KW_newton(smax['theta_c']),
       "koide_prior_table": table, "kodama_hayward_today": fac, "kodama_hayward_history": hist}
json.dump(out, open("s2_checks_output.json", "w"), indent=1, default=float)
for r in RES:
    print(("PASS " if r['pass'] else "FAIL ") + r['id'] + "  " + r['detail'][:150])
print(f"passed {npass}, failed {nfail}")
