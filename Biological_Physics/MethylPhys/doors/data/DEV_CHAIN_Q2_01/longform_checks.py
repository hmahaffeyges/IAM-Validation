import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
#!/usr/bin/env python3
"""SPIDER long-form checks, 2026-11-18 (private; proposals only).
Every step tagged DERIVED HERE in IAM_LAW_LONG_FORM_2026-11-18.md has a check id here.
Run: python3 spider_longform_checks.py  -> spider_longform_checks_output.json
Light: pure sympy/mpmath plus one chunked numpy Monte Carlo (D11)."""
import json, math
import sympy as sp
import mpmath as mp

RES = []
def check(cid, desc, ok, detail=""):
    RES.append({"id": cid, "desc": desc, "pass": bool(ok), "detail": str(detail)[:300]})

def is_zero(expr):
    return sp.simplify(sp.expand(expr)) == 0

t = sp.symbols('t', real=True)
# ---------------------------------------------------------------- A. virial from first principles
# A1 Lagrange identity: (1/2) d2I/dt2 = 2T + sum r_i.F_i   (Newton m_i r_i'' = F_i)
m1, m2 = sp.symbols('m1 m2', positive=True)
r1 = sp.Matrix([sp.Function(f'x1{c}')(t) for c in 'xyz'])
r2 = sp.Matrix([sp.Function(f'x2{c}')(t) for c in 'xyz'])
I = m1*r1.dot(r1) + m2*r2.dot(r2)
T = sp.Rational(1, 2)*(m1*r1.diff(t).dot(r1.diff(t)) + m2*r2.diff(t).dot(r2.diff(t)))
F1 = m1*r1.diff(t, 2); F2 = m2*r2.diff(t, 2)          # Newton: F = m a
lhs = sp.Rational(1, 2)*I.diff(t, 2)
rhs = 2*T + r1.dot(F1) + r2.dot(F2)
check("A1", "Lagrange identity (1/2) I'' = 2T + sum r.F", is_zero(lhs - rhs))

# A2 Euler for a pairwise power-law potential, 3 bodies in 3D: sum r_i.grad_i V = k V
k = sp.symbols('k', real=True)
P = [sp.Matrix(sp.symbols(f'p{i}x p{i}y p{i}z', real=True)) for i in range(3)]
c = sp.symbols('c01 c02 c12', real=True)
pairs = [(0, 1, c[0]), (0, 2, c[1]), (1, 2, c[2])]
V = sum(cc*((P[i]-P[j]).dot(P[i]-P[j]))**(k/2) for i, j, cc in pairs)
eul = sum(P[i].dot(sp.Matrix([sp.diff(V, s) for s in P[i]])) for i in range(3))
check("A2", "Euler: sum r_i.grad_i V = k V for V = sum c_ij |r_i-r_j|^k", is_zero(sp.simplify(eul - k*V)))

# A3 pairing by Newton's third law: r_i.F_ij + r_j.F_ji = (r_i - r_j).F_ij
Fij = sp.Matrix(sp.symbols('fx fy fz', real=True))
check("A3", "third-law pairing r_i.F_ij + r_j.F_ji = (r_i-r_j).F_ij",
      is_zero(P[0].dot(Fij) + P[1].dot(-Fij) - (P[0]-P[1]).dot(Fij)))

# A4 time averages on a bound Kepler orbit (eccentric anomaly): <1/r> = 1/a, so <V> = 2E, <T> = -E
a, e, GMm = sp.symbols('a e GMm', positive=True)
Ean = sp.symbols('E_an', real=True)
r_E = a*(1 - e*sp.cos(Ean)); dt_dE = (1 - e*sp.cos(Ean))          # dt = (1 - e cos E) dE / n
avg_inv_r = sp.integrate(dt_dE/r_E, (Ean, 0, 2*sp.pi))/sp.integrate(dt_dE, (Ean, 0, 2*sp.pi))
Etot = -GMm/(2*a)
Vavg = -GMm*avg_inv_r; Tavg = Etot - Vavg
check("A4", "Kepler orbit: <V> = 2E, <T> = -E, 2<T> + <V> = 0",
      is_zero(Vavg - 2*Etot) and is_zero(2*Tavg + Vavg), f"<1/r>={sp.simplify(avg_inv_r)}")

# A5 unbound counterexample: free particle r = r0 + v t has (1/2) I'' = 2T != 0 with V = 0 (no time average kills it)
v0, x0 = sp.symbols('v0 x0', positive=True)
xf = x0 + v0*t
check("A5", "unbound free motion: (1/2) I'' = m v^2 = 2T > 0, the virial average fails",
      is_zero(sp.Rational(1, 2)*sp.diff(m1*xf**2, t, 2) - m1*v0**2))

# A6 2<T> = k<V> on a circular orbit for V = C r^k (attractive: C k > 0)
C, r = sp.symbols('C r', positive=True)
Vr = C*r**k
mv2 = r*sp.diff(Vr, r)                      # m v^2 = r dV/dr (centripetal balance)
check("A6", "circular orbit in V = C r^k: 2T = k V", is_zero(mv2 - k*Vr))

# A7 trace of the electromagnetic energy-momentum tensor vanishes (degree-2 field energy; the k = 2 branch)
eps0, mu0 = sp.symbols('epsilon0 mu0', positive=True)
Ev = sp.Matrix(sp.symbols('Ex Ey Ez', real=True)); Bv = sp.Matrix(sp.symbols('Bx By Bz', real=True))
u = (eps0*Ev.dot(Ev) + Bv.dot(Bv)/mu0)/2
sigma = sp.Matrix(3, 3, lambda i, j: eps0*(Ev[i]*Ev[j] - sp.KroneckerDelta(i, j)*Ev.dot(Ev)/2)
                  + (Bv[i]*Bv[j] - sp.KroneckerDelta(i, j)*Bv.dot(Bv)/2)/mu0)
trace = -u + sum(-sigma[i, i] for i in range(3))     # T^mu_mu, signature (-+++), T^ij = -sigma_ij
check("A7", "EM stress-energy tensor is traceless (radiation p = u/3)", is_zero(trace))

# A8 particle trace: E - p.v = m c^2 / gamma
mm, cc_, vv = sp.symbols('m c v', positive=True)
gam = 1/sp.sqrt(1 - vv**2/cc_**2)
check("A8", "E - p.v = m c^2/gamma", is_zero(gam*mm*cc_**2 - gam*mm*vv**2 - mm*cc_**2/gam))

# A9 Landau-Lifshitz virial E_tot = <sum m c^2/gamma>: non-relativistic limit gives E - Mc^2 = -<T>
ser = sp.series(mm*cc_**2*sp.sqrt(1 - vv**2/cc_**2), vv, 0, 4).removeO()
check("A9", "m c^2/gamma = m c^2 - m v^2/2 + O(v^4): LL virial reduces to E = -<T>",
      is_zero(ser - (mm*cc_**2 - mm*vv**2/2)))

# A10 p.v in terms of kinetic energy K: p.v = K(K + 2mc^2)/(K + mc^2)  (-> 2K slow, -> K fast)
K = sp.symbols('K', positive=True)
gK = 1 + K/(mm*cc_**2); vK2 = cc_**2*(1 - 1/gK**2)
pv = gK*mm*vK2
check("A10", "p.v = K(K+2mc^2)/(K+mc^2); limits 2K (slow) and K (fast)",
      is_zero(pv - K*(K + 2*mm*cc_**2)/(K + mm*cc_**2))
      and sp.limit(pv/K, K, 0) == 2 and sp.limit(pv/K, K, sp.oo) == 1)

# A11 Smarr as Euler (G = c = 1): M(A,J) homogeneous of degree 1/2 -> M = 2 T S + 2 Omega J
A_, J = sp.symbols('A J', positive=True)
M_AJ = sp.sqrt(A_/(16*sp.pi) + 4*sp.pi*J**2/A_)
lam = sp.symbols('lambda', positive=True)
hom = sp.simplify(M_AJ.subs({A_: lam*A_, J: lam*J}, simultaneous=True)/M_AJ)
MA = sp.diff(M_AJ, A_); MJ = sp.diff(M_AJ, J)
Tbh = 4*MA                # dM = T dS + Omega dJ with S = A/4  ->  T = 4 dM/dA
Sbh = A_/4
check("A11", "Kerr M(A,J) degree 1/2; Euler gives M = 2TS + 2 Omega J",
      sp.simplify(hom - sp.sqrt(lam)) == 0 and is_zero(sp.simplify(M_AJ - 2*Tbh*Sbh - 2*MJ*J)))

# A12 Komar mass of Schwarzschild: kappa A/(4 pi) = M = 2 T S (G = c = hbar = k = 1)
M = sp.symbols('M', positive=True)
kap = 1/(4*M); Ah = 16*sp.pi*M**2
check("A12", "Komar: kappa A/(4 pi) = M = 2 T_H S", is_zero(kap*Ah/(4*sp.pi) - M) and
      is_zero(2*(kap/(2*sp.pi))*(Ah/4) - M))

# ---------------------------------------------------------------- B. the record and its split
# B1 Landauer price of compressing a one-molecule gas to half volume: W = kT ln 2
kB, Tt, Vol = sp.symbols('k_B T V', positive=True)
Vs = sp.symbols('Vs', positive=True)
W_comp = -sp.integrate(kB*Tt/Vs, (Vs, Vol, Vol/2))
check("B1", "isothermal halving of a one-molecule gas costs kT ln 2 (all of it leaves as heat)",
      is_zero(sp.simplify(W_comp - kB*Tt*sp.log(2))))

# B2 sudden connection of a degree-n store to a fixed generalized force: held 1/n, spent (n-1)/n
n, cst, F = sp.symbols('n c F', positive=True)
qf = sp.symbols('q_f', positive=True)
U = cst*qf**n
W_src = qf*sp.diff(cst*qf**n, qf)             # F = dU/dq at the final equilibrium, work = F q_f
check("B2", "fixed-force (sudden) charging of U = c q^n: W = nU, held/W = 1/n, spent = (n-1)U",
      is_zero(W_src - n*U) and sp.simplify((U/W_src).subs(n, 2)) == sp.Rational(1, 2))

# B3 RC step charging: dissipated = C V^2/2 for any R
R_, Cc, V0 = sp.symbols('R C V_0', positive=True)
Icur = V0/R_*sp.exp(-t/(R_*Cc))
diss = sp.integrate(Icur**2*R_, (t, 0, sp.oo))
check("B3", "RC step: heat = C V^2/2 = stored, independent of R", is_zero(diss - Cc*V0**2/2))

# B4 maximum power transfer: matched load takes half, source resistance burns half
Rs, RL = sp.symbols('R_s R_L', positive=True)
PL = V0**2*RL/(Rs + RL)**2
crit = sp.solve(sp.diff(PL, RL), RL)
check("B4", "max power at R_L = R_s, efficiency 1/2", crit == [Rs] and
      sp.simplify((RL/(Rs + RL)).subs(RL, Rs)) == sp.Rational(1, 2))

# B5 binding from rest under degree k: radiated Q = -(1+k/2)<V>, retained K = (k/2)<V>; equal at k = -1
Vb = sp.symbols('Vbar', negative=True)
Kret = k/2*Vb; Qrad = -(Kret + Vb)
check("B5", "binding from rest: Q_radiated = K_retained only at k = -1",
      sp.solve(sp.Eq(Qrad, Kret), k) == [-1])

# B6 one harmonic mode: <K> = <V> = E/2; instantaneous K = V only at 45 deg of phase
w, Aamp = sp.symbols('omega A', positive=True)
q = Aamp*sp.cos(w*t); pmom = -mm*w*Aamp*sp.sin(w*t)
Kt = pmom**2/(2*mm); Vt = mm*w**2*q**2/2
Tper = 2*sp.pi/w
Kav = sp.integrate(Kt, (t, 0, Tper))/Tper; Vav = sp.integrate(Vt, (t, 0, Tper))/Tper
check("B6", "harmonic mode: <K> = <V> = E/2; K(t) = V(t) only at wt = pi/4 (mod pi/2)",
      is_zero(Kav - Vav) and is_zero(Kav - mm*w**2*Aamp**2/4) and
      is_zero((Kt - Vt).subs(t, sp.pi/(4*w))))

# B7 travelling wave: u_E = u_B at every point and time; standing wave: only on average
xx = sp.symbols('x', real=True); E0, kw, cl = sp.symbols('E0 k c', positive=True)
f = sp.Function('f')
Etr = E0*f(xx - cl*t); Btr = Etr/cl
uE = eps0*Etr**2/2; uB = Btr**2/(2*mu0)
uEB_tr = sp.simplify((uE - uB).subs(mu0, 1/(eps0*cl**2)))
Est = E0*sp.cos(kw*xx)*sp.cos(kw*cl*t); Bst = E0/cl*sp.sin(kw*xx)*sp.sin(kw*cl*t)
duSt = (eps0*Est**2/2 - Bst**2/(2*mu0)).subs(mu0, 1/(eps0*cl**2))
avg_st = sp.simplify(sp.integrate(sp.integrate(duSt, (xx, 0, 2*sp.pi/kw)), (t, 0, 2*sp.pi/(kw*cl))))
check("B7", "free travelling wave u_E = u_B pointwise; standing wave equal only on average",
      uEB_tr == 0 and avg_st == 0 and sp.simplify(duSt.subs({xx: 0, t: 0})) != 0)

# B8 line at surge-impedance load: V = Z0 I gives C V^2/2 = L I^2/2 (charging MVAr = I^2 X)
L_, C_, Ii = sp.symbols('L C I', positive=True)
Vsil = sp.sqrt(L_/C_)*Ii
check("B8", "SIL: (1/2) C V^2 = (1/2) L I^2 at V = sqrt(L/C) I", is_zero(C_*Vsil**2/2 - L_*Ii**2/2))

# B9 pure dephasing exchanges no energy: [H_S, H_int] = 0 for H_S = w sz, H_int = sz (x) B
sz = sp.Matrix([[1, 0], [0, -1]]); Bop = sp.Matrix(sp.MatrixSymbol('b', 2, 2))
HS = sp.kronecker_product(w*sz, sp.eye(2)); Hint = sp.kronecker_product(sz, Bop)
check("B9", "pure dephasing: [H_S, H_int] = 0, the system's energy is conserved while coherence is lost",
      sp.simplify(HS*Hint - Hint*HS) == sp.zeros(4, 4))

# ---------------------------------------------------------------- C. distance and temperature
G, hb, cL, mbar, alpha, Rr, Mm, Npart = sp.symbols('G hbar c mbar alpha R M N', positive=True)
# C1 virial temperature: 2(3/2 N k T) = alpha G M^2/R ;  kT/(mbar c^2) = (alpha/3) Xi
kT = sp.symbols('kT', positive=True)
kTsol = sp.solve(sp.Eq(3*Npart*kT, alpha*G*Mm**2/Rr), kT)[0].subs(Npart, Mm/mbar)
Xi = G*Mm/(Rr*cL**2)
check("C1", "k T_vir = alpha G M mbar/(3R); k T_vir/(mbar c^2) = (alpha/3) G M/(R c^2)",
      is_zero(kTsol/(mbar*cL**2) - alpha/3*Xi))

# C2 contraction with adiabatic index gamma: E = (3g-4)/(3(g-1)) W; kept 1/(3(g-1)) of released |dW|
g = sp.symbols('gamma', positive=True)
Wg = sp.symbols('W', negative=True)
Uint = -Wg/(3*(g - 1)); Eg = Uint + Wg
check("C2", "E = (3g-4)/(3(g-1)) W; g = 5/3 -> half radiated, g = 4/3 -> E = 0 (nothing radiated)",
      is_zero(Eg - (3*g - 4)/(3*(g - 1))*Wg) and
      sp.simplify((Eg/Wg).subs(g, sp.Rational(5, 3))) == sp.Rational(1, 2) and
      sp.simplify((Eg/Wg).subs(g, sp.Rational(4, 3))) == 0)

# C3 gas fraction beta (Eddington standard model): U = -W(1 - beta/2), E = beta W/2
be, Pint = sp.symbols('beta Pint', positive=True)
Ugas = sp.Rational(3, 2)*be*Pint; Urad = 3*(1 - be)*Pint
Wv = -3*Pint                                     # virial: 3 int P dV + W = 0
check("C3", "radiation pressure: E = beta W/2; radiated share of released |dW| = beta/2",
      is_zero(Ugas + Urad + Wv - be*Wv/2))

# C4 relativistic thermal gas: <p.v> = 3kT at every temperature; K/|W| moves 1/2 -> 1 (numeric, mpmath)
def ratio_K_over_W(z):   # z = m c^2 / kT ; <K>/(3kT) with <E> = mc^2 [K1/K2 + 3/z]
    z = mp.mpf(z)
    Ebar_over_mc2 = mp.besselk(1, z)/mp.besselk(2, z) + 3/z
    return (Ebar_over_mc2 - 1)*z/3
def pv_over_kT(z):  # 3D Maxwell-Juttner: <p.v>/kT by direct integral
    z = mp.mpf(z)
    Ep = lambda p: mp.sqrt(p**2 + 1)                    # units m = c = 1, momentum in mc
    num = mp.quad(lambda p: p**2*(p**2/Ep(p))*mp.e**(-z*(Ep(p) - 1)), [0, mp.inf])
    den = mp.quad(lambda p: p**2*mp.e**(-z*(Ep(p) - 1)), [0, mp.inf])
    return num/den*z
vals = {zz: (float(pv_over_kT(zz)), float(ratio_K_over_W(zz))) for zz in [1e3, 10, 1, 0.1, 1e-3]}
check("C4", "generalized equipartition <p.v> = 3kT for all z = mc^2/kT; virial 3NkT = |W| always; "
      "K/|W| = 0.5 (cold) -> 1.0 (ultra-relativistic)",
      all(abs(v[0] - 3) < 1e-6 for v in vals.values()) and abs(vals[1e3][1] - 0.5) < 2e-3
      and abs(vals[1e-3][1] - 1.0) < 2e-3, vals)

# C5 photon diffusion: N = tau^2 steps of length R/tau -> t_diff = tau R/c = tau t_lc
tau = sp.symbols('tau', positive=True)
check("C5", "t_diff / t_light-crossing = tau", is_zero((tau**2*(Rr/tau)/cL)/(Rr/cL) - tau))

# C6 virial temperature + radiative diffusion: L ~ R T^4/(kappa rho) independent of R, ~ M^3
kap_o, mu_, mp_, kb_, aR = sp.symbols('kappa mu m_p k_b a_rad', positive=True)
Tc = mu_*mp_*G*Mm/(kb_*Rr); rho = 3*Mm/(4*sp.pi*Rr**3)
Lum = sp.Rational(16, 3)*sp.pi*aR*cL*Rr*Tc**4/(kap_o*rho)
check("C6", "L ~ (16 pi/3) a c R T^4/(kappa rho) with virial T: R drops out, L ~ M^3/kappa",
      sp.simplify(sp.diff(sp.log(Lum), Rr)) == 0 and sp.simplify(Mm*sp.diff(sp.log(Lum), Mm)) == 3)

# C7 surface vs volume: heating q (4/3 pi R^3) = sigma T^4 (4 pi R^2) -> T ~ R^(1/4)
qh, sig = sp.symbols('q sigma', positive=True)
Tbal = sp.solve(sp.Eq(qh*4*sp.pi*Rr**3/3, sig*Tt**4*4*sp.pi*Rr**2), Tt)[0]
check("C7", "volumetric heating vs surface radiation: T ~ R^(1/4)", sp.simplify(Rr*sp.diff(sp.log(Tbal), Rr)) == sp.Rational(1, 4))

# C8 dynamical vs light-crossing time: t_dyn/t_lc = Xi^(-1/2); sqrt(2) at the horizon (Xi = 1/2)
tdyn = sp.sqrt(Rr**3/(G*Mm)); tlc = Rr/cL
check("C8", "t_dyn/t_lc = (GM/Rc^2)^(-1/2); = sqrt 2 at GM/(Rc^2) = 1/2",
      is_zero(sp.simplify(tdyn/tlc - Xi**sp.Rational(-1, 2))))

# C9 flat FRW shell: (1/2) H^2 R^2 = G M/R identically (E = 0, K = |V|); GM/(Rc^2) = 1/2 at R = c/H
rho_, H = sp.symbols('rho H', positive=True)
Msh = sp.Rational(4, 3)*sp.pi*rho_*Rr**3
H2 = 8*sp.pi*G*rho_/3
check("C9", "flat FRW: K = |V| per unit mass (E = 0); GM/(Rc^2) = 1/2 at R = c/H",
      is_zero(H2*Rr**2/2 - G*Msh/Rr) and
      is_zero(sp.simplify((G*Msh/(Rr*cL**2)).subs(rho_, 3*H**2/(8*sp.pi*G)).subs(Rr, cL/H)) - sp.Rational(1, 2)))

# C10 Bekenstein bound saturated by a black hole: S_BH = E/T_R with k T_R = hbar c/(2 pi r_s); T_R = 2 T_H
kb = sp.symbols('k_B', positive=True)
rs = 2*G*Mm/cL**2
SBH = 4*sp.pi*G*kb*Mm**2/(hb*cL)
TR = hb*cL/(2*sp.pi*kb*rs); TH = hb*cL**3/(8*sp.pi*G*kb*Mm)
check("C10", "black hole: T_R S = M c^2 (Bekenstein saturated), T_R(r_s) = 2 T_H, T_H S = M c^2/2",
      is_zero(TR*SBH - Mm*cL**2) and is_zero(TR - 2*TH) and is_zero(TH*SBH - Mm*cL**2/2))

# C11 Hubble sphere: E = c^4 R/(2G), S = pi c^3 R^2 k/(G hbar): T_GH S = E; dE/dS = T_GH/2
EH = cL**4*Rr/(2*G); SH = sp.pi*cL**3*Rr**2*kb/(G*hb); TGH = hb*cL/(2*sp.pi*kb*Rr)
check("C11", "Hubble sphere: T_GH S = E; first-law dE/dS = T_GH/2",
      is_zero(TGH*SH - EH) and is_zero(sp.diff(EH, Rr)/sp.diff(SH, Rr) - TGH/2))

# C12 Euler, degree 2: S = s E^2 -> (dE/dS) S = E/2 ; the increment temperature is half the capacity temperature E/S
s_, Ee = sp.symbols('s E', positive=True)
Sfun = s_*Ee**2
check("C12", "S ~ E^2: (dE/dS) S = E/2, i.e. T_first-law = (E/S)/2", is_zero(Sfun/sp.diff(Sfun, Ee) - Ee/2))

# C13 Cai-Kim: T_GH dS = energy flux A (rho+p) H R dt across the fixed apparent horizon (hbar=c=k=1)
Hf = sp.Function('H')(t)
RA = 1/Hf
S_ah = sp.pi*RA**2/G; T_gh = 1/(2*sp.pi*RA)
rho_t = 3*Hf**2/(8*sp.pi*G); rp = -sp.diff(Hf, t)/(4*sp.pi*G)     # rho + p from Raychaudhuri
flux = 4*sp.pi*RA**2*rp*Hf*RA
check("C13", "Cai-Kim: T_GH dS/dt = A (rho+p) H R_A (the heat crossing equals T_GH dS)",
      is_zero(sp.simplify(T_gh*sp.diff(S_ah, t) - flux)))

# C14 Akbar-Cai unified first law: dE_MS = T dS + W dV with T = kappa/2pi, kappa = -(1/R)(1 - R'/(2HR)), W = (rho-p)/2
E_ms = RA/(2*G); V_ah = sp.Rational(4, 3)*sp.pi*RA**3
p_t = rp - rho_t
Rdot = sp.diff(RA, t)
kappa_KH = -(1/RA)*(1 - Rdot/(2*Hf*RA))
lhs = sp.diff(E_ms, t)
rhs = kappa_KH/(2*sp.pi)*sp.diff(S_ah, t) + (rho_t - p_t)/2*sp.diff(V_ah, t)
check("C14", "Akbar-Cai: dE_MS = T dS + W dV (Kodama-Hayward kappa, work density (rho-p)/2)",
      is_zero(sp.simplify(lhs - rhs)))

# C15 Kodama-Hayward temperature / T_GH = 1 - eps/2, eps = -H'/H^2 = (3/2)(1+w)
wq, Om = sp.symbols('w Omega_m', real=True)
eps = sp.Rational(3, 2)*(1 + wq)
ratio_KH = 1 - eps/2
vals15 = {"deSitter": ratio_KH.subs(wq, -1), "matter": ratio_KH.subs(wq, 0), "radiation": ratio_KH.subs(wq, sp.Rational(1, 3)),
          "LCDM_today(Om=0.3153)": float(1 - sp.Rational(3, 4)*sp.Rational(3153, 10000))}
check("C15", "T_KH/T_GH = 1 - (3/4)(1+w): de Sitter 1, matter 1/4, radiation 0; LCDM today 0.7635",
      vals15["deSitter"] == 1 and vals15["matter"] == sp.Rational(1, 4) and vals15["radiation"] == 0
      and abs(vals15["LCDM_today(Om=0.3153)"] - 0.763525) < 1e-6, vals15)

# C16 local virial temperature over the horizon temperature at the same radius: R cancels
Tvir_over_TR = (alpha*G*Mm*mbar/(3*Rr))/(hb*cL/(2*sp.pi*Rr))
check("C16", "k T_vir / k T_R = (2 pi alpha/3) G M mbar/(hbar c): independent of R",
      sp.diff(sp.simplify(Tvir_over_TR), Rr) == 0)

# C17 heat carried outward gains entropy; Bekenstein forces T >= T_R (heat) or (4/3) T_R (black-body gas)
Q_, T1, T2 = sp.symbols('Q T1 T2', positive=True)
dS = Q_/T2 - Q_/T1
check("C17", "dS = Q(1/T2 - 1/T1) > 0 for T2 < T1; S = Q/T <= Q/T_R -> T >= T_R; S = 4E/3T -> T >= 4T_R/3",
      sp.simplify(dS.subs(T1, 2*T2)) == Q_/(2*T2) and
      sp.solve(sp.Eq(4*Q_/(3*Tt), Q_/sp.Symbol('T_R', positive=True)), Tt) == [sp.Rational(4, 3)*sp.Symbol('T_R', positive=True)])

# C18 Newtonian compactness 1/2 <=> escape speed c
check("C18", "v_esc = c  <=>  G M/(R c^2) = 1/2", sp.solve(sp.Eq(sp.sqrt(2*G*Mm/Rr), cL), Rr) == [2*G*Mm/cL**2])

# C19 black hole and Hubble sphere share E(R) = c^4 R/2G and S(R) = pi c^3 R^2 k/(G hbar); only kappa R/c^2 differs (1/2 vs 1)
EBH_r = (Mm*cL**2).subs(Mm, Rr*cL**2/(2*G)); SBH_r = SBH.subs(Mm, Rr*cL**2/(2*G))
kapR_BH = sp.simplify((cL**4/(4*G*Mm))*rs/cL**2); kapR_dS = 1
check("C19", "BH (R = r_s) and Hubble sphere have identical E(R), S(R) and capacity temperature E/S = T_R; "
      "surface gravity kappa R/c^2 = 1/2 (BH) vs 1 (de Sitter) is the only difference",
      is_zero(EBH_r - EH) and is_zero(SBH_r - SH) and kapR_BH == sp.Rational(1, 2))

# ---------------------------------------------------------------- D. Koide
s1, s2, s3 = sp.symbols('s1 s2 s3', positive=True)
svec = sp.Matrix([s1, s2, s3])
Cl = sp.sqrt(sp.Rational(2, 3))*sp.Matrix([[1, -sp.Rational(1, 2), -sp.Rational(1, 2)],
                                           [0, sp.sqrt(3)/2, -sp.sqrt(3)/2],
                                           [1/sp.sqrt(2), 1/sp.sqrt(2), 1/sp.sqrt(2)]])
vv_ = Cl*svec
Qk = svec.dot(svec)/(s1 + s2 + s3)**2
check("D1", "power-invariant Clarke matrix orthogonal; Q = (v0^2 + |v_ab|^2)/(3 v0^2)",
      sp.simplify(Cl*Cl.T - sp.eye(3)) == sp.zeros(3, 3) and
      is_zero(sp.simplify(Qk - (vv_[2]**2 + vv_[0]**2 + vv_[1]**2)/(3*vv_[2]**2))))

x_, y_, dl, th = sp.symbols('x y delta theta', real=True)
sB = sp.Matrix([x_ + y_*sp.cos(dl + 2*sp.pi*j/3) for j in range(3)])
vB = Cl*sB
check("D2", "Brannen form: v0^2 = 3x^2, |v_ab|^2 = 3y^2/2", is_zero(sp.simplify(vB[2]**2 - 3*x_**2)) and
      is_zero(sp.simplify(sp.expand_trig(vB[0]**2 + vB[1]**2) - 3*y_**2/2)))

# D3 the natural three-phase oscillator: rotation about (1,1,1) is generated by H = w(al^2+be^2)/2;
#     {v0, H} = 0 (v0 a conserved Casimir), the conjugate of the action I = (al^2+be^2)/2 is the phase delta, not x
al, bt, v0s = sp.symbols('alpha beta v0', real=True)
def PB(f_, g_):   # (alpha, beta) canonical
    return sp.diff(f_, al)*sp.diff(g_, bt) - sp.diff(f_, bt)*sp.diff(g_, al)
Hrot = w*(al**2 + bt**2)/2; Iact = (al**2 + bt**2)/2; phase = sp.atan2(bt, al)
Q_ab = sp.simplify(Qk.subs({s1: sB[0], s2: sB[1], s3: sB[2]}))
check("D3", "Z3/phase rotation: {v0,H} = 0, {I, phase} = 1 (action-angle pair), Q independent of delta (Q cannot be set by this oscillator)",
      PB(v0s, Hrot) == 0 and sp.simplify(PB(Iact, phase)) == 1 and sp.simplify(sp.expand_trig(sp.diff(Q_ab, dl))) == 0)

# D4 a mode that rotates s toward and away from (1,1,1): <v0^2> = <|v_ab|^2> (k=2 virial) but the
#    per-flavour RMS triple does NOT give Q = 2/3, and <s_k> = 0
Rm = sp.symbols('R_m', positive=True)
uk = [sp.sqrt(sp.Rational(2, 3))*sp.cos(dl + 2*sp.pi*j/3) for j in range(3)]
s_t = [Rm*sp.cos(th)/sp.sqrt(3) + Rm*sp.sin(th)*uk[j] for j in range(3)]
avg = lambda ex: sp.simplify(sp.integrate(ex, (th, 0, 2*sp.pi))/(2*sp.pi))
v0sq = avg((Rm*sp.cos(th))**2); vabsq = avg((Rm*sp.sin(th))**2)
ms = [avg(s_t[j]**2) for j in range(3)]
Qrms = sp.lambdify(dl, sum(ms)/(sum(sp.sqrt(mj) for mj in ms))**2)
grid = [Qrms(d*math.pi/180) for d in range(0, 121)]
check("D4", "theta-rotating mode: <v0^2> = <|v_ab|^2>, <s_k> = 0; per-flavour RMS Q ranges far from 2/3",
      is_zero(v0sq - vabsq) and all(avg(sj) == 0 for sj in s_t) and max(abs(gq - 2/3) for gq in grid) > 0.2,
      f"Q_rms in [{min(grid):.4f}, {max(grid):.4f}] over delta")

# D5 Q = (1 + CV^2)/3 for three values (population CV)
mu_s = (s1 + s2 + s3)/3; var_s = ((s1 - mu_s)**2 + (s2 - mu_s)**2 + (s3 - mu_s)**2)/3
check("D5", "Q = (1 + CV^2)/3, CV = population std/mean of the three square roots", is_zero(sp.simplify(Qk - (1 + var_s/mu_s**2)/3)))

# D6 chi-square with nu quadratic degrees of freedom: CV^2 = 2/nu -> Q_nu = (nu+2)/(3 nu)
nu = sp.symbols('nu', positive=True)
from sympy.stats import Gamma as GammaRV, E as Ex, variance
X = GammaRV('X', nu/2, 2)
cv2 = sp.simplify(variance(X)/Ex(X)**2)
Qnu = (1 + cv2)/3
check("D6", "chi^2_nu: CV^2 = 2/nu; Q = 2/3 at nu = 2 (one canonical pair), 1 at nu = 1, 5/9 at nu = 3, 1/3 as nu -> oo",
      is_zero(cv2 - 2/nu) and Qnu.subs(nu, 2) == sp.Rational(2, 3) and Qnu.subs(nu, 1) == 1 and
      Qnu.subs(nu, 3) == sp.Rational(5, 9) and sp.limit(Qnu, nu, sp.oo) == sp.Rational(1, 3))

# D7 one thermal mode: E = (q^2 + p^2)/2 with q, p ~ N(0, kT) is exponential with CV = 1
from sympy.stats import Normal
qn = Normal('qn', 0, 1); pn = Normal('pn', 0, 1)
Emode = (qn**2 + pn**2)/2
cv_mode = sp.simplify(variance(Emode)/Ex(Emode)**2)
check("D7", "energy of one thermal harmonic mode (q,p Gaussian): CV^2 = 1 (exponential law)", cv_mode == 1)

# D8 circulant square-root mass matrix aI + bC + conj(b) C^2: eigenvalues a + 2|b| cos(delta + 2 pi k/3);
#    Koide <=> ||diagonal||_F = ||off-diagonal||_F
aa, bb = sp.symbols('a b', positive=True)
Cs = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
bc = bb*sp.exp(sp.I*dl)
Mc = aa*sp.eye(3) + bc*Cs + sp.conjugate(bc)*Cs**2
om = sp.exp(2*sp.pi*sp.I/3)
eig_ok = all(sp.simplify(sp.expand_complex(aa + bc*om**j + sp.conjugate(bc)*om**(-j) - (aa + 2*bb*sp.cos(dl + 2*sp.pi*j/3)))) == 0
             for j in range(3))
vec_ok = all(sp.simplify(sp.expand_complex((Mc*sp.Matrix([1, om**j, om**(2*j)]) -
             (aa + bc*om**j + sp.conjugate(bc)*om**(-j))*sp.Matrix([1, om**j, om**(2*j)])))) == sp.zeros(3, 1) for j in range(3))
Dn = 3*aa**2; On = sp.simplify(sum(sp.Abs(Mc[i, j])**2 for i in range(3) for j in range(3) if i != j))
check("D8", "circulant: eigenvalues a + 2|b|cos(delta + 2pi k/3); x = a, y = 2|b|; Koide <=> 3a^2 = 6|b|^2 (equal Frobenius norms)",
      eig_ok and vec_ok and is_zero(On - 6*bb**2) and
      sp.solve(sp.Eq(aa**2, (2*bb)**2/2), aa) == [sp.sqrt(2)*bb] and sp.solve(sp.Eq(Dn, On), aa) == [sp.sqrt(2)*bb])

# D9 undamped sudden step of an oscillator: q = q_f(1 - cos wt) -> x = q_f, y = q_f -> Q = 1/2 (not Koide)
Qxy = (1 + y_**2/(2*x_**2))/3
check("D9", "held/spent step (y = x) gives Q = 1/2; Koide needs y = sqrt(2) x",
      sp.simplify(Qxy.subs(y_, x_)) == sp.Rational(1, 2) and sp.simplify(Qxy.subs(y_, sp.sqrt(2)*x_)) == sp.Rational(2, 3))

# D10 leptons (PDG 2024 values as tabulated in the book): CV^2 = 3Q - 1
me, mmu, mtau = mp.mpf('0.51099895000'), mp.mpf('105.6583755'), mp.mpf('1776.93')
sq = [mp.sqrt(me), mp.sqrt(mmu), mp.sqrt(mtau)]
Qlep = (me + mmu + mtau)/sum(sq)**2
mu_l = sum(sq)/3; cv2_l = sum((sv - mu_l)**2 for sv in sq)/3/mu_l**2
check("D10", "leptons: CV^2 of sqrt masses = 3Q - 1 = 0.999993", abs(cv2_l - (3*Qlep - 1)) < 1e-12 and abs(cv2_l - 1) < 1e-4,
      f"Q={mp.nstr(Qlep, 9)}, CV^2={mp.nstr(cv2_l, 8)}")

# D11 how often three independent single-mode (exponential) draws land as close to CV^2 = 1 as the leptons do
import numpy as np
rng = np.random.default_rng(20261118)
win = 3*5.08e-6                       # |CV^2 - 1| = 3|Q - 2/3|; window = 1 sigma of Q from the book (5.08e-6)
hits = 0; ntot = 0
for _ in range(20):
    sdraw = rng.exponential(size=(1_000_000, 3))
    Qd = (sdraw**2).sum(1)/sdraw.sum(1)**2
    hits += int((np.abs(3*Qd - 3*(2/3)) < win).sum()); ntot += sdraw.shape[0]
    # density near 2/3 for a stabler estimate
dens_hits = 0
rng2 = np.random.default_rng(7)
for _ in range(20):
    sdraw = rng2.exponential(size=(1_000_000, 3))
    Qd = (sdraw**2).sum(1)/sdraw.sum(1)**2
    dens_hits += int((np.abs(Qd - 2/3) < 0.005).sum())
p_est = dens_hits/2e7/(2*0.005)*(2*5.08e-6)
check("D11", "three independent exponential draws: P(|Q - 2/3| < 5.08e-6) ~ 1e-5 (chance does not explain the precision)",
      p_est < 1e-4, f"direct hits {hits}/{ntot}; density estimate p = {p_est:.2e}")

# ---------------------------------------------------------------- output
npass = sum(r_['pass'] for r_ in RES); nfail = len(RES) - npass
json.dump({"checks": RES, "passed": npass, "failed": nfail}, open("spider_longform_checks_output.json", "w"), indent=1)
for r_ in RES:
    print(("PASS " if r_['pass'] else "FAIL ") + r_['id'] + "  " + (r_['detail'][:110] if r_['detail'] else ""))
print(f"passed {npass}, failed {nfail}")
