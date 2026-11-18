import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""Spider S4 checks: why only timelike degrees of freedom write records, and what that gives for mu and Sigma.
sympy + mpmath + scipy, one core, seconds. Writes spider_s4_checks_output.json.
Units c = 1 unless stated. Metric signature (-,+,+,+).
Conformal Newtonian gauge as in the book (p2_03:762; Ma & Bertschinger 1995):
  ds^2 = a^2 [ -(1+2 Psi) deta^2 + (1-2 Phi) dx^2 ],  Psi = time-time potential (matter), Phi = spatial potential.
mu, Sigma as in the book (p2_03:486-487; p2_17:54-55):
  k^2 Psi = -4 pi G a^2 mu rho Delta,   k^2 (Phi+Psi) = -8 pi G a^2 Sigma rho Delta.
"""
import json, sympy as sp, mpmath as mp
mp.mp.dps = 30
import numpy as np
from scipy.integrate import solve_ivp

R = {}
def rec(cid, ok, note):
    R[cid] = {"pass": bool(ok), "note": note}
    print(cid, "PASS" if ok else "FAIL", "|", note)

# ---------------------------------------------------------------- P: proper time, trace, conformal invariance
t, x, y, z = sp.symbols('t x y z', real=True)
v = sp.symbols('v', nonnegative=True)
eps = sp.symbols('epsilon')
# P1 proper time along a curve dx/dt = v in flat space; zero at v = 1; null curve of weak-field metric has dtau = 0
dtau_dt = sp.sqrt(1 - v**2)
Ps, Ph = sp.symbols('Psi Phi', real=True)
vnull = sp.sqrt((1 + 2*Ps)/(1 - 2*Ph))            # coordinate speed of light in weak-field metric
ds2_null = sp.simplify(-(1 + 2*Ps) + (1 - 2*Ph)*vnull**2)
rec("P1", dtau_dt.subs(v, 1) == 0 and ds2_null == 0,
    "dtau/dt = sqrt(1-v^2) -> 0 at v=c; along the null direction of -(1+2Psi)dt^2+(1-2Phi)dx^2, ds^2 = 0 identically")

# P2 trace of a point particle's stress tensor = -m dtau/dt (per unit delta^3); photon (m->0 at fixed E) gives 0
m, E = sp.symbols('m E', positive=True)
vx, vy, vz = sp.symbols('v_x v_y v_z', real=True)
vv = sp.Matrix([vx, vy, vz]); v2 = (vv.T*vv)[0]
gam = 1/sp.sqrt(1 - v2)
u = sp.Matrix([gam, gam*vx, gam*vy, gam*vz])        # u^mu = dx^mu/dtau
eta = sp.diag(-1, 1, 1, 1)
T = (m/gam)*u*u.T                                     # T^{mu nu} density factor (m/gamma) u u  (x delta^3)
trace = sp.simplify(sum(eta[i, i]*T[i, i] for i in range(4)))
photon_lim = sp.limit((m**2/E), m, 0)                # m/gamma = m^2/E at fixed energy E
rec("P2", sp.simplify(trace + m*sp.sqrt(1 - v2)) == 0 and photon_lim == 0,
    "T^mu_mu = -m/gamma = -m dtau/dt for a point mass; with m/gamma = m^2/E -> 0 for a massless particle")

# P3 Maxwell stress tensor is traceless (in 3+1 split): -u + sum_i T_ii = 0 with T_ij Maxwell stress
Ex, Ey, Ez, Bx, By, Bz = sp.symbols('E_x E_y E_z B_x B_y B_z', real=True)
Ev = sp.Matrix([Ex, Ey, Ez]); Bv = sp.Matrix([Bx, By, Bz])
uem = sp.Rational(1, 2)*((Ev.T*Ev)[0] + (Bv.T*Bv)[0])
sig = sp.zeros(3)
for i in range(3):
    for j in range(3):
        sig[i, j] = Ev[i]*Ev[j] + Bv[i]*Bv[j] - sp.KroneckerDelta(i, j)*uem   # Maxwell stress (Heaviside-Lorentz)
# spatial T^{ij} = -sigma_ij; T^mu_mu = -T^{00} + T^{ii} = -u - tr(sigma)
rec("P3", sp.simplify(-uem - sig.trace()) == 0, "electromagnetic T^mu_mu = -u - tr(sigma_Maxwell) = 0 (radiation has no trace)")

# P4 Weyl invariance of the Maxwell action density sqrt(-g) g^{ma} g^{nb} F F only in D = 4
Om, D = sp.symbols('Omega D', positive=True)
scale = Om**D * Om**(-4)          # sqrt(-g) -> Omega^D sqrt(-g); two inverse metrics -> Omega^-4
rec("P4", sp.simplify(scale.subs(D, 4)) == 1 and sp.simplify(scale.subs(D, 5)) != 1,
    "under g -> Omega^2 g the Maxwell density scales as Omega^(D-4): invariant in four dimensions only")

# P5 null geodesics are conformally invariant: for g = e^{2w} eta, Gamma^mu_ab k^a k^b is proportional to k^mu when k is null
X = [t, x, y, z]
w = sp.Function('w')(*X)
g = sp.exp(2*w)*eta
ginv = g.inv()
def christoffel(g, ginv, X):
    n = len(X)
    G = [[[0]*n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                G[a][b][c] = sp.simplify(sum(ginv[a, d]*(sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
                                             for d in range(n))/2)
    return G
Gm = christoffel(g, ginv, X)
k1, k2, k3 = sp.symbols('k1 k2 k3', real=True)
k0 = sp.sqrt(k1**2 + k2**2 + k3**2)                   # null: -k0^2 + |k|^2 = 0
k = [k0, k1, k2, k3]
acc = [sp.simplify(sum(Gm[mu][a][b]*k[a]*k[b] for a in range(4) for b in range(4))) for mu in range(4)]
kdw = sum(k[a]*sp.diff(w, X[a]) for a in range(4))
prop = all(sp.simplify(acc[mu] - 2*kdw*k[mu]) == 0 for mu in range(4))
rec("P5", prop, "for g = e^{2w} eta and null k: Gamma^mu_ab k^a k^b = 2 (k.dw) k^mu, a reparametrization only: null geodesics are conformally invariant")

# P6 timelike geodesics are not: for timelike u the extra term -(u.u)_eta d^mu w survives
U = [sp.Symbol('U0', positive=True), sp.Symbol('U1', real=True), 0, 0]
accU = [sp.simplify(sum(Gm[mu][a][b]*U[a]*U[b] for a in range(4) for b in range(4))) for mu in range(4)]
udw = sum(U[a]*sp.diff(w, X[a]) for a in range(4))
uu = -U[0]**2 + U[1]**2
resid = [sp.simplify(accU[mu] - 2*udw*U[mu] + uu*sum(eta[mu, nn]*sp.diff(w, X[nn]) for nn in range(4))) for mu in range(4)]
transverse = sp.simplify(accU[2])                     # y-component: not proportional to U (U2 = 0)
rec("P6", all(r == 0 for r in resid) and transverse != 0,
    "for timelike U: Gamma U U = 2(U.dw)U - (U.U) d^mu w; the last term is a real force (y-component = %s): massive worldlines feel the conformal factor" % transverse)

# P7 eikonal: phase constant along a null ray; WKB phase of a massive particle advances as -m tau (Compton clock)
px = sp.symbols('p_x', real=True)
Em = sp.sqrt(px**2 + m**2)
S_phase = lambda Ev_, pv_: (lambda tt, xx: -Ev_*tt + pv_*xx)
vel = px/Em; gm = Em/m
dSdtau_massive = sp.simplify(gm*(-Em + px*vel))       # u^mu d_mu S with u = gamma(1, v)
pp = sp.symbols('p', positive=True)
dSdlam_null = sp.simplify(pp*(-pp) + pp*pp)            # S = -|p| t + p x, k = (|p|, p): k^mu d_mu S = 0
rec("P7", sp.simplify(dSdtau_massive + m) == 0 and sp.simplify(dSdlam_null) == 0,
    "massive plane wave: dS/dtau = -m (phase advances with proper time, the Compton clock); null ray: k^mu d_mu S = g^{mu nu} dS dS = 0 (phase frozen along the ray)")

# ---------------------------------------------------------------- L: what light and matter respond to
# L1 conformal factorization of the weak-field metric: only Phi+Psi shapes null cones
lhs_tt, lhs_xx = -(1 + 2*eps*Ps), (1 - 2*eps*Ph)
fac = (1 + 2*eps*Ps)
rhs_xx = fac*(1 - 2*eps*(Ph + Ps))
rec("L1", sp.series(lhs_xx - rhs_xx, eps, 0, 2).removeO() == 0,
    "-(1+2Psi)dt^2+(1-2Phi)dx^2 = (1+2Psi)[-dt^2+(1-2(Phi+Psi))dx^2] + O(eps^2): null cones see only Phi+Psi")

# L2 particle Lagrangian to first order: L = -m/gamma - m gamma (Psi + v^2 Phi)
Lag = -m*sp.sqrt((1 + 2*eps*Ps) - (1 - 2*eps*Ph)*v**2)
lin = sp.series(Lag, eps, 0, 2).removeO()
target = -m*sp.sqrt(1 - v**2) - eps*m*(Ps + v**2*Ph)/sp.sqrt(1 - v**2)
rec("L2", sp.simplify(lin - target) == 0,
    "L = -m sqrt((1+2Psi)-(1-2Phi)v^2) = -m/gamma - m gamma (Psi + v^2 Phi) + O(eps^2): a particle of speed v feels Psi + v^2 Phi")

# L3 limits: v -> 0 gives Psi (dynamics), v -> 1 gives Phi + Psi (lensing)
pot = Ps + v**2*Ph
rec("L3", pot.subs(v, 0) == Ps and pot.subs(v, 1) == Ps + Ph,
    "effective potential Psi + v^2 Phi: Psi for slow matter, Phi+Psi for light; the split is continuous in v, set by kinematics")

# L4 deflection by a point mass: alpha = 2GM(1 + gammaP v^2)/(b v^2); light with gammaP = 1: 4GM/b
G, M, b, gP, zz = sp.symbols('G M b gamma_P zeta', positive=True)
r = sp.sqrt(b**2 + zz**2)
Psi_pm = -G*M/r; Phi_pm = -gP*G*M/r
integrand = sp.diff(Psi_pm + v**2*Phi_pm, b)
alpha = sp.simplify(sp.integrate(integrand, (zz, -sp.oo, sp.oo))/v**2)
rec("L4", sp.simplify(alpha - 2*G*M*(1 + gP*v**2)/(b*v**2)) == 0 and sp.simplify(alpha.subs({v: 1, gP: 1}) - 4*G*M/b) == 0,
    "alpha = (1/v^2) Int d_b(Psi + v^2 Phi) dz = 2GM(1+gamma_P v^2)/(b v^2); light (v=1, gamma_P=1): 4GM/b (Einstein)")

# L5 mu-Sigma-slip identity
mu_, Sig, etaS, kk, a_, rho, Dl = sp.symbols('mu Sigma eta k a rho Delta', positive=True)
PsiS = -4*sp.pi*G*a_**2*mu_*rho*Dl/kk**2
PhiS = etaS*PsiS
SigmaExpr = sp.simplify((PhiS + PsiS)/(-8*sp.pi*G*a_**2*rho*Dl/kk**2))
rec("L5", sp.simplify(SigmaExpr - mu_*(1 + etaS)/2) == 0, "Sigma = mu (1 + eta)/2 with slip eta = Phi/Psi (definitions)")

# L6 Sigma = 1 forces eta = (2 - mu)/mu; with the book's mu(0) = 1/(1+beta): eta(0) = 1 + 2 beta = 1 + Omega_m
bet = sp.symbols('beta', positive=True)
eta_req = sp.solve(sp.Eq(mu_*(1 + etaS)/2, 1), etaS)[0]
eta0 = sp.simplify(eta_req.subs(mu_, 1/(1 + bet)))
beta_m = 0.15765
rec("L6", sp.simplify(eta_req - (2 - mu_)/mu_) == 0 and sp.simplify(eta0 - (1 + 2*bet)) == 0 and abs(float(eta0.subs(bet, beta_m)) - 1.315) < 6e-4,
    "Sigma = 1 => eta = (2-mu)/mu; at z = 0, mu = 1/(1+beta_m) => eta = 1 + 2 beta_m = 1 + Omega_m = %.5f (book p2_17:130: 1.315)" % float(eta0.subs(bet, beta_m)))

# L7 the book's three statements together (mu < 1, Sigma = 1, Phi = Psi) have no solution except mu = 1
sol = sp.solve([sp.Eq(mu_*(1 + 1)/2, Sig), sp.Eq(Sig, 1)], [mu_, Sig], dict=True)
rec("L7", sol == [{mu_: 1, Sig: 1}],
    "with eta = 1 (p2_03:766, p1_02 eq:law_noaniso) Sigma = mu; Sigma = 1 then forces mu = 1. mu < 1, Sigma = 1, Phi = Psi cannot hold together")

# L8 required anisotropic stress for the Level 1 form: k^2 (Phi - Psi) = 12 pi G a^2 (rho+p) sigma (Ma & Bertschinger eq 23d convention)
sigma_req = sp.simplify(kk**2*(PhiS - PsiS)/(12*sp.pi*G*a_**2) / (rho*Dl))
sigma_req_S1 = sp.simplify(sigma_req.subs(etaS, eta_req))
rec("L8", sp.simplify(sigma_req_S1 - (-sp.Rational(2, 3)*(1 - mu_))) == 0,
    "Level 1 needs (rho+p) sigma = -(2/3)(1-mu) rho_m Delta_m from some component: %.4f rho_m Delta_m today; a non-fluctuating scalar supplies none" % (-(2/3)*(1 - 1/(1 + beta_m))))

# L9 Bianchi consistency in Newtonian gauge (dust + Lambda, Phi = Psi, delta p = 0):
#  0i: k^2 (Phi' + H Phi) = 4 pi G a^2 rho theta ; ii: Phi'' + 3 H Phi' + (2 H' + H^2) Phi = 0
#  imply the matter Euler equation theta' = -H theta + k^2 Phi. A matter-sector friction H_m != H contradicts them.
Hc, Hp, rr, ra, P0, P1, P2, kq, Hm = sp.symbols('Hc Hp rho a P0 P1 P2 kq Hm', real=True)
Gs = sp.symbols('G', positive=True)
theta_0i = kq**2*(P1 + Hc*P0)/(4*sp.pi*Gs*ra**2*rr)
P2_ii = -3*Hc*P1 - (2*Hp + Hc**2)*P0
dlog_a2rho = 2*Hc - 3*Hc                                # (a^2 rho)'/(a^2 rho) with rho' = -3 H rho
theta_prime = kq**2*(P2_ii + Hp*P0 + Hc*P1)/(4*sp.pi*Gs*ra**2*rr) - dlog_a2rho*theta_0i
rhoL = sp.symbols('rho_L', positive=True)
bg = {Hp: (4*sp.pi*Gs/3)*ra**2*(-rr + 2*rhoL)}                     # second Friedmann eq (dust + Lambda), conformal time
fr1 = {rhoL: 3*Hc**2/(8*sp.pi*Gs*ra**2) - rr}                       # first Friedmann eq
euler_GR = sp.simplify(sp.expand((theta_prime - (-Hc*theta_0i + kq**2*P0)).subs(bg).subs(fr1)))
euler_Hm = sp.simplify(sp.expand((theta_prime - (-Hm*theta_0i + kq**2*P0)).subs(bg).subs(fr1)))
rec("L9", euler_GR == 0 and sp.simplify(euler_Hm - (Hm - Hc)*theta_0i) == 0,
    "unmodified Einstein 0i+ii equations force theta' = -H theta + k^2 Psi (Bianchi); a matter friction H_m leaves residual (H_m - H) theta != 0: momentum must go to another component")

# L10 the Level 2 friction is a drag on peculiar velocity, Gamma = 2(H_m - H); today Gamma/H0 = 2(sqrt(1+beta)-1)
Gam_today = 2*(np.sqrt(1 + beta_m) - 1)
rec("L10", abs(np.sqrt(1 + beta_m) - 1.0759) < 1e-4 and abs(Gam_today - 0.15188) < 1e-4,
    "delta'' + 2 H_m delta' = 4 pi G rho delta  <=>  Euler with drag Gamma = 2(H_m - H); today H_m/H = %.4f (book p1_02 table 1.0759), Gamma = %.4f H0" % (np.sqrt(1 + beta_m), Gam_today))

# L11 the book's growth numbers for forms (i) G_eff = mu G and (ii) friction on the LCDM clock (p1_02:520-528; app_C3)
Om_m = 0.3153; Om_L = 1 - Om_m
def Hl2(a): return Om_m*a**-3 + Om_L
def Ea(a): return np.exp(1 - 1/a)
def mu_a(a): return Hl2(a)/(Hl2(a) + beta_m*Ea(a))
def dlnH(a): return -1.5*Om_m*a**-3/Hl2(a)
def Omm(a): return Om_m*a**-3/Hl2(a)
def rhs(N, yv, form):
    a = np.exp(N); D_, Dp = yv
    if form == 'lcdm': return [Dp, -(2 + dlnH(a))*Dp + 1.5*Omm(a)*D_]
    if form == 'i':    return [Dp, -(2 + dlnH(a))*Dp + 1.5*Omm(a)*mu_a(a)*D_]
    if form == 'ii':
        Hm_H = np.sqrt((Hl2(a) + beta_m*Ea(a))/Hl2(a))
        return [Dp, -(dlnH(a) + 2*Hm_H)*Dp + 1.5*Omm(a)*D_]
a0 = 1e-3
out = {}
for f in ['lcdm', 'i', 'ii']:
    s = solve_ivp(rhs, [np.log(a0), 0.0], [a0, a0], args=(f,), rtol=1e-10, atol=1e-13)
    out[f] = s.y[0, -1]
dDi = 100*(out['i']/out['lcdm'] - 1); dDii = 100*(out['ii']/out['lcdm'] - 1)
rec("L11", abs(dDi + 0.78) < 0.03 and abs(dDii + 0.67) < 0.03,
    "growth today, same early amplitude, no radiation: form (i) %.3f %%, form (ii) %.3f %% (book: -0.78, -0.67)" % (dDi, dDii))

# L12 slip in the Level 1 form at the book's redshifts
zs = [0.0, 0.3, 1.0]; book = [1.315, 1.170, 1.036]
etas = [(2 - mu_a(1/(1 + zq)))/mu_a(1/(1 + zq)) for zq in zs]
rec("L12", all(abs(e - bk) < 1.5e-3 for e, bk in zip(etas, book)),
    "eta(z) = (2-mu)/mu = %s at z = 0, 0.3, 1 (book p2_17:130: 1.315, 1.170, 1.036)" % [round(e, 4) for e in etas])

# L13 linear Weyl tensor of the scalar-perturbed metric depends only on Phi+Psi (conformal factor a^2 drops out of C^a_bcd)
Xs = [t, x, y, z]
PsF = sp.Function('Psi')(*Xs); PhF = sp.Function('Phi')(*Xs); chi = sp.Function('chi')(*Xs)
def lin_weyl(PsiE, PhiE):
    h = sp.diag(-2*PsiE, -2*PhiE, -2*PhiE, -2*PhiE)
    n = 4
    Rl = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for A in range(n):
        for B in range(n):
            for C in range(n):
                for Dd in range(n):
                    Rl[A, B, C, Dd] = sp.Rational(1, 2)*(sp.diff(h[A, Dd], Xs[B], Xs[C]) + sp.diff(h[B, C], Xs[A], Xs[Dd])
                                                        - sp.diff(h[A, C], Xs[B], Xs[Dd]) - sp.diff(h[B, Dd], Xs[A], Xs[C]))
    Ric = sp.zeros(n)
    for B in range(n):
        for Dd in range(n):
            Ric[B, Dd] = sum(eta[A, A]*Rl[A, B, A, Dd] for A in range(n))
    Rs = sum(eta[B, B]*Ric[B, B] for B in range(n))
    Cw = {}
    for A in range(n):
        for B in range(n):
            for C in range(n):
                for Dd in range(n):
                    val = (Rl[A, B, C, Dd] - sp.Rational(1, 2)*(eta[A, C]*Ric[B, Dd] - eta[A, Dd]*Ric[B, C] - eta[B, C]*Ric[A, Dd] + eta[B, Dd]*Ric[A, C])
                           + Rs/6*(eta[A, C]*eta[B, Dd] - eta[A, Dd]*eta[B, C]))
                    Cw[(A, B, C, Dd)] = sp.expand(val)
    return Cw
C1 = lin_weyl(PsF, PhF)
C2 = lin_weyl(PsF - chi, PhF + chi)          # shift that keeps Phi+Psi fixed
same = all(sp.simplify(C1[kq_] - C2[kq_]) == 0 for kq_ in C1)
nonzero = any(C1[kq_] != 0 for kq_ in C1)
rec("L13", same and nonzero,
    "linear Weyl tensor of -(1+2Psi)dt^2+(1-2Phi)dx^2 is unchanged under Psi -> Psi - chi, Phi -> Phi + chi: it is a functional of Phi+Psi (the lensing potential) only")

# ---------------------------------------------------------------- S: the established route to Sigma = 1 (conformal coupling)
bt, ph, sgn = sp.symbols('beta phi s', real=True)
# S1 Jordan metric A^2 g with A = 1 + beta phi (first order): Psi_J = Psi + beta phi, Phi_J = Phi - beta phi
PsiJ = sp.series(-(1 + 2*eps*bt*ph)*(1 + 2*eps*Ps), eps, 0, 2).removeO()
PhiJ = sp.series((1 + 2*eps*bt*ph)*(1 - 2*eps*Ph), eps, 0, 2).removeO()
PsiJv = sp.simplify((-PsiJ - 1)/(2*eps)); PhiJv = sp.simplify(-(PhiJ - 1)/(2*eps))
rec("S1", sp.simplify(PsiJv - (Ps + bt*ph)) == 0 and sp.simplify(PhiJv - (Ph - bt*ph)) == 0 and sp.simplify(PsiJv + PhiJv - (Ps + Ph)) == 0,
    "conformal coupling g_J = A^2(phi) g: Psi_J = Psi + beta phi, Phi_J = Phi - beta phi, so Phi_J + Psi_J = Phi + Psi: light is blind to the scalar (Sigma = 1 exactly at linear order)")

# S2 quasi-static, massless: Einstein frame nabla^2 Psi = 4 pi G rho, scalar s nabla^2 phi = beta rho / M_pl (s = +1 healthy, -1 ghost),
#    matter feels Psi_J = Psi + beta phi / M_pl  =>  mu = 1 + 2 s beta^2
Mpl2 = 1/(8*sp.pi*G)
lapPsi = 4*sp.pi*G*rho
lapPhi_s = sgn*bt*rho/sp.sqrt(Mpl2)
mu_st = sp.simplify((lapPsi + bt*lapPhi_s/sp.sqrt(Mpl2))/lapPsi)
rec("S2", sp.simplify(mu_st - (1 + 2*sgn*bt**2)) == 0,
    "mu = 1 + 2 s beta^2 with Sigma = 1: a healthy scalar (s=+1) gives mu > 1; mu < 1 with Sigma = 1 in this class needs s = -1, a ghost")

# S3 massive (Yukawa) scalar: mu(k) = 1 + 2 s beta^2 k^2/(k^2 + a^2 m^2): same sign statement at every k
ms = sp.symbols('m_s', positive=True)
mu_k = 1 + 2*sgn*bt**2*kk**2/(kk**2 + a_**2*ms**2)
frac = sp.simplify((mu_k - 1)/(2*sgn*bt**2))
rec("S3", sp.simplify(frac - kk**2/(kk**2 + a_**2*ms**2)) == 0 and sp.simplify(1 - frac - a_**2*ms**2/(kk**2 + a_**2*ms**2)) == 0,
    "with a mass term the fifth-force factor k^2/(k^2+a^2 m^2) lies in (0,1): mu - 1 keeps the sign of s at all scales")

# ---------------------------------------------------------------- X: the trace as a covariant 'proper-time' weight (IAM proposal)
# X1 for a Maxwell-Juttner gas, (rho - 3p)/(n m) = <1/gamma> = K1(zeta)/K2(zeta), zeta = m/kT
def inv_gamma_avg(zeta):
    f = lambda p: p**2*mp.e**(-zeta*(mp.sqrt(1 + p**2) - 1))   # common factor e^{-zeta} removed
    pk = mp.sqrt(2/zeta) if zeta > 1 else 1
    pts = [0, pk/4, pk, 4*pk, 16*pk, mp.inf]
    num = mp.quad(lambda p: f(p)/mp.sqrt(1 + p**2), pts)
    den = mp.quad(f, pts)
    return num/den
vals = {zq: (inv_gamma_avg(zq), mp.besselk(1, zq)/mp.besselk(2, zq)) for zq in [0.01, 1, 10, 1000]}
okX1 = all(abs(a1 - a2) < 1e-8 for a1, a2 in vals.values())
rec("X1", okX1 and vals[1000][0] > 0.998 and vals[0.01][0] < 0.006,
    "trace weight per particle <1/gamma> = K1/K2: %s at m/kT = 0.01, 1, 10, 1000 (0 for light, 1 for cold matter)" % [round(float(vv_[0]), 5) for vv_ in vals.values()])

# X2 Landau-Lifshitz trace = bound-system energy: sum m/gamma = M - K to O(v^2), and the virial E = M + K + V = M - K for V = -2K
Kk = sp.symbols('K', positive=True)
mg = sp.series(m*sp.sqrt(1 - v**2), v, 0, 4).removeO()
rec("X2", sp.simplify(mg - (m - m*v**2/2)) == 0 and sp.simplify((m + Kk - 2*Kk) - (m - Kk)) == 0,
    "m/gamma = m - (1/2) m v^2 + O(v^4): the trace counts rest energy minus kinetic, which for a virialized 1/r system equals its total energy (LL s34; long form A9)")

out_json = {"checks": R, "n_pass": sum(1 for v_ in R.values() if v_["pass"]), "n_fail": sum(1 for v_ in R.values() if not v_["pass"]),
            "numbers": {"eta_today": float(eta0.subs(bet, beta_m)), "Gamma_over_H0_today": float(Gam_today),
                        "growth_form_i_pct": float(dDi), "growth_form_ii_pct": float(dDii), "eta_z": [float(e) for e in etas],
                        "sigma_req_today_over_rhoDelta": float(-(2/3)*(1 - 1/(1 + beta_m))),
                        "trace_weight_K1_over_K2": {str(k_): float(v_[1]) for k_, v_ in vals.items()}}}
json.dump(out_json, open("spider_s4_checks_output.json", "w"), indent=1)
print("TOTAL pass", out_json["n_pass"], "fail", out_json["n_fail"])
