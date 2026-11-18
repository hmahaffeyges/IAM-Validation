import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""which_threes_checks.py -- 2026-11-18, derivation spider (private, proposals only).
Redo IAM's horizon, continuity, critical-density, Smarr, virial, ledger and Koide identities with
d = number of SPACE dimensions (spacetime D = d+1) and N = number of GENERATIONS as separate symbols.
Units hbar = c = k_B = 1; G is the D-dimensional constant in G_ab = 8 pi G T_ab; S = A/4G.
Writes which_threes_checks_output.json. sympy + numpy only; seconds, < 150 MB.
"""
import json, sys, sympy as sp, numpy as np
R = {}
def chk(tag, desc, ok, val=None):
    ok = bool(ok); R[tag] = {"desc": desc, "pass": ok, "value": None if val is None else str(val)}
    print(("PASS " if ok else "FAIL ") + tag + ": " + desc + ("" if val is None else "  -> " + str(val)), flush=True)
z = lambda e: sp.simplify(e) == 0

d, N, nu, G, H, r, rho, p, Om, a, t = sp.symbols('d N nu G H r rho p Omega a t', positive=True)
Wd = sp.Symbol('Omega_dm1', positive=True)       # area of the unit (d-1)-sphere
A_ = Wd*r**(d-1); V_ = Wd*r**d/d                  # area and volume of a d-ball of radius r
T_ = 1/(2*sp.pi*r); S_ = A_/(4*G)                 # apparent-horizon temperature and entropy (Cai-Kim, any D)

# ---------------- A. geometry, continuity, Friedmann in d space dimensions
chk("A1", "V/(A r) = 1/d (the ledger's 1/3)", z(V_/(A_*r) - 1/d), sp.simplify(V_/(A_*r)))
chk("A2", "dlnV/dlnr = A r / V = d (the 3 of 3H)", z(sp.diff(sp.log(V_), r)*r - d))
w = sp.Symbol('w'); lna = sp.Symbol('lna')
f_ = sp.exp(-d*(1+w)*lna)                       # rho ~ a^(-d(1+w)), lna = ln a
chk("A3", "continuity drho/dlna = -d(1+w)rho -> rho ~ a^(-d(1+w)); dust a^-d, radiation (w=1/d) a^-(d+1)",
    z(sp.diff(f_, lna) + d*(1+w)*f_) and z(-d*(1+1/d) + (d+1)))
Hd = sp.Symbol('Hdot')
# Cai-Kim: -dE = A (rho+p) H r dt booked as T dS, flat r = 1/H
lhs = Wd*r**(d-1)*(rho+p)*H*r                     # per unit dt
dSdt = sp.diff(S_, r)*(-Hd/H**2)
eq = sp.Eq(lhs.subs(r, 1/H), (T_*dSdt).subs(r, 1/H))
Hdot_sol = sp.solve(eq, Hd)[0]
chk("A4", "Cai-Kim in D=d+1 gives Hdot = -8 pi G (rho+p)/(d-1)", z(Hdot_sol + 8*sp.pi*G*(rho+p)/(d-1)), Hdot_sol)
# with continuity, d(H^2)/dt = 2 H Hdot must equal (16 pi G/(d(d-1))) drho/dt, drho/dt = -d H (rho+p)
chk("A5", "Friedmann H^2 = 16 pi G rho/(d(d-1)) is consistent with A4 + continuity",
    z(2*H*Hdot_sol - 16*sp.pi*G/(d*(d-1))*(-d*H*(rho+p))))
rhoc = d*(d-1)*H**2/(16*sp.pi*G)
acc = sp.simplify(Hdot_sol + 16*sp.pi*G*rho/(d*(d-1)))
chk("A6", "a''/a = -(8 pi G/(d(d-1)))((d-2)rho + d p): active density (d-2)rho+dp, the 3 of rho+3p is d/(d-2)",
    z(acc + 8*sp.pi*G/(d*(d-1))*((d-2)*rho + d*p)))
wthr = sp.solve(sp.Eq((d-2)+d*w, 0), w)[0]
chk("A7", "acceleration threshold w < -(d-2)/d; = -1/3 at d=3", z(wthr + (d-2)/d) and wthr.subs(d, 3) == sp.Rational(-1, 3), wthr)
E = sp.exp(1-1/a); w_info = -1 - sp.diff(sp.log(E), a)*a/d
chk("A8", "record term rho ~ E(a)=exp(1-1/a): w = -1 - 1/(d a); CPL (w0,wa) = (-1-1/d, -1/d); (-4/3,-1/3) at d=3",
    z(w_info + 1 + 1/(d*a)) and z(-sp.diff(w_info, a).subs(a, 1) + 1/d) and w_info.subs({a: 1, d: 3}) == sp.Rational(-4, 3),
    (sp.simplify(w_info.subs(a, 1)), sp.simplify(-sp.diff(w_info, a).subs(a, 1))))

# ---------------- B. the critical-density identity
TSV = sp.simplify((T_*S_/V_).subs(r, 1/H))
chk("B1", "T_H S / V_H = d H^2/(8 pi G)  (the 3 here is the volume's d)", z(TSV - d*H**2/(8*sp.pi*G)), TSV)
ratio = sp.simplify(TSV/rhoc)
chk("B2", "T_H S/V_H = [2/(d-1)] rho_crit; = rho_crit only at d=3", z(ratio - 2/(d-1)) and sp.solve(sp.Eq(ratio, 1), d) == [3], ratio)
chk("B3", "the 3 of rho_crit=3H^2/8piG is d(d-1)/2 (rotation planes, dim so(d)); the 3 of V=4pi r^3/3 is d; equal only at d=3",
    sp.solve(sp.Eq(d*(d-1)/2, d), d) == [3])
EMS = (d-1)*Wd*r**(d-2)/(16*sp.pi*G)               # Misner-Sharp energy at the apparent horizon (flat)
chk("B4", "Misner-Sharp E_MS = (d-1) Omega r^(d-2)/16piG = rho_crit V_H; T S = [2/(d-1)] E_MS",
    z((EMS - rhoc*V_).subs(r, 1/H)) and z(T_*S_/EMS - 2/(d-1)))
chk("B5", "pure de Sitter: rho_L V_dS / (T_dS S_dS) = (d-1)/2 (=1 at d=3; dark-sector A10)", z((rhoc*V_/(T_*S_)).subs(r, 1/H) - (d-1)/2))
EK = lambda rr, pp: V_*((d-2)*rr + d*pp)/(d-2)    # Komar/Tolman active energy in D dims (Cai 2012 N_bulk form)
EK_L = sp.simplify(EK(rho, -rho))
chk("B6", "de Sitter Komar energy E_K = -2 rho V/(d-2); T S = [(d-2)/(d-1)] |E_K| (=1/2 at d=3, ledger E7)",
    z(EK_L + 2*rho*V_/(d-2)) and z(((T_*S_)/(-EK_L)).subs(rho, rhoc).subs(r, 1/H) - (d-2)/(d-1)))
alpha = (d-1)/(2*(d-2))
Nsur = (2*(-EK_L)/T_).subs(rho, rhoc).subs(r, 1/H)
chk("B7", "holographic equipartition in dS, N_sur = 2|E_K|/T = alpha*A/G with Cai's alpha=(d-1)/2(d-2)",
    z(Nsur - (alpha*A_/G).subs(r, 1/H)))

# ---------------- C. Smarr in D = d+1 (Tangherlini, Myers-Perry)
M_T = (d-1)*Wd*r**(d-2)/(16*sp.pi*G); T_T = (d-2)/(4*sp.pi*r); S_T = Wd*r**(d-1)/(4*G)
chk("C1", "Tangherlini: first law dM = T dS", z(sp.diff(M_T, r) - T_T*sp.diff(S_T, r)))
chk("C2", "Smarr (D-3)M = (D-2)TS, i.e. TS/M = (d-2)/(d-1); = 1/2 only at d=3",
    z(T_T*S_T/M_T - (d-2)/(d-1)) and sp.solve(sp.Eq((d-2)/(d-1), sp.Rational(1, 2)), d) == [3])
s = sp.Symbol('s', positive=True); Ms = s**((d-2)/(d-1))
chk("C3", "Euler: M is homogeneous of degree (D-3)/(D-2) = (d-2)/(d-1) in S, so Smarr = Euler (S-degree of M)",
    z(s*sp.diff(Ms, s) - (d-2)/(d-1)*Ms))
mu, aa = sp.symbols('mu a_s', positive=True)
rp = sp.sqrt(mu - aa**2); M5 = 3*sp.pi*mu/(8*G); J5 = 2*M5*aa/3
S5 = 2*sp.pi**2*mu*rp/(4*G); T5 = rp/(2*sp.pi*mu); Om5 = aa/mu
fl = [sp.simplify(sp.diff(M5, v) - T5*sp.diff(S5, v) - Om5*sp.diff(J5, v)) for v in (mu, aa)]
chk("C4", "Myers-Perry D=5, one spin: first law dM = T dS + Omega dJ holds for (mu, a)", all(f == 0 for f in fl), fl)
chk("C5", "Myers-Perry D=5: Smarr (D-3)M = (D-2)(TS + Omega J)", z(2*M5 - 3*(T5*S5 + Om5*J5)))
chk("C6", "Euler weights for Myers-Perry: S and J both scale as l^(D-2), M as l^(D-3) (dimensional homogeneity)",
    z(sp.simplify((M5.subs({mu: 4*mu, aa: 2*aa}))/M5) - 2**2) and z(sp.simplify(S5.subs({mu: 4*mu, aa: 2*aa})/S5) - 2**3) and z(sp.simplify(J5.subs({mu: 4*mu, aa: 2*aa})/J5) - 2**3))
bh = (d-2)/(d-1); cos_ms = 2/(d-1)
chk("C7", "held share of the horizon's own energy: black hole TS/M=(d-2)/(d-1), cosmic TS/E_MS=2/(d-1); (1/2, 1) at d=3; ratio (d-2)/2",
    z(bh/cos_ms - (d-2)/2) and (bh.subs(d, 3), cos_ms.subs(d, 3)) == (sp.Rational(1, 2), 1))
chk("C8", "with Komar (active) energies both horizons give TS = [(d-2)/(d-1)] E_active for every d (B6 + C2)",
    z(T_T*S_T/M_T - ((T_*S_)/(-EK_L)).subs(rho, rhoc).subs(r, 1/H)))
Dd = sp.Symbol('D', positive=True)
newt = sp.simplify(8*sp.pi*(1 - 1/(Dd-2)))   # trace-reversed R_00 = 8piG(T_00 - T g_00/(D-2)) for static dust, T=-rho, g00=-1
chk("C9", "Newtonian limit: laplacian Phi = 8piG (D-3)/(D-2) rho; 4piG at D=4; the book's 'trace factor 2' is (D-2)/(D-3), the Smarr M/TS",
    z(newt - 8*sp.pi*(Dd-3)/(Dd-2)) and newt.subs(Dd, 4) == 4*sp.pi and z(8*sp.pi/newt - 1/bh.subs(d, Dd-1)))
k = sp.Symbol('kappa', positive=True)
chk("C10", "horizon Landauer: heat T ln2 at T = kappa/2pi adds dA = 4 G ln2 for every kappa and every d (dS = dE/T, S=A/4G)",
    z((k/(2*sp.pi))*sp.log(2)/(k/(2*sp.pi))*4*G - 4*G*sp.log(2)))
chk("C11", "Bekenstein-Hawking 1/4 = (Euclidean period 2pi)/(Einstein 8pi): both D-independent", sp.Rational(1, 4) == (2*sp.pi/(8*sp.pi)))

# ---------------- D. the virial half in d space dimensions
rr = sp.Symbol('rr', positive=True)
Phi = -rr**(-(d-2))
lap = sp.simplify(sp.diff(rr**(d-1)*sp.diff(Phi, rr), rr)/rr**(d-1))
chk("D1", "Gauss in d: Phi ~ -r^-(d-2) is harmonic away from the source (degree k = -(d-2); -1 at d=3)", lap == 0)
kdeg = -(d-2)
chk("D2", "Euler: r dV/dr = kV with k=-(d-2), so Lagrange 1/2 I'' = 2T - kV = 2T + (d-2)V", z(rr*sp.diff(Phi, rr) - kdeg*Phi))
K_share = -kdeg/2; Q_share = 1 + kdeg/2
chk("D3", "binding from rest: kept kinetic K = ((d-2)/2)|V|, radiated Q = ((4-d)/2)|V|; K = Q only at d=3",
    z(K_share - (d-2)/2) and z(Q_share - (4-d)/2) and sp.solve(sp.Eq(K_share, Q_share), d) == [3])
L, m, C = sp.symbols('L m C', positive=True)
Veff = L**2/(2*m*rr**2) - C/rr**(d-2)
r0 = sp.Symbol('r0', positive=True)
Cstar = L**2*r0**(d-4)/(m*(d-2))                 # circular-orbit condition V_eff'(r0)=0 solved for C (d>2)
assert z(sp.diff(Veff, rr).subs({rr: r0, C: Cstar}))
sec = sp.simplify((sp.diff(Veff, rr, 2)*rr**4*m/L**2).subs({rr: r0, C: Cstar}))
chk("D4", "circular orbits: V_eff''(r0) = (4-d) L^2/(m r0^4) > 0 only for d<4 (Ehrenfest 1917, Tangherlini 1963)", z(sec - (4-d)), sec)
chk("D5", "virial kept share (d-2)/2 equals Smarr share (d-2)/(d-1) only at d=2 (both 0) and d=3 (both 1/2)",
    set(sp.solve(sp.Eq((d-2)/2, (d-2)/(d-1)), d)) == {2, 3})
x_ = sp.Symbol('x', positive=True); F_ = sp.Symbol('F', positive=True); Ast = sp.Symbol('A_s', positive=True)
xs = F_/(2*Ast)
chk("D6", "sudden write / capacitor (degree-2 store, degree-1 source): held/source = 1/2 for every d (no d in the store)",
    z(Ast*xs**2/(F_*xs) - sp.Rational(1, 2)))
chk("D7", "generalized Euler table in d: held/source = |n_source|/n_store; virial (d-2)/2, Smarr (d-2)/(d-1), write 1/2; all 1/2 only at d=3",
    all(sp.simplify(e.subs(d, 3)) == sp.Rational(1, 2) for e in ((d-2)/2, (d-2)/(d-1), sp.Rational(1, 2))) and
    all(sp.simplify(e.subs(d, 4)) != sp.Rational(1, 2) for e in ((d-2)/2, (d-2)/(d-1))))

# ---------------- E. the ledger identity in d space dimensions (content rule, record term per e-fold)
Om_m = sp.Symbol('Omega_m', positive=True); rm = Om_m*rhoc
flux = (A_*rm*r).subs(r, 1/H)                       # -dE per e-fold for dust: A rho r_A H dt, dt = 1/H
Mm = (rm*V_).subs(r, 1/H)
chk("E1", "matter's rest-energy flux across r_A per e-fold = d M_m (swept shell A r_A = d V_H)", z(flux - d*Mm))
dSgeo = (sp.diff(S_, r)*(-r**2)*Hdot_sol.subs({rho: rm, p: 0})/H).subs(r, 1/H)   # dS/dlna = dS/dr * dr/dt / H
chk("E2", "flux entropy = dS_geo/dlna (first law) = [d(d-1)/2] Omega_m S_geo (the 3 of 3 Omega_m S is d(d-1)/2)",
    z(flux/T_.subs(r, 1/H) - dSgeo) and z(dSgeo/S_.subs(r, 1/H) - d*(d-1)/2*Om_m))
hs = {"equipartition/write 1/2": sp.Rational(1, 2), "virial (d-2)/2": (d-2)/2, "Smarr/Komar (d-2)/(d-1)": (d-2)/(d-1)}
okE3 = all(z((h*Mm/T_.subs(r, 1/H))/(flux/T_.subs(r, 1/H)) - h/d) for h in hs.values())
chk("E3", "content rule (record = h x M_m/T per e-fold): record/flux = h/d; 1/6 at d=3 for every h", okE3)
beta = {kk: sp.simplify(h*Mm/(T_*S_).subs(r, 1/H)) for kk, h in hs.items()}
chk("E4", "beta_m(d) = h (d-1)/2 Omega_m: (d-1)/4, (d-2)(d-1)/4, (d-2)/2 times Omega_m; all Omega_m/2 at d=3, pairwise different at d=4,5",
    all(z(b.subs(d, 3) - Om_m/2) for b in beta.values()) and
    all(len({sp.simplify(b.subs(d, dd)) for b in beta.values()}) == 3 for dd in (4, 5)), beta)
Nbulk_m = 2*EK(rm, 0)/T_
NB_over_NS = sp.simplify((Nbulk_m/(alpha*A_/G)).subs(r, 1/H))
chk("E5", "Padmanabhan/Cai tiles: |N_bulk,m|/N_sur = (d-2) Omega_m/2 = the Smarr/Komar beta (ledger L3.5 route picks (d-2)/(d-1))",
    z(NB_over_NS - (d-2)*Om_m/2) and z(NB_over_NS - beta["Smarr/Komar (d-2)/(d-1)"]))
OmL = 1 - Om_m
EKtot = EK(rm, 0) + EK(OmL*rhoc, -OmL*rhoc)
dN = (alpha*A_/G) - (-2*EKtot/T_)
halfTdN = sp.simplify((T_*dN/2).subs(r, 1/H))
chk("E6", "1/2 T DeltaN = [d/(d-2)] M_m: equals the flux d M_m only at d=3 (ledger L4.2's '3 M_m')",
    z(halfTdN - d/(d-2)*Mm) and sp.solve(sp.Eq(d/(d-2), d), d) == [3])
dVdt = (sp.diff(V_, r)*(-Hd/H**2)).subs(Hd, Hdot_sol.subs({rho: rm, p: 0})).subs(r, 1/H)
chk("E7", "Cai's law dV/dt = (G/alpha) DeltaN reproduces Hdot in every d (the 1/alpha = 2(d-2)/(d-1) is 1 at d=3)",
    z(dVdt - (G/alpha)*dN.subs(r, 1/H)))
chk("E8", "Schwarzschild-Tangherlini, flux form: S = [(d-2)/(d-1)] x (M/T) = h_Smarr x (entropy per e-fold of M); no 1/d",
    z(S_T - (d-2)/(d-1)*(M_T/T_T)) and z(M_T*sp.diff(S_T, r)/sp.diff(M_T, r) - M_T/T_T))

# ---------------- F. Koide with N generations
def Qof(v): return sum(x**2 for x in v)/sum(v)**2
okF1 = True
for n in range(2, 7):
    s_ = sp.symbols('s0:%d' % n, positive=True)
    mu_ = sum(s_)/n; var = sum((x-mu_)**2 for x in s_)/n
    okF1 &= z(Qof(s_) - (1 + var/mu_**2)/n)
chk("F1", "Q = sum m/(sum sqrt m)^2 = (1+CV^2)/N for N = 2..6 (population CV of sqrt m)", okF1)
xb, rb, db = sp.symbols('x r delta', positive=True)
okF2 = True
for n in range(3, 8):
    for (xv, rv, dv) in ((1.3, 0.7, 0.37), (0.9, 1.41421356, 2.1), (2.0, 0.2, 5.0)):
        v = [xv*(1 + rv*np.cos(dv + 2*np.pi*kk/n)) for kk in range(n)]
        okF2 &= abs(sum(x*x for x in v)/sum(v)**2 - (1 + rv**2/2)/n) < 1e-12
    okF2 &= abs(sum(np.cos(0.37 + 2*np.pi*kk/n) for kk in range(n))) < 1e-12 and abs(sum(np.cos(0.37 + 2*np.pi*kk/n)**2 for kk in range(n)) - n/2) < 1e-12
chk("F2", "Brannen form on N equally spaced phases (N>=3): Q = (1 + r^2/2)/N, so r = sqrt2 gives Q = 2/N", okF2)
CV = sp.Symbol('CV', positive=True)
chk("F3", "D_1/2(p||u) = ln(N Q) = ln(1+CV^2): one bit iff CV = 1, for every N (the bit is N-free)",
    z(sp.log(N*(1+CV**2)/N) - sp.log(1+CV**2)) and z(sp.log(1+CV**2).subs(CV, 1) - sp.log(2)))
chk("F4", "with CV=1, Q = 2/N; Q = 2/3 iff N = 3 (the 3 of 2/3 is N)", sp.solve(sp.Eq(2/N, sp.Rational(2, 3)), N) == [3])
Qnu = (1 + 2/nu)/N
chk("F5", "chi^2_nu reading: CV^2 = 2/nu, Q_nu = (nu+2)/(N nu); one canonical pair nu=2 gives 2/N (the 2 is nu, d-free)",
    z(Qnu - (nu+2)/(N*nu)) and z(Qnu.subs(nu, 2) - 2/N))
chk("F6", "coincidence: Koide Q = 2/N equals CV^2 of a d-dim kinetic energy (chi^2_d) only if N = d; Koide's own nu=d=3 would give 5/9",
    sp.solve(sp.Eq(2/N, 2/d), N) == [d] and Qnu.subs({nu: 3, N: 3}) == sp.Rational(5, 9))
chk("F7", "plane orthogonal to (1,...,1) has dim N-1; one canonical pair (dim 2) iff N=3; the record (scale,Q) keeps 2 numbers for any N",
    sp.solve(sp.Eq(N-1, 2), N) == [3])
Qd1 = (1 + (d-2)/(4-d))/3; Qd2 = (1 + (4-d)/(d-2))/3
chk("F8", "book's 1/r-virial route to y/x (p4_15a:193, 'analogy') in d dims: Q = 2/(3(4-d)) or 2/(3(d-2)); 2/3 only at d=3",
    z(Qd1 - 2/(3*(4-d))) and z(Qd2 - 2/(3*(d-2))) and sp.solve(sp.Eq(Qd1, sp.Rational(2, 3)), d) == [3] and sp.solve(sp.Eq(Qd2, sp.Rational(2, 3)), d) == [3])
rng = np.random.default_rng(20261118)
frac = {}
for n in (3, 4, 5, 6):
    u = np.ones(n)/np.sqrt(n)
    g = rng.standard_normal((400000, n)); g -= np.outer(g @ u, u)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    v = u + g                                  # 45 degrees: |s_perp| = |s_par| = 1
    frac[n] = float(np.mean(np.all(v > 0, axis=1)))
se = np.sqrt(0.25*0.75/400000)
chk("F9", "general vectors on the 45-degree (CV=1) cone: all-positive fraction N=3 -> 1/4 (forward D8b); N>=4 nonzero, so the book's n<=3 bound needs the single-harmonic encoding, not CV=1 alone",
    abs(frac[3] - 0.25) < 5*se and frac[4] > 0.01, frac)
Qmeas = sp.Rational('0.666664463')
chk("F10", "measured (forward D5, PDG 2024 pole masses): CV^2 = 3Q-1 = 0.9999934; LEP N_nu = 2.9840 +/- 0.0082 (N=3, an independent count)",
    abs(float(3*Qmeas - 1) - 0.999993389) < 1e-8, float(3*Qmeas - 1))
chk("F11", "book's generation-count theorem (p4_15a:208-226): admissible-offset fraction max(0, 1-n/4) contains no d",
    all(max(0, 1 - nn/4) == v for nn, v in ((2, 0.5), (3, 0.25), (4, 0), (5, 0))))

npass = sum(v["pass"] for v in R.values()); nfail = len(R) - npass
json.dump({"checks": R, "passed": npass, "failed": nfail}, open("which_threes_checks_output.json", "w"), indent=1)
print("TOTAL", npass, "passed", nfail, "failed")
