import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
#!/usr/bin/env python3
"""R13 bookings checks, 2026-11-18 (SPIDER, private; proposals only).
Which booking prices a written bit at the cosmic apparent horizon, and does it give the book's beta_m = Omega_m/2?
Every step tagged DERIVED HERE in R13_BOOKINGS_2026-11-18.md has a check id (K1..K22) here.
Run: python3 r13_bookings_checks.py  -> r13_bookings_checks_output.json
Units hbar = c = k_B = 1 unless stated. Light: sympy + mpmath only."""
import json
import sympy as sp
import mpmath as mp

RES = []
def check(cid, desc, ok, detail=""):
    RES.append({"id": cid, "desc": desc, "pass": bool(ok), "detail": str(detail)[:400]})

def z(e):
    return sp.simplify(sp.expand(e)) == 0

t = sp.symbols('t', real=True)
G = sp.symbols('G', positive=True)
H = sp.Function('H')(t)
Sinfo = sp.Function('S_info')(t)
rho, p = sp.symbols('rho p', real=True)          # matter (all ordinary fluids) density and pressure
Hd = sp.symbols('Hdot', real=True)                # unknown dH/dt to solve for
Sd = sp.symbols('Sdot', real=True)                # dS_info/dt
Hs = sp.symbols('H', positive=True)

R = 1/Hs                                          # apparent horizon radius, flat FRW
Rd = -Hd/Hs**2                                    # dR/dt
S_geo_dot = 2*sp.pi*R*Rd/G                        # d/dt (pi R^2 / G)
T_GH = 1/(2*sp.pi*R)
V = sp.Rational(4, 3)*sp.pi*R**3
Vd = 4*sp.pi*R**2*Rd
flux = 4*sp.pi*R**2*(rho + p)*Hs*R                # Cai-Kim energy crossing the momentarily fixed horizon per dt
Hdot_GR = -4*sp.pi*G*(rho + p)

# ---------------------------------------------------------------- booking A: heat crossing
# K1 Cai-Kim with the record: flux = T_GH (S_geo' + S_info') => Hdot = -4 pi G (rho+p) + (G/2pi) H^3 S_info'
solA = sp.solve(sp.Eq(flux, T_GH*(S_geo_dot + Sd)), Hd)
HdA = solA[0]
check("K1", "Cai-Kim + S_info: Hdot = -4piG(rho+p) + (G/2pi) H^3 Sdot (book p1_02:328-329)",
      len(solA) == 1 and z(HdA - (Hdot_GR + G*Hs**3*Sd/(2*sp.pi))), HdA)

# K2 effective record fluid of booking A: rho_x + p_x = -H^3 Sdot/(8 pi^2); drho_x/dt = -3H(rho_x+p_x) = T_GH Sdot / V
rpx = sp.simplify((HdA - Hdot_GR)/(-4*sp.pi*G))
rhox_dot_A = sp.simplify(-3*Hs*rpx)
check("K2", "A: rho_x+p_x = -H^3 Sdot/(8 pi^2) (phantom for Sdot>0); drho_x/dt = T_GH Sdot / V = 3H^4 Sdot/(8pi^2)",
      z(rpx + Hs**3*Sd/(8*sp.pi**2)) and z(rhox_dot_A - T_GH*Sd/V) and z(rhox_dot_A - 3*Hs**4*Sd/(8*sp.pi**2)),
      (rpx, rhox_dot_A))

# K3 book's requirement (p1_02:333): with drho_x/dt = rho_x H/a, Sdot = 8 pi^2 rho_x/(3 a H^3); dS/dN at a=1, H=H0 = beta S_geo
a, beta, H0 = sp.symbols('a beta H_0', positive=True)
rhox = sp.symbols('rho_x', positive=True)
Sdot_A = sp.solve(sp.Eq(3*Hs**4*Sd/(8*sp.pi**2), rhox*Hs/a), Sd)[0]
dSdN_1 = (Sdot_A/Hs).subs({a: 1, Hs: H0, rhox: beta*3*H0**2/(8*sp.pi*G)})
check("K3", "A reproduces book eq:law_Sneed: Sdot = 8pi^2 rho_x/(3aH^3); dS_info/dN|_(a=1) = beta * pi/(G H0^2) = beta S_geo",
      z(Sdot_A - 8*sp.pi**2*rhox/(3*a*Hs**3)) and z(dSdN_1 - beta*sp.pi/(G*H0**2)), dSdN_1)

# K4 Akbar-Cai / Hayward unified first law with the Kodama-Hayward temperature and the record:
#     d(rho V) = T dS_total + W dV with T = kappa/2pi, kappa = -(1/R)(1 - Rdot/(2HR)) (negative), W = (rho-p)/2.
kappa = -(1/R)*(1 - Rd/(2*Hs*R))
T_signed = kappa/(2*sp.pi)
rho_dot = -3*Hs*(rho + p)
lhs = rho_dot*V + rho*Vd
rhs = T_signed*(S_geo_dot + Sd) + (rho - p)/2*Vd
solK = sp.solve(sp.Eq(lhs, rhs), Hd)
phys = [s for s in solK if not z(s + 2*Hs**2)]
spur = [s for s in solK if z(s + 2*Hs**2)]
check("K4", "Unified first law at T_KH (+ work term) with S_info gives EXACTLY booking A's Hdot; the factor (1-eps/2) cancels; "
      "the only other root is eps = 2 (T_KH = 0, degenerate)",
      len(phys) == 1 and z(phys[0] - HdA) and len(spur) == 1, solK)

# K5 the hybrid (flux at T_KH, no work term) fails already for S_info = 0: it does not return Friedmann's Hdot
T_KH = -T_signed
hyb = sp.solve(sp.Eq(flux, T_KH*S_geo_dot), Hd)
ok5 = all(not z(s - Hdot_GR) for s in hyb)
check("K5", "Hybrid 'heat crossing priced at T_KH' (no W dV) is inconsistent: with S_info = 0 it does not give Hdot = -4piG(rho+p)",
      ok5, hyb)

# K6 Kodama-Hayward / incremental temperature coincide only at eps = 1
eps = sp.symbols('epsilon', real=True)
check("K6", "T_inc = dE_MS/dS = T_GH/2 equals T_KH = (1 - eps/2) T_GH only at eps = 1",
      sp.solve(sp.Eq(1 - eps/2, sp.Rational(1, 2)), eps) == [1])

# ---------------------------------------------------------------- booking B: the tally
# K7 tally identity: E_MS = R/(2G), S_geo = pi R^2/G => dE_MS = (T_GH/2) dS_geo for any R(t) (Euler, degree 2)
Rf = sp.Function('R')(t)
E_MS_f = Rf/(2*G); S_f = sp.pi*Rf**2/G; TGH_f = 1/(2*sp.pi*Rf)
check("K7", "Tally identity: dE_MS/dt = (T_GH/2) dS_geo/dt for any R(t)  [T_inc = T_GH/2]",
      z(sp.diff(E_MS_f, t) - TGH_f/2*sp.diff(S_f, t)))

# K8 Misner-Sharp mass is the quasi-local tally (the GR Gauss law): for flat FRW, any comoving sphere r,
#     m = (R/2G)(1 - g^ab dR dR) with R = a r equals (4pi/3) rho R^3 once H^2 = 8 pi G rho/3. Not only at the horizon.
af = sp.Function('a')(t); rr = sp.symbols('r', positive=True)
Rar = af*rr
grad2 = -sp.diff(Rar, t)**2 + (1/af**2)*sp.diff(Rar, rr)**2     # g^tt = -1, g^rr = 1/a^2
m_MS = Rar/(2*G)*(1 - grad2)
Hexpr = sp.diff(af, t)/af
rho_F = 3*Hexpr**2/(8*sp.pi*G)
check("K8", "Misner-Sharp m = (R/2G)(1 - |dR|^2) = (4pi/3) rho R^3 on every comoving sphere (constraint = tally, no signal)",
      z(m_MS - sp.Rational(4, 3)*sp.pi*rho_F*Rar**3), sp.simplify(m_MS))

# K9 tally with the record: E_MS = E_m + E_info, dE_info = T_inc dS_info  =>  H^2 = (8piG/3)(rho + E_info/V)
Einfo = sp.symbols('E_info', positive=True)
Hsol = sp.solve(sp.Eq(R/(2*G), rho*V + Einfo), rho)[0]
H2B = sp.simplify(sp.Rational(8, 3)*sp.pi*G*(Hsol + Einfo/V))
check("K9", "B: E_MS = rho V + E_info is the first Friedmann equation H^2 = (8piG/3)(rho + E_info/V)",
      z(H2B - Hs**2), H2B)

# K10 B's record density: rho_x = E_info/V, d rho_x/dt = T_inc Sdot / V - 3 eps H rho_x  (eps = Rdot = -Hdot/H^2)
Ef = sp.Function('E_info')(t); Vf = sp.Rational(4, 3)*sp.pi*Rf**3
rhoxB = Ef/Vf
drhoxB = sp.diff(rhoxB, t).subs(sp.diff(Ef, t), TGH_f/2*sp.Symbol('Sdot'))
target = (TGH_f/2)*sp.Symbol('Sdot')/Vf - 3*sp.diff(Rf, t)/Rf*rhoxB
check("K10", "B: d rho_x/dt = (T_GH/2) Sdot / V - 3 (Rdot/R) rho_x = T_GH Sdot/(2V) - 3 eps H rho_x",
      z(drhoxB - target))

# K11 marginal price: a bit written now adds to the enclosed record density (T_GH/2)/V under B and T_GH/V under A: ratio 1/2 exactly
check("K11", "Marginal (incremental) price of a new entry, B/A = 1/2 exactly (Euler degree 2, the Smarr half)",
      z(((TGH_f/2)/Vf)/(TGH_f/Vf) - sp.Rational(1, 2)))

# K12 A and B cannot both hold for a growing record: equating K2 and K10 forces Sdot = -6 eps H rho_x V / T_GH < 0
Sx = sp.symbols('Sx', real=True); epsp, Hp, rp_, Vp, Tp = sp.symbols('eps_p H_p rho_p V_p T_p', positive=True)
solAB = sp.solve(sp.Eq(Tp*Sx/Vp, Tp*Sx/(2*Vp) - 3*epsp*Hp*rp_), Sx)
check("K12", "A and B together require Sdot = -6 eps H rho_x V/T_GH (< 0 for eps>0, rho_x>0): incompatible for a growing record",
      len(solAB) == 1 and z(solAB[0] + 6*epsp*Hp*rp_*Vp/Tp) and solAB[0].is_negative, solAB)

# K13 B's writing needed to reproduce the book's rho_info (d rho/dt = rho H/a): Sdot_B / Sdot_A = 2 (1 + 3 eps a)
SdB = sp.solve(sp.Eq(Tp*Sx/(2*Vp) - 3*epsp*Hp*rp_, rp_*Hp/a), Sx)[0]
SdA = sp.solve(sp.Eq(Tp*Sx/Vp, rp_*Hp/a), Sx)[0]
ratio13 = sp.simplify(SdB/SdA)
Om = sp.Rational(3153, 10000)
eps_today = sp.Rational(3, 2)*Om                     # LCDM, radiation and the record itself neglected (provisional)
val13 = float(ratio13.subs({a: 1, epsp: eps_today}))
check("K13", "B needs Sdot_B = 2(1 + 3 eps a) Sdot_A to give the book's E(a); at a=1 with LCDM eps = 1.5 Omega_m: 4.838",
      z(ratio13 - 2*(1 + 3*epsp*a)) and abs(val13 - 4.83770) < 1e-4, (ratio13, val13))

# K14 B applied to the entropy history that A needs (book's S_info): rho_B(1)/rho_A(1) = (1/2) int_0^1 E(a)/(a^2 h^3) da
mp.mp.dps = 30
OmL = 1 - 0.3153
h = lambda x: mp.sqrt(0.3153*x**-3 + OmL)
Ea = lambda x: mp.e**(1 - 1/x) if x > 0 else mp.mpf(0)
I14 = mp.quad(lambda x: Ea(x)/(x**2*h(x)**3), [0, 0.2, 0.5, 1])
R14 = 0.5*I14
# symbolic route of the integrand: E_info(1) = int T_inc Sdot_A dt, dt = da/(aH), rho_B(1) = E_info(1)/V(1)
Hq = sp.symbols('Hq', positive=True)
integrand = sp.simplify((Hq/(4*sp.pi))*(8*sp.pi**2*rhox/(3*a*Hq**3))*(1/(a*Hq))*(3*H0**3/(4*sp.pi)))
check("K14", "B priced on A's (the book's) entropy history gives beta_B = R*beta_m with R = (1/2) int E/(a^2 h^3) da = %.4f (LCDM, provisional)" % float(R14),
      z(integrand - rhox*H0**3/(2*a**2*Hq**3)) and 0 < R14 < 0.5, float(R14))

# K15 matter-share writing (dS/dN = Omega_m(a) S_geo) under B: dOmega_x/dN = Omega_m/2 - eps Omega_x;
#     today the gross entry per e-fold is Omega_m/2 of E_MS; in the matter era the fixed point is Omega_x = 1/3 (not E(a) -> 0)
Ox, Omv = sp.symbols('Omega_x Omega_mv', positive=True)
dOx = Omv/2 - epsp*Ox                                   # from K7/K10 in Omega form, d ln E_MS / dN = eps
# derivation check of the Omega form: E_info = Ox E_MS, dE_info/dN = (E_MS/(2 S_geo)) dS/dN, dS/dN = Om S_geo, dlnE_MS/dN = eps
EMS, SG = sp.symbols('E_MS S_G', positive=True)
dEinfo_dN = EMS/(2*SG)*(Omv*SG)
dOx_derived = (dEinfo_dN - Ox*EMS*epsp)/EMS
fp = sp.solve(sp.Eq(dOx.subs({Omv: 1, epsp: sp.Rational(3, 2)}), 0), Ox)
check("K15", "Matter-share writing under B: gross entry per e-fold = Omega_m/2 of E_MS (the number), but the history "
      "tracks Omega_x -> 1/3 in the matter era (the book's term must be off there)",
      z(dOx_derived - dOx) and fp == [sp.Rational(1, 3)], fp)

# K16 stock revaluation vs tally: T_inc * f S_geo = (f/2) E_MS, but the integrated tally int T_inc d(f S_geo) = f E_MS (f const)
f = sp.symbols('f', positive=True)
Rx = sp.symbols('Rx', positive=True)
stock = (1/(4*sp.pi*Rx))*f*sp.pi*Rx**2/G
tally = sp.integrate((1/(4*sp.pi*Rx))*sp.diff(f*sp.pi*Rx**2/G, Rx), (Rx, 0, Rx))
check("K16", "Revaluing a stock at today's incremental rate gives the half (f/2)E; the integrated tally of the same entries gives f E "
      "(Euler re-integrates the half away)", z(stock - f*Rx/(4*G)) and z(tally - f*Rx/(2*G)), (stock, tally))

# K17 D dimensions: Hubble sphere E = (D-2) W R^(D-3)/(16piG), S = W R^(D-2)/(4G), T_GH = 1/(2 pi R)
D = sp.symbols('D', positive=True, integer=True); Wd = sp.symbols('Omega_D', positive=True); Rd_ = sp.symbols('R', positive=True)
E_D = (D - 2)*Wd*Rd_**(D - 3)/(16*sp.pi*G); S_D = Wd*Rd_**(D - 2)/(4*G); T_D = 1/(2*sp.pi*Rd_)
Tinc_D = sp.simplify(sp.diff(E_D, Rd_)/sp.diff(S_D, Rd_))
# consistency: E_D = rho V with H^2 = 16 pi G rho /((D-1)(D-2)), V = W R^(D-1)/(D-1), R = 1/H
rhoD = (D - 1)*(D - 2)/(16*sp.pi*G*Rd_**2)
consist = z(E_D - rhoD*Wd*Rd_**(D - 1)/(D - 1))
smarr = sp.simplify(Tinc_D*S_D/E_D)
cap = sp.simplify(T_D*S_D/E_D)
virial = (D - 3)/sp.Integer(2)                     # K/|V| for a potential of degree -(D-3) (D-1 space dims)
d4 = [x.subs(D, 4) for x in (sp.simplify(Tinc_D/T_D), smarr, cap, virial)]
d5 = [x.subs(D, 5) for x in (sp.simplify(Tinc_D/T_D), smarr, cap, virial)]
check("K17", "D dims: T_inc/T_GH = (D-3)/2, tally share T_inc S/E = (D-3)/(D-2) (Smarr), capacity T_GH S/E = 2/(D-2), "
      "virial K/|V| = (D-3)/2. D=4: 1/2,1/2,1,1/2. D=5: 1,2/3,2/3,1. The tally half and the virial half coincide only in D=4",
      consist and z(Tinc_D/T_D - (D - 3)/2) and z(smarr - (D - 3)/(D - 2)) and z(cap - 2/(D - 2))
      and d4 == [sp.Rational(1, 2), sp.Rational(1, 2), 1, sp.Rational(1, 2)] and d5 == [1, sp.Rational(2, 3), sp.Rational(2, 3), 1],
      {"D4": d4, "D5": d5})

# K18 book's beta_m as matter's share of the Hubble-sphere compactness: G M_m H / c^3 = Omega_m/2 (coefficient half, D=4)
Omg = sp.symbols('Omega_m', positive=True)
Mm = Omg*3*Hs**2/(8*sp.pi*G)*sp.Rational(4, 3)*sp.pi/Hs**3
check("K18", "G M_m H = Omega_m/2 at R = 1/H: the book's number is matter's compactness share", z(G*Mm*Hs - Omg/2))

# ---------------------------------------------------------------- EGJ non-equilibrium readings
# K19 EGJ-i: d(S_geo + S_info) = dQ/T + d_iS with d_iS = dS_info  =>  Friedmann unchanged (coefficient 0)
s19 = sp.solve(sp.Eq(T_GH*(S_geo_dot + Sd), flux + T_GH*Sd), Hd)
check("K19", "EGJ reading (i): S_info production is d_iS (no heat) => Hdot = -4piG(rho+p): record coefficient 0",
      len(s19) == 1 and z(s19[0] - Hdot_GR), s19)

# K20 EGJ-ii: dS_geo = dQ/T + dS_info (record produced internally on the area)  =>  rho_x + p_x = +H^3 Sdot/(8 pi^2): coefficient -1
s20 = sp.solve(sp.Eq(T_GH*S_geo_dot, flux + T_GH*Sd), Hd)[0]
rpx20 = sp.simplify((s20 - Hdot_GR)/(-4*sp.pi*G))
check("K20", "EGJ reading (ii): area grows by dS_info beyond the heat => rho_x+p_x = +H^3 Sdot/(8pi^2) (w>-1): coefficient -1",
      z(rpx20 - Hs**3*Sd/(8*sp.pi**2)), rpx20)

# K21 EGJ's d_iS is shear-sourced; the null congruence normal to round spheres in flat FRW is shear-free:
#     K_AB = (1/2) L_l h_AB with l = d_t + (1/a) d_r, h_AB = a^2 r^2 Omega_AB  =>  K_AB proportional to h_AB, trace-free part 0
th, ph = sp.symbols('theta phi', positive=True)
hAB = sp.diag((af*rr)**2, (af*rr*sp.sin(th))**2)
lder = lambda F: sp.diff(F, t) + sp.diff(F, rr)/af
KAB = hAB.applyfunc(lambda e: lder(e)/2)
hinv = hAB.inv()
theta_l = sp.simplify((hinv*KAB).trace())
shear = sp.simplify(KAB - theta_l/2*hAB)
check("K21", "Flat FRW: outgoing null congruence orthogonal to round spheres has zero shear (EGJ d_iS ~ shear^2 = 0 in exact FRW)",
      shear == sp.zeros(2, 2), theta_l)

# K22 book's E(a) at a=1: per-e-fold increment equals the stock (d ln E/d ln a = 1/a = 1), so 'stock' and 'marginal' Omega_m/2 coincide only at a=1
Eb = sp.exp(1 - 1/a)
check("K22", "E(a)=exp(1-1/a): a dE/da = E/a, equal to E only at a = 1",
      z(a*sp.diff(Eb, a) - Eb/a) and sp.solve(sp.Eq(1/a, 1), a) == [1])

# ---------------------------------------------------------------- output
npass = sum(r_['pass'] for r_ in RES); nfail = len(RES) - npass
json.dump({"checks": RES, "passed": npass, "failed": nfail}, open("r13_bookings_checks_output.json", "w"), indent=1)
for r_ in RES:
    print(("PASS " if r_['pass'] else "FAIL ") + r_['id'] + "  " + (r_['detail'][:120] if r_['detail'] else ""))
print(f"passed {npass}, failed {nfail}")
