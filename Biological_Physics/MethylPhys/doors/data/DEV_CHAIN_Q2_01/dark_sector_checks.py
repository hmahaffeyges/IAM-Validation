import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
#!/usr/bin/env python3
"""dark_sector_checks.py  (2026-11-18, derivation spider: dark sector)

Checks every DERIVED step of DARK_SECTOR_2026-11-18.md.
Section A: exact identities (sympy, symbolic).
Section B: numbers (numpy/scipy), book inputs, Planck 2018.
Section C: the history of Lambda and the w0-wa comparison (pre-declared readings, no fitting of IAM quantities).
Section D: the virial-partner reading as a forward calculation.

Every check is recorded with pass/fail; output -> dark_sector_checks_output.json.
Light: < 100 MB, a few seconds.
"""
import json, math
import sympy as sp
import numpy as np
from scipy import integrate, optimize

R = {"checks": [], "numbers": {}}
def chk(cid, desc, ok, val=None):
    R["checks"].append({"id": cid, "desc": desc, "pass": bool(ok), "value": val})
def num(k, v):
    R["numbers"][k] = v
    return v

# ------------------------------------------------------------------ A. identities
H, G, hb, c, kB, a, Om, OL, Ob, H0 = sp.symbols('H G hbar c k_B a Omega_m Omega_L Omega_b H_0', positive=True)
lP2 = hb*G/c**3
TH = hb*H/(2*sp.pi*kB)                        # Gibbons-Hawking
Sgeo = kB*4*sp.pi*(c/H)**2/(4*lP2)            # Bekenstein-Hawking of the apparent (Hubble) horizon, flat FRW
VH = sp.Rational(4,3)*sp.pi*(c/H)**3
rho_crit_c2 = 3*H**2*c**2/(8*sp.pi*G)
rho_vac = c**7/(hb*G**2)

# A1 T_H S_geo / V_H = rho_crit c^2
chk("A1", "T_H S_geo / V_H = 3 H^2 c^2/(8 pi G) = rho_crit c^2 (any H)",
    sp.simplify(TH*Sgeo/VH - rho_crit_c2) == 0)
# A1b T_H S_geo = c^5/(2GH) (book eq:lam_smarr)
chk("A1b", "T_H S_geo = c^5/(2 G H) = M_H c^2", sp.simplify(TH*Sgeo - c**5/(2*G*H)) == 0)
# A2 the identity
lH = c/H
chk("A2", "rho_Lambda/rho_vac = Omega_L * (T_H S_geo/V_H)/rho_vac = (3 Omega_L/8pi)(l_P/l_H)^2",
    sp.simplify(OL*TH*Sgeo/VH/rho_vac - 3*OL/(8*sp.pi)*lP2/lH**2) == 0)
# A3 factorisation of 3/(8 pi) in Planck units: T (1/2pi) x S (pi) x 1/V (3/4pi)
chk("A3", "3/(8pi) = (1/2pi)*(pi)*(3/(4pi))",
    sp.simplify(sp.Rational(1,1)/(2*sp.pi)*sp.pi*3/(4*sp.pi) - 3/(8*sp.pi)) == 0)
# A4 record rate today beta_m S_geo = (1/2) M_m c^2/(k T_H)
beta = Om/2
Mm_c2 = Om*rho_crit_c2*VH
chk("A4", "beta_m S_geo/k_B = (1/2) M_m c^2/(k_B T_H), M_m = Omega_m rho_crit V_H",
    sp.simplify(beta*Sgeo/kB - sp.Rational(1,2)*Mm_c2/(kB*TH)) == 0)
# A5 dS_geo/dln a = 3 Omega_m(a) S_geo for flat matter+Lambda; record/geo ratio = E a^2/6
Ha2 = H0**2*(Om*a**-3 + OL)
Sg_a = sp.pi*c**5/(hb*G*Ha2)           # nats
dSg = sp.simplify(a*sp.diff(Sg_a, a))
Om_a = Om*a**-3*H0**2/Ha2
chk("A5", "dS_geo/dln a = 3 Omega_m(a) S_geo(a)", sp.simplify(dSg - 3*Om_a*Sg_a) == 0)
Ea = sp.exp(1 - 1/a)
dSinfo = sp.pi*c**5/(hb*G)*beta*H0**2*Ea/(a*Ha2**2)      # book eq:law_Sneed per e-fold
ratio_rec_geo = sp.simplify(dSinfo/dSg)
chk("A6", "(dS_info/dln a)/(dS_geo/dln a) = E(a) a^2/6 ; = 1/6 at a=1",
    sp.simplify(ratio_rec_geo - Ea*a**2/6) == 0 and sp.simplify(ratio_rec_geo.subs(a, 1) - sp.Rational(1,6)) == 0)
# A7 T_dS/T_H = sqrt(Omega_L) with H_L = H0 sqrt(Omega_L)
HL = H0*sp.sqrt(OL)
chk("A7", "T_dS/T_H0 = sqrt(Omega_L)", sp.simplify(TH.subs(H, HL)/TH.subs(H, H0) - sp.sqrt(OL)) == 0)
# A8 horizon Landauer equality: dE = k T ln2 with T = hbar kappa/(2 pi k c), dE = kappa c^2 dA/(8 pi G) -> dA = 4 ln2 l_P^2
kap, dA = sp.symbols('kappa dA', positive=True)
sol = sp.solve(sp.Eq(kap*c**2*dA/(8*sp.pi*G), kB*(hb*kap/(2*sp.pi*kB*c))*sp.log(2)), dA)[0]
chk("A8", "horizon Landauer equality: dA = 4 ln2 l_P^2 for every kappa", sp.simplify(sol - 4*sp.log(2)*lP2) == 0)
# A9 ln2 cancels: N_bits * kT ln2 = N_nats * kT
Nn, T = sp.symbols('N_nats T', positive=True)
chk("A9", "N_bits kT ln2 = N_nats kT (bit/nat convention carries no O(1) factor)",
    sp.simplify((Nn/sp.log(2))*kB*T*sp.log(2) - Nn*kB*T) == 0)
# A10 de Sitter self-identity rho_L V_dS = T_dS S_dS
rhoL_c2 = OL*3*H0**2*c**2/(8*sp.pi*G)
chk("A10", "rho_Lambda c^2 V_dS = T_dS S_dS exactly (de Sitter surface 'fully written', x_dS = 1)",
    sp.simplify(rhoL_c2*VH.subs(H, HL) - (TH*Sgeo).subs(H, HL)) == 0)
# A11 the expression <-> x = (16/3) f_b sqrt(Omega_L);  2/pi = (16/3)(3/8pi)
fb = sp.symbols('f_b', positive=True)
expr = 2/sp.pi*lP2/lH**2*sp.sqrt(OL)*fb
x_expr = sp.simplify(expr/(3/(8*sp.pi)*lP2/lH**2))
chk("A11", "expression / [(3/8pi)(l_P/l_H)^2] = (16/3) f_b sqrt(Omega_L); 2/pi = (16/3)(3/(8 pi))",
    sp.simplify(x_expr - sp.Rational(16,3)*fb*sp.sqrt(OL)) == 0 and sp.simplify(2/sp.pi - sp.Rational(16,3)*3/(8*sp.pi)) == 0)
# A12 relation from expression = identity
solfb = sp.solve(sp.Eq(expr, 3*OL/(8*sp.pi)*lP2/lH**2), fb)[0]
chk("A12", "expression = identity  =>  f_b = (3/16) sqrt(Omega_L)", sp.simplify(solfb - sp.Rational(3,16)*sp.sqrt(OL)) == 0)
# A13 surface assignment: K f_b (l_P/l_X)^2 (T_Y/T_H) = identity -> f_b = (3/16) Omega_L^(1-p)
res = {}
for X in ("H", "dS"):
    for Y in ("H", "dS"):
        lX2 = lH**2 if X == "H" else (c/(H*sp.sqrt(OL)))**2
        TY = 1 if Y == "H" else sp.sqrt(OL)
        e = 2/sp.pi*lP2/lX2*TY*fb
        s = sp.simplify(sp.solve(sp.Eq(e, 3*OL/(8*sp.pi)*lP2/lH**2), fb)[0]/sp.Rational(3,16))
        res[X+","+Y] = str(s)
num("A13_surface_assignment_fb_over_3_16", res)
chk("A13", "count normalisation X / price Y: (H,H)->Omega_L, (H,dS)->Omega_L^(1/2), (dS,H)->1, (dS,dS)->Omega_L^(-1/2)",
    res == {"H,H": "Omega_L", "H,dS": "sqrt(Omega_L)", "dS,H": "1", "dS,dS": "1/sqrt(Omega_L)"}, res)
# A14 relation at every epoch => Omega_L(a) constant => w_DE = 0 (flat, matter + DE)
#   Omega_L(a) = const => rho_DE/rho_m = const => d ln rho_DE/d ln a = -3 => w = -1 - (1/3)(-3) = 0
w_from = -1 - sp.Rational(1,3)*(-3)
chk("A14", "if f_b=(3/16)sqrt(Omega_L(a)) held at every a: Omega_L(a)=const, rho_DE ~ a^-3, w_DE = 0, q = +1/2 (no acceleration)",
    w_from == 0 and sp.Rational(1,2)*(1+3*0) == sp.Rational(1,2))
# A15 d ln(f_b/((3/16)sqrt(Omega_L(a))))/d ln a = -(3/2) Omega_m(a)
OL_a = OL*H0**2/Ha2
dl = sp.simplify(a*sp.diff(sp.log(1/sp.sqrt(OL_a)), a))
chk("A15", "d ln[ratio of the two sides]/d ln a = -(3/2) Omega_m(a)", sp.simplify(dl + sp.Rational(3,2)*Om_a) == 0)
# A16 history integral in the book's own variable: d/dln a [rho_acc] = T_H (dS_info/dln a)/V_H  ->  rho_acc = beta_m E(a) rho_crit0 c^2
Hs = sp.sqrt(Ha2)
integrand = (hb*Hs/(2*sp.pi))*dSinfo/(sp.Rational(4,3)*sp.pi*(c/Hs)**3)   # energy density per e-fold (J m^-3)
target = beta*Ea*3*H0**2*c**2/(8*sp.pi*G)
chk("A16", "T_H(a) (dS_info/dln a)/V_H(a) = d/dln a [beta_m E(a) rho_crit0 c^2]: the book's rule accumulates exactly to beta_m E(a) rho_crit0",
    sp.simplify(integrand - a*sp.diff(target, a)) == 0)
# A17 Padmanabhan: N_sur = 4 S_geo;  N_bulk = 2|rho+3p| V/(kT_H) ;  N_bulk/N_sur = Omega_L - Omega_m/2 = -q (flat, matter+Lambda, a=1)
Nsur = 4*Sgeo/kB
Nbulk = 2*(2*OL - Om)*rho_crit_c2*VH/(kB*TH)       # accelerating branch: -(rho+3p) = (2 Omega_L - Omega_m) rho_crit
q0 = Om/2 - OL
chk("A17", "Padmanabhan holographic equipartition: N_bulk/N_sur = Omega_L - Omega_m/2 = -q0 = Omega_L - beta_m",
    sp.simplify(Nbulk/Nsur - (OL - Om/2)) == 0 and sp.simplify((OL - Om/2) + q0) == 0)
# A18 virial balance 2K+V with K = beta_m E rho_c0, V = -Omega_m a^-3 rho_c0 : zero only at a = 1
f = 1 - 1/a + 3*sp.log(a)
chk("A18", "2K+V = Omega_m(E - a^-3) rho_c0 vanishes only at a=1 (f=1-1/a+3ln a, f'>0)",
    sp.simplify(f.subs(a, 1)) == 0 and sp.simplify(sp.diff(f, a) - (1/a**2 + 3/a)) == 0)
# A19 residual (2K+V)/|V| = E a^3 - 1 ; d/dln a at a=1 = 4
resid = Ea*a**3 - 1
chk("A19", "virial residual (2K+V)/|V| = E(a)a^3 - 1, slope d/dln a = 4 at a = 1",
    sp.simplify(a*sp.diff(resid, a)).subs(a, 1) == 4)
# A20 matter-sector rate today H_m^2 = H0^2(1+beta_m); q_m0 = -1 - (1/2)(-3 Om + beta_m)/(1+beta_m)
Hm2 = H0**2*(Om*a**-3 + OL + beta*Ea)
qm = -1 - a*sp.diff(Hm2, a)/(2*Hm2)
qm0 = sp.simplify(qm.subs(a, 1).subs(OL, 1 - Om))
chk("A20", "matter-sector q_m0 = -1 + (3 Om - beta_m)/(2(1+beta_m)) (flat, Omega_r -> 0)",
    sp.simplify(qm0 - (-1 + (3*Om - Om/2)/(2*(1 + Om/2)))) == 0, str(qm0))

# ------------------------------------------------------------------ B. numbers (book inputs)
cc, GG, hh, kk = 2.99792458e8, 6.67430e-11, 1.054571817e-34, 1.380649e-23
Mpc = 3.0856775814913673e22
H0p = 67.4e3/Mpc
Om_p, Ob_p, Or_p = 0.3153, 0.0493, 9.22e-5
OL_p = 1 - Om_p - Or_p          # 0.6846 (book ch:lambda)
fb_p = Ob_p/Om_p
lP = math.sqrt(hh*GG/cc**3); lHn = cc/H0p
ident = 3*OL_p/(8*math.pi)*(lP/lHn)**2
rho_v = cc**7/(hh*GG**2); rhoL = OL_p*3*H0p**2/(8*math.pi*GG)*cc**2
chk("B1", "rho_L/rho_vac = 1.133e-123 and identity equal", abs(rhoL/rho_v/ident - 1) < 1e-12 and abs(ident/1.133e-123 - 1) < 2e-3, ident)
e0 = 2/math.pi*(lP/lHn)**2*fb_p; e1 = e0*math.sqrt(OL_p)
chk("B2", "expression 1.380e-123, with sqrt(OL) 1.142e-123, ratios 1.218, 1.0079",
    abs(e0/1.380e-123-1) < 2e-3 and abs(e1/1.142e-123-1) < 2e-3 and abs(e0/ident-1.218) < 2e-3 and abs(e1/ident-1.0079) < 5e-4,
    [e0, e1, e0/ident, e1/ident])
x_need = num("B3_x_needed_Omega_L", OL_p)
x_expr_n = num("B3_x_expression_16_3_fb_sqrtOL", 16/3*fb_p*math.sqrt(OL_p))
num("B3_sixteen_thirds_fb", 16/3*fb_p)
# B4 surface table numerically
tab = {"H,H": fb_p/(3/16*OL_p), "H,dS": fb_p/(3/16*math.sqrt(OL_p)), "dS,H": fb_p/(3/16), "dS,dS": fb_p/(3/16/math.sqrt(OL_p))}
num("B4_ratio_of_sides_by_surface_assignment", tab)
chk("B4", "only the mixed (count on apparent horizon, price at T_dS) assignment lands within 1%",
    abs(tab["H,dS"]-1) < 0.01 and all(abs(v-1) > 0.1 for k, v in tab.items() if k != "H,dS"), tab)
# B5 why-now window (flat LCDM with radiation, Planck book values)
def OLa(aa): return OL_p/(Om_p*aa**-3 + Or_p*aa**-4 + OL_p)
def ratio_sides(aa, fb=fb_p): return fb/(3/16*math.sqrt(OLa(aa)))
a_exact = optimize.brentq(lambda aa: ratio_sides(aa) - 1, 0.8, 1.3)
def t_of_a(aa):
    return integrate.quad(lambda x: 1/(x*H0p*math.sqrt(Om_p*x**-3 + Or_p*x**-4 + OL_p)), 1e-8, aa, limit=200)[0]/(3.15576e16)
a_p1 = optimize.brentq(lambda aa: ratio_sides(aa) - 1.01, 0.8, 1.3)
a_m1 = optimize.brentq(lambda aa: ratio_sides(aa) - 0.99, 0.8, 1.3)
num("B5_a_where_relation_exact_planck", a_exact)
num("B5_t0_Gyr", t_of_a(1.0)); num("B5_t_exact_minus_t0_Gyr", t_of_a(a_exact) - t_of_a(1.0))
num("B5_window_1pct_Gyr", [t_of_a(a_p1) - t_of_a(1.0), t_of_a(a_m1) - t_of_a(1.0)])
num("B5_dlnratio_dlna_today", -1.5*Om_p/(Om_p + Or_p + OL_p))
num("B5_ratio_at_z", {str(z): ratio_sides(1/(1+z)) for z in (0, 0.1, 0.5, 1, 2)})
# 18th chain value 1.0046 +- 0.0068: time window over which the relation stays inside 1 sigma of unity
a_lo = optimize.brentq(lambda aa: ratio_sides(aa) - (1+0.0068), 0.8, 1.3)
a_hi = optimize.brentq(lambda aa: ratio_sides(aa) - (1-0.0068), 0.8, 1.3)
num("B5_window_chain_1sigma_Gyr", [t_of_a(a_lo) - t_of_a(1.0), t_of_a(a_hi) - t_of_a(1.0)])
chk("B5", "relation holds to 1% only within about +-0.3 Gyr of a=1.017", 0.15 < (t_of_a(a_m1) - t_of_a(a_p1))/2 < 0.5,
    [a_exact, t_of_a(a_m1) - t_of_a(a_p1)])

# B6 accumulation readings (pre-declared): Lambda_acc / Lambda_obs
def Hn(aa): return math.sqrt(Om_p*aa**-3 + Or_p*aa**-4 + OL_p)     # H/H0
# (i) book rule (A16): rho_acc(1) = beta_m rho_c0 ; baryon-only = Omega_b/2
num("B6_book_rule_all_matter_over_OmegaL", (Om_p/2)/OL_p)
num("B6_book_rule_baryons_over_OmegaL", (Ob_p/2)/OL_p)
num("B6_K_book_rule_vs_required", [(Om_p/2)*3/(8*math.pi)/fb_p, OL_p*3/(8*math.pi)/fb_p])
# (ii) count booking, baryons, no activation: S_w/S_geo0 = (Omega_b/2) * int a^-3 (H0/H)^4 dln a
J = integrate.quad(lambda aa: aa**-3*Hn(aa)**-4/aa, 1e-10, 1, limit=400, points=[1e-4, 1e-3, 0.1])[0]
num("B6_J_count_integral", J)
x_count = Ob_p/2*J
num("B6_count_booking_x_and_need", {"x": x_count, "need_price_TH": OL_p, "need_price_TdS": math.sqrt(OL_p)})
# (iii) energy booking (each e-fold deposits (1/2) rho_b(a) c^2, no dilution): int (1/2)Omega_b a^-3 dln a from a_EW
for aEW in (2.3e-15, 7.8e-16):
    num(f"B6_energy_booking_from_aEW_{aEW:g}", (Ob_p/2)*(aEW**-3 - 1)/3/OL_p)
# (iv) numerical neighbour (NOT derived): N_eff such that N (Omega_b/2) sqrt(OL) = OL
N_need = 2*math.sqrt(OL_p)/Ob_p
num("B6_neighbour_N_needed", N_need)
num("B6_neighbour_ln_1_over_aEW", {"2.3e-15 (printed)": math.log(1/2.3e-15), "7.8e-16 (100 GeV)": math.log(1/7.8e-16),
                                   "4.9e-16 (159.5 GeV, scaled)": math.log(1/(7.8e-16*100/159.5))})
chk("B6", "book's own accumulation rule gives Lambda_acc/Lambda_obs = beta_m/Omega_L = 0.230 (not 1)", abs((Om_p/2)/OL_p - 0.2303) < 1e-3)

# ------------------------------------------------------------------ C. Lambda(a) readings and DESI DR2 (CPL)
# Published DESI DR2 (arXiv:2503.14738, eqs. 25-27; DESI+CMB as in book table): w0, sig_w0, wa, sig_wa(avg)
DESI = {"DESI+CMB": (-0.42, 0.21, -1.75, 0.58),
        "DESI+CMB+Pantheon+": (-0.838, 0.055, -0.62, 0.205),
        "DESI+CMB+Union3": (-0.667, 0.088, -1.09, 0.29),
        "DESI+CMB+DESY5": (-0.752, 0.057, -0.86, 0.215)}
bm = Om_p/2
def Eact(aa): return math.exp(1 - 1/aa)
readings = {
    "M: matter-sector dark sector, Omega_L + beta_m E(a)": lambda aa: OL_p + bm*Eact(aa),
    "L1: whole Lambda accumulates, Omega_L E(a)": lambda aa: OL_p*Eact(aa),
    "L1b: baseline Omega_L-beta_m plus beta_m E(a)": lambda aa: (OL_p - bm) + bm*Eact(aa),
    "L2: book photon ruler, constant Lambda": lambda aa: OL_p,
}
def w_of(rho, aa, h=1e-5):
    return -1 - (math.log(rho(aa*(1+h))) - math.log(rho(aa*(1-h))))/(3*math.log((1+h)/(1-h)))
def cpl_valder(rho):
    w1 = w_of(rho, 1.0); dw = (w_of(rho, 1+1e-4) - w_of(rho, 1-1e-4))/2e-4
    return w1, -dw
def cpl_density_fit(rho, zmax=2.5):
    aa = np.linspace(1/(1+zmax), 1, 400)
    y = np.array([math.log(rho(x)/rho(1.0)) for x in aa])
    def model(p):
        w0, wa = p
        return -3*(1+w0+wa)*np.log(aa) - 3*wa*(1-aa)
    s = optimize.least_squares(lambda p: model(p) - y, [-1, 0])
    return float(s.x[0]), float(s.x[1]), float(np.max(np.abs(model(s.x) - y)))
cplres = {}
for name, rho in readings.items():
    vd = cpl_valder(rho); df = cpl_density_fit(rho)
    pulls = {k: [(vd[0]-v[0])/v[1], (df[0]-v[0])/v[1], (df[1]-v[2])/v[3]] for k, v in DESI.items()}
    cplres[name] = {"w_today": vd[0], "valder_w0_wa": vd, "densityfit_w0_wa_maxres": df,
                    "w_at_z": {str(z): w_of(rho, 1/(1+z)) for z in (0, 0.5, 1, 2)},
                    "pull_vs_DESI_[w0_valder,w0_fit,wa_fit]_sigma": pulls}
num("C1_CPL_by_reading", cplres)
wM = cplres["M: matter-sector dark sector, Omega_L + beta_m E(a)"]["w_today"]
chk("C1", "matter-sector combined w today = -1 - (beta_m/3)/(Omega_L+beta_m) = -1.062", abs(wM - (-1 - bm/3/(OL_p+bm))) < 1e-6, wM)
L1vd = cplres["L1: whole Lambda accumulates, Omega_L E(a)"]["valder_w0_wa"]
chk("C2", "L1 CPL image (value+derivative) = (-4/3, -1/3) (book eq:wz_cpl)", abs(L1vd[0]+4/3) < 1e-6 and abs(L1vd[1]+1/3) < 1e-4, L1vd)
fitM = cplres["M: matter-sector dark sector, Omega_L + beta_m E(a)"]["densityfit_w0_wa_maxres"]
chk("C3", "matter-sector dark sector sits at w0<-1, wa>0: opposite quadrant to every DESI DR2 fit", fitM[0] < -1 and fitM[1] > 0, fitM)

# C4 background test of L1 / L1b: H0 that keeps D_M(z*) (theta*) with omega_m, omega_b, omega_r fixed
om_m = 0.3153*0.6736**2; om_r = 9.22e-5*0.674**2; zstar = 1089.9
def DM(h, rhoDE):
    OLh = 1 - om_m/h**2 - om_r/h**2
    f = lambda z: 1/math.sqrt(om_m/h**2*(1+z)**3 + om_r/h**2*(1+z)**4 + OLh*rhoDE(1/(1+z), OLh))
    return integrate.quad(f, 0, zstar, limit=400, points=[1, 10, 100])[0]/(100*h)   # c=1 units of c/100
ref = DM(0.6736, lambda aa, O: 1.0)
sol_h = {}
for name, fn in {"L1": lambda aa, O: Eact(aa), "L1b": lambda aa, O: ((O - bm) + bm*Eact(aa))/O}.items():
    hsol = optimize.brentq(lambda h: DM(h, fn) - ref, 0.45, 0.9)
    sol_h[name] = 100*hsol
num("C4_H0_keeping_thetastar", sol_h)
# run 1 stated the direction wrongly ("below 67.36") and failed; kept in dark_sector_checks_output_run1.json.
# Corrected statement: dark energy that was smaller in the past lengthens D_M at fixed h, so theta* needs a HIGHER H0.
dm_fixed = DM(0.6736, lambda aa, O: Eact(aa))
chk("C4", "at fixed h, D_M(z*) of L1 exceeds LCDM's, so keeping theta* needs H0 above 67.36 (L1 and L1b)",
    dm_fixed > ref and sol_h["L1"] > 67.36 and sol_h["L1b"] > 67.36, [dm_fixed/ref, sol_h])
num("C4_Omegam_implied", {k: om_m/(v/100)**2 for k, v in sol_h.items()})

# ------------------------------------------------------------------ D. virial-partner reading, forward numbers
def Hm2n(aa): return Om_p*aa**-3 + Or_p*aa**-4 + OL_p + bm*Eact(aa)
def H2n(aa): return Om_p*aa**-3 + Or_p*aa**-4 + OL_p
def q_of(H2f, aa, h=1e-5):
    return -1 - (math.log(H2f(aa*(1+h))) - math.log(H2f(aa*(1-h))))/(2*math.log((1+h)/(1-h)))
qm0n = q_of(Hm2n, 1.0); q0n = q_of(H2n, 1.0)
zt_m = optimize.brentq(lambda z: q_of(Hm2n, 1/(1+z)), 0.1, 3.0)
zt_g = optimize.brentq(lambda z: q_of(H2n, 1/(1+z)), 0.1, 3.0)
num("D1_q0_photon_matter", [q0n, qm0n]); num("D1_z_transition_photon_matter", [zt_g, zt_m])
chk("D1", "photon z_t = 0.632 (book); matter-sector q_m0 and z_t^m computed", abs(zt_g - 0.632) < 0.005, [zt_g, zt_m, qm0n])
resid_n = {str(aa): Eact(aa)*aa**3 - 1 for aa in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)}
num("D2_virial_residual_Ea3_minus_1", resid_n)
H0_perGyr = H0p*3.15576e16
num("D2_residual_slope_per_Gyr_today", 4*H0_perGyr)
Rfut = {str(aa): Om_p*aa**-3/(bm*Eact(aa)) for aa in (0.5, 0.8, 1.0, 1.26, 1.5, 2.0, 5.0)}
num("D3_R_of_a", Rfut)
aR1 = optimize.brentq(lambda aa: Om_p*aa**-3/(bm*Eact(aa)) - 1, 1.0, 3.0)
num("D3_a_where_R_equals_1", aR1); num("D3_t_from_now_R_equals_1_Gyr", t_of_a(aR1) - t_of_a(1.0))
chk("D3", "R(a) = 2 a^-3/E(a): passes 2 at a=1 and keeps falling (R=1 at a ~ 1.2)", Rfut["1.0"] == 2.0 or abs(Rfut["1.0"]-2) < 1e-12, aR1)
num("D4_baseline_minus_q0", [OL_p - bm, -q0n])
num("D5_Omegam_matter_sector_today", Om_p/(1+bm))
num("D6_Odm_over_Ob_from_relation_and_measured", [16/(3*math.sqrt(OL_p)) - 1, (Om_p - Ob_p)/Ob_p])

R["summary"] = {"passed": sum(1 for x in R["checks"] if x["pass"]), "failed": sum(1 for x in R["checks"] if not x["pass"])}
json.dump(R, open("dark_sector_checks_output.json", "w"), indent=1, default=str)
print(json.dumps(R["summary"]))
for x in R["checks"]:
    if not x["pass"]:
        print("FAIL", x["id"], x["desc"], x["value"])
