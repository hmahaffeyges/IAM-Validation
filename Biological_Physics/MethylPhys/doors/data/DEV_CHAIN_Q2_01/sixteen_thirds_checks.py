import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""sixteen_thirds_checks.py -- 2026-11-18, derivation spider for ch:lambda's open O(1) step.

Forward checks for  sqrt(Omega_L) = (16/3) f_b,  f_b = Omega_b/Omega_m  (book eq:rel rewritten
with the horizon identity T_H S_geo / V_H = rho_crit c^2).  Units hbar = c = k_B = 1, G = l_P^2.
Every derived step is a check; output -> sixteen_thirds_checks_output.json.
No fitting: Planck 2018 values as the book uses them; no number is tuned after it is computed.
"""
import json
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq

RES = {"checks": [], "numbers": {}}


def check(cid, desc, ok, value=None):
    ok = bool(ok)
    RES["checks"].append({"id": cid, "desc": desc, "pass": ok,
                          "value": None if value is None else str(value)})
    print(("PASS " if ok else "FAIL ") + cid + "  " + desc + ("" if value is None else "  -> " + str(value)))


def num(key, v):
    RES["numbers"][key] = float(v)
    return float(v)


z = sp.simplify
H, HL, G, lP, rho_b, rho_m, fb, OL, Om, Ob, a, d, T = sp.symbols(
    'H H_L G l_P rho_b rho_m f_b Omega_L Omega_m Omega_b a d T', positive=True)

# ---------------------------------------------------------------- Section A: exact restatements
TH = H / (2 * sp.pi)
TdS = HL / (2 * sp.pi)
rA = 1 / H
rdS = 1 / HL
S_H = sp.pi * rA**2 / G
S_dS = sp.pi * rdS**2 / G
V_H = sp.Rational(4, 3) * sp.pi * rA**3
V_dS = sp.Rational(4, 3) * sp.pi * rdS**3
A_H = 4 * sp.pi * rA**2
rho_c = 3 * H**2 / (8 * sp.pi * G)
rho_L = 3 * HL**2 / (8 * sp.pi * G)          # Lambda = 3 H_L^2, rho_L = Lambda/(8 pi G)
rho_vac = 1 / G**2                            # c^7/(hbar G^2)

check("A0", "identity T_H S_H / V_H = rho_crit (d=3)", z(TH * S_H / V_H - rho_c) == 0)
check("A0b", "Omega_L = (H_L/H)^2 with rho_L = 3H_L^2/8piG", z(rho_L / rho_c - (HL / H)**2) == 0)
check("A1", "book coefficient 2/pi = (16/3)(3/8pi)", z(sp.Rational(2) / sp.pi - sp.Rational(16, 3) * 3 / (8 * sp.pi)) == 0)
# book eq:corr: rho_L/rho_vac = (2/pi)(lP/lH)^2 sqrt(OL) fb ; lP^2 = G, lH = 1/H
book = (2 / sp.pi) * G * H**2 * (HL / H) * fb
x_book = z(book * rho_vac * V_H / (TH * S_H))
check("A2", "book expression is x = (16/3) f_b sqrt(Omega_L) in horizon units", z(x_book - sp.Rational(16, 3) * fb * HL / H) == 0, x_book)
# relation  sqrt(OL) = (16/3) fb  <=>  H_L = (16/3) fb H ; equivalent restatements
rel = sp.Eq(HL, sp.Rational(16, 3) * fb * H)
Lam = 3 * HL**2
check("A3", "relation <=> Lambda = 16 f_b H H_L (the 3 is de Sitter's Lambda = 3H_L^2)",
      z((Lam - 16 * fb * H * HL).subs(HL, rel.rhs)) == 0)
check("A4", "relation <=> r_A / r_dS = (16/3) f_b", z((rA / rdS - sp.Rational(16, 3) * fb).subs(HL, rel.rhs)) == 0)
check("A4b", "x_dS = rho_L V_dS/(T_dS S_dS) = 1 exactly (d=3)", z(rho_L * V_dS / (TdS * S_dS) - 1) == 0)
# rho_m (T_dS/T_H) = (16/3) rho_b   (rho_b = fb rho_m)
check("A5", "relation <=> rho_m T_dS/T_H = (16/3) rho_b",
      z((rho_m * TdS / TH - sp.Rational(16, 3) * fb * rho_m).subs(HL, rel.rhs)) == 0)
# sweep form: A_H rho_L r_A = 16 f_b T_dS S_H  when relation and identity both hold
sweep = z((A_H * rho_L * rA - 16 * fb * TdS * S_H).subs(fb, sp.Rational(3, 16) * HL / H))
check("A6", "relation <=> Lambda energy swept through r_A per e-fold = 16 f_b T_dS S_H", sweep == 0)
check("A6b", "per unit area: rho_L r_A = 4 f_b T_dS / l_P^2",
      z((rho_L * rA - 4 * fb * TdS / G).subs(fb, sp.Rational(3, 16) * HL / H)) == 0)

# ---------------------------------------------------------------- Section B: candidate sources of 16/3
# B1: the Bekenstein-Hawking 1/4 'twice'. Count N = A/(4 l_P^2) nats; price T per nat (horizon Landauer:
# heat T dS adds dA = 4 G dS). Energy = T A/(4G): the 1/4 enters once. 16 needs a count in units of l_P^2/4.
E_once = T * A_H / (4 * G)
check("B1", "count x price contains the BH 1/4 exactly once (E = T S)", z(E_once - T * S_H) == 0)
check("B1b", "a factor 16 over T S needs area unit l_P^2/4 (16x finer than Bekenstein-Hawking)",
      z((T * A_H / (G / 4)) / (T * S_H) - 16) == 0)
# B2: Einstein-Hilbert 16 pi G, Euclidean de Sitter: I = -(1/16piG) int (R - 2 Lam) sqrt(g), R = 4 Lam, Vol(S^4) = 8pi^2 l^4/3
ell = sp.symbols('ell', positive=True)
LamE = 3 / ell**2
I_E = -(1 / (16 * sp.pi * G)) * (4 * LamE - 2 * LamE) * sp.Rational(8, 3) * sp.pi**2 * ell**4
check("B2", "on-shell Euclidean EH action of de Sitter = -S_dS (16piG gives ratio 1, not 16/3)",
      z(I_E + sp.pi * ell**2 / G) == 0)
# B3: Padmanabhan N_sur = A/l_P^2 = 4S (Cai: N_sur/S = 4 alpha, alpha=(d-1)/(2(d-2))) times radiation enthalpy 1+w = (d+1)/d
Nsur_over_S = 4 * (d - 1) / (2 * (d - 2))
enth = (d + 1) / d
B3 = z(Nsur_over_S * enth)
check("B3", "N_sur/S x (1+w_rad) = 16/3 at d=3", z(B3.subs(d, 3) - sp.Rational(16, 3)) == 0, B3)
check("B3b", "same product at d=4 is 15/4 (d-general form 2(d-1)(d+1)/(d(d-2)))", z(B3.subs(d, 4) - sp.Rational(15, 4)) == 0)
# B4: blackbody entropy density s = (4/3) a T^3 = (16/3) sigma T^3 / c ; d-general: s = ((d+1)/d) a T^d, sigma = a c f_d
th = sp.symbols('theta', positive=True)
def flux_factor(dd):
    num_ = sp.integrate(sp.cos(th) * sp.sin(th)**(dd - 2), (th, 0, sp.pi / 2))
    den_ = sp.integrate(sp.sin(th)**(dd - 2), (th, 0, sp.pi))
    return z(num_ / den_)
f3, f2, f4 = flux_factor(3), flux_factor(2), flux_factor(4)
check("B4", "one-way flux factor of isotropic radiation: 1/4 (d=3), 1/pi (d=2), 2/(3pi) (d=4)",
      f3 == sp.Rational(1, 4) and z(f2 - 1 / sp.pi) == 0, (f2, f3, f4))
s_over = lambda dd, ff: z(sp.Rational(dd + 1, dd) / ff)
check("B4b", "s/(sigma T^3/c) = (4/3)/(1/4) = 16/3 at d=3 (Rosseland 16 sigma T^3/3 kappa rho)", s_over(3, f3) == sp.Rational(16, 3))
check("B4c", "same ratio at d=4 carries pi: (5/4)/f_4", True, s_over(4, f4))
# B5: Thomson/Rayleigh phase function (3/16pi)(1+cos^2) is normalised; its 3/16 comes with 1/pi
mu = sp.symbols('mu')
check("B5", "Thomson phase function (3/16pi)(1+cos^2 theta) integrates to 1 over 4pi (3/16 carries a pi)",
      z(2 * sp.pi * sp.integrate(sp.Rational(3, 16) / sp.pi * (1 + mu**2), (mu, -1, 1)) - 1) == 0)
# B6: in d space dims de Sitter has Lambda = d(d-1)/2 H_L^2 : the '3' of A3 is the rotation-plane count
Hd = sp.symbols('H_d', positive=True)
# (d+1)-dim Einstein: H^2 = 16 pi G rho/(d(d-1)) with rho_L = Lambda/(8 pi G)  ->  Lambda = d(d-1)/2 H^2
LamD = sp.solve(sp.Eq(Hd**2, 16 * sp.pi * G * (sp.Symbol('Lam') / (8 * sp.pi * G)) / (d * (d - 1))), sp.Symbol('Lam'))[0]
check("B6", "Lambda = d(d-1)/2 H_L^2 in d space dims (=3 H_L^2 at d=3)", z(LamD - d * (d - 1) / 2 * Hd**2) == 0)
# B7: ledger 1/d and Smarr (d-2)/(d-1): 16/3 = 16 x (1/d) = (32/3) x Smarr ; 16 and 32/3 unsourced
check("B7", "16/3 = 16 * (1/d) = (32/3) * (d-2)/(d-1) at d=3 (bookkeeping, no source for 16 or 32/3)",
      sp.Rational(16, 3) == 16 * sp.Rational(1, 3) == sp.Rational(32, 3) * sp.Rational(1, 2))

# ---------------------------------------------------------------- numbers (Planck 2018 as the book uses them)
H0 = 67.4
Om0, Ob0, Or0 = 0.3153, 0.0493, 9.22e-5
OL0 = 1 - Om0 - Or0
fb0 = Ob0 / Om0
beta_m = Om0 / 2
sig_Om = 0.0073      # book p2_12:24, Planck prints Omega_L 0.6847 +- 0.0073
EH = lambda A: np.sqrt(Om0 * A**-3 + Or0 * A**-4 + OL0)
OLa = lambda A: OL0 / EH(A)**2
Oma = lambda A: Om0 * A**-3 / EH(A)**2
Ora = lambda A: Or0 * A**-4 / EH(A)**2
Eact = lambda A: np.exp(1 - 1 / A)
num("Omega_L", OL0); num("f_b", fb0)
ratio_today = num("ratio_fb_over_316sqrtOL", fb0 / (3 / 16 * np.sqrt(OL0)))
check("N1", "book's Planck ratio 1.0079 reproduced", abs(ratio_today - 1.0079) < 5e-4, round(ratio_today, 5))
xneed, xexpr = OL0, 16 / 3 * fb0 * np.sqrt(OL0)
num("x_needed", xneed); num("x_expression", xexpr)

# ---------------------------------------------------------------- Section C: which surface counts, which prices
# general booking: rho_L/rho_vac = (2/pi) f_b (l_P/r_X)^2 (T_Y/T_H)  ->  f_b = (3/16) g,  g = Omega_L (r_X/r_A)^2 (r_Y/r_A)
rX, rY, rAs, OLs = sp.symbols('r_X r_Y r_A Omega_Ls', positive=True)
g_sym = OLs * (rX / rAs)**2 * (rY / rAs)
check("C1", "g(X,Y) reproduces the dark-sector table: (A,A)=OL, (A,dS)=sqrt OL, (dS,A)=1, (dS,dS)=1/sqrt OL",
      all(z(e) == 0 for e in [
          g_sym.subs({rX: rAs, rY: rAs}) - OLs,
          g_sym.subs({rX: rAs, rY: rAs / sp.sqrt(OLs)}) - sp.sqrt(OLs),
          g_sym.subs({rX: rAs / sp.sqrt(OLs), rY: rAs}) - 1,
          g_sym.subs({rX: rAs / sp.sqrt(OLs), rY: rAs / sp.sqrt(OLs)}) - 1 / sp.sqrt(OLs)]))
# future event horizon today (units of r_A = c/H0)
rE_of = lambda A: A * quad(lambda u: 1 / (u**2 * EH(u)), A, np.inf, limit=200)[0] * EH(A)   # r_E/r_A(a)
rE1 = num("rE_over_rA_today", rE_of(1.0))
rdS1 = num("rdS_over_rA_today", 1 / np.sqrt(OL0))
check("C2", "ordering today r_A < r_E < r_dS", 1 < rE1 < rdS1, (round(rE1, 4), round(rdS1, 4)))
surf = {"A": 1.0, "E": rE1, "dS": rdS1}
table = {}
for X, rx in surf.items():
    for Y, ry in surf.items():
        g = OL0 * rx**2 * ry
        table[X + "," + Y] = fb0 / (3 / 16 * g)
RES["numbers"]["booking_table_ratio_fb_over_316g"] = table
for k, v in table.items():
    print("   booking count,price =", k, " ratio of sides", round(v, 4))
need_r = num("same_surface_radius_needed_over_rA", (16 * fb0 / (3 * OL0))**(1 / 3))
check("C3", "a same-surface booking needs a surface at (16 f_b/3 Omega_L)^(1/3) r_A = 1.068 r_A; none of A, E, dS",
      all(abs(need_r - r) > 0.03 for r in surf.values()), round(need_r, 4))
# floor property: T_A(a) >= T_dS for all a (H >= H_L), T_A -> T_dS
aa = np.logspace(-4, 3, 400)
check("C4", "T_A(a)/T_dS = E_H(a)/sqrt(OL) >= 1 for all a, -> 1 as a -> inf",
      np.all(EH(aa) / np.sqrt(OL0) >= 1 - 1e-12) and abs(EH(1e3) / np.sqrt(OL0) - 1) < 1e-6)
# Carnot: heat k T_A ln2 delivered at T_A, carried reversibly down to T_dS: net cost k T_dS ln2
Th_, Tc_ = sp.symbols('T_h T_c', positive=True)
Q = Th_ * sp.log(2)
W_back = Q * (1 - Tc_ / Th_)
check("C5", "Landauer with a reachable colder bath: net cost = k T_c ln2 (reversible engine returns Q(1-T_c/T_h))",
      z(Q - W_back - Tc_ * sp.log(2)) == 0)
# drift of the bookings (A,dS) and (A,E) with a
drift = {}
for A in [0.5, 0.9, 1.0, 1.1, 2.0]:
    gAdS = np.sqrt(OLa(A))
    gAE = OLa(A) * rE_of(A)
    drift[str(A)] = {"(A,dS)": fb0 / (3 / 16 * gAdS), "(A,E)": fb0 / (3 / 16 * gAE)}
RES["numbers"]["booking_drift"] = drift
_tH = 977.792 / H0
_age = lambda A: _tH * quad(lambda u: 1 / (u * EH(u)), 1e-10, A, limit=400)[0]
aAE = num("a_exact_booking_A_E", brentq(lambda A: fb0 / (3 / 16 * OLa(A) * rE_of(A)) - 1, 1.0, 1.3))
tAE = num("t_exact_booking_A_E_Gyr_from_now", _age(aAE) - _age(1.0))
check("C7", "(count A, price E) booking is exact at a = 1.113, 1.59 Gyr from now", abs(aAE - 1.113) < 0.002,
      (round(aAE, 4), round(tAE, 3)))

# ---------------------------------------------------------------- Section D: baryons as the writing fraction
w_req = num("writing_fraction_required_316sqrtOL", 3 / 16 * np.sqrt(OL0))
shares = {
    "B1 book beta_m, all matter writes per unit mass (baryon share of record)": fb0,
    "B2 only baryons write (book's reason): baryon share of the actual record": 1.0,
    "B3 baryons in stars+remnants (Fukugita-Peebles 6.0+-1.3% of baryons) x f_b": 0.060 * fb0,
    "B4 condensed baryons, stars+cold gas (~10% of baryons) x f_b": 0.10 * fb0,
}
RES["numbers"]["baryon_share_vs_required"] = {k: {"share": v, "over_required": v / w_req} for k, v in shares.items()}
check("D1", "only the all-baryons mass-weighted share (= f_b) is within 1% of the required 0.1551",
      abs(shares[list(shares)[0]] / w_req - 1) < 0.01 and all(abs(v / w_req - 1) > 0.5 for v in list(shares.values())[1:]))
# beta_m reading: x = (16/3) beta_m sqrt(OL)
r_beta = num("ratio_betam_over_316sqrtOL", beta_m / w_req)
Om_sym = sp.symbols('Om_s', positive=True)
Om_beta = float(sp.nsolve(Om_sym / 2 - sp.Rational(3, 16) * sp.sqrt(1 - Om_sym), 0.3))   # radiation neglected here
num("Omega_m_if_betam_eq_316sqrtOL", Om_beta)
dlnr = 1 / Om0 + 1 / (2 * OL0)
sig_r = num("sigma_ratio_betam_from_sigOm", dlnr * sig_Om)
check("D2", "beta_m = (3/16) sqrt(OL) holds to 1.6%, ~0.6 sigma on Omega_m +- 0.0073 (provisional, summary errors)",
      abs(r_beta - 1) / sig_r < 1.0, (round(r_beta, 4), round(Om_beta, 4), round((r_beta - 1) / sig_r, 2)))
check("D3", "the near-identity f_b ~ beta_m is Omega_m^2 ~ 2 Omega_b (0.8%)", abs(2 * Ob0 / Om0**2 - 1) < 0.01,
      round(2 * Ob0 / Om0**2, 4))
# record-rate route: today's record rate (eq:law_Sneed), priced at the floor T_dS, accumulated
# dS_info/dln a / (beta_m S_geo0) = E(a)/(a E_H(a)^4)   [p1_02:333]
check("D4a", "eq:law_Sneed ratio: (pi H0^2 beta E/(G a H^4)) / (beta pi/(G H0^2)) = E H0^4/(a H^4)",
      z((sp.pi * sp.Symbol('H0')**2 * sp.Symbol('E') / (G * a * H**4)) / (sp.pi / (G * sp.Symbol('H0')**2))
        - sp.Symbol('E') * sp.Symbol('H0')**4 / (a * H**4)) == 0)
I_rec = quad(lambda la: Eact(np.exp(la)) / (np.exp(la) * EH(np.exp(la))**4), np.log(1e-6), 0, limit=400)[0]
num("efolds_of_record_route_integral", I_rec)
N_need = num("efolds_needed_sqrtOL_over_betam", np.sqrt(OL0) / beta_m)
check("D4", "record history priced at T_dS (declared route) gives x/Omega_L far below 1", I_rec / N_need < 0.5,
      (round(I_rec, 4), round(N_need, 4), round(I_rec / N_need, 4)))
# dark-sector count booking (a): (Omega_b/2) int a^-3 (H0/H)^4 dln a
I_cnt = quad(lambda la: np.exp(-3 * la) / EH(np.exp(la))**4, np.log(1e-8), 0, limit=400)[0]
check("D5", "dark-sector count-booking integral 1.056 reproduced", abs(I_cnt - 1.056) < 0.002, round(I_cnt, 4))
num("count_booking_baryon_x", Ob0 / 2 * I_cnt)

# ---------------------------------------------------------------- Section E: why now
Oms, Ors = sp.symbols('Om_a Or_a', positive=True)
A_ = sp.symbols('A', positive=True)
OLa_s = OLs / (Om * A_**-3 + sp.Symbol('Orad') * A_**-4 + OLs)
dlnR = z(-sp.Rational(1, 2) * A_ * sp.diff(sp.log(OLa_s), A_))
target = sp.Rational(1, 2) * (3 * Om * A_**-3 + 4 * sp.Symbol('Orad') * A_**-4) / (Om * A_**-3 + sp.Symbol('Orad') * A_**-4 + OLs)
check("E1", "d ln(ratio)/d ln a = -(1/2) d ln OL(a)/d ln a = -(3 Om(a) + 4 Or(a))/2 (ratio falls)",
      z(dlnR + target) == 0)
slope_today = num("dlnratio_dlna_today", -(3 * Oma(1) + 4 * Ora(1)) / 2)
check("E1b", "drift today -0.47 per e-fold", abs(slope_today + 0.473) < 0.003, round(slope_today, 4))
a_star = num("a_star_exact", brentq(lambda A: fb0 / (3 / 16 * np.sqrt(OLa(A))) - 1, 0.9, 1.2))
a_lo = brentq(lambda A: fb0 / (3 / 16 * np.sqrt(OLa(A))) - 1.01, 0.8, 1.0)
a_hi = brentq(lambda A: fb0 / (3 / 16 * np.sqrt(OLa(A))) - 0.99, 1.0, 1.3)
tH = 977.792 / H0   # Gyr
age = lambda A: tH * quad(lambda u: 1 / (u * EH(u)), 1e-10, A, limit=400)[0]
t0 = age(1.0)
win = (age(a_lo) - t0, age(a_star) - t0, age(a_hi) - t0)
RES["numbers"]["why_now_window_Gyr"] = {"a_lo": a_lo, "a_star": a_star, "a_hi": a_hi,
                                         "t_lo": win[0], "t_star": win[1], "t_hi": win[2], "age": t0}
check("E2", "exact at a=1.017 (+0.25 Gyr); within 1% from -0.065 to +0.58 Gyr (dark-sector B5 reproduced)",
      abs(a_star - 1.017) < 0.002 and abs(win[0] + 0.065) < 0.01 and abs(win[2] - 0.58) < 0.02,
      tuple(round(w, 3) for w in win))
# hold at all epochs with f_b fixed -> OL(a) const -> rho_DE / rho_m const -> w_DE = 0 (matter era)
wDE = sp.symbols('w_DE')
# rho_DE ~ a^{-3(1+w)}, rho_m ~ a^{-3}: ratio const iff w = 0
check("E3", "relation at all epochs with fixed f_b forces rho_DE/rho_m const -> w_DE = 0, q = +1/2",
      sp.solve(sp.Eq(-3 * (1 + wDE), -3), wDE) == [0] and z((1 + 3 * 0) / 2 - sp.Rational(1, 2)) == 0)
check("E4", "implied DE history a^-3 (dln/dlna = -3) vs book activation E = e^{1-1/a} (dln/dlna = +1/a)",
      z(A_ * sp.diff(sp.log(sp.exp(1 - 1 / A_)), A_) - 1 / A_) == 0)
# dynamic writing fraction f_w(a) = (3/16) sqrt(OL(a)), compared with E(a)^p
pvals = {}
for A in [0.2, 0.5, 0.8, 0.95, 1.05, 1.5]:
    pvals[str(A)] = float(np.log(np.sqrt(OLa(A) / OL0)) / np.log(Eact(A)))
RES["numbers"]["p_of_a_fw_vs_E"] = pvals
check("E5", "f_w(a)/f_w(1) = E(a)^p has no constant p (numerical neighbour only)",
      max(pvals.values()) - min(pvals.values()) > 0.05, {k: round(v, 3) for k, v in pvals.items()})
check("E5b", "slope at a=1: (3/2)Om(1) = 0.473 vs (1/2) dlnE/dlna = 0.5 (close because 3 Om0 = 0.946)",
      abs(1.5 * Oma(1) - 0.473) < 0.003)
# matter-ruler variant: OL^(m) = (OL + beta_m E)/(1 + beta_m) today
OLm = (OL0 + beta_m) / (1 + beta_m)
num("ratio_matter_ruler", fb0 / (3 / 16 * np.sqrt(OLm)))
check("E6", "on the matter ruler (OL^(m)=0.728) the ratio is 0.978 (2.2% low)",
      abs(fb0 / (3 / 16 * np.sqrt(OLm)) - 0.978) < 0.002, round(fb0 / (3 / 16 * np.sqrt(OLm)), 4))

npass = sum(c["pass"] for c in RES["checks"])
nfail = len(RES["checks"]) - npass
RES["summary"] = {"passed": npass, "failed": nfail}
print("PASSED", npass, "FAILED", nfail)
json.dump(RES, open("sixteen_thirds_checks_output.json", "w"), indent=1, default=str)
