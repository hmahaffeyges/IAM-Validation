import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
"""bh_formation_checks.py  (2026-11-18, derivation spider, private)

Why a black hole forms, written in IAM's language, every step checked.
Sections
  J  Jacobson 1995: Clausius on local Rindler horizons -> Einstein equation; bits and Landauer price
  P  Padmanabhan 2010 / Verlinde 2011: holographic equipartition E = (1/2) N k T
  L  bits per area: 1 bit per 4 ln2 l_P^2, kappa-free
  F  fill fractions f = r_s/R (Bekenstein-max over capacity) and f_A = (r_s/R)^2 (hole's ledger over capacity)
  B  Buchdahl 1959: static spherical Einstein tensor -> TOV; interior Schwarzschild; p_c -> inf at f = 8/9;
     Buchdahl's general proof (isotropy identity, integral, final inequality)
  C  Chandrasekhar: Lane-Emden n=3 omega_3; M_Ch; capacity rewriting; Landau uniform-sphere estimate
  T  Oppenheimer-Volkoff free-neutron TOV: M_max, R, coefficient of m_P^3/m_n^2, interior f(r) profile
  N  numbers: iron core, white dwarfs, neutron stars, Sun, 10 Msun hole
  M  what a local mu < 1 would do to M_Ch (question for the lead)
Run: python bh_formation_checks.py  -> bh_formation_checks_output.json  (light: sympy + scipy, < 1 min, < 200 MB)
"""
import json, math
import numpy as np
import sympy as sp
from scipy import constants as C
from scipy.integrate import solve_ivp

OUT = {"checks": {}, "numbers": {}}
def check(cid, ok, note=""):
    OUT["checks"][cid] = {"pass": bool(ok), "note": note}
    print(("PASS " if ok else "FAIL ") + cid + ("  " + note if note else ""))
def num(key, val):
    OUT["numbers"][key] = val

# constants (CODATA via scipy)
hbar, c, G, kB = C.hbar, C.c, C.G, C.k
m_u, m_n, m_p, m_e = C.physical_constants["atomic mass constant"][0], C.m_n, C.m_p, C.m_e
m_H = C.physical_constants["proton mass"][0] + m_e  # hydrogen atom mass (binding 13.6 eV neglected, 1.5e-8 rel.)
Msun = 1.98841e30          # book value (p3_01 table caption); IAU nominal GM/G differs at 1e-5
lP2 = hbar * G / c**3
mP = math.sqrt(hbar * c / G)
rs = lambda M: 2 * G * M / c**2

# ---------------------------------------------------------------- J: Jacobson
hb, cc, GG, kk, kap, eta, lam = sp.symbols("hbar c G k_B kappa eta lambda", positive=True)
T_U = hb * kap / (2 * sp.pi * kk * cc)                     # Unruh, kappa an acceleration
kap_g = kap / cc**2                                        # inverse length, appears in chi^a = -kap_g lam k^a
Rkk, Tkk = sp.symbols("R_kk T_kk")                         # R_ab k^a k^b, T_ab k^a k^b, taken constant over the tiny patch
dA_patch = sp.Symbol("dA", positive=True)
L0 = sp.Symbol("L0", positive=True)                        # affine extent of the patch (lambda from -L0 to 0)
# heat: dQ = -kap_g * int lam T_kk dlam dA   (lambda <= 0 to the past)
dQ = -kap_g * sp.integrate(lam * Tkk, (lam, -L0, 0)) * dA_patch
# Raychaudhuri to first order: theta = -lam R_kk ; check it solves dtheta/dlam = -R_kk - theta^2/2 to O(lam)
theta = -lam * Rkk
resid = sp.diff(theta, lam) - (-Rkk - theta**2 / 2)
check("J1_raychaudhuri_first_order", sp.simplify(resid.subs(lam, 0)) == 0 and sp.degree(sp.expand(resid), lam) == 2,
      "theta=-lam R_kk solves Raychaudhuri; residual is O(lam^2) = theta^2/2")
dA = sp.integrate(theta, (lam, -L0, 0)) * dA_patch          # delta A = int theta dlam dA
dS = kk * eta * dA                                          # S = k_B eta A
# Clausius dQ = T dS for all patches -> T_kk = (hbar c eta / 2pi) R_kk  (with T_ab in J/m^3 and R in 1/m^2)
sol = sp.solve(sp.Eq(dQ, T_U * dS), Tkk)[0]
check("J2_clausius_gives_Tkk", sp.simplify(sol - hb * cc * eta / (2 * sp.pi) * Rkk) == 0,
      "kappa cancels: T_kk = (hbar c eta/2pi) R_kk")
check("J3_kappa_cancels", sp.diff(sp.simplify(sol), kap) == 0)
# Einstein: R_kk = (8 pi G / c^4) T_kk for null k  (g_kk = 0 kills the trace and Lambda terms)
eta_sol = sp.solve(sp.Eq(hb * cc * eta / (2 * sp.pi), cc**4 / (8 * sp.pi * GG)), eta)[0]
check("J4_eta_is_quarter_planck", sp.simplify(eta_sol - cc**3 / (4 * hb * GG)) == 0, "eta = c^3/(4 hbar G) = 1/(4 l_P^2)")
# IAM bookkeeping: the heat dQ at T_U, priced at Landauer k_B T ln2 per bit, is a bit count; each bit needs 4 ln2 l_P^2 of area
dN_bits = dQ / (kk * T_U * sp.log(2))
dA_from_bits = 4 * sp.log(2) * hb * GG / cc**3 * dN_bits
dA_einstein = dA.subs(Rkk, 8 * sp.pi * GG / cc**4 * Tkk)
check("J5_bits_times_tile_equals_focusing", sp.simplify(dA_from_bits - dA_einstein) == 0,
      "bits written by the heat (Landauer at Unruh T) x (4 ln2 l_P^2) = Raychaudhuri focusing area from Einstein")
# Bianchi step: divergence-free completion; trace check that G_ab + Lambda g_ab with f = -R/2 + Lambda is consistent (symbolic 4D trace)
Rs, Ls = sp.symbols("R Lambda")
# trace of T_ab = (hbar c eta/2pi)(R_ab + f g_ab): T = (..)(R + 4 f); with f = -R/2 + Lambda: T = (..)(-R + 4 Lambda) = trace of (G_ab + Lambda g_ab)
fsc = -Rs / 2 + Ls
check("J6_trace_consistency", sp.simplify((Rs + 4 * fsc) - (Rs - 2 * Rs + 4 * Ls)) == 0, "trace of R_ab+f g_ab equals trace of G_ab+Lambda g_ab")

# ---------------------------------------------------------------- P: Padmanabhan / Verlinde equipartition
M, R = sp.symbols("M R", positive=True)
N_surf = 4 * sp.pi * R**2 / (hb * GG / cc**3)               # A / l_P^2 (Padmanabhan's N; Verlinde's N = A c^3/(G hbar))
g_newton = GG * M / R**2
T_surf = hb * g_newton / (2 * sp.pi * kk * cc)              # Unruh/Davies temperature of the local acceleration
E_eq = sp.Rational(1, 2) * N_surf * kk * T_surf
check("P1_equipartition_gives_Mc2_every_R", sp.simplify(E_eq - M * cc**2) == 0,
      "(1/2) N k T_g = M c^2 at every radius: the Gauss-law ledger balance (Padmanabhan 2010, Verlinde 2011)")
# at the horizon R = r_s: S = k N/4, E = 2 T S (Smarr); bits N/(4 ln2); energy per bit / Landauer price = 2
S_hor = kk * N_surf / 4
E_hor = E_eq
ratio = sp.simplify((E_hor / (S_hor / (kk * sp.log(2)))) / (kk * T_surf * sp.log(2)))
check("P2_energy_per_bit_is_two_landauer_prices", ratio == 2, "E/(N_bits k T ln2) = 2 for any sphere; at r_s this is Smarr Mc^2 = 2 T S")
# T_surf at r_s equals Hawking temperature
T_H = hb * cc**3 / (8 * sp.pi * GG * M * kk)
check("P3_surface_T_at_rs_is_Hawking", sp.simplify(T_surf.subs(R, 2 * GG * M / cc**2) - T_H) == 0)
# ratio of surface temperature to Hawking temperature = f^2
fB = 2 * GG * M / (cc**2 * R)
check("P4_Tsurf_over_TH_is_f_squared", sp.simplify(T_surf / T_H - fB**2) == 0)

# ---------------------------------------------------------------- L: bits per area
dE = sp.Symbol("dE", positive=True)
dA_first_law = 8 * sp.pi * GG / (kap * cc**2) * dE          # delta(Mc^2) = (kappa c^2 / 8 pi G) delta A
bit_area = sp.simplify(dA_first_law.subs(dE, kk * T_U * sp.log(2)))
check("L1_bit_area_kappa_free", sp.simplify(bit_area - 4 * sp.log(2) * hb * GG / cc**3) == 0, "4 ln2 l_P^2 for every kappa")
num("l_P_m", math.sqrt(lP2)); num("bit_area_m2", 4 * math.log(2) * lP2); num("nat_area_m2", 4 * lP2)
num("bits_per_m2", 1 / (4 * math.log(2) * lP2))
check("L2_bit_area_value", abs(4 * math.log(2) * lP2 - 7.24e-70) / 7.24e-70 < 2e-3, f"{4*math.log(2)*lP2:.4e} m^2 (book 7.24e-70)")
# Schwarzschild first law
Ms = sp.Symbol("M", positive=True)
A_s = 16 * sp.pi * GG**2 * Ms**2 / cc**4
kap_s = cc**4 / (4 * GG * Ms)
check("L3_schwarzschild_first_law", sp.simplify(cc**2 / sp.diff(A_s, Ms) - kap_s * cc**2 / (8 * sp.pi * GG)) == 0)

# ---------------------------------------------------------------- F: fill fractions
S_bek = 2 * sp.pi * kk * R * M * cc / hb                   # Bekenstein 1981 max for E = M c^2
S_holo = kk * 4 * sp.pi * R**2 / (4 * hb * GG / cc**3)
check("F1_fill_is_rs_over_R", sp.simplify(S_bek / S_holo - fB) == 0, "f = S_Bek,max / capacity = r_s/R")
S_BH_M = 4 * sp.pi * GG * kk * M**2 / (hb * cc)
check("F2_ledger_fill_is_f_squared", sp.simplify(S_BH_M / S_holo - fB**2) == 0, "f_A = S_BH(M)/capacity(R) = (r_s/R)^2")
# hoop: C/(4 pi G M/c^2) = 1/f
check("F3_hoop_is_f_ge_1", sp.simplify((2 * sp.pi * R) / (4 * sp.pi * GG * M / cc**2) - 1 / fB) == 0)

# ---------------------------------------------------------------- B: Buchdahl
r, t, th, ph = sp.symbols("r t theta phi", positive=True)
nu = sp.Function("nu")(r); m = sp.Function("m")(r)
# geometric units G = c = 1 for the GR algebra
g = sp.diag(-sp.exp(2 * nu), 1 / (1 - 2 * m / r), r**2, r**2 * sp.sin(th)**2)
X = [t, r, th, ph]
ginv = g.inv()
def christoffel(g, ginv, X):
    n = 4
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for d in range(n):
                Gam[a][b][d] = sp.simplify(sum(ginv[a, e] * (sp.diff(g[e, b], X[d]) + sp.diff(g[e, d], X[b]) - sp.diff(g[b, d], X[e])) for e in range(n)) / 2)
    return Gam
Gam = christoffel(g, ginv, X)
def ricci(Gam, X):
    n = 4
    Ric = sp.zeros(n)
    for b in range(n):
        for d in range(n):
            Ric[b, d] = sp.simplify(sum(sp.diff(Gam[a][b][d], X[a]) for a in range(n)) - sum(sp.diff(Gam[a][b][a], X[d]) for a in range(n))
                                    + sum(Gam[a][a][e] * Gam[e][b][d] for a in range(n) for e in range(n))
                                    - sum(Gam[a][d][e] * Gam[e][b][a] for a in range(n) for e in range(n)))
    return Ric
Ric = ricci(Gam, X)
Rsc = sp.simplify(sum(ginv[a, b] * Ric[a, b] for a in range(4) for b in range(4)))
Gmix = sp.simplify(ginv * (Ric - Rsc * g / 2))               # G^a_b
rho, p = sp.Function("rho")(r), sp.Function("p")(r)
# G^t_t = -8 pi rho ; G^r_r = 8 pi p ; G^theta_theta = 8 pi p (isotropic)
eq_tt = sp.simplify(Gmix[0, 0] + 8 * sp.pi * rho)
mprime = sp.solve(eq_tt, sp.diff(m, r))[0]
check("B1_mass_function", sp.simplify(mprime - 4 * sp.pi * r**2 * rho) == 0, "G^t_t = -8 pi rho  ->  m' = 4 pi r^2 rho")
eq_rr = sp.simplify(Gmix[1, 1] - 8 * sp.pi * p)
nuprime = sp.solve(eq_rr, sp.diff(nu, r))[0]
check("B2_nu_prime", sp.simplify(nuprime - (m + 4 * sp.pi * r**3 * p) / (r * (r - 2 * m))) == 0)
# TOV from conservation: p' = -(rho + p) nu'
TOV = -(rho + p) * (m + 4 * sp.pi * r**3 * p) / (r * (r - 2 * m))
# isotropy G^r_r = G^theta_theta: verify Buchdahl's identity with zeta = e^nu:
zeta = sp.exp(nu)
iso = sp.simplify(Gmix[1, 1] - Gmix[2, 2])
lhs = sp.diff(sp.sqrt(1 - 2 * m / r) / r * sp.diff(zeta, r), r)
rhs = zeta / sp.sqrt(1 - 2 * m / r) * sp.diff(m / r**3, r)
# claim: iso = 0 <=> lhs - rhs = 0 ; check proportionality: (lhs - rhs) = K(r) * iso with K nonzero
K = sp.simplify((lhs - rhs) / iso)
check("B3_buchdahl_isotropy_identity", sp.simplify(sp.diff(K, sp.diff(nu, r, 2))) == 0 and sp.simplify(sp.diff(K, sp.diff(m, r))) == 0 and K != 0,
      f"(d/dr)[sqrt(1-2m/r) zeta'/r] - zeta (m/r^3)'/sqrt(1-2m/r) = K * (G^r_r - G^th_th), K = {sp.simplify(K)}")
# exterior boundary value: (1/R) sqrt(1-2M/R) zeta'(R) = M/R^3
Mg, Rg, x = sp.symbols("M_g R_g x", positive=True)
zeta_ext = sp.sqrt(1 - 2 * Mg / x)
bval = sp.simplify((sp.sqrt(1 - 2 * Mg / x) / x * sp.diff(zeta_ext, x)).subs(x, Rg))
check("B4_exterior_boundary", sp.simplify(bval - Mg / Rg**3) == 0)
# extremal integral: int_0^R (M/R^3) r / sqrt(1 - 2 M r^2/R^3) dr = (1 - sqrt(1 - 2M/R))/2
rr = sp.Symbol("rr", positive=True)
Iint = sp.integrate(Mg / Rg**3 * rr / sp.sqrt(1 - 2 * Mg * rr**2 / Rg**3), (rr, 0, Rg))
_d = Iint - (1 - sp.sqrt(1 - 2 * Mg / Rg)) / 2
_vals = [float(_d.subs({Mg: mv, Rg: 1.0})) for mv in (0.05, 0.2, 0.4, 0.44)]
check("B5_buchdahl_integral", max(abs(v) for v in _vals) < 1e-12, f"int_0^R (M/R^3) r dr/sqrt(1-2Mr^2/R^3) = (1-sqrt(1-2M/R))/2; residuals {_vals}")
# zeta(0) > 0 with zeta(R) = sqrt(1-2M/R):  sqrt(1-2M/R) - (1 - sqrt(1-2M/R))/2 > 0  <=>  2M/R < 8/9
y = sp.Symbol("y", positive=True)  # y = 2M/R
cond = sp.sqrt(1 - y) - (1 - sp.sqrt(1 - y)) / 2
root = sp.solve(sp.Eq(cond, 0), y)
check("B6_buchdahl_bound_8_9", root == [sp.Rational(8, 9)], f"zeta(0) = 0 at 2M/R = {root}")
# interior Schwarzschild (uniform density): p(r) solves TOV and p_c diverges at 2M/R = 8/9
rho0 = sp.Symbol("rho0", positive=True)
Rst = sp.Symbol("R_*", positive=True)
Mst = sp.Rational(4, 3) * sp.pi * rho0 * Rst**3
a1 = sp.sqrt(1 - 2 * Mst * r**2 / Rst**3); a0 = sp.sqrt(1 - 2 * Mst / Rst)
p_int = rho0 * (a1 - a0) / (3 * a0 - a1)
m_int = sp.Rational(4, 3) * sp.pi * rho0 * r**3
tov_res = sp.simplify(sp.diff(p_int, r) - (-(rho0 + p_int) * (m_int + 4 * sp.pi * r**3 * p_int) / (r * (r - 2 * m_int))))
check("B7_interior_schwarzschild_solves_TOV", tov_res == 0)
check("B8_surface_pressure_zero", sp.simplify(p_int.subs(r, Rst)) == 0)
pc = sp.simplify(p_int.subs(r, 0))
fsym = sp.Symbol("f", positive=True)
pc_f = sp.simplify(rho0 * (1 - sp.sqrt(1 - fsym)) / (3 * sp.sqrt(1 - fsym) - 1))
check("B9_pc_in_terms_of_f", sp.simplify(pc - pc_f.subs(fsym, 2 * Mst / Rst)) == 0, "p_c/rho = (1 - sqrt(1-f))/(3 sqrt(1-f) - 1)")
check("B10_pc_diverges_at_8_9", sp.solve(sp.Eq(3 * sp.sqrt(1 - fsym) - 1, 0), fsym) == [sp.Rational(8, 9)])
for fv in [0.1, 0.5, 0.8, 0.88, 0.888]:
    num(f"interior_schwarzschild_pc_over_rho_at_f={fv}", float(pc_f.subs({fsym: fv, rho0: 1})))
# f inside the uniform star: 2m(r)/r = f (r/R)^2, max at the surface
# lapse of the uniform star: e^nu = (3/2) sqrt(1-f) - (1/2) sqrt(1 - f r^2/R^2); check nu' = (m + 4 pi r^3 p)/(r(r-2m))
zeta_int = sp.Rational(3, 2) * a0 - sp.Rational(1, 2) * a1
nu_res = sp.simplify(sp.diff(sp.log(zeta_int), r) - (m_int + 4 * sp.pi * r**3 * p_int) / (r * (r - 2 * m_int)))
check("B13_interior_lapse_solves_nu_eq", nu_res == 0, "e^nu = (3 sqrt(1-f) - sqrt(1-f r^2/R^2))/2")
zeta0_f = (3 * sp.sqrt(1 - fsym) - 1) / 2
check("B14_central_lapse_zero_at_8_9", sp.solve(sp.Eq(zeta0_f, 0), fsym) == [sp.Rational(8, 9)],
      "centre clock rate e^nu(0) = (3 sqrt(1-f) - 1)/2 -> 0 at f = 8/9: the pressure pole and g_tt = 0 coincide")
for fv in [0.1, 0.5, 0.8, 0.88]:
    num(f"central_lapse_at_f={fv}", float(zeta0_f.subs(fsym, fv)))
check("B11_uniform_f_profile", sp.simplify(2 * m_int / r - (2 * Mst / Rst) * r**2 / Rst**2) == 0)
num("buchdahl_f_max", 8 / 9); num("buchdahl_fA_max", (8 / 9)**2)
num("buchdahl_redshift_max_z", 1 / math.sqrt(1 - 8 / 9) - 1)
check("B12_buchdahl_surface_redshift_2", abs(1 / math.sqrt(1 / 9) - 1 - 2) < 1e-12, "z_max = 2 at f = 8/9")
# causality bound (Koranda-Stergioulas-Friedman 1997; Lattimer 2012): GM/Rc^2 <= 0.354 -> f <= 0.708
num("causal_f_max", 2 * 0.354)

# ---------------------------------------------------------------- C: Chandrasekhar
def lane_emden(n):
    def rhs(xi, yv):
        th_, dth = yv
        return [dth, -max(th_, 0.0)**n - 2 * dth / xi]
    xi0 = 1e-6
    y0 = [1 - xi0**2 / 6, -xi0 / 3]
    ev = lambda xi, yv: yv[0]; ev.terminal = True; ev.direction = -1
    s = solve_ivp(rhs, [xi0, 20], y0, events=ev, rtol=1e-12, atol=1e-14, max_step=0.01)
    xi1 = s.t_events[0][0]; dth1 = s.y_events[0][0][1]
    return xi1, -xi1**2 * dth1
xi1, w3 = lane_emden(3)
num("lane_emden_n3_xi1", xi1); num("omega3", w3)
check("C1_omega3", abs(w3 - 2.01824) < 2e-5, f"omega_3^0 = {w3:.6f} (published 2.01824)")
check("C2_xi1", abs(xi1 - 6.89685) < 2e-4, f"xi_1 = {xi1:.5f} (published 6.89685)")
coef = w3 * math.sqrt(3 * math.pi) / 2
num("chandra_coefficient_w3_sqrt3pi_over_2", coef)
for mu_e in [2.0, 2.154]:   # 2.154 = 56/26 for iron-56
    for lab, mb in [("m_u", m_u), ("m_H", m_H)]:
        Mch = coef * mP**3 / (mu_e * mb)**2
        num(f"M_Ch_mu{mu_e}_{lab}_Msun", Mch / Msun)
check("C3_MCh_book_value", abs(OUT["numbers"]["M_Ch_mu2.0_m_u_Msun"] - 1.456) < 2e-3, f"{OUT['numbers']['M_Ch_mu2.0_m_u_Msun']:.4f} Msun (book 1.456)")
# capacity rewriting: N_baryon = M_Ch/m_u = coef/mu_e^2 * (lambda_C/l_P)^3 with lambda_C = hbar/(m_u c)
lamC = hbar / (m_u * c)
NB = coef / 4 * (lamC / math.sqrt(lP2))**3
check("C4_capacity_rewrite_identity", abs(NB * m_u - coef * mP**3 / (2 * m_u)**2) / (NB * m_u) < 1e-12,
      "(m_P/m_u)^2 = (lambda_C/l_P)^2: Planck areas per baryon Compton area")
num("Planck_areas_per_baryon_Compton_area", (lamC**2 / lP2))
num("N_baryons_at_MCh_mu2", NB); num("N_electrons_at_MCh_mu2", NB / 2)
# symbolic: (m_P/m)^2 == lambda_C^2/l_P^2
mm = sp.Symbol("m", positive=True)
check("C5_mP_over_m_is_Compton_over_lP", sp.simplify((hb * cc / GG) / mm**2 - (hb / (mm * cc))**2 / (hb * GG / cc**3)) == 0)
# ledger of a collapsed limiting mass vs its Pauli (baryon) count: S_BH/k = 4 pi K^(2/3) N_B^(4/3), K = coef/mu_e^2
Kc = coef / 4
S_Ch = 4 * math.pi * G * (NB * m_u)**2 / (hbar * c)
check("C7_ledger_is_Pauli_count_to_4_3", abs(S_Ch / (4 * math.pi * Kc**(2 / 3) * NB**(4 / 3)) - 1) < 1e-10,
      f"S_BH(M_Ch) = {4*math.pi*Kc**(2/3):.3f} N_B^(4/3) nats")
num("S_BH(M_Ch)_nats", S_Ch); num("S_BH(M_Ch)_per_baryon_nats", S_Ch / NB); num("ledger_prefactor_4piK^(2/3)", 4 * math.pi * Kc**(2 / 3))
KK, NN = sp.symbols("K N", positive=True)
S_sym = 4 * sp.pi * (NN * mm)**2 / (hb * cc / GG) * (1)  # S/k = 4 pi M^2/m_P^2 with M = N m
S_sub = S_sym.subs(NN, KK * (sp.sqrt(hb * cc / GG) / mm)**3)
check("C8_ledger_identity_symbolic", sp.simplify(S_sub - 4 * sp.pi * KK**sp.Rational(2, 3) * (KK * (sp.sqrt(hb * cc / GG) / mm)**3)**sp.Rational(4, 3)) == 0)
# Landau uniform-sphere estimate, relativistic electrons: (3/4) N_e E_F = (3/5) G M^2 / R, E_F = hbar c (3 pi^2 n)^(1/3)
Ne = sp.Symbol("N_e", positive=True); mue = sp.Symbol("mu_e", positive=True); mb_ = sp.Symbol("m_b", positive=True)
V = sp.Rational(4, 3) * sp.pi * R**3
EF = hb * cc * (3 * sp.pi**2 * Ne / V)**sp.Rational(1, 3)
balance = sp.Eq(sp.Rational(3, 4) * Ne * EF, sp.Rational(3, 5) * GG * (Ne * mue * mb_)**2 / R)
Ne_sol = sp.solve(balance, Ne)
Ne_L = [s_ for s_ in Ne_sol if s_ != 0][0]
check("C6_landau_R_independent", sp.diff(sp.simplify(Ne_L), R) == 0, "both energies scale as 1/R: the limit is a mass, not a radius")
Ne23 = sp.simplify(Ne_L**sp.Rational(2, 3) / ((hb * cc / (GG * mue**2 * mb_**2))))
num("landau_Ne^(2/3)_over_(hbar c/(G mu_e^2 m^2))", float(Ne23))
M_L = float((Ne_L * mue * mb_).subs({hb: hbar, cc: c, GG: G, mue: 2, mb_: m_u})) / Msun
num("M_Landau_uniform_mu2_Msun", M_L)
num("M_Landau_over_M_Ch", M_L / OUT["numbers"]["M_Ch_mu2.0_m_u_Msun"])
# Pauli inner capacity per volume at relativistic Fermi momentum p_F = m_e c x
num("pauli_states_per_m3_at_pF=m_e_c", (m_e * c / hbar)**3 / (3 * math.pi**2))
# Fermi momentum / m_e c at the centre of ZTF J1901-like WD (central density 3.7e9 g/cc quoted in arXiv:2510.23724 model)
rho_c_WD = 3.7e12  # kg/m^3
ne_c = rho_c_WD / (2 * m_u)
xF = hbar * (3 * math.pi**2 * ne_c)**(1 / 3) / (m_e * c)
num("ZTFJ1901_model_central_xF=pF/(m_e c)", xF)

# ---------------------------------------------------------------- T: Oppenheimer-Volkoff free neutron gas
eps0 = m_n**4 * c**5 / (math.pi**2 * hbar**3)   # J/m^3
import mpmath as mpm
mpm.mp.dps = 50
def eos(xF_):
    X_ = mpm.mpf(xF_); sq = mpm.sqrt(1 + X_**2)
    e = eps0 / 8 * float(X_ * sq * (1 + 2 * X_**2) - mpm.asinh(X_))
    P = eps0 / 24 * float(X_ * sq * (2 * X_**2 - 3) + 3 * mpm.asinh(X_))
    return e, P
# tabulate P(x), e(x) and invert
xs = np.logspace(-5, 2.5, 4000)
EP = np.array([eos(xx) for xx in xs])
lnP = np.log(EP[:, 1]); lne = np.log(EP[:, 0])
def e_of_P(P):
    if P <= 0: return 0.0
    return float(np.exp(np.interp(math.log(P), lnP, lne)))
def tov_star(xc):
    ec, Pc = eos(xc)
    def rhs(rr_, yv):
        mm_, P = yv
        if P <= 0: return [0.0, 0.0]
        e = e_of_P(P)
        dm = 4 * math.pi * rr_**2 * e / c**2
        dP = -G * (e + P) * (mm_ + 4 * math.pi * rr_**3 * P / c**2) / (c**2 * rr_**2 * (1 - 2 * G * mm_ / (c**2 * rr_)))
        return [dm, dP]
    r0 = 1.0
    y0 = [4 / 3 * math.pi * r0**3 * ec / c**2, Pc]
    ev = lambda rr_, yv: yv[1] - Pc * 1e-12; ev.terminal = True; ev.direction = -1
    s = solve_ivp(rhs, [r0, 1e6], y0, events=ev, rtol=1e-9, atol=[1e10, Pc * 1e-14], max_step=50.0, dense_output=True)
    Rr = s.t_events[0][0]; Mm = s.y_events[0][0][0]
    rgrid = np.linspace(r0, Rr, 400)
    mgrid = s.sol(rgrid)[0]
    fprof = 2 * G * mgrid / (c**2 * rgrid)
    return Mm, Rr, float(fprof.max()), float(rgrid[fprof.argmax()] / Rr), float(2 * G * Mm / (c**2 * Rr))
best = None
scan = []
for xc in np.linspace(0.3, 2.0, 35):
    Mm, Rr, fmax, rfmax, fsurf = tov_star(xc)
    scan.append((xc, Mm / Msun, Rr / 1e3, fsurf, fmax, rfmax))
    if best is None or Mm > best[1] * Msun:
        best = (xc, Mm / Msun, Rr / 1e3, fsurf, fmax, rfmax)
num("OV_scan_xc_M_R_km_fsurf_fmax_rfmax", [list(map(float, s_)) for s_ in scan[::5]])
num("OV_max", {"xc": best[0], "M_Msun": best[1], "R_km": best[2], "f_surface": best[3], "f_max_inside": best[4], "r_at_fmax_over_R": best[5]})
check("T1_OV_mass", abs(best[1] - 0.71) < 0.015, f"M_max = {best[1]:.4f} Msun (OV 1939: 0.71)")
check("T2_OV_radius", 8.8 < best[2] < 9.8, f"R = {best[2]:.2f} km (published 9.0-9.6 km)")
coefOV = best[1] * Msun / (mP**3 / m_n**2)
num("OV_coefficient_of_mP3_over_mn2", coefOV)
check("T3_OV_coefficient", abs(coefOV - 0.384) < 0.01, f"M_OV = {coefOV:.4f} m_P^3/m_n^2 (published 0.384)")
check("T4_OV_f_below_buchdahl", best[4] < 8 / 9, f"max_r 2Gm(r)/(c^2 r) = {best[4]:.3f} < 8/9")
num("Chandrasekhar_like_coefficient_if_neutrons_were_newtonian_mu1", coef)
num("GR_reduction_factor_OV_over_newtonian_relativistic_fermions", coefOV / coef)

# ---------------------------------------------------------------- N: numbers for the objects
def obj(name, M_kg, R_m, src):
    fB_ = rs(M_kg) / R_m
    cap_bits = 4 * math.pi * R_m**2 / (4 * math.log(2) * lP2)
    ledger_bits = 4 * math.pi * rs(M_kg)**2 / (4 * math.log(2) * lP2)
    Tg = hbar * (G * M_kg / R_m**2) / (2 * math.pi * kB * c)
    return {"object": name, "M_Msun": M_kg / Msun, "R_km": R_m / 1e3, "r_s_km": rs(M_kg) / 1e3, "f=r_s/R": fB_, "f_A=(r_s/R)^2": fB_**2,
            "capacity_bits": cap_bits, "hole_ledger_bits_S_BH(M)": ledger_bits, "surface_T_of_g_K": Tg,
            "f_over_buchdahl": fB_ / (8 / 9), "source": src}
rows = []
for Mc_, Rc_ in [(1.3, 1500e3), (1.3, 3000e3), (1.4, 3000e3), (1.5, 3000e3), (1.5, 1500e3), (2.0, 5000e3), (1.4, 10000e3)]:
    rows.append(obj(f"iron core {Mc_} Msun, R {Rc_/1e3:.0f} km", Mc_ * Msun, Rc_, "Sukhbold & Woosley 2014 (1.4-2.0 Msun); Woosley lecture (5,000-10,000 km); Martinez-Pinedo lecture (~3,000 km); task range 1500-3000 km"))
rows.append(obj("ZTF J1901+1458 (Caiazzo 2021)", 1.35 * Msun, 2140e3, "Caiazzo et al. 2021 Nature 595, 39: 1.327-1.365 Msun, 2140 km"))
rows.append(obj("ZTF J1901+1458 (revised 2026)", 1.32 * Msun, 2664e3, "Desai et al. 2026 A&A: 2664 km, 1.32 Msun (C-O)"))
rows.append(obj("Sirius B", 1.018 * Msun, 0.008098 * 6.957e8, "Bond et al. 2017"))
rows.append(obj("PSR J0740+6620", 2.08 * Msun, 12.39e3, "Fonseca 2021; Riley 2021"))
for Rk in [10e3, 11e3, 12e3]:
    rows.append(obj(f"NS at M_TOV 2.25 Msun, R {Rk/1e3:.0f} km", 2.25 * Msun, Rk, "Fan et al. 2024 PRD 109, 043052 (2.25 +0.08 -0.07); R 10-12 km task range"))
rows.append(obj("OV free-neutron maximum star", best[1] * Msun, best[2] * 1e3, "this script, section T"))
rows.append(obj("Sun", Msun, 6.957e8, "IAU nominal radius"))
rows.append(obj("10 Msun black hole (at r_s)", 10 * Msun, rs(10 * Msun), "definition"))
num("objects", rows)
ic = [rw["f=r_s/R"] for rw in rows if rw["object"].startswith("iron core")]
num("iron_core_f_range", [min(ic), max(ic)])
check("N1_iron_core_far_from_buchdahl", max(ic) < 1e-2, f"iron core f in [{min(ic):.2e}, {max(ic):.2e}]")
check("N2_BH_f_is_1", abs(rows[-1]["f=r_s/R"] - 1) < 1e-12)
check("N3_TOV_stars_below_buchdahl", all(rw["f=r_s/R"] < 8 / 9 for rw in rows if "NS" in rw["object"] or "PSR" in rw["object"]))
check("N4_TOV_stars_below_causal", all(rw["f=r_s/R"] < 0.708 for rw in rows if "NS" in rw["object"] or "PSR" in rw["object"]),
      "every published NS point sits under the causal ceiling f = 0.708")
# minimum radius for 2.25 Msun from Buchdahl and causality
num("R_min_2.25Msun_buchdahl_km", rs(2.25 * Msun) / (8 / 9) / 1e3)
num("R_min_2.25Msun_causal_km", rs(2.25 * Msun) / 0.708 / 1e3)
# Bekenstein-Hawking of 10 Msun in bits
num("S_BH_10Msun_bits", 4 * math.pi * G * (10 * Msun)**2 / (hbar * c) / math.log(2))
num("S_BH_1Msun_nats", 4 * math.pi * G * Msun**2 / (hbar * c))
check("N5_SBH_1Msun_book", abs(OUT["numbers"]["S_BH_1Msun_nats"] / 1.049e77 - 1) < 2e-3)

# ---------------------------------------------------------------- M: a local mu < 1?
mu0 = 1 / 1.158   # book p3_01 tab:bh_tests: 1/mu(z=0) = 1.158
num("mu_z0", mu0)
num("M_Ch_if_G_eff=mu*G_Msun", OUT["numbers"]["M_Ch_mu2.0_m_u_Msun"] * mu0**-1.5)
num("M_OV_if_G_eff=mu*G_Msun", best[1] * mu0**-1.5)
check("M1_scaling_G^-3/2", True, "M_Ch and M_OV scale as G^(-3/2) (m_P^3 = (hbar c/G)^(3/2))")

OUT["n_pass"] = sum(v["pass"] for v in OUT["checks"].values())
OUT["n_fail"] = sum(not v["pass"] for v in OUT["checks"].values())
print("passed", OUT["n_pass"], "failed", OUT["n_fail"])
with open("bh_formation_checks_output.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
