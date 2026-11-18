import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
"""ledger_half_checks.py  (private, proposals only; 2026-11-18)

Checks every DERIVED step of LEDGER_HALF_2026-11-18.md in sympy (symbolic) and,
where a number is quoted, in SI with mpmath/scipy-free quadrature (sympy/mpmath only).

Book inputs used as given (weave/book):
  p1_02:305-312  Cai-Kim: -dE = A_H (rho+P) r_A H dt, T_H = H/2pi, S_geo = pi/(G H^2)
  p1_02:322-335  eq:law_F2info, eq:law_Sneed (record term, required rate)
  p1_02:337      balance law dlnE/dlna = 1 - ln E ("a writer at one per e-fold")
  p1_02:370-379  eq:law_constraint, eq:Ea  E(a) = exp(1-1/a)
  p1_02:440-446  eq:law_rhoinfo1, eq:law_beta  rho_info(1) = rho_m/2, beta_m = Omega_m/2
  p7_05d:36-46   eq:eh_area, eq:eh_dS (Euler halves at the two horizons)
  p2_12:240-243  eq:lam_smarr  T_GH S = M_H c^2
  p1_03:151      eq:vl_komar  M_K c^2 = 2 T_H S
Padmanabhan 2010 (arXiv:0912.3165) eqs (1),(7),(8); Padmanabhan 2012 (arXiv:1206.4916) dV/dt = L_P^2 (N_sur - N_bulk).
Run: python3 ledger_half_checks.py
"""
import sympy as sp
import mpmath as mp

PASS, FAIL = [], []
def check(tag, cond, note=""):
    ok = bool(cond)
    (PASS if ok else FAIL).append(tag)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag} {note}")

def is_zero(expr):
    return sp.simplify(expr) == 0

# ---------- symbols (units hbar = c = k_B = 1, G explicit) ----------
G, H, H0, a, Om, rho, P, Sdot, M, t = sp.symbols('G H H0 a Omega_m rho P Sdot M t', positive=True)
pi = sp.pi
T_GH  = H/(2*pi)
S_geo = pi/(G*H**2)
r_A   = 1/H
V_H   = sp.Rational(4,3)*pi*r_A**3
A_H   = 4*pi*r_A**2
E_MS  = r_A/(2*G)
rho_c = 3*H**2/(8*pi*G)

# A1  T_GH S_geo = E_MS for any H  (book eq:lam_smarr; ESTABLISHED, re-checked)
check("A1 T_GH*S_geo == E_MS (any H)", is_zero(T_GH*S_geo - E_MS))
# A2  E_MS = rho_c V_H  (Friedmann)
check("A2 E_MS == rho_c*V_H", is_zero(E_MS - rho_c*V_H))
# A3  identity 1 at a=1: beta_m S_geo = (1/2) M_m / T_GH, M_m = Omega_m rho_c V_H
M_m = Om*rho_c*V_H
beta_m = Om/2
check("A3 beta_m*S_geo == (1/2)M_m/T_GH", is_zero(beta_m*S_geo - sp.Rational(1,2)*M_m/T_GH))

# ---------- B: the book's record term as a horizon entry ----------
# B1 from eq:law_F2info: rho_x + P_x = -H^3 Sdot/(8 pi^2); continuity rho_x' = -3H(rho_x+P_x)
rhoP_x   = -H**3*Sdot/(8*pi**2)
rhodot_x = -3*H*rhoP_x
check("B1 T_GH*Sdot_info == V_H*rhodot_x (any history, any w)", is_zero(T_GH*Sdot - V_H*rhodot_x))
# B1b sign: rho_x+P_x < 0 for Sdot>0
check("B1b rho_x+P_x < 0 for Sdot>0", sp.simplify(rhoP_x).is_negative)
# B1c the same entry is the Cai-Kim flux of the component: A_H (rho+P) r_A H dt with sign reversed
check("B1c T_GH*Sdot == -A_H*(rho_x+P_x)*r_A*H", is_zero(T_GH*Sdot + A_H*rhoP_x*r_A*H))

# B2 activation (given): E(a) = exp(1-1/a); balance law dlnE/dlna = 1 - lnE = 1/a
E = sp.exp(1 - 1/a)
dlnE_dlna = sp.simplify(a*sp.diff(sp.log(E), a))
check("B2 dlnE/dlna == 1/a", is_zero(dlnE_dlna - 1/a))
check("B2b dlnE/dlna == 1 - lnE", is_zero(dlnE_dlna - (1 - sp.log(E))))
check("B2c dlnE/dlna at a=1 == 1 (one per e-fold today)", dlnE_dlna.subs(a,1) == 1)
# record constraint rho' = rho H/a is the same statement
rho_info0 = sp.symbols('rho_i0', positive=True)
rho_info = rho_info0*E
check("B2d rhodot_info == rho_info*H/a (eq:law_constraint)", is_zero(sp.diff(rho_info,a)*a*H - rho_info*H/a))

# B3 the book's required Sdot (eq:law_Sneed) = V_H rhodot_info / T_GH with rhodot = rho H / a
Sdot_book = 8*pi**2*rho_info/(3*a*H**3)
check("B3 eq:law_Sneed == V_H*rhodot_info/T_GH", is_zero(Sdot_book - V_H*(rho_info*H/a)/T_GH))
# B4 per e-fold: T_GH dS_info/dlna = E_info * dlnE/dlna, E_info = rho_info V_H
E_info = rho_info*V_H
check("B4 T_GH*(Sdot/H) == E_info*dlnE/dlna", is_zero(T_GH*Sdot_book/H - E_info*dlnE_dlna))

# ---------- C: general a, flat matter + Lambda ----------
Om0, h = sp.symbols('Omega_m0 h', positive=True)   # h = H/H0
Hh = H0*h
rho_c0 = 3*H0**2/(8*pi*G)
rho_m_a = Om0*rho_c0*a**-3
rho_info_a = (Om0/2)*rho_c0*E
T_a, S_a, V_a = Hh/(2*pi), pi/(G*Hh**2), sp.Rational(4,3)*pi/Hh**3
M_m_a = rho_m_a*V_a
dSinfo_dlna = 8*pi**2*rho_info_a/(3*a*Hh**3)/Hh            # eq:law_Sneed / H
book_form = (pi/G)*(Om0/2)*H0**2*E/(a*Hh**4)                # count doc 1.1 form
check("C1 eq:law_Sneed/H == (pi/G) beta_m H0^2 E/(a H^4)", is_zero(dSinfo_dlna - book_form))
check("C2 dS_info/dlna == (2 rho_info/rho_m)*(dlnE/dlna)*(1/2)M_m/T", is_zero(dSinfo_dlna - (2*rho_info_a/rho_m_a)*dlnE_dlna*sp.Rational(1,2)*M_m_a/T_a))
check("C3 (2 rho_info/rho_m)*(dlnE/dlna) == E a^2", is_zero((2*rho_info_a/rho_m_a)*dlnE_dlna - E*a**2))
# C4 one sixth: with Omega_m(a) = rho_m/rho_c(a), dlnH/dlna = -(3/2)Omega_m(a) (flat m+Lambda)
Oma = sp.symbols('Omega_ma', positive=True)
S_of = pi/(G*H**2)
dS_geo_dlna = sp.diff(S_of, H)*H*(-sp.Rational(3,2)*Oma)
half_rest = sp.Rational(1,2)*Oma*rho_c*V_H/T_GH
check("C4 (1/2)M_m/T_GH == (1/6) dS_geo/dlna (any a)", is_zero(half_rest - dS_geo_dlna/6))
# C5 the Cai-Kim crossing per e-fold for matter + Lambda: 3 V (rho_m + 0) = 3 M_m = T dS_geo/dlna
check("C5 T_GH dS_geo/dlna == 3 M_m (matter enthalpy flux)", is_zero(T_GH*dS_geo_dlna - 3*Oma*rho_c*V_H))
# C6 Friedmann check of dlnH/dlna = -(3/2)Omega_m(a): Hdot = -4 pi G rho_m
check("C6 -4piG rho_m/H^2 == -(3/2)Omega_m(a)", is_zero(-4*pi*G*Oma*rho_c/H**2 + sp.Rational(3,2)*Oma))

# ---------- D: Padmanabhan's counting ----------
# 2010: E = (1/2) n T, n = A/L_P^2 (L_P^2 = G); S = n/4  =>  E = 2 T S
n_sur = A_H/G
check("D1 N_sur == 4 S_geo", is_zero(n_sur - 4*S_geo))
# 2012: N_bulk = -2 E_K/T, E_K = (rho+3p)V ; matter+Lambda: rho+3p = rho_m - 2 rho_L
OL = 1 - Oma
E_K = (Oma - 2*OL)*rho_c*V_H
N_bulk = -2*E_K/T_GH
dN = n_sur - N_bulk
check("D2 Delta N == 6 Omega_m S_geo", is_zero(dN - 6*Oma*S_geo))
check("D3 (1/2) T Delta N == 3 M_m == T dS_geo/dlna", is_zero(sp.Rational(1,2)*T_GH*dN - 3*Oma*rho_c*V_H) and is_zero(sp.Rational(1,2)*T_GH*dN - T_GH*dS_geo_dlna))
# D4 Padmanabhan's law dV/dt = L_P^2 Delta N reproduces Hdot = -4 pi G rho_m
Hdot = -4*pi*G*Oma*rho_c
dVdt = sp.diff(V_H, H)*Hdot
check("D4 dV/dt == G*Delta N (flat m+Lambda)", is_zero(dVdt - G*dN))
# D5 matter bulk dof alone: |N_bulk,m| = 2 M_m/T ; record per e-fold in tiles (n = 4 S) at a=1
N_bulk_m = 2*Oma*rho_c*V_H/T_GH
tiles_rec = 4*(Oma/2)*S_geo     # 4 * beta_m S_geo at a=1 (Omega_ma -> Omega_m today)
check("D5 record tiles per e-fold (a=1) == |N_bulk,m|", is_zero(tiles_rec - N_bulk_m))
check("D6 |N_bulk,m|/N_sur == Omega_m/2 == beta_m", is_zero(N_bulk_m/n_sur - Oma/2))
check("D7 (1/2) T per tile over record tiles == M_m (Padmanabhan E = n T/2)", is_zero(sp.Rational(1,2)*T_GH*tiles_rec - Oma*rho_c*V_H))
check("D8 record / crossing per e-fold == 1/6 == (1/2)/3", is_zero((sp.Rational(1,2)*Oma*rho_c*V_H)/(3*Oma*rho_c*V_H) - sp.Rational(1,6)))
# D9 general a: record tiles / |N_bulk,m(a)| == E a^2
Nbm_a = 2*M_m_a/T_a
check("D9 general a: 4 dS_info/dlna / |N_bulk,m| == E a^2", is_zero(4*dSinfo_dlna/Nbm_a - E*a**2))

# ---------- E: black hole (Schwarzschild) ----------
r_s = 2*G*M
T_H = 1/(4*pi*r_s)            # = 1/(8 pi G M)
S_BH = pi*r_s**2/G            # A/4G
check("E1 Smarr: T_H S_BH == M/2", is_zero(T_H*S_BH - M/2))
check("E2 Euler: M dS/dM == 2 S", is_zero(M*sp.diff(S_BH, M) - 2*S_BH))
check("E3 first law: dS/dM == 1/T_H", is_zero(sp.diff(S_BH, M) - 1/T_H))
n_BH = 4*pi*r_s**2/G
check("E4 Padmanabhan: (1/2) n T_H == M (= Komar mass)", is_zero(sp.Rational(1,2)*n_BH*T_H - M))
check("E5 static BH: N_sur == N_bulk = 2M/T_H (holographic equipartition)", is_zero(n_BH - 2*M/T_H))
check("E6 S = E_K/(2T) for BH (Padmanabhan eq 1)", is_zero(S_BH - M/(2*T_H)))
# de Sitter: E_K = (rho+3p)V = -2 rho V ; T S = (1/2)|E_K| ; (1/2) n T = 2 E_MS = |E_K|
E_K_dS = -2*rho_c*V_H
check("E7 de Sitter: T_GH S == (1/2)|E_K|", is_zero(T_GH*S_geo - sp.Rational(1,2)*(-E_K_dS)))
check("E8 de Sitter: (1/2) N_sur T == |E_K| == N_bulk (1/2) T", is_zero(sp.Rational(1,2)*n_sur*T_GH + E_K_dS))
# cosmic Euler (book eq:eh_dS): dE_MS/dS_geo = T_GH/2
check("E9 cosmic: dE_MS/dS_geo == T_GH/2", is_zero(sp.diff(E_MS,H)/sp.diff(S_geo,H) - T_GH/2))
# common rule: T * (ledger entropy) == (1/2) * (enclosed active energy of what is booked)
#   BH total:              T_H S_BH       = (1/2) E_K(hole)          (E6)
#   cosmic per e-fold a=1: T_GH dS_info   = (1/2) E_K(matter) = M_m/2 (A3, matter p=0)
E_K_m = (Oma + 0)*rho_c*V_H
check("E10 cosmic per e-fold (a=1): T_GH beta_m S_geo == (1/2) E_K(matter)", is_zero(T_GH*(Oma/2)*S_geo - E_K_m/2))

# E11 BH per e-fold of its own mass (flow): T_H dS/dlnM == M (full rest energy, twice Smarr)
check("E11 BH flow: T_H*M*dS/dM == M (= 2 T_H S)", is_zero(T_H*M*sp.diff(S_BH, M) - M) and is_zero(T_H*M*sp.diff(S_BH, M) - 2*T_H*S_BH))
# E12 cosmic record flow equals a Smarr-type stock only because dlnE/dlna = 1 at a=1:
#     T_GH dS_info/dlna = E_info * dlnE/dlna  ->  E_info at a=1  (stock / flow coincide today)
check("E12 at a=1: T_GH dS_info/dlna == E_info (flow == stock)", is_zero((E_info*dlnE_dlna - E_info).subs(a,1)))

# ---------- F: horizon Landauer (forward Step 7) ----------
kappa, dA = sp.symbols('kappa dA', positive=True)
T_k = kappa/(2*pi)
dE = kappa*dA/(8*pi*G)
sol = sp.solve(sp.Eq(dE, T_k*sp.log(2)), dA)[0]
check("F1 heat T ln2 adds 4 ln2 G of area, any kappa", is_zero(sol - 4*sp.log(2)*G))
check("F2 record area per e-fold (a=1) == beta_m A_H", is_zero(4*G*sp.Rational(1,2)*Oma*rho_c*V_H/T_GH - (Oma/2)*A_H))

# ---------- G: numbers (SI) ----------
mp.mp.dps = 30
c_, hb, kB, Gn = mp.mpf('299792458'), mp.mpf('1.054571817e-34'), mp.mpf('1.380649e-23'), mp.mpf('6.67430e-11')
Mpc = mp.mpf('3.0856775814913673e22')
H0s = mp.mpf('67.16')*1000/Mpc       # book p1_02:335 uses H0 = 67.16
Omv = mp.mpf('0.3153')
Sgeo = mp.pi*c_**5/(hb*Gn*H0s**2)
TGH = hb*H0s/(2*mp.pi*kB)
EMS = c_**5/(2*Gn*H0s)
Mm = Omv*EMS/c_**2
rate_bits = (Omv/2)*Sgeo/mp.log(2)
half_bits = (mp.mpf(1)/2*Mm*c_**2/(kB*TGH))/mp.log(2)
print(f"   S_geo = {mp.nstr(Sgeo,6)} nats; T_GH = {mp.nstr(TGH,5)} K; M_m = {mp.nstr(Mm,5)} kg; rate = {mp.nstr(rate_bits,5)} bits/e-fold")
check("G1 beta_m S_geo == (1/2)M_m c^2/(k T_GH) numerically", abs(rate_bits/half_bits - 1) < mp.mpf('1e-25'))
check("G2 rate ~ 5.2e121 bits/e-fold (book p1_02:335)", abs(rate_bits/mp.mpf('5.2e121') - 1) < 0.01)
sig = mp.mpf('213e3')
kin = sig**2/(2*c_**2)
print(f"   kinetic/rest for sigma=213 km/s (p2_12b:89): (1/2)sigma^2/c^2 = {mp.nstr(kin,3)}")
check("G3 kinetic share << 1 (<1e-6)", kin < mp.mpf('1e-6'))
# G4 accumulated S_info(a=1) from a->0, in units of (1/2)M_m c^2/kT_GH today, flat m+Lambda (+ radiation 9.1e-5 as book p1_02:580)
Omr = mp.mpf('9.1e-5'); OmL = 1 - Omv - Omr
hfun = lambda x: mp.sqrt(Omv/x**3 + Omr/x**4 + OmL)
Ef = lambda x: mp.e**(1 - 1/x)
integrand = lambda x: Ef(x)/(x**2*hfun(x)**4)    # (dS_info/dlna)/(beta_m S_geo0) * (1/a)
acc = mp.quad(integrand, [mp.mpf('1e-6'), mp.mpf('0.1'), mp.mpf('0.5'), 1])
print(f"   S_info(a=1) accumulated / [(1/2)M_m c^2/kT_GH] = {mp.nstr(acc,6)}")
check("G4 accumulated record today is a finite O(1) fraction (0<x<1) of one e-fold's entry", 0 < acc < 1)
# G5 per-e-fold factor E a^2 at a = 0.5, 1, 2
for x in (mp.mpf('0.5'), mp.mpf(1), mp.mpf(2)):
    print(f"   E(a)a^2 at a={mp.nstr(x,2)}: {mp.nstr(Ef(x)*x**2,5)}")
check("G5 E(1)*1^2 == 1", Ef(mp.mpf(1)) == 1)
# G6 Padmanabhan Delta N today and record tiles (numbers)
print(f"   Delta N today = 6 Omega_m S_geo = {mp.nstr(6*Omv*Sgeo,5)} tiles; record tiles/e-fold = |N_bulk,m| = {mp.nstr(2*Omv*Sgeo,5)}")
check("G6 record tiles = Delta N / 3", abs((2*Omv*Sgeo)/(6*Omv*Sgeo) - mp.mpf(1)/3) < mp.mpf('1e-25'))


# ---------- I: the lead's hand result and the 1/6 (report section 1) ----------
rm = sp.symbols('rho_m', positive=True)
dt1 = 1/H                                            # one e-fold
flux = A_H*rm*r_A*H*dt1                              # -dE, matter (P=0) through r_A in one e-fold
check("I1 -dE per e-fold == 4 pi rho_m / H^3", is_zero(flux - 4*pi*rm/H**3))
dS_flux = flux/T_GH
check("I2 dS_flux == 8 pi^2 rho_m / H^4", is_zero(dS_flux - 8*pi**2*rm/H**4))
rm_O = 3*Oma*H**2/(8*pi*G)
check("I3 dS_flux == 3 Omega_m S_geo", is_zero(dS_flux.subs(rm, rm_O) - 3*Oma*S_geo))
check("I4 dS_flux == matter part of dS_geo/dlna (flat m+Lambda)", is_zero(dS_flux.subs(rm, rm_O) - dS_geo_dlna))
check("I5 beta_m S_geo == dS_flux/6", is_zero((Oma/2)*S_geo - dS_flux.subs(rm, rm_O)/6))
M_m_r = sp.Rational(4,3)*pi*rm*r_A**3
check("I6 beta_m S_geo == (1/2) M_m / T_H, M_m = (4pi/3) rho_m r_A^3", is_zero((Oma/2)*S_geo - (sp.Rational(1,2)*M_m_r/T_GH).subs(rm, rm_O)))
# the 1/3: four statements, one number
check("I7a flux swept per e-fold / content == A_H r_A / V_H == 3", is_zero(flux/M_m_r - 3) and is_zero(A_H*r_A/V_H - 3))
w = sp.symbols('w')
check("I7b continuity: -(d rho/dlna)/(rho+P) == 3 (any w)", is_zero(3*H*(rho + w*rho)/H/(rho*(1+w)) - 3))
d = sp.symbols('d', positive=True, integer=True)
Om_d = sp.symbols('Omega_d', positive=True)       # solid angle in d dims
V_d, A_d, rr = Om_d*sp.Symbol('r')**d/d, Om_d*sp.Symbol('r')**(d-1), sp.Symbol('r')
check("I7c in d spatial dims A r / V == d (one per spatial direction)", is_zero(A_d*rr/V_d - d))
check("I7c' d ln V / d ln r == d", is_zero(sp.simplify(rr*sp.diff(V_d, rr)/V_d) - d))
check("I7d Padmanabhan: (1/2) T Delta N == 3 M_m == flux; Delta N == 3 |N_bulk,m|", is_zero(sp.Rational(1,2)*T_GH*dN - 3*Oma*rho_c*V_H) and is_zero(dN - 3*N_bulk_m))
check("I8 1/6 == (1/2 of content) x (content/flux = 1/3)", is_zero(((sp.Rational(1,2)*M_m_r)/flux) - sp.Rational(1,6)))
# radiation: the matter-only 3 becomes 3(rho_m + 4/3 rho_r): the record/flux ratio is 1/6 only for the matter flux
rr_ = sp.symbols('rho_r', positive=True)
flux_mr = A_H*(rm + sp.Rational(4,3)*rr_)*r_A
check("I9 with radiation, record/(total flux) == (1/6) rho_m/(rho_m + 4/3 rho_r)", is_zero((sp.Rational(1,2)*M_m_r)/flux_mr - sp.Rational(1,6)*rm/(rm + sp.Rational(4,3)*rr_)))
# Schwarzschild: same rule
check("I10 BH: flux of rest energy per e-fold of M, booked at T_H: dS/dlnM == M/T_H == 2 S", is_zero(M*sp.diff(S_BH, M) - M/T_H) and is_zero(M/T_H - 2*S_BH))
check("I11 BH: S == (1/2) M/T_H == (1/2) x flux entropy per e-fold (Smarr); no 1/3", is_zero(S_BH - sp.Rational(1,2)*M/T_H))
# 'content' rule at both horizons: ledger entropy = (1/2) x (enclosed active energy)/T
check("I12 content rule: BH S = (1/2) M/T_H and cosmic dS_info/dlna(a=1) = (1/2) M_m/T_GH",
      is_zero(S_BH - M/(2*T_H)) and is_zero((Oma/2)*S_geo - Oma*rho_c*V_H/(2*T_GH)))
# the BH's 1/2 is Euler degree 2 (S ~ M^2); the cosmic geometric horizon's own Euler half sits in dE_MS/dS = T/2 (E9)
check("I13 BH 1/2 == 1/(Euler degree of S in M) == 1/2", is_zero(S_BH/(M*sp.diff(S_BH, M)) - sp.Rational(1,2)))

# ---------- H: the four candidates (numbers from candidates_compare.py, same file set) ----------
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import candidates_compare as CC
import bulk_virial_count as BV
R = CC.res
# H1 under any heat-at-T_GH booking, candidate/required at a=1 is an ENERGY ratio: (energy per e-fold)/((1/2) M_m c^2)
#    (T_GH cancels: required = (1/2)M_m c^2/kT_GH by A3).  Checked for the book rule (ST) with its own H0.
bits, Q, T, V, dK = BV.bulk_bits(1.0, BV.f_ST)
half_rest_BV = 0.5*BV.Om*BV.rho_c0*BV.c**2*V
r_energy = Q/half_rest_BV
r_bits = bits/((half_rest_BV/(BV.kB*T))/np.log(2))
check("H1 book-rule ratio == energy ratio (dK/dlna V)/(1/2 M_m c^2)", abs(r_energy/r_bits - 1) < 1e-12, f"({r_energy:.3e})")
check("H1b book-rule ratio vs required (H0=67.16) within 3% of energy ratio", abs(R["(1) book rule, ST, horizon-volume total"][0]/r_energy - 1) < 0.03)
# H2 (4) shape is 1/(E a^2) exactly (C3)
for k, z in enumerate((0, 1, 2)):
    a = 1/(1+z)
    check(f"H2 (4) ratio at z={z} == 1/(E a^2)", abs(R["(4) (1/2) M_m c^2 at kT_GH per e-fold"][k]*float(sp.exp(1-1/sp.Rational(1,1+z)))*a**2 - 1) < 2e-3)
# H3 starlight/BH ratio scales as [psi(z)/psi(0)] (1+z)^2 e^z / h(z)  (L_com V_com,H /H /T  over  E/(aH^4))
for k, z in enumerate((1, 2), start=1):
    pred = CC.psi(z)/CC.psi(0)*(1+z)**2*np.exp(z)/CC.Eb(1/(1+z))
    got = R["(3) starlight at kT_GH ln2"][k]/R["(3) starlight at kT_GH ln2"][0]
    check(f"H3 starlight shape z={z} == psi ratio (1+z)^2 e^z / h(z)", abs(got/pred - 1) < 1e-9, f"({got:.3g})")
    # H4 the count document's quoted scaling psi (1+z)^2 e^z (x63, x586) differs from H3 by exactly h(z) (the e-fold duration 1/H)
    cd = CC.psi(z)/CC.psi(0)*(1+z)**2*np.exp(z)
    check(f"H4 count-doc scaling / H3 scaling == h(z) at z={z}", abs((cd/pred)/CC.Eb(1/(1+z)) - 1) < 1e-12, f"(count-doc x{cd:.0f}, here x{pred:.1f})")
# H5 shape test: E(a) shape requires r(z)/r(0) == 1; record each candidate's departure (reported, not a pass/fail on physics)
shape_ok = {k: (abs(v[1]/v[0]-1) < 0.1 and abs(v[2]/v[0]-1) < 0.1) for k, v in R.items() if np.isfinite(v[1])}
check("H5 no candidate follows the E(a) shape within 10% (statement of the computed table)", not any(shape_ok.values()))
# H7-H10 implied record histories (implied_history.py): the integration returns the book's E(a) from the book's shape,
#   the analytic (4a) history E = 4/3 - a^-3/3, the book's own mu values (p1_02:580 caption: 0.948, 0.982), and (4a)'s switch-on z.
import implied_history as IH
Ebk = IH.implied_E(IH.shapes["book E(a) (measured shape)"])
check("H7 ledger rule + book shape returns exp(1-1/a)", float(np.max(np.abs(Ebk - np.exp(1 - 1/IH.A)))) < 1e-4)
E4a = IH.implied_E(IH.shapes["(4a) 1/2 M_m(a) c^2 (matter inside horizon now)"])
check("H8 (4a) history == 4/3 - a^-3/3", float(np.max(np.abs(E4a - (4/3 - IH.A**-3/3)))) < 2e-3)
mb = IH.out["book E(a) (measured shape)"]["mu"]
check("H9 book mu(z=0.5)=0.948, mu(z=1)=0.982 reproduced", abs(mb[0.5]-0.948) < 6e-4 and abs(mb[1.0]-0.982) < 6e-4)
check("H10 (4a) switch-on z == 4^(1/3) - 1", abs(IH.out["(4a) 1/2 M_m(a) c^2 (matter inside horizon now)"]["z_on"] - (4**(1/3) - 1)) < 2e-3)
amp_ok = {k: abs(v[0]-1) < 0.1 for k, v in R.items()}
check("H6 only (4) meets the amplitude today within 10%", [k for k, ok in amp_ok.items() if ok] == ["(4) (1/2) M_m c^2 at kT_GH per e-fold"])

print(f"\nchecks passed: {len(PASS)}  failed: {len(FAIL)}")
if FAIL: print("FAILED:", FAIL)
