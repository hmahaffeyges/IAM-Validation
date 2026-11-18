import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""SPIDER S5: the 1/4 and one bit per minimum area (private, proposals only).
Every DERIVED step of SPIDER_S5_QUARTER_BIT_2026-11-18.md has a check here.
sympy + scipy.constants (CODATA 2018) only; runs in seconds, < 100 MB.
Output: spider_s5_checks_output.json
"""
import json, math
import sympy as sp
from scipy import constants as C

RES = []
def chk(cid, desc, ok, val=None):
    RES.append({"id": cid, "desc": desc, "pass": bool(ok), "value": None if val is None else str(val)})

hb, G, c, k, M, kap, A, eta, E, S, T, R, H = sp.symbols('hbar G c k M kappa A eta E S T R H', positive=True)
lP2 = hb*G/c**3

# ---------------- A. units, nats and bits ----------------
eta_BH = c**3/(4*hb*G)
chk("A1", "eta = c^3/(4 hbar G) = 1/(4 l_P^2)", sp.simplify(eta_BH - 1/(4*lP2)) == 0)
hbN, GN, cN, kN = C.hbar, C.G, C.c, C.k
lP2N = hbN*GN/cN**3
chk("A2a", "l_P^2 = 2.6121e-70 m^2", abs(lP2N/2.6121e-70 - 1) < 1e-4, lP2N)
chk("A2b", "eta = 9.570e68 m^-2 (book p3_01a:279)", abs(1/(4*lP2N)/9.570e68 - 1) < 2e-4, 1/(4*lP2N))
bit_area = 4*math.log(2)*lP2N
chk("A2c", "area per bit 4 ln2 l_P^2 = 7.24e-70 m^2 (book p3_01a:338)", abs(bit_area/7.24e-70 - 1) < 1e-3, bit_area)
bits_per_m2 = 1/bit_area
chk("A2d", "bits per m^2 = 1.38e69 (Bousso 2002 abstract: 1.4e69)", abs(bits_per_m2/1.4e69 - 1) < 0.02, bits_per_m2)
chk("A3a", "area per bit / l_P^2 = 4 ln 2 = 2.7726", abs(4*math.log(2) - 2.7725887) < 1e-6, 4*math.log(2))
chk("A3b", "bits per Planck area = 1/(4 ln 2) = 0.3607; nats per Planck area = 1/4",
    abs(1/(4*math.log(2)) - 0.360674) < 1e-6, 1/(4*math.log(2)))
chk("A4", "1 nat = 1/ln2 bits = 1.4427 bits (glossary p. app_F:98)", abs(1/math.log(2) - 1.442695) < 1e-6)
chk("A5", "cell: M = 20.94 kT per ATP = 30.21 bits (book p6_02:289)", round(20.94/math.log(2), 2) == 30.21, 20.94/math.log(2))

# ---------------- B. Hawking's route to the 1/4 ----------------
TH = hb*c**3/(8*sp.pi*G*M*k)                     # Hawking temperature
S_int = sp.integrate(c**2/TH, (M, 0, M))           # dS = d(Mc^2)/T, S(0) = 0
A_s = 16*sp.pi*G**2*M**2/c**4
chk("B1", "integrating d(Mc^2) = T_H dS from M = 0 gives S = k A/(4 l_P^2)", sp.simplify(S_int - k*A_s/(4*lP2)) == 0)
kapS = c**4/(4*G*M)
dAdE = sp.diff(A_s, M)/sp.diff(M*c**2, M)
chk("B2", "Schwarzschild: dA/d(Mc^2) = 8 pi G/(kappa c^2)", sp.simplify(dAdE - 8*sp.pi*G/(kapS*c**2)) == 0)
Tunruh = hb*kap/(2*sp.pi*k*c)
dA_nat = 8*sp.pi*G/(kap*c**2)*k*Tunruh
chk("B3a", "k T moves the area by 4 l_P^2 for every kappa", sp.simplify(dA_nat - 4*lP2) == 0)
chk("B3b", "k T ln2 moves the area by 4 ln2 l_P^2", sp.simplify(dA_nat*sp.log(2) - 4*sp.log(2)*lP2) == 0)
# B4: circularity. Any S = eta*A and any E(A); T := dE/dS. Area per k T of energy = 1/eta.
Ef = sp.Function('Ef')
Tdef = sp.diff(Ef(A), A)/(k*eta)
dA_per_kT = (k*Tdef)/sp.diff(Ef(A), A)
chk("B4", "for ANY E(A) and S = k eta A with T = dE/dS, the area per k T is 1/eta: 'delta A_min' restates eta",
    sp.simplify(dA_per_kT - 1/eta) == 0)
# B5: Jacobson structure equation
eta_sol = sp.solve(sp.Eq(hb*eta/(2*sp.pi), c**3/(8*sp.pi*G)), eta)[0]
chk("B5", "hbar eta/2pi = c^3/(8 pi G)  =>  eta = c^3/(4 hbar G)", sp.simplify(eta_sol - eta_BH) == 0)
# B6: the numeral 1/4 is tied to the choice of Planck unit
h = 2*sp.pi*hb
lh2 = h*G/c**3                 # Planck area built with h
lred2 = 8*sp.pi*hb*G/c**3      # reduced Planck area (8 pi G = 1)
chk("B6a", "with l_h^2 = hG/c^3: eta = (pi/2)/l_h^2", sp.simplify(eta_BH - (sp.pi/2)/lh2) == 0)
chk("B6b", "with reduced l^2 = 8 pi hbar G/c^3: eta = 2 pi / l^2", sp.simplify(eta_BH - 2*sp.pi/lred2) == 0)
# B7: dimensional rewriting does not remove G: eta*hbar*G/c^3 is the pure number 1/4
chk("B7", "eta l_P^2 = 1/4 exactly; G enters only through l_P", sp.simplify(eta_BH*lP2 - sp.Rational(1, 4)) == 0)

# ---------------- C. where ln2 and 1/4 meet ----------------
S_BH = k*A_s/(4*lP2)
Nbits = S_BH/(k*sp.log(2))
chk("C1", "N_bits * k T_H ln2 = T_H S_BH = Mc^2/2 (ln 2 cancels; Smarr)", sp.simplify(Nbits*k*TH*sp.log(2) - M*c**2/2) == 0)
b = sp.symbols('b', positive=True)
Nb = S_BH/(k*sp.log(b))
chk("C2", "for any base b: (S/k ln b) * k T ln b = T S (energy is base-free)", sp.simplify(Nb*k*TH*sp.log(b) - TH*S_BH) == 0)
area_per_unit = lambda base: 4*math.log(base)
chk("C3", "area per unit of base-b information = 4 ln b l_P^2: 4 (nat), 2.773 (bit), 4.394 (trit, Hod 1998)",
    abs(area_per_unit(2) - 2.7725887) < 1e-6 and abs(area_per_unit(3) - 4.3944492) < 1e-6 and abs(area_per_unit(math.e) - 4) < 1e-12)
# C4: Bekenstein 1973 mechanical minimum
mu, bb = sp.symbols('mu b_p', positive=True)
dE_mech = mu*kap*bb                   # Killing energy of a particle of proper size b lowered to the horizon (redshift kappa*b/c^2)
dA_mech = 8*sp.pi*G/(kap*c**2)*dE_mech
dA_min_B = sp.simplify(dA_mech.subs(bb, hb/(mu*c)))
chk("C4a", "Bekenstein 1973: Delta A_min = 8 pi G mu b/c^2 -> 8 pi l_P^2 at b = hbar/(mu c), kappa cancels",
    sp.simplify(dA_min_B - 8*sp.pi*lP2) == 0)
eta_Bek = sp.log(2)/(8*sp.pi*lP2)
chk("C4b", "one bit per 8 pi l_P^2 gives eta_B = ln2/(8 pi l_P^2); eta_BH/eta_B = 2 pi/ln2 = 9.06",
    abs(float(sp.simplify(eta_BH/eta_Bek)) - 2*math.pi/math.log(2)) < 1e-9, 2*math.pi/math.log(2))
chk("C4c", "with the correct eta, 8 pi l_P^2 carries 2 pi nats = Bekenstein bound 2 pi E R/(hbar c) of a Compton-size object",
    sp.simplify(eta_BH*dA_min_B - 2*sp.pi) == 0 and sp.simplify(2*sp.pi*(mu*c**2)*(hb/(mu*c))/(hb*c) - 2*sp.pi) == 0)
eps = sp.symbols('epsilon', positive=True)
chk("C5", "the first law has no minimum: energy eps*kT moves the area by eps*4 l_P^2 for any eps > 0",
    sp.simplify(8*sp.pi*G/(kap*c**2)*eps*k*Tunruh - eps*4*lP2) == 0)
# C6: the Landauer bit at T_H is a quantum of wavelength ~114 r_s
omega = k*TH*sp.log(2)/hb
lam = 2*sp.pi*c/omega
rs = 2*G*M/c**2
ratio = sp.simplify(lam/rs)
chk("C6", "a quantum carrying k T_H ln2 has wavelength (8 pi^2/ln2) r_s = 113.9 r_s",
    sp.simplify(ratio - 8*sp.pi**2/sp.log(2)) == 0, float(ratio))
# C7: Bekenstein bound on a carrier of size r_s with E = k T_H ln2
Smax = 2*sp.pi*(k*TH*sp.log(2))*rs/(hb*c)
chk("C7", "a carrier of size <= r_s holding E = k T_H ln2 can hold at most ln2/2 nats (half a bit)",
    sp.simplify(Smax - sp.log(2)/2) == 0)
# C8: Bekenstein-Mukhanov line: A_n = 4 ln2 l_P^2 n -> Delta(Mc^2) per step = k T_H ln2 (large n)
n = sp.symbols('n', positive=True)
Mn = sp.sqrt(4*sp.log(2)*lP2*n*c**4/(16*sp.pi*G**2))
dM = sp.diff(Mn, n)*c**2
chk("C8", "BM 1995 spectrum A_n = 4 ln2 l_P^2 n: d(Mc^2)/dn = k T_H ln2 (one Landauer bit per level)",
    sp.simplify(dM - (k*TH*sp.log(2)).subs(M, Mn)) == 0)

# ---------------- D. the cosmic horizon: booking decides the energy per nat ----------------
E_MS = c**4*R/(2*G)
A_R = 4*sp.pi*R**2
S_R = k*A_R/(4*lP2)
T_GH = hb*c/(2*sp.pi*k*R)
chk("D1a", "T_GH S = E_MS (capacity: ratio 1)", sp.simplify(T_GH*S_R - E_MS) == 0)
Tinc = sp.diff(E_MS, R)/sp.diff(S_R, R)
chk("D1b", "dE_MS/dS = T_GH/2 (increment temperature)", sp.simplify(Tinc - T_GH/2) == 0)
dA_tally = sp.diff(A_R, R)/sp.diff(E_MS, R)*k*T_GH
chk("D2", "tally booking: an increment k T_GH of Misner-Sharp energy moves the area by 8 l_P^2 (2 nats)",
    sp.simplify(dA_tally - 8*lP2) == 0)
chk("D3", "heat booking (Clausius, Cai-Kim): heat k T_GH raises S by k, area by 4 l_P^2",
    sp.simplify((k*T_GH/T_GH)/k * (4*lP2) - 4*lP2) == 0)
chk("D4", "area per nat is 4 l_P^2 at every horizon with S = kA/4l_P^2, whatever the temperature (definition)",
    sp.simplify(sp.diff(A_R, R)/sp.diff(S_R/k, R) - 4*lP2) == 0)
# D5: identical E(R), S(R) for hole at R = r_s and Hubble sphere at R = c/H; kappa differs by 2
chk("D5", "black hole: kappa = c^2/(2 r_s); de Sitter: kappa = c^2/R; same E(R), S(R)",
    sp.simplify(kapS.subs(M, R*c**2/(2*G)) - c**2/(2*R)) == 0 and sp.simplify(E_MS.subs(R, rs) - M*c**2) == 0)

# ---------------- E. numbers in the book ----------------
Msun = 1.98847e30
S_sun_nats = 4*math.pi*GN*Msun**2/(hbN*cN)
chk("E1a", "solar-mass hole: A/(4 l_P^2) = 1.05e77 nats = 1.51e77 bits", abs(S_sun_nats/1.05e77 - 1) < 0.01 and abs(S_sun_nats/math.log(2)/1.51e77 - 1) < 0.01,
    f"{S_sun_nats:.4e} nats, {S_sun_nats/math.log(2):.4e} bits")
chk("E1b", "book p6_05:36 prints 'A/4l_P^2 (1.5e77 bits)': the formula gives nats; 1.5e77 is the bit count A/(4 ln2 l_P^2)",
    abs(S_sun_nats/1.5e77 - 1) > 0.25 and abs(S_sun_nats/math.log(2)/1.5e77 - 1) < 0.01)
TH_sun = hbN*cN**3/(8*math.pi*GN*Msun*kN)
chk("E1c", "solar T_H = 6.2e-8 K; k T_H ln2 = 5.9e-31 J (p6_05:35-37)", abs(TH_sun/6.17e-8 - 1) < 0.01 and abs(kN*TH_sun*math.log(2)/5.9e-31 - 1) < 0.01,
    f"{TH_sun:.4e} K, {kN*TH_sun*math.log(2):.4e} J")
Mpc = 3.0856775814913673e22
def hubble(H0):
    Hs = H0*1e3/Mpc; Rh = cN/Hs
    nats = 4*math.pi*Rh**2/(4*lP2N)
    return Hs, Rh, nats
Hs, Rh, nats674 = hubble(67.4)
chk("E2a", "Hubble horizon (H0 67.4): A/(4 l_P^2) = 2.265e122 nats = 3.268e122 bits (p2_12:96)",
    abs(nats674/2.265e122 - 1) < 2e-3 and abs(nats674/math.log(2)/3.268e122 - 1) < 2e-3, f"{nats674:.4e}")
chk("E2b", "E_bit = hbar H0 ln2/(2 pi) = 2.541e-53 J at H0 67.4 (p2_12:88)", abs(hbN*Hs*math.log(2)/(2*math.pi)/2.541e-53 - 1) < 2e-3)
_, _, nats6716 = hubble(67.16)
chk("E2c", "eq:law_Sneed at H0 67.16: beta_m S_geo = 3.6e121 nats = 5.2e121 bits (p1_02:335 in bits; X6 in nats)",
    abs(0.15765*nats6716/3.6e121 - 1) < 0.01 and abs(0.15765*nats6716/math.log(2)/5.2e121 - 1) < 0.01,
    f"{0.15765*nats6716:.4e} nats")
chk("E3", "Planck kappa c^2/l_P = 5.56e51 m s^-2; cosmic kappa = c H0 = 6.5e-10 m s^-2 (p3_01a fig caption)",
    abs(cN**2/math.sqrt(lP2N)/5.56e51 - 1) < 2e-3 and abs(cN*Hs/6.5e-10 - 1) < 0.01)
# E4: a 'Planck tile' of area l_P^2 holding one bit would overcount by 4 ln2
chk("E4", "one bit per l_P^2 would give S = 4 ln2 x S_BH = 2.77 S_BH and T_H S = 1.39 Mc^2 (fails Smarr)",
    sp.simplify((k*sp.log(2)*A_s/lP2)/S_BH - 4*sp.log(2)) == 0 and abs(float(4*sp.log(2)/2) - 1.386294) < 1e-6)
# E5: Compton sphere (X19, p4_14:246): area count pi (m_P/m)^2 vs Bekenstein 2 pi
me = C.m_e; mP = math.sqrt(hbN*cN/GN)
chk("E5", "electron Compton sphere: area count pi (m_P/m_e)^2 = 1.79e45 vs Bekenstein bound 2 pi (p4_14:249-250)",
    abs(math.pi*(mP/me)**2/1.79e45 - 1) < 0.01, f"{math.pi*(mP/me)**2:.4e}")
# E6: T_GH S_geo / V_H = rho_crit c^2 (R14 (2))
chk("E6", "T_GH S_geo = E_MS = rho_crit c^2 V_H (one nat per 4 l_P^2 priced at T_GH carries the critical density)",
    sp.simplify(T_GH*S_R - (3*(c/R)**2/(8*sp.pi*G))*c**2*sp.Rational(4, 3)*sp.pi*R**3) == 0)

# ---------------- F. Koide paper Sec. III.D-E (as written) ----------------
# dimensions as (m, kg, s) exponent tuples
def dim(**e): return (e.get('m', 0), e.get('kg', 0), e.get('s', 0))
def mul(*ds): return tuple(sum(x) for x in zip(*ds))
def pw(d, p): return tuple(p*x for x in d)
dG = dim(m=3, kg=-1, s=-2); dc = dim(m=1, s=-1); dE = dim(m=2, kg=1, s=-2); dkapg = dim(m=-1)
d_eq8 = mul(dG, dE, pw(dc, -4))
chk("F1", "paper Eq. 8 'delta A = (8 pi G/c^4) E' has the dimension of a LENGTH (m), not an area; first law needs 1/kappa_g",
    d_eq8 == dim(m=1) and mul(d_eq8, pw(dkapg, -1)) == dim(m=2), d_eq8)
chk("F2", "paper Eq. 10 'eta delta A = 1 ... exactly one bit': eta counts nats; 1 nat = 1.4427 bits", abs(1/math.log(2) - 1.4427) < 1e-4)
x, y, s = sp.symbols('x y s', positive=True)
Qxy = (x**2 + y**2/2)/x**2/3     # Q = (x^2 + y^2/2)/(3 x^2) for the Brannen form
chk("F3", "Q is degree 0 in (x, y): any 'total W fixed by the area' cancels from Q, so Step 5's normalisation carries nothing to Koide",
    sp.simplify(Qxy.subs({x: s*x, y: s*y}) - Qxy) == 0)

out = {"script": "spider_s5_checks.py", "passed": sum(r["pass"] for r in RES), "failed": sum(not r["pass"] for r in RES), "checks": RES}
json.dump(out, open("spider_s5_checks_output.json", "w"), indent=1)
print(out["passed"], "passed;", out["failed"], "failed")
for r in RES:
    if not r["pass"]:
        print("FAIL", r["id"], r["desc"], r["value"])
