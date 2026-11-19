import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
"""bh_writer_checks.py  (2026-11-18, derivation spider, private)

ONE question: is BLACK-HOLE GROWTH the writer of the book's record term?

PASS BARS (copied unchanged from missing_writer_checks.py / floor_pricing_checks.py;
fixed here BEFORE any black-hole number is computed; never moved afterwards):
  AMPLITUDE passes if 0.5 <= R(z=0) <= 2, i.e. |log R(0)| <= log 2
            R = (ledger energy the writer delivers) / (ledger energy the book's record term needs)
  SHAPE     passes if 0.5 <= S(z) <= 2 at z = 1, 2, 3, i.e. |log S| <= log 2,  S(z) = R(z)/R(0)
  BINNED SHAPE (only for histories published as epoch shares, Merloni 2004):
            S_bin = [mean R over ln a in z in (1,3)] / [mean R over ln a in z in (0,1)], same 0.5-2 bar.

REQUIREMENT (book eq:law_Sneed, p1_02_iams_law.tex:330-333, beta_m = Om/2):
  B_req_HH(a)    = (1/2) E(a) a^2                   [units M_m(a) c^2 / k T_H(a) per e-fold]
  B_req_floor(a) = (1/2) E(a) a^2 / sqrt(OL(a))     [booking P: written at T_H, booked at T_dS]
  (floor_pricing_checks.py R5; reproduced here as R1-R3)

ROWS (declared before computing; every row is reported):
  W1a SMBH accretion, rest-mass energy dM c^2, BHAR = psi_MD14(z)/3300 (MD14 Fig.15 scaling)
  W1b SMBH accretion, rest-mass energy, MD14 shape normalised by Soltan to rho_BH0 = 4.6e5 (Marconi 2004)
  W1c SMBH accretion, rest-mass energy, Merloni 2004 epoch shares (half at z<1, 85% at z<3), BINNED shape
  W2  SMBH accretion, Smarr half d(T_BH S_BH) = (1/2) dM c^2 (book eq:qr_smarr, p4_14:275), W1b history
  W3  SMBH accretion heat radiated, L = eps/(1-eps) dM c^2, eps = 0.1 (MD14 sec 5.7), W1b history
  W4a SMBH Bekenstein-Hawking count dS_BH, S_BH = 4 pi G k M^2/(hbar c) per hole, summed through the published
      mass-function entropy s_SMBH,0 = 8.4e23 k m^-3 (Egan & Lineweaver 2010 Table 1, from the Graham 2007 BHMF);
      convention alpha = 2: every hole grows by the same fraction (mass function shifts in M)
  W4b same, alpha = 1: growth by new holes at fixed mass scale (number density grows)
  W4c W4a times (1 + 7.4/2.89) = merger upper end of the bottom-up count report (one equal-mass merger per hole
      per e-fold; an assumption of that report, not a measured rate); amplitude only, same history
  W5  stellar BH formation, rest mass: d rho_sBH/dt = f_BH psi(z), f_BH from rho_sBH,0 = 5e7 Msun/Mpc^3 (Sicilia+2022)
  W6a stellar BH formation, BH count, all holes 10 Msun;  W6b all holes 35 Msun (GWTC-3 mass peaks)
  W7a BBH mergers (GWTC-3), area-theorem count, m1=m2=10 Msun, R(0.2) = 17.9 Gpc^-3/yr, kappa = 2.9
  W7b same, m1=m2=35 Msun, R(0.2) = 44 Gpc^-3/yr (upper)
      remnant from NR: M_f = 0.95162 M, chi_f = 0.68646 (equal mass, non-spinning)
      z > 1: (1+z)^kappa continued (GWTC-3 constrains z <~ 1 only; provisional)
  W8  BBH mergers, GW energy radiated (4.838 % of M), W7b numbers (upper)
  W9a SMBH stock rest energy rho_BH(z) c^2, once per e-fold;  W9b its Smarr half;  W9c SMBH entropy stock, count
BOOKINGS (floor_pricing_checks.py R4: R -> R x T_L/T_W):
  energy rows:  HH (T_W = T_L = T_H; identical to FF), P (T_W = T_H, T_L = T_dS), RV (T_W = T_dS, T_L = T_H)
  count rows :  L_TH (ledger T_H), L_TdS (ledger T_dS).  A count is entropy itself; by the first law
                T_BH dS_BH = dM c^2 it is the energy written at the hole's own temperature T_BH.

A "check" is an identity (sympy), a unit conversion, or a reproduction of a published / stored number.
Whether a row passes is a RESULT and is not counted as a check.
"""
import json, sys
import numpy as np
import sympy as sp
from scipy import integrate

HIST = sys.argv[1]
hist = json.load(open(HIST))
OUT = {"bars": {"amp": [0.5, 2.0], "shape": [0.5, 2.0], "shape_z": [1, 2, 3]}}
CHECKS = []


def check(cid, cond, note):
    CHECKS.append({"id": cid, "pass": bool(cond), "note": note})
    print(("PASS " if cond else "FAIL ") + cid + ": " + note)


# ------------------------------------------------------------- constants (as the two prior scripts)
hbar = 1.054571817e-34; G = 6.67430e-11; c = 2.99792458e8; kB = 1.380649e-23
Msun = 1.98841e30; Mpc = 3.0856775814913673e22; yr = 3.15576e7; Gpc = 1e3 * Mpc
H0kms, Om, Ob, Or = 67.4, 0.3153, 0.0493, 9.22e-5
OL = 1 - Om - Or
H0 = H0kms * 1e3 / Mpc
zs = [0.0, 1.0, 2.0, 3.0]
EH = lambda z: np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL)
Ef = lambda z: np.exp(1 - (1 + z))          # E(a) = exp(1 - 1/a) = exp(-z)
OLa = lambda z: OL / EH(z) ** 2
T_H = lambda z: hbar * H0 * EH(z) / (2 * np.pi * kB)
rho_c0 = 3 * H0 ** 2 / (8 * np.pi * G)                     # kg/m^3
rho_m0 = Om * rho_c0
MsunMpc3 = Msun / Mpc ** 3                                 # kg/m^3 per (Msun/Mpc^3)

# ============================================================== R: requirement (sympy)
a, H0s, Gs, Oms, Ors, OLs, b, hb, kb, cs, M, Hs_ = sp.symbols(
    "a H_0 G Omega_m Omega_r Omega_L beta_m hbar k_B c M H", positive=True)
E = sp.exp(1 - 1 / a)
Hsq = H0s ** 2 * (Oms / a ** 3 + Ors / a ** 4 + OLs)
Hs = sp.sqrt(Hsq)
Oma = Oms * H0s ** 2 / (a ** 3 * Hsq); Ora = Ors * H0s ** 2 / (a ** 4 * Hsq); OLas = OLs * H0s ** 2 / Hsq
rho_info = 3 * H0s ** 2 * b * E / (8 * sp.pi * Gs)
Sdot = 8 * sp.pi ** 2 * rho_info / (3 * a * Hs ** 3)
unit_H = Oma * sp.pi / (Gs * Hs ** 2)                     # M_m c^2/kT_H = Om(a) S_geo (G=c=hbar=k=1 inside)
Breq_H = sp.simplify((Sdot / Hs).subs(b, Oms / 2) / unit_H)
check("R1", sp.simplify(Breq_H - E * a ** 2 / 2) == 0,
      "book eq:law_Sneed (beta_m = Om/2) -> B_req = E a^2/2 in units M_m c^2/kT_H (missing-writer R2, floor R1)")
Breq_P = sp.simplify(Breq_H / sp.sqrt(OLas))
check("R2", sp.simplify(sp.diff(sp.log(Breq_P), a) * a - (2 + 1 / a - (3 * Oma + 4 * Ora) / 2)) == 0,
      "floor booking P: B_req = (E a^2/2)/sqrt(OL(a)), log-slope (3+z) - (3 Om(a)+4 Or(a))/2 (floor R5, R6)")
num = {Oms: Om, Ors: Or, OLs: OL, H0s: 1}
fBH = sp.lambdify(a, Breq_H.subs(num), "numpy"); fBP = sp.lambdify(a, Breq_P.subs(num), "numpy")
check("R3", abs(2 * fBP(1.0) - 1 / np.sqrt(OL)) < 1e-12,
      "under P the all-matter writing fraction needed today = 1/sqrt(OL0) = %.4f (floor R8: 1.2086)" % (2 * fBP(1.0)))

# black-hole identities (sympy)
S_sch = 4 * sp.pi * Gs * kb * M ** 2 / (hb * cs)
T_bh = hb * cs ** 3 / (8 * sp.pi * Gs * M * kb)
check("K1", sp.simplify(sp.diff(S_sch, M) * T_bh - cs ** 2) == 0,
      "first law of a Schwarzschild hole: T_BH dS_BH/dM = c^2, so the count dS_BH equals dM c^2 written at T_BH")
check("K2", sp.simplify(T_bh * S_sch / (M * cs ** 2) - sp.Rational(1, 2)) == 0,
      "Smarr: T_BH S_BH = (1/2) M c^2 (book eq:qr_smarr, p4_14:275)")
TH_s = hb * Hs_ / (2 * sp.pi * kb)
Msol = sp.solve(sp.Eq(T_bh, TH_s), M)[0]
check("K3", sp.simplify(2 * Gs * Msol / cs ** 2 - cs / (2 * Hs_)) == 0,
      "T_BH = T_H only for M = c^3/(4 G H), i.e. r_s = c/(2H): a hole half the Hubble radius")
# count/energy ratio: count booking at T_H delivers (T_H/T_BH) of the energy booking, for every M
check("K4", sp.simplify((sp.diff(S_sch, M) * TH_s) / cs ** 2 - TH_s / T_bh) == 0,
      "count booked at T_H = energy booking x T_H/T_BH(M) exactly (both per unit accreted mass)")
chi = sp.symbols("chi", nonnegative=True)
S_kerr = 2 * sp.pi * Gs * kb * M ** 2 * (1 + sp.sqrt(1 - chi ** 2)) / (hb * cs)
check("K5", sp.simplify(S_kerr.subs(chi, 0) - S_sch) == 0, "Kerr entropy 2 pi G k M^2 (1+sqrt(1-chi^2))/(hbar c) -> Schwarzschild at chi = 0")
# entropy gain of an equal-mass non-spinning merger (NR remnant), and the count report's 80 %
Mf_frac, chif = 0.95162, 0.68646
gain = float(((2 * Mf_frac) ** 2 * (1 + np.sqrt(1 - chif ** 2)) / 2) / 2)   # S_f / (S1+S2), m1=m2
gain_schw = (2 * Mf_frac) ** 2 / 2
OUT["merger_entropy_gain"] = {"kerr_NR": gain, "schwarzschild_remnant": gain_schw}
check("K6", abs(gain_schw - 1.811) < 2e-3 and abs(gain - 1.564) < 2e-3,
      "equal-mass merger: S_f/(S1+S2) = %.3f with the NR Kerr remnant; %.3f if the remnant spin is ignored (bottom-up report's '+80 %%')" % (gain, gain_schw))

# requirement table
req = {}
for z in [0.0, 0.5, 0.75, 1.0, 2.0, 3.0]:
    aa = 1 / (1 + z)
    Sgeo = np.pi * c ** 5 / (hbar * G * (H0 * EH(z)) ** 2)
    Oma_n = Om * (1 + z) ** 3 / EH(z) ** 2
    req[str(z)] = {"B_req_HH": float(fBH(aa)), "B_req_P": float(fBP(aa)),
                   "shape_HH": float(fBH(aa) / fBH(1.0)), "shape_P": float(fBP(aa) / fBP(1.0)),
                   "nats_per_efold_HubbleSphere_HH": float(fBH(aa) * Oma_n * Sgeo),
                   "nats_per_efold_HubbleSphere_P": float(fBP(aa) * Oma_n * Sgeo),
                   "sqrt_OL_a": float(np.sqrt(OLa(z))), "E_H": float(EH(z))}
OUT["requirement"] = req
S_geo0_6716 = np.pi * c ** 5 / (hbar * G * (67.16e3 / Mpc) ** 2)
check("R4", abs(Om / 2 * S_geo0_6716 / 3.597e121 - 1) < 2e-3,
      "beta_m S_geo at H0 = 67.16 = %.4e nats per e-fold (bottom-up report 3.597e121)" % (Om / 2 * S_geo0_6716))

# ============================================================== published inputs
psi = lambda z: 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)     # MD14 eq.15, Msun/yr/Mpc^3
H_yr = lambda z: H0 * EH(z) * yr                                           # 1/yr
dtdz = lambda z: 1 / ((1 + z) * H_yr(z))                                   # yr
ZMAX = 10.0
form_tot = integrate.quad(lambda z: psi(z) * dtdz(z), 0, ZMAX, limit=200)[0]
check("P1", abs(form_tot * (1 - 0.27) / 5.9e8 - 1) < 0.03,
      "MD14 psi integrated (z<10) x (1-R), R = 0.27: rho_* = %.3e Msun/Mpc^3 (MD14 / missing-writer D2: ~5.8-5.9e8)" % (form_tot * 0.73))
BHAR0_direct = psi(0) / 3300
check("P2", abs(BHAR0_direct / 4.53e-6 - 1) < 3e-3,
      "BHAR(0) = psi(0)/3300 = %.3e Msun/yr/Mpc^3 (bottom-up report D1: 4.53e-6)" % BHAR0_direct)
frac_lt = lambda z1: integrate.quad(lambda z: psi(z) * dtdz(z), 0, z1)[0] / form_tot
OUT["MD14_shape_fraction_accreted"] = {"z<1": frac_lt(1.0), "z<3": frac_lt(3.0)}
rhoBH0 = {"Marconi2004": 4.6e5, "Shankar2009_lo": 3.2e5, "Shankar2009_hi": 5.4e5, "MerloniHeinz2008": 4.3e5}
dir_total = BHAR0_direct / psi(0) * form_tot
OUT["MD14_direct_total_accreted_Msun_Mpc3"] = dir_total
# reproduce the bottom-up report's 2.9e102 (alpha = 2, Shankar 2004 rho0 = 4.2e5, H0 = 67.16, V_H)
H0b = 67.16e3 / Mpc
VH = 4 * np.pi / 3 * (c / H0b) ** 3
S_stock = 8.4e23 * VH
dlnrho = BHAR0_direct / (H0b * yr) / 4.2e5
check("P3", abs(S_stock / 9.20e102 - 1) < 5e-3 and abs(dlnrho / 0.157 - 1) < 0.01 and abs(2 * S_stock * dlnrho / 2.89e102 - 1) < 0.01,
      "bottom-up report reproduced: S_SMBH(Hubble sphere) = %.3e, dln rho/dln a = %.3f, alpha=2 count = %.3e nats/e-fold" % (S_stock, dlnrho, 2 * S_stock * dlnrho))
# implied entropy-weighted mass <M^2>/<M> of the published stock
kS = 4 * np.pi * G * kB * Msun ** 2 / (hbar * c) / kB                      # nats per Msun^2
M_S = 8.4e23 / (kS * 5.0e5 * MsunMpc3 / Msun)                               # Graham-2007-consistent rho 5.0e5
OUT["SMBH_entropy_weighted_mass_Msun"] = M_S
check("P4", 1e8 < M_S < 2e9, "entropy-weighted SMBH mass <M^2>/<M> from s = 8.4e23 k/m^3 and rho = 5.0e5: %.2e Msun (stock dominated by ~1e9 holes, Egan-Lineweaver)" % M_S)

# ============================================================== writer histories B_HH(z)
# B (units M_m(a) c^2/kT_H(a) per e-fold) = [energy per comoving volume per e-fold] / (rho_m0 c^2)   (T_W = T_H)
#                                         = [nats per comoving volume per e-fold] kT_H(z) / (rho_m0 c^2) (count)
zg = np.linspace(0, 8, 161)


def rhoBH_of(z, rho0):     # Soltan: rho_BH(z) = rho0 [1 - F(<z)], MD14 shape normalised to rho0 (z<10)
    return rho0 * (1 - frac_lt(z))


def B_energy_from_rate(rate_Msun_yr_Mpc3, z):      # rate per comoving volume -> per e-fold -> units of rho_m0 c^2
    return rate_Msun_yr_Mpc3 / H_yr(z) * MsunMpc3 / rho_m0


def B_count_from_nats(nats_per_m3_per_efold, z):
    return nats_per_m3_per_efold * kB * T_H(z) / (rho_m0 * c ** 2)


rows = {}
r0 = rhoBH0["Marconi2004"]
BHAR_norm = lambda z: r0 * psi(z) / form_tot
rows["W1a"] = ("SMBH accretion rest energy, BHAR = psi/3300", "energy", lambda z: B_energy_from_rate(psi(z) / 3300, z))
rows["W1b"] = ("SMBH accretion rest energy, MD14 shape, Soltan-normalised to 4.6e5", "energy", lambda z: B_energy_from_rate(BHAR_norm(z), z))
rows["W2"] = ("SMBH accretion Smarr half (1/2) dM c^2", "energy", lambda z: 0.5 * B_energy_from_rate(BHAR_norm(z), z))
rows["W3"] = ("SMBH accretion heat radiated, eps = 0.1", "energy", lambda z: (0.1 / 0.9) * B_energy_from_rate(BHAR_norm(z), z))
s0 = 8.4e23                                                                  # nats per m^3 (comoving, z = 0)
for key, alpha in [("W4a", 2), ("W4b", 1)]:
    def f(z, alpha=alpha):
        x = rhoBH_of(z, r0) / r0
        dlnrho_dlna = BHAR_norm(z) / H_yr(z) / rhoBH_of(z, r0)
        return B_count_from_nats(alpha * s0 * x ** alpha * dlnrho_dlna, z)
    rows[key] = ("SMBH Bekenstein-Hawking count, alpha = %d" % alpha, "count", f)
rows["W4c"] = ("SMBH count alpha=2 x merger upper end (x3.56, assumption)", "count", lambda z: (1 + 7.4 / 2.89) * rows["W4a"][2](z))
rho_s0 = 5e7
fBHs = rho_s0 / form_tot
OUT["stellar_BH_mass_fraction_of_formed_stars"] = fBHs
rows["W5"] = ("stellar BH formation rest energy (Sicilia 2022 rho = 5e7)", "energy", lambda z: B_energy_from_rate(fBHs * psi(z), z))
for key, m in [("W6a", 10.0), ("W6b", 35.0)]:
    rows[key] = ("stellar BH formation count, holes of %g Msun" % m, "count",
                 lambda z, m=m: B_count_from_nats(fBHs * psi(z) / H_yr(z) * kS * m * (1 / Mpc ** 3), z))
Rz = lambda z, R02, kap=2.9: R02 * ((1 + z) / 1.2) ** kap / Gpc ** 3       # per m^3 per yr (comoving, source frame)
for key, m, R02 in [("W7a", 10.0, 17.9), ("W7b", 35.0, 44.0)]:
    dS = kS * m ** 2 * (gain - 1) * 2                                        # nats per merger: S_f - S1 - S2
    rows[key] = ("BBH merger count, %g+%g Msun, R(0.2) = %g" % (m, m, R02), "count",
                 lambda z, dS=dS, R02=R02: B_count_from_nats(Rz(z, R02) * dS / H_yr(z), z))
rows["W8"] = ("BBH merger GW energy, 35+35, R(0.2) = 44", "energy",
              lambda z: Rz(z, 44.0) * 0.04838 * 70 * Msun / H_yr(z) / rho_m0)
rows["W9a"] = ("SMBH stock rest energy once per e-fold", "energy", lambda z: rhoBH_of(z, r0) * MsunMpc3 / rho_m0)
rows["W9b"] = ("SMBH stock Smarr half once per e-fold", "energy", lambda z: 0.5 * rhoBH_of(z, r0) * MsunMpc3 / rho_m0)
rows["W9c"] = ("SMBH entropy stock (count) once per e-fold, alpha = 2", "count",
               lambda z: B_count_from_nats(s0 * (rhoBH_of(z, r0) / r0) ** 2, z))

# checks on the histories
check("P6", abs(rows["W1a"][2](0) / 0.5 / 3.3e-6 - 1) < 0.08,
      "W1a amplitude today R = %.3e reproduces the bottom-up report's 3.3e-6 (H0 differs: 67.4 vs 67.16)" % (rows["W1a"][2](0) / 0.5))

# ============================================================== score every row under every booking
mult_E = {"HH": lambda z: 1.0, "P": lambda z: np.sqrt(OLa(z)), "RV": lambda z: 1 / np.sqrt(OLa(z))}
mult_C = {"L_TH": lambda z: 1.0, "L_TdS": lambda z: np.sqrt(OLa(z))}


def verdict(R):
    S = [R[i] / R[0] for i in range(1, 4)]
    return {"amp": R[0], "S": S, "log10_amp": float(np.log10(R[0])),
            "amp_pass": bool(0.5 <= R[0] <= 2), "shape_pass": bool(all(0.5 <= s <= 2 for s in S))}


def mu_of(z, X):          # book eq:tw_mu (p2_18:67), X = record term in units of rho_c0
    HL2 = Om * (1 + z) ** 3 + OL
    return HL2 / (HL2 + X)


lna = -np.log(1 + zg)
scored = []
for key, (lab, kind, f) in rows.items():
    Bz = np.array([f(z) for z in zs])
    for bk, m in (mult_E if kind == "energy" else mult_C).items():
        R = [Bz[i] * m(zs[i]) / fBH(1 / (1 + zs[i])) for i in range(4)]
        v = verdict(R)
        # forward record by the book's rule d(rho_info/rho_c0)/dln a = B x mult x Om a^-3, from z = 8
        dens = np.array([f(z) * m(z) for z in zg]) * Om * (1 + zg) ** 3
        cm = np.array([-np.trapezoid(dens[k:], lna[k:]) for k in range(len(zg))])
        X = lambda zz: float(np.interp(zz, zg, cm))
        tot = cm[0]
        v.update({"key": key, "label": lab, "kind": kind, "booking": bk,
                  "record_today_over_beta_m": float(tot / (Om / 2)),
                  "mu_absolute": {str(zz): mu_of(zz, X(zz)) for zz in [0.0, 0.5, 0.75, 1.0]},
                  "mu_shape_only": {str(zz): mu_of(zz, (Om / 2) * X(zz) / tot) for zz in [0.0, 0.5, 0.75, 1.0]},
                  "B_HH": [float(x) for x in Bz]})
        v["mu0_minus_1_absolute"] = v["mu_absolute"]["0.0"] - 1
        v["mu0_minus_1_shape_only"] = v["mu_shape_only"]["0.0"] - 1
        scored.append(v)

# W1c: Merloni 2004 epoch shares, binned (energy): mean dF/dln a over z in (0,1) = 0.5/ln2, over (1,3) = 0.35/ln2
binned = []
for lab, r in rhoBH0.items():
    for bk, m in mult_E.items():
        lna_b = lambda z0, z1: np.log((1 + z1) / (1 + z0))
        Bm01 = 0.50 * r * MsunMpc3 / rho_m0 / lna_b(0, 1)
        Bm13 = 0.35 * r * MsunMpc3 / rho_m0 / lna_b(1, 3)
        def meanreq(z0, z1):
            zz = np.linspace(z0, z1, 201)
            y = fBH(1 / (1 + zz)) / np.array([m(x) for x in zz])
            return -np.trapezoid(y, -np.log(1 + zz)) / lna_b(z0, z1)
        R01, R13 = Bm01 / meanreq(0, 1), Bm13 / meanreq(1, 3)
        binned.append({"key": "W1c", "rho_BH0": lab, "booking": bk, "R_mean_z0_1": R01, "R_mean_z1_3": R13,
                       "S_bin": R13 / R01, "amp_pass": bool(0.5 <= R01 <= 2), "shape_pass": bool(0.5 <= R13 / R01 <= 2)})
OUT["rows"] = scored
OUT["binned_rows"] = binned
OUT["passes"] = [(r["key"], r["booking"]) for r in scored if r["amp_pass"] and r["shape_pass"]] + \
                [("W1c", r["rho_BH0"], r["booking"]) for r in binned if r["amp_pass"] and r["shape_pass"]]
OUT["shape_passes"] = [(r["key"], r["booking"], r["amp"]) for r in scored if r["shape_pass"]] + \
                      [("W1c", r["rho_BH0"], r["booking"], r["R_mean_z0_1"]) for r in binned if r["shape_pass"]]
OUT["amp_passes"] = [(r["key"], r["booking"]) for r in scored if r["amp_pass"]]

# ============================================================== the dispatch curve a black-hole writer would need
need = {}
for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
    aa = 1 / (1 + z)
    need[str(z)] = {
        "BHAR_needed_over_today_HH": float(fBH(aa) / fBH(1.0) * EH(z)),
        "BHAR_needed_over_today_P": float(fBH(aa) / fBH(1.0) * EH(z) / (np.sqrt(OLa(z)) / np.sqrt(OL))),
        "BHAR_published_over_today_MD14": float(psi(z) / psi(0)),
        "count_dSdt_needed_over_today_HH": float(fBH(aa) / fBH(1.0) * EH(z) / EH(z) * 1.0),
    }
OUT["dispatch_needed"] = need

# ============================================================== book mu reproduced, and scale of what BHs would need
bookmu = {str(z): mu_of(z, (Om / 2) * np.exp(-z)) for z in [0.0, 0.5, 0.75, 1.0]}
check("F1", abs(bookmu["0.5"] - 0.9482) < 2e-4 and abs(bookmu["0.75"] - 0.970) < 1e-3 and abs(bookmu["1.0"] - 0.9822) < 2e-4
      and abs(bookmu["0.0"] - 1 + 0.136) < 1e-3 and abs(bookmu["0.5"] - hist["book E(a) (measured shape)"]["mu"]["0.5"]) < 1e-4,
      "book mu(z) reproduced: %.4f, %.4f, %.4f, %.4f at z = 0, 0.5, 0.75, 1 (book 0.864 [mu0-1 = -0.136], 0.948, 0.970, 0.982)"
      % (bookmu["0.0"], bookmu["0.5"], bookmu["0.75"], bookmu["1.0"]))
OUT["book_mu"] = bookmu
# method check: forward rule with B = B_req returns beta_m [E(1) - E(1/9)]
dens = fBH(1 / (1 + zg)) * Om * (1 + zg) ** 3
check("F2", abs(-np.trapezoid(dens, lna) / ((Om / 2) * (1 - np.exp(-8))) - 1) < 2e-3,
      "forward rule with B = B_req returns beta_m [E(1) - E(1/9)] (method check, as missing-writer F1)")
# what rho_BH would have to be to supply the amplitude today with T_W = T_L = T_H: (dρ/dln a)/rho_m0 = 1/2
need_rate = 0.5 * rho_m0 / MsunMpc3 * H_yr(0)
OUT["BHAR_needed_today_Msun_yr_Mpc3_HH"] = need_rate
OUT["checks"] = CHECKS
OUT["checks_passed"] = sum(x["pass"] for x in CHECKS)
OUT["checks_failed"] = sum(not x["pass"] for x in CHECKS)
json.dump(OUT, open("bh_writer_checks_output.json", "w"), indent=1, default=float)
print("passed", OUT["checks_passed"], "failed", OUT["checks_failed"])
for r in scored:
    print("%-4s %-6s amp %.2e  S %s  rec/beta %.2e  mu0-1 %.2e  shape-only mu %s" % (
        r["key"], r["booking"], r["amp"], ", ".join("%.3g" % s for s in r["S"]), r["record_today_over_beta_m"],
        r["mu0_minus_1_absolute"], ", ".join("%.4f" % r["mu_shape_only"][k] for k in ["0.0", "0.5", "0.75", "1.0"])))
for r in binned:
    print("W1c %-16s %-3s R01 %.2e R13 %.2e S_bin %.3g" % (r["rho_BH0"], r["booking"], r["R_mean_z0_1"], r["R_mean_z1_3"], r["S_bin"]))
print("dispatch", json.dumps(need))
print("frac", OUT["MD14_shape_fraction_accreted"], "direct total", dir_total, "fBHs", fBHs, "need", need_rate, "M_S", M_S)
print("passes", OUT["passes"], "shape passes", OUT["shape_passes"])
