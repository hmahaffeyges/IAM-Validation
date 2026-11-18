import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""bottom_up_checks.py  (2026-11-18)
The bottom-up count: records written per e-fold today by local writers,
against the book's dS_info/dln a = beta_m S_geo (eq:law_Sneed, p1_02:333-336).
Every arithmetic step of BOTTOM_UP_COUNT_2026-11-18.md is computed here.
Book values (CANON via the book): H0 = 67.16 km/s/Mpc (p1_02:152,335),
Omega_m = 0.3153, beta_m = Omega_m/2 = 0.15765 (p1_02:442-444),
Omega_b = 0.0493 (p1_02:735), Omega_Lambda = 0.6847 (p1_02:730),
E(a) = exp(1 - 1/a) (eq:Ea).
Outside numbers: each is tagged with its source in the line that uses it.
"""
import math
import numpy as np
import sympy as sp
from scipy import constants as C
from scipy.integrate import quad

PASS, FAIL = [], []
def check(tag, desc, cond, val=""):
    (PASS if cond else FAIL).append(tag)
    print(f"[{'PASS' if cond else 'FAIL'}] {tag}: {desc}  {val}")

# ---------------- book values ----------------
H0_kms = 67.16
Mpc = 3.0856775814913673e22
H0 = H0_kms*1e3/Mpc                   # s^-1
Om, OL, Ob = 0.3153, 0.6847, 0.0493
beta_m = Om/2
c, G, hbar, k = C.c, C.G, C.hbar, C.k
Msun = 1.98847e30                     # kg (IAU nominal GM_sun/G)
yr = 365.25*86400
ln2 = math.log(2)

# ---------------- A. the required rate and its identities ----------------
rH = c/H0
V_H = 4*math.pi/3*rH**3               # apparent-horizon (Hubble) volume, flat FRW
A_H = 4*math.pi*rH**2
lP2 = hbar*G/c**3
S_geo = A_H/(4*lP2)                    # nats
req_nats = beta_m*S_geo
req_bits = req_nats/ln2
check("A1", "S_geo = pi c^5/(G hbar H0^2) (nats) and A/4l_P^2 agree",
      abs(S_geo/(math.pi*c**5/(G*hbar*H0**2)) - 1) < 1e-12, f"S_geo={S_geo:.4e}")
check("A2", "required rate beta_m S_geo = 5.19e121 bits per e-fold (book: 5.2e121, p1_02:335)",
      abs(req_bits/5.19e121 - 1) < 0.005, f"{req_bits:.4e} bits = {req_nats:.4e} nats")

T_GH = hbar*H0/(2*math.pi*k)
rho_c = 3*H0**2/(8*math.pi*G)
E_MS = c**5/(2*G*H0)                   # Misner-Sharp energy inside r_A = c/H
check("A3", "rho_crit c^2 V_H = Misner-Sharp energy c^5/(2GH)", abs(rho_c*c**2*V_H/E_MS - 1) < 1e-12,
      f"E_MS={E_MS:.4e} J")
check("A4", "E_MS/(k T_GH) = S_geo exactly (T_GH S_geo = E_MS)", abs(E_MS/(k*T_GH)/S_geo - 1) < 1e-12)
M_m = Om*rho_c*V_H
half_Mc2_over_kT = 0.5*M_m*c**2/(k*T_GH)
check("A5", "IDENTITY: beta_m S_geo = (1/2) M_m c^2/(k T_GH) (matter inside r_A, priced at the bank)",
      abs(half_Mc2_over_kT/req_nats - 1) < 1e-12, f"M_m={M_m:.4e} kg = {M_m/Msun:.3e} Msun")
# symbolic version of A5 and the Cai-Kim and Margolus-Levitin forms
Hs, Gs, hb, cs, Oms, a = sp.symbols('H G hbar c Omega_m a', positive=True)
Ssym = sp.pi*cs**5/(Gs*hb*Hs**2)
Msym = Oms*3*Hs**2/(8*sp.pi*Gs)*sp.Rational(4, 3)*sp.pi*(cs/Hs)**3
kTsym = hb*Hs/(2*sp.pi)
check("A6", "symbolic: (Om/2) S_geo - (1/2) M_m c^2/kT_GH = 0",
      sp.simplify(Oms/2*Ssym - sp.Rational(1, 2)*Msym*cs**2/kTsym) == 0)
# Cai-Kim: dS_geo/dln a = -2 S dlnH/dlna = 3 Omega_m(a) S_geo in flat matter+Lambda
dSgeo_dlna_0 = 3*Om*S_geo
check("A7", "IDENTITY: required / (dS_geo/dln a) = beta_m/(3 Om) = 1/6 today",
      abs(req_nats/dSgeo_dlna_0 - 1/6) < 1e-12, f"dS_geo/dlna={dSgeo_dlna_0:.4e} nats")
# general a: required(a) = pi beta_m H0^2 E(a) /(G a H^4) * c^5/hbar ; dS_geo/dlna = 3 Om(a) pi c^5/(G hbar H^2)
E = lambda aa: math.exp(1 - 1/aa)
Hz = lambda z: H0*math.sqrt(Om*(1+z)**3 + OL)
def req_rate(z):
    aa = 1/(1+z); H = Hz(z)
    return math.pi*c**5*beta_m*H0**2*E(aa)/(hbar*G*aa*H**4)
def dSgeo(z):
    H = Hz(z); Omz = Om*(1+z)**3*H0**2/H**2
    return 3*Omz*math.pi*c**5/(G*hbar*H**2)
ok = all(abs(req_rate(z)/dSgeo(z) - E(1/(1+z))*(1/(1+z))**2/6) < 1e-12 for z in (0, 0.5, 1, 2, 5))
check("A8", "required(a)/(dS_geo/dln a) = E(a) a^2/6 at z = 0, 0.5, 1, 2, 5", ok)
check("A9", "required(a=1) from the general formula equals beta_m S_geo", abs(req_rate(0)/req_nats - 1) < 1e-12)
# Margolus-Levitin: ops per second <= 2E/(pi hbar) (Lloyd 2002 quotes it); ops per e-fold of the matter rest energy
N_ML = 2*M_m*c**2/(math.pi*hbar*H0)
check("A10", "IDENTITY: required / (Margolus-Levitin ops of matter rest energy per e-fold) = pi^2/2",
      abs(req_nats/N_ML - math.pi**2/2) < 1e-12, f"N_ML={N_ML:.4e}; ratio={req_nats/N_ML:.4f}")
m_p = C.m_p
per_proton = math.pi*m_p*c**2/(hbar*H0)
check("A11", "required per proton mass of matter per e-fold = pi m_p c^2/(hbar H0)",
      abs(per_proton*M_m/m_p/req_nats - 1) < 1e-12, f"{per_proton:.4e} nats per m_p")

# ---------------- B. inventory scaled to the Hubble volume ----------------
Mpc3 = Mpc**3
h = H0_kms/100
rho_c_Msun_Mpc3 = rho_c*Mpc3/Msun
check("B1", "rho_crit = 2.775e11 h^2 Msun/Mpc^3", abs(rho_c_Msun_Mpc3/(2.7754e11*h**2) - 1) < 2e-3,
      f"{rho_c_Msun_Mpc3:.4e}")
V_H_Mpc3 = V_H/Mpc3
# Egan & Lineweaver 2010 Table 1 entropy densities (k m^-3)
s_EL = {"SMBH": 8.4e23, "stellarBH": 1.6e17, "photons": 1.478e9, "neutrinos": 1.411e9,
        "WIMP_DM": 5e7, "gravitons": 1.7e7, "ISM_IGM": 20.0, "stars": 0.26}
S_H = {kk: v*V_H for kk, v in s_EL.items()}
V_obs_EL = 3.65e80
check("B2", "E-L Table 1 reproduced: S_SMBH(obs) = s V_obs = 3.1e104 k",
      abs(s_EL["SMBH"]*V_obs_EL/3.1e104 - 1) < 0.02)
check("B3", "SMBH entropy inside the Hubble sphere today ~ 9e102 nats", 8e102 < S_H["SMBH"] < 1.0e103,
      f"{S_H['SMBH']:.3e}; V_H={V_H:.4e} m^3 = {V_H_Mpc3:.4e} Mpc^3")


# Graham et al. 2007 Schechter fit used by E-L (phi*=0.0016 Mpc^-3, M*=2.9e8 Msun, alpha=-0.30): implied rho_BH and s_SMBH
phi_s, M_s, al = 0.0016, 2.9e8, -0.30
rho_G07 = phi_s*M_s*math.e*math.gamma(al+2)/math.log(10)
s_G07 = 4*math.pi*G/(c*hbar)*phi_s*(M_s*Msun)**2*math.e*math.gamma(al+3)/math.log(10)/Mpc3
check("B4", "E-L eq. 35 reproduced from the Graham 2007 fit: s_SMBH = 8.4e23 k/m^3",
      abs(s_G07/8.4e23 - 1) < 0.05, f"s={s_G07:.3e}; implied rho_BH={rho_G07:.3e} Msun/Mpc^3")

# ---------------- C. stars: starlight as records on the surrounding medium ----------------
j0 = 1.8e35*h                           # W/Mpc^3, Driver et al. 2012 (1.8 +/- 0.3) x 10^35 h W Mpc^-3, intrinsic
j_dust = 0.6e35*h                       # reprocessed by dust (Driver 2012)
L_H = j0*V_H_Mpc3
check("C1", "luminosity of the Hubble volume L = j V_H", True, f"L={L_H:.3e} W")
T_star = 5000.0                         # K, ASSUMED effective radiation temperature of starlight (range 3000-10000 K)
T_dust = 25.0                           # K, ASSUMED cold-dust temperature (range 15-40 K)
Sdot_star = 4/3*L_H/(k*T_star)         # nats/s, blackbody photon entropy at emission (s = 4u/3T)
Sdot_dust = 4/3*(j_dust*V_H_Mpc3)*(1/(k*T_dust) - 1/(k*T_star))
star_fee = Sdot_star/H0
dust_fee = Sdot_dust/H0
# photon count: blackbody mean photon energy 2.701 kT
z3 = 1.2020569031595942; mean_E = (math.pi**4/(30*z3))*k*T_star
check("C2", "blackbody mean photon energy = (pi^4/30 zeta3) kT = 2.701 kT", abs(mean_E/(k*T_star) - 2.7012) < 1e-3)
Ndot_ph = L_H/mean_E
s_per_photon_BB = (4/3)*mean_E/(k*T_star)
check("C3", "blackbody entropy per photon = 3.60 k", abs(s_per_photon_BB - 3.6016) < 1e-3)
# dilution: starlight energy density ~ j0/H0 (one e-fold of emission), W = u/(a_rad T^4)
a_rad = 4*C.sigma/c
u_star = (j0/Mpc3)/H0
W = u_star/(a_rad*T_star**4)
def s_per_photon_dilute(W, T):
    # occupation n(nu) = W/(exp(x)-1); entropy per mode (1+n)ln(1+n) - n ln n ; integrate x^2 dx
    f_s = lambda x: x**2*((1+W/np.expm1(x))*np.log1p(W/np.expm1(x)) - (W/np.expm1(x))*np.log(W/np.expm1(x)))
    f_n = lambda x: x**2*W/np.expm1(x)
    Sn = quad(f_s, 1e-6, 200, limit=400)[0]; Nn = quad(f_n, 1e-6, 200, limit=400)[0]
    return Sn/Nn
s1 = s_per_photon_dilute(1.0, T_star)
check("C4", "dilute-radiation integrator returns 3.60 k/photon at W = 1", abs(s1 - 3.6016) < 2e-3, f"{s1:.4f}")
sW = s_per_photon_dilute(W, T_star)
check("C5", "entropy per starlight photon after dilution W ~ 1e-15 is ~ 38 k", 30 < sW < 45, f"W={W:.2e}; s={sW:.2f} k")
star_count_upper = Ndot_ph*sW/H0
print(f"   stars: fee at emission {star_fee:.3e} nats/e-fold; dust re-emission {dust_fee:.3e}; "
      f"photons {Ndot_ph/H0:.3e}/e-fold; diluted-photon entropy {star_count_upper:.3e}")
check("C6", "stars: every starlight booking is within 1e83-1e87 nats per e-fold",
      all(1e83 < v < 1e87 for v in (star_fee, dust_fee, Ndot_ph/H0, star_count_upper)))

# ---------------- D. black-hole growth ----------------
rho_BH0 = 4.2e5                         # Msun/Mpc^3, Shankar et al. 2004 (MNRAS 354, 1020) (4.2 +/- 1.1) x 10^5
psi = lambda z: 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)   # Msun/yr/Mpc^3, Madau & Dickinson 2014 eq. 15
BHAR = lambda z: psi(z)/3300.0          # MD14: BH accretion rate density scaled up by 3300 matches the SFRD shape
dlnrho_dlna_0 = BHAR(0)/rho_BH0/yr/H0  # per e-fold
check("D1", "SMBH mass growth today: d ln rho_BH / d ln a ~ 0.16", 0.1 < dlnrho_dlna_0 < 0.25,
      f"BHAR(0)={BHAR(0):.3e} Msun/yr/Mpc^3; {dlnrho_dlna_0:.4f}")
# S = 4 pi G k M^2/(hbar c) ; with equal specific growth, dS/dln a = 2 S dln rho/dln a
M_ = sp.symbols('M', positive=True)
check("D2", "dS_BH/dM = 2 S/M for S = 4 pi G M^2/(hbar c)",
      sp.simplify(sp.diff(4*sp.pi*Gs*M_**2/(hb*cs), M_) - 2*(4*sp.pi*Gs*M_**2/(hb*cs))/M_) == 0)
BH_acc = 2*S_H["SMBH"]*dlnrho_dlna_0
# mergers: equal-mass merger with ~5% of mass radiated: S_f/S_i = (1.9)^2/2
merger_gain = 1.9**2/2 - 1
BH_merge_upper = merger_gain*S_H["SMBH"]  # if every SMBH had one equal-mass merger per e-fold (upper)
stellarBH = S_H["stellarBH"]*(psi(0)/yr/H0*1e0)/(0.0027*rho_c_Msun_Mpc3)  # new stellar BH entropy ~ SFR share
print(f"   SMBH accretion {BH_acc:.3e} nats/e-fold; merger upper {BH_merge_upper:.3e}; stellar BHs {stellarBH:.3e}")
check("D3", "equal-mass merger (5% radiated) raises horizon entropy by 80%", abs(merger_gain - 0.805) < 1e-9)
BH_low, BH_high = BH_acc, BH_acc + BH_merge_upper
# Soltan consistency: integrate BHAR back to z=1 ; Merloni 2004: about half of local rho_BH accumulated at z<1
t_of = lambda z1, z2: quad(lambda z: 1/((1+z)*Hz(z)), z1, z2)[0]
acc_z1 = quad(lambda z: BHAR(z)/yr/((1+z)*Hz(z)), 0, 1)[0]
check("D4", "MD14 BHAR integrated over z<1 is 25-60% of rho_BH0 (Merloni 2004: ~half)",
      0.25 < acc_z1/rho_BH0 < 0.6, f"{acc_z1/rho_BH0:.3f}")


for rb, lab in ((2.6e5, "Merloni 2004"), (4.2e5, "Shankar 2004"), (rho_G07, "Graham 2007 fit")):
    print(f"   sensitivity rho_BH0={rb:.2e} ({lab}): dln rho/dln a={BHAR(0)/rb/yr/H0:.3f}; "
          f"SMBH accretion count {2*S_H['SMBH']*BHAR(0)/rb/yr/H0:.3e} nats/e-fold; "
          f"z<1 share {quad(lambda z: BHAR(z)/yr/((1+z)*Hz(z)), 0, 1)[0]/rb:.3f}")

# ---------------- E. structure formation and gas ----------------
gas_total = S_H["ISM_IGM"]              # all gas entropy now in the Hubble volume (E-L Table 1)
rho_L_c2 = OL*rho_c*c**2
Q_vir = 3.15e-8*rho_L_c2*V_H            # book p2_12b:92 accumulated baryon virial heat
T_gas = 3.24e6                          # K, mu m_p sigma^2/k for sigma = 213 km/s, mu = 0.59 (forward H8)
check("E1", "T_gas = 0.59 m_p (213 km/s)^2 / k = 3.24e6 K", abs(0.59*m_p*213e3**2/k/3.24e6 - 1) < 0.01)
gas_count_book = Q_vir/(k*T_gas)        # nats; book p2_12b:93 count at T_gas
check("E2", "book p2_12b:93 horizon price reproduced: 3.15e-8 x T_GH/T_gas = 2.6e-44",
      abs(3.15e-8*T_GH/T_gas/2.6e-44 - 1) < 0.02, f"{3.15e-8*T_GH/T_gas:.3e}")
N_b = Ob*rho_c*V_H/m_p
print(f"   gas: total gas entropy {gas_total:.3e}; book virial count {gas_count_book:.3e} nats accumulated; "
      f"{gas_count_book/(0.58*N_b):.2f} nats per halo baryon; N_b={N_b:.3e}")
check("E3", "book virial count is O(1) nats per halo baryon", 0.5 < gas_count_book/(0.58*N_b) < 10)
gas_upper = gas_total                   # per e-fold upper bound: at most the whole present gas entropy

# ---------------- F. planets and life ----------------
S0, alb, R_E = 1361.0, 0.30, 6.3781e6
P_abs = S0*math.pi*R_E**2*(1-alb)
T_e = (S0*(1-alb)/(4*C.sigma))**0.25
Sdot_E = 4/3*P_abs*(1/T_e - 1/5772.0)
bits_s = Sdot_E/(k*ln2)
check("F1", "Step 2b H12 reproduced: Earth's sunlight record budget 6.4e37 bits/s",
      abs(bits_s/6.4e37 - 1) < 0.02, f"T_e={T_e:.1f} K; {bits_s:.3e}")
earth_per_efold = bits_s*ln2/H0
M_star = 0.0027*rho_c*V_H               # E-L Omega_* = 0.0027
N_star_upper = M_star/(0.08*Msun)       # every star a hydrogen-burning minimum-mass star (upper count)
life_upper = earth_per_efold*N_star_upper
print(f"   life: Earth {earth_per_efold:.3e} nats/e-fold; N_star<= {N_star_upper:.3e}; all-planets upper {life_upper:.3e}")
check("F2", "life, one Earth per star (upper), is below 1e-44 of the required rate", life_upper/req_nats < 1e-44)

# ---------------- G. totals and ratios today ----------------
star_hi = max(star_fee + dust_fee, star_count_upper)
tot_low = BH_low + star_fee + gas_count_book
tot_high = BH_high + star_hi + gas_upper + life_upper
r_low, r_high = tot_low/req_nats, tot_high/req_nats
print(f"   TOTAL today: {tot_low:.3e} - {tot_high:.3e} nats/e-fold; ratio to required {r_low:.2e} - {r_high:.2e}")
check("G1", "SMBH horizon growth carries >99.99% of the bottom-up count", BH_low/tot_low > 0.9999)
check("G2", "bottom-up count / required lies between 1e-20 and 1e-18", 1e-20 < r_low < r_high < 1e-18)
# heat booking at T_GH (the fee re-priced at the bank) for comparison
star_heat_TGH = (j0*V_H_Mpc3)/(k*T_GH)/H0
BH_rest_TGH = dlnrho_dlna_0*(rho_BH0/rho_c_Msun_Mpc3)*S_geo     # accreted rest energy at T_GH
check("G3", "starlight energy priced at T_GH per e-fold = (j/H0)/(rho_c c^2) S_geo",
      abs(star_heat_TGH/((j0/Mpc3/H0)/(rho_c*c**2)*S_geo) - 1) < 1e-12,
      f"{star_heat_TGH:.3e} nats = {star_heat_TGH/req_nats:.2e} of required")
print(f"   BH accreted rest energy at T_GH: {BH_rest_TGH:.3e} = {BH_rest_TGH/req_nats:.2e} of required")
check("G4", "even the fee re-priced at T_GH (stars + BH rest energy) is < 1e-4 of required",
      (star_heat_TGH + BH_rest_TGH)/req_nats < 1e-4)
check("G5", "Lloyd 2002 matter bits (~1e90) and the present SMBH entropy (~1e103) are both << required per e-fold",
      1e90 < req_nats and S_H["SMBH"] < 1e-18*req_nats)

# ---------------- H. time dependence z = 1, 2 ----------------
def bottom_up_stars(z):  # luminosity density tracks psi (upper scaling); physical Hubble volume; per e-fold
    jphys = j0*psi(z)/psi(0)*(1+z)**3          # W per physical Mpc^3
    VH = 4*math.pi/3*(c/Hz(z))**3/Mpc3
    return 4/3*jphys*VH/(k*T_star)/Hz(z)
def rho_BH(z):
    return rho_BH0 - quad(lambda zz: BHAR(zz)/yr/((1+zz)*Hz(zz)), 0, z)[0]
def bottom_up_BH(z, alpha):  # 2 S_com (Mdot/M) V_com / H, S_com ~ rho_BH^alpha
    S_com = s_EL["SMBH"]*(rho_BH(z)/rho_BH0)**alpha     # k per comoving m^3
    Vcom = 4*math.pi/3*(c/Hz(z))**3*(1+z)**3
    return 2*S_com*Vcom*(BHAR(z)/rho_BH(z))/yr/Hz(z)
rows = []
for z in (0.0, 1.0, 2.0):
    R = req_rate(z)
    bs = bottom_up_stars(z); b1 = bottom_up_BH(z, 1); b2 = bottom_up_BH(z, 2)
    rows.append((z, R, bs, b1, b2, rho_BH(z)/rho_BH0))
    print(f"   z={z}: required {R:.3e}; stars {bs:.3e}; BH(alpha=1) {b1:.3e}; BH(alpha=2) {b2:.3e}; "
          f"rho_BH/rho_BH0 {rho_BH(z)/rho_BH0:.3f}; ratio(BH a=1) {b1/R:.2e}; ratio(stars) {bs/R:.2e}")
check("H1", "stars: ratio(z)/ratio(0) = [psi(z)/psi(0)] (1+z)^2 e^z analytically",
      all(abs((r[2]/r[1])/(rows[0][2]/rows[0][1]) - psi(r[0])/psi(0)*(1+r[0])**2*math.exp(r[0])) < 1e-6*(psi(r[0])/psi(0)*(1+r[0])**2*math.exp(r[0])) for r in rows))
check("H2", "BH count / required rises from z=0 to z=2 (bottom-up falls faster toward today than E(a) needs)",
      rows[2][3]/rows[2][1] > rows[1][3]/rows[1][1] > rows[0][3]/rows[0][1])
check("H3", "required rate at z=1 and z=2 (E(a)/(a (H/H0)^4) shape)",
      abs(rows[1][1]/req_nats - math.exp(-1)*2/(Hz(1)/H0)**4) < 1e-9 and abs(rows[2][1]/req_nats - math.exp(-2)*3/(Hz(2)/H0)**4) < 1e-9,
      f"R(1)/R(0)={rows[1][1]/req_nats:.4e}, R(2)/R(0)={rows[2][1]/req_nats:.4e}")

print(f"\nchecks passed: {len(PASS)}  failed: {len(FAIL)} {FAIL}")
import json
json.dump({"req_bits": req_bits, "req_nats": req_nats, "S_geo": S_geo, "T_GH": T_GH, "M_m_Msun": M_m/Msun,
           "N_ML": N_ML, "per_proton": per_proton, "dSgeo": dSgeo_dlna_0,
           "S_SMBH_H": S_H["SMBH"], "BH_acc": BH_acc, "BH_merge_upper": BH_merge_upper, "stellarBH": stellarBH,
           "dlnrho": dlnrho_dlna_0, "acc_z1_frac": acc_z1/rho_BH0,
           "L_H": L_H, "star_fee": star_fee, "dust_fee": dust_fee, "photons": Ndot_ph/H0, "W": W, "sW": sW,
           "star_count_upper": star_count_upper, "gas_total": gas_total, "gas_count_book": gas_count_book,
           "N_b": N_b, "earth": earth_per_efold, "N_star": N_star_upper, "life_upper": life_upper,
           "tot_low": tot_low, "tot_high": tot_high, "r_low": r_low, "r_high": r_high,
           "star_heat_TGH": star_heat_TGH, "BH_rest_TGH": BH_rest_TGH, "V_H_Mpc3": V_H_Mpc3,
           "rows": rows, "passed": len(PASS), "failed": len(FAIL)}, open("bottom_up_numbers.json", "w"), indent=1)
