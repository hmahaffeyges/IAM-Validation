import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""time_rate_checks.py -- TIME_AND_RATE_2026-11-18 (private, proposals only).
Every derived step of the report: maximum actualization rates vs the record term,
the two clocks (H vs H_m), null geodesics and c.  sympy for identities, scipy for integrals.
Inputs: book eq:law_Sneed (p1_02_iams_law.tex:333), eq Hm (p2_11_dark_energy.tex:58),
beta_m = Om/2, E(a)=exp(1-1/a).  Light (<100 MB, a few seconds)."""
import json, math
import sympy as sp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

checks = []; nums = {}
def check(cid, ok, what):
    checks.append({"id": cid, "result": "pass" if ok else "FAIL", "what": what})

# ---------------- constants (CODATA 2018 exact/recommended) ----------------
c = 299792458.0; hbar = 1.054571817e-34; h = 2*math.pi*hbar; G = 6.67430e-11
kB = 1.380649e-23; mp = 1.67262192369e-27; Mpc = 3.0856775814913673e22
Gyr = 3.15576e16; lnn2 = math.log(2)
tP = math.sqrt(hbar*G/c**5)

# ---------------- symbols ----------------
a, m, E_, R, H, H0, Om, bm, GG, hb, cc, k = sp.symbols('a m E R H H0 Omega_m beta_m G hbar c k_B', positive=True)
Ea = sp.exp(1-1/a)

# ======================= PART A: maximum rates =======================
Gam_ML   = 2*E_/(sp.pi*hb)                    # Margolus-Levitin ops/s, energy E above ground
Gam_MT   = 2*(E_/2)/(sp.pi*hb)                # Mandelstam-Tamm, Delta E <= E/2 for spectrum in [0,E]
I_Bek81  = sp.pi*E_/hb                        # Bekenstein 1981 PRL rate, nats/s (pi E/(hbar ln2) bits/s)
S_Bek    = 2*sp.pi*E_*R/(hb*cc)               # Bekenstein 1981 PRD entropy bound, nats
Brem     = E_/(2*sp.pi*hb)*sp.log(2)          # Bremermann mc^2/h bits/s -> nats/s

# A0: Bekenstein's rate = his entropy bound / message crossing time 2R/c
check("A0", sp.simplify(S_Bek/(2*R/cc) - I_Bek81) == 0,
      "Bekenstein 1981 rate pi E/hbar (nats/s) = entropy bound 2 pi E R/(hbar c) divided by crossing time 2R/c")
# A1: Lloyd 2000 ultimate laptop 2mc^2/(pi hbar) for 1 kg
lap = 2*1.0*c**2/(math.pi*hbar); nums["ML_1kg_ops_per_s"] = lap
check("A1", abs(lap/5.4258e50-1) < 2e-4, f"Margolus-Levitin for 1 kg = {lap:.5e} ops/s (Lloyd 2000: 5.4258e50)")
# A2: Bremermann c^2/h per kg
br = c**2/h; nums["Bremermann_bits_per_s_per_kg"] = br
check("A2", abs(br/1.3563925e50-1) < 1e-6, f"Bremermann c^2/h = {br:.7e} bits/s/kg (published 1.3563925e50)")

# A3: required per-particle rate from eq:law_Sneed, all symbolic
rho_info = 3*H0**2*bm*Ea/(8*sp.pi*GG)                     # mass density units with c: energy = rho c^2
Sdot = 8*sp.pi**2*(rho_info)/(3*a*H**3) * cc**5/(hb*GG**0) # book form in G=c=hbar=1; restore below
# do it cleanly in SI: S_geo = pi c^5/(hbar G H^2); dS_info/dln a = beta_m E H0^2/(a H^2) * S_geo (from eq:law_Sneed)
S_geo = sp.pi*cc**5/(hb*GG*H**2)
dSinfo_dlna_book = 8*sp.pi**2*(3*H0**2*bm*Ea/(8*sp.pi))/(3*a*H**4)   # G=c=hbar=1 form, then times c^5/(hbar G) units
dSinfo_dlna = dSinfo_dlna_book * cc**5/(hb*GG)   # (8pi^2 rho/3aH^4) with rho = 3H0^2 bE/(8pi G) carries 1/G -> c^5/(hbar G)
check("A3a", sp.simplify(dSinfo_dlna.subs(a,1).subs(H,H0) - bm*S_geo.subs(H,H0)) == 0,
      "eq:law_Sneed at a=1 reproduces dS_info/dln a = beta_m S_geo (book p1_02:333)")
Hsq = H0**2*Om/a**3   # matter part only needed for N; Omega_m(a)=Om a^-3 H0^2/H^2
rho_m = 3*H0**2*Om/(8*sp.pi*GG*a**3)
N = rho_m*(sp.Rational(4,3)*sp.pi*(cc/H)**3)/m
Gam_req = sp.simplify(H*dSinfo_dlna/N)
check("A3", sp.simplify(Gam_req - 2*sp.pi*(bm/Om)*Ea*a**2*m*cc**2/hb) == 0,
      "required nats per matter particle per second = 2 pi (beta_m/Om) E(a) a^2 m c^2/hbar; H cancels")
Gam_req_b = sp.simplify(Gam_req.subs(bm, Om/2))
check("A3b", sp.simplify(Gam_req_b - sp.pi*Ea*a**2*m*cc**2/hb) == 0, "with beta_m = Om/2: pi E(a) a^2 m c^2/hbar (missing-writer R3)")
# A4..A8 ratios
Ep = m*cc**2
r_ML  = sp.simplify(Gam_req_b/Gam_ML.subs(E_,Ep))
r_MT  = sp.simplify(Gam_req_b/Gam_MT.subs(E_,Ep))
r_B81 = sp.simplify(Gam_req_b/I_Bek81.subs(E_,Ep))
r_BLC = sp.simplify(Gam_req_b/(S_Bek.subs(E_,Ep)/(R/cc)))
r_Br  = sp.simplify(Gam_req_b/Brem.subs(E_,Ep))
check("A4", sp.simplify(r_ML - sp.pi**2/2*Ea*a**2) == 0, "Gamma_req/Gamma_ML = (pi^2/2) E a^2 (= 4.93 today; missing-writer R4 verified)")
check("A5", sp.simplify(r_B81 - Ea*a**2) == 0, "Gamma_req / Bekenstein-1981 rate (pi m c^2/hbar nats/s) = E(a) a^2: EXACTLY 1 today")
check("A6", sp.simplify(r_BLC - Ea*a**2/2) == 0, "Gamma_req / (Bekenstein entropy bound per light-crossing R/c) = E a^2/2, independent of R")
check("A7", sp.simplify(r_MT - sp.pi**2*Ea*a**2) == 0, "Gamma_req / Mandelstam-Tamm (Delta E <= mc^2/2) = pi^2 E a^2")
check("A8", sp.simplify(r_Br - 2*sp.pi**2/sp.log(2)*Ea*a**2) == 0, "Gamma_req / Bremermann = (2 pi^2/ln 2) E a^2 = 28.5 today")
# A9: the identity behind A5: Bekenstein rate per Hubble time, priced at T_H
TH = hb*H/(2*sp.pi*k)
check("A9", sp.simplify(I_Bek81/H - E_/(2*k*TH)) == 0 and sp.simplify(S_Bek.subs(R,cc/H) - E_/(k*TH)) == 0,
      "pi E/(hbar H) = E/(2 k T_H) and Bekenstein bound at R=c/H = E/(k T_H): pricing at T_H IS the Bekenstein bound of the Hubble radius")
# numbers
def Ef(av): return math.exp(1-1/av)
tab = []
for z in [0,1,2,3]:
    av = 1/(1+z); x = Ef(av)*av**2
    tab.append({"z": z, "E": Ef(av), "Ea2": x, "req_over_ML": math.pi**2/2*x, "req_over_Bek81": x,
                "req_over_Bek_per_crossing": x/2, "req_over_MT": math.pi**2*x, "req_over_Bremermann": 2*math.pi**2/lnn2*x,
                "ML_supplies_fraction_at_1nat_per_op": 1/(math.pi**2/2*x)})
nums["rate_table"] = tab
Ep_p = mp*c**2
nums["proton"] = {"ML_ops_s": 2*Ep_p/(math.pi*hbar), "Bek81_nats_s": math.pi*Ep_p/hbar,
                  "Bek81_bits_s": math.pi*Ep_p/(hbar*lnn2), "Gamma_req_today": math.pi*Ep_p/hbar,
                  "Bremermann_bits_s": Ep_p/h}
check("A10", abs(nums["proton"]["Gamma_req_today"]/4.48e24-1) < 3e-3, f"proton Gamma_req today = {math.pi*Ep_p/hbar:.4e} /s (missing-writer: 4.48e24)")
# future: when does requirement exceed Bekenstein rate? E a^2 = 1 at a = 1 exactly; >1 after
check("A11", abs(Ef(1.0)-1) < 1e-15 and Ef(1.5)*1.5**2 > 1 and Ef(0.99)*0.99**2 < 1,
      "E a^2 crosses 1 exactly at a=1: before today below the Bekenstein-1981 rate, after today above it")
nums["Ea2_future"] = {str(av): Ef(av)*av**2 for av in [1.25,1.5,2,3,10]}

# A12: epoch at which each bound stops sufficing (all matter writes at the bound, 1 nat per op/nat)
cross = {}
for lab, fac in [("Margolus-Levitin", math.pi**2/2), ("Mandelstam-Tamm", math.pi**2), ("Bekenstein-1981", 1.0),
                 ("Bekenstein-per-light-crossing", 0.5), ("Bremermann", 2*math.pi**2/lnn2)]:
    ac = brentq(lambda av: fac*Ef(av)*av**2 - 1, 0.05, 5)
    cross[lab] = {"a": ac, "z": 1/ac-1}
nums["bound_crossover"] = cross
check("A12", abs(cross["Bekenstein-1981"]["a"]-1) < 1e-9, "requirement meets the Bekenstein-1981 rate at a = 1 exactly; other bounds' crossover epochs computed")

# ---------------- black hole and horizon ----------------
M, Hh = sp.symbols('M H_h', positive=True)
Rs = 2*GG*M/cc**2
S_BH = 4*sp.pi*GG*M**2/(hb*cc)
check("H0", sp.simplify(S_Bek.subs({E_:M*cc**2, R:Rs}) - S_BH) == 0, "black hole saturates Bekenstein: 2 pi Mc^2 R_s/(hbar c) = 4 pi G M^2/(hbar c)")
ops_per_nat_per_cross_BH = sp.simplify(Gam_ML.subs(E_,M*cc**2)/S_BH*(Rs/cc))
check("H0b", sp.simplify(ops_per_nat_per_cross_BH - 1/sp.pi**2) == 0, "black hole at Margolus-Levitin: 1/pi^2 operations per nat per light-crossing R_s/c (Lloyd 2000: a bit flips about once per crossing time, up to pi^2/ln2)")
MH = cc**3/(2*GG*H)   # mass of Hubble sphere at critical density
check("H1", sp.simplify(sp.Rational(4,3)*sp.pi*(cc/H)**3*3*H**2/(8*sp.pi*GG) - MH) == 0 and sp.simplify(TH*k*S_geo - MH*cc**2) == 0,
      "M_H = c^3/(2GH); T_H S_geo = M_H c^2")
check("H2", sp.simplify(S_Bek.subs({E_:MH*cc**2, R:cc/H}) - S_geo) == 0 and sp.simplify(2*GG*MH/cc**2 - cc/H) == 0,
      "Hubble sphere at critical density saturates Bekenstein exactly (S_geo), and c/H is its own Schwarzschild radius")
check("H3", sp.simplify(Gam_ML.subs(E_,MH*cc**2) - S_geo*H/sp.pi**2) == 0, "horizon energy at Margolus-Levitin: S_geo H/pi^2 ops/s (1/pi^2 ops per nat per Hubble time, as for a black hole)")
OmA = sp.symbols('Omega_ma', positive=True)   # Omega_m(a)
rec_rate = Ea*a**2/6*3*OmA*S_geo*H             # dS_info/dt
check("H4", sp.simplify((rec_rate/(S_geo*H)).subs({a:1, OmA:2*bm}) - bm) == 0,
      "record rate / (Bekenstein bound of Hubble sphere per light-crossing) = (E a^2/2) Om(a); = beta_m today")
# numbers today
H0SI = 67.4e3/Mpc; Om0 = 0.3153; bm0 = Om0/2
Sg0 = math.pi*c**5/(hbar*G*H0SI**2)
nums["horizon_today_H0_67.4"] = {"S_geo_nats": Sg0, "S_geo_bits": Sg0/lnn2, "Bek_per_crossing_nats_s": Sg0*H0SI,
    "ML_ops_s": Sg0*H0SI/math.pi**2, "record_rate_nats_s": bm0*Sg0*H0SI, "dSgeo_dt_nats_s": 3*Om0*Sg0*H0SI,
    "record_nats_per_ML_op": bm0*math.pi**2, "dSgeo_over_Bek_per_crossing": 3*Om0,
    "record_per_tile_per_tile_crossing": bm0*H0SI*2*math.sqrt(lnn2)*tP}
# Omega_m(a)=1/3 crossing (dS_geo/dt = S_geo H)
OL0 = 1-Om0
acr = (2*Om0/OL0)**(1/3); nums["a_where_dSgeo_dt_equals_SgeoH"] = acr
check("H5", abs(Om0*acr**-3/(Om0*acr**-3+OL0) - 1/3) < 1e-12, f"horizon area growth dS_geo/dt = 3 Om(a) S_geo H equals S_geo H at a = {acr:.4f} (z = {1/acr-1:.4f}); above it before")
# Lloyd 2002 cross-check: total ML ops of the horizon mass over history, int S_geo H/pi^2 dt = int (pi c^5/(hbar G H^2))/pi^2 * H dt
Orad = 4.18e-5/0.674**2
def Eh(av, Om=Om0, Or=Orad): return math.sqrt(Om*av**-3 + Or*av**-4 + (1-Om-Or))
Itot = quad(lambda av: (c**5/(hbar*G*(H0SI*Eh(av))**2))/math.pi / (av*H0SI*Eh(av)) * (H0SI*Eh(av)), 1e-12, 1, limit=400)[0]
# int ops = int (S_geo H/pi^2) dt = int S_geo/pi^2 dln a
Itot = quad(lambda lna: (c**5/(hbar*G*(H0SI*Eh(math.exp(lna)))**2))/math.pi, math.log(1e-12), 0, limit=400)[0]
nums["horizon_ML_ops_whole_history"] = Itot
check("H6", 1e119 < Itot < 1e122, f"horizon mass at Margolus-Levitin over the whole history = {Itot:.2e} ops (Lloyd 2002: ~1e120); record accumulated 1.11e121 nats (missing-writer)")

# ======================= PART B: two clocks =======================
def Hm_over_H0(av, H0k, Om, Or):
    return math.sqrt(Eh(av,Om,Or)**2 + (Om/2)*Ef(av))
def age(H0k, Om, Or, matter=False, zmin=0.0):
    H0s = H0k*1e3/Mpc
    f = (lambda av: 1/(av*Hm_over_H0(av,H0k,Om,Or))) if matter else (lambda av: 1/(av*Eh(av,Om,Or)))
    return quad(f, 1e-10, 1/(1+zmin), limit=500, epsabs=0, epsrel=1e-10)[0]/H0s/Gyr
# B1 reproduce book two-ruler table (H0=67.16, Om=0.3153, no radiation)
tbl = {}
for av in [1,2,10,1e6]:
    Hph = 67.16*Eh(av,Om0,0.0); Hm = 67.16*Hm_over_H0(av,67.16,Om0,0.0); tbl[str(av)] = (round(Hph,2), round(Hm,2), round(Hm/Hph,3))
nums["two_ruler_reproduced"] = tbl
book = {"1":(67.16,72.26,1.076),"2":(57.15,66.62,1.166),"10":(55.59,69.56,1.251)}
okB1 = all(abs(tbl[k_][0]-v[0])<0.011 and abs(tbl[k_][1]-v[1])<0.011 for k_,v in book.items())
check("B1", okB1 and abs(tbl["1000000.0"][1]-70.86)<0.01, "book two-ruler table (p2_11_dark_energy.tex:180-186) reproduced: H, H_m at a=1,2,10,inf")
check("B2", abs(math.sqrt(1+bm0)-1.0759) < 5e-5, "H_m/H today = sqrt(1+beta_m) = 1.0759 (book p2_06:346)")
clock = {}
for (lab,H0k) in [("L2_67.16",67.16),("Planck_67.36",67.36),("Planck_67.4",67.4)]:
    Or = 4.18e-5/(H0k/100)**2
    tg = age(H0k,Om0,Or); tm = age(H0k,Om0,Or,matter=True)
    lb = {}
    for z in [0.5,1,2,3]:
        lb[str(z)] = {"photon": tg-age(H0k,Om0,Or,zmin=z), "matter": tm-age(H0k,Om0,Or,True,zmin=z)}
    clock[lab] = {"age_photon_Gyr": tg, "age_matter_Gyr": tm, "difference_Gyr": tg-tm, "lookback": lb,
                  "age_LCDM_at_H0_times_sqrt(1+bm)": age(H0k*math.sqrt(1+bm0),Om0,4.18e-5/(H0k*math.sqrt(1+bm0)/100)**2),
                  "age_photon_no_radiation": age(H0k,Om0,0.0), "age_matter_no_radiation": age(H0k,Om0,0.0,True)}
nums["ages"] = clock
check("B3", abs(clock["Planck_67.36"]["age_photon_Gyr"]-13.8) < 0.05, f"LCDM age at 67.36/0.3153 = {clock['Planck_67.36']['age_photon_Gyr']:.3f} Gyr (Planck/Valcin quote 13.8 +/- 0.02)")
rates = {}
for z in [0,0.5,1,2,3]:
    av = 1/(1+z); r_ = Hm_over_H0(av,67.16,Om0,0.0)/Eh(av,Om0,0.0)
    rates[str(z)] = {"Hm_over_H": r_, "dtau_m_dt": 1/r_}
nums["clock_rates"] = rates
# book appendix numbers at Level-2 posterior (Om=0.3166): z=1 H=120.44, Hm=121.53
Hz1 = 67.161*Eh(0.5,0.3166,0.0); Hmz1 = 67.161*math.sqrt(Eh(0.5,0.3166,0.0)**2+0.3166/2*Ef(0.5))
check("B4", abs(Hz1-120.44)<0.02 and abs(Hmz1-121.53)<0.02, f"book app_C3:278 z=1 rates reproduced: H={Hz1:.2f}, H_m={Hmz1:.2f}")
# B5 Pound-Rebka prediction, B6 GPS
g=9.80665; hPR=22.6
nums["PoundRebka_gh_over_c2"] = g*hPR/c**2
check("B5", abs(g*hPR/c**2 - 2.47e-15) < 0.02e-15, f"Pound-Rebka predicted fractional shift gh/c^2 over 22.6 m = {g*hPR/c**2:.3e}")
GM_E = 3.986004418e14; RE = 6.371e6; rGPS = 26.5616e6
grav = GM_E/c**2*(1/RE-1/rGPS)*86400e6; vel = -(GM_E/rGPS)/(2*c**2)*86400e6
nums["GPS_us_per_day"] = {"gravitational": grav, "velocity": vel, "net": grav+vel}
check("B6", abs(grav-45.7)<0.3 and abs(grav+vel-38.6)<0.4, f"GPS clock offsets: +{grav:.1f} grav, {vel:.1f} velocity, net {grav+vel:.1f} us/day (Ashby 2003)")
# record clock rate (per particle per second) relative to today
nums["record_clock_rate_rel_today"] = {str(z): Ef(1/(1+z))/(1+z)**2 for z in [0,0.5,1,2,3]}

# ======================= PART C: null geodesics and c =======================
t, chi = sp.symbols('t chi', real=True); af = sp.Function('a')(t)
dt_, dchi = sp.symbols('dt dchi')
ds2 = -cc**2*dt_**2 + af**2*dchi**2
check("C1", sp.simplify(ds2.subs(dchi, cc*dt_/af)) == 0, "FRW radial null ray dchi = c dt/a gives ds^2 = 0, d tau = 0 identically")
r_, rs_ = sp.symbols('r r_s', positive=True)
ds2S = -(1-rs_/r_)*cc**2*dt_**2 + dchi**2/(1-rs_/r_)
check("C2", sp.simplify(ds2S.subs(dchi, cc*(1-rs_/r_)*dt_)) == 0, "Schwarzschild radial null ray dr/dt = c(1 - r_s/r): ds^2 = 0 (coordinate speed varies, local speed c)")
D26 = 26*Mpc
up = c*1.74/D26; lo = c*(1.74-10)/D26
nums["GW170817"] = {"upper": up, "lower": lo}
check("C3", abs(up/7e-16-1) < 0.1 and abs(lo/-3e-15-1) < 0.1, f"GW170817 interval reproduced from 1.74 s over 26 Mpc (+{up:.1e}) and a 10 s emission allowance ({lo:.1e})")
# C4: propagation effect needed for a 7.6% siren offset at GW170817's redshift
zG = 0.0098
delta_needed = math.log(math.sqrt(1+bm0))/zG
nums["siren_friction_needed"] = delta_needed
check("C4", delta_needed > 1, f"a path-accumulated GW amplitude effect giving d_GW/d_EM = 1.0759 at z={zG} needs an effective friction coefficient ~{delta_needed:.1f} per unit z (any such ratio -> 1 as z -> 0)")
# C5: light-travel lag from a local write to the apparent (Hubble) horizon, LCDM background (H0=67.4)
def chi_(ae, ar): return quad(lambda av: 1/(av**2*Eh(av,Om0,Orad)), ae, ar, limit=300)[0]
lag = {}
for z in [0,1,2,3]:
    ae = 1/(1+z)
    f = lambda ar: chi_(ae,ar) - 1/(ar*Eh(ar,Om0,Orad))
    ar = brentq(f, ae*1.0000001, 50)
    tlag = quad(lambda av: 1/(av*Eh(av,Om0,Orad)), ae, ar)[0]/H0SI/Gyr
    lag[str(z)] = {"a_write": ae, "a_reach_horizon": ar, "dln_a": math.log(ar/ae), "years_Gyr": tlag,
                   "E_write": Ef(ae), "E_reach": Ef(ar)}
nums["write_to_horizon_lag"] = lag
check("C5", all(v["dln_a"] > 0 for v in lag.values()), "a light signal from a local write reaches the Hubble sphere after a finite lag (computed per z)")
# C6: Hubble sphere recession speed dR/dt = c(1+q); in matter era 3c/2
q = sp.symbols('q'); Ht = sp.Function('H')(t)
Rh = cc/Ht
check("C6", sp.simplify(sp.diff(Rh,t).subs(sp.Derivative(Ht,t), -(1+q)*Ht**2) - cc*(1+q)) == 0, "Hubble sphere recedes at dR_H/dt = c(1+q) (faster than c when q > 0)")

# C7: which write epoch reaches the Hubble sphere exactly today (signal reading of the ledger)
def reach(ae):
    return brentq(lambda ar: chi_(ae,ar) - 1/(ar*Eh(ar,Om0,Orad)), ae*1.0000001, 50)
ae_today = brentq(lambda ae: reach(ae) - 1.0, 0.2, 0.99)
nums["write_epoch_reaching_horizon_today"] = {"a_e": ae_today, "z_e": 1/ae_today-1, "E_at_write": Ef(ae_today)}
check("C7", abs(reach(ae_today)-1) < 1e-8, f"signal reading: records booked at the Hubble sphere today were written at z = {1/ae_today-1:.3f} (E = {Ef(ae_today):.3f})")
nums["Bek81_bits_per_s_per_kg"] = math.pi*c**2/(hbar*lnn2)
nums["record_over_horizon_ML_history"] = 1.11e121/Itot

out = {"checks": checks, "n_pass": sum(1 for x in checks if x["result"]=="pass"),
       "n_fail": sum(1 for x in checks if x["result"]!="pass"), "numbers": nums}
json.dump(out, open("time_rate_checks_output.json","w"), indent=1, default=float)
print(out["n_pass"], out["n_fail"])
for x in checks:
    if x["result"]!="pass": print(x)
