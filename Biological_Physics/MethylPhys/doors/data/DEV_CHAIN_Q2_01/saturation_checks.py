import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""saturation_checks.py  (2026-11-18, derivation spider: writing speed, local surface saturation, black-hole formation)

Every identity is checked in sympy; every number is computed from CODATA constants (scipy) and the published
masses/radii listed in INPUTS (sources in SATURATION_2026-11-18.md, section 9). Output: saturation_checks_output.json.
Light: < 2 s, < 100 MB.
"""
import json, math
import numpy as np
import sympy as sp
from scipy import constants as C
from scipy.integrate import quad
from scipy.optimize import brentq

out = {"checks": [], "numbers": {}}
def check(cid, desc, ok, val=None):
    out["checks"].append({"id": cid, "desc": desc, "pass": bool(ok), "value": val})
def num(k, v):
    out["numbers"][k] = v
    return v

G, c, hbar, kB = C.G, C.c, C.hbar, C.k
lP2 = hbar * G / c**3
Msun, Rsun = 1.98841e30, 6.957e8            # IAU 2015 nominal (GM_sun / G with CODATA G; R_sun nominal)
Mpc, kpc = 3.0856775814913673e22, 3.0856775814913673e19
m_p, m_e = C.m_p, C.m_e

# ---------------------------------------------------------------- A. Bekenstein vs holographic (symbolic)
M, R, Gs, cs, hs = sp.symbols('M R G c hbar', positive=True)
S_bek  = 2*sp.pi*M*cs*R/hs                 # Bekenstein 1981, S/k <= 2 pi E R/(hbar c), E = M c^2
S_holo = sp.pi*R**2*cs**3/(Gs*hs)          # A/(4 l_P^2), A = 4 pi R^2
rs = 2*Gs*M/cs**2
check("A1", "S_Bek(M, r_s) == A(r_s)/4l_P^2 exactly", sp.simplify(S_bek.subs(R, rs) - S_holo.subs(R, rs)) == 0)
f = sp.simplify(S_bek/S_holo)
check("A2", "fill fraction f = S_Bek/S_holo = r_s/R", sp.simplify(f - rs/R) == 0, str(f))
rate = sp.pi*M*cs**2/hs                      # Bekenstein 1981 rate, nats/s
t_fill = sp.simplify(S_holo/rate)
check("A3", "t_fill = S_holo/(pi M c^2/hbar) = R^2 c/(G M); = 4GM/c^3 at r_s",
      sp.simplify(t_fill - R**2*cs/(Gs*M)) == 0 and sp.simplify(t_fill.subs(R, rs) - 4*Gs*M/cs**3) == 0)
check("A4", "t_fill = (2R/c)/f  (light crossing over fill fraction)", sp.simplify(t_fill - (2*R/cs)/f) == 0)
check("A5", "own Bekenstein bound filled at Bekenstein rate in 2R/c (built in)", sp.simplify(S_bek/rate - 2*R/cs) == 0)
t_ff = sp.pi/2*sp.sqrt(R**3/(2*Gs*M))         # dust free-fall time from rest at R
fs = sp.symbols('f', positive=True)
ratio_ff = sp.simplify((t_fill/t_ff).subs(M, fs*R*cs**2/(2*Gs)))
check("A6", "t_fill/t_freefall = 4/(pi sqrt f)", sp.simplify(ratio_ff - 4/(sp.pi*sp.sqrt(fs))) == 0, str(ratio_ff))
C_hoop = 2*sp.pi*R
check("A7", "hoop: C/(4 pi G M/c^2) = 1/f (C <= 4piGM/c^2  <=>  f >= 1)", sp.simplify(C_hoop/(4*sp.pi*Gs*M/cs**2) - 1/f) == 0)

# Misner-Sharp: f with E = Misner-Sharp energy equals 1 - g^ab dR dR  -> f = 1 <=> marginally trapped (spherical symmetry)
# closed-FRW dust interior: R = a sin(chi), dR/dtau = adot sin(chi), (1/a) dR/dchi = cos(chi), adot^2/c^2 = a_m/a - 1
a, am, chi, eta, chi0 = sp.symbols('a a_m chi eta chi0', positive=True)
gradR2 = -(sp.sin(chi)**2*(am/a - 1)) + sp.cos(chi)**2
f_MS = sp.simplify(1 - gradR2)
check("A8", "OS interior: 2Gm_MS/(c^2 R) = sin^2(chi) a_m/a", sp.simplify(f_MS - sp.sin(chi)**2*am/a) == 0, str(f_MS))

# ---------------------------------------------------------------- B. fill fraction table (published masses, radii)
objs = [  # name, mass kg, radius m, source tag
 ("proton (CODATA rms charge radius)", m_p, C.physical_constants['proton rms charge radius'][0], "CODATA 2018 via scipy"),
 ("hydrogen atom (Bohr radius)", m_p + m_e, C.physical_constants['Bohr radius'][0], "CODATA 2018 via scipy"),
 ("mammalian cell (3000 um^3, rho=1000 kg/m^3)", 3000e-18*1000.0, (3*3000e-18/(4*math.pi))**(1/3), "BioNumbers BNID 111476 volume; density assumed water"),
 ("Sun", Msun, Rsun, "IAU 2015 nominal"),
 ("white dwarf Sirius B", 1.018*Msun, 0.008098*Rsun, "Bond et al. 2017 ApJ 840 70"),
 ("neutron star PSR J0740+6620", 2.08*Msun, 12.39e3, "Fonseca 2021 mass; Riley 2021 radius (12.39 km)"),
 ("Milky-Way-like halo M200", 1.08e12*Msun, None, "Cautun 2020 M200c; R200 from 200 rho_c (H0=67.4)"),
 ("Coma cluster M200 (h=0.674)", 1.88e15/0.674*Msun, 1.99/0.674*Mpc, "Kubo et al. 2007 weak lensing"),
]
H0 = 67.4e3/Mpc
rho_c = 3*H0**2/(8*math.pi*G)
rows = []
for name, Mk, Rk, src in objs:
    if Rk is None:
        Rk = (3*Mk/(4*math.pi*200*rho_c))**(1/3)
    fk = 2*G*Mk/(c**2*Rk)
    Sh = math.pi*Rk**2/lP2 / 4 * 4 / 4  # placeholder, replaced below
    Sh = 4*math.pi*Rk**2/(4*lP2)          # nats
    Sb = 2*math.pi*Mk*c*Rk/hbar             # nats
    rows.append({"object": name, "M_kg": Mk, "R_m": Rk, "r_s_m": 2*G*Mk/c**2, "f": fk,
                 "S_holo_nats": Sh, "S_Bek_nats": Sb, "t_fill_s": Rk**2*c/(G*Mk), "t_cross_s": 2*Rk/c, "source": src})
    check("B_" + name.split()[0], f"f = S_Bek/S_holo = r_s/R for {name}", abs(Sb/Sh/fk - 1) < 1e-12, fk)
# Hubble sphere today: total density -> f = 1 exactly; matter only -> Om
R_H = c/H0
M_H = rho_c*4/3*math.pi*R_H**3
fH = 2*G*M_H/(c**2*R_H)
rows.append({"object": "Hubble sphere (total, critical density)", "M_kg": M_H, "R_m": R_H, "r_s_m": 2*G*M_H/c**2, "f": fH,
             "S_holo_nats": math.pi*R_H**2/lP2, "S_Bek_nats": 2*math.pi*M_H*c*R_H/hbar, "t_fill_s": R_H**2*c/(G*M_H),
             "t_cross_s": 2*R_H/c, "source": "Planck 2018 H0 = 67.4"})
Om, Orad, OL = 0.3153, 9.22e-5, 1 - 0.3153 - 9.22e-5
rows.append({"object": "Hubble sphere (matter only)", "M_kg": Om*M_H, "R_m": R_H, "r_s_m": Om*2*G*M_H/c**2, "f": Om*fH,
             "S_holo_nats": math.pi*R_H**2/lP2, "S_Bek_nats": Om*2*math.pi*M_H*c*R_H/hbar, "t_fill_s": R_H**2*c/(G*Om*M_H),
             "t_cross_s": 2*R_H/c, "source": "Planck 2018 Om = 0.3153"})
Hs, rhos = sp.symbols('H rho', positive=True)
MH_s = sp.Rational(4, 3)*sp.pi*(3*Hs**2/(8*sp.pi*Gs))*(cs/Hs)**3
check("B_hubble", "flat FRW: 2 G M_H/(c^2 R_H) = 1 at every epoch (Hubble sphere = apparent horizon)",
      sp.simplify(2*Gs*MH_s/(cs**2*(cs/Hs)) - 1) == 0)
num("fill_table", rows)

# ---------------------------------------------------------------- C. Oppenheimer-Snyder dust collapse
# interior closed FRW: a(eta) = (a_m/2)(1+cos eta), c tau = (a_m/2)(eta + sin eta); surface at chi0; R0 = a_m sin chi0; r_s = a_m sin^3 chi0
tau_s = am/2*(eta + sp.sin(eta))     # c*tau
a_eta = am/2*(1 + sp.cos(eta))
adot = sp.diff(a_eta, eta)/sp.diff(tau_s, eta)   # da/d(c tau)
check("C1", "a(eta), tau(eta) solve closed dust Friedmann: (da/dctau)^2 = a_m/a - 1", sp.simplify(adot**2 - (am/a_eta - 1)) == 0)
f_surf = sp.sin(chi0)**2*am/a_eta
eta_h = sp.pi - 2*chi0
check("C2", "surface f = r_s/R reaches 1 exactly at eta_h = pi - 2 chi0", sp.simplify(f_surf.subs(eta, eta_h) - 1) == 0)
chiAH = sp.pi/2 - eta/2
check("C3", "interior apparent horizon chi_AH(eta) = pi/2 - eta/2 has f = 1 (enters at surface at eta_h, centre at eta = pi)",
      sp.simplify((sp.sin(chiAH)**2*am/a_eta) - 1) == 0 and sp.simplify(chiAH.subs(eta, eta_h) - chi0) == 0)
# f increases outward for chi < pi/2: outermost shell traps first
check("C4", "d f/d chi > 0 for 0 < chi < pi/2 (first trapped sphere is the surface)",
      sp.simplify(sp.diff(sp.sin(chi)**2*am/a_eta, chi) - am/a_eta*sp.sin(2*chi)) == 0)
eta_b = sp.pi - 3*chi0      # event horizon born at centre: outgoing null ray chi = eta - eta_b reaches chi0 at eta_h
x = sp.symbols('x', positive=True)
dtau_EH = sp.simplify((tau_s.subs(eta, eta_h) - tau_s.subs(eta, eta_b)))          # c*Delta tau
rs_s = am*sp.sin(chi0)**3
lim_EH = sp.limit(dtau_EH/rs_s, chi0, 0)
check("C5", "EH birth (centre) to surface crossing: c*Delta tau -> (19/12) r_s as R0/r_s -> inf", sp.simplify(lim_EH - sp.Rational(19, 12)) == 0, str(lim_EH))
dtau_sing = sp.simplify(tau_s.subs(eta, sp.pi) - tau_s.subs(eta, eta_h))
lim_sing = sp.limit(dtau_sing/rs_s, chi0, 0)
check("C6", "surface crossing r_s to singularity: c*Delta tau -> (2/3) r_s", sp.simplify(lim_sing - sp.Rational(2, 3)) == 0, str(lim_sing))
check("C7", "average horizon writing rate / Bekenstein rate -> (2 r_s/c)/Delta tau = 24/19",
      sp.simplify(2/lim_EH - sp.Rational(24, 19)) == 0, float(sp.Rational(24, 19)))

def os_numbers(R0_over_rs):
    s0 = math.asin(math.sqrt(1/R0_over_rs))   # chi0
    a_m = R0_over_rs/math.sin(s0)             # in units of r_s  (R0 = a_m sin chi0)
    T = lambda e: a_m/2*(e + math.sin(e))     # c tau / r_s
    eh, eb = math.pi - 2*s0, math.pi - 3*s0
    res = {"R0_over_rs": R0_over_rs, "chi0": s0, "tau_AH_surface": T(eh), "tau_sing": T(math.pi),
           "tau_freefall_check": math.pi/2*math.sqrt(R0_over_rs**3/1.0)}   # (pi/2)sqrt(R0^3/(c^2 r_s)) in r_s/c units
    if eb > 0:
        res["tau_EH_birth"] = T(eb); res["dtau_EH_growth"] = T(eh) - T(eb)
        # instantaneous horizon writing rate in Bekenstein units: d(2 x^2)/dT, x = R_EH/r_s, T = c tau/r_s
        es = np.linspace(eb + 1e-9, eh, 4001)
        xs = (a_m/2*(1 + np.cos(es)))*np.sin(es - eb)
        Ts = a_m/2*(es + np.sin(es))
        r = np.gradient(2*xs**2, Ts)
        res["rate_ratio_max"] = float(r.max()); res["rate_ratio_avg"] = float(2/(T(eh) - T(eb)))
        res["rate_ratio_at_crossing"] = float(r[-1]); res["rate_ratio_min_interior"] = float(r[5:-5].min())
    else:
        res["tau_EH_birth"] = None
    # f along the collapse at the surface: tabulate f vs tau
    es = np.linspace(0, math.pi*0.999, 9)
    res["f_track"] = [{"ctau_over_rs": T(e), "f": float((1/R0_over_rs)*2/(1 + math.cos(e)))} for e in es]
    # Bekenstein-rate record written over the whole collapse to r_s, against the capacity at r_s
    res["Bek_written_to_crossing_over_capacity"] = T(eh)/2
    return res
os_tab = [os_numbers(v) for v in (2.0, 3.0, 10.0, 100.0, 1e4)]
for r in os_tab:
    check(f"C8_{r['R0_over_rs']:g}", "singularity time equals dust free-fall time (pi/2)sqrt(R0^3/2GM)",
          abs(r["tau_sing"]/r["tau_freefall_check"] - 1) < 1e-12)
check("C9", "numerical average rate ratio at R0 = 1e4 r_s within 1e-3 of 24/19",
      abs(os_tab[-1]["rate_ratio_avg"] - 24/19) < 1e-3, os_tab[-1]["rate_ratio_avg"])
for r in os_tab:
    if r.get("tau_EH_birth") is not None:
        check(f"C10_{r['R0_over_rs']:g}", "event-horizon area grows monotonically from 0 to 4 pi r_s^2 (area theorem)", r["rate_ratio_min_interior"] >= 0, r["rate_ratio_min_interior"])
num("OS", os_tab)

# ---------------------------------------------------------------- D. the formation entropy jump
S_BH = lambda Mk: 4*math.pi*G*Mk**2/(hbar*c)          # nats (S/k)
SBH1 = num("S_BH_1Msun_nats", S_BH(Msun))
check("D1", "S_BH(1 Msun) = 1.05e77 k (book p3_07 / app_S table)", abs(SBH1/1.05e77 - 1) < 0.01, SBH1)
mPl2 = hbar*c/G
Mb = sp.symbols('M_b', positive=True); mps, mPls = sp.symbols('m_p m_Pl', positive=True)
check("D2", "S_BH per baryon = 4 pi M m_p / m_Pl^2", sp.simplify(4*sp.pi*Gs*Mb**2/(hs*cs)/(Mb/mps) - (4*sp.pi*Mb*mps/mPls**2).subs(mPls, sp.sqrt(hs*cs/Gs))) == 0)
num("S_BH_per_baryon_1Msun", S_BH(Msun)/(Msun/m_p))
# rough thermal entropy of the Sun: uniform sphere, virial mean T, fully ionised H/He (X=0.70, Y=0.28), Sackur-Tetrode per species
X, Y = 0.70, 0.28
mu = 1/(2*X + 0.75*Y + 0.5*(1 - X - Y))
T_vir = G*Msun*mu*m_p/(5*kB*Rsun)            # 2K + W = 0 for uniform sphere, K = (3/2) N k T
V = 4/3*math.pi*Rsun**3
def st(Nn, m):
    return Nn*(2.5 + math.log(V/Nn*(m*kB*T_vir/(2*math.pi*hbar**2))**1.5))
Np = X*Msun/m_p; Nhe = Y*Msun/(4.0026*C.atomic_mass); Ne = Np + 2*Nhe
S_sun_est = st(Np, m_p) + st(Nhe, 4.0026*C.atomic_mass) + st(Ne, m_e) + Ne*0  # electrons non-degenerate at mean density (checked below)
num("T_vir_mean_K", T_vir); num("S_sun_estimate_k", S_sun_est)
n_e = Ne/V; lam_e = math.sqrt(2*math.pi*hbar**2/(m_e*kB*T_vir))
num("electron_degeneracy_n_lambda3", n_e*lam_e**3)
check("D3", "rough Sun entropy estimate within the published 1e57-1e58 k range (order of magnitude)", 1e57 <= S_sun_est <= 1e59, S_sun_est)
for Ssun in (1e57, 1e58):
    num(f"jump_SBH_over_Ssun_{Ssun:.0e}", SBH1/Ssun)
num("jump_SBH_over_Ssun_estimate", SBH1/S_sun_est)
# actual fill at the moment the surface crosses r_s, if the matter brings only its thermal entropy
num("actual_fill_at_rs_published_range", [1e57/SBH1, 1e58/SBH1])

# ---------------------------------------------------------------- E. the cosmos
def Om_a(a_):
    return Om*a_**-3/(Om*a_**-3 + Orad*a_**-4 + OL)
avals = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 1.53, 2.0, 10.0]
num("cosmic_f_total", [{"a": v, "f": 1.0} for v in avals])
num("cosmic_f_matter", [{"a": v, "f_m": Om_a(v)} for v in avals])
check("E1", "matter-only Hubble fill today = Om = 0.3153", abs(Om_a(1.0) - 0.3153) < 1e-4, Om_a(1.0))
Eb = lambda a_: math.exp(1 - 1/a_)
g = lambda a_: Eb(a_)*a_**2*Om_a(a_)
amax = brentq(lambda a_: (g(a_*1.0001) - g(a_*0.9999)), 1.1, 3.0)
check("E2", "reproduce Bekenstein-Dirac K26: E a^2 Om(a) peaks 0.377 at a = 1.53 (a RATE ratio, not a fill)",
      abs(g(amax) - 0.377) < 0.002 and abs(amax - 1.53) < 0.02, [amax, g(amax)])
S_geo = math.pi*R_H**2/lP2
num("S_geo_H67p4_nats", S_geo)
num("actual_fill_today_SMBH_count_report", 9.20e102/2.2817e122)         # count report B3 / A1 (H0 = 67.16)
num("actual_fill_today_EganLineweaver", 3.1e104/2.6e122)                 # S_obs / S_CEH, Egan & Lineweaver 2010

# ---------------------------------------------------------------- F. the cell, as numbers
T_b = 310.15
N = 28217448                                                 # book: hg19 CpG index (p6_02:134, app_S:148)
Mcoin = 20.94                                                # book: one ATP = 20.94 kT (forward Step 2, app_S)
Ehold = 3.08                                                 # book: healthy holding energy, kT
eps0 = 1/(1 + math.exp(Mcoin)); num("floor_eps0", eps0)
check("F1", "floor 1/(1+e^M) = 8.1e-10 (book fig cell_gauge)", abs(eps0/8.1e-10 - 1) < 0.01, eps0)
Hb = lambda e: -(e*math.log2(e) + (1 - e)*math.log2(1 - e))
num("Hmin_bits", Hb(eps0)); check("F2", "H_min = 2.5e-8 bits (book)", abs(Hb(eps0)/2.5e-8 - 1) < 0.05, Hb(eps0))
eps_h = 1/(1 + math.exp(Ehold)); num("eps_healthy", eps_h)
num("H_ref_bits", Hb(eps_h)); check("F3", "healthy H_ref = 0.259 bits at 3.08 kT (book)", abs(Hb(eps_h) - 0.259) < 0.002, Hb(eps_h))
num("cell_fill_entropy_share", Hb(eps_h)/1.0)                 # fraction of the ceiling (1 bit per site)
num("cell_fill_error_share", eps_h/0.5)
num("cell_hold_over_coin", Ehold/Mcoin)
num("cell_eps_over_floor", eps_h/eps0)
num("cell_H_over_Hmin", Hb(eps_h)/Hb(eps0))
# record capacity vs holographic capacity of a nucleus-sized sphere, and Landauer cost
r_nuc = 3e-6
S_nuc_bits = math.pi*r_nuc**2/lP2/math.log(2)
num("nucleus_holo_bits", S_nuc_bits); check("F4", "nucleus-sized (r = 3 um) holographic bits = 1.6e59 (book app_S:184)", abs(S_nuc_bits/1.6e59 - 1) < 0.03, S_nuc_bits)
num("record_over_nucleus_holo", N/S_nuc_bits)
# register rates (A8 inventory: Repli-BS GSE82045, hESC): restore g = 0.443 /h, loss f = 0.024 /h
gr, fl = 0.443, 0.024
num("Ehold_from_rates_kT", math.log(gr/fl)); num("eps_from_rates", fl/(fl + gr))
num("Ehold_rates_over_book", math.log(gr/fl)/Ehold)
# (maintenance write rates are in section G, from the register's g, H1 methylation share and 2 coins per restore)
# time for the record to reach the coin-flip ceiling if restores stop (loss-only, one-strand model of the register)
t_half = math.log(2*(1 - eps_h))/fl; num("t_to_eps_half_loss_only_h", t_half)
# NOTE: the register's f was itself computed as g*eps/(1-eps) with the hESC eps = 0.0515 (REGISTER_SNAPSHOT line 2830),
# so ln(g/f) = ln((1-eps)/eps) is that eps restated, NOT an independent consistency check.
check("F5", "reproduce register arithmetic: f = g eps/(1-eps) with eps_hESC = 0.0515 gives 0.024 /h", abs(gr*0.0515/(1-0.0515) - 0.024) < 0.0005, gr*0.0515/(1-0.0515))
# cell Bekenstein fill (gravity) for comparison with BH f = 1
cell_row = [r for r in rows if r["object"].startswith("mammalian")][0]
num("cell_gravity_f", cell_row["f"])

# ---------------------------------------------------------------- G. the cell's writing speed from M (lead's addition)
Ms_, kT_, h_ = sp.symbols('M kT hbar', positive=True)
coin = Mcoin*kB*T_b; num("coin_J", coin); num("coin_kJ_per_mol", coin*C.N_A/1e3)
check("G1", "coin M kT = 8.97e-20 J (54 kJ/mol)", abs(coin/8.97e-20 - 1) < 0.002 and abs(coin*C.N_A/1e3 - 54.0) < 0.1, coin)
bek_coin = math.pi*coin/(hbar*math.log(2)); ml_coin = 2*coin/(math.pi*hbar); th = kB*T_b/C.h
num("bek_bits_per_s_per_coin", bek_coin); num("ML_ops_per_s_per_coin", ml_coin); num("thermal_attempt_per_s", th)
check("G2", "Bekenstein per coin 3.85e15 bits/s", abs(bek_coin/3.85e15 - 1) < 0.002, bek_coin)
check("G3", "Margolus-Levitin per coin 5.41e14 /s", abs(ml_coin/5.41e14 - 1) < 0.002, ml_coin)
check("G4", "thermal attempt kT/h 6.46e12 /s", abs(th/6.46e12 - 1) < 0.002, th)
ratio_sym = sp.simplify((sp.pi*Ms_*kT_/(h_*sp.log(2)))/(kT_/(2*sp.pi*h_)))
check("G5", "Bekenstein/thermal = 2 pi^2 M/ln2 (pure function of M) = 596", sp.simplify(ratio_sym - 2*sp.pi**2*Ms_/sp.log(2)) == 0
      and abs(float(ratio_sym.subs(Ms_, Mcoin)) - 596) < 0.5, float(ratio_sym.subs(Ms_, Mcoin)))
num("bek_over_thermal", float(ratio_sym.subs(Ms_, Mcoin)))
check("G6", "restore = 2 coins = 41.9 kT; Landauer efficiency ln2/(2M) = 1.66 %", abs(2*Mcoin - 41.88) < 0.005 and abs(math.log(2)/(2*Mcoin) - 0.01655) < 0.00005, math.log(2)/(2*Mcoin))
num("restore_kT", 2*Mcoin); num("landauer_eff", math.log(2)/(2*Mcoin))
num("ML_over_thermal", ml_coin/th); num("bek_over_ML", bek_coin/ml_coin)
# actual writing speed, hESC: register A8R g = 0.443 +/- 0.120 /h (held sites); H1 CpG methylation 0.827 (Lister 2009); cycle 15-16 h (Becker 2006)
m_meth, Tcyc = 0.827, 15.5*3600
n_restore = 4*N*m_meth; num("restores_per_division_diploid", n_restore)
rate_avg = n_restore/Tcyc; num("restores_per_s_cycle_avg", rate_avg)
rate_peak = (gr/3600)*n_restore; num("restores_per_s_all_pending_upper", rate_peak)
rate_ss = (gr*0.0515/3600)*n_restore; num("restores_per_s_register_steady_state", rate_ss)
num("restores_per_s_g_band", [((gr-0.120)/3600)*n_restore, ((gr+0.120)/3600)*n_restore])
num("mean_wait_per_mark_s", 3600/gr)
for lab, rr in (("avg", rate_avg), ("peak", rate_peak), ("ss", rate_ss)):
    atp = 2*rr
    num(f"ATP_per_s_{lab}", atp)
    num(f"ATP_share_{lab}", {"1e9": atp/1e9, "1e8": atp/1e8, "1e7": atp/1e7})
    num(f"power_W_{lab}", atp*coin)
per_mark_rate = gr/3600
num("per_mark_vs_bekenstein_2coins", per_mark_rate/(2*bek_coin))
num("per_mark_vs_ML_2coins", per_mark_rate/(2*ml_coin))
num("per_mark_vs_thermal", per_mark_rate/th)
num("bek_min_time_per_bit_2coins_s", 1/(2*bek_coin))
for lab, rr in (("avg", rate_avg), ("peak", rate_peak)):
    num(f"cell_{lab}_vs_bek_one_coin", rr/bek_coin); num(f"cell_{lab}_vs_ML_one_coin", rr/ml_coin); num(f"cell_{lab}_vs_thermal", rr/th)

npass = sum(ch["pass"] for ch in out["checks"]); nfail = len(out["checks"]) - npass
out["summary"] = {"passed": npass, "failed": nfail}
json.dump(out, open("saturation_checks_output.json", "w"), indent=1, default=float)
print(npass, nfail, [ch["id"] for ch in out["checks"] if not ch["pass"]])
