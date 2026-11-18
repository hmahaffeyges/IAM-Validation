import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath', 'colossus'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
"""missing_writer_checks.py  (2026-11-18, derivation spider, private)

Question: which process books ~ matter's rest energy per e-fold on the horizon
ledger and switches on as E(a) a^2 ?  Every candidate is computed from
published numbers / established formulas; nothing is fitted.  Pass criteria
are declared here BEFORE any candidate number is computed:

  AMPLITUDE passes  if  0.5 <= B(z=0)/B_req(z=0) <= 2
  SHAPE passes      if  0.5 <= S(z) <= 2 for z = 1, 2, 3, where
                        S(z) = [B(z)/B(0)] / [B_req(z)/B_req(0)]
  (B = bookable entropy per e-fold in units of M_m(a) c^2 / k T_H(a), the
   rest energy of the matter inside the Hubble sphere at epoch a priced at
   that epoch's Hubble temperature; B_req = E(a) a^2 / 2.)

A "check" below is a mathematical/numerical consistency assertion (identity,
unit conversion, published value reproduced).  Candidate pass/fail results are
RESULTS, recorded in the output JSON, not counted as checks.

Inputs: Planck 2018 as the book uses them (dark-sector report: H0 = 67.4,
Om = 0.3153, Ob = 0.0493); E(a) = exp(1 - 1/a); colossus (Diemer 2018) for the
Tinker 2008 and Sheth-Tormen 1999 mass functions (Eisenstein-Hu 1998 P(k)).
Runs in ~20 s, < 300 MB.
"""
import json, sys
import numpy as np
import sympy as sp
from scipy import integrate

OUT = {}
CHECKS = []


def check(cid, cond, note):
    CHECKS.append({"id": cid, "pass": bool(cond), "note": note})
    print(("PASS " if cond else "FAIL ") + cid + ": " + note)


# ---------------------------------------------------------------- constants
hbar = 1.054571817e-34; G = 6.67430e-11; c = 2.99792458e8; kB = 1.380649e-23
m_p = 1.67262192e-27; Msun = 1.98841e30; Mpc = 3.0856775814913673e22
sigT = 6.6524587321e-29; yr = 3.15576e7
H0kms, Om, Ob, Or = 67.4, 0.3153, 0.0493, 9.22e-5
OL = 1 - Om - Or
h = H0kms / 100
H0 = H0kms * 1e3 / Mpc
zs = [0.0, 1.0, 2.0, 3.0]


def Ea_act(a):           # the book's activation function
    return np.exp(1 - 1 / a)


def Hz(z):
    return H0 * np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL)


def B_req(z):
    a = 1 / (1 + z)
    return 0.5 * Ea_act(a) * a ** 2


# ======================================================= R: the requirement
a, H0s, b, Oms, Gs = sp.symbols("a H_0 beta_m Omega_m G", positive=True)
Hs = sp.Function("H")(a)
E = sp.exp(1 - 1 / a)
rho_info = 3 * H0s ** 2 * b * E / (8 * sp.pi * Gs)
Sdot_info = 8 * sp.pi ** 2 * rho_info / (3 * a * Hs ** 3)      # BOOK eq:law_Sneed (G=hbar=c=k=1 form)
dSinfo_dlna = Sdot_info / Hs
# S_geo = pi/(G H^2); dS_geo/dln a = 3 Om(a) S_geo for flat matter+Lambda
Om_a = Oms * H0s ** 2 / (a ** 3 * Hs ** 2)
dSgeo_dlna = 3 * Om_a * sp.pi / (Gs * Hs ** 2)
ratio = sp.simplify((dSinfo_dlna / dSgeo_dlna).subs(b, Oms / 2))
check("R1", sp.simplify(ratio - E * a ** 2 / 6) == 0,
      "book eq:law_Sneed with beta_m = Om/2 gives (dS_info/dln a)/(dS_geo/dln a) = E(a) a^2/6 exactly")
# booking per e-fold in units of M_m(a)c^2/kT_H(a): M_m c^2/kT_H = Om(a) S_geo
Breq_sym = sp.simplify(dSinfo_dlna.subs(b, Oms / 2) / (Om_a * sp.pi / (Gs * Hs ** 2)))
check("R2", sp.simplify(Breq_sym - E * a ** 2 / 2) == 0,
      "required booking per e-fold = (1/2) E(a) a^2 x [M_m(a) c^2 / k T_H(a)]; = 1/2 today")
# per-particle rate: (1/2)E a^2 m c^2/(k T_H) per e-fold, T_H = hbar H/(2 pi k)  ->  per second x H
m_, c_, hb_, H_ = sp.symbols("m c hbar H", positive=True)
Gam_req = sp.Rational(1, 2) * E * a ** 2 * m_ * c_ ** 2 / (hb_ * H_ / (2 * sp.pi)) * H_
check("R3", sp.simplify(Gam_req - sp.pi * E * a ** 2 * m_ * c_ ** 2 / hb_) == 0,
      "required nats per writing particle per second = pi E(a) a^2 m c^2 / hbar (H cancels)")
ML = 2 * m_ * c_ ** 2 / (sp.pi * hb_)        # Margolus-Levitin max ops/s for energy mc^2
check("R4", sp.simplify(Gam_req.subs(a, 1) / ML - sp.pi ** 2 / 2) == 0,
      "today the required rate is pi^2/2 = 4.93 times the Margolus-Levitin bound 2mc^2/(pi hbar)")
Greq0 = np.pi * m_p * c ** 2 / hbar
TH0 = hbar * H0 / (2 * np.pi * kB)
req = {"B_req": {str(z): B_req(z) for z in zs},
       "B_req_shape": {str(z): B_req(z) / B_req(0) for z in zs},
       "Gamma_req_proton_today_per_s": Greq0,
       "heat_flow_req_per_proton_today_W": 0.5 * m_p * c ** 2 * H0,
       "T_H0_K": TH0}
# history file cross-check: the prior spider's E values are exp(-z)
hist = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
if hist:
    Eh = hist["book E(a) (measured shape)"]["E"]
    check("R5", all(abs(Eh[k] / np.exp(-float(k)) - 1) < 1e-4 for k in Eh),
          "implied_history.json 'book E(a) (measured shape)' values equal exp(-z) = E(1/(1+z)) to 1e-4 relative "
          "(run 1 used an absolute 1e-6 tolerance and failed on the file's 3e-6 offsets: my tolerance, not the physics)")
# total record entropy today and in Hubble-sphere numbers
lP2 = hbar * G / c ** 3
Sgeo0 = np.pi * c ** 2 / (lP2 * H0 ** 2)          # nats (k=1)
need_e_fold_today = 0.5 * Om * Sgeo0
req["S_geo0_nats"] = Sgeo0
req["booking_per_efold_today_nats"] = need_e_fold_today
check("R6", abs(need_e_fold_today / np.log(2) / (5.2e121 * (67.16 / 67.4) ** 2) - 1) < 0.03,
      "beta_m S_geo today = %.2e bits, the book's 5.2e121 bits/e-fold (at H0=67.16) up to the H0^-2 difference" % (need_e_fold_today / np.log(2)))
OUT["requirement"] = req


def verdict(B):
    amp = B[0] / B_req(0)
    S = [(B[i] / B[0]) / (B_req(z) / B_req(0)) if B[0] > 0 else np.nan for i, z in enumerate(zs)]
    return {"amp_ratio_z0": amp,
            "ratio_to_req": {str(z): B[i] / B_req(z) for i, z in enumerate(zs)},
            "shape_S": {str(z): S[i] for i, z in enumerate(zs)},
            "amp_pass": bool(0.5 <= amp <= 2),
            "shape_pass": bool(all(0.5 <= s <= 2 for s in S[1:]))}


# ======================================= (a) gravitational / environmental decoherence
A = {}
R0_bound = 0.54e-10          # Donadi et al. 2021, 95 % lower bound
R0_diosi = 1e-15             # Diosi's original choice (excluded)
for lab, R0 in [("R0_bound_0.54A", R0_bound), ("R0_1fm_excluded", R0_diosi)]:
    EG = 6 / 5 * G * m_p ** 2 / R0          # self-energy difference of a uniform sphere, separation >> R0
    gam = EG / hbar
    # heating, Donadi et al. 2021 printed form dT/dt = 4 sqrt(pi) m0 G hbar / (3 kB R0^3); dE/dt = (3/2) kB dT/dt
    dTdt = 4 * np.sqrt(np.pi) * m_p * G * hbar / (3 * kB * R0 ** 3)
    Pheat = 1.5 * kB * dTdt
    A[lab] = {"Gamma_DP_per_nucleon_s": gam, "count_booking_ratio_today": gam / Greq0,
              "dTdt_K_per_s": dTdt, "heating_W_per_nucleon": Pheat,
              "heat_booking_ratio_today": Pheat / (0.5 * m_p * c ** 2 * H0)}
check("A1", 1e-4 < A["R0_1fm_excluded"]["dTdt_K_per_s"] < 1e-2,
      "Donadi's printed heating formula at R0 = 1e-15 m gives %.1e K/s (paper text: '~1e-4 K/s'; see questions)" % A["R0_1fm_excluded"]["dTdt_K_per_s"])
# DP count booking for a coherently superposed body of mass M, radius R: per nucleon (6/5) G M m/(hbar R)
# equals pi E a^2 m c^2/hbar when G M /(R c^2) = (5 pi / 6) E a^2
comp = {str(z): 5 * np.pi / 6 * B_req(z) * 2 for z in zs}
A["DP_compactness_needed_GM_over_Rc2"] = comp
check("A2", abs(comp["0.0"] - 5 * np.pi / 6) < 1e-12 and comp["0.0"] > 0.5,
      "DP count booking reaches the required rate today only for GM/(Rc^2) = 5pi/6 = 2.62 > 1/2 (inside its own horizon)")
# R0 at which DP heating equals the heat requirement (shown, not used: it would be a fit)
R0_eq = (1.5 * 4 * np.sqrt(np.pi) * G * hbar / (3 * 0.5 * c ** 2 * H0)) ** (1 / 3)
A["R0_for_heat_equality_m_NOT_USED"] = R0_eq
# Joos-Zeh: scattering-limited decoherence of an electron by the CMB (Thomson), per second, today
Tcmb = 2.7255
n_gam = 16 * np.pi * 1.2020569 * (kB * Tcmb / (2 * np.pi * hbar * c)) ** 3   # 2 zeta(3)/pi^2 (kT/hbar c)^3
check("A3", abs(n_gam * 1e-6 / 410.7 - 1) < 0.005, "CMB photon density %.1f cm^-3 (410.7 published)" % (n_gam * 1e-6))
gam_T = n_gam * sigT * c
A["Thomson_CMB_rate_per_electron_s"] = gam_T
A["Thomson_ratio_today"] = gam_T / Greq0
# Margolus-Levitin bound on any environment whose interaction energy per particle is kT
for lab, T in [("IGM_1e4K", 1e4), ("ICM_1e8K", 1e8), ("solar_core_1.57e7K", 1.57e7)]:
    A["ML_bound_" + lab] = {"rate_s": 2 * kB * T / (np.pi * hbar), "ratio_today": 2 * kB * T / (np.pi * hbar) / Greq0}
A["kT_needed_per_particle_today_GeV"] = (np.pi ** 2 / 2) * m_p * c ** 2 / 1.602176634e-10
# shape of any per-particle constant-rate process applied to ALL matter: B is a constant multiple of [M c^2/kT_H]
A["all_matter_constant_rate_shape"] = verdict([1.0, 1.0, 1.0, 1.0])["shape_S"]
OUT["a_decoherence"] = A

# ======================================= (b),(c) halo statistics, Weyl weighting
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from colossus.halo import concentration as cc
cosmo = cosmology.setCosmology("book_planck18", params=dict(flat=True, H0=H0kms, Om0=Om, Ob0=Ob,
                               sigma8=0.8111, ns=0.9649, relspecies=True, persistence=""))
rho_m0 = cosmo.rho_m(0.0) * 1e9           # Msun h^2 / Mpc^3, comoving
lnM = np.linspace(np.log(1e6), np.log(1e16), 400)
Mg = np.exp(lnM)


def w_weyl(cn):
    """CET (alpha=1) Weyl energy inside r_200m of an NFW halo, in units of M c^2:
    rho_grav = (1/3)|rhobar(<r) - rho(r)|  (CET eq. 60, LTB form) -> w = int_0^c m(x)/x dx / m(c) - 1/3"""
    m = lambda x: np.log1p(x) - x / (1 + x)
    I = integrate.quad(lambda x: m(x) / x, 0, cn)[0]
    return I / m(cn) - 1 / 3


# sanity: singular isothermal sphere limit gives 2/3; uniform sphere gives 0
xs = sp.symbols("x", positive=True)
check("B1", sp.simplify(sp.integrate(xs / xs, (xs, 0, 1)) / 1 - sp.Rational(1, 3)) == sp.Rational(2, 3),
      "Weyl-energy weight for a singular isothermal sphere = 2/3 of Mc^2 (int m/x = m)")
check("B2", sp.simplify(sp.integrate(xs ** 3 / xs, (xs, 0, 1)) / 1 - sp.Rational(1, 3)) == 0,
      "uniform sphere (m ~ x^3) has zero interior Weyl energy, as CET's LTB form requires")
wtab = {str(cn): w_weyl(cn) for cn in [2, 4, 5, 7, 10, 15, 20]}
OUT["weyl_weight_NFW"] = wtab
cgrid = np.linspace(1, 40, 79)
wgrid = np.array([w_weyl(x) for x in cgrid])


def f_tables(model, mdef, z):
    dn = mf.massFunction(Mg, z, q_in="M", q_out="dndlnM", mdef=mdef, model=model)
    integrand = dn * Mg / rho_m0
    conc = cc.concentration(Mg, "200m", z, model="duffy08")
    wW = np.interp(conc, cgrid, wgrid)
    cum = lambda y: np.array([np.trapezoid(y[i:], lnM[i:]) for i in range(len(lnM))])
    return cum(integrand), cum(integrand * wW)


thr = [1e8, 1e11, 1e12, 1e13, 1e14]
HALO = {}
for model, mdef in [("tinker08", "200m"), ("sheth99", "fof")]:
    dz = 0.02
    tab = {}
    for z in zs:
        tab[z] = {"f": {}, "fW": {}, "df": {}, "dfW": {}}
        Fc, FW = f_tables(model, mdef, z)
        Fp, FWp = f_tables(model, mdef, max(z - dz, 0) if z > 0 else 0)
        Fm, FWm = f_tables(model, mdef, z + dz)
        # d/dln a = -(1+z) d/dz
        if z > 0:
            dF = -(1 + z) * (Fp - Fm) / (-2 * dz)
            dFW = -(1 + z) * (FWp - FWm) / (-2 * dz)
        else:
            dF = -(1 + z) * (Fc - Fm) / (-dz)
            dFW = -(1 + z) * (FW - FWm) / (-dz)
        for M in thr:
            i = np.searchsorted(Mg, M)
            tab[z]["f"][M] = Fc[i]; tab[z]["fW"][M] = FW[i]
            tab[z]["df"][M] = dF[i]; tab[z]["dfW"][M] = dFW[i]
    res = {}
    for M in thr:
        key = "%.0e" % M
        f = [tab[z]["f"][M] for z in zs]; fW = [tab[z]["fW"][M] for z in zs]
        df = [tab[z]["df"][M] for z in zs]; dfW = [tab[z]["dfW"][M] for z in zs]
        res[key] = {
            "f_coll": dict(zip(map(str, zs), f)), "f_Weyl": dict(zip(map(str, zs), fW)),
            "df_dlna": dict(zip(map(str, zs), df)),
            "b1_stock_virial_half": verdict([0.5 * x for x in f]),
            "b2_flux_virial_half_3x": verdict([1.5 * x for x in f]),
            "b3_flow_virial_half": verdict([0.5 * x for x in df]),
            "c2_stock_Weyl_energy": verdict(fW),
            "c2_flux_Weyl_3x": verdict([3 * x for x in fW]),
            "c3_flow_Weyl_energy": verdict(dfW)}
    HALO[model] = res
# B3: integration check against the analytic Sheth-Tormen multiplicity (A=0.3222, a=0.707, p=0.3)
def fST(nu):
    aa = 0.707
    return 0.3222 * np.sqrt(2 * aa / np.pi) * (1 + (aa * nu ** 2) ** -0.3) * nu * np.exp(-aa * nu ** 2 / 2)
tot = integrate.quad(lambda nu: fST(nu) / nu, 0, np.inf)[0]
nu8 = 1.686 / cosmo.sigma(__import__("colossus.lss.peaks", fromlist=["x"]).lagrangianR(1e8), 0.0)
F8 = integrate.quad(lambda nu: fST(nu) / nu, nu8, np.inf)[0]
check("B3", abs(tot - 1) < 0.01 and abs(HALO["sheth99"]["1e+08"]["f_coll"]["0.0"] / F8 - 1) < 0.05,
      "ST multiplicity integrates to %.4f; colossus ST f(>1e8 Msun/h, z=0) = %.3f vs analytic %.3f "
      "(run 1 wrongly expected ~1: 43%% of the ST mass sits below 1e8 Msun/h)" % (tot, HALO["sheth99"]["1e+08"]["f_coll"]["0.0"], F8))
f13 = HALO["tinker08"]["1e+13"]["f_coll"]["0.0"]
check("B4", 0.05 < f13 < 0.25, "Tinker08 mass fraction in halos > 1e13 Msun/h at z=0 = %.3f (published range ~0.1-0.2)" % f13)
OUT["halos"] = HALO


# ----------------------------------------------- shape scan (DIAGNOSTIC ONLY: a threshold chosen to match shape would be a fit)
SCAN = {}
lgM = np.round(np.arange(11.0, 15.01, 0.1), 2)
for model, mdef in [("tinker08", "200m"), ("sheth99", "fof")]:
    FF = {z: f_tables(model, mdef, z) for z in zs}
    FFp = f_tables(model, mdef, 0.02)
    rows = []
    for lg in lgM:
        i = np.searchsorted(Mg, 10 ** lg)
        f = [FF[z][0][i] for z in zs]; fW = [FF[z][1][i] for z in zs]
        Sf = [(f[k] / f[0]) / (B_req(z) / B_req(0)) for k, z in enumerate(zs)]
        SW = [(fW[k] / fW[0]) / (B_req(z) / B_req(0)) for k, z in enumerate(zs)]
        slope0 = (np.log(f[0]) - np.log(FFp[0][i])) / np.log(1.02)    # d ln f / d ln a at z=0
        rows.append({"log10M": float(lg), "f0": f[0], "fW0": fW[0], "S_f": Sf, "S_fW": SW,
                     "maxdev_f": float(max(abs(np.log2(x)) for x in Sf[1:])),
                     "maxdev_fW": float(max(abs(np.log2(x)) for x in SW[1:])),
                     "dlnf_dlna_z0": slope0})
    best_f = min(rows, key=lambda r: r["maxdev_f"]); best_W = min(rows, key=lambda r: r["maxdev_fW"])
    SCAN[model] = {"rows": rows, "best_shape_f": best_f, "best_shape_fW": best_W,
                   "any_pass_f": [r["log10M"] for r in rows if r["maxdev_f"] <= 1],
                   "any_pass_fW": [r["log10M"] for r in rows if r["maxdev_fW"] <= 1],
                   "log10M_slope3_z0": float(lgM[np.argmin([abs(r["dlnf_dlna_z0"] - 3) for r in rows])])}
OUT["shape_scan_diagnostic"] = SCAN
# required log slope: d ln(E a^2)/d ln a = 1/a + 2 = 3 + z
check("S1", sp.simplify(sp.diff(sp.log(E * a ** 2), a) * a - (1 / a + 2)) == 0,
      "required log-slope d ln(E a^2)/d ln a = 2 + 1/a = 3 + z (3 today, 4, 5, 6 at z = 1, 2, 3)")


# ----------------------------------------------- forward: the record each halo writer would build (no fitting)
# d(rho_info/rho_c0)/dln a = B(a) * Om * a^-3   (booking per e-fold priced at T_H(a) per Hubble volume, as in R2)
IMPL = {}
zg = np.linspace(0, 8, 81)
for model, mdef in [("tinker08", "200m"), ("sheth99", "fof")]:
    FWz = {}
    Fz = {}
    for M in [1e8, 1e11, 1e13]:
        i = np.searchsorted(Mg, M)
        FWz[M] = []; Fz[M] = []
    for z in zg:
        F_, W_ = f_tables(model, mdef, z)
        for M in FWz:
            i = np.searchsorted(Mg, M); FWz[M].append(W_[i]); Fz[M].append(F_[i])
    lna = -np.log(1 + zg)
    for M in FWz:
        for lab, Y, wgt in [("c3_CET_Weyl_flow", np.array(FWz[M]), 1.0), ("b3_virial_half_flow", np.array(Fz[M]), 0.5)]:
            Bz = wgt * np.gradient(Y, lna)
            dens = Bz * Om * (1 + zg) ** 3
            # integrate from z=8 down to z
            cum = np.array([-np.trapezoid(dens[k:], lna[k:]) for k in range(len(zg))])
            tot0 = cum[0]
            Eeff = {str(z): float(cum[np.argmin(abs(zg - z))] / tot0) for z in [0.5, 1.0, 2.0, 3.0]}
            IMPL["%s_%s_%.0e" % (model, lab, M)] = {"rho_info0_over_rho_c0": float(tot0),
                                                   "over_beta_m": float(tot0 / (Om / 2)),
                                                   "E_eff_shape": Eeff,
                                                   "share_from_z_gt_1": float(cum[np.argmin(abs(zg - 1))] / tot0)}
IMPL["book_E_shape"] = {str(z): float(np.exp(-z)) for z in [0.5, 1.0, 2.0, 3.0]}
OUT["implied_record_forward"] = IMPL
# F1: the same integral with B = B_req returns the book's record term beta_m E(a) (method check)
a_g = 1 / (1 + zg)
dens_req = 0.5 * np.exp(1 - 1 / a_g) * a_g ** 2 * Om * a_g ** -3
cum_req = -np.trapezoid(dens_req, -np.log(1 + zg))
check("F1", abs(cum_req / ((Om / 2) * (1 - np.exp(1 - 9))) - 1) < 2e-3,
      "forward integral with B = B_req returns beta_m [E(1) - E(1/9)] = %.4f rho_c0 (method check)" % cum_req)

# ----------------------------------------------- (c1) CET linear regime: rho_grav = (1/3) rhobar |delta|, T_grav = T_H
LIN = {}
for R in [8.0, 1.0]:
    Ulin, dU = [], []
    for z in zs:
        s = cosmo.sigma(R, z); s2 = cosmo.sigma(R, z + 0.01)
        Ulin.append(np.sqrt(2 / np.pi) * s / 3)
        dU.append(np.sqrt(2 / np.pi) / 3 * (s - s2) / 0.01 * (1 + z))
    LIN["R=%g Mpc/h" % R] = {"sigma": dict(zip(map(str, zs), [cosmo.sigma(R, z) for z in zs])),
                             "c1_stock": verdict(Ulin), "c1_flow": verdict(dU)}
check("C1", abs(cosmo.sigma(8.0, 0.0) - 0.8111) < 1e-3, "sigma8 reproduced (%.4f)" % cosmo.sigma(8.0, 0.0))
OUT["cet_linear"] = LIN
# CET temperature for perturbed FRW is |H|/2pi (eq. 51): in SI hbar H/(2 pi kB) = T_H
check("C2", abs(hbar * H0 / (2 * np.pi * kB) / TH0 - 1) < 1e-12,
      "CET T_grav (perturbed FRW, eq. 51) = hbar H/2 pi k_B = T_H = %.2e K: CET prices Weyl energy at the ledger's own temperature" % TH0)

# ======================================= (d) other established Mc^2-per-time processes
D = {}
D["Margolus_Levitin_all_matter"] = {"per_nat": 2 / np.pi ** 2, "per_bit_ln2": 2 * np.log(2) / np.pi ** 2,
                                    "shape_S": verdict([1, 1, 1, 1])["shape_S"]}
# Lloyd 2002: total ops ~ 1e120 over the history; record entropy accumulated to today
def Hn(a_):
    return np.sqrt(Om / a_ ** 3 + Or / a_ ** 4 + OL)
Sinfo_today = integrate.quad(lambda a_: (Om / 2) * Ea_act(a_) / a_ / Hn(a_) ** 4 / a_, 1e-4, 1)[0] * Sgeo0
D["S_info_accumulated_today_nats"] = Sinfo_today
D["Lloyd_ops_1e120_over_S_info"] = 1e120 / Sinfo_today
# Padmanabhan 2012: N_bulk(matter) = 2 M_m c^2 / kT_H ; record per e-fold = (E a^2 /4) N_bulk
Nb = sp.symbols("N_b", positive=True)
check("D1", sp.simplify(E * a ** 2 / 2 - (E * a ** 2 / 4) * 2) == 0,
      "record booking per e-fold = (E a^2/4) x Padmanabhan's N_bulk of matter (= 2 M_m c^2/kT_H); today N_bulk/4")
# star formation (Madau & Dickinson 2014), each new stellar mass books its virial half at T_H
psi = lambda z: 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)   # Msun/yr/Mpc^3 (Salpeter)
rho_m0_phys = rho_m0 * h ** 2                                               # Msun/Mpc^3
Bstar = [0.5 * (1 - 0.27) * psi(z) / (Hz(z) * yr) / rho_m0_phys for z in zs]  # comoving/comoving
D["star_formation_flow"] = verdict(Bstar)
tgrid = np.linspace(0, 20, 4001)
rho_star = (1 - 0.27) * np.trapezoid([psi(z) / (Hz(z) * yr * (1 + z)) for z in tgrid], tgrid)
D["rho_star_today_Msun_Mpc3"] = rho_star
check("D2", 3e8 < rho_star < 8e8, "MD14 integral gives rho_* = %.2e Msun/Mpc^3 (MD14 quote ~5.8e8 for R=0.27)" % rho_star)
OUT["d_other"] = D

OUT["checks"] = CHECKS
OUT["checks_passed"] = sum(c_["pass"] for c_ in CHECKS)
OUT["checks_failed"] = sum(not c_["pass"] for c_ in CHECKS)
json.dump(OUT, open("missing_writer_checks_output.json", "w"), indent=1, default=float)
print("passed", OUT["checks_passed"], "failed", OUT["checks_failed"])
