import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath', 'colossus'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
"""cooling_scale_checks.py  (2026-11-18, derivation spider, private)

Question: does an ESTABLISHED galaxy-formation mass scale put the halo threshold
where the Weyl-energy writer (Clifton-Ellis-Tavakol 2013 free gravitational field
energy, NFW, alpha = 1) matches the record term in amplitude AND history under
booking P (written at T_H, booked at T_dS: the booking that reproduces the book's
Lambda relation; floor-pricing report Section 0 item 1)?

Nothing is fitted.  Every threshold M_th(z) below is a published number or a
published fitting formula, web-verified 2026-11-18 (sources in the report), with
its own mass definition and h, converted to the pipeline's M_200m [Msun/h] by
colossus (duffy08 NFW concentrations, the pipeline's own concentration model).

PASS BARS (copied unchanged from missing_writer_checks.py / floor_pricing_checks.py,
declared here before any candidate number is computed):
  AMPLITUDE passes  if 0.5 <= R(0) <= 2
  SHAPE passes      if 0.5 <= S(z) <= 2 at z = 1, 2, 3,  S(z) = R(z)/R(0)
  R(z) = B(z) * sqrt(Omega_L(z)) / [E(a) a^2 / 2]     (booking P)
  Reading (primary) = c2f, "CET Weyl, horizon flux": B = 3 f_W(>M_th(z), z), the
  reading in which the floor-pricing window opened.  Reported also: c2 stock
  (B = f_W) and c3 flow above the moving threshold (secondary, not used for the call).
  A candidate PASSES if it passes with BOTH Tinker08 (200m) and Sheth-Tormen (FOF);
  passing with one mass function only is reported as "one-HMF pass".

PRE-DECLARED CANDIDATES (published value, definition, h):
  A1  DB06 M_shock, shock at 0.1 R_v, Z0 = 0.1: 6e11 Msun, z-independent; M_vir(BN98), h = 0.7
      (band: factor 2 down/up for metallicity, DB06 Sec 3.7)
  A2  DB06 M_shock, shock at R_v, Z0 = 0.03, u_s = 0: ~2e12 Msun (the text's upper limit; its Fig. 2
      z-dependence was not digitised, so it is taken flat like the inner-halo curve DB06 call
      'quite independent of redshift' -- an assumption, flagged); variant 2.6e12 (eq. 35, u_s = 1/7)
  A3  DB06 eq. 43 "all-hot" M_crit(z): M_s = 10^11.8 Msun, f = 3; = M_s for z < z_crit,
      = M_s^2/(f M_*(z)) above.  M_*(z) = Press-Schechter 1-sigma mass in the BOOK cosmology
      (colossus nonLinearMass); variant: DB06's own log M_* = 13.1 - 1.3 z
  A4  Rees-Ostriker 1977 / White-Rees 1978 cooling = free-fall scale ~1e12 Msun (order of magnitude;
      read as halo M_vir); Silk 1977 M_cr ~ 5e11 Msun
  B1  Behroozi+13 ApJL 762 L31 instantaneous SFE peak 10^11.7 Msun, ~z-independent; M_vir, h = 0.7
  B2  Behroozi+13 ApJ 770 57 SMHM-ratio peak, from their eq. 3-4 best fit; M_vir, h = 0.7
  B3  Moster+13 MNRAS 428 3121 M_max(z) = M1 (beta/gamma)^(1/(beta+gamma)); M_200c, h = 0.704
  B4  Moster+18 MNRAS 477 1822 (Emerge) instantaneous-efficiency peak M_max(z); M_vir, h = 0.6781
  B5  Behroozi+19 MNRAS 488 3143 (UniverseMachine) SMHM-ratio peak, Table J1 'True All Cen Excl';
      M_peak ~ M_vir(BN98), h = 0.678
  C   virial-shock stability (BD03 / DB06): identical physics to A1/A2 (BD03: 'a few 1e11')
  D1  Bower+17 MNRAS 465 32 AGN take-over M_crit = 1e12 Delta_z^(-3/8) Msun, Delta_z^3 = Om(1+z)^3 + OL,
      Om = 0.307; M_200c, h = 0.6777
Physical masses (Msun) are put on the pipeline's Msun/h scale with the book h = 0.674 (using the
publication's h instead shifts every threshold by <= 0.017 dex).  For Sheth-Tormen (FOF b = 0.2)
the same M_200m threshold is used (FOF b=0.2 ~ M_200m within ~10-20 %; stated, not tuned).
A "check" is a numerical identity / reproduction; candidate verdicts are RESULTS, not checks.
"""
import json, sys
import numpy as np
from scipy import integrate, optimize

OUT = {}; CHECKS = []


def check(cid, cond, note):
    CHECKS.append({"id": cid, "pass": bool(cond), "note": note})
    print(("PASS " if cond else "FAIL ") + cid + ": " + note)


H0kms, Om, Ob, Or = 67.4, 0.3153, 0.0493, 9.22e-5
OL = 1 - Om - Or
h = H0kms / 100
zs = [0.0, 1.0, 2.0, 3.0]
EH = lambda z: np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL)
OLa = lambda z: OL / EH(z) ** 2
Breq = lambda z: 0.5 * np.exp(1 - (1 + z)) / (1 + z) ** 2        # E(a) a^2/2, E = exp(1-1/a)
check("K1", abs(np.sqrt(OLa(0)) - 0.8274) < 1e-4, "sqrt(Omega_L0) = %.4f (book eq:lam_corr_num 0.8274)" % np.sqrt(OLa(0)))

mw = json.load(open(sys.argv[1]))           # missing-writer output (stored curves)
hist = json.load(open(sys.argv[2]))         # implied_history.json

from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf, peaks
from colossus.halo import concentration as cc, mass_adv
cosmo = cosmology.setCosmology("book_planck18", params=dict(flat=True, H0=H0kms, Om0=Om, Ob0=Ob,
                               sigma8=0.8111, ns=0.9649, relspecies=True, persistence=""))
rho_m0 = cosmo.rho_m(0.0) * 1e9
lnM = np.linspace(np.log(1e6), np.log(1e16), 400)
Mg = np.exp(lnM)


def w_weyl(cn):
    m = lambda x: np.log1p(x) - x / (1 + x)
    return integrate.quad(lambda x: m(x) / x, 0, cn)[0] / m(cn) - 1 / 3


cgrid = np.linspace(1, 40, 79)
wgrid = np.array([w_weyl(x) for x in cgrid])


def rcum(y):
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(lnM)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def integrands(model, mdef, z):
    dn = mf.massFunction(Mg, z, q_in="M", q_out="dndlnM", mdef=mdef, model=model)
    I = dn * Mg / rho_m0
    conc = cc.concentration(Mg, "200m", z, model="duffy08")
    return I * np.interp(conc, cgrid, wgrid)


HMF = [("tinker08", "200m"), ("sheth99", "fof")]
# ---------------------------------------------------------------- tables on the forward grid
zg = np.round(np.linspace(0, 8, 81), 4)
zall = sorted(set(list(zg) + zs + [0.02, 1.02, 2.02, 3.02, 0.98, 1.98, 2.98]))
TAB = {}
for model, mdef in HMF:
    TAB[model] = {z: rcum(integrands(model, mdef, z)) for z in zall}
# K2: reproduce stored f_Weyl at the stored thresholds
dev = []
for model, _ in HMF:
    for key, d in mw["halos"][model].items():
        i = np.searchsorted(Mg, float(key))
        for z in zs:
            dev.append(abs(TAB[model][z][i] / d["f_Weyl"][str(z)] - 1))
check("K2", max(dev) < 1e-6, "pipeline reproduces stored missing-writer f_Weyl (5 thresholds x 4 z x 2 HMF), max rel dev %.1e" % max(dev))


def fW_at(model, z, M200m_h):
    return float(np.interp(np.log(M200m_h), lnM, TAB[model][z]))


def verdict(R):
    S = [R[k] / R[0] for k in range(1, 4)]
    return {"amp": R[0], "S": S, "amp_pass": bool(0.5 <= R[0] <= 2),
            "shape_pass": bool(all(0.5 <= s <= 2 for s in S)),
            "max_abs_log2S": float(max(abs(np.log2(s)) for s in S))}


def R_P(B, z):
    return B * np.sqrt(OLa(z)) / Breq(z)


# K3: reproduce the floor-pricing 10^12.3 numbers (scan used grid index via searchsorted)
fpref = {"tinker08": (1.26, [1.81, 1.51, 0.81]), "sheth99": (1.11, [2.08, 2.02, 1.33])}
ok = True; rep = {}
for model, _ in HMF:
    i = np.searchsorted(Mg, 10 ** 12.3)
    v = verdict([R_P(3 * TAB[model][z][i], z) for z in zs])
    rep[model] = v
    ok &= abs(v["amp"] - fpref[model][0]) < 0.006 and all(abs(a - b) < 0.006 for a, b in zip(v["S"], fpref[model][1]))
check("K3", ok, "floor-pricing 10^12.3 (c2f, P) reproduced: Tinker %.2f; %s  ST %.2f; %s" % (
    rep["tinker08"]["amp"], ", ".join("%.2f" % s for s in rep["tinker08"]["S"]),
    rep["sheth99"]["amp"], ", ".join("%.2f" % s for s in rep["sheth99"]["S"])))
OUT["repro_12.3"] = rep

# ---------------------------------------------------------------- published thresholds M_th(z)
def conv(Mphys, z, mdef_in):
    """physical Msun (definition mdef_in) -> M_200m [Msun/h] at z (book h; duffy08 NFW)"""
    Mh = Mphys * h
    if mdef_in == "200m":
        return Mh
    return float(mass_adv.changeMassDefinitionCModel(Mh, z, mdef_in, "200m", c_model="duffy08")[0])


r0 = conv(1e12, 0.0, "vir") / 1e12 / h; r3 = conv(1e12, 3.0, "vir") / 1e12 / h
check("K4", 1.05 < r0 < 1.35 and 0.9 < r3 < 1.05,
      "M_200m/M_vir for 1e12 Msun: %.3f at z=0, %.3f at z=3 (Delta_vir -> 18pi^2 x mean at high z)" % (r0, r3))


def b13_peak(z):            # Behroozi+13 eq. 3-4 best fit, SMHM-ratio peak
    a = 1 / (1 + z); nu = np.exp(-4 * a * a)
    M1 = 11.514 + (-1.793 * (a - 1) - 0.251 * z) * nu
    al = -1.412 + 0.731 * (a - 1) * nu
    de = 3.508 + (2.608 * (a - 1) - 0.043 * z) * nu
    ga = 0.316 + (1.319 * (a - 1) + 0.279 * z) * nu
    f = lambda x: -np.log10(10 ** (al * x) + 1) + de * np.log10(1 + np.exp(x)) ** ga / (1 + np.exp(10 ** -x))
    x = optimize.minimize_scalar(lambda x: -(f(x) - x), bounds=(-3, 3), method="bounded").x
    return 10 ** (M1 + x)


def m13_peak(z):
    s = z / (1 + z)
    M1 = 11.590 + 1.195 * s; be = 1.376 - 0.826 * s; ga = 0.608 + 0.329 * s
    return 10 ** M1 * (be / ga) ** (1 / (be + ga))


def m18_peak(z):
    s = z / (1 + z)
    M1 = 11.339 + 0.692 * s; be = 3.344 - 2.079 * s; ga = 0.966
    return 10 ** M1 * (be / ga) ** (1 / (be + ga))


def um19_peak(z):           # Table J1, True / All / Cen / Excl
    a = 1 / (1 + z); la = np.log(a)
    M1 = 12.074 + 4.600 * (a - 1) - 4.423 * la - 0.732 * z
    al = 1.974 - 2.468 * (a - 1) + 1.816 * la + 0.182 * z      # alpha = a0 + aa(a-1) - alna ln a + az z, alna = -1.816
    be = 0.470 - 0.875 * (a - 1) - 0.487 * z
    de = 0.382
    ga = 10 ** (-1.160 - 3.634 * (a - 1) - 1.219 * z)
    g = lambda x: -np.log10(10 ** (-al * x) + 10 ** (-be * x)) + ga * np.exp(-0.5 * (x / de) ** 2) - x
    xs = np.linspace(-2.5, 2.5, 5001)
    return 10 ** (M1 + xs[np.argmax(g(xs))])


def bower_crit(z):
    D = (0.307 * (1 + z) ** 3 + 0.693) ** (1 / 3)
    return 1e12 * D ** (-3 / 8)


Mstar_book = lambda z: peaks.nonLinearMass(z) / h          # Msun (Lagrangian, PS nu = 1)
Ms = 10 ** 11.8


def db06_crit(z, Mstar):
    fM = 3 * Mstar(z)
    return Ms if fM >= Ms else Ms * Ms / fM


Mstar_db06 = lambda z: 10 ** (13.1 - 1.3 * z)
# checks on the threshold implementations
# K5-K7 (run 1 compared formula values to loose text statements with tolerances that were mine, not the
# papers': Moster's '~12.3 at higher z' refers to his per-redshift fits, B13's App. E '10^12 -> 10^12.5 at z=2'
# is its simpler direct abundance matching, and Emerge's '1.1e12' is a 2-figure rounding of 10^12.031)
s2 = 2 / 3
check("K5", abs(np.log10(m13_peak(0)) - (11.590 + np.log10(1.376 / 0.608) / (1.376 + 0.608))) < 1e-12
      and abs(np.log10(m13_peak(0)) - 11.7) < 0.1 and abs(np.log10(m13_peak(2)) - 12.3) < 0.15,
      "Moster+13 eq. 3 + eq. 11-14: M_max = 10^%.2f at z=0 (paper text ~11.7), 10^%.2f at z=2 (text: per-z fits peak ~12.3 near z~2), 10^%.2f at z=3"
      % (np.log10(m13_peak(0)), np.log10(m13_peak(2)), np.log10(m13_peak(3))))
check("K6", all(abs(np.log10(b13_peak(z)) - 12.0) < 0.3 for z in zs),
      "Behroozi+13 eq. 3-4 best fit: SMHM peak 10^%s at z = 0..3 (abstract: 'halos near 1e12 Msun are the most efficient at all redshifts'; "
      "App. E simpler method quotes +0.5 dex by z=2, main fit gives +%.2f dex)" % (
          ", ".join("%.2f" % np.log10(b13_peak(z)) for z in zs), np.log10(b13_peak(2) / b13_peak(0))))
check("K7", 2.15e11 <= 10 ** 11.339 < 2.25e11 and 1.05e12 <= 10 ** (11.339 + 0.692) < 1.15e12,
      "Emerge M1 = 10^11.339 = %.2e at z=0 and 10^12.031 = %.2e at z->inf (paper Sec 3.5: 2.2e11, 1.1e12); M_max(0) = 10^%.2f"
      % (10 ** 11.339, 10 ** 12.031, np.log10(m18_peak(0))))
check("K8", 11.7 < np.log10(um19_peak(0)) < 12.4,
      "UniverseMachine SMHM peak at z=0 = 10^%.2f (paper: haloes reach peak SMHM ratio near 1e12)" % np.log10(um19_peak(0)))
zc_book = optimize.brentq(lambda z: 3 * Mstar_book(z) - Ms, 0, 5)
zc_db = (13.1 + np.log10(3) - 11.8) / 1.3
check("K9", abs(zc_db - 1.37) < 0.05, "DB06 eq.43 with log M* = 13.1-1.3z, f=3: z_crit = %.2f (paper ~1.4); book cosmology z_crit = %.2f" % (zc_db, zc_book))
# DB06's own claim that M_shock is ~z-independent: V_v = 120 + 28 z km/s through eq. A6-A8 (their cosmology)
def db06_M_from_V(z, V):
    a = 1 / (1 + z); Oma = 0.3 / (0.3 + 0.7 * a ** 3); OLa_ = 1 - Oma
    D = (18 * np.pi ** 2 - 82 * OLa_ - 39 * OLa_ ** 2) / Oma
    A = (D / 200) ** (-1 / 3) * a
    return 6.06e11 * (V / 100) ** 3 * A ** 1.5
mv = [db06_M_from_V(z, 120 + 28 * z) for z in zs]
check("K10", all(3e11 < m < 1.2e12 for m in mv),
      "DB06 V_v = 120+28z through their eq. A6-A8 gives M = %s Msun at z = 0..3 (their '6e11 quite independent of z', factor-2 band)" % ", ".join("%.1e" % m for m in mv))

CAND = {
    "A1_DB06_Mshock_0.1Rv_6e11": (lambda z: 6e11, "vir"),
    "A1lo_DB06_Mshock_3e11": (lambda z: 3e11, "vir"),
    "A1hi_DB06_Mshock_1.2e12": (lambda z: 1.2e12, "vir"),
    "A2_DB06_Mshock_Rv_2e12": (lambda z: 2e12, "vir"),
    "A2b_DB06_Mshock_Rv_eq35_2.6e12": (lambda z: 2.6e12, "vir"),
    "A3_DB06_eq43_allhot_bookMstar": (lambda z: db06_crit(z, Mstar_book), "vir"),
    "A3b_DB06_eq43_allhot_DB06Mstar": (lambda z: db06_crit(z, Mstar_db06), "vir"),
    "A4_ReesOstriker_WhiteRees_1e12": (lambda z: 1e12, "vir"),
    "A4b_Silk77_5e11": (lambda z: 5e11, "vir"),
    "B1_Behroozi13_SFEpeak_10^11.7": (lambda z: 10 ** 11.7, "vir"),
    "B2_Behroozi13_SMHMpeak": (b13_peak, "vir"),
    "B3_Moster13_Mmax": (m13_peak, "200c"),
    "B4_Moster18_Emerge_SFEpeak": (m18_peak, "vir"),
    "B5_Behroozi19_UM_SMHMpeak": (um19_peak, "vir"),
    "D1_Bower17_AGN_Mcrit": (bower_crit, "200c"),
}

# forward-record helpers
lna = -np.log(1 + zg)
sOL = np.sqrt(OLa(zg))


def mu_of(z, X):        # book eq:tw_mu (p2_18:67)
    HL2 = Om * (1 + z) ** 3 + OL
    return HL2 / (HL2 + X)


dens_req = Breq(zg) / np.sqrt(OLa(zg)) * sOL * Om * (1 + zg) ** 3
cum_req = -np.trapezoid(dens_req, lna)
check("K11", abs(cum_req / ((Om / 2) * (1 - np.exp(-8))) - 1) < 2e-3,
      "forward integral with R = 1 under P returns beta_m [E(1) - E(1/9)] = %.4f rho_c0 (method)" % cum_req)
bookmu = {z: mu_of(z, (Om / 2) * np.exp(-z)) for z in [0.0, 0.5, 1.0]}
check("K12", abs(bookmu[0.0] - 0.864) < 1e-3 and abs(bookmu[0.5] - 0.9482) < 2e-4 and abs(bookmu[1.0] - 0.9822) < 2e-4
      and abs(hist["book E(a) (measured shape)"]["mu"]["0.5"] - bookmu[0.5]) < 2e-3,
      "book mu: %.4f, %.4f, %.4f at z = 0, 0.5, 1 (book 0.864, 0.9482, 0.9822; implied_history %.4f)" % (
          bookmu[0.0], bookmu[0.5], bookmu[1.0], hist["book E(a) (measured shape)"]["mu"]["0.5"]))
zq = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
BOOK = {str(z): mu_of(z, (Om / 2) * np.exp(-z)) for z in zq}
OUT["book_mu"] = BOOK

RES = {}
for name, (fn, mdef_in) in CAND.items():
    Mth = {z: conv(fn(z), z, mdef_in) for z in zall}
    r = {"M_pub_Msun": {str(z): float(fn(z)) for z in zs}, "mdef_pub": mdef_in,
         "log10_M200m_h": {str(z): float(np.log10(Mth[z])) for z in zs}}
    for model, _ in HMF:
        fW = [fW_at(model, z, Mth[z]) for z in zs]
        c2f = verdict([R_P(3 * fW[k], z) for k, z in enumerate(zs)])
        c2 = verdict([R_P(fW[k], z) for k, z in enumerate(zs)])
        # c3 flow above the moving threshold: d/dln a at fixed M, integrated above M_th(z)
        fl = []
        for z in zs:
            zp, zm = (z, z + 0.02) if z == 0 else (z - 0.02, z + 0.02)
            dlna = np.log((1 + zm) / (1 + zp))
            fp_ = fW_at(model, round(zp, 4), Mth[z]); fm_ = fW_at(model, round(zm, 4), Mth[z])
            fl.append((fp_ - fm_) / dlna)
        c3 = verdict([R_P(fl[k], z) for k, z in enumerate(zs)])
        # forward record, c2f reading, booking P
        B = np.array([3 * fW_at(model, z, Mth[z]) for z in zg])
        d_ = B * sOL * Om * (1 + zg) ** 3
        cm = np.array([-np.trapezoid(d_[k:], lna[k:]) for k in range(len(zg))])
        Xi = lambda zz: float(np.interp(zz, zg, cm))
        fwd = {"over_beta_m": float(cm[0] / (Om / 2)),
               "mu_absolute": {str(zz): mu_of(zz, Xi(zz)) for zz in zq},
               "mu_shape_only": {str(zz): mu_of(zz, (Om / 2) * Xi(zz) / cm[0]) for zz in zq}}
        r[model] = {"c2f_horizon_flux": c2f, "c2_stock": c2, "c3_flow_moving_thr": c3, "forward_c2f_P": fwd,
                    "pass": bool(c2f["amp_pass"] and c2f["shape_pass"])}
    r["PASS_both_HMF"] = bool(r["tinker08"]["pass"] and r["sheth99"]["pass"])
    r["one_HMF_pass"] = [m for m, _ in HMF if r[m]["pass"]]
    RES[name] = r
    t, s = r["tinker08"]["c2f_horizon_flux"], r["sheth99"]["c2f_horizon_flux"]
    print("%-34s logM200m/h z0..3 = %s | T amp %.2f S %s | ST amp %.2f S %s | %s" % (
        name, " ".join("%.2f" % v for v in r["log10_M200m_h"].values()), t["amp"], " ".join("%.2f" % x for x in t["S"]),
        s["amp"], " ".join("%.2f" % x for x in s["S"]), "PASS" if r["PASS_both_HMF"] else ("one:" + ",".join(r["one_HMF_pass"]) if r["one_HMF_pass"] else "fail")))
OUT["candidates"] = RES

# diagnostic: constant-threshold pass window recomputed with interpolation (not used for any call)
win = {}
for model, _ in HMF:
    ok_ = []
    for lg in np.round(np.arange(11.5, 13.21, 0.05), 2):
        v = verdict([R_P(3 * fW_at(model, z, 10 ** lg), z) for z in zs])
        if v["amp_pass"] and v["shape_pass"]:
            ok_.append(float(lg))
    win[model] = ok_
OUT["constant_threshold_window_c2f_P_diag"] = win
print("window (const M200m/h, c2f, P):", {m: (min(v), max(v)) if v else None for m, v in win.items()})

# ---------------------------------------------------------------- sensitivity of the near-passers (diagnostic; verdicts above stand)
HPUB = {"B3_Moster13_Mmax": 0.704, "B2_Behroozi13_SMHMpeak": 0.7, "B5_Behroozi19_UM_SMHMpeak": 0.678,
        "A2_DB06_Mshock_Rv_2e12": 0.7, "A2b_DB06_Mshock_Rv_eq35_2.6e12": 0.7}
SENS = {}
def run_thr(Mth):
    o = {}
    for model, _ in HMF:
        o[model] = verdict([R_P(3 * fW_at(model, z, Mth[z]), z) for z in zs])
        o[model]["pass"] = bool(o[model]["amp_pass"] and o[model]["shape_pass"])
    return o
for name, hp in HPUB.items():
    fn, mdef_in = CAND[name]
    SENS[name] = {
        "h_publication": run_thr({z: conv(fn(z), z, mdef_in) * hp / h for z in zs}),
        "conv_dutton14": run_thr({z: float(mass_adv.changeMassDefinitionCModel(fn(z) * h, z, mdef_in, "200m", c_model="dutton14")[0]) for z in zs}),
        "minus_0.1dex": run_thr({z: conv(fn(z), z, mdef_in) * 10 ** -0.1 for z in zs}),
        "plus_0.1dex": run_thr({z: conv(fn(z), z, mdef_in) * 10 ** 0.1 for z in zs})}
# Moster+13 one-parameter-at-a-time 1-sigma shifts (Table 1 sigma; covariance not published in the table)
def m13_var(dM10=0, dM11=0, db10=0, db11=0, dg10=0, dg11=0):
    def f(z):
        s_ = z / (1 + z)
        M1 = 11.590 + dM10 + (1.195 + dM11) * s_; be = 1.376 + db10 + (-0.826 + db11) * s_; ga = 0.608 + dg10 + (0.329 + dg11) * s_
        return 10 ** M1 * (be / ga) ** (1 / (be + ga))
    return f
sig = dict(dM10=0.236, dM11=0.353, db10=0.153, db11=0.225, dg10=0.059, dg11=0.173)
MV = {}
for k, v in sig.items():
    for sgn in (+1, -1):
        f = m13_var(**{k: sgn * v})
        MV["%s%+.3f" % (k, sgn * v)] = run_thr({z: conv(f(z), z, "200c") for z in zs})
SENS["B3_Moster13_one_param_1sigma"] = MV
OUT["sensitivity"] = SENS
for name in SENS:
    if name.startswith("B3_Moster13_one"):
        npass = sum(1 for v in MV.values() if v["tinker08"]["pass"] and v["sheth99"]["pass"])
        print("Moster13 1-sigma one-at-a-time: both-HMF pass in %d of %d" % (npass, len(MV)))
        continue
    print(name, {k: (v["tinker08"]["pass"], v["sheth99"]["pass"], round(v["sheth99"]["S"][0], 2), round(v["sheth99"]["S"][1], 2)) for k, v in SENS[name].items()})
for name in ["B3_Moster13_Mmax", "B5_Behroozi19_UM_SMHMpeak", "A2b_DB06_Mshock_Rv_eq35_2.6e12", "A1_DB06_Mshock_0.1Rv_6e11"]:
    for model, _ in HMF:
        fw = RES[name][model]["forward_c2f_P"]
        print("mu", name, model, "over_beta_m %.3f" % fw["over_beta_m"], "abs", " ".join("%.4f" % fw["mu_absolute"][str(z)] for z in zq),
              "| shape", " ".join("%.4f" % fw["mu_shape_only"][str(z)] for z in zq))
print("mu book", " ".join("%.4f" % BOOK[str(z)] for z in zq))

OUT["checks"] = CHECKS
OUT["checks_passed"] = sum(c["pass"] for c in CHECKS)
OUT["checks_failed"] = sum(not c["pass"] for c in CHECKS)
json.dump(OUT, open("cooling_scale_checks_output.json", "w"), indent=1, default=float)
print("passed", OUT["checks_passed"], "failed", OUT["checks_failed"])
