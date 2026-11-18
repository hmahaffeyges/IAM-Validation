import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath', 'colossus'], check=True)
import numpy as _np
if not hasattr(_np, 'trapezoid'): _np.trapezoid = _np.trapz
"""floor_pricing_checks.py  (2026-11-18, derivation spider, private)

Question: if records are priced at the de Sitter FLOOR T_dS (the booking that
reproduces the book's Lambda relation, 1.0079) instead of T_H(a), how does the
required writer history change, and does any missing-writer candidate then
match BOTH amplitude and shape with nothing fitted?

PASS BARS (copied unchanged from missing_writer_checks.py; fixed here BEFORE any
new number is computed; never moved afterwards):
  AMPLITUDE passes if 0.5 <= R(z=0) <= 2, R = (ledger energy the writer delivers)
                       / (ledger energy the book's record term needs)
  SHAPE passes     if 0.5 <= S(z) <= 2 at z = 1, 2, 3, S(z) = R(z)/R(0)

BOOKINGS (declared before computing).  A writer makes entropy at its writing
temperature T_W: S_W = U_W / k T_W.  The ledger turns entropy into the energy
the field equations see at the ledger temperature T_L: E_L = k T_L S_W.
  HH : T_W = T_H, T_L = T_H         (missing-writer baseline, CET eq. 51)
  P  : T_W = T_H, T_L = T_dS        (book p2_12:209 'writing temperature T_H and
                                      surface temperature T_dS'; the 1.0079 booking)
  FF : T_W = T_dS, T_L = T_dS       (records priced at the floor on both sides)
  RV : T_W = T_dS, T_L = T_H        (the reverse, also named at p2_12:209)
  LOC: T_W = local material temperature (CMB, constant gas T, halo virial T,
       Deser-Levin local Unruh temperature at r_200m), T_L = T_dS (and T_H)

A "check" is an identity, unit conversion or reproduction of a stored / published
number.  Pass/fail of a candidate is a RESULT and is not counted as a check.

Inputs (literal paths, read only):
  missing-writer output JSON (stored curves), implied_history.json (book E, mu).
Background: H0 = 67.4, Om = 0.3153, Or = 9.22e-5, OL = 1 - Om - Or (as the
missing-writer script), E(a) = exp(1 - 1/a), beta_m = Om/2.
Colossus is used only (i) to reproduce stored f values (check) and (ii) for
per-mass temperature weights and the forward record on the same z grid and the
same thresholds as missing_writer_checks.py (no threshold re-tuned).
"""
import json, sys
import numpy as np
import sympy as sp

MW = sys.argv[1]
HIST = sys.argv[2]
mw = json.load(open(MW))
hist = json.load(open(HIST))

OUT = {"bars": {"amp": [0.5, 2.0], "shape": [0.5, 2.0], "shape_z": [1, 2, 3]}}
CHECKS = []


def check(cid, cond, note):
    CHECKS.append({"id": cid, "pass": bool(cond), "note": note})
    print(("PASS " if cond else "FAIL ") + cid + ": " + note)


# ---------------------------------------------------------------- constants (as missing_writer_checks.py)
hbar = 1.054571817e-34; G = 6.67430e-11; c = 2.99792458e8; kB = 1.380649e-23
m_p = 1.67262192e-27; Msun = 1.98841e30; Mpc = 3.0856775814913673e22
H0kms, Om, Ob, Or = 67.4, 0.3153, 0.0493, 9.22e-5
OL = 1 - Om - Or
h = H0kms / 100
H0 = H0kms * 1e3 / Mpc
zs = [0.0, 1.0, 2.0, 3.0]

# ======================================================= R: the requirement under floor pricing (sympy)
a, H0s, Gs, Oms, Ors, OLs, b = sp.symbols("a H_0 G Omega_m Omega_r Omega_L beta_m", positive=True)
hb, kb = sp.symbols("hbar k_B", positive=True)
E = sp.exp(1 - 1 / a)
Hsq = H0s ** 2 * (Oms / a ** 3 + Ors / a ** 4 + OLs)          # book background (flat, photon sector unchanged)
Hs = sp.sqrt(Hsq)
HL = H0s * sp.sqrt(OLs)                                      # de Sitter rate H_Lambda = H0 sqrt(OL)
T_H = hb * Hs / (2 * sp.pi * kb)                             # apparent-horizon temperature, flat FRW (Cai-Cao-Hu 2009)
T_dS = hb * HL / (2 * sp.pi * kb)                            # book eq:lam_TdS (p2_12:196)
OLa = OLs * H0s ** 2 / Hsq                                   # Omega_Lambda(a)
Oma = Oms * H0s ** 2 / (a ** 3 * Hsq)
Ora = Ors * H0s ** 2 / (a ** 4 * Hsq)

# R1: reproduce the missing-writer requirement from eq:law_Sneed (T_H booking)
rho_info = 3 * H0s ** 2 * b * E / (8 * sp.pi * Gs)
Sdot = 8 * sp.pi ** 2 * rho_info / (3 * a * Hs ** 3)
unit_H = Oma * sp.pi / (Gs * Hs ** 2)                        # M_m(a) c^2 / k T_H(a) = Om(a) S_geo
Breq_H = sp.simplify((Sdot / Hs).subs(b, Oms / 2) / unit_H)
check("R1", sp.simplify(Breq_H - E * a ** 2 / 2) == 0,
      "book eq:law_Sneed (beta_m = Om/2) -> B_req = E a^2/2 in units M_m c^2/kT_H (missing-writer R2 reproduced)")

# R2: T_dS/T_H = sqrt(Omega_Lambda(a)) exactly, with radiation
rT = sp.simplify(T_dS / T_H)
check("R2", sp.simplify(rT ** 2 - OLa) == 0, "T_dS/T_H(a) = sqrt(Omega_Lambda(a)) exactly on the book background (radiation kept)")
# R3: floor property T_dS <= T_H at every a (H^2 - H_L^2 = H0^2 (Om a^-3 + Or a^-4) > 0)
check("R3", sp.simplify(Hsq - HL ** 2 - H0s ** 2 * (Oms / a ** 3 + Ors / a ** 4)) == 0,
      "H^2 - H_Lambda^2 = H0^2 (Om a^-3 + Or a^-4) > 0, so T_dS < T_H at all finite a (floor)")

# R4: general booking theorem: ledger energy = U_W T_L / T_W; requirement in writer energy = U_req T_W / T_L
UW, Ureq, TW, TL = sp.symbols("U_W U_req T_W T_L", positive=True)
R_gen = (UW / TW) * TL / Ureq
check("R4", sp.simplify(R_gen.subs(TW, TL) - UW / Ureq) == 0,
      "if writer and ledger use the same temperature (HH or FF or local/local) it cancels: R = U_W/U_req, S(z) unchanged")

# R5: booking P (writer T_H, ledger T_dS): required writer energy share
Breq_P = sp.simplify(Breq_H * T_H / T_dS)                    # units M_m c^2 (writer priced at T_H)
check("R5", sp.simplify(Breq_P ** 2 - (E * a ** 2 / 2) ** 2 / OLa) == 0,
      "booking P: B_req_floor(a) = (E a^2/2)/sqrt(Omega_Lambda(a)) = (E a^2/2) H(a)/H_Lambda")
# R6: log slope under P
slope_P = sp.simplify(a * sp.diff(sp.log(Breq_P), a))
slope_P_claim = 2 + 1 / a - (3 * Oma + 4 * Ora) / 2
check("R6", sp.simplify(slope_P - slope_P_claim) == 0,
      "d ln B_req_floor/d ln a = 2 + 1/a - (3 Om(a) + 4 Or(a))/2 = (3+z) - (3 Om(a)+4 Or(a))/2")
# R7: reverse booking RV (writer T_dS, ledger T_H)
Breq_RV = sp.simplify(Breq_H * T_dS / T_H)
slope_RV = sp.simplify(a * sp.diff(sp.log(Breq_RV), a))
check("R7", sp.simplify(slope_RV - (2 + 1 / a + (3 * Oma + 4 * Ora) / 2)) == 0,
      "booking RV: B_req = (E a^2/2) sqrt(Omega_L(a)); slope (3+z) + (3 Om(a)+4 Or(a))/2")
# R8: required all-matter writing fraction today under P equals r_dS/r_A = 1/sqrt(OL0)
frac_P = sp.simplify(2 * Breq_P)
num = {Oms: Om, Ors: Or, OLs: OL, H0s: 1}
fracP0 = float(frac_P.subs(num).subs(a, 1))
check("R8", abs(fracP0 - 1 / np.sqrt(OL)) < 1e-12 and abs(fracP0 - 1.2086) < 5e-4,
      "under P the all-matter virial-half writing fraction needed today is E a^2/sqrt(OL(a)) = 1/sqrt(OL0) = %.4f = r_dS/r_A (sixteen-thirds C2: 1.2086)" % fracP0)
# R9: per-particle rate under P, ratio to Margolus-Levitin today
m_, c_ = sp.symbols("m c", positive=True)
Gam_P = Breq_P * m_ * c_ ** 2 / (kb * T_H) * Hs            # writer nats per particle per second
ML = 2 * m_ * c_ ** 2 / (sp.pi * hb)
gP = sp.simplify(Gam_P / ML)
gP0 = float(gP.subs(num).subs(a, 1))
check("R9", abs(gP0 - np.pi ** 2 / (2 * np.sqrt(OL))) < 1e-9,
      "under P each writing particle must book pi E a^2 m c^2/(hbar sqrt(OL(a))) nats/s; today %.3f x Margolus-Levitin (was pi^2/2 = 4.935)" % gP0)
# numbers
fB = sp.lambdify(a, Breq_H.subs(num), "numpy")
fBP = sp.lambdify(a, Breq_P.subs(num), "numpy")
fBRV = sp.lambdify(a, Breq_RV.subs(num), "numpy")
fsl = sp.lambdify(a, slope_P.subs(num), "numpy")
fslRV = sp.lambdify(a, slope_RV.subs(num), "numpy")
fOLa = sp.lambdify(a, OLa.subs(num), "numpy")
EH = lambda z: np.sqrt(Om * (1 + z) ** 3 + Or * (1 + z) ** 4 + OL)
req = {}
for z in zs + [0.5, 0.75]:
    aa = 1 / (1 + z)
    req[str(z)] = {"B_req_HH": float(fB(aa)), "B_req_P_floor": float(fBP(aa)), "B_req_RV": float(fBRV(aa)),
                   "shape_HH": float(fB(aa) / fB(1.0)), "shape_P": float(fBP(aa) / fBP(1.0)),
                   "slope_HH_3plusz": 3 + z, "slope_P": float(fsl(aa)), "slope_RV": float(fslRV(aa)),
                   "sqrt_OL_a": float(np.sqrt(fOLa(aa))), "E_H": float(EH(z)),
                   "frac_allmatter_needed_P": float(2 * fBP(aa))}
OUT["requirement"] = req
check("R10", abs(req["0.0"]["sqrt_OL_a"] - 0.8274) < 1e-4, "sqrt(Omega_L0) = %.4f = book eq:lam_corr_num 0.8274 (p2_12:205)" % req["0.0"]["sqrt_OL_a"])
# epoch where the needed all-matter fraction under P crosses 1 (diagnostic)
from scipy.optimize import brentq
a_one = brentq(lambda x: 2 * fBP(x) - 1, 0.3, 1.0)
OUT["a_where_fraction_P_equals_1"] = {"a": a_one, "z": 1 / a_one - 1}


# ======================================================= S: stored curves, every row, every booking
def SP_mult(z):      # P: T_L/T_W = sqrt(OL(z)); relative to today
    return np.sqrt(fOLa(1 / (1 + z)))


def verdict(R):      # R list at zs
    S = [R[i] / R[0] for i in range(4)]
    return {"amp": R[0], "S": S[1:], "amp_pass": bool(0.5 <= R[0] <= 2),
            "shape_pass": bool(all(0.5 <= s <= 2 for s in S[1:]))}


Breq_z = np.array([fB(1 / (1 + z)) for z in zs])
mult = {"HH": np.ones(4), "FF": np.ones(4),
        "P": np.array([SP_mult(z) for z in zs]),
        "RV": np.array([1 / SP_mult(z) for z in zs])}
rows = []
# check: stored ratio_to_req x B_req reproduces 0.5 f_coll for the b1 reading
b1chk = []
for model in ["tinker08", "sheth99"]:
    for thr, d in mw["halos"][model].items():
        f = np.array([d["f_coll"][str(z)] for z in zs])
        r = np.array([d["b1_stock_virial_half"]["ratio_to_req"][str(z)] for z in zs])
        b1chk.append(np.max(np.abs(r * Breq_z / (0.5 * f) - 1)))
        for reading in ["b1_stock_virial_half", "b2_flux_virial_half_3x", "b3_flow_virial_half",
                        "c2_stock_Weyl_energy", "c2_flux_Weyl_3x", "c3_flow_Weyl_energy"]:
            R_old = np.array([d[reading]["ratio_to_req"][str(z)] for z in zs])
            row = {"model": model, "M_min": thr, "reading": reading}
            for bk, m in mult.items():
                row[bk] = verdict(list(R_old * m))
            rows.append(row)
check("S1", max(b1chk) < 1e-9, "stored ratio_to_req x E a^2/2 reproduces 0.5 f_coll for all 10 b1 rows (stored curves read correctly)")
# check: HH recomputed S equals stored shape_S
dev = max(abs(r["HH"]["S"][k] - mw["halos"][r["model"]][r["M_min"]][r["reading"]]["shape_S"][str(zs[k + 1])])
          / mw["halos"][r["model"]][r["M_min"]][r["reading"]]["shape_S"][str(zs[k + 1])] for r in rows for k in range(3))
check("S2", dev < 1e-9, "HH booking recomputed from stored ratios returns the stored shape_S for all 60 halo rows")
# independent P check for c2 stock: B_W = f_W directly
d = mw["halos"]["tinker08"]["1e+13"]
RW = np.array([d["f_Weyl"][str(z)] for z in zs]) / np.array([fBP(1 / (1 + z)) for z in zs])
rowP = [r for r in rows if r["model"] == "tinker08" and r["M_min"] == "1e+13" and r["reading"] == "c2_stock_Weyl_energy"][0]["P"]
check("S3", abs(RW[0] / rowP["amp"] - 1) < 1e-9 and abs((RW[1] / RW[0]) / rowP["S"][0] - 1) < 1e-9,
      "P computed directly as f_W / B_req_floor equals the multiplier route (Tinker, 1e13, c2 stock)")
OUT["halo_rows"] = rows
OUT["halo_passes"] = {bk: [(r["model"], r["M_min"], r["reading"]) for r in rows if r[bk]["amp_pass"] and r[bk]["shape_pass"]] for bk in mult}
OUT["halo_shape_passes"] = {bk: [(r["model"], r["M_min"], r["reading"]) for r in rows if r[bk]["shape_pass"]] for bk in mult}

# other stored candidates
other = []
for R, lab in [(mw["cet_linear"]["R=8 Mpc/h"]["c1_stock"], "c1 linear Weyl R=8 stock"),
               (mw["cet_linear"]["R=8 Mpc/h"]["c1_flow"], "c1 linear Weyl R=8 flow"),
               (mw["cet_linear"]["R=1 Mpc/h"]["c1_stock"], "c1 linear Weyl R=1 stock"),
               (mw["cet_linear"]["R=1 Mpc/h"]["c1_flow"], "c1 linear Weyl R=1 flow"),
               (mw["d_other"]["star_formation_flow"], "star formation (MD14) virial half")]:
    R_old = np.array([R["ratio_to_req"][str(z)] for z in zs])
    o = {"candidate": lab}
    for bk, m in mult.items():
        o[bk] = verdict(list(R_old * m))
    other.append(o)
# constant-per-particle all-matter processes (ML at 1 nat/op; decoherence family): B = const x 1
for lab, amp0 in [("Margolus-Levitin all matter, 1 nat/op", mw["d_other"]["Margolus_Levitin_all_matter"]["per_nat"]),
                  ("Diosi-Penrose heating at R0 bound (heat booking)", mw["a_decoherence"]["R0_bound_0.54A"]["heat_booking_ratio_today"])]:
    R_old = np.array([amp0 * Breq_z[0] / Breq_z[k] for k in range(4)])
    o = {"candidate": lab}
    for bk, m in mult.items():
        o[bk] = verdict(list(R_old * m))
    other.append(o)
OUT["other_candidates"] = other
check("S4", abs(other[-2]["HH"]["S"][0] - mw["a_decoherence"]["all_matter_constant_rate_shape"]["1.0"]) < 1e-9,
      "constant-per-particle shape under HH reproduces the stored S(z=1) = %.2f" % other[-2]["HH"]["S"][0])

# shape scan (diagnostic only; thresholds never used to claim a pass)
scan = {}
for model in ["tinker08", "sheth99"]:
    rr = []
    for r in mw["shape_scan_diagnostic"][model]["rows"]:
        SfP = [r["S_f"][k] * mult["P"][k] / mult["P"][0] for k in range(4)]
        SWP = [r["S_fW"][k] * mult["P"][k] / mult["P"][0] for k in range(4)]
        rr.append({"log10M": r["log10M"], "S_f_P": SfP[1:], "S_fW_P": SWP[1:],
                   "pass_f_P": bool(all(0.5 <= s <= 2 for s in SfP[1:])),
                   "pass_fW_P": bool(all(0.5 <= s <= 2 for s in SWP[1:]))})
    best_f = min(rr, key=lambda x: max(abs(np.log2(s)) for s in x["S_f_P"]))
    best_W = min(rr, key=lambda x: max(abs(np.log2(s)) for s in x["S_fW_P"]))
    scan[model] = {"rows": rr, "best_f_P": best_f, "best_fW_P": best_W,
                   "any_pass_f_P": [x["log10M"] for x in rr if x["pass_f_P"]],
                   "any_pass_fW_P": [x["log10M"] for x in rr if x["pass_fW_P"]]}
OUT["shape_scan_P_diagnostic"] = scan

# ======================================================= L: local writing temperatures (colossus, same settings)
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from colossus.halo import concentration as cc
from scipy import integrate
cosmo = cosmology.setCosmology("book_planck18", params=dict(flat=True, H0=H0kms, Om0=Om, Ob0=Ob,
                               sigma8=0.8111, ns=0.9649, relspecies=True, persistence=""))
rho_m0 = cosmo.rho_m(0.0) * 1e9           # Msun h^2 / Mpc^3, comoving
lnM = np.linspace(np.log(1e6), np.log(1e16), 400)
Mg = np.exp(lnM)


def w_weyl(cn):
    m = lambda x: np.log1p(x) - x / (1 + x)
    I = integrate.quad(lambda x: m(x) / x, 0, cn)[0]
    return I / m(cn) - 1 / 3


cgrid = np.linspace(1, 40, 79)
wgrid = np.array([w_weyl(x) for x in cgrid])


def rcum(y):   # identical to missing-writer cum(): trapezoid from index i to the end
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(lnM)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


rho_c0_SI = 3 * H0 ** 2 / (8 * np.pi * G)
mu_gas = 0.59                       # fully ionized primordial gas (Barkana & Loeb 2001 convention)


def halo_temps(z):
    """per-mass writing temperatures at r_200m (physical): virial T and Deser-Levin local Unruh T"""
    Mphys = Mg / h * Msun
    rho_m = Om * rho_c0_SI * (1 + z) ** 3
    r200 = (3 * Mphys / (4 * np.pi * 200 * rho_m)) ** (1 / 3)
    Vc2 = G * Mphys / r200
    Tvir = mu_gas * m_p * Vc2 / (2 * kB)
    g = G * Mphys / r200 ** 2
    HL_SI = H0 * np.sqrt(OL)
    T_DL = hbar / (2 * np.pi * c * kB) * np.sqrt(g ** 2 + c ** 2 * HL_SI ** 2)    # Deser-Levin 1997 (dS)
    T_DLH = hbar / (2 * np.pi * c * kB) * np.sqrt(g ** 2 + c ** 2 * (H0 * EH(z)) ** 2)   # FRW analogue (IAM IDENTIFICATION)
    return Tvir, T_DL, T_DLH, g


def integrands(model, mdef, z):
    dn = mf.massFunction(Mg, z, q_in="M", q_out="dndlnM", mdef=mdef, model=model)
    I = dn * Mg / rho_m0
    conc = cc.concentration(Mg, "200m", z, model="duffy08")
    return I, I * np.interp(conc, cgrid, wgrid)


T_H_of = lambda z: hbar * H0 * EH(z) / (2 * np.pi * kB)
T_dS0 = hbar * H0 * np.sqrt(OL) / (2 * np.pi * kB)
thr = [1e8, 1e11, 1e12, 1e13, 1e14]
LOC = {}
repro = []
dz = 0.02
for model, mdef in [("tinker08", "200m"), ("sheth99", "fof")]:
    res = {"%.0e" % M: {} for M in thr}
    for z in zs:
        I, IW = integrands(model, mdef, z)
        Ip, IWp = integrands(model, mdef, max(z - dz, 0) if z > 0 else 0)
        Im, IWm = integrands(model, mdef, z + dz)
        if z > 0:
            dIW = -(1 + z) * (IWp - IWm) / (-2 * dz)
        else:
            dIW = -(1 + z) * (IW - IWm) / (-dz)
        Tvir, TDL, TDLH, g = halo_temps(z)
        TH = T_H_of(z)
        for M in thr:
            i = np.searchsorted(Mg, M)
            key = "%.0e" % M
            stored_fW = mw["halos"][model][key]["f_Weyl"][str(z)]
            repro.append(abs(rcum(IW)[i] / stored_fW - 1))
            Breq = fB(1 / (1 + z))
            out = {}
            # ledger-energy ratio = int integrand * T_L/T_W dlnM / B_req   (integrand in units of M_m c^2)
            for TLlab, TL in [("ledger_TdS", T_dS0), ("ledger_TH", TH)]:
                for TWlab, TW in [("Tvir", Tvir), ("T_DeserLevin_dS", TDL), ("T_DL_FRW", TDLH),
                                  ("T_CMB", 2.7255 * (1 + z) * np.ones_like(Mg)), ("T_const_1e4K", 1e4 * np.ones_like(Mg))]:
                    out[TLlab + "|" + TWlab + "|c2_stock"] = rcum(IW * TL / TW)[i] / Breq
                    out[TLlab + "|" + TWlab + "|c3_flow"] = rcum(dIW * TL / TW)[i] / Breq
            # temperature diagnostics at the threshold mass
            out["_Tvir_thr_K"] = float(Tvir[i]); out["_g_over_cHL_thr"] = float(g[i] / (c * H0 * np.sqrt(OL)))
            out["_TDL_over_TdS_thr"] = float(TDL[i] / T_dS0)
            res[key][str(z)] = out
    LOC[model] = res
check("L1", max(repro) < 1e-6, "colossus pipeline reproduces the stored f_Weyl at all 5 thresholds, 4 redshifts, 2 mass functions (max rel dev %.1e)" % max(repro))

# collect local verdicts
loc_rows = []
labels = [k for k in LOC["tinker08"]["1e+13"]["0.0"] if not k.startswith("_")]
for model in LOC:
    for key in LOC[model]:
        for lab in labels:
            R = [LOC[model][key][str(z)][lab] for z in zs]
            if R[0] <= 0:
                continue
            v = verdict(R)
            loc_rows.append({"model": model, "M_min": key, "booking": lab, **v})
OUT["local_rows"] = loc_rows
OUT["local_passes"] = [(r["model"], r["M_min"], r["booking"]) for r in loc_rows if r["amp_pass"] and r["shape_pass"]]
OUT["local_shape_passes"] = [(r["model"], r["M_min"], r["booking"], r["amp"]) for r in loc_rows if r["shape_pass"]]
OUT["local_temp_diag"] = {m: {k: {z: {kk: v for kk, v in LOC[m][k][z].items() if kk.startswith("_")} for z in LOC[m][k]} for k in LOC[m]} for m in LOC}
# check: Deser-Levin at halo g << c H_L returns the floor to < 1 %
gmax = max(LOC[m]["1e+14"]["0.0"]["_g_over_cHL_thr"] for m in LOC)
check("L2", gmax < 0.2 and all(abs(LOC[m][k]["0.0"]["_TDL_over_TdS_thr"] - 1) < 0.02 for m in LOC for k in LOC[m]),
      "Deser-Levin local temperature at r_200m equals T_dS to < 2 %% for all thresholds today (g/cH_L <= %.3f)" % gmax)
# check: Tvir formula sanity, 1e12 Msun/h at z=0 within 1e5.5 - 1e6.5 K
T12 = LOC["tinker08"]["1e+12"]["0.0"]["_Tvir_thr_K"]
check("L3", 3e5 < T12 < 3e6, "T_vir(1e12 Msun/h, z=0, Delta=200m, mu=0.59) = %.2e K (Milky-Way-size halo ~1e6 K)" % T12)

# ======================================================= F: forward record under P and the mu(z) it predicts
zg = np.linspace(0, 8, 81)
lna = -np.log(1 + zg)
ag = 1 / (1 + zg)
sOL = np.sqrt(fOLa(ag))
# F1: method check: B_req_floor x sqrt(OL(a)) through the book's accumulation rule returns beta_m [E(1) - E(1/9)]
dens = fBP(ag) * sOL * Om * ag ** -3
cum = -np.trapezoid(dens, lna)
check("F1", abs(cum / ((Om / 2) * (1 - np.exp(1 - 9))) - 1) < 2e-3,
      "forward integral of B_req_floor priced at T_dS returns beta_m [E(1)-E(1/9)] = %.4f rho_c0" % cum)


def mu_of(z, X):        # book eq:tw_mu (p2_18:67), X = record term in units of rho_c0 (beta_m E for the book)
    HL2 = Om * (1 + z) ** 3 + OL
    return HL2 / (HL2 + X)


bookmu = {str(z): mu_of(z, (Om / 2) * np.exp(-z)) for z in [0.0, 0.5, 1.0]}
check("F2", abs(bookmu["0.5"] - 0.9482) < 2e-4 and abs(bookmu["1.0"] - 0.9822) < 2e-4 and abs(bookmu["0.0"] - 0.864) < 1e-3,
      "book mu(z) reproduced: %.4f, %.4f, %.4f at z = 0, 0.5, 1 (book 0.864, 0.9482, 0.9822; implied_history %.4f, %.4f)"
      % (bookmu["0.0"], bookmu["0.5"], bookmu["1.0"], hist["book E(a) (measured shape)"]["mu"]["0.5"], hist["book E(a) (measured shape)"]["mu"]["1.0"]))
FWD = {}
zq = [0.5, 0.6, 0.75, 0.9, 1.0, 2.0, 3.0]
for model, mdef in [("tinker08", "200m"), ("sheth99", "fof")]:
    Ys = {M: {"W": [], "F": []} for M in [1e8, 1e11, 1e13]}
    for z in zg:
        I, IW = integrands(model, mdef, z)
        FW, FF_ = rcum(IW), rcum(I)
        for M in Ys:
            i = np.searchsorted(Mg, M)
            Ys[M]["W"].append(FW[i]); Ys[M]["F"].append(FF_[i])
    for M in Ys:
        for lab, Y, wgt in [("c3_CET_Weyl_flow", np.array(Ys[M]["W"]), 1.0), ("b3_virial_half_flow", np.array(Ys[M]["F"]), 0.5)]:
            Bz = wgt * np.gradient(Y, lna)
            for bk, fac in [("HH", np.ones_like(zg)), ("P", sOL)]:
                d_ = Bz * fac * Om * (1 + zg) ** 3
                cm = np.array([-np.trapezoid(d_[k:], lna[k:]) for k in range(len(zg))])
                tot = cm[0]
                Xi = lambda zz: float(np.interp(zz, zg, cm))
                FWD["%s_%s_%.0e_%s" % (model, lab, M, bk)] = {
                    "rho_info0_over_rho_c0": float(tot), "over_beta_m": float(tot / (Om / 2)),
                    "E_eff": {str(zz): Xi(zz) / tot for zz in zq},
                    "mu_shape_only": {str(zz): mu_of(zz, (Om / 2) * Xi(zz) / tot) for zz in [0.0] + zq},
                    "mu_absolute": {str(zz): mu_of(zz, Xi(zz)) for zz in [0.0] + zq}}
# HH method reproduction of stored forward rows
dv = max(abs(FWD["%s_%s_%.0e_HH" % (m, l, M)]["over_beta_m"] / mw["implied_record_forward"]["%s_%s_%.0e" % (m, l, M)]["over_beta_m"] - 1)
         for m in ["tinker08", "sheth99"] for l in ["c3_CET_Weyl_flow", "b3_virial_half_flow"] for M in [1e8, 1e11, 1e13])
check("F3", dv < 1e-6, "forward HH rows reproduce the stored missing-writer forward amplitudes (max rel dev %.1e)" % dv)
FWD["book"] = {"E_eff": {str(zz): float(np.exp(-zz)) for zz in zq},
               "mu": {str(zz): mu_of(zz, (Om / 2) * np.exp(-zz)) for zz in [0.0] + zq}}
OUT["forward"] = FWD
OUT["checks"] = CHECKS
OUT["checks_passed"] = sum(x["pass"] for x in CHECKS)
OUT["checks_failed"] = sum(not x["pass"] for x in CHECKS)
json.dump(OUT, open("floor_pricing_checks_output.json", "w"), indent=1, default=float)
print("passed", OUT["checks_passed"], "failed", OUT["checks_failed"])
