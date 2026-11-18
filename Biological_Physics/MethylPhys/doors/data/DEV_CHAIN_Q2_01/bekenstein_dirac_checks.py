import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', 'sympy', 'numpy', 'scipy', 'mpmath'], check=True)
"""Bekenstein 1981 rate, Dirac's large numbers, and the saturation prediction (IAM web, 2026-11-18).
Forward only. Every derived step is a check K#. Writes bekenstein_dirac_checks_output.json.
Inputs: book eq:law_Sneed (p1_02_iams_law.tex:330-333), eq:Hm (:344), E(1)=1 by definition (:380),
p2_20_wz_far_future.tex table and far-future numbers; missing-writer JSON; implied-history JSON.
"""
import json, math, sys
import sympy as sp
import numpy as np
from scipy import constants as C
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

MW = sys.argv[1] if len(sys.argv) > 1 else None   # missing_writer_checks_output.json
IH = sys.argv[2] if len(sys.argv) > 2 else None   # implied_history.json

checks, out = [], {}
def chk(cid, ok, note, **vals):
    ok = bool(ok)
    checks.append({"id": cid, "pass": ok, "note": note, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in vals.items()}})
    print(("PASS " if ok else "FAIL ") + cid + "  " + note[:110])

# ---------------- symbols ----------------
E_, R, c, hbar, kB, G, H, H0, m, M, a = sp.symbols('E R c hbar k_B G H H_0 m M a', positive=True)
Om, bm, Ea, OL, Ob = sp.symbols('Omega_m beta_m E_a Omega_L Omega_b', positive=True)

# K1 Bekenstein rate = entropy bound / minimum crossing time 2R/c ; R and c cancel
S_bound = 2*sp.pi*R*E_/(hbar*c)            # nats (k_B=1), Bekenstein 1981 PRD
rate = sp.simplify(S_bound/(2*R/c))
chk("K1", sp.simplify(rate - sp.pi*E_/hbar) == 0, "Bekenstein PRL 1981: (2 pi E R/hbar c)/(2R/c) = pi E/hbar nats/s; R and c cancel", )

# K2 for E = m c^2: rate = pi c / lambda_bar (nats/s); rate/H = pi (c/H)/lambda_bar = pi * D
lam = hbar/(m*c)
rate_m = (sp.pi*E_/hbar).subs(E_, m*c**2)
chk("K2", sp.simplify(rate_m - sp.pi*c/lam) == 0 and sp.simplify(rate_m/H - sp.pi*(c/H)/lam) == 0,
    "pi m c^2/hbar = pi x (light-crossing frequency c/lambda_bar of the reduced Compton wavelength); divided by H it is pi x (Hubble radius / lambda_bar) = pi D")

# K3 pi E/(hbar H) = E/(2 k T_H), T_H = hbar H/(2 pi k)
TH = hbar*H/(2*sp.pi*kB)
chk("K3", sp.simplify(sp.pi*E_/(hbar*H) - E_/(2*kB*TH)) == 0, "Bekenstein rate x one Hubble time = E/(2 k_B T_H)")

# K4 Bekenstein entropy of mass M at R=c/H; critical-density Hubble mass gives S_geo
S_geo = sp.pi*c**5/(hbar*G*H**2)
M_H = c**3/(2*G*H)                                    # rho_crit (4pi/3)(c/H)^3
S_bek = lambda MM: 2*sp.pi*MM*c**2*(c/H)/(hbar*c)
chk("K4", sp.simplify(S_bek(M_H) - S_geo) == 0 and sp.simplify(sp.pi*M*c**2/(hbar*H) - S_bek(M)/2) == 0,
    "S_Bek(M_H, c/H) = S_geo; pi M c^2/(hbar H) = S_Bek(M, c/H)/2")

# K5 book record per e-fold (eq:law_Sneed general a) / [pi M_m c^2/(hbar H)]
# dS/dln a = beta_m E(a) H0^2/(a H^2) * S_geo(H);  M_m(a) = Omega_m rho_c0 a^-3 V_H(H)
rho_c0 = 3*H0**2/(8*sp.pi*G)
M_m = Om*rho_c0*a**-3*sp.Rational(4, 3)*sp.pi*(c/H)**3
rec = bm*Ea*H0**2/(a*H**2)*S_geo
ratio = sp.simplify(rec/(sp.pi*M_m*c**2/(hbar*H)))
chk("K5", sp.simplify(ratio - 2*bm*Ea*a**2/Om) == 0, "record/(pi M_m c^2/hbar H) = 2 beta_m E(a) a^2/Omega_m at every a (H, G, hbar, c cancel)", ratio=str(ratio))

# K6 = 1 at a=1 with beta_m = Omega_m/2 and E(1)=1 (normalization by definition, p1_02:380)
chk("K6", sp.simplify(ratio.subs({bm: Om/2, Ea: 1, a: 1}) - 1) == 0, "exact today: needs beta_m=Omega_m/2 AND E(1)=1 (the latter by definition)")

# K7 H0 cancels: the identity is equivalent to beta_m = Omega_m/2 for ANY H0, m, G, hbar
ident = sp.simplify(rec.subs({a: 1, Ea: 1, H: H0}) - sp.pi*M_m.subs({a: 1, H: H0})*c**2/(hbar*H0))
sol = sp.solve(sp.Eq(ident, 0), bm)
chk("K7", sol == [Om/2], "identity today <=> beta_m = Omega_m/2; no H0, m, G, hbar survive, so it fixes no H0 and no large number", solution=str(sol))

# ---------------- numbers ----------------
mp, me, e_ch, eps0 = C.m_p, C.m_e, C.e, C.epsilon_0
hb, cc, GG, kk = C.hbar, C.c, C.G, C.k
Mpc = 3.0856775814913673e22
Om_v, Ob_v, Orad = 0.3153, 0.0493, 9.22e-5
bm_v = Om_v/2
def Hs(h): return h*1e3/Mpc
num = {}
for h in (67.4, 67.16):
    H0s = Hs(h)
    D = mp*cc**2/(hb*H0s)
    num[h] = dict(D_p=D, piD=math.pi*D, piD_bits=math.pi*D/math.log(2),
                  N_m=Om_v*cc**3/(2*GG*H0s*mp), S_geo=math.pi*cc**5/(hb*GG*H0s**2))
    num[h]["rec_today"] = bm_v*num[h]["S_geo"]
out["per_proton"] = num
n = num[67.4]
chk("K8", abs(n["piD"]/2.05e42 - 1) < 5e-3 and abs(n["D_p"]/6.5e41 - 1) < 1e-2,
    "lead's numbers: pi m_p c^2/(hbar H0) = %.4e nats per e-fold, m_p c^2/(hbar H0) = %.4e (H0=67.4)" % (n["piD"], n["D_p"]),
    piD=n["piD"], D_p=n["D_p"], piD_6716=num[67.16]["piD"])

mwj = json.load(open(MW)) if MW else None
book_booking = mwj["requirement"]["booking_per_efold_today_nats"] if mwj else float('nan')
chk("K9", abs(n["N_m"]*n["piD"]/n["rec_today"] - 1) < 1e-12 and abs(n["rec_today"]/book_booking - 1) < 2e-3,
    "N_m x pi D_p = beta_m S_geo = %.4e nats per e-fold; missing-writer booking %.4e" % (n["rec_today"], book_booking),
    N_m=n["N_m"], rec_today=n["rec_today"], missing_writer=book_booking)

bit_rate = math.pi*mp*cc**2/(hb*math.log(2))
lam_p = hb/(mp*cc)
chk("K10", abs(bit_rate/6.46e24 - 1) < 2e-3 and abs(cc/bit_rate/(math.log(2)/math.pi*lam_p) - 1) < 1e-12,
    "Bekenstein rate per proton %.4e bits/s; light travels (ln2/pi) lambda_bar_p = %.3e m per bit" % (bit_rate, cc/bit_rate),
    bits_per_s=bit_rate, m_per_bit=cc/bit_rate, lambda_bar_p=lam_p)

# Dirac / Eddington / Weinberg numbers
alphaG_pp = GG*mp**2/(hb*cc)
alphaG_pe = GG*mp*me/(hb*cc)
alpha = C.fine_structure
N1 = e_ch**2/(4*math.pi*eps0*GG*mp*me)
t_e = e_ch**2/(4*math.pi*eps0*me*cc**3)
H0s = Hs(67.4)
Ez = lambda z: math.sqrt(Om_v*(1+z)**3 + Orad*(1+z)**4 + (1-Om_v-Orad))
t0 = quad(lambda z: 1/((1+z)*Ez(z)), 0, np.inf)[0]/H0s
N2_H = (1/H0s)/t_e; N2_t = t0/t_e
D_p = n["D_p"]
EW = hb**2*H0s/(GG*cc*mp**3)    # Eddington-Weinberg: hbar^2 H ~ G c m^3  <=>  D ~ alpha_G^-1
N_Edd = 136*2**256
out["dirac"] = dict(N1_force_ratio=N1, alpha=alpha, alphaG_pp_inv=1/alphaG_pp, alphaG_pe_inv=1/alphaG_pe,
                    N2_age_over_e2_mc3_using_1overH0=N2_H, N2_using_t0=N2_t, t0_Gyr=t0/(3.15576e16),
                    D_p=D_p, D_e=me*cc**2/(hb*H0s), EddWeinberg_ratio=EW, D_over_alphaGinv_pp=D_p*alphaG_pp,
                    D_over_alphaGinv_pe=D_p*alphaG_pe, N_m_hubble=n["N_m"], N_b_hubble=Ob_v/Om_v*n["N_m"],
                    N_Edd=float(N_Edd), N1_over_N2t=N1/N2_t)
chk("K11", 1e39 < N1 < 1e40 and abs(N1/(alpha/alphaG_pe) - 1) < 1e-10,
    "Dirac 1937 force ratio e^2/(4 pi eps0 G m_p m_e) = %.3e = alpha x hbar c/(G m_p m_e)" % N1, N1=N1)
chk("K12", 1e40 < N2_t < 1e41, "Dirac epoch t0/(e^2/4 pi eps0 m_e c^3) = %.3e (t0 = %.3f Gyr), 1/H0 version %.3e" % (N2_t, t0/3.15576e16, N2_H), N2=N2_t)

# K13 Friedmann alone: N_m = (Omega_m/2) alpha_G^-1 D  (exact)  -> the IAM identity contains D and alpha_G only as this product
Nm_sym = Om*c**3/(2*G*H*m)
D_sym = m*c**2/(hbar*H); aG_sym = G*m**2/(hbar*c)
chk("K13", sp.simplify(Nm_sym - Om/2*D_sym/aG_sym) == 0 and abs(n["N_m"]/(Om_v/2*D_p/alphaG_pp) - 1) < 1e-12,
    "Friedmann: N_m = (Omega_m/2) D / alpha_G exactly; record per e-fold = pi N_m D = (Omega_m/2) pi D^2/alpha_G")

# K14 the identity cannot fix D x alpha_G (Dirac/Weinberg coincidence): after Friedmann substitution the residual is free of m
resid = sp.simplify((bm*S_geo - sp.pi*Nm_sym*m*c**2/(hbar*H)).subs(bm, Om/2))
chk("K14", resid == 0 and not resid.has(m), "with Friedmann + beta_m=Omega_m/2 the identity is 0=0 for every particle mass: D alpha_G is unconstrained",
    D_alphaG_pp=D_p*alphaG_pp, D_alphaG_pe=D_p*alphaG_pe, EddWeinberg=EW)

# K15 holographic count = Dirac N x Dirac D (numerically, Hubble sphere, proton-mass units)
chk("K15", abs(n["S_geo"]/(n["N_m"]*D_p) - 2*math.pi/Om_v) < 1e-6*2*math.pi/Om_v,
    "S_geo = (2 pi/Omega_m) N_m D_p: %.3e = %.3e x %.3e x %.2f" % (n["S_geo"], n["N_m"], D_p, 2*math.pi/Om_v),
    N_m=n["N_m"], N_m_over_Eddington=n["N_m"]/N_Edd)

# ---------------- saturation ----------------
# K16 general: continuity of the record component (eq:law_F2info) gives d rho_info/d ln a = (3 H^4/8 pi^2 G) dS/dln a... with G kept:
# rho_x+P_x = -H^3 Sdot/(8 pi^2) in hbar=c=k=1 units (p1_02:329). d rho/dln a = 3 (rho+P)(-1) = 3 H^4 (dS/dln a)/(8 pi^2)
s = sp.symbols('s', positive=True)
dS = s*sp.pi*M_m*c**2/(hbar*H)               # writing at fraction s of matter's Bekenstein budget per Hubble time
drho = 3*H**4*dS/(8*sp.pi**2)
drho = drho.subs({c: 1, hbar: 1})
rho_m = Om*rho_c0*a**-3
chk("K16", sp.simplify(drho - s*rho_m/2) == 0,
    "d rho_info/d ln a = (s/2) rho_m(a): at the Bekenstein ceiling (s=1) the record gains half of matter's rest-energy density per e-fold")
Ef = sp.exp(1 - 1/a)
s_book = sp.simplify((bm*rho_c0*a*sp.diff(Ef, a))/(rho_m/2)).subs(bm, Om/2)   # dE/dln a = a dE/da
chk("K17", sp.simplify(s_book - a**2*Ef) == 0, "book: s(a) = a^3 dE/dln a = E(a) a^2 (the ceiling condition is a^3 dE/dln a <= 1, not E a^2 <= 1 in general)")

aa = np.linspace(0.05, 10, 4000)
sb = aa**2*np.exp(1 - 1/aa)
chk("K18", np.all(np.diff(sb) > 0) and abs(brentq(lambda x: x**2*math.exp(1-1/x) - 1, 0.5, 2) - 1) < 1e-12,
    "E a^2 rises monotonically (d ln/d ln a = 2 + 1/a > 0) and crosses 1 only at a = 1", values={str(x): float(x**2*math.exp(1-1/x)) for x in (0.5, 1.25, 1.5, 2, 3, 10)})

# K19 saturated continuation for a>=1: dE/dln a = a^-3, E(1)=1  ->  E_sat = (4 - a^-3)/3
Es = (4 - a**-3)/3
chk("K19", sp.simplify(a*sp.diff(Es, a) - a**-3) == 0 and Es.subs(a, 1) == 1 and sp.limit(Es, a, sp.oo) == sp.Rational(4, 3),
    "E_sat(a) = (4 - a^-3)/3 for a >= 1; E_sat -> 4/3 (book: E -> e); rate continuous at a=1 (both = 1)")

ihj = json.load(open(IH)) if IH else None
if ihj:
    r4a = ihj["(4a) 1/2 M_m(a) c^2 (matter inside horizon now)"]["E"]
    pred = {z: (4 - (1+float(z))**3)/3 for z in r4a}
    dev = max(abs(r4a[z] - pred[z]) for z in r4a)
else:
    r4a, pred, dev = {}, {}, float('nan')
chk("K19b", dev < 2e-3, "ledger-history route (4a) '1/2 M_m(a) c^2' at z=0.5,1,2,3 equals (4-a^-3)/3: the saturated law run backwards (fails in the past, E<0 at z>0.59)",
    route4a={k: float(v) for k, v in r4a.items()}, formula={k: float(v) for k, v in pred.items()}, max_dev=dev)

# K20 maturity: fraction of the final record written today
chk("K20", abs(float(1/sp.Rational(4, 3)) - 0.75) < 1e-15, "saturated: 3/4 of the final record written today (book: 1/e = 0.368)")

# K21 w for the record component
w_book = -1 - sp.diff(sp.log(Ef), a)*a/3
w_sat = sp.simplify(-1 - (a*sp.diff(Es, a))/(3*Es))
chk("K21", sp.simplify(w_sat - (-1 - 1/(4*a**3 - 1))) == 0 and w_sat.subs(a, 1) == sp.Rational(-4, 3)
    and sp.simplify(sp.diff(w_sat, a).subs(a, 1) - sp.Rational(4, 3)) == 0 and sp.simplify(sp.diff(w_book, a).subs(a, 1) - sp.Rational(1, 3)) == 0,
    "w_sat = -1 - 1/(4a^3-1): -4/3 today (same as book), slope dw/da = 4/3 just after today (book 1/3); -> -1",
    w_sat_a2=float(w_sat.subs(a, 2)), w_book_a2=float(w_book.subs(a, 2)), w_sat_a1p5=float(w_sat.subs(a, 1.5)), w_book_a1p5=float(w_book.subs(a, 1.5)))

OL_v = 1 - Om_v - Orad
OL_b = 0.685; Om_b = 0.315; bm_b = 0.15765     # p2_20 table inputs
chk("K22", abs(bm_b*math.e/OL_b - 0.626) < 1e-3 and abs(bm_b/OL_b - 0.230) < 1e-3,
    "book p2_20 reproduced: rho_info/rho_L = 0.230 today, 0.626 at E->e; saturated asymptote %.3f" % (bm_b*4/3/OL_b),
    sat_asym=bm_b*4/3/OL_b, sat_share=bm_b*4/3/(OL_b+bm_b*4/3))

Hinf = {}
for h, OLx in ((67.16, 1 - 0.3153 - 9.22e-5), (67.4, 0.685)):
    Hinf[h] = dict(book=h*math.sqrt(OLx + bm_v*math.e), sat=h*math.sqrt(OLx + bm_v*4/3), photon=h*math.sqrt(OLx))
out["H_inf"] = Hinf
chk("K23", abs(Hinf[67.16]["book"] - 70.86) < 0.02 and abs(Hinf[67.4]["book"] - 71.12) < 0.02,
    "book H_m,inf 70.86 (67.16), 71.12 (67.4) reproduced; saturated %.2f, %.2f; photon %.2f, %.2f" % (Hinf[67.16]["sat"], Hinf[67.4]["sat"], Hinf[67.16]["photon"], Hinf[67.4]["photon"]))

# K24 bound on H_m for a>=1
x = sp.symbols('x', positive=True)    # x = a^-3 in (0,1]
Hm2_sat = Om*x + OL + bm*(4 - x)/3
dHdx = sp.diff(Hm2_sat, x)
chk("K24", float(dHdx.subs({Om: Om_v, bm: bm_v})) > 0 and abs(math.sqrt(1 + bm_v) - 1.0759) < 1e-3 and abs(math.sqrt(1 + bm_b*math.e) - 1.195) < 1e-3,
    "saturated: H_m^2 decreases monotonically for a>=1 (d/d(a^-3) = Omega_m - beta_m/3 > 0), so H_m <= sqrt(1+beta_m) H0 = 1.0759 H0 (book bound 1.195 H0)")

# K25 matter-sector total dark-sector w (Lambda + record)
def wtot(av, sat):
    if sat and av >= 1: E, dE = (4 - av**-3)/3, av**-3
    else: E = math.exp(1 - 1/av); dE = E/av
    return -1 - bm_v*dE/(3*(OL_v + bm_v*E))
wt = {str(av): dict(book=wtot(av, False), sat=wtot(av, True)) for av in (0.5, 1.0, 1.5, 2.0, 5.0)}
out["w_tot_matter_sector"] = wt
chk("K25", abs(wt["1.0"]["book"] - (-1.0624)) < 2e-4 and wt["0.5"]["book"] == wt["0.5"]["sat"],
    "matter-sector Lambda+record w today %.4f in both readings (dark-sector report -1.062); identical for every a<1" % wt["1.0"]["book"])

# K26 horizon-wide ceiling: record per e-fold / (S_geo/2) = E a^2 Omega_m(a) (book); max < 1
Om_a = lambda av: Om_v*av**-3/(Om_v*av**-3 + Orad*av**-4 + OL_v)
f = lambda av: av**2*math.exp(1 - 1/av)*Om_a(av)
res = minimize_scalar(lambda av: -f(av), bounds=(0.2, 20), method='bounded')
chk("K26", -res.fun < 1, "horizon-wide ceiling (all of M_H c^2 at Bekenstein): book ratio E a^2 Omega_m(a) peaks at %.3f at a = %.3f; never binding" % (-res.fun, res.x),
    peak=-res.fun, a_peak=res.x, today=f(1.0))

# K27 baryons-only writers
ab = brentq(lambda av: av**2*math.exp(1 - 1/av) - Ob_v/Om_v, 0.2, 1)
chk("K27", 0 < ab < 1, "if only baryons write, book rate exceeds their Bekenstein ceiling from a = %.3f (z = %.2f); today by Omega_m/Omega_b = %.2f" % (ab, 1/ab - 1, Om_v/Ob_v),
    a_cross=ab, z_cross=1/ab - 1, today_ratio=Om_v/Ob_v)

# K28 DESI DR2 (arXiv:2503.14738, as printed): all three w0 > -1; IAM record w(0) = -4/3, matter-sector total -1.062; both readings identical over the data range (a<1)
desi = {"Pantheon+": (-0.838, 0.055, -0.62), "Union3": (-0.667, 0.088, -1.09), "DESY5": (-0.752, 0.057, -0.86)}
cross = {k: 1 + (v[0] + 1)/v[2] for k, v in desi.items()}   # a where w0 + wa(1-a) = -1
chk("K28", all(v[0] > -1 for v in desi.values()) and all(abs(cross[k] - t) < 2e-3 for k, t in (("Pantheon+", 0.739), ("Union3", 0.694), ("DESY5", 0.712))),
    "DESI DR2 w0 > -1 in all three; CPL crosses -1 at a = 0.739/0.694/0.712 (book p2_20 reproduced); data at a<1 cannot see the saturation (it acts only for a>1)", cross=cross)

# K29 book w_info table check
chk("K29", abs(float(w_book.subs(a, 1)) + 4/3) < 1e-12 and abs(float(w_book.subs(a, sp.Rational(1, 2))) + 5/3) < 1e-12, "book w_info = -1 - 1/(3a): -4/3 today, -5/3 at z=1 reproduced")

# K30 rate ratio Bekenstein/expansion per proton is pi D: the 'match' to H0 is per e-fold, not per second
chk("K30", abs((math.pi*mp*cc**2/hb)/H0s/n["piD"] - 1) < 1e-12, "(Bekenstein rate per proton)/H0 = pi D_p = %.3e: the record per proton per e-fold, not a match of rates" % n["piD"])

out["checks"] = checks
out["checks_passed"] = sum(c_["pass"] for c_ in checks)
out["checks_failed"] = sum(not c_["pass"] for c_ in checks)
json.dump(out, open("bekenstein_dirac_checks_output.json", "w"), indent=1, default=float)
print("passed", out["checks_passed"], "failed", out["checks_failed"])
