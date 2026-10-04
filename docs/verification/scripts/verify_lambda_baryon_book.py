#!/usr/bin/env python3
"""Book carriage check for Part 2, Chapters 'The cosmological constant' (p2_12_lambda.tex), 'The history of the
cosmological constant' (p2_12b_lambda_history.tex), 'The baryon density' (p2_13_baryon.tex) and 'The baryon density
from the microwave background alone' (p2_13b_baryon_chain.tex).

Every equation and number those chapters carry is recomputed here; algebra with sympy, numbers with CODATA
(scipy.constants). Chains: Cosmological_Physics/mgcamb_validation/chains/*.1.txt, 30 % burn-in, weighted (book convention).
Run from the repository root:  python docs/verification/scripts/verify_lambda_baryon_book.py
Output: docs/verification/scripts/verify_lambda_baryon_book_output.txt
Extends verify_cc_and_baryon.py (same inputs, same chains); adds the sympy steps, the history integral in the form
printed, the cut-off factors, the horizon bit counts, the look-elsewhere count of this book, and the comparators traced to
Planck 2018 VI (Table 2) and Cyburt et al. 2016 (Table IV).
"""
import itertools, numpy as np, pandas as pd, scipy.constants as C, sympy as sp
from fractions import Fraction
from scipy.integrate import quad

hbar, c, G, kB = C.hbar, C.c, C.G, C.k
Mpc = 3.0857e22; GeV = 1e9 * C.e
lP = np.sqrt(hbar * G / c**3); EP = np.sqrt(hbar * c**5 / G); MP_GeV = EP / GeV
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

P("== A. Algebra (sympy) ==")
H, Gs, hb, cs, OL_, Ob_, Om_, lp, lh, pi = sp.symbols("H G hbar c Omega_L Omega_b Omega_m l_P l_H pi", positive=True)
rho_vac = cs**7 / (hb * Gs**2)                                   # E_P^4/(hbar c)^3
rho_L = OL_ * 3 * H**2 / (8 * sp.pi * Gs) * cs**2
ratio = sp.simplify(rho_L / rho_vac)
ident = 3 * OL_ / (8 * sp.pi) * (hb * Gs / cs**3) / (cs / H)**2
P("A1 E_P^4/(hbar c)^3 == c^7/(hbar G^2):", sp.simplify((sp.sqrt(hb * cs**5 / Gs))**4 / (hb * cs)**3 - rho_vac) == 0)
P("A2 rho_L/rho_vac - (3 OL/8pi)(l_P/l_H)^2 == 0:", sp.simplify(ratio - ident) == 0)
# de Sitter horizon: T_GH S_BH = M_H c^2, any H
T = hb * H / (2 * sp.pi); S = (4 * sp.pi * (cs / H)**2) / (4 * hb * Gs / cs**3)   # k_B = 1
MH = 3 * H**2 / (8 * sp.pi * Gs) * sp.Rational(4, 3) * sp.pi * (cs / H)**3 * cs**2
P("A3 T_GH S / (M_H c^2) =", sp.simplify(T * S / MH))
x = sp.symbols("x", positive=True)
sol = sp.solve(sp.Eq(2 / sp.pi * sp.sqrt(OL_) * x, 3 * OL_ / (8 * sp.pi)), x)
P("A4 (2/pi) sqrt(OL) Ob/Om = 3 OL/(8 pi)  =>  Ob/Om =", sol)
sol2 = sp.solve(sp.Eq(2 / sp.pi * x, 3 * OL_ / (8 * sp.pi)), x)
P("A5 without sqrt(OL): Ob/Om =", sol2)
Aeff = 2 * sp.pi * lh**2
P("A6 l_P^2/(A_eff/4pi), A_eff = 2 pi l_H^2:", sp.simplify(lp**2 / (Aeff / (4 * sp.pi))), " ; N = A_eff/(4 l_P^2) =", sp.simplify(Aeff / (4 * lp**2)))
P("A7 f_geo = l_P^2/(4 pi l_H^2) =", sp.simplify(lp**2 / (4 * sp.pi * lh**2)), "; (2/pi)/f_geo coefficient ratio =", sp.simplify((2 / sp.pi) / (1 / (4 * sp.pi))))
a = sp.symbols("a", positive=True); E = sp.exp(1 - 1 / a)
P("A8 dE/da =", sp.simplify(sp.diff(E, a)), "; at a=1:", sp.diff(E, a).subs(a, 1), "; E(oo) =", sp.limit(E, a, sp.oo))
w = sp.simplify(-1 - sp.Rational(1, 3) * a * sp.diff(sp.log(E), a))
P("A9 rho propto E(a): w = -1 - (1/3) dln rho/dln a =", w, "; w(1) =", w.subs(a, 1), " (growing density => w < -1)")
P("A10 virial ratio Om/[beta_m E(1)] with beta_m = Om/2:", sp.simplify(Om_ / (Om_ / 2 * E.subs(a, 1))))

P("\n== B. Planck 2018 inputs as used (H0 67.4, Ob 0.0493, Om 0.3153, OL 0.6846) ==")
H0 = 67.4e3 / Mpc; lH = c / H0; Ob, Om, OL = 0.0493, 0.3153, 0.6846
rvac = EP**4 / (hbar * c)**3; rc = 3 * H0**2 / (8 * np.pi * G); rL = OL * rc * c**2; obs = rL / rvac
P(f"B1 E_P = {EP:.3e} J; l_P = {lP:.4e} m; H0 = {H0:.4e} s^-1; l_H = {lH:.4e} m; l_H/l_P = {lH/lP:.3e}")
P(f"B2 rho_vac = {rvac:.4e} J/m3; rho_L = {rL:.4e} J/m3; ratio = {obs:.4e}; log10 = {np.log10(obs):.3f}")
g = (lP / lH)**2; P(f"B3 (l_P/l_H)^2 = {g:.4e}; identity (3 OL/8pi) g = {3*OL/(8*np.pi)*g:.4e}; 3 OL/8pi = {3*OL/(8*np.pi):.5f}")
fb = Ob / Om; base = 2 / np.pi * g * fb; corr = base * np.sqrt(OL)
P(f"B4 Ob/Om = {fb:.4f}; g x Ob/Om = {g*fb:.4e} (x{g*fb/obs:.3f}); baseline (2/pi) g Ob/Om = {base:.4e} (x{base/obs:.4f}); g alone x{g/obs:.2f}")
P(f"B5 sqrt(OL) = {np.sqrt(OL):.4f}; corrected = {corr:.4e}; offset {100*(corr/obs-1):+.2f} %; exponent closing baseline p = {np.log(obs/base)/np.log(OL):.4f}")
P(f"B6 required coefficient K in K g Ob/Om = obs: {obs/(g*fb):.4f}; (3/16) sqrt(OL) = {3/16*np.sqrt(OL):.4f} vs Ob/Om {fb:.4f}; ratio {fb/(3/16*np.sqrt(OL)):.4f}")
Nn = 4 * np.pi * lH**2 / (4 * lP**2); P(f"B7 horizon today: A/(4 l_P^2) = {Nn:.3e} nats = {Nn/np.log(2):.3e} bits; A/l_P^2 = {4*np.pi*lH**2/lP**2:.3e}")
for nm, M in (("electroweak 100 GeV", 100.0), ("QCD 200 MeV", 0.2)):
    P(f"B8 cut-off {nm}: rho_vac falls by (E_P/M)^4 = {(MP_GeV/M)**4:.2e}; rho_L/rho_vac(M) = {obs*(MP_GeV/M)**4:.2e}")
TH = hbar * H0 / (2 * np.pi * kB); P(f"B9 T_GH(H0) = {TH:.3e} K; T_dS = {TH*np.sqrt(OL):.3e} K; E_bit = k T ln2 = {kB*TH*np.log(2):.3e} J")

P("\n== C. History integral, as printed: (Ob/Otot)(a) / (l_H(a)^2/l_P^2) da / (a^2 H/H0), from a_EW = 2.3e-15 ==")
Orad = 2.473e-5 / 0.6736**2 * (1 + 0.2271 * 3.046)   # photons (T0 = 2.7255 K) + 3.046 massless neutrinos
Hn = lambda a_: np.sqrt(Om / a_**3 + Orad / a_**4 + OL)
fbt = lambda a_: (Ob / a_**3) / (Om / a_**3 + Orad / a_**4 + OL)
integrand = lambda a_: fbt(a_) * Hn(a_)**2 / (a_**2 * Hn(a_))       # (l_P/l_H(a))^2 = g (H/H0)^2, g factored out
lnA = np.linspace(np.log(2.3e-15), 0, 200001); A_ = np.exp(lnA)
I = np.trapezoid(integrand(A_) * A_, lnA)
P(f"C1 Omega_rad = {Orad:.3e}; integral I (times g) = {I:.3e}; as coefficient K = I/(Ob/Om) = {I/fb:.2e} against the required {obs/(g*fb):.3f}")
aEW100 = (3.91 / 106.75)**(1 / 3) * 2.7255 * kB / GeV / 100; aEW160 = aEW100 * 100 / 159.5
for a1 in (aEW100, aEW160):
    lnB = np.linspace(np.log(a1), 0, 200001); B_ = np.exp(lnB)
    P(f"C1b lower limit a = {a1:.2e} (entropy-conserving a at {'100' if a1 == aEW100 else '159.5'} GeV): K = {np.trapezoid(integrand(B_)*B_, lnB)/fb:.2e}")
for a1 in (1e-10, 1e-6, 1e-3):
    m = A_ >= a1; P(f"C2 lower limit {a1:.0e}: K = {np.trapezoid(integrand(A_[m])*A_[m], lnA[m])/fb:.3e}")
P(f"C3 early-time slope dln(integrand)/dln a at a=1e-10: {np.gradient(np.log(integrand(A_)), lnA)[np.searchsorted(A_,1e-10)]:.3f}")
# activation-weighted forms, 18th-chain Omega_m = 0.3197 (as CC_AND_BARYON_CHECK item 4), and Planck 0.3153
for Omx in (0.3197, 0.3153):
    Hx = lambda a_: np.sqrt(Omx / a_**3 + 1 - Omx); dE = lambda a_: np.exp(1 - 1 / a_) / a_**2
    vals = [quad(lambda a_: Hx(a_)**p * dE(a_), 1e-6, 1, limit=200)[0] for p in (-2, -1, 0, 1, 2)]
    P(f"C4 Om = {Omx}: int (H/H0)^p dE for p = -2..2: " + ", ".join(f"{v:.3f}" for v in vals))

P("\n== D. Look-elsewhere count of this book ==")
# Family: K = q * pi^k * (Ob/Om)^i * OL^j ; q in distinct p/r with p, r = 1..6; k in {-1,0,1}; i in {0,1}; j in {0,1/2,1}.
qs = sorted({Fraction(p, r) for p in range(1, 7) for r in range(1, 7)})
target = obs / g; n = hit = 0; hits = []
for q, k, i, j in itertools.product(qs, (-1, 0, 1), (0, 1), (0, 0.5, 1)):
    v = float(q) * np.pi**k * fb**i * OL**j; n += 1
    if abs(v / target - 1) < 0.01: hit += 1; hits.append(f"{q}*pi^{k}*(Ob/Om)^{i}*OL^{j}")
P(f"D1 rationals {len(qs)}; forms {n}; within 1 % of (rho_L/rho_vac)/(l_P/l_H)^2 = {target:.5f}: {hit} ({100*hit/n:.1f} %)")
P("D2 the hits:", "; ".join(hits))

P("\n== E. Chains (30 % burn-in, weighted); eta = 273.9e-10 Ob h^2 (Steigman 2006) ==")
def chain(stem):
    f = f"Cosmological_Physics/mgcamb_validation/chains/{stem}.1.txt"; cols = open(f).readline().lstrip("#").split()
    X = pd.read_csv(f, sep=r"\s+", comment="#", names=cols); return X, X.iloc[int(0.3 * len(X)):]
def ms(v, w): m = np.average(v, weights=w); return m, np.sqrt(np.average((v - m)**2, weights=w))
for st in ("iam_baryon_test", "lcdm_baseline", "planck_bao_lcdm_baseline", "planck_pantheon_lcdm_baseline", "planck_rsd_lcdm_baseline"):
    Xall, X = chain(st); w = X.weight.values; h = X.H0.values / 100; ob = X.ombh2.values; olx = X.omegal.values; om = 1 - olx
    r = ob / h**2 / om / (3 / 16 * np.sqrt(olx))
    mob, sob = ms(ob, w); mH, sH = ms(X.H0.values, w); mOm, sOm = ms(om, w); mfr, sfr = ms(ob / h**2 / om, w); mr, sr = ms(r, w)
    P(f"E1 {st:30s} rows {len(Xall)} -> {len(X)}; ombh2 {mob:.6f} +/- {sob:.6f}; eta {273.9*mob:.3f} +/- {273.9*sob:.3f}; "
      f"H0 {mH:.2f} +/- {sH:.2f}; Om {mOm:.4f} +/- {sOm:.4f}; OL {1-mOm:.4f}; Ob/Om {mfr:.4f} +/- {sfr:.4f}; ratio {mr:.4f} +/- {sr:.4f} ({(mr-1)/sr:.2f} sigma)")
    if st == "iam_baryon_test":
        P(f"E2 18th chain: eta with 2.74e-8 (the record's factor) = {274.0*mob:.4f}e-10; with 273.9 = {273.9*mob:.4f}e-10; "
          f"no burn-in ombh2 = {ms(Xall.ombh2.values, Xall.weight.values)[0]:.6f}; sum of weights after burn-in {w.sum():.0f}")
        P(f"E3 18th chain ombh2 range in chain: {ob.min():.5f} - {ob.max():.5f} (prior 0.010-0.040)")
prog = pd.read_csv("Cosmological_Physics/mgcamb_validation/chains/iam_baryon_test.progress", sep=r"\s+", comment="#", names=["N", "time", "acc", "R1", "R1cl"])
P(f"E4 progress file: first {prog.time.iloc[0]}, last {prog.time.iloc[-1]}; last N {prog.N.iloc[-1]:.0f}; last R-1 {prog.R1.iloc[-1]:.6f}; min R-1 {prog.R1.min():.6f}")

P("\n== F. Comparators traced to the sources ==")
for nm, v, s in (("Planck 2018 TT,TE,EE+lowE+lensing (Table 2)", 0.02237, 0.00015), ("Planck 2018 +BAO (Table 2)", 0.02242, 0.00014)):
    P(f"F1 {nm}: Ob h^2 {v} +/- {s} -> eta {273.9*v:.3f} +/- {273.9*s:.3f}")
P("F2 Cyburt et al. 2016 Table IV (eta x 1e10): CMB-only 6.108 +/- 0.060; BBN+D 6.180 +/- 0.195; BBN+Yp+D 6.172 +/- 0.195; CMB+BBN 6.098 +/- 0.042; "
  "text: Planck (2015) 6.10 +/- 0.04. The value 6.137 +/- 0.017 printed as 'Observed (BBN)' is not in that source.")
P(f"F3 18th chain 6.113 against Planck TT,TE,EE+lowE+lensing 6.127: {100*(6.113/6.127-1):+.2f} %; against BBN+D 6.180: {100*(6.113/6.180-1):+.2f} % ({(6.113-6.180)/np.hypot(0.037,0.195):.2f} sigma)")
omh2 = 0.3153 * 0.6736**2; OLp = 0.6847
P(f"F4 Planck Om h^2 = 0.3153 x 0.6736^2 = {omh2:.4f}; (3/16) sqrt(OL) Om h^2 = {3/16*np.sqrt(OLp)*omh2:.5f} -> eta {273.9*3/16*np.sqrt(OLp)*omh2:.3f}; "
  f"(3/16) OL Om h^2 = {3/16*OLp*omh2:.5f} -> eta {273.9*3/16*OLp*omh2:.3f}")
P(f"F5 eta x 1e9 photons per baryon: 1/eta = {1/6.113e-10:.3e}; dark-to-baryon (0.265+0.685)/0.049 = {(0.265+0.685)/0.049:.2f}")

P("\n== G. QCD transition (T = 150 MeV, g* = 17.25) ==")
Tq = 0.150; Hq = 1.66 * np.sqrt(17.25) * Tq**2 / MP_GeV * GeV / hbar; lq = c / Hq
aq = (3.91 / 17.25)**(1 / 3) * 2.7255 * kB / GeV / Tq
Nq = np.pi * lq**2 / lP**2
P(f"G1 a_QCD = {aq:.3e}; H = {Hq:.3e} s^-1; l_H = {lq:.4e} m = {lq/3.0857e16:.2e} pc; horizon A/(4 l_P^2) = {Nq:.2e} nats = {Nq/np.log(2):.2e} bits")
Tew = 159.5; Hew = 1.66 * np.sqrt(106.75) * Tew**2 / MP_GeV * GeV / hbar
P(f"G2 a_EW (T = 100 GeV, g* = 106.75) = {(3.91/106.75)**(1/3)*2.7255*kB/GeV/100:.2e}; crossover 159.5 GeV: t = 1/(2H) = {1/(2*Hew):.2e} s")

open("docs/verification/scripts/verify_lambda_baryon_book_output.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
