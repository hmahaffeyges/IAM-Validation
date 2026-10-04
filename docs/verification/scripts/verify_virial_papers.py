#!/usr/bin/env python3
"""Recomputes every equation step and number carried into the book from the seven papers of the virial and gravitational-decoherence group:
The Virial Partition from Atoms to the Horizon; The Thermodynamic Identity Governing the Virial Theorem (PRL version); Virial Efficiency and
Effective Nonlinear Exponent; The Virial Partition Across Wide Range of Physical Scales; Dark Matter and Dark Energy as Virial Partners;
Gravitational Decoherence, the Virial Partition and the Emergence of Classical Structure; Gravitational Decoherence from Dual-Sector
Thermodynamics (quantum level).
Chapters: part1/p1_03_virial_law.tex, part1/p1_04_virial_identity.tex, part2/p2_02_virial.tex, part2/p2_02b_virial_tests.tex,
part5/p5_05_gravdec.tex, part5/p5_05b_virial_partners.tex, part5/p5_05c_virial_decoherence.tex.
CODATA 2018 via scipy.constants. Planck 2018: Omega_m = 0.3153, H0 = 67.36; Level 2 chains from Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv.
Run from the repository root:  python docs/verification/scripts/verify_virial_papers.py > docs/verification/scripts/verify_virial_papers_output.txt
"""
import csv, pathlib
import numpy as np, sympy as sp, scipy.constants as C
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

REPO = pathlib.Path(__file__).resolve().parents[3]
G, c, hbar, k = C.G, C.c, C.hbar, C.k
Msun, Rsun, Lsun, yr, Mpc = 1.98847e30, 6.957e8, 3.828e26, 3.15576e7, 3.0856775814913673e22
Om, H0P = 0.3153, 67.36
OL = 1 - Om
bm = Om / 2
ln2 = np.log(2)

def hdr(s): print("\n" + s)

# ------------------------------------------------------------------------------------------------------------------
hdr("A. The theorem (sympy)")
lam, x, y, z_ = sp.symbols("lambda x y z", positive=True)
kdeg = sp.symbols("k")
V = (x**2 + y**2 + z_**2) ** sp.Rational(-1, 2)              # 1/r, homogeneous of degree -1
euler = sp.simplify(x*sp.diff(V, x) + y*sp.diff(V, y) + z_*sp.diff(V, z_) + V)
print("   Euler for V = 1/r: r.grad V - (-1) V =", euler, " -> 2<K> = -<V>, <K> = |<V>|/2, E = -<K>")
n_ = sp.symbols("n", positive=True)
Vn = -(x**2 + y**2 + z_**2) ** (-n_/2)
deg = sp.simplify((x*sp.diff(Vn, x) + y*sp.diff(Vn, y) + z_*sp.diff(Vn, z_)) / Vn)
print("   V = -1/r^n has degree", deg, "-> 2<T> = (-n)<V> = n|<V>|, <T> = (n/2)<|V|>")
print("   linear potential (k = +1): 2<K> = +<V>: <K> = <V>/2 for confinement; quadratic (k = 2): <K> = <V> (equipartition is not the 1/r half)")

hdr("B. Atoms and molecules")
Eh = C.physical_constants["Hartree energy in eV"][0]
print(f"   hydrogen: E1 = {-Eh/2:.4f} eV, <K> = {Eh/2:.4f} eV, <V> = {-Eh:.4f} eV, <K>/|<V>| = 0.5000, -<K>/E = 1.0000")
for nm, E in (("H", -0.5), ("Ne", -128.547), ("Xe", -7232.138)):
    print(f"   {nm}: E = {E} Eh, <T> = {-E} Eh (virial), -<T>/E = {(-E)/(-E):.6f}, <T>/|<V>| = {(-E)/abs(2*E):.6f}")
print("   (the virial ratio printed as 'T/|V| = 1.0000' is -T/E = 2T/|V| = 1; T/|V| = 1/2)")
a0 = C.physical_constants["Bohr radius"][0]
RH = c / (H0P * 1e3 / Mpc)
print(f"   span: Bohr radius {a0:.3e} m to the Hubble radius c/H0 = {RH:.3e} m: {np.log10(RH/a0):.2f} orders; atom (1e-10 m) to cluster (1e23 m): 33 orders")

hdr("C. Stars")
U = -1.5 * G * Msun**2 / Rsun
print(f"   Sun, n = 3 polytrope: U = {U:.3e} J; |U|/2 = {abs(U)/2:.3e} J; t_KH = |U|/2L = {abs(U)/2/Lsun/yr/1e6:.1f} Myr (age 4,568 Myr)")
Uu = -0.6 * G * Msun**2 / Rsun
print(f"   uniform sphere (U = -3GM^2/5R): t = {abs(Uu)/2/Lsun/yr/1e6:.1f} Myr")
mu_ = C.physical_constants["atomic mass constant"][0]
for mue in (2.0, 2.15):
    M = 2.01824*np.sqrt(3*np.pi)/2*(hbar*c/G)**1.5/(mue*mu_)**2
    print(f"   Chandrasekhar, mu_e = {mue}: {M/Msun:.3f} Msun")

hdr("D. Simulated halos: published 2T/|U| as K/|W|")
for lo, hi, nm in ((1.1, 1.3, "within r_vir (Bett, Neto, Power)"), (1.02, 1.17, "with surface pressure (Klypin 2016)")):
    print(f"   2T/|U| = {lo}-{hi} {nm}: K/|W| = {lo/2:.3f}-{hi/2:.3f}; reciprocal |U|/2T = {1/hi:.2f}-{1/lo:.2f}")
print(f"   Neto relaxation cut 2T/|U| < 1.35 -> |U|/2T > {1/1.35:.3f}")
print(f"   eta_vir = 1/(2 f_coll) with f_coll = 0.62: {1/(2*0.62):.3f} (a definition: f_coll eta_vir = 1/2)")
print(f"   Omega_m f_coll eta_vir = 0.315 x 0.62 x 0.815 = {0.315*0.62*0.815:.4f} (product of a definition, not a measurement)")
rows = list(csv.DictReader(open(REPO/"docs/verification/virial/NBODY_TRACE_massfunction_slopes.csv")))
print("   d ln F(>M)/d ln D = f/F (NBODY_TRACE_massfunction_slopes.csv, Planck 2018, z = 0):")
for m in ("1e+10", "1e+12", "1e+14"):
    vals = [f"{r['mf']} {float(r['dlnF_dlnD']):.2f}" for r in rows if abs(np.log10(float(r["M_hinv_Msun"])) - np.log10(float(m))) < 1e-6]
    print(f"     M = {m}: " + ", ".join(vals))

hdr("E. Black-hole horizon")
for m in (1.0, 4.3e6, 6.5e9):
    M = m*Msun; T = hbar*c**3/(8*np.pi*G*M*k); S = k*4*np.pi*G*M**2/(hbar*c); N = S/(k*ln2)
    print(f"   M = {m:.3g} Msun: T_H = {T:.3e} K, N = {N:.3e} bits, N k T ln2/(M c^2) = {N*k*T*ln2/(M*c**2):.10f}")
Ms, cs_, Gs, hs = sp.symbols("M c G hbar", positive=True)
TH = hs*cs_**3/(8*sp.pi*Gs*Ms); SB = 4*sp.pi*Gs*Ms**2/(hs*cs_)      # k = 1
print("   sympy: T_H S / (M c^2) =", sp.simplify(TH*SB/(Ms*cs_**2)))
for chi in (0.5, 0.9, 0.998):
    print(f"   Kerr chi = {chi}: T_H S/Mc^2 = sqrt(1-chi^2)/2 = {np.sqrt(1-chi**2)/2:.4f}")
print(f"   (ln 2/2) M c^2 = {ln2/2:.4f} M c^2 counts ln 2 twice; with N = S/(k ln 2) and k T ln 2 per bit the cost is T S = M c^2/2")
P = hs*cs_**6/(15360*sp.pi*Gs**2*Ms**2)
print("   Hawking power / (k T_H ln 2) =", sp.simplify(P/(TH*sp.log(2))), " (bits per second; = c^3/(1920 G M ln 2))")
t_ev = 5120*np.pi*G**2*Msun**3/(hbar*c**4)
print(f"   evaporation time of 1 Msun: {t_ev/yr:.2e} yr")

hdr("F. The thermodynamic identity (sympy)")
K_, Vv, T_ = sp.symbols("K V T", real=True)
Q = -(K_ + Vv)                      # first law, E_i = 0
dS = Q/T_                           # bound met
EL = T_*dS
print("   E_L = T dS =", sp.simplify(EL), "= Q;  with 2K + V = 0:", sp.simplify(Q.subs(Vv, -2*K_)), "= K;  |V|/2 =", sp.simplify((2*K_)/2))

hdr("G. The cosmic coupling and its functions")
E = lambda a: np.exp(1 - 1/a)
H2 = lambda a: Om*a**-3 + OL
mu = lambda a: H2(a)/(H2(a) + bm*E(a))
print(f"   beta_m = Omega_m/2 = {bm:.5f}; Omega_m = 0.3166 (Level 2 posterior) -> {0.3166/2:.5f}")
print(f"   three-channel: 0.076618 + 0.078825 + 0.002207 = {0.076618+0.078825+0.002207:.5f}; geometric = beta_m/2 = {bm/2:.6f}; shares "
      f"{100*0.076618/bm:.1f} / {100*0.078825/bm:.1f} / {100*0.002207/bm:.1f} %")
asym = sp.symbols("a", positive=True)
Es = sp.exp(1 - 1/asym)
print("   E(a) = e^{-z}:", sp.simplify(Es.subs(asym, 1/(1+sp.Symbol('z')))), "; dE/da =", sp.simplify(sp.diff(Es, asym)),
      "; E(inf) =", sp.limit(Es, asym, sp.oo), "; inflection a =", sp.solve(sp.diff(Es, asym, 2), asym))
print("   dE/dln a = E/a peaks at a =", sp.solve(sp.diff(Es/asym, asym), asym))
print(f"   E(z=10) = {np.exp(-10):.2e}; E reaches 10/50/90 % at z = {-np.log(0.1):.2f}/{-np.log(0.5):.2f}/{-np.log(0.9):.2f}")
print("   z      E(a)    mu      1-mu    R(a)    record share of (Omega_L + beta_m E)   d(beta_m E)/da / |d(Omega_m a^-3)/da|")
for z in (0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
    a = 1/(1+z)
    print(f"   {z:<5}  {E(a):.3f}   {mu(a):.4f}  {100*(1-mu(a)):5.2f} %  {Om*a**-3/(bm*E(a)):7.1f}   {100*bm*E(a)/(OL+bm*E(a)):5.1f} %   "
          f"{100*E(a)*a**2/6:5.2f} %")
print(f"   R(1) = Omega_m/(beta_m E(1)) = {Om/bm:.4f} (by construction); beta_m/Omega_L = {100*bm/OL:.1f} % (not the share)")
print(f"   Omega_DM/Omega_L today (0.26/0.69) = {0.26/0.69:.2f}")
zeq_l = brentq(lambda z: Om*(1+z)**3 - OL, 0, 2)
zeq_i = brentq(lambda z: Om*(1+z)**3 - OL - bm*np.exp(-z), 0, 2)
print(f"   matter = vacuum (+record) at z = {zeq_l:.3f} (LCDM) and {zeq_i:.3f} (matter-sector rate)")
q = lambda z: 0.5*Om*(1+z)**3/(Om*(1+z)**3+OL) - OL/(Om*(1+z)**3+OL)
print(f"   deceleration-acceleration transition with the background unmodified: z_t = {brentq(q, 0, 2):.3f}")
print(f"   mu0 = mu(1) - 1 = {mu(1)-1:.4f}; mu(z=0.3) = {mu(1/1.3):.4f}; mu(0.5) = {mu(1/1.5):.4f}")
for z in (0.4, 0.5):
    print(f"   Omega_m mu(z={z}) = {Om*mu(1/(1+z)):.4f}")
print(f"   DESI DR1 FS+BAO 0.2962 +/- 0.0095 vs 0.2990: {(0.2990-0.2962)/0.0095:.2f} sigma (not a growth-only measurement)")
print(f"   1 - 0.2962/0.3153 = {100*(1-0.2962/0.3153):.1f} %")

hdr("H. Growth (linear, Planck 2018 background, same early amplitude, Level 1 mu*G form)")
def grow(m):
    def r(l, yv):
        a = np.exp(l); return [yv[1], -(2 - 1.5*Om*a**-3/H2(a))*yv[1] + 1.5*(Om*a**-3/H2(a))*m(a)*yv[0]]
    return solve_ivp(r, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
Lg, Ig = grow(lambda a: 1.0), grow(mu)
D = lambda S, a: S.sol(np.log(a))[0]; f = lambda S, a: S.sol(np.log(a))[1]/S.sol(np.log(a))[0]
for z in (0.0, 0.295, 0.3, 0.5, 1.0, 1.491):
    a = 1/(1+z); r = f(Ig, a)*D(Ig, a)/(f(Lg, a)*D(Lg, a))
    print(f"   z = {z:5.3f}: 1-mu = {100*(1-mu(a)):5.2f} %   f sigma8 deficit = {100*(1-r):5.2f} %   E_G change f_L/f_I - 1 = {100*(f(Lg,a)/f(Ig,a)-1):+5.2f} %")

hdr("I. Chains (Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv)")
ch = {r["chain"]: r for r in csv.DictReader(open(REPO/"Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv"))}
print(f"   number of chains in the record: {len(ch)}  (levels: " + ", ".join(sorted({r['level'] for r in ch.values()})) + ")")
for nm in ("iam_level2_runA", "iam_level2_runD", "iam_level2_runC_lcdm", "iam_l2b_runA", "iam_l2b_runD"):
    r = ch[nm]
    print(f"   {nm:22s} sigma8 {float(r['sigma8']):.4f}+/-{float(r['sigma8_sd']):.4f}  H0 {float(r['H0']):.2f}+/-{float(r['H0_sd']):.2f}  "
          f"Om {float(r['omegam']):.4f}+/-{float(r['omegam_sd']):.4f}  S8 {float(r['S8']):.3f}+/-{float(r['S8_sd']):.3f}  chi2min {float(r['chi2_min']):.3f}  R-1 {r['R-1_final(progress)']}")
dchi = float(ch["iam_level2_runA"]["chi2_min"]) - float(ch["iam_level2_runC_lcdm"]["chi2_min"])
print(f"   Level 2 Delta chi2 (IAM Run A - LCDM Run C) = {dchi:+.2f}  (IAM higher: consistent with Planck)")
print(f"   L1 fixed vs LCDM: Delta chi2 = {float(ch['iam_fixed_mu0 (r2 final)']['chi2_min'])-float(ch['lcdm_baseline']['chi2_min']):+.2f}; "
      f"sigma8 {float(ch['lcdm_baseline']['sigma8']):.4f} -> {float(ch['iam_fixed_mu0 (r2 final)']['sigma8']):.4f}")
OmA, OmAsd = float(ch["iam_level2_runA"]["omegam"]), float(ch["iam_level2_runA"]["omegam_sd"])
print(f"   beta_m (fixed 0.15765) / Omega_m(Run A) = {0.15765/OmA:.3f} +/- {0.15765*OmAsd/OmA**2:.3f}; Omega_m(Run A)/2 = {OmA/2:.4f}")
s8A, s8Asd = float(ch["iam_level2_runA"]["sigma8"]), float(ch["iam_level2_runA"]["sigma8_sd"])
s8C, s8Csd = float(ch["iam_level2_runC_lcdm"]["sigma8"]), float(ch["iam_level2_runC_lcdm"]["sigma8_sd"])
print(f"   sigma8 vs joint KiDS-Legacy+DES Y3+DESI+Pantheon+ 0.802 (+0.022 -0.018): IAM {(0.802-s8A)/np.hypot(0.018, s8Asd):.2f} sigma, "
      f"LCDM (L2) {(s8C-0.802)/np.hypot(0.022, s8Csd):.2f} sigma, LCDM (Planck 0.8111) {(0.8111-0.802)/0.022:.2f} sigma")
S8A, S8Asd = float(ch["iam_level2_runA"]["S8"]), float(ch["iam_level2_runA"]["S8_sd"])
print(f"   S8 Level 2 {S8A:.3f} vs KiDS-Legacy 0.815 (+0.016 -0.021): {(S8A-0.815)/np.hypot(0.016, S8Asd):.2f} sigma")

hdr("J. Two expansion rates and the H0 census (worldline rule: supernova ladder and masers on the matter ruler; CMB and lensing time delays on photon paths)")
Hg = float(ch["iam_level2_runA"]["H0"]); Hgsd = float(ch["iam_level2_runA"]["H0_sd"])
Hm = Hg*np.sqrt(1+bm)
print(f"   H0_matter = {Hg:.2f} x sqrt(1 + {bm}) = {Hm:.2f}; with Planck 67.36: {67.36*np.sqrt(1+bm):.2f}; sqrt(1+beta_m) - 1 = {100*(np.sqrt(1+bm)-1):.2f} %")
data = [("Planck 2018 CMB", 67.36, 0.54, "photon"), ("ACT DR4 + WMAP", 67.6, 1.1, "photon"), ("SH0ES Cepheids", 73.04, 1.04, "matter"),
        ("H0LiCOW time delays", 73.3, 1.8, "photon"), ("Megamasers (MCP)", 73.9, 3.0, "matter"), ("SBF", 73.3, 2.5, "matter")]
chi_i = 0
for nm, h, s, sec in data:
    pred = Hg if sec == "photon" else Hm
    chi_i += ((h-pred)/s)**2
    print(f"   {nm:22s} {h:6.2f} +/- {s:4.2f}  {sec:6s}  vs 67.16: {(h-Hg)/s:+5.1f} sigma   vs 72.26: {(h-Hm)/s:+5.1f} sigma   vs own sector: {(h-pred)/s:+5.1f}")
print(f"   chi2 with each probe against its own sector (6 probes): {chi_i:.2f}; time delays on the photon ruler contribute {((73.3-Hg)/1.8)**2:.2f}")
w = np.array([1/s**2 for _, _, s, _ in data]); hv = np.array([h for _, h, _, _ in data])
hbest = (w*hv).sum()/w.sum()
print(f"   one H0 fitted to all six: {hbest:.2f}, chi2 = {(w*(hv-hbest)**2).sum():.1f}; held at 67.36: chi2 = {(w*(hv-67.36)**2).sum():.1f}")
print(f"   SH0ES vs 72.26: {(73.04-72.26)/1.04:.2f} sigma; Planck vs 67.16 with Planck's error: {(67.36-67.16)/0.54:.2f} sigma")

hdr("K. Gravitational decoherence, virial paper: checks of the equations carried as conjecture")
T0 = hbar*H0P*1e3/Mpc/(2*np.pi*k)
print(f"   Gibbons-Hawking temperature today: {T0:.3e} K")
Mm, sig, Gg, rho = sp.symbols("M sigma G rho", positive=True)
R = Gg*Mm/sig**2; rh = 3*Mm/(4*sp.pi*R**3)
print("   t_dyn = 1/sqrt(G rho) with sigma^2 = GM/R:", sp.simplify(1/sp.sqrt(Gg*rh)), " (prefactor sqrt(4 pi/3) = 2.05; printed sqrt(pi/6) = 0.72)")
Hn = H0P*1e3/Mpc
print(f"   sigma_crit from M_min = 4 Omega_m sigma^3/(G H) at 10^8.4 Msun: {(G*Hn*10**8.4*Msun/(4*Om))**(1/3)/1e3:.2f} km/s")

hdr("L. Gravitational decoherence, quantum level")
rho_s, T_ = 2200.0, 0.010
def EG(m): Rr = (3*m/(4*np.pi*rho_s))**(1/3); return G*m**2/Rr, Rr
def tauI(m, T): return hbar*(k*T)**2*ln2/EG(m)[0]**3
def tauD(m): return hbar/EG(m)[0]
Eg, Rr = EG(1e-12)
print(f"   m = 1e-12 kg, rho = 2200: R = {Rr*1e6:.2f} um, E_G = {Eg:.3e} J, tau_PD = {tauD(1e-12)*1e6:.2f} us, tau_IAM(10 mK) = {tauI(1e-12, T_):.1f} s, ratio {tauI(1e-12,T_)/tauD(1e-12):.2e}")
print(f"   T^2: tau(20)/tau(10) = {tauI(1e-12,0.02)/tauI(1e-12,0.01):.1f}, tau(40)/tau(10) = {tauI(1e-12,0.04)/tauI(1e-12,0.01):.1f}")
sl = lambda fn: (np.log(fn(2e-12)) - np.log(fn(1e-12)))/np.log(2)
print(f"   slopes: tau_IAM ~ m^{sl(lambda m: tauI(m, T_)):.3f}, tau_PD ~ m^{sl(tauD):.3f}; difference {sl(tauD)-sl(lambda m: tauI(m,T_)):.3f}")
mx = brentq(lambda lm: np.log(tauI(10**lm, T_)/tauD(10**lm)), -14, -8)
print(f"   crossover tau_IAM = tau_PD at 10 mK: {10**mx:.2e} kg")
for T in (0.001, 0.01, 0.1, 1.0, 300.0):
    print(f"     T = {T:7.3f} K: tau_IAM(1e-15 kg) = {tauI(1e-15,T):.2e} s, tau_IAM(1e-12) = {tauI(1e-12,T):.2e} s, tau_IAM(1e-10) = {tauI(1e-10,T):.2e} s")
for tt in (1.0, 1e-3):
    mm = brentq(lambda lm: np.log(tauI(10**lm, T_)/tt), -16, -6)
    print(f"   tau_IAM = {tt} s at 10 mK for m = {10**mm:.2e} kg")
amu = C.physical_constants["atomic mass constant"][0]
print(f"   1e13-1e14 amu = {1e13*amu:.2e}-{1e14*amu:.2e} kg: tau_IAM(10 mK) = {tauI(1e13*amu,T_):.2e}-{tauI(1e14*amu,T_):.2e} s")
EGs, Ts, hb, kk, tt = sp.symbols("E_G T hbar k_B t", positive=True)
Gamma = EGs**2/(hb*kk*Ts*sp.log(2)); Sb = kk*Ts/EGs
lnE = sp.integrate(Gamma/Sb, (tt, 0, tt))
print("   integral of Gamma_info/S_boundary dt =", sp.simplify(lnE), " -> E_q = exp(t/tau): exponential, constant integrand")
tau_s = hb*kk**2*Ts**2*sp.log(2)/EGs**3
print("   tau_IAM units check: [hbar k^2 T^2/E^3] -> J s J^2 / J^3 = s;  1/tau =", sp.simplify(1/tau_s))
eta = sp.symbols("eta", positive=True)
Eq = sp.exp(1 - 1/eta); rate = sp.diff(Eq, eta)
print("   ramp E_q = exp(1-1/eta): rate dE_q/deta =", sp.simplify(rate), "; peak at eta =", sp.solve(sp.diff(rate, eta), eta), "; slope at 0+ =", sp.limit(rate, eta, 0, "+"))
Pw = kk*Ts*sp.log(2)/tau_s
print("   k_B T ln 2 / tau_IAM =", sp.simplify(Pw), " (= printed P_IAM);  bit rate x k_B T ln 2 =", sp.simplify(Gamma*kk*Ts*sp.log(2)))
w0 = 2*np.pi*1e5
PI = Eg**3/(hbar*k*T_)
print(f"   P_IAM(1e-12 kg, 10 mK) = {PI:.2e} W; dn/dt = P/(hbar omega0) at 100 kHz = {PI/(hbar*w0):.2f} phonons/s; E_G^2/hbar = {Eg**2/hbar:.2e} W")
for T in (0.01, 0.02, 0.04, 0.3):
    print(f"     T = {T} K: dn/dt(1e-12 kg) = {Eg**3/(hbar*k*T)/(hbar*w0):.3f} /s")

xs = np.log(np.array([1, 2.15, 4.64, 10.0])); sig_a = 0.1/np.sqrt(((xs-xs.mean())**2).sum())
print(f"   four masses equally spaced in ln m over one decade, 10 % on tau: sigma(alpha) = {sig_a:.3f}; Delta alpha/sigma = {(10/3)/sig_a:.0f}")
print(f"   three temperatures 10/40 mK, 20 % on each tau: ln16 / (0.2 sqrt 2) = {np.log(16)/(0.2*np.sqrt(2)):.1f} sigma against constant tau")
mm1 = brentq(lambda lm: np.log(Eg**3*(10**lm/1e-12)**5/(hbar*k*T_)/(hbar*w0)), -14, -10)
print(f"   dn/dt = 1 phonon/s at 10 mK for m = {10**mm1:.2e} kg")
print(f"   timeline: from 1e-19 kg to 3.48e-12 kg = {np.log10(3.48e-12/1e-19):.2f} decades; at 3-5 yr per decade: {3*np.log10(3.48e-12/1e-19):.0f}-{5*np.log10(3.48e-12/1e-19):.0f} yr")

hdr("M. Lindblad dephasing with L = x (dimensionless, x = (a + a^dag)/sqrt 2), H neglected, coherent |alpha = 2>")
N = 40
a_ = np.diag(np.sqrt(np.arange(1, N)), 1); X = (a_ + a_.T)/np.sqrt(2); nop = a_.T @ a_
from math import factorial
al = 2.0
psi = np.array([np.exp(-al**2/2)*al**n/np.sqrt(float(factorial(n))) for n in range(N)]); rho0 = np.outer(psi, psi).astype(complex)
def D_(r): return X@r@X - 0.5*(X@X@r + r@X@X)
def evolve(gam, tmax=5.0, dt=1e-3):
    r = rho0.copy(); out = []; t = 0.0
    while t <= tmax + 1e-12:
        out.append((t, np.real(np.trace(r@r)), np.real(np.trace(nop@r)), abs(r[0, 4])/abs(rho0[0, 4])))
        k1 = gam(t)*D_(r); k2 = gam(t+dt/2)*D_(r+dt/2*k1); k3 = gam(t+dt/2)*D_(r+dt/2*k2); k4 = gam(t+dt)*D_(r+dt*k3)
        r = r + dt/6*(k1+2*k2+2*k3+k4); t += dt
    return np.array(out)
gI = lambda t: 0.0 if t <= 1e-9 else np.exp(1-1/t)/t**2
std = evolve(lambda t: 1.0); ramp = evolve(gI)
dP = ramp[:, 1] - std[:, 1]; i = np.argmax(dP)
print(f"   purity difference P_ramp - P_const peaks at eta = {ramp[i,0]:.3f}, Delta P = {dP[i]:.3f}")
for tq in (0.5, 1.0, 2.0, 5.0):
    j = int(round(tq/1e-3))
    print(f"   eta = {tq}: purity ramp {ramp[j,1]:.3f} const {std[j,1]:.3f};  <n> ramp {ramp[j,2]:.2f} const {std[j,2]:.2f} (initial {std[0,2]:.2f})")
print("   <n> grows: d<n>/dt = Gamma/2 for L = x (momentum diffusion heats); position dephasing does not conserve phonon number")
# (state histories are plotted by docs/book/figscripts/fig_virial_book.py)
