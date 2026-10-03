"""Checks for the coverage of three summary papers (Technical Reference for Physicists,
Test Validation Compendium, Official Score Card): every number used in the insertion blocks.
Run: python verify_summary_papers.py > verify_summary_papers_output.txt"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq

print("=== 1. Fixed constants")
Om, OL = 0.3153, 1 - 0.3153
bm = Om / 2
print(f"beta_m = Omega_m/2 = {bm:.5f}")
mu0 = 1 / (1 + bm)
print(f"mu(z=0) = {mu0:.6f}; 1-mu = {100*(1-mu0):.2f} %; mu0-1 = {mu0-1:.4f}")
print(f"Omega_m mu(0) = {Om*mu0:.4f}")
print(f"matter-sector H0 = 67.16 sqrt(1+beta_m) = {67.16*np.sqrt(1+bm):.2f}")

E = lambda a, c=1.0: np.exp(c * (1 - 1 / a))
H2 = lambda a: Om * a**-3 + OL          # LCDM background, H0 = 1

def mu(a, beta=bm, c=1.0):
    return H2(a) / (H2(a) + beta * E(a, c))

for z in (0, 0.2, 0.5, 1, 2, 10):
    a = 1 / (1 + z)
    print(f"  mu(z={z}) = {mu(a):.4f}")

print("=== 2. Growth with mu, LCDM background, same early amplitude")
def growth(beta, c=1.0):
    def rhs(lna, y):
        a = np.exp(lna)
        h2 = H2(a)
        dlnH = -1.5 * Om * a**-3 / h2
        d, dp = y
        return [dp, -(2 + dlnH) * dp + 1.5 * Om * a**-3 / h2 * mu(a, beta, c) * d]
    a0 = 1e-3
    s = solve_ivp(rhs, (np.log(a0), 0), [a0, a0], dense_output=True, rtol=1e-10, atol=1e-13)
    return s.sol

solL, solI = growth(0.0), growth(bm)
DL1, DI1 = solL(0)[0], solI(0)[0]
s8L = 0.811
print(f"D_IAM(1)/D_LCDM(1) = {DI1/DL1:.5f}  (growth factor change {100*(1-DI1/DL1):.2f} %)")
def fs8(sol, z, s8_0, D1):
    y = sol(np.log(1 / (1 + z)))
    return (y[1] / y[0]) * s8_0 * y[0] / D1
for z in (0, 0.3, 0.5, 1):
    r = fs8(solI, z, s8L * DI1 / DL1, DI1) / fs8(solL, z, s8L, DL1)
    print(f"  f sigma8 deficit z={z}: {100*(1-r):.2f} %")

print("=== 3. Free-coupling fit (3 H0 + 7 f sigma8)")
H0d = [("Planck", 67.40, 0.50, "photon"), ("SH0ES", 73.04, 1.04, "matter"), ("TRGB", 70.39, 1.94, "matter")]
fsd = np.array([[0.067, 0.423, 0.055], [0.150, 0.530, 0.160], [0.380, 0.497, 0.045], [0.510, 0.459, 0.038],
                [0.700, 0.473, 0.041], [0.850, 0.315, 0.095], [1.480, 0.462, 0.045]])
def chi2_parts(beta):
    hH = 0.0
    for n, v, s, sec in H0d:
        pred = 67.4 if sec == "photon" else 67.4 * np.sqrt(1 + beta)
        hH += ((v - pred) / s) ** 2
    sol = growth(beta); D1 = sol(0)[0]
    s8 = s8L * D1 / DL1
    pr = np.array([fs8(sol, z, s8, D1) for z in fsd[:, 0]])
    hG = np.sum(((fsd[:, 1] - pr) / fsd[:, 2]) ** 2)
    return hH, hG
for b in (0.0, bm):
    h, g = chi2_parts(b)
    print(f"  beta={b:.5f}: chi2_H0={h:.2f} chi2_fs8={g:.2f} total={h+g:.2f}")
tot = lambda b: sum(chi2_parts(b))
best = minimize_scalar(tot, bounds=(0, 0.4), method="bounded").x
cmin = tot(best)
lo1 = brentq(lambda b: tot(b) - cmin - 1, 0.0, best); hi1 = brentq(lambda b: tot(b) - cmin - 1, best, 0.4)
lo2 = brentq(lambda b: tot(b) - cmin - 4, 0.0, best); hi2 = brentq(lambda b: tot(b) - cmin - 4, best, 0.45)
hb, gb = chi2_parts(best)
print(f"  best beta = {best:.4f} (+{hi1-best:.4f} -{best-lo1:.4f} at 68 %; +{hi2-best:.4f} -{best-lo2:.4f} at 95 %)")
print(f"  at best: chi2_H0={hb:.2f} chi2_fs8={gb:.2f} total={cmin:.2f}")
h0, g0 = chi2_parts(0.0); hf, gf = chi2_parts(bm)
print(f"  Delta chi2 (beta=0 minus best) = {h0+g0-cmin:.2f}: H0 part {h0-hb:.2f}, fs8 part {g0-gb:+.2f}")
print(f"  Delta chi2 (beta=0 minus beta_m fixed) = {h0+g0-hf-gf:.2f}: H0 part {h0-hf:.2f}, fs8 part {g0-gf:+.2f}")
# fs8 alone
gonly = lambda b: chi2_parts(b)[1]
bg = minimize_scalar(gonly, bounds=(-0.3, 0.8), method="bounded").x
gm = gonly(bg)
print(f"  f sigma8 alone: best beta {bg:.3f}; chi2 {gm:.2f}; Delta chi2 at 0 = {gonly(0)-gm:.2f}, at beta_m = {gonly(bm)-gm:.2f}")
try:
    l = brentq(lambda b: gonly(b) - gm - 1, -0.3, bg); u = brentq(lambda b: gonly(b) - gm - 1, bg, 0.8)
    print(f"  f sigma8 alone 68 %: {l:.3f} to {u:.3f}")
except ValueError as e:
    print("  f sigma8 alone 68 %: bracket failed", e)
# local H0 restated
w = np.array([1/1.04**2, 1/1.94**2]); v = np.array([73.04, 70.39])
Hloc = np.sum(w*v)/np.sum(w); sH = 1/np.sqrt(np.sum(w))
print(f"  weighted local H0 = {Hloc:.2f} +- {sH:.2f}; (Hloc/67.4)^2-1 = {(Hloc/67.4)**2-1:.4f}; zero-tension beta for SH0ES alone {(73.04/67.4)**2-1:.4f}")
print(f"  beta_m gives local H0 {67.4*np.sqrt(1+bm):.2f}: SH0ES {(73.04-67.4*np.sqrt(1+bm))/1.04:.2f} sigma")
print(f"  Planck-SH0ES difference {(73.04-67.4)/np.hypot(0.5,1.04):.2f} sigma")
n, k = 10, 2
dAIC = (h0+g0) - (cmin + 2*k); dBIC = (h0+g0) - (cmin + k*np.log(n))
print(f"  AIC: Delta = {dAIC:.2f}; BIC (n=10, k=2): Delta = {dBIC:.2f}; exp(-dAIC/2) = {np.exp(-dAIC/2):.2e}")
dAICf = (h0+g0) - (hf+gf)   # k = 0 when beta is fixed
print(f"  beta fixed at Omega_m/2 (k=0): Delta chi2 = Delta AIC = Delta BIC = {dAICf:.2f}")

print("=== 4. Exponent of the activation function and mu0")
for c in (0.8, 1.0, 1.2):
    sol = growth(bm, c); r = sol(0)[0] / DL1
    print(f"  c={c}: mu(z=0) = {mu(1.0, bm, c):.6f}; sigma8 = {s8L*r:.4f}")
s8s = [s8L * growth(bm, c)(0)[0] / DL1 for c in np.linspace(0.8, 1.2, 9)]
print(f"  sigma8 spread over c in [0.8,1.2]: {max(s8s)-min(s8s):.4f} ({100*(max(s8s)-min(s8s))/np.mean(s8s):.2f} %)")

print("=== 5. Equation of state of the record term, future")
for a in (1, 2, 5, 100):
    print(f"  a={a}: w_info = {-1-1/(3*a):.4f}; E = {E(a):.4f}")
print(f"  E(a->inf) = e = {np.e:.6f}")
for z in (0, 1):
    a = 1/(1+z); bE = bm*E(a)
    weff = (-OL + (-1-1/(3*a))*bE)/(OL+bE)
    print(f"  w_eff(z={z}) = {weff:.4f}")

print("=== 6. Horizon temperature and Hubble rate")
print(f"  T_H/H = 1/(2 pi) = {1/(2*np.pi):.6f} (hbar = k_B = 1)")
print(f"  H_m^2(1)/H0^2 = 1 + beta_m = {1+bm:.5f}")

print("=== 7. Why qubits are cold")
h, kB = 6.62607015e-34, 1.380649e-23
for f, T in ((5e9, 0.015), (5e9, 0.050)):
    x = h*f/(kB*T)
    print(f"  f={f/1e9:.0f} GHz, T={1e3*T:.0f} mK: hf/kT = {x:.1f}; thermal excited population exp(-hf/kT) = {np.exp(-x):.1e}")
Delta = 182e-6*1.602176634e-19
print(f"  Delta_Al/k_B = {Delta/kB:.3f} K; BCS T_c = Delta/(1.764 k_B) = {Delta/(1.764*kB):.2f} K")
for T in (0.015, 0.1, 0.2):
    # thermal quasiparticle density x_qp ~ sqrt(2 pi kT/Delta) exp(-Delta/kT)
    x = np.sqrt(2*np.pi*kB*T/Delta)*np.exp(-Delta/(kB*T))
    print(f"  thermal x_qp at {1e3*T:.0f} mK = {x:.1e}")
print(f"  Landauer k_B T ln2 at 15 mK vs 300 K: ratio {0.015/300:.1e}; per bit at 15 mK = {kB*0.015*np.log(2):.2e} J")

print("=== 8. Vacuum ratio coefficient")
OLp = 0.6847
print(f"  3 Omega_L/(8 pi) = {3*OLp/(8*np.pi):.4f}")
print("=== 9. Collapsed fraction check")
print(f"  Omega_m f_coll = 0.315 x 0.5927 = {0.315*0.5927:.4f}; 1/(2 f_coll) = {1/(2*0.5927):.3f}")

print("=== 10. Reproduction of the summary papers' own arithmetic (background-diluted form, TRGB +-1.89)")
Omc, Orc = 0.315, 9.24e-5; OLc = 1 - Omc - Orc
def chi2H(Hm, sT=1.89):
    return ((67.40-67.4)/0.5)**2 + ((73.04-Hm)/1.04)**2 + ((70.39-Hm)/sT)**2
print(f"  chi2_H0 LCDM = {chi2H(67.4):.2f}; with 72.5 = {chi2H(72.5):.2f}; 67.4 sqrt(1.157) = {67.4*np.sqrt(1.157):.2f}")
def gback(beta, s8):
    Omeff = lambda a: Omc*a**-3/(Omc*a**-3+Orc*a**-4+OLc+beta*E(a))
    rhs = lambda l, y: [y[1], -(2-1.5*Omeff(np.exp(l)))*y[1]+1.5*Omeff(np.exp(l))*y[0]]
    s = solve_ivp(rhs, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-8).sol
    D1 = s(0)[0]; pr = []
    for z in fsd[:, 0]:
        y = s(np.log(1/(1+z))); pr.append(y[1]/y[0]*s8*y[0]/D1)
    return np.sum(((fsd[:, 1]-np.array(pr))/fsd[:, 2])**2)
gl, gi = gback(0, 0.811), gback(0.157, 0.800)
print(f"  growth chi2 (their form): LCDM {gl:.2f}, beta 0.157 {gi:.2f}, difference {gi-gl:.2f}")
tl, ti = chi2H(67.4)+gl, chi2H(72.5)+gi
print(f"  totals {tl:.2f} vs {ti:.2f}; Delta {tl-ti:.2f}; sqrt {np.sqrt(tl-ti):.2f}")
print(f"  AIC Delta {tl-(ti+4):.2f}; BIC Delta {tl-(ti+2*np.log(10)):.2f}; exp(-dAIC/2) {np.exp(-(tl-ti-4)/2):.2e}")
print(f"  Omega_m(a=1; beta=0.157) = {Omc/(Omc+Orc+OLc+0.157):.4f}")
print(f"  photon coupling ratio: 0.0039/beta_m = {0.0039/bm:.4f} -> photons couple >= {bm/0.0039:.0f} x more weakly")

print("=== 11. Chain sector values against the local ladder")
Hm = 67.16*np.sqrt(1+bm)
print(f"  67.16 sqrt(1+beta_m) = {Hm:.2f}; SH0ES {(73.04-Hm)/1.04:.2f} sigma; TRGB {(70.39-Hm)/1.94:.2f} sigma")
print(f"  chi2_H0 with chain values (67.16 photon vs Planck 67.40+-0.50; 72.26 local) = {((67.40-67.16)/0.5)**2+((73.04-Hm)/1.04)**2+((70.39-Hm)/1.94)**2:.2f}")
