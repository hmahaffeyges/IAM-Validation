"""Limber estimate of the change in the CMB lensing power C_L^phiphi from the informational term.

IAM: matter growth with mu(a) = H^2/(H^2 + beta_m E(a) H0^2), E(a) = exp(1 - 1/a), beta_m = Omega_m/2; Sigma = 1, so the lensing
potential follows the density field and the only change is the growth D(z). Both models start from the same early amplitude
(a_i = 1e-4). Linear power: Eisenstein & Hu (1998) no-wiggle transfer, n_s = 0.9649. Background: Planck 2018 (flat LCDM).
C_L^kappakappa (Limber) = integral dchi W_kappa(chi)^2/chi^2 P((L+1/2)/chi, z), W_kappa = (3/2) Om H0^2 (1+z) chi (chi*-chi)/chi*;
C_L^phiphi = 4 C_L^kappakappa / [L(L+1)]^2, so the ratio is the same; constants cancel.
Prints the fractional change 1 - C_L(IAM)/C_L(LCDM) for 30 <= L <= 1000.
"""
import json, pathlib, numpy as np
from scipy.integrate import solve_ivp, quad

ROOT = pathlib.Path(__file__).resolve().parents[3]
canon = json.load(open(ROOT / "CANON" / "iam_canon.json"))
h, Om, Ob_h2, ns = 0.6736, 0.3153, 0.02237, 0.9649
beta = Om / 2
Orad = 9.24e-5
OL = 1 - Om - Orad

def H2(a):                      # (H/H0)^2
    return Om / a**3 + Orad / a**4 + OL

def growth(iam):
    def rhs(x, y):
        a = np.exp(x); h2 = H2(a)
        dlnH = (-3 * Om / a**3 - 4 * Orad / a**4) / (2 * h2)
        mu = h2 / (h2 + beta * np.exp(1 - 1 / a)) if iam else 1.0
        D, Dp = y
        return [Dp, -(2 + dlnH) * Dp + 1.5 * (Om / a**3) / h2 * mu * D]
    a0 = 1e-4
    return solve_ivp(rhs, [np.log(a0), 0], [a0, a0], method="DOP853", rtol=1e-10, atol=1e-14, dense_output=True)

G = {m: growth(m) for m in (False, True)}
def D(a, iam): return G[iam].sol(np.log(a))[0]

def T_eh(k):                    # k in 1/Mpc; Eisenstein & Hu 1998 no-wiggle
    Omh2 = Om * h * h; fb = Ob_h2 / Omh2; th = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Ob_h2**0.75)
    aG = 1 - 0.328 * np.log(431 * Omh2) * fb + 0.38 * np.log(22.3 * Omh2) * fb**2
    Gam = Om * h * (aG + (1 - aG) / (1 + (0.43 * k * s)**4))
    q = k * th**2 / (Gam * h)
    L0 = np.log(2 * np.e + 1.8 * q); C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)

def P(k): return k**ns * T_eh(k)**2

c = 299792.458; H0 = 100 * h
def chi(z): return quad(lambda zz: c / (H0 * np.sqrt(H2(1 / (1 + zz)))), 0, z, limit=200)[0]
zstar = 1089.9; chis = chi(zstar)
zg = np.concatenate([np.linspace(1e-3, 10, 1200), np.geomspace(10.01, zstar * 0.999, 400)])
chig = np.array([chi(z) for z in zg])
ag = 1 / (1 + zg)
W = ((chis - chig) / chis)**2 * (1 + zg)**2 / np.sqrt(H2(ag))   # W_kappa^2/chi^2; dchi = c dz/H
Dl = np.array([D(a, False) for a in ag]); Di = np.array([D(a, True) for a in ag])

def CL(L, Dg):
    k = (L + 0.5) / chig
    return np.trapezoid(W * P(k) * Dg**2, zg)

Ls = [30, 50, 100, 200, 400, 700, 1000]
out = {}
for L in Ls:
    out[L] = 1 - CL(L, Di) / CL(L, Dl)
    print(f"L = {L:5d}: C_L^phiphi lower by {100 * out[L]:.3f} %")
band = [1 - CL(L, Di) / CL(L, Dl) for L in range(30, 1001, 10)]
print(f"30 <= L <= 1000: {100 * min(band):.3f} - {100 * max(band):.3f} %")
print(f"   full precision: {100 * min(band):.6f} - {100 * max(band):.6f} %")
print(f"sigma8 ratio D_IAM(1)/D_LCDM(1) = {D(1, True) / D(1, False):.5f}")

# Single-number summary over the Planck 2018 lensing range 8 <= L <= 400: the change in the lensing amplitude, weighting each L
# by its Fisher information for an amplitude, (2L+1) (a cosmic-variance weight; the amplitude estimator sum_L w_L dC/C / sum w_L).
Lb = np.arange(8, 401)
wb = 2 * Lb + 1
dfr = np.array([1 - CL(L, Di) / CL(L, Dl) for L in Lb])
print(f"Planck lensing range 8 <= L <= 400, (2L+1)-weighted: {100 * np.sum(wb * dfr) / np.sum(wb):.3f} %")
print(f"Planck lensing range 8 <= L <= 400, unweighted mean: {100 * dfr.mean():.3f} %")
