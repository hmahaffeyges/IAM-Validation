"""Shared cosmology for the Part 1, 2 and 5 figures. Same equations as docs/verification/scripts/verify_s8_trend.py and
verify_sector_tension.py: LambdaCDM background (Planck 2018, Om 0.3153, OL 0.6847), beta_m = Om/2 = 0.15765, E(a) = exp(1 - 1/a),
mu(a) = H^2/(H^2 + beta_m E(a) H0^2), Sigma = 1, linear growth with the same early amplitude (a_i = 1e-3)."""
import numpy as np
from scipy.integrate import solve_ivp

Om, OL = 0.3153, 0.6847
bm = Om / 2                     # 0.15765, fixed in every chain
MU0_MGCAMB = -0.13495           # MGCAMB amplitude used in the Level 1 chains

E = lambda a: np.exp(1 - 1 / a)
H2 = lambda a: Om * a**-3 + OL                       # (H/H0)^2, photon sector
mu = lambda a: H2(a) / (H2(a) + bm * E(a))           # exact coupling
mu_mgcamb = lambda a: 1 + MU0_MGCAMB * (OL / H2(a)) / OL
Oma = lambda a: Om * a**-3 / H2(a)


def grow(m):
    def r(l, y):
        a = np.exp(l)
        dlnH = -1.5 * Om * a**-3 / H2(a)
        return [y[1], -(2 + dlnH) * y[1] + 1.5 * Oma(a) * m(a) * y[0]]
    return solve_ivp(r, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)


def D(S, a):
    return S.sol(np.log(a))[0]


def f(S, a):
    y = S.sol(np.log(a))
    return y[1] / y[0]


LCDM = grow(lambda a: 1.0)
IAM = grow(mu)
MGC = grow(mu_mgcamb)


def fs8_deficit(z, S=None):
    """Percent deficit of f sigma8 (IAM vs LambdaCDM, same early amplitude)."""
    S = IAM if S is None else S
    a = 1 / (1 + np.asarray(z, float))
    return 100 * (1 - f(S, a) * D(S, a) / (f(LCDM, a) * D(LCDM, a)))


def amp_deficit(z, S=None):
    S = IAM if S is None else S
    a = 1 / (1 + np.asarray(z, float))
    return 100 * (1 - D(S, a) / D(LCDM, a))
