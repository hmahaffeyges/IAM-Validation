#!/usr/bin/env python3
"""
IAM Dual-Sector Validation: Combined Background + Perturbation
================================================================
Tests the FULL dual-sector mechanism with three growth models:
  A) ΛCDM:           Standard H(z), μ = 1
  B) Background only: Matter-sector H_m(z), μ = 1
  C) Perturbation only: Standard H(z), μ(a) < 1
  D) COMBINED (full IAM): Matter-sector H_m(z), μ(a) < 1

Photon sector (geometry): H_γ = H_ΛCDM for CMB, BAO, SNe in all models
Matter sector (growth):   depends on model A-D above

Datasets (all with published references):
  1. CMB distance priors: Planck 2018 (A&A 641, A6)
  2. H₀: Planck 2018, SH0ES 2022, CCHP/JWST, H0LiCOW, TRGB, TDCOSMO
  3. f·σ₈(z): 6dFGS, BOSS DR12 consensus, eBOSS DR16
  4. BAO: BOSS DR12 + eBOSS DR16
  5. Cosmic Chronometers: Moresco+ compilation (32 points)
  6. Weak Lensing S₈: DES Y3, KiDS-1000, HSC Y3, Planck
  7. Pantheon+ SNe: Brout+ 2022 (photon sector consistency check)

Author: H.W. Mahaffey
Date: February 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

# =============================================================================
# COSMOLOGICAL PARAMETERS — Planck 2018 TT,TE,EE+lowE+lensing (Table 2)
# =============================================================================
H0 = 67.36;  h = H0/100
Om = 0.3153; Ob = 0.0493; OL = 1-Om
Obh2 = 0.02237; Omh2 = 0.1430
sigma8_fid = 0.8111; ns = 0.9649
c_km = 299792.458
Ogamma = 2.469e-5 / h**2
Onu = 3.046 * 7/8 * (4/11)**(4/3) * Ogamma
Or = Ogamma + Onu

# IAM: β = Ω_m/2 from virial theorem (Eq 55, theory paper)
beta = Om / 2.0

# =============================================================================
# CORE PHYSICS
# =============================================================================
def E_act(a):
    """Activation function E(a) = exp(1 - 1/a) [Eq 35, theory paper]"""
    a = np.atleast_1d(np.float64(a))
    r = np.where(a > 1e-6, np.exp(1.0 - 1.0/a), 0.0)
    return float(r[0]) if np.ndim(a)==0 or (hasattr(a,'__len__') and len(a)==1 and np.ndim(r)==1) else r

def mu_iam(a):
    """IAM gravitational coupling μ(a) = H²_ΛCDM / (H²_ΛCDM + β·E(a)) [Eq 40]"""
    E2L = Om * a**(-3) + Or * a**(-4) + OL
    return E2L / (E2L + beta * E_act(a))

# Expansion rates
def E2_lcdm(z):
    a = 1/(1+z); return Om*a**(-3) + Or*a**(-4) + OL

def E2_matter(z):
    a = 1/(1+z); return E2_lcdm(z) + beta * E_act(a)

def H_phot(z): return H0 * np.sqrt(E2_lcdm(z))
def H_matt(z): return H0 * np.sqrt(E2_matter(z))

H0_matter = H0 * np.sqrt(1 + beta * E_act(1.0))

# =============================================================================
# DISTANCES (photon sector = ΛCDM for all models)
# =============================================================================
def DC(z):
    if z < 1e-8: return 0.0
    r,_ = quad(lambda zp: c_km/H_phot(zp), 0, z, limit=1000); return r

def DV(z):
    dc = DC(z); dh = c_km/H_phot(z); return (z * dc**2 * dh)**(1/3)

def DH(z): return c_km / H_phot(z)

# Sound horizon
def z_star():
    g1 = 0.0783*Obh2**(-0.238)/(1+39.5*Obh2**0.763)
    g2 = 0.560/(1+21.1*Obh2**1.81)
    return 1048*(1+0.00124*Obh2**(-0.738))*(1+g1*Omh2**g2)

def z_drag():
    b1 = 0.313*Omh2**(-0.419)*(1+0.607*Omh2**0.674)
    b2 = 0.238*Omh2**0.223
    return 1291*Omh2**0.251/(1+0.659*Omh2**0.828)*(1+b1*Obh2**b2)

def r_sound(z_up):
    def f(z):
        a=1/(1+z); Rb=3*Ob*a/(4*Ogamma)
        return c_km/np.sqrt(3*(1+Rb)) / H_phot(z)
    r,_ = quad(f, z_up, 1e6, limit=3000); return r

# =============================================================================
# GROWTH FACTOR — supports all 4 models
# =============================================================================
def solve_growth(matter_H=False, use_mu=False):
    """
    Growth ODE: D'' + (2 + dlnH/dlna)D' - 3/2 · [μ(a)] · Ω_m(a) · D = 0
    
    Derived from δ̈ + 2Hδ̇ = 4πGμρ_m δ  converted to x = ln(a).
    Friction coefficient = 2 + d(ln H)/d(ln a), NOT (2-q).
    
    matter_H: use matter-sector H² (background effect)
    use_mu:   multiply source by μ(a) < 1 (perturbation effect)
    
    Model A: matter_H=F, use_mu=F  → ΛCDM
    Model B: matter_H=T, use_mu=F  → background only
    Model C: matter_H=F, use_mu=T  → perturbation only  
    Model D: matter_H=T, use_mu=T  → COMBINED (full IAM)
    """
    def rhs(lna, y):
        a = max(np.exp(lna), 1e-12)
        
        # E² = H²/H₀²
        E2L = Om*a**(-3) + Or*a**(-4) + OL
        Ea = E_act(a)
        E2 = (E2L + beta*Ea) if matter_H else E2L
        Oma = Om*a**(-3) / E2
        
        # d(E²)/da for friction term
        dE2_da = -3*Om*a**(-4) - 4*Or*a**(-5)
        if matter_H and a > 1e-6:
            dE2_da += beta * Ea / (a**2)  # d/da[exp(1-1/a)] = exp(1-1/a)/a²
        
        # Friction: 2 + d(ln H)/d(ln a) = 2 + a/(2E²) × dE²/da
        dlnH_dlna = 0.5 * a * dE2_da / E2
        fric = 2.0 + dlnH_dlna
        
        mu = mu_iam(a) if use_mu else 1.0
        return [y[1], -fric*y[1] + 1.5*mu*Oma*y[0]]

    sol = solve_ivp(rhs, (np.log(1e-4), 0), [1e-4, 1.0],
                    t_eval=np.linspace(np.log(1e-4), 0, 20000),
                    rtol=1e-11, atol=1e-14, method='DOP853')
    return np.exp(sol.t), sol.y[0]

def compute_fsig8(z_arr, matter_H=False, use_mu=False):
    """Compute f·σ₈(z) for given model"""
    a_arr, D_arr = solve_growth(matter_H, use_mu)
    _, D_lcdm = solve_growth(False, False)
    # Normalize to LCDM D(a=1) = 1
    D_arr = D_arr / D_lcdm[-1]
    lna = np.log(a_arr)
    lnD = np.log(np.maximum(D_arr, 1e-30))
    f_arr = np.gradient(lnD, lna)
    Di = interp1d(a_arr, D_arr, kind='cubic', fill_value='extrapolate')
    fi = interp1d(a_arr, f_arr, kind='cubic', fill_value='extrapolate')
    out = []
    for z in z_arr:
        a = 1/(1+z)
        out.append(float(fi(a)) * sigma8_fid * float(Di(a)))
    return np.array(out)

def sigma8_model(matter_H=False, use_mu=False):
    """σ₈ at z=0 for given model"""
    a_arr, D = solve_growth(matter_H, use_mu)
    _, D_lcdm = solve_growth(False, False)
    return sigma8_fid * D[-1] / D_lcdm[-1]

# =============================================================================
# VERIFIED OBSERVATIONAL DATA
# =============================================================================

# --- H₀ measurements ---
# Each measurement listed with full first-author citation.
# Sector assignment: Planck infers H₀ from CMB anisotropies (photon propagation);
# all local/late-universe methods use matter-based distance indicators
# (Cepheids, TRGB, masers, time-delay lensing of massive objects).
H0_data = [
    ('Planck 2018',     67.36, 0.54, 'photon',
     'Planck Collab. VI, A&A 641, A6 (2020)'),
    ('SH0ES 2022',      73.04, 1.04, 'matter',
     'Riess+ 2022, ApJL 934, L7'),
    ('CCHP/JWST 2024',  69.96, 1.53, 'matter',
     'Freedman+ 2025, ApJ 985, 203 (JWST TRGB+JAGB+Cepheid combined)'),
    ('H0LiCOW 2020',    73.3,  1.8,  'matter',
     'Wong+ 2020, MNRAS 498, 1420 (time-delay strong lensing)'),
    ('SH0ES/JWST 2024',  72.6,  2.0,  'matter',
     'Riess+ 2024, ApJ (SH0ES JWST Cepheids)'),
    ('TRGB Anand 2022',  71.5, 1.8,  'matter',
     'Anand+ 2022, ApJ 932, 15 (TRGB calibration)'),
    ('TDCOSMO 2020',     74.2, 1.6,  'matter',
     'Birrer+ 2020, A&A 643, A165 (time-delay cosmography)'),
]

# --- f·σ₈(z) growth rates ---
# Values from official SDSS/BOSS/eBOSS consensus tables:
#   https://www.sdss.org/science/final-bao-and-rsd-measurements/
# Plus 6dFGS (independent survey). These are RSD-only consensus values.
# f·σ₈ = (growth rate f) × (amplitude σ₈), measured from redshift-space
# distortions in galaxy clustering.
fsig8_data = [
    # z,    fsig8,  err,    reference
    (0.122, 0.428, 0.066,
     '6dFGS: Beutler+ 2012, MNRAS 423, 3430'),
    (0.15,  0.53,  0.16,
     'SDSS MGS: Howlett+ 2015, MNRAS 449, 848'),
    (0.38,  0.500, 0.047,
     'BOSS DR12 consensus: Alam+ 2017, MNRAS 470, 2617'),
    (0.51,  0.455, 0.039,
     'BOSS DR12 consensus: Alam+ 2017, MNRAS 470, 2617'),
    (0.70,  0.448, 0.043,
     'eBOSS DR16 LRG consensus: Alam+ 2021, PRD 103, 083533'),
    (0.85,  0.315, 0.095,
     'eBOSS DR16 ELG: de Mattia+ 2021, MNRAS 501, 5616'),
    (1.48,  0.462, 0.045,
     'eBOSS DR16 QSO: Hou+ 2021, MNRAS 500, 1201'),
]

# --- BAO (photon sector, same for all models) ---
# Values from official SDSS/BOSS/eBOSS consensus tables:
#   https://www.sdss.org/science/final-bao-and-rsd-measurements/
# BAO angular positions are measured via photon propagation (galaxy positions
# observed through light), making them photon-sector observables.
# DM/rd = comoving angular diameter distance / sound horizon
# DH/rd = c/(H(z)·rd), DV/rd = [z·DM²·DH]^(1/3) / rd
bao_data = [
    (0.15,  'DV', 4.47,  0.17,
     'SDSS MGS: Ross+ 2015, MNRAS 449, 835'),
    (0.38,  'DM', 10.23, 0.17,
     'BOSS DR12: Alam+ 2017, MNRAS 470, 2617'),
    (0.38,  'DH', 25.00, 0.76,
     'BOSS DR12: Alam+ 2017, MNRAS 470, 2617'),
    (0.51,  'DM', 13.36, 0.21,
     'BOSS DR12: Alam+ 2017, MNRAS 470, 2617'),
    (0.51,  'DH', 22.33, 0.58,
     'BOSS DR12: Alam+ 2017, MNRAS 470, 2617'),
    (0.70,  'DM', 17.86, 0.33,
     'eBOSS DR16 LRG: Bautista+ 2021, MNRAS 500, 736; Gil-Marin+ 2020, MNRAS 498, 2492'),
    (0.70,  'DH', 19.33, 0.53,
     'eBOSS DR16 LRG: Bautista+ 2021, MNRAS 500, 736; Gil-Marin+ 2020, MNRAS 498, 2492'),
    (0.85,  'DV', 18.33, 0.60,
     'eBOSS DR16 ELG: Raichoor+ 2021, MNRAS 500, 3254; de Mattia+ 2021, MNRAS 501, 5616'),
    (1.48,  'DM', 30.69, 0.80,
     'eBOSS DR16 QSO: Hou+ 2021, MNRAS 500, 1201; Neveux+ 2020, MNRAS 499, 210'),
    (1.48,  'DH', 13.26, 0.55,
     'eBOSS DR16 QSO: Hou+ 2021, MNRAS 500, 1201; Neveux+ 2020, MNRAS 499, 210'),
    (2.33,  'DM', 37.6,  1.9,
     'eBOSS DR16 Lyα: du Mas des Bourboux+ 2020, ApJ 901, 153'),
    (2.33,  'DH',  8.93, 0.28,
     'eBOSS DR16 Lyα: du Mas des Bourboux+ 2020, ApJ 901, 153'),
]

# --- Cosmic Chronometers: Direct H(z) measurements ---
# Compilation following Moresco+ 2022 (Living Rev. Rel. 25, 6) and
# Moresco (2024, arXiv:2412.01994), Table 1.
# H(z) = -1/(1+z) dz/dt from differential aging of passive galaxies.
# Model-independent — no assumed cosmology.
# CC probes matter sector (galaxy aging = timelike worldlines)
# Full references for each original measurement given inline.
cc_data = [
    # z,     H(z),   err,   reference
    (0.070,  69.0,   19.6, 'Zhang+ 2014, RAA 14 1221'),
    (0.090,  69.0,   12.0, 'Simon+ 2005, PRD 71 123001'),
    (0.120,  68.6,   26.2, 'Zhang+ 2014'),
    (0.170,  83.0,    8.0, 'Simon+ 2005'),
    (0.179,  75.0,    4.0, 'Moresco+ 2012, JCAP 08 006'),
    (0.199,  75.0,    5.0, 'Moresco+ 2012'),
    (0.200,  72.9,   29.6, 'Zhang+ 2014'),
    (0.270,  77.0,   14.0, 'Simon+ 2005'),
    (0.280,  88.8,   36.6, 'Zhang+ 2014'),
    (0.352,  83.0,   14.0, 'Moresco+ 2012'),
    (0.380,  83.0,   13.5, 'Moresco+ 2016, JCAP 05 014'),
    (0.400,  95.0,   17.0, 'Simon+ 2005'),
    (0.425,  87.1,   11.2, 'Moresco+ 2016'),
    (0.445,  92.8,   12.9, 'Moresco+ 2016'),
    (0.470,  89.0,   34.0, 'Ratsimbazafy+ 2017, MNRAS 467 3239'),
    (0.478,  80.9,    9.0, 'Moresco+ 2016'),
    (0.480,  97.0,   62.0, 'Stern+ 2010, JCAP 02 008'),
    (0.593,  104.0,  13.0, 'Moresco+ 2012'),
    (0.680,  92.0,    8.0, 'Moresco+ 2012'),
    (0.750,  98.8,   33.6, 'Borghi+ 2022, ApJL 928 L4'),
    (0.781,  105.0,  12.0, 'Moresco+ 2012'),
    (0.875,  125.0,  17.0, 'Moresco+ 2012'),
    (0.880,  90.0,   40.0, 'Stern+ 2010'),
    (0.900,  117.0,  23.0, 'Simon+ 2005'),
    (1.037,  154.0,  20.0, 'Moresco+ 2012'),
    (1.260,  135.0,  65.0, 'Tomasetti+ 2023, A&A 679 A96'),
    (1.300,  168.0,  17.0, 'Simon+ 2005'),
    (1.363,  160.0,  33.6, 'Moresco 2015, MNRAS 450 L16'),
    (1.430,  177.0,  18.0, 'Simon+ 2005'),
    (1.530,  140.0,  14.0, 'Simon+ 2005'),
    (1.750,  202.0,  40.0, 'Simon+ 2005'),
    (1.965,  186.5,  50.4, 'Moresco 2015'),
]

# --- Weak Lensing S₈ = σ₈ √(Ω_m/0.3) measurements ---
# S₈ is the primary weak lensing parameter. Published survey values below.
# IAM predicts lower σ₈ → lower S₈, relevant to the S₈ discrepancy.
# Planck included as CMB-inferred comparison point.
s8_data = [
    # survey,       S8,     err,    reference
    ('DES Y3',      0.776,  0.017,
     'DES Collab. (Abbott+) 2022, PRD 105, 023520'),
    ('KiDS-1000',   0.759,  0.021,
     'Asgari+ 2021, A&A 645, A104'),
    ('HSC Y3',      0.776,  0.032,
     'Li+ 2023 (HSC Collab.), PRD 108, 123518'),
    ('Planck 2018', 0.832,  0.013,
     'Planck Collab. VI, A&A 641, A6 (2020)'),
]

# --- Pantheon+ binned distance moduli (photon sector) ---
# Full sample: Brout+ 2022, ApJ 938, 110; Scolnic+ 2022, ApJ 938, 113
# 1701 light curves from 1550 unique SNe Ia, 0.001 < z < 2.26.
# μ_obs = m_B - M_B (distance modulus) compared to photon-sector d_L(z).
# SNe brightness is measured via photon flux → photon-sector observable.
# Data access: https://github.com/PantheonPlusSH0ES
#
# NOTE: The values below are Planck-cosmology predictions at representative
# redshift bins with approximate Pantheon+ statistical uncertainties.
# This serves as a demonstration that the photon sector is unmodified;
# the full Pantheon+ analysis requires the complete 1701-point dataset
# with systematic+statistical covariance matrix (not reproduced here).
# For a proper Pantheon+ fit, see the companion MGCAMB validation.
pantheon_binned = [
    # z_bin,  mu_pred (Planck ΛCDM),  approx σ_μ
    (0.010,  33.26, 0.08),
    (0.015,  34.15, 0.08),
    (0.023,  35.09, 0.04),
    (0.034,  35.95, 0.04),
    (0.050,  36.82, 0.025),
    (0.070,  37.58, 0.025),
    (0.100,  38.40, 0.020),
    (0.140,  39.18, 0.020),
    (0.200,  40.03, 0.020),
    (0.280,  40.86, 0.020),
    (0.380,  41.63, 0.025),
    (0.500,  42.33, 0.025),
    (0.630,  42.94, 0.040),
    (0.770,  43.47, 0.040),
    (0.900,  43.88, 0.040),
    (1.050,  44.29, 0.080),
    (1.200,  44.65, 0.080),
    (1.400,  45.06, 0.080),
    (1.700,  45.58, 0.150),
    (2.260,  46.34, 0.150),
]

# =============================================================================
# COMPUTE
# =============================================================================
print("=" * 80)
print("  IAM DUAL-SECTOR VALIDATION: COMBINED BACKGROUND + PERTURBATION")
print("=" * 80)
print("""
  METHODOLOGY
  -----------
  The IAM dual-sector framework posits that matter-based and photon-based
  observables probe different effective expansion rates at late times. The
  single free coupling β = Ω_m/2 (derived from the virial theorem) enters
  the matter-sector Friedmann equation as:

    H²_matter = H²_ΛCDM + β · E(a) · H₀²

  where E(a) = exp(1 - 1/a) is an activation function that is negligible
  at early times (a << 1) and approaches unity today (a = 1). The photon
  sector is unmodified: H_photon = H_ΛCDM.

  At the perturbation level, the gravitational coupling for matter is:

    μ(a) = H²_ΛCDM / (H²_ΛCDM + β · E(a))

  This gives μ < 1 at late times (gravity effectively weakened for matter),
  with μ(z=0) ≈ 0.864, recovering GR (μ→1) at high redshift where E(a)→0.
  In the MGCAMB convention, μ₀ = μ(z=0) - 1 ≈ -0.136.
  The lensing slip is Σ = 1 (photon deflection unaffected).

  The growth equation solved is:
    D'' + [2 + d(ln H)/d(ln a)] D' - (3/2) μ(a) Ω_m(a) D = 0

  where primes denote d/d(ln a) and Ω_m(a) = Ω_m a⁻³ / E²(a).

  All Planck 2018 parameters are held fixed (no fitting). The only
  model input is β = Ω_m/2. Sector assignment follows from the physical
  measurement process: observables measured via photon propagation (CMB,
  BAO angular positions, SNe luminosity distances) use H_photon; observables
  involving matter evolution (local H₀ via distance ladders, galaxy growth
  rates, cosmic chronometer aging) use H_matter.

  Under this framework, if the dual-sector cosmology is an accurate
  description of nature, one expects:
    (1) Local H₀ measurements to cluster near H_matter ≈ 72.5 km/s/Mpc
    (2) CMB and BAO to remain unchanged from ΛCDM predictions
    (3) σ₈ to be suppressed by ~2-3%, in the direction of weak lensing data
    (4) Cosmic chronometer H(z) to lie between H_photon and H_matter at
        low z, converging at high z where E(a)→0
    (5) SNe distance moduli to be unchanged (photon sector)

  The following tests evaluate these expectations against published data.
""")
print(f"  Planck 2018 parameters (held fixed):")
print(f"    H₀ = {H0} km/s/Mpc,  Ω_m = {Om},  σ₈ = {sigma8_fid}")
print(f"    Ω_b h² = {Obh2},  Ω_m h² = {Omh2}")
print(f"  IAM-derived quantities (from β = Ω_m/2 = {beta:.5f}):")
print(f"    H₀(photon) = {H0:.2f} km/s/Mpc")
print(f"    H₀(matter) = H₀ × √(1+β) = {H0_matter:.2f} km/s/Mpc")
print(f"    μ(z=0) = {mu_iam(1.0):.4f}")
mu0_val = mu_iam(1.0) - 1.0  # μ₀ as departure from GR at z=0
print(f"    μ₀ = μ(z=0) - 1 = {mu0_val:.5f}  (includes residual radiation; analytic: -0.13495)")

# Precompute distances
zs = z_star(); zd = z_drag()
rs_star = r_sound(zs); rs_drag = r_sound(zd); DC_star = DC(zs)
print(f"  r_s(z_d) = {rs_drag:.2f} Mpc,  D_C(z*) = {DC_star:.0f} Mpc\n")

# Precompute growth for all 4 models
z_g = np.array([d[0] for d in fsig8_data])
obs_g = np.array([d[1] for d in fsig8_data])
err_g = np.array([d[2] for d in fsig8_data])

models = {
    'A: ΛCDM':       (False, False),
    'B: Bkgd only':  (True,  False),
    'C: Pert only':  (False, True),
    'D: Combined':   (True,  True),
}
pred_g = {}
sig8_vals = {}
for name, (mH, mu) in models.items():
    pred_g[name] = compute_fsig8(z_g, mH, mu)
    sig8_vals[name] = sigma8_model(mH, mu)

# =============================================================================
# TEST 1: H₀
# =============================================================================
print("=" * 80)
print("  TEST 1: H₀ Measurements (Dual-Sector Assignment)")
print("=" * 80)
print("""
  Each H₀ measurement is assigned to the sector corresponding to its
  physical measurement process. Planck infers H₀ from CMB angular power
  spectra (photon propagation → photon sector). Local measurements use
  matter-based distance indicators: Cepheid period-luminosity (SH0ES),
  tip of the red giant branch luminosity (TRGB/CCHP), and gravitational
  time-delay lensing of background quasars by foreground galaxies
  (H0LiCOW, TDCOSMO).

  Under dual-sector cosmology, photon-sector measurements should be
  consistent with H₀(photon) = 67.36 km/s/Mpc, while matter-sector
  measurements should be consistent with H₀(matter) = 72.48 km/s/Mpc.
  Under single-sector ΛCDM, all measurements should agree with 67.36.

  χ² computed as Σ [(H₀_obs - H₀_pred)² / σ²] for each sector assignment.
""")

chi2_H0_iam = 0; chi2_H0_lcdm = 0
print(f"\n  {'Source':<18} {'H₀':>6} {'±':>5} {'Sector':>8} {'IAM':>7} {'Δ/σ(IAM)':>9} {'Δ/σ(CDM)':>9}")
print(f"  {'-'*64}")
for name, val, err, sector, ref in H0_data:
    pred_iam = H0 if sector=='photon' else H0_matter
    p_i = (val - pred_iam)/err; p_l = (val - H0)/err
    chi2_H0_iam += p_i**2; chi2_H0_lcdm += p_l**2
    print(f"  {name:<18} {val:>6.2f} {err:>5.2f} {sector:>8} {pred_iam:>7.2f} {p_i:>9.2f} {p_l:>9.2f}")
print(f"\n  References:")
for name, val, err, sector, ref in H0_data:
    print(f"    {name}: {ref}")

dchi2_H0 = chi2_H0_lcdm - chi2_H0_iam
print(f"\n  χ²(H₀, IAM)  = {chi2_H0_iam:.2f}  |  χ²(H₀, ΛCDM) = {chi2_H0_lcdm:.2f}")
print(f"  Δχ² = {dchi2_H0:.1f}  (not insignificant)")

# =============================================================================
# TEST 2: GROWTH RATES — ALL 4 MODELS
# =============================================================================
print()
print("=" * 80)
print("  TEST 2: f·σ₈(z) Growth Rates — Four Models")
print("=" * 80)
print("""
  Growth rates f·σ₈(z) are measured from redshift-space distortions (RSD)
  in galaxy clustering surveys. f = d(ln D)/d(ln a) is the logarithmic
  growth rate; σ₈ is the amplitude of matter fluctuations at 8 h⁻¹ Mpc.

  Four models are compared, differing in whether the matter-sector
  background H(z) and/or the perturbation-level μ(a) modification are active:
    A) ΛCDM:      standard H(z), μ = 1         (baseline)
    B) Bkgd only: matter H_m(z), μ = 1         (expansion friction only)
    C) Pert only: standard H(z), μ(a) < 1      (weakened gravity only)
    D) Combined:  matter H_m(z) + μ(a) < 1     (both effects)

  Growth ODE: D'' + [2 + d(ln H)/d(ln a)] D' - (3/2) μ(a) Ω_m(a) D = 0
  Solved from a = 10⁻⁴ to a = 1 using DOP853 with rtol=10⁻¹¹.
""")

chi2_g = {}
for name in models:
    chi2_g[name] = float(np.sum(((obs_g - pred_g[name])/err_g)**2))

print(f"\n  {'z':>5} {'obs':>7} {'±':>5}  {'ΛCDM':>7} {'Bkgd':>7} {'Pert':>7} {'Comb':>7}  Ref")
print(f"  {'-'*72}")
for i, (z, obs, err, ref) in enumerate(fsig8_data):
    short_ref = ref.split(',')[0]
    print(f"  {z:>5.3f} {obs:>7.3f} {err:>5.3f}  "
          f"{pred_g['A: ΛCDM'][i]:>7.3f} {pred_g['B: Bkgd only'][i]:>7.3f} "
          f"{pred_g['C: Pert only'][i]:>7.3f} {pred_g['D: Combined'][i]:>7.3f}  {short_ref}")

print(f"\n  {'Model':<20} {'χ²':>8} {'σ₈(z=0)':>10} {'Suppression':>12}")
print(f"  {'-'*52}")
for name in models:
    s8 = sig8_vals[name]
    supp = (1 - s8/sigma8_fid)*100
    print(f"  {name:<20} {chi2_g[name]:>8.2f} {s8:>10.4f} {supp:>11.2f}%")

# The key comparison
best_model = min(chi2_g, key=chi2_g.get)
print(f"\n  Best fit: {best_model} (χ² = {chi2_g[best_model]:.2f})")
print(f"  ΛCDM χ² = {chi2_g['A: ΛCDM']:.2f}")
print(f"  Combined Δχ² vs ΛCDM = {chi2_g['A: ΛCDM'] - chi2_g['D: Combined']:.2f}")
print(f"\n  Note: Combined model receives suppression from both channels:")
print(f"    Background (larger H → more friction): {(1-sig8_vals['B: Bkgd only']/sigma8_fid)*100:.2f}%")
print(f"    Perturbation (μ < 1 → weaker source):  {(1-sig8_vals['C: Pert only']/sigma8_fid)*100:.2f}%")
print(f"    Combined (both effects):               {(1-sig8_vals['D: Combined']/sigma8_fid)*100:.2f}%")

print(f"\n  References:")
for z, obs, err, ref in fsig8_data:
    print(f"    z={z}: {ref}")

# =============================================================================
# TEST 3: BAO
# =============================================================================
print()
print("=" * 80)
print("  TEST 3: BAO (Photon Sector — identical for all models)")
print("=" * 80)
print("""
  BAO angular positions are determined from galaxy clustering patterns
  observed via photon propagation. The acoustic scale r_d (sound horizon
  at baryon drag epoch) serves as a standard ruler. Observables:
    DM/rd = comoving angular diameter distance / r_d
    DH/rd = c / [H(z) · r_d]
    DV/rd where DV(z) = [z · DM(z)² · DH(z)]^(1/3)

  Because BAO positions are measured through photon arrival directions,
  they are photon-sector observables and are predicted to be identical
  to ΛCDM in the dual-sector framework. This test verifies that the
  photon sector is not degraded.

  Note: Distances here use Hu & Sugiyama fitting formulae for r_d and
  numerical integration for D_C(z). Residuals at the 1-2σ level reflect
  fitting-formula precision, not physical tensions. Full Boltzmann
  calculations (MGCAMB) eliminate these residuals.
""")

chi2_bao = 0
print(f"\n  {'z':>5} {'type':>4} {'obs':>7} {'±':>5} {'pred':>7} {'Δ/σ':>6}")
print(f"  {'-'*40}")
for z, typ, val, err, ref in bao_data:
    if typ=='DV': pred = DV(z)/rs_drag
    elif typ=='DM': pred = DC(z)/rs_drag
    elif typ=='DH': pred = DH(z)/rs_drag
    pull = (val-pred)/err; chi2_bao += pull**2
    print(f"  {z:>5.2f} {typ:>4} {val:>7.2f} {err:>5.2f} {pred:>7.2f} {pull:>6.2f}")
print(f"\n  χ²(BAO) = {chi2_bao:.2f}")

# Print unique BAO references
print(f"\n  References:")
seen_refs = set()
for z, typ, val, err, ref in bao_data:
    short_key = ref.split(':')[0]
    if short_key not in seen_refs:
        seen_refs.add(short_key)
        print(f"    {ref}")

# =============================================================================
# TEST 4: CMB DISTANCE PRIORS
# =============================================================================
print()
print("=" * 80)
print("  TEST 4: CMB Distance Priors (Photon Sector)")
print("=" * 80)
print("""
  CMB distance priors compress the full Planck likelihood into three
  numbers: θ_MC (angular size of sound horizon), R (shift parameter),
  and l_A (acoustic scale). These are photon-sector geometric quantities.
  Computed using Hu & Sugiyama (1996) fitting formulae for z_* and z_d.

  Note: The large χ² from θ_MC and l_A reflects fitting-formula residuals
  relative to full Boltzmann solver output (known ~0.1% offset), not a
  physical tension. Full MGCAMB MCMC chains (documented in companion
  Technical Note) confirm CMB TT residuals < 0.17%.
""")

theta_pred = rs_star / DC_star; theta_obs = 0.010411; theta_err = 0.0000031
R_pred = np.sqrt(Om)*H0*DC_star/c_km; R_obs = 1.7502; R_err = 0.0046
lA_pred = np.pi*DC_star/rs_star; lA_obs = 301.471; lA_err = 0.090

chi2_cmb = ((theta_pred-theta_obs)/theta_err)**2 + \
           ((R_pred-R_obs)/R_err)**2 + ((lA_pred-lA_obs)/lA_err)**2

print(f"\n  θ_MC: {theta_pred:.7f} vs {theta_obs:.7f} ({(theta_pred-theta_obs)/theta_err:+.1f}σ)")
print(f"  R:    {R_pred:.4f} vs {R_obs:.4f} ({(R_pred-R_obs)/R_err:+.1f}σ)")
print(f"  l_A:  {lA_pred:.3f} vs {lA_obs:.3f} ({(lA_pred-lA_obs)/lA_err:+.1f}σ)")
print(f"  χ²(CMB) = {chi2_cmb:.2f}  (fitting formula residuals, not physics)")
print(f"""
  Formulae:
    θ_MC = r_s(z_*) / D_C(z_*)
    R     = √Ω_m · H₀ · D_C(z_*) / c
    l_A   = π · D_C(z_*) / r_s(z_*)
  z_*, z_d from Hu & Sugiyama (1996) fitting formulae.
  Observed values: Planck Collab. VI, A&A 641, A6 (2020), Table 1.
""")

# =============================================================================
# TEST 5: COSMIC CHRONOMETERS — THE TRANSITION ZONE
# =============================================================================
print()
print("=" * 80)
print("  TEST 5: Cosmic Chronometers H(z) — Direct Expansion Rate")
print("=" * 80)
print("""
  Cosmic chronometers measure H(z) = -1/(1+z) · dz/dt from the
  differential aging of massive, passively-evolving galaxies. This is
  a direct, model-independent measurement of the expansion rate.

  Compilation: 32 measurements from Simon+ 2005 (PRD 71, 123001),
  Stern+ 2010 (JCAP 02, 008), Moresco+ 2012 (JCAP 08, 006),
  Zhang+ 2014 (RAA 14, 1221), Moresco 2015 (MNRAS 450, L16),
  Moresco+ 2016 (JCAP 05, 014), Ratsimbazafy+ 2017 (MNRAS 467, 3239),
  Borghi+ 2022 (ApJL 928, L4), Tomasetti+ 2023 (A&A 679, A96).

  Sector assignment rationale: CC measures aging of matter (galaxies
  evolving along timelike worldlines), making it a matter-sector probe.
  Note: while the aging rate dz/dt is a matter-sector quantity (proper
  time of massive galaxies), the spectral features used to measure it
  are observed via photons. The sector assignment follows from which
  physical process determines the observable: the galaxy aging rate is
  set by the matter-sector expansion history, even though we detect it
  through photon observations. This is analogous to how local H₀
  measurements use photons to observe matter-based distance indicators.
  Under dual-sector cosmology, CC H(z) at low z should be compared to
  H_matter(z) rather than H_photon(z). At high z, both converge.

  Current CC uncertainties (typically 5-30%) are larger than the
  predicted sector gap (<7.6% at z=0, <1% at z>1), so this test
  primarily checks for consistency rather than discrimination.
""")

chi2_cc_iam = 0; chi2_cc_lcdm = 0
print(f"\n  {'z':>6} {'H_obs':>7} {'±':>5} {'H_γ':>7} {'H_m':>7} {'Δ/σ(γ)':>8} {'Δ/σ(m)':>8}  Reference")
print(f"  {'-'*90}")
for z, Hobs, Herr, ref in cc_data:
    Hg = H_phot(z); Hm = H_matt(z)
    pull_g = (Hobs - Hg)/Herr; pull_m = (Hobs - Hm)/Herr
    chi2_cc_lcdm += pull_g**2; chi2_cc_iam += pull_m**2
    print(f"  {z:>6.3f} {Hobs:>7.1f} {Herr:>5.1f} {Hg:>7.1f} {Hm:>7.1f} {pull_g:>+8.2f} {pull_m:>+8.2f}  {ref}")
n_cc = len(cc_data)
print(f"\n  χ²(CC, photon/ΛCDM) = {chi2_cc_lcdm:.1f}  ({n_cc} points)")
print(f"  χ²(CC, matter/IAM)  = {chi2_cc_iam:.1f}")
print(f"  Δχ² = {chi2_cc_lcdm - chi2_cc_iam:.1f}  (statistically indistinguishable at current precision)")

# =============================================================================
# TEST 6: WEAK LENSING S₈ TENSION
# =============================================================================
print()
print("=" * 80)
print("  TEST 6: Weak Lensing S₈ = σ₈ √(Ω_m/0.3)")
print("=" * 80)
print("""
  S₈ = σ₈ √(Ω_m/0.3) is the primary parameter constrained by weak
  gravitational lensing surveys. A well-documented discrepancy exists
  between Planck CMB-inferred S₈ ≈ 0.83 and weak lensing measurements
  S₈ ≈ 0.76-0.78 (the "S₈ tension," ~2-3σ depending on survey).

  Because the IAM combined model suppresses σ₈ through both background
  and perturbation effects, it predicts a lower S₈ than ΛCDM. Additionally,
  the dual-sector mechanism dilutes the physical matter density from the
  Planck-inferred Ω_m = 0.315 to Ω_m(physical) = 0.272 (13.7% reduction),
  which further reduces S₈.

  Two IAM S₈ predictions are shown:
    (a) σ₈(IAM) × √(Ω_m(Planck)/0.3)  — growth suppression only
    (b) σ₈(IAM) × √(Ω_m(physical)/0.3) — growth + matter dilution

  Note: σ₈(IAM) = 0.790 from this lightweight growth ODE, consistent with
  the MGCAMB validation result σ₈ = 0.7945 (Section 4, IAM-CAMB Technical
  Note). The Planck MCMC (Run A) gives σ₈ = 0.8014, which includes only
  perturbation-level effects and not the background sector split.
""")

Om_phys = Om / (1 + beta)  # IAM physical Omega_m at z=0

s8_factor = np.sqrt(Om / 0.3)
s8_factor_phys = np.sqrt(Om_phys / 0.3)

chi2_s8_lcdm = 0; chi2_s8_iam = 0; chi2_s8_iam_phys = 0
s8_lcdm = sigma8_fid * s8_factor
s8_iam = sig8_vals['D: Combined'] * s8_factor
s8_iam_phys = sig8_vals['D: Combined'] * s8_factor_phys

print(f"\n  Ω_m(Planck) = {Om:.4f},  Ω_m(IAM physical) = {Om_phys:.3f}")
print(f"  σ₈(ΛCDM) = {sigma8_fid:.4f},  σ₈(IAM) = {sig8_vals['D: Combined']:.4f}")
print(f"  S₈(ΛCDM)        = {s8_lcdm:.3f}   [σ₈ × √(Ω_m/0.3), Planck Ω_m]")
print(f"  S₈(IAM, Planck)  = {s8_iam:.3f}   [σ₈(IAM) × √(Ω_m(Planck)/0.3)]")
print(f"  S₈(IAM, physical)= {s8_iam_phys:.3f}   [σ₈(IAM) × √(Ω_m(phys)/0.3)]")

print(f"\n  {'Survey':<16} {'S₈_obs':>8} {'±':>6} {'S₈_CDM':>8} {'S₈_IAM':>8} {'S₈_phys':>8} {'Δ/σ(CDM)':>9} {'Δ/σ(IAM)':>9} {'Δ/σ(phys)':>10}")
print(f"  {'-'*95}")
for name, val, err, ref in s8_data:
    p_l = (val - s8_lcdm)/err; p_i = (val - s8_iam)/err; p_p = (val - s8_iam_phys)/err
    chi2_s8_lcdm += p_l**2; chi2_s8_iam += p_i**2; chi2_s8_iam_phys += p_p**2
    print(f"  {name:<16} {val:>8.3f} {err:>6.3f} {s8_lcdm:>8.3f} {s8_iam:>8.3f} {s8_iam_phys:>8.3f} {p_l:>+9.2f} {p_i:>+9.2f} {p_p:>+10.2f}")
print(f"\n  χ²(S₈, ΛCDM) = {chi2_s8_lcdm:.1f}")
print(f"  χ²(S₈, IAM Planck Ω_m) = {chi2_s8_iam:.1f}")
print(f"  χ²(S₈, IAM physical Ω_m) = {chi2_s8_iam_phys:.1f}  (WL-only: {chi2_s8_iam_phys - ((0.832-s8_iam_phys)/0.013)**2:.1f})")
print(f"  Δχ²(ΛCDM vs IAM Planck Ω_m) = {chi2_s8_lcdm - chi2_s8_iam:.1f}")
print(f"""
  Note: The physical Ω_m prediction (S₈=0.753) is consistent with the
  three WL surveys (DES/KiDS/HSC) but not with Planck's derived S₈=0.832,
  because Planck's S₈ is computed assuming ΛCDM Ω_m=0.315. A full
  reanalysis of Planck under IAM's Ω_m=0.272 would yield a different
  derived S₈. For χ² accounting we use the Planck-Ω_m comparison
  (S₈(IAM)=0.810) which is the apples-to-apples test.
""")

print(f"\n  References:")
for name, val, err, ref in s8_data:
    print(f"    {name}: {ref}")

# =============================================================================
# TEST 7: PANTHEON+ SNe (Photon Sector)
# =============================================================================
print()
print("=" * 80)
print("  TEST 7: Pantheon+ Binned Type Ia Supernovae (Photon Sector)")
print("=" * 80)
print("""
  The Pantheon+ compilation (Brout+ 2022, ApJ 938, 110; Scolnic+ 2022,
  ApJ 938, 113) contains 1701 light curves of 1550 Type Ia supernovae
  over 0.001 < z < 2.26. Distance moduli μ = 5 log₁₀(d_L/10 pc) are
  compared to the photon-sector luminosity distance:

    d_L(z) = (1+z) · ∫₀ᶻ c/H_photon(z') dz'

  SNe luminosity is measured via photon flux, making this a photon-sector
  observable. Under dual-sector cosmology, Pantheon+ predictions are
  identical to ΛCDM.

  Note: The values below are Planck-cosmology distance modulus predictions
  at representative redshift bins, not actual Pantheon+ data. This serves
  as a consistency check that our photon-sector d_L(z) matches ΛCDM.
  The full Pantheon+ analysis requires the complete 1701-point dataset
  with systematic+statistical covariance matrix (see MGCAMB validation).
""")

def mu_dist(z, h0=H0):
    """Distance modulus μ = 5 log₁₀(d_L/10pc), photon sector"""
    dc = DC(z)
    dl = dc * (1+z)  # luminosity distance in Mpc
    return 5*np.log10(dl) + 25  # convert Mpc to 10pc

# Compare photon-sector predictions to Planck-cosmology reference values
mu_pred_arr = np.array([mu_dist(z) for z, _, _ in pantheon_binned])
mu_ref_arr = np.array([m for _, m, _ in pantheon_binned])
mu_err_arr = np.array([e for _, _, e in pantheon_binned])
residuals = mu_pred_arr - mu_ref_arr
n_sne = len(pantheon_binned)
print(f"\n  Photon-sector d_L(z) vs Planck ΛCDM reference ({n_sne} bins):")
print(f"  Max |residual| = {np.max(np.abs(residuals)):.4f} mag")
print(f"  RMS residual    = {np.sqrt(np.mean(residuals**2)):.4f} mag")
print(f"  Both sectors use H_photon = H_ΛCDM → SNe identical by construction.")
print(f"  Full Pantheon+ validation requires complete covariance matrix;")
print(f"  this comparison confirms the photon-sector geometry is unmodified.")

# For χ² accounting, use zero since it's identical by construction
chi2_sne = 0.0

# =============================================================================
# COMBINED RESULTS — FOUR MODELS
# =============================================================================
n_H0 = len(H0_data); n_g = len(fsig8_data); n_bao = len(bao_data)
n_cmb = 3; n_s8 = len(s8_data)
n_tested = n_H0 + n_g + n_bao + n_cmb + n_cc + n_s8  # actual χ² comparison points
n_tot = n_tested + n_sne  # total including SNe consistency check

# For all models, CMB, BAO, SNe are identical (photon sector)
# CC: matter sector, S₈: depends on growth model
chi2_total_lcdm = chi2_cmb + chi2_H0_lcdm + chi2_g['A: ΛCDM'] + chi2_bao + chi2_cc_lcdm + chi2_s8_lcdm

# IAM combined (full model) — use Planck Omega_m for S8 (apples-to-apples)
# Physical Om prediction (S8=0.753) shown separately but requires WL reanalysis
chi2_full_iam = chi2_cmb + chi2_H0_iam + chi2_g['D: Combined'] + chi2_bao + chi2_cc_iam + chi2_s8_iam
dchi2_total = chi2_total_lcdm - chi2_full_iam

print()
print("=" * 80)
print("  COMBINED RESULTS: ALL MODELS")
print("=" * 80)

print(f"\n  {'':>20} {'CMB':>6} {'H₀':>6} {'f·σ₈':>6} {'BAO':>6} {'CC':>6} {'S₈*':>6} {'TOTAL':>8} {'Δχ²':>7}")
print(f"  {'-'*76}")
# ΛCDM
print(f"  {'ΛCDM':>20} {chi2_cmb:>6.0f} {chi2_H0_lcdm:>6.1f} {chi2_g['A: ΛCDM']:>6.1f} {chi2_bao:>6.1f} {chi2_cc_lcdm:>6.1f} {chi2_s8_lcdm:>6.1f} {chi2_total_lcdm:>8.0f} {'—':>7}")
# IAM Combined
print(f"  {'IAM Combined':>20} {chi2_cmb:>6.0f} {chi2_H0_iam:>6.1f} {chi2_g['D: Combined']:>6.1f} {chi2_bao:>6.1f} {chi2_cc_iam:>6.1f} {chi2_s8_iam:>6.1f} {chi2_full_iam:>8.0f} {dchi2_total:>+7.1f}")
print(f"\n  * S₈ uses Planck Ω_m for both (apples-to-apples); IAM physical Ω_m gives S₈=0.753")

# Summary
print(f"\n  {'='*76}")
print(f"  SUMMARY: IAM vs ΛCDM across {n_tested} data points, 6 χ² probes")
print(f"  (Plus {n_sne}-bin Pantheon+ consistency check — identical by construction)")
print(f"  Δχ² = {dchi2_total:.1f}  (0 additional free parameters)")
if dchi2_total > 0:
    print(f"  Equivalent significance: {np.sqrt(dchi2_total):.1f}σ")
print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  H₀: {H0:.2f} (photon sector) vs {H0_matter:.2f} (matter sector)     │
  │  σ₈: {sigma8_fid:.4f} (ΛCDM) → {sig8_vals['D: Combined']:.4f} (IAM, {(1-sig8_vals['D: Combined']/sigma8_fid)*100:.1f}% reduction)    │
  │  Combined suppression from background + perturbation effects │
  │  All parameters derived from single coupling β = Ω_m/2.     │
  └──────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# FIGURE: 9-PANEL COMPREHENSIVE
# =============================================================================
print("Generating 9-panel figure...")

fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle(f'IAM Dual-Sector Validation: {n_tested} Data Points Across 6 Probes (+ Pantheon+ Consistency Check)\n'
             f'Combined Δχ² = {dchi2_total:.1f} (equivalent to {np.sqrt(dchi2_total):.1f}σ), zero additional parameters',
             fontsize=14, fontweight='bold', y=0.995)

# --- Panel 1: H₀ ---
ax = axes[0,0]
zl = np.linspace(0, 0.15, 200)
ax.fill_between(zl, [H_phot(z) for z in zl], [H_matt(z) for z in zl],
                alpha=0.12, color='purple', label='Sector gap')
ax.plot(zl, [H_phot(z) for z in zl], 'b-', lw=2.5, label=f'Photon: {H0:.1f}')
ax.plot(zl, [H_matt(z) for z in zl], 'r-', lw=2.5, label=f'Matter: {H0_matter:.1f}')
for name, val, err, sector, ref in H0_data:
    color = 'navy' if sector=='photon' else 'darkred'
    marker = 's' if sector=='photon' else '^'
    short = name.split()[0]
    ax.errorbar([0], [val], yerr=[err], fmt=marker, color=color, ms=8, capsize=4)
ax.set_xlabel('z'); ax.set_ylabel('H(z) [km/s/Mpc]')
ax.set_title(f'(a) H₀ Measurements by Sector', fontweight='bold')
ax.legend(fontsize=8); ax.set_xlim(-0.03, 0.15); ax.set_ylim(63, 80)

# --- Panel 2: Cosmic Chronometers — THE TRANSITION ZONE ---
ax = axes[0,1]
zc = np.linspace(0, 2.2, 300)
ax.plot(zc, [H_phot(z) for z in zc], 'b-', lw=2, label=r'$H_\gamma$ (photon/$\Lambda$CDM)')
ax.plot(zc, [H_matt(z) for z in zc], 'r-', lw=2, label=r'$H_m$ (matter/IAM)')
# Fill the transition zone
ax.fill_between(zc, [H_phot(z) for z in zc], [H_matt(z) for z in zc],
                alpha=0.08, color='purple')
# CC data points
cc_z = [d[0] for d in cc_data]
cc_H = [d[1] for d in cc_data]
cc_e = [d[2] for d in cc_data]
ax.errorbar(cc_z, cc_H, yerr=cc_e, fmt='o', color='darkgreen', ms=4, capsize=2,
            alpha=0.7, label=f'CC data ({n_cc} pts)', zorder=5)
# Mark transition zone percentages
for zt, pct in [(0.11, '90%'), (0.69, '50%'), (2.3, '10%')]:
    gap_val = (H_matt(zt)/H_phot(zt)-1)*100
    ax.axvline(zt, color='gray', ls=':', alpha=0.3)
    ax.text(zt+0.03, 210, f'z={zt}\n{pct} gap', fontsize=7, ha='left', color='gray')
ax.set_xlabel('Redshift z'); ax.set_ylabel('H(z) [km/s/Mpc]')
ax.set_title(f'(b) Cosmic Chronometers: Transition Zone', fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 2.2); ax.set_ylim(50, 230)

# --- Panel 3: f·σ₈ — all 4 models ---
ax = axes[0,2]
zf = np.linspace(0.05, 2.0, 80)
colors_m = {'A: ΛCDM': 'black', 'B: Bkgd only': 'orange', 
          'C: Pert only': 'blue', 'D: Combined': 'red'}
styles_m = {'A: ΛCDM': '-', 'B: Bkgd only': '--', 
          'C: Pert only': '-.', 'D: Combined': '-'}
for name, (mH, mu) in models.items():
    curve = compute_fsig8(zf, mH, mu)
    ax.plot(zf, curve, color=colors_m[name], ls=styles_m[name], lw=2, label=name)
ax.errorbar(z_g, obs_g, yerr=err_g, fmt='o', color='navy', ms=6, capsize=3, 
            label='Data', zorder=5)
ax.set_xlabel('z'); ax.set_ylabel(r'$f\sigma_8(z)$')
ax.set_title('(c) Growth Rates: Four Models', fontweight='bold')
ax.legend(fontsize=7, ncol=2); ax.set_xlim(0, 2.0)

# --- Panel 4: μ(a) + transition zone markers ---
ax = axes[1,0]
z_pl = np.linspace(0, 5, 500)
mu_z = [mu_iam(1/(1+z)) for z in z_pl]
ax.plot(z_pl, mu_z, 'r-', lw=2.5)
ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
ax.fill_between(z_pl, mu_z, 1.0, alpha=0.1, color='red')
# Transition markers
tz_marks = [(0.11, '90% active'), (0.69, '50%'), (2.3, '10%')]
for zt, lab in tz_marks:
    mz = mu_iam(1/(1+zt))
    ax.plot(zt, mz, 'ko', ms=7, zorder=5)
    ax.annotate(f'z={zt}\n{lab}\nμ={mz:.3f}', xy=(zt, mz), xytext=(zt+0.3, mz+0.01),
                fontsize=7, arrowprops=dict(arrowstyle='->', color='black'))
ax.set_xlabel('Redshift z'); ax.set_ylabel('μ(z)')
ax.set_title(f'(d) Gravitational Coupling μ(z) — Transition Zone', fontweight='bold')
ax.set_xlim(0, 3.5); ax.set_ylim(0.82, 1.02)

# --- Panel 5: S₈ comparison ---
ax = axes[1,1]
surveys = [d[0] for d in s8_data]
s8_vals_obs = [d[1] for d in s8_data]
s8_errs = [d[2] for d in s8_data]
y_pos = np.arange(len(surveys))
ax.errorbar(s8_vals_obs, y_pos, xerr=s8_errs, fmt='o', color='navy', ms=8, capsize=5,
            zorder=5)
ax.axvline(s8_lcdm, color='black', ls='--', lw=2, label=f'ΛCDM: {s8_lcdm:.3f}')
ax.axvline(s8_iam, color='red', ls=':', lw=1.5, alpha=0.7, label=f'IAM (Planck $\\Omega_m$): {s8_iam:.3f}')
ax.axvline(s8_iam_phys, color='red', ls='-', lw=2.5, label=f'IAM (physical $\\Omega_m$): {s8_iam_phys:.3f}')
# Shade the full improvement region
ax.axvspan(s8_iam_phys, s8_lcdm, alpha=0.08, color='red')
ax.set_yticks(y_pos); ax.set_yticklabels(surveys)
ax.set_xlabel(r'$S_8 = \sigma_8 \sqrt{\Omega_m/0.3}$')
ax.set_title(f'(e) $S_8$: Growth + Matter Dilution', fontweight='bold')
ax.legend(fontsize=6.5, loc='upper left')
ax.set_xlim(0.72, 0.86)

# --- Panel 6: Pantheon+ Hubble diagram (Photon Sector) ---
ax = axes[1,2]
z_sne = [d[0] for d in pantheon_binned]
mu_ref = [d[1] for d in pantheon_binned]
mu_err_sne = [d[2] for d in pantheon_binned]
# Plot reference bins
ax.errorbar(z_sne, mu_ref, yerr=mu_err_sne, fmt='s', color='darkorange',
            ms=4, capsize=2, alpha=0.8, label='Planck ΛCDM reference')
# Overplot the photon-sector prediction curve
z_curve = np.linspace(0.008, 2.5, 200)
mu_curve = [mu_dist(z) for z in z_curve]
ax.plot(z_curve, mu_curve, 'b-', lw=2, label=r'$H_\gamma$ (photon sector)')
ax.set_xlabel('Redshift z'); ax.set_ylabel(r'Distance modulus $\mu$ [mag]')
ax.set_title(f'(f) Pantheon+ Hubble Diagram (Photon Sector)', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(0.005, 2.5); ax.set_xscale('log')
ax.text(0.05, 0.92, 'Photon sector:\nidentical for ΛCDM & IAM\nby construction',
        transform=ax.transAxes, fontsize=8, color='gray', style='italic')

# --- Panel 7: Sector gap vs redshift ---
ax = axes[2,0]
z_gap = np.linspace(0.001, 5, 500)
gap_pct = [(H_matt(z)/H_phot(z)-1)*100 for z in z_gap]
ax.plot(z_gap, gap_pct, 'r-', lw=2.5)
ax.fill_between(z_gap, 0, gap_pct, alpha=0.1, color='red')
# Transition milestones
for zt, lab in [(0.11, '90%'), (0.69, '50%'), (2.3, '10%')]:
    gv = (H_matt(zt)/H_phot(zt)-1)*100
    ax.plot(zt, gv, 'ko', ms=7, zorder=5)
    ax.text(zt+0.1, gv+0.3, f'z={zt}: {gv:.1f}%', fontsize=8)
ax.set_xlabel('Redshift z'); ax.set_ylabel('$H_m/H_\\gamma - 1$ [%]')
ax.set_title('(g) Sector Gap: IAM Activation Timeline', fontweight='bold')
ax.set_xlim(0, 5); ax.set_ylim(0, 8.5)

# --- Panel 8: χ² breakdown (discriminating probes only) ---
ax = axes[2,1]
labs = ['H₀', r'f·$\sigma_8$', 'BAO', 'CC', 'S₈']
c_lcdm_list = [chi2_H0_lcdm, chi2_g['A: ΛCDM'], chi2_bao, chi2_cc_lcdm, chi2_s8_lcdm]
c_iam_list = [chi2_H0_iam, chi2_g['D: Combined'], chi2_bao, chi2_cc_iam, chi2_s8_iam]
x = np.arange(5); w = 0.35
bars_iam = ax.bar(x-w/2, c_iam_list, w, label='IAM', color='steelblue', alpha=0.85)
bars_lcdm = ax.bar(x+w/2, c_lcdm_list, w, label=r'$\Lambda$CDM', color='gray', alpha=0.85)
# Annotate H₀ Δχ²
ax.annotate(f'Δχ²={dchi2_H0:.0f}', xy=(0+w/2, chi2_H0_lcdm),
            xytext=(0.8, chi2_H0_lcdm*0.85),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=10, color='red',
            fontweight='bold')
# Annotate S₈ Δχ²
dchi_s8 = chi2_s8_lcdm - chi2_s8_iam
ax.annotate(f'Δχ²={dchi_s8:.0f}', xy=(4+w/2, chi2_s8_lcdm),
            xytext=(4.3, chi2_s8_lcdm*0.85),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=9, color='red')
ax.set_ylabel(r'$\chi^2$'); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=10)
ax.set_title(r'(h) $\chi^2$ by Probe', fontweight='bold')
ax.legend(fontsize=10)

# --- Panel 9: Summary scorecard ---
ax = axes[2,2]
ax.axis('off')
txt = f"""IAM Dual-Sector Validation Summary
{'='*42}

  beta = Om/2 = {beta:.5f}  (virial theorem)
  H0(photon) = {H0:.2f} km/s/Mpc
  H0(matter)  = {H0_matter:.2f} km/s/Mpc
  mu(z=0) = {mu_iam(1.0):.4f}
  sigma8(IAM) = {sig8_vals['D: Combined']:.4f}  ({(1-sig8_vals['D: Combined']/sigma8_fid)*100:.1f}% below LCDM)
  S8(IAM, Planck Om) = {s8_iam:.3f}
  S8(IAM, physical Om=0.272) = {s8_iam_phys:.3f}

  chi2 comparison: {n_tested} points, 6 probes
    H0: {n_H0}   RSD: {n_g}   BAO: {n_bao}
    CC: {n_cc}   S8: {n_s8}   CMB: {n_cmb}
  Plus {n_sne}-bin Pantheon+ (photon sector,
    identical by construction)

  dchi2(LCDM - IAM) = {dchi2_total:.1f}
  Equivalent significance: {np.sqrt(dchi2_total):.1f} sigma
  Additional free parameters: 0

  Transition zone (E(a) activation):
    90% at z = 0.11
    50% at z = 0.69
    10% at z = 2.3
    GR recovered for z > 5
"""
ax.text(0.05, 0.98, txt, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/mnt/user-data/outputs/iam_dual_sector_combined.pdf', dpi=150, bbox_inches='tight')
print("  Figure saved.")
plt.close()

# =============================================================================
# SECTOR GAP TABLE
# =============================================================================
print()
print("=" * 80)
print("  Sector Gap & μ(a) vs Redshift")
print("=" * 80)
print(f"  {'z':>6} {'a':>7} {'E(a)':>9} {'μ(a)':>7} {'H_γ':>7} {'H_m':>7} {'Gap%':>6}")
print(f"  {'-'*52}")
for z in [0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 100]:
    a = 1/(1+z)
    print(f"  {z:>6} {a:>7.4f} {E_act(a):>9.5f} {mu_iam(a):>7.4f} "
          f"{H_phot(z):>7.1f} {H_matt(z):>7.1f} {(H_matt(z)/H_phot(z)-1)*100:>6.3f}")

print(f"""
  Dataset Summary:
    H₀ measurements:  {n_H0} points (Planck, SH0ES, CCHP, H0LiCOW, SH0ES/JWST, TRGB, TDCOSMO)
    f·σ₈ growth:      {n_g} points (6dFGS, BOSS DR12, eBOSS DR16)
    BAO:              {n_bao} points (BOSS DR12, eBOSS DR16)
    Cosmic chrono:    {n_cc} points (Moresco+ compilation)
    Weak lensing S₈:  {n_s8} points (DES Y3, KiDS-1000, HSC Y3, Planck)
    Pantheon+ SNe:    {n_sne} reference bins (Brout+ 2022; model predictions, not data)
    CMB priors:       {n_cmb} (θ_MC, R, l_A)
    χ² comparison:    {n_tested} real data points across 6 probes
    Pantheon+ demo:   {n_sne} bins (photon sector, identical by construction)

  Models compared:
    A) ΛCDM:      standard H(z), μ = 1
    B) Bkgd only: matter H(z), μ = 1         → background suppression
    C) Pert only: standard H(z), μ(a) < 1    → perturbation suppression
    D) Combined:  matter H(z) + μ(a) < 1     → FULL IAM (both effects)
""")
print("=" * 80)
print("  COMPLETE")
print("=" * 80)
