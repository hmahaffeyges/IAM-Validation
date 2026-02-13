#!/usr/bin/env python3
"""
TEST A: Pantheon+ SNe with PLANCK (Photon) H₀ Prior
Tests if SNe prefer photon-sector expansion (β_m → 0)
"""

import numpy as np
from scipy.optimize import minimize
import time

print("="*80)
print("TEST A: PANTHEON+ WITH PLANCK (PHOTON) H₀ PRIOR")
print("="*80)
print()

# Load data
data_file = '/Users/hmahaffeyges/Desktop/IAM-Validation/data/pantheon_repo/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'

data = []
with open(data_file, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        zCMB = float(parts[4])
        m_b = float(parts[8])
        m_b_err = float(parts[9])
        if 0.01 < zCMB < 2.5:
            data.append([zCMB, m_b, m_b_err])

data = np.array(data)
z_sne = data[:, 0]
mb_obs = data[:, 1]
dmb_obs = data[:, 2]

print(f"Loaded: {len(z_sne)} SNe")
print()

# Parameters
Om0_fid = 0.315
H0_PLANCK = 67.4
H0_PLANCK_err = 0.5
c_km_s = 299792.458

# Functions
def activation(a):
    return np.exp(1.0 - 1.0/a)

def H_IAM(z, Om0, H0, beta_m):
    a = 1.0 / (1.0 + z)
    OmL = 1.0 - Om0
    E_a = activation(a)
    H_squared = Om0 * a**-3 + OmL + beta_m * E_a
    return H0 * np.sqrt(H_squared)

def dL_IAM(z, Om0, H0, beta_m):
    if z < 1e-6:
        return 1e-10
    z_arr = np.linspace(0, z, 500)
    H_arr = H_IAM(z_arr, Om0, H0, beta_m)
    integrand = c_km_s / H_arr
    d_C = np.trapezoid(integrand, z_arr)
    return (1 + z) * d_C

def mu_IAM(z, Om0, H0, beta_m, M):
    dL = dL_IAM(z, Om0, H0, beta_m)
    return M + 5.0 * np.log10(dL) + 25.0

def chi2_IAM(params):
    Om0, H0, beta_m, M = params
    if not (0.2 < Om0 < 0.4):
        return 1e10
    if not (60.0 < H0 < 75.0):
        return 1e10
    if not (-0.3 < beta_m < 0.3):
        return 1e10
    if not (-20.0 < M < -18.0):
        return 1e10
    
    mu_model = np.array([mu_IAM(z, Om0, H0, beta_m, M) for z in z_sne])
    chi2_sne = np.sum(((mb_obs - mu_model) / dmb_obs)**2)
    
    # PLANCK H₀ PRIOR (photon sector)
    chi2_H0 = ((H0 - H0_PLANCK) / H0_PLANCK_err)**2
    
    return chi2_sne + chi2_H0

print("Fitting with Planck H₀ = 67.4 ± 0.5 km/s/Mpc...")
t0 = time.time()

x0 = [Om0_fid, H0_PLANCK, 0.0, -19.3]
result = minimize(chi2_IAM, x0, method='Nelder-Mead',
                 options={'maxiter': 5000, 'disp': False})

Om0, H0, beta_m, M = result.x
chi2 = result.fun
t1 = time.time()

H0_matter = H0 * np.sqrt(1 + beta_m)

print()
print("="*80)
print("TEST A RESULTS: PLANCK PRIOR")
print("="*80)
print(f"  Ωₘ        = {Om0:.4f}")
print(f"  H₀_base   = {H0:.2f} km/s/Mpc")
print(f"  β_m       = {beta_m:+.4f}")
print(f"  M         = {M:.4f}")
print(f"  χ²        = {chi2:.2f}")
print(f"  χ²/dof    = {chi2/(len(z_sne)-4):.4f}")
print()
print(f"  H₀(matter) = {H0_matter:.2f} km/s/Mpc")
print(f"  Time       = {t1-t0:.1f} sec")
print()

if abs(beta_m) < 0.05:
    print("✓ β_m ≈ 0: SNe prefer PHOTON-SECTOR behavior")
elif beta_m > 0.1:
    print("✓ β_m > 0: SNe prefer MATTER-SECTOR behavior")
elif beta_m < -0.1:
    print("⚠ β_m < 0: Unexpected (would lower H₀)")
print()
