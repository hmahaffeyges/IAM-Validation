#!/usr/bin/env python3
"""
Extract Run A (IAM fixed mu0 = -0.13495) posteriors from MCMC chains.
Reproduces Table 5 of the IAM-CAMB Technical Note.

Usage:
    python extract_run_a.py

Requires:
    - getdist (pip install getdist)
    - Chain files in iam_planck_chains/iam_fixed_mu0.*
"""

from getdist.mcsamples import loadMCSamples
import numpy as np

print("=" * 60)
print("  RUN A: IAM FIXED (mu0 = -0.13495) POSTERIORS")
print("=" * 60)

samples = loadMCSamples('iam_planck_chains/iam_fixed_mu0')
stats = samples.getMargeStats()

print()
print("  Sampled parameters:")
for par in ['H0', 'sigma8', 'ombh2', 'omch2', 'tau', 'ns', 'A_planck']:
    s = stats.parWithName(par)
    if s is not None:
        print(f'    {par:>12s}: {s.mean:.4f} +/- {s.err:.4f}  '
              f'({s.limits[0].lower:.4f} to {s.limits[0].upper:.4f} 68%)')

# Best-fit chi2
loglikes = samples.loglikes
best_idx = np.argmin(loglikes)
best_chi2 = 2 * loglikes[best_idx]
print(f'\n  Best-fit -lnL: {loglikes[best_idx]:.2f}')
print(f'  Best-fit chi2: {best_chi2:.2f}')

# Derived parameters
print('\n  Derived parameters:')
for par in ['omegal', 's8omegamp5']:
    s = stats.parWithName(par)
    if s is not None:
        print(f'    {par:>12s}: {s.mean:.4f} +/- {s.err:.4f}')

# Convergence
print(f'\n  Accepted samples: {len(samples.weights)}')
print(f'  Effective samples: {samples.norm:.0f}')

print()
print("  Comparison points:")
print("    Planck 2018 LCDM:  H0 = 67.36, sigma8 = 0.811, S8 = 0.832")
print("    IAM prediction:    H0(matter) = 72.51, sigma8 ~ 0.80")
print("    mu0 fixed at:      -0.13495 (derived from beta_m = Omega_m/2)")
print("=" * 60)
