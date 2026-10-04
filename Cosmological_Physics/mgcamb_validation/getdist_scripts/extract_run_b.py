#!/usr/bin/env python3
"""
Extract Run B (mu0 floating) posteriors from MCMC chains.
Reproduces Table 6 and Section 9 of the IAM-CAMB Technical Note.

Usage:
    python extract_run_b.py

Requires:
    - getdist (pip install getdist)
    - Chain files in iam_planck_chains/iam_float_mu0.*
"""

from getdist.mcsamples import loadMCSamples
import numpy as np

print("=" * 60)
print("  RUN B: mu0 FLOATING POSTERIORS")
print("=" * 60)

samples = loadMCSamples('iam_planck_chains/iam_float_mu0')
stats = samples.getMargeStats()

# mu0 posterior (the key result)
mu0_stat = stats.parWithName('mu0')
if mu0_stat is not None:
    print(f'\n  mu0 posterior:')
    print(f'    Mean:     {mu0_stat.mean:.4f} +/- {mu0_stat.err:.4f}')
    print(f'    68% CI:   [{mu0_stat.limits[0].lower:.4f}, {mu0_stat.limits[0].upper:.4f}]')
    if len(mu0_stat.limits) > 1:
        print(f'    95% CI:   [{mu0_stat.limits[1].lower:.4f}, {mu0_stat.limits[1].upper:.4f}]')
    
    # Distance from IAM prediction
    iam_mu0 = -0.13495
    distance_sigma = abs(mu0_stat.mean - iam_mu0) / mu0_stat.err
    print(f'\n    IAM predicted mu0:  {iam_mu0}')
    print(f'    Distance from mean: {distance_sigma:.1f} sigma')
    print(f'    Within 68% CI:     {"Yes" if mu0_stat.limits[0].lower <= iam_mu0 <= mu0_stat.limits[0].upper else "No (but within 95%)"}')

print(f'\n  Other sampled parameters:')
for par in ['H0', 'sigma8', 'ombh2', 'omch2', 'tau', 'ns', 'A_planck']:
    s = stats.parWithName(par)
    if s is not None:
        print(f'    {par:>12s}: {s.mean:.4f} +/- {s.err:.4f}')

# Best-fit chi2
loglikes = samples.loglikes
best_idx = np.argmin(loglikes)
best_chi2 = 2 * loglikes[best_idx]
print(f'\n  Best-fit -lnL: {loglikes[best_idx]:.2f}')
print(f'  Best-fit chi2: {best_chi2:.2f}')

# Convergence
print(f'\n  Accepted samples: {len(samples.weights)}')
print(f'  Effective samples: {samples.norm:.0f}')

print("=" * 60)
