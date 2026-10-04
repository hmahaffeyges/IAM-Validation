#!/usr/bin/env python3
"""
Extract Run C (LCDM baseline, mu0 = 0) posteriors from MCMC chains.
Reproduces Table 7 of the IAM-CAMB Technical Note.

Usage:
    python extract_run_c.py

Requires:
    - getdist (pip install getdist)
    - Chain files in iam_planck_chains/lcdm_baseline.*
"""

from getdist.mcsamples import loadMCSamples
import numpy as np

print("=" * 60)
print("  RUN C: LCDM BASELINE POSTERIORS")
print("=" * 60)

samples = loadMCSamples('iam_planck_chains/lcdm_baseline')
stats = samples.getMargeStats()

print()
print("  Sampled parameters:")
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

# Derived parameters
print('\n  Derived parameters:')
for par in ['omegal', 's8omegamp5']:
    s = stats.parWithName(par)
    if s is not None:
        print(f'    {par:>12s}: {s.mean:.4f} +/- {s.err:.4f}')

# Convergence
print(f'\n  Accepted samples: {len(samples.weights)}')
print(f'  Effective samples: {samples.norm:.0f}')

print("=" * 60)
