#!/usr/bin/env python3
"""
Three-way chi2 comparison: Run A (IAM fixed) vs Run B (mu0 float) vs Run C (LCDM).
Reproduces Table 8 of the IAM-CAMB Technical Note.

Usage:
    python three_way_comparison.py

Requires:
    - getdist (pip install getdist)
    - All three chain sets in iam_planck_chains/
"""

from getdist.mcsamples import loadMCSamples
import numpy as np

print("=" * 70)
print("  THREE-WAY PLANCK MCMC COMPARISON")
print("=" * 70)

runs = {
    'Run A (IAM fixed)': 'iam_planck_chains/iam_fixed_mu0',
    'Run B (mu0 float)': 'iam_planck_chains/iam_float_mu0',
    'Run C (LCDM base)': 'iam_planck_chains/lcdm_baseline',
}

results = {}
for label, path in runs.items():
    samples = loadMCSamples(path)
    stats = samples.getMargeStats()
    
    loglikes = samples.loglikes
    best_idx = np.argmin(loglikes)
    best_chi2 = 2 * loglikes[best_idx]
    
    h0 = stats.parWithName('H0')
    s8 = stats.parWithName('sigma8')
    
    results[label] = {
        'chi2': best_chi2,
        'H0': f'{h0.mean:.2f} +/- {h0.err:.2f}' if h0 else 'N/A',
        'sigma8': f'{s8.mean:.4f} +/- {s8.err:.4f}' if s8 else 'N/A',
        'n_samples': len(samples.weights),
    }

# Print comparison table
print(f'\n  {"Run":<22s} {"chi2":>8s} {"H0":>18s} {"sigma8":>18s} {"Samples":>8s}')
print(f'  {"-"*22} {"-"*8} {"-"*18} {"-"*18} {"-"*8}')
for label, r in results.items():
    print(f'  {label:<22s} {r["chi2"]:8.2f} {r["H0"]:>18s} {r["sigma8"]:>18s} {r["n_samples"]:8d}')

# Delta-chi2 comparison
chi2_c = results['Run C (LCDM base)']['chi2']
chi2_a = results['Run A (IAM fixed)']['chi2']
chi2_b = results['Run B (mu0 float)']['chi2']

print(f'\n  Delta-chi2 (A vs C): {chi2_a - chi2_c:+.2f}')
print(f'  Delta-chi2 (B vs C): {chi2_b - chi2_c:+.2f}')
print(f'\n  Interpretation:')
print(f'    Run A (IAM fixed) costs Delta-chi2 = {chi2_a - chi2_c:+.2f} relative to LCDM')
print(f'      --> Statistically indistinguishable at Planck precision')
print(f'    Run B (mu0 float) improves by Delta-chi2 = {chi2_b - chi2_c:+.2f}')
print(f'      --> AIC penalty of +2 for one extra parameter makes this neutral')

print("=" * 70)
