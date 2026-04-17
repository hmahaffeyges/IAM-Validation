#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
# GAPE Bootstrap Comparison — Independent Validation of G-003b MCMC Posteriors
# ════════════════════════════════════════════════════════════════════════════
#
# Purpose: Cross-check G-003b MCMC H_min posteriors against a simpler,
# assumption-free bootstrap estimator using IDENTICAL reference cell data.
#
# Why this exists:
# G-003b used Bayesian MCMC (emcee, 5 chains × 32 walkers × 5,500 steps) to
# infer H_min posteriors for 4 substrates × 8 architecture classes = 32 values.
# A reasonable referee question is: "Is MCMC necessary, or would a simpler
# non-parametric approach give the same answer?"
#
# This script runs 10,000 bootstrap resamples of the same reference cells and
# reports mean + 95% CI per (substrate, class). If MCMC posterior means fall
# within bootstrap 95% CIs, both methods agree — MCMC is not introducing
# a methodology-dependent artifact.
#
# Methodology:
#   1. Import reference cell databases from gape_mcmc_g003b (zero data duplication)
#   2. For each (substrate, class), extract the measured values from reference cells
#   3. Compute per-cell Shannon binary entropy H(β)
#   4. Bootstrap 10,000 resamples of the class members
#   5. Compute mean H per resample → bootstrap distribution for H_min
#   6. Report bootstrap mean, SD, 2.5%-97.5% CI, agreement with MCMC posterior
#
# No free parameters. No priors. No convergence diagnostics required.
# Just the reference cells and resampling.
#
# Output: bootstrap_vs_mcmc_comparison.tsv (machine-readable)
#         plus printed table matching G-003b posterior format
#
# Reference: Efron & Tibshirani 1993, "An Introduction to the Bootstrap"
# Cite as: Mahaffey HW (2026). GAPE bootstrap cross-check.
#          Zenodo: doi:10.5281/zenodo.19547624
# ════════════════════════════════════════════════════════════════════════════

import math
import numpy as np
import sys
import os

# ── Add evidence folder to path so we can import the MCMC reference data ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = SCRIPT_DIR if os.path.basename(SCRIPT_DIR) == 'evidence' \
    else os.path.join(os.path.dirname(SCRIPT_DIR), 'evidence')
sys.path.insert(0, EVIDENCE_DIR)

# ── Shannon binary entropy (same as G-003b) ──────────────────────────────────
def H(b):
    """Shannon binary entropy in bits. H(0) = H(1) = 0. H(0.5) = 1."""
    if b <= 0 or b >= 1:
        return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)

# ── Try to import G-003b reference databases ─────────────────────────────────
try:
    from gape_mcmc_g003b import DB_NUCL, DB_FUZZ, DB_WPS, DB_FRAG
    SOURCE = "imported from gape_mcmc_g003b.py"
except ImportError:
    print("ERROR: Could not import reference databases from gape_mcmc_g003b.py")
    print(f"  Looked in: {EVIDENCE_DIR}")
    print("  Please run this script from the same directory or from")
    print("  IAM-Validation/Biological_Physics/evidence/")
    sys.exit(1)

# ── G-003b MCMC posteriors (for comparison) ──────────────────────────────────
# From gape_mcmc_g003b run April 16, 2026. R-hat < 1.001 across all chains.
MCMC_POSTERIORS = {
    'nucl': {
        'stem_pluri': (0.799818, 0.009230),
        'stem_adult': (0.960866, 0.011131),
        'progenitor': (0.972790, 0.011009),
        'terminal':   (0.992027, 0.005948),
        'cycling':    (0.980072, 0.008427),
        'immune':     (0.989930, 0.006463),
        'secretory':  (0.982560, 0.009638),
        'stromal':    (0.985667, 0.008815),
    },
    'fuzz': {
        'stem_pluri': (0.962920, 0.011135),
        'stem_adult': (0.980754, 0.009944),
        'progenitor': (0.961900, 0.011166),
        'terminal':   (0.736973, 0.007371),
        'cycling':    (0.819030, 0.007359),
        'immune':     (0.830377, 0.008299),
        'secretory':  (0.847947, 0.009769),
        'stromal':    (0.832386, 0.009645),
    },
    'wps': {
        'stem_pluri': (0.905004, 0.012671),
        'stem_adult': (0.988964, 0.008174),
        'progenitor': (0.988046, 0.008611),
        'terminal':   (0.958909, 0.011203),
        'cycling':    (0.627429, 0.005649),
        'immune':     (0.589644, 0.006792),
        'secretory':  (0.634534, 0.008996),
        'stromal':    (0.612686, 0.008810),
    },
    'frag': {
        'stem_pluri': (0.973583, 0.015681),
        'stem_adult': (0.841327, 0.011784),
        'progenitor': (0.808978, 0.016338),
        'terminal':   (0.624938, 0.007288),
        'cycling':    (0.687936, 0.006878),
        'immune':     (0.711534, 0.007067),
        'secretory':  (0.697718, 0.009890),
        'stromal':    (0.724691, 0.014423),
    },
}

# ── Bootstrap helper ─────────────────────────────────────────────────────────
def bootstrap_h_min(beta_values, n_boot=10000, seed=42):
    """
    Bootstrap confidence interval for H_min estimated from a set of reference
    cell β values belonging to one architecture class.

    Parameters
    ----------
    beta_values : array-like of float
        Reference cell measurements (β for methylation, occupancy for nucl, etc.)
    n_boot : int
        Number of bootstrap resamples (default 10000 — standard)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict with keys:
        mean: bootstrap mean of H(β) over resamples
        std:  bootstrap SD
        ci_lower: 2.5th percentile
        ci_upper: 97.5th percentile
        n: number of reference cells
        n_boot: number of bootstrap iterations
    """
    rng = np.random.default_rng(seed)
    arr = np.array(beta_values, dtype=float)
    n = len(arr)

    if n == 0:
        return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan,
                'ci_upper': np.nan, 'n': 0, 'n_boot': 0}

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        resample = rng.choice(arr, size=n, replace=True)
        # Compute per-cell H, then mean H — same as G-003b likelihood
        h_values = np.array([H(b) for b in resample])
        boot_means[i] = np.mean(h_values)

    return {
        'mean': float(np.mean(boot_means)),
        'std':  float(np.std(boot_means, ddof=1)),
        'ci_lower': float(np.percentile(boot_means, 2.5)),
        'ci_upper': float(np.percentile(boot_means, 97.5)),
        'n': n,
        'n_boot': n_boot,
    }

# ── Main bootstrap pass ──────────────────────────────────────────────────────
SUBSTRATES = [
    ('nucl', 'Nucleosome occupancy', DB_NUCL),
    ('fuzz', 'Nucleosome fuzziness', DB_FUZZ),
    ('wps',  'Windowed protection score', DB_WPS),
    ('frag', 'Fragment size (DELFI)', DB_FRAG),
]

CLASSES = ['stem_pluri', 'stem_adult', 'progenitor', 'terminal',
           'cycling',    'immune',     'secretory',  'stromal']

print("="*88)
print("GAPE Bootstrap Comparison — Cross-check of G-003b MCMC Posteriors")
print("="*88)
print(f"Methodology: 10,000 bootstrap resamples per (substrate, class)")
print(f"Reference data: {SOURCE}")
print(f"Comparison: MCMC posterior mean vs bootstrap mean, within 95% CI?")
print("="*88)

all_results = {}

for sub_key, sub_name, db in SUBSTRATES:
    print(f"\n{'─'*88}")
    print(f"SUBSTRATE: {sub_name}")
    print(f"{'─'*88}")
    print(f"\n  Reference cells: {len(db)} across {len(CLASSES)} architecture classes")
    print(f"\n  {'Class':<13} {'n':>3} {'Bootstrap mean':>15} {'SD':>9} "
          f"{'2.5%':>9} {'97.5%':>9} {'MCMC mean':>11} "
          f"{'MCMC σ':>8} {'MCMC in CI?':>12}")
    print(f"  {'─'*92}")

    sub_results = {}
    for cls in CLASSES:
        # Extract β values for this class
        beta_vals = [row[2] for row in db if row[1] == cls]

        if len(beta_vals) == 0:
            print(f"  {cls:<13} {'0':>3} {'(no reference cells)':>45}")
            continue

        # Bootstrap
        bs = bootstrap_h_min(beta_vals, n_boot=10000)

        # MCMC for comparison
        mcmc_mean, mcmc_std = MCMC_POSTERIORS[sub_key][cls]
        in_ci = bs['ci_lower'] <= mcmc_mean <= bs['ci_upper']

        mark = "✓" if in_ci else "✗"

        print(f"  {cls:<13} {bs['n']:>3} "
              f"{bs['mean']:>15.6f} {bs['std']:>9.6f} "
              f"{bs['ci_lower']:>9.6f} {bs['ci_upper']:>9.6f} "
              f"{mcmc_mean:>11.6f} {mcmc_std:>8.6f} "
              f"{mark:>12}")

        sub_results[cls] = {
            'n': bs['n'],
            'bootstrap_mean': bs['mean'],
            'bootstrap_std':  bs['std'],
            'bootstrap_ci_lower': bs['ci_lower'],
            'bootstrap_ci_upper': bs['ci_upper'],
            'mcmc_mean': mcmc_mean,
            'mcmc_std':  mcmc_std,
            'mcmc_in_bootstrap_ci': in_ci,
            'abs_difference': abs(mcmc_mean - bs['mean']),
            'rel_difference_pct': 100.0 * abs(mcmc_mean - bs['mean']) /
                                       bs['mean'] if bs['mean'] > 0 else 0.0,
        }

    all_results[sub_key] = sub_results

# ── Agreement summary ────────────────────────────────────────────────────────
print(f"\n{'='*88}")
print("AGREEMENT SUMMARY")
print(f"{'='*88}")

total_cells = 0
in_ci_cells = 0
sum_rel_diff = 0.0
max_rel_diff = 0.0
max_case = None

for sub_key, sub_name, _ in SUBSTRATES:
    sub_in_ci = sum(1 for cls_data in all_results[sub_key].values()
                    if cls_data['mcmc_in_bootstrap_ci'])
    sub_total = len(all_results[sub_key])
    print(f"  {sub_name:<30} {sub_in_ci}/{sub_total} classes — "
          f"MCMC posterior within bootstrap 95% CI")
    total_cells += sub_total
    in_ci_cells += sub_in_ci
    for cls, d in all_results[sub_key].items():
        sum_rel_diff += d['rel_difference_pct']
        if d['rel_difference_pct'] > max_rel_diff:
            max_rel_diff = d['rel_difference_pct']
            max_case = (sub_name, cls, d['mcmc_mean'], d['bootstrap_mean'])

print(f"\n  OVERALL: {in_ci_cells}/{total_cells} posteriors within bootstrap 95% CI")
print(f"  Mean relative difference (MCMC vs bootstrap): "
      f"{sum_rel_diff/total_cells:.3f}%")
print(f"  Max relative difference: {max_rel_diff:.3f}% "
      f"({max_case[0]} / {max_case[1]}: MCMC={max_case[2]:.6f}, "
      f"bootstrap={max_case[3]:.6f})")

# ── Write TSV for evidence deposit ───────────────────────────────────────────
output_path = os.path.join(SCRIPT_DIR, 'bootstrap_vs_mcmc_comparison.tsv')
with open(output_path, 'w') as f:
    f.write("substrate\tclass\tn_cells\tbootstrap_mean\tbootstrap_std\t"
            "bootstrap_ci_lower\tbootstrap_ci_upper\tmcmc_mean\tmcmc_std\t"
            "mcmc_in_bootstrap_ci\tabs_difference\trel_difference_pct\n")
    for sub_key, sub_name, _ in SUBSTRATES:
        for cls, d in all_results[sub_key].items():
            f.write(f"{sub_key}\t{cls}\t{d['n']}\t"
                    f"{d['bootstrap_mean']:.6f}\t{d['bootstrap_std']:.6f}\t"
                    f"{d['bootstrap_ci_lower']:.6f}\t{d['bootstrap_ci_upper']:.6f}\t"
                    f"{d['mcmc_mean']:.6f}\t{d['mcmc_std']:.6f}\t"
                    f"{d['mcmc_in_bootstrap_ci']}\t"
                    f"{d['abs_difference']:.6f}\t"
                    f"{d['rel_difference_pct']:.3f}\n")

print(f"\n  TSV output: {output_path}")

# ── Interpretation ───────────────────────────────────────────────────────────
print(f"\n{'='*88}")
print("INTERPRETATION")
print(f"{'='*88}")
print("""
  The bootstrap and MCMC should agree when:
    • The reference data is the primary driver of the H_min estimate
    • The MCMC prior is not overwhelming the likelihood
    • The likelihood surface is smooth and well-approximated by its mean
    • No prior-only parameters are constrained (no degeneracies)

  Disagreement would indicate:
    • MCMC prior is pulling posteriors toward prior center (prior-dominated)
    • Bootstrap resampling misses joint structure MCMC captures
    • Reference data has outliers the bootstrap amplifies

  When MCMC posteriors fall INSIDE bootstrap 95% CIs:
    → Both methods estimate the same parameter from the same data
    → MCMC's marginal answer is not methodology-dependent
    → Bootstrap provides a simpler sanity check anyone can reproduce

  Typical good agreement: MCMC mean within bootstrap CI for >90% of classes,
  with mean relative difference <5% and max <10%.
""")
print(f"{'='*88}")
print(f"DONE — paste full output to Walther")
print(f"{'='*88}")
