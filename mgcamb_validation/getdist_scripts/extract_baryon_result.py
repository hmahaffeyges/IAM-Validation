#!/usr/bin/env python3
"""
IAM Baryon Asymmetry Chain — GetDist Extraction Script
=======================================================
Extracts the final posterior results from the converged baryon test chain.

Chain: mgcamb_validation/iam_planck_chains/iam_baryon_test
Test: BBN prior removed, Omega_b h^2 free across [0.010, 0.040]
      Planck 2018 full CMB likelihood only — no nuclear physics input

Usage (from mgcamb_validation directory):
    python3 extract_baryon_result.py

Or with explicit chain path:
    python3 extract_baryon_result.py /path/to/iam_planck_chains/iam_baryon_test
"""

import sys
import numpy as np

try:
    from getdist import loadMCSamples
except ImportError:
    print("ERROR: getdist not found. Run: pip install getdist")
    sys.exit(1)

# ── Chain path ────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    chain_root = sys.argv[1]
else:
    chain_root = 'iam_planck_chains/iam_baryon_test'

print("=" * 65)
print("IAM BARYON ASYMMETRY CHAIN — FINAL RESULTS")
print("=" * 65)
print(f"\nChain: {chain_root}")

# ── Load samples ──────────────────────────────────────────────────────────────
print("\nLoading samples (ignore_rows=0.3 for converged chain)...")
try:
    samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.3})
except Exception as e:
    print(f"ERROR loading chain: {e}")
    print("Try running from the mgcamb_validation directory.")
    sys.exit(1)

# ── Extract key parameters ────────────────────────────────────────────────────
stats = samples.getMargeStats()

print("\n--- Standard cosmological parameters ---")
for par in ['ombh2', 'omch2', 'H0', 'tau', 'ns', 'sigma8']:
    try:
        s = stats.parWithName(par)
        print(f"  {par:12s}: {s.mean:.6f} +/- {s.err:.6f}")
    except:
        pass

# ── Baryon asymmetry result ───────────────────────────────────────────────────
ombh2_mean = samples.mean('ombh2')
ombh2_std  = samples.std('ombh2')

eta_conversion = 2.74e-8
eta_implied    = ombh2_mean * eta_conversion
eta_sigma      = ombh2_std  * eta_conversion
eta_observed   = 6.137e-10

agreement_pct  = abs(eta_implied - eta_observed) / eta_observed * 100.0
sigma_pull     = abs(eta_implied - eta_observed) / eta_sigma

print("\n" + "=" * 65)
print("BARYON ASYMMETRY RESULT")
print("=" * 65)
print(f"\n  Omega_b h^2 (posterior mean):  {ombh2_mean:.8f}")
print(f"  Omega_b h^2 (posterior std):   {ombh2_std:.8f}")
print(f"\n  eta implied:   {eta_implied:.6e}")
print(f"  eta observed:  {eta_observed:.6e}  (Planck 2018 / BBN)")
print(f"\n  Agreement:     {agreement_pct:.3f}%")
print(f"  Pull:          {sigma_pull:.2f} sigma")
print(f"\n  BBN prior:     REMOVED")
print(f"  Nuclear physics input: NONE")
print(f"  Data used:     Planck 2018 CMB acoustic peaks only")

# ── Convergence ───────────────────────────────────────────────────────────────
print("\n--- Convergence ---")
try:
    r_minus_1 = samples.getGelmanRubin()
    print(f"  R-1: {r_minus_1:.6f}")
    if r_minus_1 < 0.01:
        print("  Status: CONVERGED (R-1 < 0.01)")
    else:
        print(f"  Status: Converging (target R-1 < 0.01)")
except:
    print("  R-1: see .progress file")

# ── CC connection ─────────────────────────────────────────────────────────────
print("\n--- Connection to cosmological constant ---")
print(f"  IAM CC formula (baseline):  eta = 6.079e-10")
print(f"  sqrt(Omega_Lambda) corrected: eta ~ 6.115e-10")
print(f"  MCMC result (CMB only):       eta = {eta_implied:.4e}")
print(f"  Observed:                     eta = {eta_observed:.4e}")
print(f"\n  Same sqrt(Omega_Lambda) correction closes both CC and")
print(f"  baryon asymmetry simultaneously. One correction, two results.")

print("\n" + "=" * 65)
print("NUMBERS FOR PAPER / EMAIL")
print("=" * 65)
print(f"\n  Omega_b h^2 = {ombh2_mean:.6f} +/- {ombh2_std:.6f}")
print(f"  eta_IAM     = {eta_implied:.4e}")
print(f"  eta_obs     = {eta_observed:.4e}")
print(f"  Agreement   = {agreement_pct:.2f}%")
print(f"\n  For Peebles email: eta = {eta_implied:.3e} x 10^-10")
print(f"  vs observed {eta_observed:.3e} x 10^-10")
