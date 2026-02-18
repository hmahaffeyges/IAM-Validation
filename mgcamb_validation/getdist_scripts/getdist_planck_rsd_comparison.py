#!/usr/bin/env python3
"""
GetDist extraction and three-way comparison for Planck + RSD runs (D/E/F).

Usage:
    cd mgcamb_validation/getdist_scripts/
    python getdist_planck_rsd_comparison.py

Requires:
    - GetDist (pip install getdist)
    - Chain files in ../chains/ (planck_rsd_iam_fixed, planck_rsd_mu0_float, planck_rsd_lcdm_baseline)

Outputs:
    - Parameter summary tables for each run
    - Three-way chi2 comparison
    - Matches Table 2 in observational paper and Section 6 of Technical Note
"""

import numpy as np

try:
    from getdist import loadMCSamples
except ImportError:
    print("ERROR: GetDist not installed. Run: pip install getdist")
    exit(1)

# ============================================================================
# Configuration
# ============================================================================

CHAIN_DIR = "../chains"
IAM_MU0_PREDICTED = -0.13495

RUNS = {
    "D": {
        "name": "IAM fixed (Planck + RSD)",
        "prefix": f"{CHAIN_DIR}/planck_rsd_iam_fixed",
        "mu0_fixed": True,
        "mu0_value": IAM_MU0_PREDICTED,
    },
    "E": {
        "name": "mu0 floating (Planck + RSD)",
        "prefix": f"{CHAIN_DIR}/planck_rsd_mu0_float",
        "mu0_fixed": False,
    },
    "F": {
        "name": "LCDM baseline (Planck + RSD)",
        "prefix": f"{CHAIN_DIR}/planck_rsd_lcdm_baseline",
        "mu0_fixed": True,
        "mu0_value": 0.0,
    },
}

# Parameters to extract
PARAMS = [
    "H0", "sigma8", "omegabh2", "omegach2", "omegal", "ns",
    "logA", "tau",
]

# ============================================================================
# Extract posteriors
# ============================================================================

print("=" * 80)
print("PLANCK + RSD: THREE-WAY COMPARISON (Runs D / E / F)")
print("=" * 80)

results = {}

for run_id, run_info in RUNS.items():
    print(f"\n{'─' * 60}")
    print(f"Run {run_id}: {run_info['name']}")
    print(f"{'─' * 60}")

    try:
        samples = loadMCSamples(run_info["prefix"])
    except Exception as e:
        print(f"  ERROR loading chains: {e}")
        print(f"  Expected: {run_info['prefix']}.1.txt")
        continue

    # Basic chain info
    stats = samples.getMargeStats()
    n_samples = samples.numrows
    print(f"  Samples: {n_samples}")

    # Extract parameters
    run_results = {"n_samples": n_samples}

    for param in PARAMS:
        try:
            p = stats.parWithName(param)
            if p is not None:
                mean = p.mean
                err = p.err
                print(f"  {param:12s} = {mean:.4f} +/- {err:.4f}")
                run_results[param] = (mean, err)
        except Exception:
            pass

    # mu0 if floating
    if not run_info["mu0_fixed"]:
        try:
            p = stats.parWithName("mu0")
            if p is not None:
                print(f"  {'mu0':12s} = {p.mean:.4f} +/- {p.err:.4f}")
                run_results["mu0"] = (p.mean, p.err)

                # Distance from IAM prediction
                dist = abs(p.mean - IAM_MU0_PREDICTED) / p.err
                print(f"  Distance from IAM prediction: {dist:.1f} sigma")
                run_results["mu0_distance_sigma"] = dist
        except Exception:
            pass

    # Best-fit chi2
    try:
        best_chi2 = -2 * samples.getLikeStats().logLike_best
        print(f"  Best chi2  = {best_chi2:.2f}")
        run_results["chi2"] = best_chi2
    except Exception:
        # Try from minimum log-likelihood in chain
        try:
            loglikes = samples.loglikes
            best_chi2 = 2 * np.min(loglikes)
            print(f"  Best chi2  = {best_chi2:.2f} (from chain min)")
            run_results["chi2"] = best_chi2
        except Exception:
            print("  WARNING: Could not extract chi2")

    # Convergence (R-1)
    try:
        from getdist import chains
        print(f"  R-1        = (check .progress file)")
    except Exception:
        pass

    results[run_id] = run_results

# ============================================================================
# Three-way comparison
# ============================================================================

print(f"\n{'=' * 80}")
print("THREE-WAY COMPARISON TABLE")
print(f"{'=' * 80}")

if "F" in results and "chi2" in results.get("F", {}):
    lcdm_chi2 = results["F"]["chi2"]

    print(f"\n{'Run':<8} {'mu0':<20} {'sigma8':<18} {'chi2':<12} {'Delta-chi2':<12} {'Interpretation'}")
    print("─" * 80)

    for run_id in ["F", "D", "E"]:
        if run_id not in results:
            continue
        r = results[run_id]

        # mu0 string
        if run_id == "F":
            mu0_str = "0 (fixed)"
        elif run_id == "D":
            mu0_str = f"{IAM_MU0_PREDICTED:.3f} (fixed)"
        else:
            mu0_mean, mu0_err = r.get("mu0", (0, 0))
            mu0_str = f"{mu0_mean:.3f} +/- {mu0_err:.3f}"

        # sigma8 string
        s8_mean, s8_err = r.get("sigma8", (0, 0))
        s8_str = f"{s8_mean:.4f} +/- {s8_err:.4f}"

        # chi2
        chi2 = r.get("chi2", 0)
        dchi2 = chi2 - lcdm_chi2

        # Interpretation
        if run_id == "F":
            interp = "Baseline"
        elif abs(dchi2) < 2:
            interp = "Compatible"
        else:
            interp = "Tension" if dchi2 > 4 else "Marginal"

        print(f"{run_id:<8} {mu0_str:<20} {s8_str:<18} {chi2:<12.2f} {dchi2:<+12.2f} {interp}")

    print()
    print(f"LCDM baseline chi2 = {lcdm_chi2:.2f}")
    if "D" in results and "chi2" in results["D"]:
        print(f"IAM fixed Delta-chi2 = {results['D']['chi2'] - lcdm_chi2:+.2f}")
    if "E" in results and "chi2" in results["E"]:
        print(f"IAM float Delta-chi2 = {results['E']['chi2'] - lcdm_chi2:+.2f}")

# ============================================================================
# sigma8 shift summary
# ============================================================================

print(f"\n{'=' * 80}")
print("SIGMA_8 SHIFT SUMMARY")
print(f"{'=' * 80}")

if "F" in results and "D" in results:
    s8_lcdm = results["F"].get("sigma8", (0, 0))[0]
    s8_iam = results["D"].get("sigma8", (0, 0))[0]
    shift = s8_iam - s8_lcdm
    pct = 100 * shift / s8_lcdm if s8_lcdm > 0 else 0
    print(f"  LCDM (Run F):     sigma8 = {s8_lcdm:.4f}")
    print(f"  IAM fixed (Run D): sigma8 = {s8_iam:.4f}")
    print(f"  Shift:             {shift:+.4f} ({pct:+.1f}%)")
    print(f"  Direction: {'Toward weak lensing (correct)' if shift < 0 else 'Away from weak lensing'}")

print()
print("Script complete. Compare with Technical Note Section 6 and observational paper Table 2.")
