#!/bin/bash
# =============================================================================
# IAM Virial Scaling Test: β_m = Ω_m/2
# =============================================================================
# Runs MCMC with three different Ω_m values to verify that
# the fitted β_m tracks Ω_m/2 (virial theorem prediction).
#
# Expected runtime: ~6 minutes total (3 × ~2 min each)
# =============================================================================

echo "================================================================="
echo "  IAM VIRIAL SCALING TEST: β_m = Ω_m/2"
echo "  Testing with Ω_m = 0.30, 0.315, 0.33"
echo "================================================================="
echo ""

echo "[1/3] Running Ω_m = 0.30..."
python3 mcmc_omega_scaling_low.py 2>&1 | tee virial_test_low.log
echo ""

echo "[2/3] Running Ω_m = 0.315..."
python3 mcmc_omega_scaling_mid.py 2>&1 | tee virial_test_mid.log
echo ""

echo "[3/3] Running Ω_m = 0.33..."
python3 mcmc_omega_scaling_high.py 2>&1 | tee virial_test_high.log
echo ""

echo "================================================================="
echo "  ALL RUNS COMPLETE - Check logs for virial scaling results"
echo "================================================================="
echo ""
echo "Summary: grep for 'beta_m / Omega_m' in each log:"
grep "beta_m / Omega_m" virial_test_low.log
grep "beta_m / Omega_m" virial_test_mid.log
grep "beta_m / Omega_m" virial_test_high.log
