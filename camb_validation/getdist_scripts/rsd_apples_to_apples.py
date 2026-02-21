#!/usr/bin/env python3
"""Quick apples-to-apples RSD comparison for Level 2a Run D"""
import numpy as np
from getdist.mcsamples import loadMCSamples
import camb, sys, os

chain_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/CAMB_IAM_L2/chains')

z_data = np.array([0.067, 0.150, 0.380, 0.510, 0.700, 0.850, 1.480])
fsig8_data = np.array([0.423, 0.530, 0.497, 0.459, 0.473, 0.315, 0.462])
fsig8_err = np.array([0.055, 0.160, 0.045, 0.038, 0.041, 0.095, 0.045])

# Load LCDM baseline
lcdm = loadMCSamples(os.path.join(chain_dir, 'iam_level2_runC_lcdm'), settings={'ignore_rows': 0.3})
ls = lcdm.getMargeStats()

# Compute LCDM fsigma8
pars = camb.CAMBparams()
pars.set_cosmology(H0=ls.parWithName('H0').mean, ombh2=ls.parWithName('omegabh2').mean,
                   omch2=ls.parWithName('omegach2').mean, tau=ls.parWithName('tau').mean)
pars.InitPower.set_params(As=np.exp(ls.parWithName('logA').mean)*1e-10, ns=ls.parWithName('ns').mean)
pars.set_matter_power(redshifts=z_data.tolist(), kmax=2.0)
pars.want_transfer = True
pars.NonLinear = camb.model.NonLinear_none
results = camb.get_results(pars)
fsig8_lcdm = results.get_fsigma8()
chi2_lcdm_rsd = np.sum(((fsig8_lcdm - fsig8_data) / fsig8_err) ** 2)

# Load IAM+RSD chain
rsd = loadMCSamples(os.path.join(chain_dir, 'iam_level2_runD'), settings={'ignore_rows': 0.3})
rs = rsd.getMargeStats()
iam_total = rs.parWithName('chi2').mean
iam_cmb = rs.parWithName('chi2__CMB').mean
iam_rsd = iam_total - iam_cmb
lcdm_cmb = ls.parWithName('chi2').mean

print("=" * 70)
print("  APPLES-TO-APPLES Level 2a Run D Breakdown")
print("=" * 70)
print(f"\n  {'Component':20s} {'IAM':>10s} {'LCDM':>10s} {'Delta':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
print(f"  {'CMB':20s} {iam_cmb:>10.2f} {lcdm_cmb:>10.2f} {iam_cmb - lcdm_cmb:>+10.2f}")
print(f"  {'RSD (7 pts)':20s} {iam_rsd:>10.2f} {chi2_lcdm_rsd:>10.2f} {iam_rsd - chi2_lcdm_rsd:>+10.2f}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
delta_total = (iam_cmb - lcdm_cmb) + (iam_rsd - chi2_lcdm_rsd)
print(f"  {'TOTAL (apples)':20s} {'':>10s} {'':>10s} {delta_total:>+10.2f}")

print(f"\n  LCDM fsigma8 vs data:")
print(f"  {'z':>8s} {'data':>8s} {'err':>8s} {'LCDM':>8s} {'pull':>8s}")
for i in range(len(z_data)):
    pull = (fsig8_lcdm[i] - fsig8_data[i]) / fsig8_err[i]
    print(f"  {z_data[i]:8.3f} {fsig8_data[i]:8.3f} {fsig8_err[i]:8.3f} {fsig8_lcdm[i]:8.4f} {pull:>+8.2f}")

print(f"\n  PASS: |Delta-chi2 (apples)| = {abs(delta_total):.2f} < 5.0? {'PASS' if abs(delta_total) < 5.0 else 'FAIL'}")
print("=" * 70)
