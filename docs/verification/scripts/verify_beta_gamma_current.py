#!/usr/bin/env python3
"""The beta_gamma bound of verify_beta_gamma.py, on the current public CMB fits (downtime work; numpy, scipy). The model and method are
verify_beta_gamma.py's, unchanged: H_photon = H0 sqrt(Om a^-3 + Or a^-4 + OL + beta_g e^(1-1/a)), r_* fixed, theta_* compared with the fit's
own; every input of one row from ONE fit. Inputs: beta_gamma_current_inputs.csv, copied from the ACT DR6 LCDM table (arXiv:2503.14452;
columns ACT, Planck, W-ACT, P-ACT; the combinations with BAO are left out, since BAO is late-time data IAM modifies). SPT-3G D1
(arXiv:2506.20707) is left out: its table reports the angle under another definition (theta_s near 1.0418) with no matching r_* or z_*.
Simplified as the original: r_* held fixed; massless neutrinos for Omega_r. The profile bound depends on how far that simplified baseline
misses each fit's measured angle (printed, in sigma); the shift-only bound does not, and is the one to quote."""
import csv, os, numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
c = 299792.458; HERE = os.path.dirname(os.path.abspath(__file__))
for r in csv.DictReader(open(os.path.join(HERE, "beta_gamma_current_inputs.csv"))):
    H0, Om, zs, rs, obs, err = (float(r[k]) for k in ("H0", "Om", "zs", "rs", "theta", "err"))
    Or = 2.4728e-5*(1+0.2271*3.046)/(H0/100)**2; OL = 1-Om-Or
    H = lambda a, b: H0*np.sqrt(Om*a**-3 + Or*a**-4 + OL + b*np.exp(1-1/a))
    theta = lambda b: rs/(quad(lambda z: 1/H(1/(1+z), b), 0, zs, limit=500, epsabs=0, epsrel=1e-12)[0]*c)
    t0 = theta(0.0); c0 = ((t0-obs)/err)**2
    lim = {lev: brentq(lambda b: ((theta(b)-obs)/err)**2 - c0 - lev, 1e-7, 0.05) for lev in (1, 4)}
    print(f"{r['fit']}: theta_*(beta_g=0) = {t0:.8f} ({(t0-obs)/err:+.1f} sigma, fixed r_*) | beta_g < {lim[1]:.4f} (68 %), < {lim[4]:.4f} (95 %), beta_g/beta_m < {lim[4]/(Om/2):.3f} (95 %)")
    # the same model, the bound from the shift alone: theta_*(beta_g) - theta_*(0) within 2 of the fit's own errors (does not depend on how far
    # the simplified baseline misses the measured angle, which makes the profile bound above tighter when the baseline sits high)
    sh = brentq(lambda b: (theta(b)-t0)/err - 2.0, 1e-7, 0.05)
    print(f"   shift-only: beta_g < {sh:.4f} (2 sigma shift), beta_g/beta_m < {sh/(Om/2):.3f}")
