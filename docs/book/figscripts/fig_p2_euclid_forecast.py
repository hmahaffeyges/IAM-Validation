"""Part 2, Chapter 'What the surveys will measure' (p2_16_survey_predictions.tex), Section sec:sp_euclid.
fig_survey_euclid_forecast: the Fisher forecast with the IAM mu(z) for Euclid, Planck CMB lensing and DESI (CALCULATED FORECAST).
(a) the f sigma8 deficit of IAM against LambdaCDM, mu(z) = 1 + A [mu_IAM(z) - 1], A = 1, with the 1 sigma bands of A for the pessimistic
and optimistic Euclid + Planck lensing + DESI combinations, DESI f sigma8 errors (DESI 2016, k < 0.1 h/Mpc) and the Euclid spectroscopic
f sigma8 errors of this Fisher matrix; (b) the change of the Euclid 3x2pt spectra (shear 5x5, galaxy-galaxy lensing G3 x L9, photometric
clustering 3x3) with their errors.
The figure is drawn by the forecast package, docs/verification/forecasts/euclid_fisher_iam_mu/analysis.py (stage C), from
out/fishers.json (the production Fisher matrices) and one CAMB run; it needs camb (tested 2.0.4). This script runs stage C and copies
out/iam_signal_forecast.pdf to figures/part2/fig_survey_euclid_forecast.pdf. Stages A and B (run_forecast.py) rebuild fishers.json.
"""
import sys, shutil, subprocess, pathlib

HERE = pathlib.Path(__file__).resolve().parent
BOOK, REPO = HERE.parent, HERE.parent.parent.parent
PKG = REPO / "docs" / "verification" / "forecasts" / "euclid_fisher_iam_mu"
subprocess.run([sys.executable, "analysis.py"], cwd=PKG, check=True)
dst = BOOK / "figures" / "part2" / "fig_survey_euclid_forecast.pdf"
shutil.copyfile(PKG / "out" / "iam_signal_forecast.pdf", dst)
print("wrote", dst.relative_to(REPO))
