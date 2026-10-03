# IAM mu(z) Fisher forecast for Euclid (+ Planck CMB lensing + DESI) - CALCULATED FORECAST

Requirements: python 3 with `pip install camb numpy scipy pandas matplotlib joblib` (tested with camb 2.0.4;
numpy 1.26 and 2.5).

Steps:

```
python run_forecast.py [n_jobs]   # stage A (51 CAMB runs) + B (919 derivative tasks, 59 Fisher matrices) -> out/fishers.json
python analysis.py                # stage C: out/results.{csv,md}, out/validation.{csv,md}, out/iam_signal_forecast.{png,pdf}
python step_test.py               # optional: finite-difference step-size stability check -> out/step_test.json
```

Stage B took 16.6 min on 128 cores; per-worker setup dominates that time. A serial run takes roughly 10-20 min on a laptop (0.5 s per task locally).
`out/fishers.json` from the production run is included, so `analysis.py` can be run without stage A/B.

Files:

- `iamfisher.py`: model (CAMB, growth ODE with mu(a), HALOFIT, no-wiggle), probes (GCsp, 3x2pt, CMB lensing, DESI)
  and the Fisher utilities.
- `run_forecast.py`: case list and parallel driver.
- `analysis.py`: scenario combinations, scale-matching, tables and figure.
- `data/scaledmeanlum-E2Sa.dat`: the IST:F <L>/L* table used by the IA model.

METHODS.md lists the assumptions, sources and approximations. In the repository at `docs/verification/forecasts/euclid_fisher_iam_mu/`; the book carries its results in Chapter 'What the surveys will measure', Section 'Euclid, DESI and the turn-on of growth' (docs/book/part2/p2_16_survey_predictions.tex), and `docs/book/figscripts/fig_p2_euclid_forecast.py` copies its figure into the book.
