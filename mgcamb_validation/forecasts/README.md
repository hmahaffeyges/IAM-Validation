# Forecasting and Observational Prospects

Four complementary analyses projecting IAM's detectability with upcoming surveys.
These correspond to Section 10 of the IAM-CAMB Technical Note.

## Scripts

| Script | Section | Key Result |
|--------|---------|------------|
| `iam_fisher_forecast.py` | 10.1 | Detection timeline: Euclid+DESI projected 5.4σ (idealized) |
| `iam_isw_prediction.py` | 10.2 | ISW enhancement: A_ISW = 1.13 (13% above LCDM) |
| `iam_binned_mu_reconstruction.py` | 10.3 | Tomographic mu(z) shape recovery with Euclid bins |
| `iam_transition_zone.py` | 10.4 | Transition zone: 50% activation at z = 0.69 |

## Usage

```bash
# Each script is self-contained and generates a multi-panel PDF figure
python iam_fisher_forecast.py          # -> iam_fisher_forecast.pdf
python iam_isw_prediction.py           # -> iam_isw_prediction.pdf
python iam_binned_mu_reconstruction.py # -> iam_binned_mu_reconstruction.pdf
python iam_transition_zone.py          # -> iam_transition_zone.pdf
```

## Requirements

- Python 3.8+ with numpy, scipy, matplotlib

No external cosmology packages required — all calculations are analytical.

## Caveat

These Fisher forecasts assume idealized Gaussian posteriors, linear-scale modeling,
and neglect survey systematics (e.g., nonlinear bias, baryonic effects, photometric
calibration). The quoted significances represent optimistic sensitivity estimates
rather than guaranteed detection levels.
