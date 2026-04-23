# VAL-054 — Age-Confounding Assessment on AIBL Holdout

**Date:** 2026-04-23
**Status:** Complete — pre-registered, hash-sealed, executed in two parts.
**Parent:** VAL-051
**Outcome:** VAL-054a non-test; VAL-054b **STRONG** (p=0.003 HC-internal permutation bound)

---

## Headline

VAL-051's AIBL holdout d = +0.624 was computed without age adjustment because AIBL's GEO deposit does not include chronological age metadata. VAL-054 attempts two complementary tests:

### VAL-054a — cellular-age regression (non-test)
Applied Alpha-Omega §E.5 80-cell immune-class baseline to compute cellular age per sample. All 148 holdout samples mapped to cellular age = 95.0 yr (saturated at oldest decade).

**Diagnosis:** the IMM_CPGS panel β_mean is centered at 0.52, but the 80-cell baseline β_mean is 0.73-0.78 (class-wide β, not panel-subset β). Panel was selected by Xu 2020 for *differential-methylation signal*, not class-average β. Applying class-wide baseline to panel subset is a methodological mismatch. **Honest non-test; direct age regression requires AIBL direct-access metadata.**

### VAL-054b — HC-internal permutation bound (VALID TEST)
10,000 random splits of HC (n=95) into subsets of size 33 and 62. If observed AD signal exceeds what random HC-internal splits produce, the signal cannot be attributed to any within-HC variance source collectively (including age, sex, batch, cell composition).

| Metric | Value |
|---|---|
| Observed d | +0.624 |
| Null mean | −0.002 |
| Null SD | 0.219 |
| Null 99th pctile | +0.520 |
| **P(null d ≥ d_obs)** | **0.003** |
| Z of observed in null | +2.85 |

**Verdict: STRONG.** Observed AD signal is at the 99.7th percentile of HC-internal splits.

---

## Relationship to VAL-052

VAL-054b and VAL-052 produce complementary (not contradictory) readings:

- **VAL-054b on AIBL:** within-HC variance cannot explain the signal (p = 0.003)
- **VAL-052 on AddNeuroMed:** age linearly regresses out 60% of the signal (residual d = +0.12)

These test different questions. VAL-054b is a permutation bound: *is the signal bigger than HC-internal noise?* VAL-052 is a linear regression decomposition: *how much of the signal does age linearly predict?*

A panel can pass the first and still be partly age-driven. Combined: the AD panel detects real AD-associated signal plus age-associated drift that is itself accelerated in AD. Age-adjusted Z-score (Alpha-Omega §E.5) is the correct clinical output.

---

## Files

| File | Role |
|---|---|
| `VAL_054_PREREG.md` | Pre-registration for both approaches |
| `VAL_054_SEAL.txt` | SHA-256 hashes sealed before analysis |
| `val054_age_regression.py` | VAL-054a cellular-age regression attempt |
| `val054b_permutation_bound.py` | VAL-054b HC-internal permutation bound |
| `VAL_054_RESULTS.json` | VAL-054a results (non-test flag) |
| `VAL_054b_RESULTS.json` | VAL-054b results (strong positive) |
| `VAL_054_REPORT.md` | Human-readable report combining both |

---

## Reproduction

```bash
# AIBL data + split + panel live in parent folders
cp ../val_050_aibl/aibl_manifest.json .
cp ../val_050_aibl/aibl_imm_betas.json .
cp ../val_051_ad_directional/val051_split_map.json .
cp ../val_051_ad_directional/val051_panel_ruleA.json .

python3 val054_age_regression.py     # VAL-054a (produces non-test flag)
python3 val054b_permutation_bound.py # VAL-054b (produces p=0.003 bound)
```

All stdlib Python 3.9+. Seed 42. Outputs byte-identical.
