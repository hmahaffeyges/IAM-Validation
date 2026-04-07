# Biological Physics

Application of the Informational Actualization Model (IAM) to biological systems.

The same thermodynamic principle that governs gravitational decoherence and cosmic expansion
also sets the minimum information maintenance cost for living cells. The Landauer cost of
irreversible DNA methylation maintenance at physiological temperature defines architecture-class
specific entropy floors for all mammalian somatic cell types.

## Contents

### papers/
First-principles derivations and validation manuscripts.

- `Mahaffey_2026_cell_thermodynamics.tex` / `.pdf` — Thermodynamic Operating Constraints of
  Mammalian Somatic Cell Classes: A First-Principles Derivation from DNA Methylation
  Maintenance Information Costs. Eight architecture classes, MCMC-validated H_min floors,
  zero free parameters, 27/28 TCGA cancer types confirmed.

### evidence/
Validation data, MCMC chains, and supporting scripts.

- G-002: MCMC posterior H_min values for all 8 architecture classes
  (5 independent chains, R-hat < 1.001, 8×10^5 samples)
- G-008: Zero-free-parameter cancer floor breach validation
  (27/28 TCGA cancer types, n = 4,304 matched tumor-normal pairs)
- `fig_thermodynamic_validation.py` — Reproduces the four-panel validation figure
- `build_cell_architecture_card.py` — Generates the terminal class architecture card

## Key Results

| Result | Value | Method |
|--------|-------|--------|
| Global Landauer floor H_min_global | 0.756500 | Frontal cortex neuron, Lister 2013 |
| Terminal class floor H_min | 0.772837 | G-002 MCMC posterior |
| Cancer types confirmed (direction) | 27/28 | G-008, zero free parameters |
| AD A-score (high neuropathology) | 1.020 | De Jager 2014 |
| LGG A-score | 1.285 | Ceccarelli 2016 |
| GBM A-score | 1.256 | Ceccarelli 2016 |
| DunedinPACE t_max | 120.3 ± 7.1 yr | MCMC posterior |

## Core Equation

```
A = H(β) / H_min(class)
```

where H(β) is the Shannon binary entropy of the mean genome-wide methylation beta,
and H_min(class) is the architecture-class floor derived from the Landauer cost of
DNMT1-mediated methylation maintenance at T_body = 310.15 K.

## Zenodo

Canonical DOI: [10.5281/zenodo.18702042](https://doi.org/10.5281/zenodo.18702042)

## Status

Pre-clinical. All predictions tested against published data only.
Prospective clinical validation has not been performed.
