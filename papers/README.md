# IAM Research Papers

[![OSF Preprint](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![viXra](https://img.shields.io/badge/viXra-2512.0029-orange)](https://ai.vixra.org/abs/2512.0029)
[![Papers](https://img.shields.io/badge/Papers-3-green)](papers/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This directory contains research papers related to the Informational Actualization Model (IAM).

## Structure

### [`/cosmology/`](cosmology/)
Core cosmological framework resolving the Hubble tension.

**Status**: Strong statistical evidence (Δχ² = 32.09, 5.7σ)
- Published: OSF Preprints (DOI: 10.17605/OSF.IO/KCZD9)
- Preliminary version: viXra:2512.0029 (December 2025)
- Under review: arXiv endorsement requested (February 2026)

### Observational Paper
"Constraints on Late-Time f·σ₈ Suppression from µ < 1, Σ = 1: Planck 2018 and Large-Scale Structure"

**Status**: Submitted to *Universe* (February 19, 2026)
- 12 MCMC chains across 4 dataset combinations (Planck, +RSD, +BAO, +Pantheon+)
- All Δχ² below exclusion threshold; σ₈ shift of −0.013 ± 0.001 universal across datasets
- Full chain data and figures in [`../mgcamb_validation/`](../mgcamb_validation/)

### [`/speculative/`](speculative/)
Speculative extensions exploring implications beyond cosmology.

**Status**: Exploratory / not peer-reviewed
- Quantum computing limits
- Thermodynamic applications
- Foundational physics implications
- Developed via human-AI collaboration

## Timeline

| Date | Milestone |
|------|-----------|
| Dec 2025 | Preliminary cosmology framework (viXra:2512.0029) |
| Feb 2026 | Refined cosmology analysis (OSF DOI) |
| Feb 2026 | Speculative extensions paper |
| Feb 2026 | arXiv endorsement requested (Lloyd Knox, UC Davis) |
| Feb 13, 2026 | Outreach emails to MGCAMB/µ-Σ researchers (Wang, Pogosian, Silvestri) |
| Feb 18, 2026 | All 12 MCMC runs converged (Runs A–L) |
| **Feb 19, 2026** | **Observational paper submitted to *Universe*** |
| TBD | arXiv publication |
| TBD | Level 2 background-modified CAMB results |

## Citation

For the core cosmological IAM:
```bibtex
@article{Mahaffey2026IAM,
  author = {Heath W. Mahaffey},
  title = {Holographic Black-Hole Cosmology: An Informational Resolution of the Hubble Tension},
  year = {2026},
  publisher = {OSF Preprints},
  doi = {10.17605/OSF.IO/KCZD9},
  url = {https://osf.io/kczd9}
}
```

For the observational validation:
```bibtex
@article{Mahaffey2026obs,
  author = {Heath W. Mahaffey},
  title = {Constraints on Late-Time $f\sigma_8$ Suppression from $\mu < 1$, $\Sigma = 1$: Planck 2018 and Large-Scale Structure},
  journal = {Universe},
  year = {2026},
  note = {Submitted February 2026. Code: \url{https://github.com/hmahaffeyges/IAM-Validation}}
}
```
