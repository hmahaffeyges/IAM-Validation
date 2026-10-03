# IAM's Law and Order

**The Actualization of Reality** — *the cost of recording it, and the price of maintaining it.*

> Not a new model: a new perspective. Jacobson's exact formulas, taken one step further. General Relativity, with a new piece of information: information.

**How decoherence writes the classical world, and the energy that holds it against thermal noise.**
Heath W. Mahaffey, independent researcher.

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18702042-blue)](https://doi.org/10.5281/zenodo.18702042) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The law

Every irreversible transition from quantum superposition to a classical record costs at least `k_B T ln 2` per bit, paid at the
nearest surface on which the record is written, at that surface's temperature. The cost is Landauer's and has been measured; the
surface and its temperature are Bekenstein's and Hawking's; Jacobson showed that gravity follows from the thermodynamics of such
surfaces. IAM's one added identification is that the records belong in the surface's entropy. It is not a rival to general relativity
or the standard model: it is the cost both already imply wherever a record is made.

Followed across some 37 orders of magnitude in size, the same accounting applies at the cosmic horizon, at black holes, at qubits and
transistors, and at the methylation pattern a cell holds to remain the cell it is.

## Start here

| what | where |
|---|---|
| **The book** (five parts, LaTeX, compiles as-is in Overleaf) | [`docs/book/`](docs/book/) — open `main.tex` |
| Corrections to every paper (authoritative where a paper and this list disagree) | [`docs/verification/PAPER_ERRATA.md`](docs/verification/PAPER_ERRATA.md) |
| Every constant and name, defined once | [`CANON/GLOSSARY.md`](CANON/GLOSSARY.md) (generated from `CANON/iam_canon.json`; `python3 CANON/canon_check.py` lists files that must follow a change) |
| Every prediction in the papers, with its verdict and reason | [`CANON/predictions_register.csv`](CANON/predictions_register.csv), [`CANON/predictions_triage_2026-10-02.json`](CANON/predictions_triage_2026-10-02.json); the live list is Part 5, *Falsifiable predictions* |
| The cell instrument (chain v3, development build) | [`Biological_Physics/MethylPhys/`](Biological_Physics/MethylPhys/) |
| Cosmology chains | [`mgcamb_validation/`](mgcamb_validation/) (Level 1), [`camb_validation/`](camb_validation/) (Level 2) |
| Recomputation scripts for the book's numbers | [`docs/verification/scripts/`](docs/verification/scripts/) |
| Derivation checks for the device and cell numbers | DERIVATIONS QPROC_CHIP_MPHYS/ (archived privately) (`python3 mphys_derivation_tests.py`, and the quantum-processor report and the semiconductor report files) |

Every result carries one status label: DERIVED, CALCULATED, CALIBRATED, MEASURED, OBSERVED, FITTED, CONJECTURE, PREDICTION or OPEN.

## Results, with their status

| place | result | status |
|---|---|---|
| cosmic horizon | matter-sector coupling `β_m = Ω_m/2`, so `μ(0) = 0.864` (`μ_0 = −0.136`), `Σ = 1` | DERIVED (from the partition), CALCULATED |
| cosmic horizon | Planck 2018 fit with the coupling fixed: `Δχ² = +0.54` against ΛCDM; only `σ_8` moves (0.8087 → 0.7998) | MEASURED (chains) |
| cosmic horizon | `fσ_8` 4.25 % below ΛCDM today, 2.17 % at z = 0.3, 0.41 % at z = 1; lensing unchanged | CALCULATED; PREDICTION |
| cosmic horizon | two Hubble rates: 67.16 (light) and 72.26 = 67.161·√1.1575 km/s/Mpc (matter flow) | CALCULATED; PREDICTION |
| black holes | horizon bits priced at the horizon's temperature total `½Mc²` (Smarr) | DERIVED |
| qubits | gate error read against a material floor; Quantinuum Helios about 8× above its floor | floors CALIBRATED |
| chips | Ryzen 9 9950X switches at 576 × `k_B T_j ln 2` | CALCULATED |
| cells | neutrophils read against their own healthy reference: held out, SD 0.020 around 1 | CALIBRATED, MEASURED |
| cells | DNMT1 blocked: Met-A 1.16–1.85 (arrays), IAM-A 1.65–1.97 (molecules); tumour copy error above the same patient's normal tissue in all 10 pairs read (6 colorectal, 4 oral) | MEASURED (development) |

What is not shown yet: no floor is derived from first principles; the cell instrument is not commissioned on real whole blood; breach
and the cancer region on the cell gauge are not placed. The full list, with the plan for each item, is Part 5, *What is open*.
The decisive cosmology test is Euclid: its first complete data release (mid-2027) tests `μ_0` at about 1.7σ, the final survey at about 3.4σ.

## The papers

The papers are the working record. Where a paper and the corrections list disagree, the corrections list and the book are current.

**The law**
- [IAM's Law: the thermodynamic cost of classical existence - the law itself; the cosmological model is one derived implementation of it](docs/papers/IAM_Law.pdf)
- [The thermodynamic identity governing the virial theorem - physical identification of K, with evidence across domains](docs/papers/PRL_Version_Thermodynamic_Identity_Governing_Virial_Theorem.pdf)
- [The virial partition across a wide range of physical scales - the cross-domain validation](docs/papers/Virial_Partitian_Across_Wide_Domains.pdf)
- [The virial partition from atoms to the horizon](docs/papers/The_Virial_Partition_from_Atoms_to_the_Horizon.pdf)
- The Informational Actualization Model: a technical reference for physicists (archived privately)

**Cosmology and gravitation**
- [Horizon thermodynamics and gravitational decoherence as the origin of mu < 1, Sigma = 1](docs/papers/IAM_Theory_Paper.pdf)
- [Master consolidated preprint - dual-sector cosmology, zero parameters beyond LCDM](docs/papers/IAM_Master_Preprint.pdf)
- [Technical companion - the quick overview](docs/papers/IAM_Overview_Companion.pdf)
- [IAM-CAMB technical note: mapping, Boltzmann validation, full Planck MCMC](docs/papers/IAM_CAMB_Technical_Note.pdf)
- [Dual-sector perturbation cosmology: the modified CAMB implementation](docs/papers/Dual_Sector_Perturbation_Cosmology_CAMB.pdf)
- [Type Ia supernovae validate matter-sector H0 normalisation](docs/papers/Dual_Sector_Validation_Paper.pdf)
- [Why the sector split is already in general relativity](docs/papers/IAM_Dual_Sector_Note.pdf)
- [Constraints on late-time fsigma8 suppression: Planck 2018 and large-scale structure](docs/papers/Late_Time_Growth_Suppression_in_the_mu_Sigma_Framework__Confrontation_with_Planck_and_Large_Scale_Structure.pdf)
- [Confrontation with DESI full-shape growth rates and joint weak lensing](docs/papers/Dark_Energy_or_Sector_Tension.pdf)
- [The redshift-dependent S8 trend](docs/papers/The_Redshift_Dependent_S_8_Trend_in_the_Context_of_IAM.pdf)
- [The cosmological constant as actualised vacuum energy - a zero-parameter resolution](docs/papers/The_Cosmological_Constant_as_Actualized_Vacuum_Energy.pdf)
- [Dark energy evolution and the far future of an IAM universe](docs/papers/wz_far_future.pdf)
- [Falsifiable predictions for Euclid, DESI and next-generation surveys](docs/papers/IAM_Survey_Predictions_Paper.pdf)
- [Missing satellites: the virial partition closure condition and the two mechanisms](docs/papers/Missing_Satellites.pdf)
- [Virial efficiency and the effective nonlinear exponent - published N-body confirmation](docs/papers/Virial_Efficiency_and_Effective_Nonlinear_Exponent.pdf)
- [Lensing-dynamics mass discrepancy as a redshift-dependent signature](docs/papers/IAM_Lensing_Dynamics_Paper.pdf)
- [Three-way mass discrepancy in galaxy clusters: eROSITA, Planck SZ, DES](docs/papers/3Way_Mass_Discrepancy_in_Galaxy_Clusters.pdf)
- [Quantum Darwinism at cosmological scales - the cosmic horizon and the emergence of classicality](docs/papers/Quantum_Darwinism_at_Cosmological_Scales.pdf)
- [Black hole horizons as thermodynamic encoding surfaces](docs/papers/IAM_BH_Thermodynamics.pdf)
- [The cessation of projection: the information paradox](docs/papers/IAM_Black_Hole_Information_Paradox.pdf)
- [The geometric origin of the Bekenstein-Hawking entropy coefficient](docs/papers/Bekenstein_coefficient.pdf)
- [A note on entropic gravity and the thermodynamic-gravity conjecture](docs/papers/A_Note_on_Entropic_Gravity__Saridakis_.pdf)
- [Validation scorecard - the complete test ledger](docs/papers/IAM_Official_Score_Card.pdf)
- [Complete test validation compendium](docs/papers/IAM_Test_Validation_Compendium.pdf)
- [Supplementary methods and reproducibility guide](docs/papers/Supplementary_Methods_Reproducibility_Guide.pdf)
- [Dark matter and dark energy as virial partners](docs/papers/Dark_Matter_and_Dark_Energy_as_Virial_Partners.pdf)

**Quantum and particle physics**
- [Landauer-based model for the minimum quasiparticle density in Al/AlOx/Al Josephson junctions - the quantum-processor report foundation](docs/papers/IAM_Xqp_Mahaffey.pdf)
- [Electron rest mass from holographic horizon thermodynamics - a fixed-point equation](docs/papers/Electron_Rest_Mass_from__IAM.pdf)
- [Three charged lepton generations and the Koide ratio from horizon information equipartition](docs/papers/Koide_Mahaffey.pdf)
- [Electroweak symmetry breaking and the matter sector](docs/papers/Electroweak_Symmetry_Breaking_and_the_Matter_Sector.pdf)
- [Matter-antimatter asymmetry and the information-writing constraint](docs/papers/Matter_Antimatter_Asymmetry_and_the_Information_Writing_Constraint.pdf)
- [The baryon asymmetry as a derived quantity - CMB evidence without a BBN prior](docs/papers/Baryon_Asymmetry_as_a_Derived_Quantity_CMB_Evidence_Without_BBN_Prior.pdf)
- [The measurement problem dissolved: decoherence as irreversible sector crossing](docs/papers/IAM_Measurement_Problem_Quantum.pdf)
- [Gravitational decoherence from dual-sector thermodynamics - predictions for optomechanical experiments](docs/papers/Gravitational_Decoherence_Quantum_Level.pdf)
- [Gravitational decoherence, the virial partition, and the emergence of classical structure](docs/papers/Gravitational_Decoherence_and_the_Virial_Partition.pdf)
- [Entanglement, decoherence, and the thermodynamic cost of classical records](docs/papers/Entanglement_Decoherence_and_Classical_Records.pdf)
- [The two faces of time: coordinate time, proper time, and accumulated decoherence](docs/papers/The_Two_Faces_of_Time.pdf)

**The cell**
- [Physics of methylation: Landauer metrology](docs/papers/Physics_of_Methylation__Landauer_Metrology.pdf)

**Exploratory, marked as such** — not part of the validation record:
- [Gravitational engineering and interstellar transit: a first-principles exploration (exploratory)](docs/papers/IAM_Gravitational_Engineering_Exploration.pdf)

**Earlier revisions, kept for provenance:**
- [Earlier revision of the Bekenstein coefficient paper](docs/papers/iam_bekenstein_coefficient.pdf)
- [Retired v1 manuscript](docs/Retired_V1_IAM_Manuscript.pdf)
- [Retired technical clarifications guide](docs/Retired_IAM_Technical_Clarifications_Guide.pdf)

## Reproduce

```
git clone https://github.com/hmahaffeyges/IAM-Validation && cd IAM-Validation
pip install numpy scipy pandas matplotlib
python3 CANON/canon_check.py                                   # constants and names consistent
# python3 "DERIVATIONS QPROC_CHIP_MPHYS/mphys_derivation_tests.py"   # cell derivation checks (folder archived privately)
python3 docs/verification/scripts/verify_encoding_ladder.py    # the places table of Part 1
```
The cell chain is run as in [`Biological_Physics/MethylPhys/README.md`](Biological_Physics/MethylPhys/README.md). The book compiles with
`pdflatex`/`bibtex` (or Overleaf) from [`docs/book/main.tex`](docs/book/main.tex).

## Citation

```bibtex
@misc{Mahaffey2026,
  author    = {Mahaffey, Heath W.},
  title     = {Dual-Sector Cosmology from Structure-Driven Expansion: The Informational Actualization Model},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18702042},
  url       = {https://doi.org/10.5281/zenodo.18702042}
}
```

## Contact

Heath W. Mahaffey, independent researcher. Email: hmahaffeyges@gmail.com. GitHub: [@hmahaffeyges](https://github.com/hmahaffeyges).
Questions, checks and replications are welcome as GitHub issues.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

The work rests on Landauer, Bennett, Bekenstein, Hawking, Gibbons, Jacobson, Cai and Kim, and Zurek. Data: the Planck, SDSS/BOSS/eBOSS,
SH0ES, Pantheon+, DESI, KiDS and DES collaborations, and the authors of every public methylation data set read here (named in each
record). Software: CAMB (Lewis, Challinor), MGCAMB (Wang, Mirpoorian, Pogosian, Silvestri, Zhao), Cobaya (Torrado, Lewis), HEALPix, NumPy,
SciPy, Matplotlib, GetDist.

The previous README (April 2026) is kept unchanged at docs/RETIRED_2026-10/README_2026-04.md (archived privately).
