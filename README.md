# IAM's Law and Order

**The Actualization of Reality** — *The Cost of Recording It, and the Price to Maintain It.*

> Not a new model: a new perspective. Jacobson's exact formulas, taken one step further. General Relativity, with a new piece of information: information.

Heath W. Mahaffey, independent researcher.

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18702042-blue)](https://doi.org/10.5281/zenodo.18702042) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The law

Every irreversible transition from quantum superposition to a classical record costs at least `k_B T ln 2` per bit, paid at the
nearest surface on which the record is written, at that surface's temperature. The cost is Landauer's and has been measured; the
surface and its temperature are Bekenstein's and Hawking's; Jacobson showed that gravity follows from the thermodynamics of such
surfaces. IAM's one added identification is that the records belong in the surface's entropy. It is not a rival to general relativity
or the standard model: it is the cost both already imply wherever a record is made.

Followed across some 37 orders of magnitude in size, the same accounting applies at the cosmic horizon, at black holes, at qubits and
transistors, and at the methylation pattern a cell holds to remain the cell it is. What holds the stars in order holds the cell in order.

## Why pick it up

- **For a cosmologist:** one parameter-free line fixes `μ_0`, the `fσ_8` ramp, `Σ = 1`, `E_G`, `S_8` and the two Hubble rates together, and
  the full Planck likelihood with the coupling fixed fits as well as ΛCDM. The surveys now running measure its size.
- **For a quantum physicist:** a gate error and a quasiparticle density have a thermal floor at the device's own temperature, from
  `k_B T ln 2` alone. Every device is read against its own as-built reference.
- **For a geneticist:** the methylation pattern is a record a cell pays to keep. Each cell type is read against its own healthy state,
  so one person can be measured, measured again, and compared with themselves.

Every claim carries one status label (DERIVED, CALCULATED, CALIBRATED, MEASURED, OBSERVED, FITTED, CONJECTURE, PREDICTION or OPEN), and
every number in the book is recomputed by a script in this repository. The invitation is to try to break it.

## Start here

| what | where |
|---|---|
| **The book**, *IAM's Law and Order* (seven parts, LaTeX, compiles as-is in Overleaf) | [`docs/book/`](docs/book/) — open `main.tex` |
| Corrections to every source paper (authoritative where a paper and this list disagree) | [`docs/verification/PAPER_ERRATA.md`](docs/verification/PAPER_ERRATA.md) |
| Every constant and name, defined once | [`CANON/GLOSSARY.md`](CANON/GLOSSARY.md) (generated from `CANON/iam_canon.json`; `python3 CANON/canon_check.py`) |
| Every prediction, with its test and status | the book's predictions chapter and register appendix; source list [`CANON/predictions_register.csv`](CANON/predictions_register.csv) |
| Recomputation scripts for the book's numbers | [`docs/verification/scripts/`](docs/verification/scripts/) |
| Cosmology chains | [`mgcamb_validation/`](mgcamb_validation/) (Level 1), [`camb_validation/`](camb_validation/) (Level 2) |
| The cell instrument (chain v3, in development) | [`Biological_Physics/MethylPhys/`](Biological_Physics/MethylPhys/) — SOP, operations manual, toolkit, development records in `doors/` |

## Results, with their status

| place | result | status |
|---|---|---|
| cosmic horizon | matter-sector coupling `β_m = Ω_m/2 = 0.15765`, so `μ(0) = 0.864` (`μ_0 = −0.136`), `Σ = 1` | DERIVED (from the partition), CALCULATED |
| cosmic horizon | Planck 2018 fit with the coupling fixed: `Δχ² = +0.54` against ΛCDM; `σ_8` 0.8087 → 0.7998 | MEASURED (chains) |
| cosmic horizon | `fσ_8` 4.25 % below ΛCDM today, 2.17 % at z = 0.3, 0.41 % at z = 1; lensing unchanged | CALCULATED; PREDICTION |
| cosmic horizon | two Hubble rates: 67.16 (light) and 72.26 = 67.161·√1.1575 km/s/Mpc (matter flow) | CALCULATED; PREDICTION |
| cosmic horizon | putting the term in the background instead moves `H_0` to 61.5 (10.9σ): the background does not change in IAM | MEASURED (chains) |
| surveys | growth surveys through the mid-2030s reach about 2.5–3σ on the deficit with the CMB fixing the early amplitude; DESI DR2 full shape is next | CALCULATED (Fisher forecast) |
| black holes | horizon bits priced at the horizon's temperature total `½Mc²` (Smarr) | DERIVED |
| qubits | thermal floor of a transmon gate at its own temperature, from `k_B T ln 2` | DERIVED, CALCULATED |
| chips | Ryzen 9 9950X switches at 576–593 × `k_B T_j ln 2` (published TDP, clock and transistor count) | CALCULATED |
| cells | neutrophils read against their own healthy reference: held out, spread 0.020 around 1 | CALIBRATED, MEASURED |
| cells | DNMT1 blocked, tumour against the same patient's normal tissue, species ageing | development records (`doors/`), not yet commissioned |

The cell instrument is being commissioned stage by stage: base chain first, then atlas deconvolution, NILC, per-cell readings,
directional decomposition and the sky tools. Each stage enters the chain only after its own check passes; the order and status are in the
SOP, section 2b.

## The papers

The papers are the working record. Where a paper and the corrections list disagree, the corrections list and the book are current.

**The law**
- [IAM's Law: the thermodynamic cost of classical existence - the law itself; the cosmological model is one derived implementation of it](docs/papers/IAM_Law.pdf)
- [The thermodynamic identity governing the virial theorem - physical identification of K, with evidence across domains](docs/papers/PRL_Version_Thermodynamic_Identity_Governing_Virial_Theorem.pdf)
- [The virial partition across a wide range of physical scales - the cross-domain validation](docs/papers/Virial_Partitian_Across_Wide_Domains.pdf)
- [The virial partition from atoms to the horizon](docs/papers/The_Virial_Partition_from_Atoms_to_the_Horizon.pdf)

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
- [Landauer-based model for the minimum quasiparticle density in Al/AlOx/Al Josephson junctions](docs/papers/IAM_Xqp_Mahaffey.pdf)
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
- Physics of methylation: Landauer metrology - retired; its content is carried in the book, Part VI (the Landauer chapter)

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
python3 docs/verification/scripts/verify_encoding_ladder.py    # the places table of Part I
```
The cell chain is run as in [`Biological_Physics/MethylPhys/README.md`](Biological_Physics/MethylPhys/README.md) and its operations manual;
`python3 Biological_Physics/MethylPhys/kit/release_check.py` checks it end to end. The book compiles with `pdflatex`/`bibtex` (or Overleaf)
from [`docs/book/main.tex`](docs/book/main.tex).

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

