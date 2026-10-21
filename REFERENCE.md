# IAM: reference

The long-form sections that used to sit on the front README, moved here 2026-10-15 so the front page stays short. Nothing was removed; the front README links here.

## What IAM is

The Informational Actualization Model is a testable set of equations inside general relativity, the piece Jacobson's thermodynamic
derivation of Einstein's equations left open. At the cosmic horizon it fixes a late-time coupling with no free parameter: IAM fits the
Planck CMB as well as ΛCDM and predicts slower growth of structure with lensing unchanged, a test Euclid and DESI will make. In the
cell, the same price per bit sets a floor under how faithfully a methylation pattern can be held, and two instruments read each
cell's informational fidelity against that floor and against its own healthy state. The substrate changes. The noise changes. The measured inputs
change. The thermodynamic accounting does not.

## Why pick it up

- **For a cosmologist:** one parameter-free line fixes `μ_0`, the `fσ_8` ramp, `Σ = 1`, `E_G`, `S_8` and the two Hubble rates together, and
  the full Planck likelihood with the coupling fixed fits as well as ΛCDM. The surveys now running measure its size.
- **For a quantum physicist:** a gate error and a quasiparticle density have a thermal floor at the device's own temperature, from
  `k_B T ln 2` alone. The book gives the physics, with makers' published device values as illustrations.
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
| Check every derivation in the book | [`docs/book/verify_book.py`](docs/book/verify_book.py): `python3 docs/book/verify_book.py` (results and inventory in [`docs/book/verification/`](docs/book/verification/)) |
| Cosmology and gravitation | [`Cosmological_Physics/`](Cosmological_Physics/) — Level 1 and Level 2 chains, data, early tests |
| Cellular physics | [`Biological_Physics/`](Biological_Physics/) — the cell instrument (chain v3): SOP, operations manual, toolkit |
| Quantum and particle physics | [`Quantum_and_Particle_Physics/`](Quantum_and_Particle_Physics/) — the scripts behind Parts IV and V |
| Development logs | [`development/`](development/) — every development finding, one running log per project |

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
| cells | IAM-Atlas v2.1: 79 healthy cell types at 813,951 array sites, one Bayesian model across 21 public sources; commissioned for three cell references and the colon and lung tissue tests (2026-10-16), and it met all three sealed bars against both published reference atlases. Atlas v2 (74 cell types, seven sources) is its predecessor | MEASURED, commissioned |
| cells | cancer fingerprints (prostate in two laboratories, breast lines, breast and colorectal patients), a physics prediction met in mice, three-laboratory agreement, Met-A commissioned on neutrophils | dated, each with its record, in the table below |



<!-- STATUS:BEGIN -->
**Status (generated by `CANON/status_lines.py`):** commissioned: Met-A on neutrophils, EPIC v1, isolated neutrophils ([note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md)); the Met-A C-score on whole blood, EPIC v1 ([note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_CSCORE.md)); IAM-Atlas v2.1 for three cell references (450K) and the colon and lung tissue tests' blind sites ([note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_ATLAS_V21.md)); the Deconvolution tool for blood-cell shares on arrays ([note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_DECONV_V21.md)); IAM-A on healthy blood CD8 T cells (sequencing, its own reference) ([note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_IAMA.md)); Group-A on groups of whole bloods run in one series, EPIC v1 ([note](Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_GROUPA.md)). Everything else is in development; not a diagnostic test. IAM-Atlas v2 is the reference in use; v2.1 is commissioned (79 cell types; met all three sealed bars against both published reference atlases, 2026-10-16; commissioned for three cell references and the colon and lung tissue tests). Every instrument and its next step: [Biological_Physics/MethylPhys/INSTRUMENTS.md](Biological_Physics/MethylPhys/INSTRUMENTS.md); what is commissioned and the recent results: [STATUS.md](Biological_Physics/MethylPhys/STATUS.md).
<!-- STATUS:END -->

The cell instrument is commissioned stage by stage: base chain first, then atlas deconvolution, NILC, per-cell readings,
directional decomposition and the sky tools. Each stage enters the chain only after its own check passes; the order and status are in the
SOP, section 2b.

## The papers

The book is the one current text. The papers it was built from were written and shared along the way; they are not kept in this
repository, because the book corrects and supersedes them. Every correction, with the original passage beside the corrected one, is in
[`docs/verification/PAPER_ERRATA.md`](docs/verification/PAPER_ERRATA.md). The originals remain public on
[OSF](https://doi.org/10.17605/OSF.IO/KCZD9) and [Zenodo](https://doi.org/10.5281/zenodo.18702042).

## Reproduce

**Check the book's derivations in one command:** `python3 docs/book/verify_book.py`. Every derivation and calculated number in the book is checked and passes (4,643 checks, 0 failures), in the book's order, each with its equation label (`--part N`, `--label L`, `--fails`, `--json`).

**Development logs:** every development finding, good or bad, is logged as it happens in [`development/`](development/), one running log per project (Met-A, IAM-A and C-scores; wild versus hatchery fish). The book carries results once the chain that produced them is commissioned.


```
git clone https://github.com/hmahaffeyges/IAM-Validation && cd IAM-Validation
pip install numpy scipy pandas matplotlib
python3 CANON/canon_check.py                                   # constants and names consistent
python3 docs/verification/scripts/verify_encoding_ladder.py    # the encoding surfaces; see the Saturation appendix's identity table
```
The cell chain is run as in [`Biological_Physics/MethylPhys/README.md`](Biological_Physics/MethylPhys/README.md) and its operations manual;
`python3 Biological_Physics/MethylPhys/kit/release_check.py` checks it end to end. The book compiles with `pdflatex`/`bibtex` (or Overleaf)
from [`docs/book/main.tex`](docs/book/main.tex).

