# Verification: independent recomputation of the IAM physics papers

This folder holds the record of checking the theory, cosmology and physics papers in `docs/papers/`. For each paper, the record gives:
- every quantitative claim, recomputed from the paper's own equations and stated inputs (Planck 2018 unless noted);
- every cited literature value, traced to its source and page;
- the result: **reproduced**, **corrected** (with the evidence), or **open** (with what is needed).

Nothing here is a summary of a paper. A paper's argument is in the paper and in its book chapter (`docs/book/`); this folder holds the checks
those chapters rely on. A paper is listed only after it has been read in full. The reading plan and status for all 47 papers is in
`docs/book/BOOK_READING_TODO.md`.

## Contents
| Folder | File | Paper(s) checked | Status |
|---|---|---|---|
| `theory/` | `THEORY_CHECK.md` | IAM Theory Paper (14 Apr 2026) | Reproduced except five items, all resolved: exponent n = 7/2 (Eq. 41), w_a = −0.012, N-body tables restated, Fig. 2 label, §11.5 normalisation |
| `theory/` | `EXPONENT_LINE_BY_LINE.md` | Theory Paper Eqs. 28–41 | n = 7/2 analytic and numerical |
| `virial/` | `VIRIAL_CHECK.md` | The five Virial papers (25 Feb – 18 Mar 2026) | Reproduced; corrections listed (β_m posterior, E(a) column, dark-energy share, z_t, sector census, citations) |
| `virial/` | `NBODY_TRACE.md` | The N-body virial-ratio, n_eff and f_coll values cited in the Virial and Theory papers | Traced to arXiv full text with page numbers; Neto 2007 and Power 2012 re-read by me |
| `cosmological_constant_and_baryon/` | `CC_AND_BARYON_CHECK.md` | Cosmological Constant paper; Baryon Asymmetry paper; the 18th (baryon) chain | One present-epoch relation Ω_b/Ω_m ≈ (3/16)√Ω_Λ, 0.7σ on the CMB-only chain; derivation open |
| `observations/` | `TWO_RULER_DESI_TEST.md` | Dark Energy or Sector Tension mechanism vs DESI DR2 (mock) | Development measurement, distances only |
| `observations/` | `MISSING_SATELLITES_CHECK.md` (+ `data/`) | Missing Satellites (Mar 2026) | Mechanism B tested against the Local Volume Database; not in the book by ruling, kept as the record |
| `scripts/` | `verify_theory_paper.py` (+ output) | Theory Paper | Reruns every number in `theory/` in about 2 s (numpy, scipy, sympy) |

## Related records elsewhere in the repository
- Chain numbers: `mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv` and `CHAIN_PAIRS_FINAL.csv`, one documented extraction of all 18 final chain files (30 % burn-in). Every Δχ², σ8 and µ0 in the book comes from these.
- Side tests: `results/side_tests/` (cusp–core).
- Derivation test suite: `tests/iam_derivation_tests.py`.

## Conventions
- Δχ² is IAM minus ΛCDM, so a positive value means IAM fits slightly worse. "Consistent with Planck" means |Δχ²| of order 1 with no extra parameters.
- "Reproduced" means my number agrees with the paper's to the precision the paper prints. A difference that comes from the paper's inputs
  (for example Ω_m = 0.315 vs 0.3153) is stated, not called an error.
- Corrections are made only where the paper's own equations, or the cited source read directly, show the printed value is wrong. The author has
  approved corrections on that basis (2026-10-02).
