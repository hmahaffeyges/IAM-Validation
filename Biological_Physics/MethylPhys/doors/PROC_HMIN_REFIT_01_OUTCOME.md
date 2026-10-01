# PROC-HMIN-REFIT-01 — outcome (2026-09-30): option A measured; NOT adoptable as it stands

Run as pre-registered (PROC_HMIN_REFIT_01_PREREG.md): 314 purified-cell samples of the atlas v2 roster (+ GSE63409 normal marrow),
813,951 loci in the atlas universe U. h_A = mean over U of H(β) per sample; cell = median of its samples; class = G-002 closed-form
optimum Σh²/Σh. Nothing in the chain changed.

## The numbers
| class | current floor | A, WGBS (cells) | A, array (cells) | B = H(mean β), WGBS |
|---|---|---|---|---|
| pluripotent | 0.9822 | 0.388 (1) | — | 0.897 |
| adult stem | 0.8737 | — | 0.424 (1) | — |
| progenitor | 0.8522 | 0.641 (1) | 0.422 (5) | 0.996 |
| cycling | 0.8561 | 0.402 (13) | 0.507 (2) | 0.981 |
| immune | 0.8389 | 0.409 (15) | 0.498 (15) | 0.947 |
| secretory | 0.8433 | 0.404 (13) | 0.525 (4) | 0.971 |
| stromal | 0.8630 | 0.476 (10) | 0.553 (2) | 0.979 |
| terminal | 0.7728 | 0.416 (4) | 0.474 (1) | 0.957 |

## The checks
- **C1 platform — FAILED on the raw scale.** On all 17 cells measured on both, arrays read h higher than WGBS by +0.03 to +0.17 (median +0.11):
  arrays never read a true 0 or 1. With WGBS put on the array scale by its measured source term, the median gap is +0.005 but single cells
  still differ by −0.057 to +0.063. The floor therefore depends on the platform by more than the ±0.05 tolerance.
- **C2 depth — FAILED at the 0.02 level.** WGBS median h rises 0.408 → 0.415 → 0.426 at depth ≥ 10, 20, 30: shallow reads round loci to
  exactly 0 or 1. A needs a depth correction (the binomial bias of H is computable) before it is a physical number.
- **C4 classes — the eight do not separate on A.** On WGBS, cycling 0.402, immune 0.409, secretory 0.404 and terminal 0.416 lie within
  0.014 of each other, while cells within one class spread by SD 0.03–0.07. Only stromal (0.476) stands apart; pluripotent (1 cell) is now the
  LOWEST, not the highest.
- **B** is 0.90–1.00 everywhere: the entropy of a whole-methylome mean is near one bit and carries no cell information, as expected.

## What this decides
1. Option A cannot replace the floors today: its value moves with platform and depth by more than the gauge's tolerance.
2. On whole methylomes the eight classes are not eight entropy levels. Either the class floors are a property of specific loci, not the
   whole methylome, or the class structure is not what the floors say. That is the question CLASS-COUNT-01 was meant to answer, now
   with a measurement behind it.
3. Before any refit: a depth-corrected entropy (exact binomial correction per locus), measured on WGBS only (the platform with true 0 and 1),
   and reported per cell with its interval. Then the class question is asked of those numbers.

## Follow-up (PROC-HMIN-REFIT-02, same day): depth-corrected, sequencing only
183 WGBS samples. Self-test with no population: each sample's own loci with ≥ 40 reads, thinned to 10–30 reads.
- The plain estimate reads low (−0.052 at 10 reads); first-order bias correction reads low by half that (−0.026); the Bayesian estimate reads
  high by the same amount (+0.025). The true value sits between the last two: at the samples' real depth the remaining uncertainty is about ±0.01.
- Depth-corrected whole-methylome entropy per class (cells): cycling 0.438 (13), immune 0.445 (16), secretory 0.442 (13), terminal 0.436 (5),
  pluripotent 0.420 (1), stromal 0.500 (11), progenitor 0.551 (2, spread 0.38–0.65).
- Within a class, cells spread by SD 0.02–0.06. Five classes lie within 0.025 of each other. **On whole methylomes a cell holds about
  0.44 bits per CpG whatever its class; stromal cells hold about 0.50.** The eight class floors do not appear as eight whole-methylome levels
  under any of the three estimators.
