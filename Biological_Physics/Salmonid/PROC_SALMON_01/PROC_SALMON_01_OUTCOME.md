# PROC-SALMON-01 — outcome (2026-10-01)

Pre-registration: PROC_SALMON_01_PREREG.md (sha 4e65a4c06246cf56). Data: Gavery et al. 2018 (G3), PRJNA325786; 20 Methow River steelhead males
(10 hatchery, 10 natural), RBC and sperm, RRBS (PBAT), all 40 specimens aligned (Bismark --pbat, Omyk_1.0) and scored with score_salmon.py.

**Deviations, recorded before scoring.** Per-fish age and hatchery (WH/WNFH) are not in the public metadata, so P3's within-age-4 check could
not be run. Published DMR coordinates were not obtained, so P4 reports sites above the look-elsewhere threshold only. Run halves are split by read
parity within each specimen (the extractor's 'parity' mode), not by sequencing run.

| prediction | result | |
|---|---|---|
| P0 instrument consistency | ICC 0.998; median half-difference 0.00011 vs between-fish SD 0.0025 (RBC) / 0.0029 (sperm) | **pass** |
| P1 fish differ more than halves | between-fish SD 0.0025 / 0.0029 vs within 0.0005 / 0.0002 | **pass** (but see below) |
| P2 temperature: RBC ε_corr in 0.021–0.027 | median RBC 0.0354 (3.31 kT) | **fail** |
| P3 hatchery vs natural | RBC 0.0356 vs 0.0352, p 0.31; sperm 0.0165 vs 0.0184, p 0.34 | **no difference** |
| P4 difference map | 82,006 RBC / 83,245 sperm sites; max |z| 6.0 / 5.9 vs thresholds 16.0 / 23.0 | **no site above threshold** |

**Readings.** RBC ε_corr 0.028–0.037 (median 0.0354, E = 3.31 kT); sperm 0.014–0.024 (median 0.0177, E = 4.02 kT). Conversion failure
0.004–0.014; sequencing error 0.006–0.008.

**What the fish-to-fish spread is.** It tracks the library, not the fish: within RBC, ε_corr falls with conversion failure (Spearman −0.58,
p 0.007) and with masked-site count (−0.69, p 0.001); within sperm with conversion failure (−0.54, p 0.014) and depth (−0.47, p 0.04); one
sperm sequencing lane reads higher (0.0224 vs 0.016–0.019). The same fish's RBC and sperm readings do not correlate (ρ 0.06, p 0.81). P1 passes
on its stated terms, but the between-specimen spread is dominated by library/batch, so a per-fish trait is not shown.

**What this says.**
1. The reading is precise within a specimen (P0) — the instrument side works on salmon RRBS.
2. Steelhead RBC at ~10 °C read 3.31 kT, the same as healthy human cells at 37 °C (3.29–3.51 kT on this statistic). The fixed-energy prediction
   (gap constant in joules, so larger in kT units when colder) fails. Either the copy error is held at a fixed value in kT units (temperature-
   compensated), or RRBS CpG-dense regions differ from the whole-genome molecules the human value came from — the limit stated in advance. A
   whole-genome bisulfite fish sample, or one species at two temperatures, separates these.
3. No hatchery/natural difference in copy error or in any single site, on these 20 fish.
4. Sperm holds its pattern more tightly than RBC (4.0 vs 3.3 kT).
