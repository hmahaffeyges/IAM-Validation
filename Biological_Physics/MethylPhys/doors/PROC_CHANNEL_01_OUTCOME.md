# PROC-CHANNEL-01 — each cell type's error budget and holding energy, from single DNA molecules (2026-09-30)

Loyfer 2023 read-level WGBS (hg19 .pat), 56 cell types, 153 samples, the same 399 windows (798,000 CpGs across all autosomes) read for every sample:
median 2.96 M read-patterns and 561,000 adjacent CpG pairs per sample. No labels and no floor used in the measurement.

## 1. Holding energy per site: nearly the same in every cell type
Copy error ε = isolated unmethylated CpGs inside otherwise methylated molecules; de novo error = isolated methylated CpGs inside unmethylated molecules.
Boltzmann: E_hold = kT·ln((1−ε)/ε). Share of one ATP (M = ΔG_ATP/RT = 20.94): φ = E_hold / M.

| channel | ε across 56 cell types | E_hold (kT) | φ (share of one ATP) |
|---|---|---|---|
| methylated sites (copy error) | 0.024–0.042 | 3.41 ± 0.12 (3.13–3.71) | **0.163 ± 0.006** (0.150–0.177) |
| unmethylated sites (de novo error) | 0.009–0.016 | 4.37 ± 0.13 (4.11–4.66) | **0.209 ± 0.006** (0.196–0.222) |

Lowest error: naive T cells, macrophages, cortical neurons (φ 0.173–0.177). Highest: smooth muscle, heart fibroblasts, erythroid progenitors (0.150–0.152).
Stated before this result arrived (to the author, 2026-09-30), without a numeric threshold: "similar φ across cell types."
Limits: (a) E is a logarithm, so it compresses spread — the error rates themselves differ by up to 1.75×; (b) the isolated-error counts include
sequencing and bisulfite-conversion error, a technical floor not yet subtracted; (c) one reference: WGBS from one consortium.

## 2. Do cells group by error fingerprint? (neighbour agreement, copy / de novo errors, erasure and gain runs, discordant reads)
- Best separation: 2 groups (silhouette 0.53). Most reproducible on held-out samples: 4–5 groups (ARI 0.50–0.52). Eight groups: ARI 0.22.
- Eight groups vs the draft eight classes: ARI 0.234 (relabelling null p95 0.045) — well above chance, far from identical.
- The five groups: (i) resting immune + macrophages, lowest copy error (0.0275), highest neighbour agreement (0.866); (ii) activated / effector and
  memory lymphocytes (0.0349); (iii) epithelial and secretory cells of gut, stomach, pancreas, liver, lung, thyroid (0.0326); (iv) endothelium, kidney,
  adipocytes, muscle, neurons (0.0323); (v) fibroblasts, smooth muscle, erythroid progenitors, highest error and most discordant molecules (0.0395).
- Immune splits by activation state, not lineage: the error budget follows function.

**Reading.** The error fingerprint carries class-related structure but does not reproduce eight as the stable count; four to five groups are
reproducible. The holding energy per site is close to universal (φ ≈ 0.16 methylated, 0.21 unmethylated). Next: subtract the technical floor with
spike-in or unmethylated-control data, pre-register the cross-species prediction (normal ε shifts with M), and test the energy reading of the
Warburg drift against measured ATP output.

## 3. Can one physics-only floor read every healthy cell? (2026-09-30, exploratory)
Floor from the mean holding energy alone: ε₀ = 1 / (1 + e^(φ·M)), with φ the 56-cell mean and M = 20.94. No reference person, no cohort.

| channel | φ | ε₀ | healthy cell types on this floor | inside Normal (0.95–1.05) |
|---|---|---|---|---|
| methylated | 0.1629 | 0.0320 | A 0.794–1.225 (median 1.009) | 27 / 56 |
| unmethylated | 0.2088 | 0.0125 | A 0.793–1.237 (median 0.996) | 20 / 56 |

Lowest: naive CD8 and CD4 T cells, cortical neurons (non-dividing / quiescent). Highest: smooth muscle, heart fibroblasts, erythroid progenitors
(dividing). One universal floor is not yet precise enough for an individual reading; the residual spread follows how much the cell divides — the
candidate physical term (copy error accrues per division). To test: the floor as φ·M plus a division term from each cell type's known turnover.

## 4. Does turnover explain the spread? (2026-09-30) — NO
Independent cell lifespans: Sender & Milo 2021, Nat Med (github.com/milo-lab/cellular_turnover, Summary.xlsx). 26 of the 56 cell types matched.
Spearman ρ(shorter lifespan, copy error) = 0.03 (p 0.87); de novo error −0.32 (p 0.12); neighbour agreement −0.15; discordant reads 0.10.
Colon epithelium (3.4-day lifespan) carries copy error 0.031, cardiomyocytes (56,000 days) 0.031. Monocytes (3.5 d) 0.028, neurons 0.024.
The spread in copy error is not set by how often the cell is replaced. The suggestion in section 3 — that the residual follows division — is not
supported. Cell lifespan counts replacement of the differentiated cell, not divisions in its lineage; a lineage-division count is a separate test.

## 5. Do copy errors concentrate where the cell works? (PROC-WORK-01, 2026-09-30) — NOT overall
Working regions = CpGs a cell keeps unmethylated (β ≤ 0.3) while the median other cell type keeps them methylated (≥ 0.7): 979–23,159 per cell.
Copy error on the cell's methylated CpGs within 50 CpGs of them vs in windows with none; control = the same sites in every other cell type.
55 cell types evaluable. Near/far ratio in the working cell ÷ the same ratio in idle cells ("excess"): median 0.975
(IQR 0.932–1.038); Wilcoxon p = 0.231.
Sites near working regions carry LESS copy error than far sites in 50 of 55 cells, in working and idle cells alike: a property of the
regions, not of the work. Excess > 1.10 only in: bladder epithelium, colon enteroendocrine, colon epithelium, pancreatic acinar, small intestine epithelium, t effector memory cd4.
Limit: "work" here = regulatory activity; the operation of methylation maintenance itself is copying at replication, not measured by this test.

## 6. One physics floor for every cell: do the cells' positions follow their jobs? (2026-09-30)
Floor per channel from the mean holding energy only: ε₀ = 1/(1+e^(φM)), φ_meth 0.1629, φ_unmeth 0.2088, M 20.94 (ε₀ 0.0320 / 0.0125).
A = H(ε_cell)/H(ε₀), each healthy cell type read on the same floor. Variance in A explained by the draft classes (56 cells; 6 of the 8 classes present):
methylated 0.28 (permutation null p95 0.19, p 0.003); unmethylated 0.30 (null p95 0.19, p 0.002). By the five error-fingerprint groups: 0.67 / 0.40
(circular: the groups were built from these error rates).

| draft class (n) | A_meth median (range) | A_unmeth median (range) |
|---|---|---|
| cycling (13) | 0.999 (0.912–1.127) | 0.943 (0.830–1.035) |
| immune (15) | 0.901 (0.794–1.115) | 1.080 (0.891–1.183) |
| progenitor (1) | 1.175 (1.175–1.175) | 1.150 (1.150–1.150) |
| secretory (13) | 1.004 (0.935–1.139) | 0.963 (0.793–1.119) |
| stromal (10) | 1.039 (1.006–1.225) | 1.006 (0.937–1.237) |
| terminal (4) | 0.961 (0.803–1.060) | 0.965 (0.946–1.075) |

Reading: on one floor from physics, healthy cells sit at different places, and part of that placement (about 30 %) follows the job classes; immune cells
sit LOW on the methylated channel and HIGH on the unmethylated one — a two-channel signature no single floor or single A shows. With a fixed floor,
any change of state multiplies the cell's position by its own-standard A, so directions are preserved by construction; what the physics floor adds is
each architecture's absolute position. Floor from sequencing data applies to sequencing reads only (arrays compress β).

## 7. The same bits in every cell: architecture error on a common benchmark (2026-09-30)
To remove what each cell keeps methylated (its program), copy error was read on the 26,800 CpGs that ALL 56 cell types keep methylated
(β ≥ 0.8, ≥ 10 methylated-read coverage in every cell) — the cell's equivalent of running one benchmark circuit on every platform.
- Copy error on the common sites: median 0.0165, range 0.0136–0.0226 (1.66× from lowest to highest). Rank agreement with the all-site copy error ρ = 0.87.
- The draft classes explain **51 %** of the variance on common sites (permutation null p95 0.20, p < 0.0002) — up from 28 % on all sites.
- Lowest: naive CD8 T cells, lung macrophages, B cells (0.0136–0.0139). Highest: smooth muscle, heart fibroblasts, erythroid progenitors (0.0201–0.0226).
- Class medians: immune 0.0149, cycling 0.0154, terminal 0.0164, secretory 0.0166, stromal 0.0174, progenitor 0.0226.
**Reading.** On identical bits, cells still differ in how well they hold them, and the difference follows the classes: the architecture difference
is in the cell's own maintenance, not in which regions it keeps. Instrument error (≈ 0.1–0.3 % per substitution in ENCODE; not measurable in
these files) is an order of magnitude smaller than the spread, but has not been subtracted here.
- Replicate agreement (per sample, same 26,800 common sites): 53 cell types with ≥ 2 samples (150 samples, different donors): intraclass
  correlation 0.80; within-cell SD 0.0008 vs between-cell SD 0.0017. Not driven by coverage (ρ = −0.11). The architecture error is a stable
  property of each cell type across people.

## 8. Cross-check against the copying enzyme's measured selectivity (2026-09-30, literature, not pre-registered)
Hopfield's relation: an enzyme discriminating right from wrong with an energy gap ΔE makes errors ≈ e^(−ΔE/kT), so ΔE = kT·ln(selectivity).
Published in-vitro DNMT1 preference for hemimethylated over unmethylated CpG: 7–21× (Pradhan 1999), ~17× (truncated, CXXC-containing),
30–50× (several reports), 80× on average across flanking sequences (comprehensive analysis, 2023) → ΔE = 1.9–4.4 kT.
Measured holding energy per site: copy channel 3.41 kT (Loyfer, uncorrected), 3.77 kT (ENCODE normal, instrument-corrected); de novo channel 4.37 kT
(Loyfer, uncorrected; mostly instrument after ENCODE correction).
**Reading.** The cell's measured holding energy falls inside the range set by the copying enzyme's own measured selectivity — same order, from
independent biochemistry. Limits: HM/UM selectivity maps most directly onto the de novo (unmethylated-site) channel, whose cellular value is not yet
separable from the instrument; in the cell UHRF1 targeting, DNMT3 and TET also act. A consistency check, not a confirmation.

## 9. Do class floors read cell types they were not calibrated on? (2026-09-30, leave-one-cell-type-out)
Each healthy cell type hidden in turn; its floor predicted from the other members of its draft class; Normal = A within 0.95–1.05.
- Error on common bits (sequencing): class floor puts 29/55 in Normal (0.55); one floor for all 0.45; shuffled classes p95 0.55 (p 0.054).
  Immune 0.27, cycling 0.62, secretory 0.69, stromal 0.70, terminal 0.50.
- All-site copy error (sequencing): class floor 27/55 (0.49); one floor 0.53; shuffled p 0.35. Immune 0.33, cycling 0.62, secretory 0.46, stromal 0.80, terminal 0.00.
- Old-A quantity H(mean β) (Loyfer): class-median floor 0.55; G-002's most-methylated-cell rule 0.22; one floor 0.60.
**Reading.** Class floors carry real information (classes explain 51 % of error on common bits) but do not yet read an unseen cell type to ±5 %.
On common bits immune is the weakest class (0.27): it splits by activation state (resting vs effector), so one immune floor cannot serve both. On all-site copy error terminal is weakest (0/4; only 4 cell types), then immune (0.33).

**Reproducibility (2026-10-09).** The per-cell genome-wide table this note reports is committed (`doors/data/PROC_CHANNEL_01/
channel_cells_genomewide.csv`, recovered from the author's 2026-09-30 copy); `derive_constants.py` reproduces every value above and the
CANON constants E_hold_meth 3.41, phi 0.1628 and eps0_meth 0.032 from it. The job that made the table (channel.py, window generator, run
script) is rebuilt from the session record; the per-sample table is to be regenerated by rerunning it. Range top corrected 3.72 to 3.71
(3.7146).
