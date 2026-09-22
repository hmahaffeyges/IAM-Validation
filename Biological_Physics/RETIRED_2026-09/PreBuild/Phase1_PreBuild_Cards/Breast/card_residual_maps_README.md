# Per-Card Disease Residual Maps

**Layer 3 base maps** — per-CpG residual signatures per card cohort. Where the disease signal lives at sub-cellular resolution after cellular composition has been factored out.

## What a residual map is

For each card, every patient runs through Stages 1+2 of the EDEAR pipeline:
1. **Stage 1 (Walther IAM Deconvolver)** produces per-class fractions for that patient.
2. **Stage 2 reconstruction**: at each marker CpG, predicted β = Σ (class_fraction × class_reference_β).
3. **Per-CpG residual** = observed β − reconstructed β. This isolates the disease-specific signal from cellular-composition variation.
4. **Per-CpG case-vs-HC Cohen's d** on the residuals identifies the loci where the card's disease signature lives — orthogonal to whatever cellular composition shifts have already been captured at the class-A-score level.

## Files in this folder

| File | Card | Cohort | Tool | N (case/HC) | Concordant CpGs |
|---|---|---|---|---|---|
| `breast_epic_residual_map_v0_1.csv` | breast-epic | GSE51057 + GSE51032 >10yr breast pre-dx | production Walther IAM Deconvolver | 47 / 601 | 1,392 (|d|>0.3 both cohorts, same sign) |

Each CSV has columns: `cpg`, `d_GSE51057`, `d_GSE51032`, `concordant_strong`, `mean_abs_d`, optionally `in_xu538`.

## Breast-epic v0.1 — key finding (2026-05-29)

- **1,173 hypomethylated CpGs (case β < reconstructed β)** vs only 219 hypermethylated — 5.4-to-1 ratio.
- **Dominant signature is loss of methylation** below cellular-composition expectation at 10+ years pre-diagnosis. Classic field-effect cancerization signature, quantified.
- **1,389 NEW candidate CpGs** not in the Xu-538 disease-trained panel. Candidates for an expanded breast-epic panel that complements Xu-538.
- Top concordant loci: cg20124336 (d=−2.17/−1.89), cg16188349 (d=−1.67/−1.67), cg27467249 (d=−2.17/−1.17). All hypomethylated, all replicating across cohorts.

## How the card consumes these maps

For each customer through the breast-epic card:
1. Production deconvolver → class fractions
2. Reconstructed β at the 7,114 deconvolver class markers
3. Per-CpG observed − reconstructed = patient residual vector
4. Compare patient residual at the concordant 1,392 CpGs to the cohort residuals here
5. Per-CpG z-score → card-specific layer 3 evidence
