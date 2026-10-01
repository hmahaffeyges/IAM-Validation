# IAM-A floor for neutrophils — which floor is defensible (development, 2026-10-01)

**Asked:** "test both and see which one is more defensible and the true IAM physics way."
**Data:** Loyfer 2023 blood granulocytes (≥ 90 % neutrophils), 3 donors, single-molecule reads (.pat). Copy error ε = an isolated unmethylated CpG between
two methylated neighbours, on molecules with ≥ 6 CpGs and ≥ 80 % methylated. Counts: 16–20 M opportunities per donor.
IAM-A = H(ε)/H(floor). ε₀ = 1/(1+e^(φM)) = 0.0320 (canon).

| | (a) bare physics floor ε₀ | (b) own healthy baseline (other donors) | (c) ε₀ × frozen neutrophil position P |
|---|---|---|---|
| derived or measured | derived (φM from the law) | measured, 2 donors per reading | ε₀ derived; P measured once and frozen (P = 1.099, range 1.084–1.108) |
| healthy donors in Normal (0.95–1.05) | **0/3** (1.084, 1.127, 1.084) | 3/3 (0.978–1.040) | 3/3 (same numbers as b) |
| repeat (odd vs even molecules) | max diff 0.003 | 0.002 | 0.002 |
| 2 % known copy damage, read above 1.05 | 3/3 (median 1.425) | 3/3 (1.289) | 3/3 |
| same healthy blood, other lab/pipeline (ENCODE B, mono, NK, T) | **0.70–0.79** uncorrected (0.67–0.77 corrected) | n/a (cancels) | cancels |

**What decides it.**
1. **The bare floor reads the lab, not the cell.** The same kind of healthy blood cell reads 0.70–0.79 on ENCODE's pipeline (uncorrected, like for like) and 1.08–1.13 on Loyfer's.
   Absolute ε depends on read length, chemistry and molecule filter. Dividing by a fixed absolute number carries all of that into the reading.
2. **The floor is still the anchor.** Across 56 healthy cell types (same pipeline as the Loyfer granulocytes), the median on ε₀ is **1.009**. The law's floor is where healthy cells sit
   as a group, and each cell type holds its own fixed position near it: neutrophils 1.10 ± 0.01 across donors (CV 1.2 %).
3. **(c) keeps both.** The reading divides by ε₀ × P. ε₀ is the physics floor and P is the cell architecture's stated position, measured on healthy cells
   with the same instrument and frozen. Instrument effects then cancel the same way they do for Met-A. This is the Met-A structure (the cell's own floor), with the physics floor explicit.

**Recommendation: (c).** IAM-A(neutrophil) = H(ε) / H(ε₀·P_neu), with P_neu frozen from healthy granulocytes on the same pipeline. A same-run tare applies when references exist.
**Limits:** 3 donors; uncorrected ε (.pat files carry no sequence, so sequencing error can't be subtracted); one pipeline. P must be re-measured for any
other read-level pipeline before it's used there. The author's decision is needed before the canon changes.
