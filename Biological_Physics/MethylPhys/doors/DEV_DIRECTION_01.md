# DEV-DIRECTION-01 — stage 10 directional decomposition (development; check written 2026-10-03 before any data were read)

**Commissioning step 4.** Check: on replicates it returns no direction; on a known treated series it returns the direction the treatment is
known to push. Arrays that would be used: GSE250556 pooled replicates (no direction expected); DNMT-inhibitor series in the bucket — GSE123140
(AML cell line, azacitidine vs vehicle), GSE165185 (azacitidine), GSE187291 (decitabine vs DMSO): known push = loss of methylation, i.e. toward
disorder at methylated identity sites.

**Module as built.** `Runtime Matrices/Directional Panel/bidirectional_decomposition.py` with `directional_panels_v1_0.json`: per-CpG
z = (beta - mean_hc_train) / sd_hc_train, multiplied by a disease direction (+1 disease-up / -1 disease-down) from VAL-051, averaged per class.
Before reading any data: the z is taken against a healthy training-set mean and SD (a population term, SOP section 3 rule 5), the sign is a
disease direction, and the output is per class, not per cell. The check asks for "toward disorder / toward over-order"; the module's output axis
cannot express it. **The check is not assessable on the module as built; stage 10 is not run and not wired. Author decision needed** on a
physics-only definition (e.g. per identity site the signed move of beta toward 0.5 or away from it, against the cell's own floor pattern).

---
## Outcome
Not run (reason above, recorded before data). No reading taken.
