# What moves a leukocyte on the gauge — the two ends are different physics (2026-09-27)

Author: "Just because the patient may be sick, the immune cells working hard to make repairs… that doesn't mean that the informational
fidelity is compromised." And: "an elevated A score is not the same as the severely suppressed A score. When it can no longer maintain its
identity that is when the score is low. When it can no longer find enough 2D surface to write because the printer is in overdrive… that
is when the tumour forms, just like that black hole."

**The two ends.** A = H(mean β over the cell's identity loci) / H_min of its class.
- **SUPPRESSED (A < 0.95)** — the cell holds less entropy than its class floor: the write process has slowed or stopped, identity
  maintenance is failing, the cell is running down. Senescence, exhaustion, a lineage dying out.
- **ELEVATED (A ≥ 1.05; Warburg line 1.07; BREACH 1.10)** — the write process is in overdrive: more being written than the encoding
  surface holds, the surface saturating. This is the tumour direction — the black-hole analogue (Mahaffey number at saturation).
- **Activity is neither.** A neutrophil clearing an infection does more work while keeping its identity; the physics does not predict
  that a busy immune cell moves on the gauge, and nothing on the report may say so.

| # | mechanism | end of the gauge | which cells | the test |
|---|---|---|---|---|
| 1 | **The leukocyte is the tumour**: leukaemia (AML, CML, ALL, CLL), lymphoma, myeloma; clonal haematopoiesis (CHIP) as the pre-malignant form | **ELEVATED** — overdrive on that lineage's identity loci | the affected lineage; the others read 1.00 (CLL: B-cell A rises, neutrophils do not). The diseased cell is the majority cell in the tube — the fraction confound cannot fake it | PROC-BLOODCANCER-01: public 450K whole-blood / PBMC AML and CLL cohorts, per-cell A on the affected lineage vs the others |
| 2 | **Ageing** | **ELEVATED — measured, not assumed.** The retired 1,379-donor curve rose by decade and the oldest Uppsala arrays read 1.037 (PROC-STAGE2D-03). The pre-build era had this backwards (author, 2026-09-27): an old immune system is on the overdrive side — more written over a lifetime onto the same surface, less headroom; the same direction as CHIP | slow, per person, over years | SATSA (E-MTAB-7309): 287 people, 2–5 draws over 10–19 years — serial mode measures it in the same person |
| 3 | **T-cell exhaustion / immunosenescence** (AD) | **SUPPRESSED — the AD-era record says so, on the pre-atlas surface**: CPG-VAL-008 read uniformly negative immune A in AD; the covariance PC2 was a T-cell axis with AD displaced toward *reduced* T-cell readout (67 % of healthy inter-sample variance). Ageing in healthy cohorts reads ELEVATED (row 2): two mechanisms on two ends, not one phenomenon read backwards | CD4 / CD8 specifically; neutrophils and monocytes at or above 1.00 in the same specimen | PROC-DIRECTION-01 on the AD cohorts on disk, per cell against 1.00 |
| 4 | **Emergency haematopoiesis**: marrow pushes immature cells out under a tumour or severe inflammation | composition first (progenitor / immature fraction up); immature cells not yet fully written may read **SUPPRESSED** | progenitor class; the immune class as a whole | composition + progenitor-class reading on the same cohorts; not an A claim on mature cells |

**What this says about today's 86-year-old arrays** (immune A 1.037, 5 % non-blood-like mass, PROC-STAGE2D-03): the elevated side,
consistent with ageing as measured (row 2) and with what clonal haematopoiesis would look like on this gauge (row 1's pre-malignant
form). Whether old blood's elevation is the ageing slope or a clone riding on it is what per-cell trajectories in SATSA's oldest
draws can separate: a slope is every cell together; a clone is one lineage leaving the others.

**What the leukocyte gauge does NOT do.** Detect a solid tumour through the blood cells' A. Immune cells do not take on a tumour's
overdrive by fighting it. A solid cancer, if it shows in whole blood at all, shows through composition (row 4) or shed cells — and
PROC-STAGE2D-03 measured that shed cells below ~2 % of the DNA are below what this array can see in whole blood.

**Where the 'bidirectional' picture came from.** The AD-era tests (GIFT AD d = +0.68, PSP −0.38, the directional panel) were the first
place the immune metric moved in opposite directions by condition — but those were pooled-class, case-minus-control numbers, pre-atlas
and pre-per-cell. Two things produce 'bidirectional' there without any cell going both ways: pooling (one cell up, another down, netting
to either sign with composition) and the comparison itself (a control group sitting above or below 1.00 flips the sign of a departure).
Neither exists on the commissioned chain. The record itself (AD directional-score principle): "AD has bidirectional per-CpG drift; pooled entropy is NULL;
directional weighting recovers d = +0.62." The cancellation was at the LOCUS level inside one pooled metric — some CpGs drifting up,
some down, H(mean β) over all of them netting to nothing — and, the author's reading, at the CELL level too (T cells one way, B cells or
neutrophils the other, netted by a composition that itself moves with disease and age). Either way the pooled surface was blind; the
biology was not ambiguous. On the commissioned chain each cell's identity loci are a narrow band near H_min_β chosen so the statistic
is unimodal, and the per-CpG opposite drifts that nulled the pooled metric do not share a mean there. **PROC-DIRECTION-01** re-reads those
cohorts per cell against 1.00: if T cells read 1.06 and B cells 0.97 in one specimen, that prints as two rows and nothing cancels.

**Status.** Hypotheses with tests, not findings.
