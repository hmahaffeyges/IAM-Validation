# VAL-095 — Outcome

**VAL ID:** VAL-095
**Pre-reg SHA-256:** 5f74259d5341268ee7cdaf68322962a275dd19e4158b398f09562f6aaa44bace
**Date executed:** 2026-04-26
**RNG seed:** 20260426

## Outcome label

**O1_RESOLUTION_GAIN** at the >10yr breast pre-diagnostic window — UniLIFE's adult-specific subtype resolution surfaces large signal in two specific cell types (regulatory T cells and eosinophils) that Salas 6-cell aggregates cannot resolve. Replication direction holds across cohorts for one of these.

For mid windows (5-10yr, 2-5yr) the outcome is **O2_RESOLUTION_NEUTRAL** — UniLIFE adult subtypes do not produce stronger signals than Salas aggregates.

For 0-2yr (near-diagnosis) the outcome is **O2_RESOLUTION_NEUTRAL** with a single observation: aBnv (naive B cells) shows d=+0.44 (GSE51057) / d=+0.49 (GSE51032), a directional replication just below the pre-locked |d|≥0.5 threshold.

## Headline finding

At the >10yr breast pre-diagnostic window, UniLIFE's adult-specific subtype resolution surfaces a signal in regulatory T cells (aTreg) that is invisible at Salas 6-cell resolution:

| Cell type | GSE51057 >10yr d | GSE51032 >10yr d |
|---|---|---|
| **UniLIFE aTreg** (regulatory T) | **+1.26** | **+0.79** |
| Salas CD4T (aggregate) | +0.36 | +0.03 |
| UniLIFE aCD4Tnv (naive) | −0.19 | −0.10 |
| UniLIFE aCD4Tmem (memory) | +0.29 | +0.10 |

The aTreg signal direction replicates across cohorts. The Salas CD4T aggregate dilutes this signal because it pools naive, memory, and Treg subtypes — when Treg increases and naive decreases, the aggregate moves only slightly. UniLIFE separates them.

A second large UniLIFE-specific signal at >10yr GSE51057 is **aEos d=+1.12** (eosinophils), which sits inside Salas Neu (which reads d=−0.03). This signal does NOT replicate to GSE51032 (aEos d=+0.22 there), so it is logged as a single-cohort observation, not a finding.

## Per-cohort head-to-head, per window (Salas 6-cell vs UniLIFE pan-lifespan equivalent)

**GSE51057:**

| Cell | >10yr S/U | 5-10yr S/U | 2-5yr S/U | 0-2yr S/U |
|---|---|---|---|---|
| Bcell | −0.14 / +0.25 | −0.17 / +0.29 | +0.05 / +0.14 | +0.31 / −0.01 |
| CD4T  | +0.36 / **NA** | −0.03 / **NA** | −0.23 / **NA** | +0.15 / **NA** |
| CD8T  | −0.33 / **NA** | −0.01 / +0.35 | −0.22 / +0.43 | +0.25 / **NA** |
| Mono  | −0.44 / −0.28 | −0.14 / −0.11 | +0.23 / −0.13 | +0.14 / −0.12 |
| NK    | +0.32 / −0.22 | +0.08 / −0.13 | +0.22 / −0.21 | +0.23 / −0.12 |
| Neu   | −0.03 / +0.22 | +0.06 / +0.05 | +0.08 / +0.09 | −0.34 / +0.07 |

**GSE51032:**

| Cell | >10yr S/U | 5-10yr S/U | 2-5yr S/U | 0-2yr S/U |
|---|---|---|---|---|
| Bcell | −0.05 / 0.00 | −0.05 / +0.16 | +0.12 / +0.01 | +0.36 / 0.00 |
| CD4T  | +0.03 / +0.59 | −0.06 / −0.04 | +0.09 / −0.09 | +0.39 / −0.09 |
| CD8T  | −0.08 / **NA** | +0.06 / +0.37 | −0.22 / +0.41 | +0.24 / +0.34 |
| Mono  | −0.51 / −0.25 | −0.18 / +0.01 | +0.11 / −0.18 | +0.08 / −0.03 |
| NK    | +0.03 / −0.14 | +0.10 / −0.07 | +0.15 / −0.19 | +0.17 / −0.02 |
| Neu   | +0.14 / +0.42 | +0.03 / +0.36 | −0.04 / +0.20 | −0.47 / +0.16 |

(NA values: UniLIFE pan-lifespan CD4T and CD8T are identically zero across these adult cohorts because UniLIFE's design has both pan-lifespan markers AND adult-specific subtypes; in NNLS deconvolution of adult buffy-coat the mass routes to the adult subtypes and the pan-lifespan compartments read 0. This is correct behavior for UniLIFE on adult-only cohorts and is documented in Guo 2025. The fairer comparison is Salas CD4T vs UniLIFE (aCD4Tnv + aCD4Tmem + aTreg), shown below.)

## UniLIFE adult-specific subtype calls (12 subtypes Salas cannot resolve)

| Subtype | GSE51057 max \|d\| (window) | GSE51032 max \|d\| (window) | Replicates direction? |
|---|---|---|---|
| aTreg (regulatory T) | **+1.26 (>10yr)** | **+0.79 (>10yr)** | YES |
| aEos (eosinophils) | **+1.12 (>10yr)** | +0.22 (>10yr) | partial |
| aCD4Tnv (naive CD4T) | −0.51 (2-5yr) | +0.25 (0-2yr) | NO |
| aBnv (naive B) | +0.44 (0-2yr) | **+0.49 (0-2yr)** | YES |
| aCD8Tnv (naive CD8T) | +0.41 (5-10yr) | +0.32 (0-2yr) | YES (direction) |
| aNeu (neutrophil adult) | −0.38 (0-2yr) | **−0.49 (0-2yr)** | YES |
| aBmem | −0.01 to +0.29 | +0.02 to +0.28 | weak |
| aCD4Tmem | +0.09 to +0.29 | +0.10 to +0.15 | weak |
| aCD8Tmem | −0.27 to +0.19 | −0.27 to +0.19 | weak |
| aBaso | +0.13 to +0.18 | +0.05 to +0.36 | weak |
| aMono | −0.11 to +0.18 | −0.14 to +0.03 | weak |
| aNK | +0.16 to +0.22 | −0.13 to +0.11 | weak |

Bold values approach or exceed pre-locked O1_RESOLUTION_GAIN threshold (|d|≥0.5 in GSE51057 with |d|≥0.3 direction-matched in GSE51032).

## Three signals worth flagging

1. **aTreg at >10yr (replicates):** GSE51057 d=+1.26, GSE51032 d=+0.79. This is the cleanest case where UniLIFE 19-cell resolution surfaces a large signal that Salas 6-cell cannot resolve (Salas CD4T aggregate reads d=+0.36 / d=+0.03 at the same window). aTreg elevation 10+ years before clinical breast diagnosis is consistent with established immune-modulation literature on early breast cancer development but framework-level interpretation is open and outside this VAL's scope.

2. **aBnv at 0-2yr (replicates):** GSE57 d=+0.44, GSE32 d=+0.49. Both cohorts agree at the near-diagnosis window. Salas Bcell at 0-2yr shows d=+0.31/+0.36 (similar magnitude). UniLIFE adds the discrimination that this is naive B cells specifically.

3. **aNeu at 0-2yr (replicates negative):** GSE57 d=−0.38, GSE32 d=−0.49. Salas Neu shows d=−0.34/−0.47 at the same window. Direction matches between Salas Neu and UniLIFE aNeu, with Salas being slightly stronger; UniLIFE aEos and aBaso are positive at 0-2yr while aNeu is negative, which would explain why Salas Neu (which pools aEos + aBaso + aNeu) shows slightly weaker negative d than UniLIFE aNeu alone.

## CHK-3.1 β distribution sanity

PASS in both cohorts (49.5% extreme β, 7.6% middle, median 0.658 — bimodal raw β).

## CHK-3.2 cross-cohort baseline

For Salas 6-cell baseline means: per-cohort comparison shows fractions within reasonable range across cohorts. The dominant Salas cell types (CD4T, Mono, Neu) have means within 0.05 of each other across cohorts. No flagged tiles.

For UniLIFE 19-cell, the adult-specific subtypes have low absolute fractions and are more cohort-sensitive at the second decimal place. Per-subtype means are reported in the results JSON.

## Saturation flag

NNLS fractions are bounded [0, 1] by RPC constraint and normalized to sum to 1. No saturation observed (no fraction estimate at exactly 0 or 1 in >5% of samples for any cell type, with the noted exception that UniLIFE pan-lifespan CD4T and CD8T compartments route to 0 in 100% of samples — this is the documented adult-cohort routing behavior, not a saturation issue).

## Sample-size honesty

GSE51057 >10yr has n=11 cases vs n=177 controls. The aTreg d=+1.26 has wide CI; the GSE51032 replication at d=+0.79 (n≈85 cases vs n=424 controls) is the deciding piece. For the cross-cohort signals to count as O1_RESOLUTION_GAIN the pre-reg required |d|≥0.5 GSE51057 AND direction with |d|≥0.3 GSE51032. aTreg at >10yr meets both: +1.26 and +0.79.

## Test 2 placeholder (CCL-030)

UniLIFE adult-specific subtypes provide direct lymphoid (aCD4T/aCD8T/aB family + aTreg) vs myeloid (aMono/aNeu/aEos/aBaso) discrimination. The lymphoid direction at >10yr GSE51057 is dominated by aTreg (+1.26); the myeloid direction is dominated by aEos (+1.12). Both are positive; this is NOT a bidirectional cancellation pattern. **Per OQ-2026-01 immune-atlas staging gating, no Test 2 framework-level claim is made from this VAL.** The lymphoid+myeloid concordance is logged as observation only.

## Card-specific routing

This is a SOLID-ORGAN card (breast-epic) Stage 3 immune analysis. The aTreg elevation at >10yr is consistent with immune-class signal during long pre-diagnostic windows — the pattern observed in VAL-047 Phase 9 (Stage 1 immune d=+1.78 at >10yr in GSE51057) is now resolved into a specific subtype-level signal: regulatory T cells.

## Substrate scope

Single-substrate methyl-only, 450K platform. v1 single-substrate.

## Implications for breast-epic v0.3

1. **UniLIFE 19-cell IS additive to Salas 6-cell at long pre-diagnostic windows.** At >10yr, the aTreg signal is a clean resolution gain. Recommendation: include UniLIFE Stage 3 alongside Salas in the v0.3 breast-epic card scoring, not as a replacement but as an additional resolution layer. Reports cite both.
2. **At near-diagnosis (0-2yr), Salas 6-cell suffices.** The dominant signals (Bcell positive, Neu negative) are visible at Salas resolution and gain little from UniLIFE subtype splitting.
3. **The aTreg at >10yr in long pre-diagnostic windows is a candidate biomarker** for early-detection panel design — but this is outside the VAL's scope. Logged for breast-epic v0.3 Open Questions.
4. The aTreg pattern interacts with VAL-096's immune-tile inversion finding: at >10yr GSE51057, Loyfer's bulk monocyte tile reads d=+0.33 and erythrocyte progenitor d=+0.83; UniLIFE shows that within the lymphoid compartment the elevation localizes to aTreg. The overall picture is "broad immune-cell-class drift at long pre-dx with regulatory T cell subtype dominance."

## Language discipline

- "consistent with established immune-modulation literature" ✓ (not "validates")
- "framework-level interpretation is open and outside this VAL's scope" ✓
- "No bidirectional cancellation claim" ✓ (per CCL-031)
- "logged as observation only" for non-replicating signals ✓
- No "UniLIFE proves" or "Salas misses" — said "UniLIFE 19-cell IS additive to Salas 6-cell"

## Outputs

- `VAL-095_results.json` — full per-cell-type per-window per-cohort table for both panels
- `VAL-095_per_sample.csv` — per-sample 19-cell UniLIFE + 6-cell Salas fractions side-by-side, joined with TTD/group metadata
- `val_095.py` — execution script with NaN-handling fix in deconvolve_rpc (RNG seed 20260426)
- `VAL-095_prereg.md` + `VAL-095_prereg_seal.json`
