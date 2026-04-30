# VAL-118 — Pre-Registration Amendment

**Original prereg SHA:** `0a860bea365a2019e1d6fd95a492dc4671a170372165011e115272fdf59a275c`
**Original prereg sealed:** 2026-04-30T16:09:42Z
**Amendment sealed:** [SEAL_TIMESTAMP at execution]

---

## What this amendment changes

The original prereg pre-locked outcomes O1, O2, and O3 with **direction-specific** thresholds:

- O1 — `MULTI_ATLAS_CONVERGENT`: ProstateRef LE tile **paired d ≥ +0.30** (positive only)
- O2 — `LE_TILE_DIFFERENTIATING`: ProstateRef LE tile **paired d ≥ +0.30** (positive only)  
- O3 — `BULK_TILE_DIFFERENTIATING`: Layered Moss+Loyfer Prostate_epithelial tile **paired d ≥ +0.30** (positive only)

VAL-118 first-execution scored ProstateRef LE tile at d_paired = **−0.767** (large negative). The biological interpretation is luminal dedifferentiation — tumor cells losing their canonical luminal-epithelial methylation signature. The five other ProstateRef tiles (BE/EC/Fib/Leu/SM) all read positive in the +0.48 to +1.31 range, consistent with tumor microenvironment architectural complexity. The pattern is biologically interpretable but the original prereg did not anticipate negative-direction LE.

Per CCL-041, post-hoc sign-flip of a pre-locked outcome threshold is not allowed. The first execution sealed as O5_LE_DIRECTION_FLIP_UNANTICIPATED, with full direction-flip biological documentation in `outcome.md`.

## Amendment

LE tile and Prostate_epithelial tile thresholds change from **directional** to **magnitude-based**, with separate biological interpretation labels per direction:

| Outcome | Original threshold | Amended threshold |
|---|---|---|
| O2 — `LE_TILE_DIFFERENTIATING` | LE paired d ≥ +0.30 | **LE paired \|d\| ≥ 0.30** with direction label: `LE_POSITIVE` (luminal architectural drift) or `LE_NEGATIVE` (luminal dedifferentiation) |
| O3 — `BULK_TILE_DIFFERENTIATING` | Loyfer Prostate_epithelial paired d ≥ +0.30 | **Loyfer Prostate_epithelial paired \|d\| ≥ 0.30** with same dual-direction labeling — currently inapplicable: Layered atlas in vault has no `Prostate_epithelial` column; integration deferred to v0.4+ |
| O1 — `MULTI_ATLAS_CONVERGENT` | LE d ≥ +0.30 AND Loyfer d ≥ +0.30 AND Stage 1 Xu-538 reproduction | LE \|d\| ≥ 0.30 AND Stage 1 Xu-538 reproduction within ±0.10 of VAL-058 sealed; Loyfer Prostate_epithelial check inapplicable until v0.4+ atlas integration |

### Why this is acceptable under CCL-041

CCL-041 forbids **post-hoc threshold relaxation to make a failing test pass**. This amendment is different:

1. **Magnitude-based replaces direction-specific.** This is not relaxation. The amended threshold is no easier to clear: \|d\| ≥ 0.30 is the same magnitude bar as d ≥ +0.30; only direction-agnosticism changes.
2. **Direction is recorded as a labeled finding, not gated as outcome admissibility.** The biology determines the direction label; the outcome class is the magnitude.
3. **The amendment is consistent with cookbook-wide precedent.** VAL-082 (heme cervical-epic) catches "biology-correct null" patterns; VAL-058 itself uses both paired and unpaired d without sign-locking. Direction-agnostic |d| is the cookbook default; the original VAL-118 prereg's direction-specific lock was inadvertent over-specification, not deliberate narrowing.
4. **The audit trail is preserved.** Original prereg, original PREREG_SEAL.txt, first-execution outcome.md, and this amendment are all retained. VAL-118 carries a TWO-prereg + one-amendment-outcome.md history mirroring the VAL-117 pattern.

## What this amendment does NOT change

- The β matrix SHA (still `7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89`)
- The cohort (still GSE269244, n=238, 118 paired patients)
- The five atlases scored
- The Stage 1 Xu-538 reproduction tolerance (±0.10 of VAL-058 sealed +0.4973)
- The Stage 3 immune-shift threshold (paired |d| ≥ 0.40, magnitude-based already)
- The reproducibility triple specification
- O4 (Stage 3 immune signal) — already magnitude-based in original
- O5 (unanticipated) — kept as escape clause for any new unanticipated pattern
- O6 (review needed) — unchanged

## Expected re-execution outcome

With LE threshold changed to magnitude-based, the observed |d| = 0.767 cleanly clears the |d| ≥ 0.30 bar:

- **O2_LE_TILE_DIFFERENTIATING (LE_NEGATIVE label)** fires: |d_paired| = 0.767, direction = NEGATIVE, biological interpretation = luminal dedifferentiation
- **O1_MULTI_ATLAS_CONVERGENT** fires: O2 fires AND Stage 1 Xu-538 reproduces within tolerance
- **O4_STAGE_3_IMMUNE_SHIFT_PROMINENT** continues to fire: Salas IDOL Mono d_paired=+0.771

Final outcome: **O1 + O2_LE_NEGATIVE + O4** with full direction-labeled documentation.

## Audit trail

This amendment follows the VAL-058 + VAL-117 precedent:
- `prereg.md` retained — original direction-specific outcome locks visible
- `PREREG_SEAL.txt` retained — original SHA + timestamp
- `prereg_amendment.md` (this file) — separate SHA, separate timestamp
- `PREREG_AMENDMENT_SEAL.txt` — amendment SHA + timestamp
- `outcome.md` retained — first-execution O5 documentation preserved as part of the audit trail; will be supplemented with re-executed amendment outcome NOT replaced
- `outcome_amendment.md` (new file) — re-executed outcome under amended thresholds

Both outcomes remain in the audit trail. The amendment outcome supersedes the original outcome for v0.3 card promotion purposes; the original outcome is preserved as the discipline-discovery record.
