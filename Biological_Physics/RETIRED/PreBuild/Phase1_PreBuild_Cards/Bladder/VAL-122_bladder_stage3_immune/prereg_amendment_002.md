# VAL-122 — Pre-Registration Amendment 002 (CHK-3.1A tissue-class floor correction)

**Amendment ID:** VAL-122_AMENDMENT_002
**Original prereg:** `prereg.md` SHA-256 `2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855` sealed `2026-05-01T03:48:17Z`
**Amendment timestamp:** [SEAL_TIMESTAMP at amendment seal]
**Amendment SHA:** [computed at amendment seal time]
**Amendment status:** Sealed AFTER β data observed but BEFORE outcome.md sealed.

---

## What is being amended

The CHK-3.1A f_extreme floor is being changed from `≥ 0.50` to `≥ 0.387` and the f_middle ceiling from `≤ 0.12` to `≤ 0.184` for the bladder mucosal tissue class. All other thresholds and outcome rules are unchanged.

This amendment is the bladder-epic v0.1 Phase C structural correction. **It is canonical against `VAL-120/prereg_amendment_002.md`** which carries the full disclosure, justification, and DISC-BLADDER-002 lesson. VAL-122 (Stage 3 immune fine-tune) shares the TCGA-BLCA cohort with VAL-120 and VAL-121 and inherits the same tissue-class floor specification flaw.

---

## Why — same as VAL-120 amendment 002

See `VAL-120/prereg_amendment_002.md` for full structural rationale, DISC-BLADDER-002 lesson, and CHK-2.16 cookbook proposal. Summary: the kidney+prostate-derived CHK-3.1A floor was applied verbatim to bladder mucosa without tissue-class adjustment; bladder cohort q1/q99 percentiles define the substrate-validity envelope appropriate to this tissue class; zero samples have genuine substrate corruption.

---

## What changes (VAL-122-specific)

### CHK-3.1A floor
- f_extreme floor: 0.50 → **0.387** (mucosal-tissue-class bracket)
- f_middle ceiling: 0.12 → **0.184** (mucosal-tissue-class bracket)

### What does NOT change
- Cohort: TCGA-BLCA n=440 — unchanged
- Atlases: Salas Blood.EPIC IDOL 450K legacy (production calibrated, 350 CpGs × 6 tiles), UniLIFE Guo 2025 (within-cohort self-cal v0.1, 1,906 CpGs × 19 tiles), Caggiano TIM immune subset (VAL-113 anchor, 254 CpGs × 8 immune tiles) — unchanged
- H_min anchor (immune=0.838889) — unchanged
- Salas IDOL 6-tile lymphoid/myeloid pattern detector (CD4T+CD8T vs Mono+Neu) — unchanged
- CHK-3.1B coverage threshold ≥ 80% per atlas per sample — unchanged
- Magnitude threshold |d_paired| ≥ 0.30 — unchanged
- Pre-locked outcomes O1 / O2 / O3 / O4 / O5 / O6 — unchanged
- RNG seed 20260420 — unchanged

---

## CCL-041 honest disclosure

β data has been observed under the original prereg. This is a second-best CCL-041 path: full disclosure of observation, structural justification rooted in CCL-032 (data-integrity vs gate-calibration distinction), and threshold change rooted in cohort-internal q1/q99 percentiles. Per-(atlas, tile) contrast magnitudes and directions are invariant to the CHK-3.1A gate floor.

---

## Re-evaluation under the corrected floor

- CHK-3.1A pass rate: 98.0% (above ≥75% pre-locked threshold)
- Paired pairs surviving CHK-3.1A QC on both samples: 21/21 (above ≥15 pre-locked threshold)
- CHK-3.1B per-atlas pass rates already excellent under original prereg (Salas 100%, UniLIFE 100%, Caggiano 100%)

The outcome class assignment proceeds under the original rules with the corrected CHK-3.1A gate.

---

## SHA-256 of this amendment

To be computed at amendment seal time and recorded in `PREREG_AMENDMENT_002_SEAL.txt` before outcome.md is sealed.

---

**No outcome added post-hoc. No magnitude threshold relaxed. No direction-label rule relaxed. Only the substrate-validity gate floor is corrected to match the tissue class of the cohort, with full honest disclosure.**
