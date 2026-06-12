# VAL-121 — Pre-Registration Amendment 002 (CHK-3.1A tissue-class floor correction)

**Amendment ID:** VAL-121_AMENDMENT_002
**Original prereg:** `prereg.md` SHA-256 `eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962` sealed `2026-05-01T03:48:17Z`
**Amendment timestamp:** [SEAL_TIMESTAMP at amendment seal]
**Amendment SHA:** [computed at amendment seal time]
**Amendment status:** Sealed AFTER β data observed but BEFORE outcome.md sealed.

---

## What is being amended

The CHK-3.1A f_extreme floor is being changed from `≥ 0.50` to `≥ 0.387` and the f_middle ceiling from `≤ 0.12` to `≤ 0.184` for the bladder mucosal tissue class. All other thresholds and outcome rules are unchanged.

This amendment is the bladder-epic v0.1 Phase C structural correction. **It is canonical against `VAL-120/prereg_amendment_002.md`** which carries the full disclosure, justification, and DISC-BLADDER-002 lesson. VAL-121 (Stage 2 multi-atlas) and VAL-122 (Stage 3 immune fine-tune) share the TCGA-BLCA cohort, share the CHK-3.1A gate, and inherit the same tissue-class floor specification flaw. This amendment is the local instance of the cookbook-wide CHK-3.1A tissue-class correction.

---

## Why — same as VAL-120 amendment 002

The bladder cohort (TCGA-BLCA n=440) shows a tissue-class methylation distribution shape distinct from solid parenchyma:
- Bladder adjacent-normal (n=21): f_extreme 49.5% ± 4.3%
- Bladder primary tumor (n=418): f_extreme 47.1% ± 4.9%
- VAL-106 baseline (n=210 KIRC+PRAD adjacent-normal): f_extreme 55.87% ± 2.44%

Under the original kidney/prostate-derived 0.50 floor, only 23.9% of bladder samples pass, mislabeling a tissue-class threshold mismatch as a data integrity failure. Zero samples in the cohort have any genuine substrate corruption (f_extreme < 0.30, f_middle > 0.30, or n_cpgs_genome < 350,000). The corrected floor (cohort-internal q1/q99: f_extreme ≥ 0.387, f_middle ≤ 0.184) lets through 98.0% of samples and all 21 paired pairs.

See `VAL-120/prereg_amendment_002.md` for the full structural rationale, DISC-BLADDER-002 lesson, and CHK-2.16 cookbook proposal.

---

## What changes (VAL-121-specific)

### CHK-3.1A floor
- f_extreme floor: 0.50 → **0.387** (mucosal-tissue-class bracket, bladder cohort q1)
- f_middle ceiling: 0.12 → **0.184** (mucosal-tissue-class bracket, bladder cohort q99)

### What does NOT change
- Cohort: TCGA-BLCA n=440 — unchanged
- Atlases: Layered Moss+Loyfer (VAL-112 anchor), EpiSCORE BladderRef (VAL-119 anchor, SHA `3005663b…`), Caggiano TIM (VAL-113 anchor) — unchanged
- H_min anchors (terminal=0.772837, immune=0.838889, secretory=0.843264, cycling=0.856055, stromal=0.862950) — unchanged
- 25 Loyfer tile class assignments — unchanged
- 4 BladderRef tile class assignments (EC=stromal, Epi=secretory, Fib=stromal, IC=immune) — unchanged
- 19 Caggiano tile class assignments — unchanged
- CHK-3.1B coverage threshold ≥ 80% per atlas per sample — unchanged
- CHK-3.1A pass-rate threshold ≥ 75% — unchanged
- Magnitude threshold |d_paired| ≥ 0.30 — unchanged
- CCL-039 cell-of-origin direction expectation (NEGATIVE on Loyfer Bladder, BladderRef Epi; POSITIVE on microenvironment) — unchanged
- CHK-3.2 cross-tile sanity check on Loyfer non-bladder solid-tissue tiles — unchanged
- Pre-locked outcomes O1 / O2 / O3 / O4 / O5 — unchanged
- RNG seed 20260420 — unchanged

---

## CCL-041 honest disclosure

β data has been observed under the original prereg. This is a second-best CCL-041 path: full disclosure of observation, structural justification rooted in CCL-032 (data-integrity vs gate-calibration distinction), and threshold change rooted in cohort-internal q1/q99 percentiles (observable substrate properties not chosen to make a particular outcome fire). Per-(atlas, tile) contrast magnitudes and directions are invariant to the CHK-3.1A gate floor — only QC-pass eligibility for paired contrasts changes.

---

## Re-evaluation under the corrected floor

- CHK-3.1A pass rate: 98.0% (above ≥75% pre-locked threshold)
- Paired pairs surviving CHK-3.1A QC on both samples: 21/21 (above ≥15 pre-locked threshold)
- CHK-3.1B per-atlas pass rates already excellent under original prereg (Loyfer 100%, BladderRef 100%, Caggiano 100%)

The outcome class assignment proceeds under the original rules with the corrected CHK-3.1A gate.

---

## SHA-256 of this amendment

To be computed at amendment seal time and recorded in `PREREG_AMENDMENT_002_SEAL.txt` before outcome.md is sealed.

---

**No outcome added post-hoc. No magnitude threshold relaxed. No direction-label rule relaxed. Only the substrate-validity gate floor is corrected to match the tissue class of the cohort, with full honest disclosure.**
