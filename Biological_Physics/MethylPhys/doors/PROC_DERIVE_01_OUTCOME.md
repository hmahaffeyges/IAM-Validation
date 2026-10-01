# PROC-DERIVE-01 — outcome (2026-09-30). Pre-registration: PROC_DERIVE_01_PREREG.md (sha 7521c01ccf37e041).

Sites methylated in HUES64 bulk (β ≥ 0.8, ≥ 10 reads): 5,953,404. Restoration r(t) = β_nascent / β_bulk, pooled.

| series | 0 h | 1 h | 2 h | 4 h | 8 h | 16 h | shortfall f (16 h) | E (kT) |
|---|---|---|---|---|---|---|---|---|
| replicate 3 (pre-registered series) | 0.644 | 0.729 | 0.821 | 0.958 | 0.944 | 0.980 | **0.0203** | 3.88 |
| replicate 1 | – | 0.801 | – | 0.916 | – | 0.9964 | **0.0036** | 5.63 |
| replicate 2 | – | – | – | – | – | 0.9961 | **0.0039** | 5.55 |

Measured steady copy error in HUES64 (ENCODE single molecules, instrument-corrected): 0.0186.

| prediction | replicate 3 | replicates 1–2 | verdict |
|---|---|---|---|
| P1 f within 0.0093–0.037 | 0.0203 PASS | 0.0036–0.0039 FAIL | **not reproducible** |
| P2 E within 1.9–4.4 kT | 3.88 PASS | 5.55–5.63 FAIL | **not reproducible** |

**Reading.** The pre-registered series matches the measured error within 9 %, but the two other 16-h replicates of the same experiment give a
shortfall 5× smaller. Replicates also disagree at 1 h and 4 h in opposite directions (rep1 higher at 1 h, lower at 4 h), so the spread is between
replicates, not a consistent processing offset. The copy error cannot yet be derived from these kinetics: the per-cycle shortfall is not measured
precisely enough in this dataset (label dilution by 16 h and ~2–3 M reads per time point). A derivation needs a kinetics dataset with tighter
replicates, or hairpin (both-strand) data that read maintenance directly.
