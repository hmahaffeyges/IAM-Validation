# PROC-PREDX-SEQUENCE-01 — pre-registration (written 2026-09-30, before any EPIC-Italy array is read by the current chain)

**Question.** The pre-commissioning VALs showed a pre-diagnostic sequence in blood: immune cells move first; then several secretory cells elevate
together while the immune cells settle; near diagnosis, breast luminal alone stays elevated. Does the current chain (atlas v2 composition, then
each present cell's A = H(mean β over its v2 identity loci) / H_min(class), tiers v1.5) show it again, on people the VALs did not use?

**Data.** EPIC-Italy, GSE51032 (450K, whole blood, raw IDATs through our Stage 1), time to diagnosis per case.
**Held out (primary):** every GSE51032 array that is NOT in GSE51057 (the 329 arrays the VALs were discovered on). Expected ~78 breast cases,
~72 colorectal cases, ~247 cancer-free controls. All other arrays: descriptive only.
Lead-time bins (years before diagnosis): > 8, 5–8, 2–5, < 2. Each specimen is read alone; the table counts individual readings.

**Readings per specimen.** For every cell present at ≥ 1 %: fraction, A, tier. "Moved" = A outside Normal (< 0.95 or ≥ 1.05).
Immune = the v2 immune cells; secretory = the v2 secretory cells; breast = breast luminal epithelium (A, and fraction ≥ 1 %).

**Predictions (held-out breast cases vs held-out controls; one-sided Fisher exact test, p < 0.05 each):**
- P1 immune first: lead > 8 y — share of cases with ≥ 1 immune cell moved exceeds controls.
- P2 secretory together: lead 2–8 y — share of cases with ≥ 2 secretory cells at A ≥ 1.05 exceeds controls; AND the immune share in this bin is
  lower than in the > 8 y bin.
- P3 breast at the end: lead < 2 y — share of cases with breast luminal present (≥ 1 %) or at A ≥ 1.05 exceeds controls; AND the share with ≥ 2
  other secretory cells at A ≥ 1.05 is not higher than controls.
- P4 cross-cancer: held-out colorectal cases, lead > 8 y — the P1 test.
Verdict: the sequence reproduces if P1, P2 and P3 all pass.

**Stated limits now.**
- Minor-cell A in one whole-blood specimen is below per-specimen resolution (PROC-SCORE-03); a pass means readings move consistently across
  people, not that one person's minor-cell reading is reliable.
- Arrays give the PROGRAM reading only; the error reading (IAM's law on single molecules) cannot be made on this data.
- H_min(class) here is the current chain's floor, not the physics floor. Controls give no information about how the floor was set.
- No subject identifiers: no serial pairs in this set.
