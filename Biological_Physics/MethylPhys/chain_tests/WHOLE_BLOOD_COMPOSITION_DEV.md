# Whole-blood neutrophil Met-A in chain v3: composition and tare (development, 2026-10-01)

**Problem found by running the chain end to end.** The atlas v2 solver (WGBS + array references) under-reads neutrophils in EPIC blood
by ~0.05 and splits T cells into subtypes with no EPIC purified profile. With its fractions the expectation is wrong (W2: 5/6 → after a
subtype map 1/6; the second figure was a scoring-script fault, the diagnostic reproduces the first).

**Fix (chain v3 Stage A).** Composition on the same platform and reference as the expectation: 8 groups from Salas purified EPIC cells,
963 markers (not neutrophil sites; within-group SD ≤ 0.05; margin ≥ 0.25), NNLS, sum 1. Frozen: blood_composition_EPIC_v1.json.

**Known mixtures with ≥ 50 % neutrophils (n = 6; profiles from the other study):**

| reading | healthy in Normal | 2 % neutrophil damage > 1.05 |
|---|---|---|
| untared A (EPIC NNLS) | 3/6 (0.943–0.968; offset −0.05, SD 0.009) | 0/6 |
| fraction re-fitted on neutrophil sites | 2/6 | 0/6 (the fit absorbs the damage) |
| **A_rel, tared against the other healthy blood specimens** | **6/6 (0.992–1.021, SD 0.010)** | **6/6 (1.052–1.090; shift +0.061)** |

**Rule adopted for the chain.** In whole blood the neutrophil Met-A carries a near-constant composition offset; the gauge state is printed
only after the tare against healthy whole bloods run the same way (same slide/batch, ≥ 3) — the same-slide tare already in the canon.
Untared whole-blood A is reported as a number with state "untared". Isolated neutrophils read against their own floor, no tare needed.
Limits: 6 DNA mixtures, simulated damage; real healthy whole blood and repeat pairs are next.
