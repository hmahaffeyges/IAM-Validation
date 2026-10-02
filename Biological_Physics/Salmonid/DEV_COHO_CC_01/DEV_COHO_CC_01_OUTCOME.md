# DEV-COHO-CC-01 — outcome (2026-10-02)

Scored as written in `DEV_COHO_CC_01_NOTE.md` (committed 2bf27f6 before scoring; scorer 76ee851). 39 of 39 fish, none failed.
Development reading; nothing here is a commissioning result.

| check | bar | result | |
|---|---|---|---|
| D1 half-split ICC of ε_cc (common sites) | ≥ 0.9 | **0.826** | not met |
| D2 \|ρ\| ε_cc vs conversion failure | < 0.3 | **0.384** (ρ −0.384) | not met |
| D2 \|ρ\| ε_cc vs depth (qualifying molecules) | < 0.3 | 0.224 | met |

Both checks are needed, so **hatchery vs wild and female vs male are not described**, as the note requires.

## What the run shows
- **The per-read conversion filter did not change the reading.** It kept 92.8 % of qualifying molecules and moved ε by +0.00008 on
  average; ρ with conversion failure is −0.384 before and after. So the library term is not carried by incompletely converted molecules.
  It is a library-level property that travels with conversion efficiency (higher conversion failure, lower ε). Within sequencing lanes the
  sign is mostly negative too (ρ from −1.0 to +0.6 in 8 lanes of 3–7 fish), and neither ε nor conversion failure differs by lane (p 0.36, 0.33).
- **The spread between fish is small.** ε on 43,554 common sites runs 0.0337–0.0373 (median 0.0356). Between-fish SD is 0.00089 against
  a within-fish half-split noise of 0.00039, which is why the ICC is 0.83 and not 0.99 as in the WGBS sets (Rimouski 0.996, Methow 0.998).
- **The holding energy reads 3.25–3.36 kT (median 3.30)**, the same as Methow steelhead red cells (3.31 kT) and human cells (3.29–3.51 kT),
  not the value a fixed gap in joules would give at river temperature. Descriptive only.

## Next development step
The library term has to be removed at the library level, not the read level: a per-library spike-in or an internal control whose true
copy error is known (for example, a fully methylated control region read in every library), so ε can be read against what that library
does to a known pattern. RRBS cannot be deduplicated, so a set with WGBS or EM-seq and unique molecular identifiers is the better next
fish set. Per-fish tables: `coho_cc_fish.csv`; summary `coho_cc_summary.json`; site tables and BAMs in S3 `downloads/coho_work/leluyer_cc/`.
