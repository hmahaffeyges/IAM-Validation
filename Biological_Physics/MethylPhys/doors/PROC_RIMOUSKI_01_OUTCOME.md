# PROC-RIMOUSKI-01 — outcome (scored 2026-10-02 exactly as pre-registered; scorer `score_rimouski.py`, same statistic and code path as PROC-CHARR-01)

**Data.** Atlantic salmon fin, Rimouski hatchery programme: 32 F0 adults (16 stocked, 16 wild origin) and 32 F1 offspring (4 parental crosses × 8). All 64
fish aligned and extracted (one fish re-fetched once, as allowed). Reading: single-molecule copy error ε_corr (sequencing error subtracted) and E = ln((1−ε)/ε) in kT.

| prediction | result | verdict |
|---|---|---|
| P0 instrument: two run-halves agree, ICC ≥ 0.80 | ICC 0.996 | **met** |
| P1 fish not library: \|ρ\| < 0.30 with conversion failure, duplicates, masked fraction | conversion failure ρ = +0.38; duplicates −0.20; masked −0.15 | **not met** (conversion failure) |
| P2 scale: median F0 in 3.6–4.3 kT | 3.47 kT | **not met** (just below the window) |
| P3 origin (F0) | stocked − wild ε = +0.0021 (95 % CI 0.0010–0.0031), p = 0.0003 | significant, **not interpreted** (P1 failed) |
| P4 parents (F1) | father +0.0001 (p 0.71); mother +0.0003 (p 0.27) | none; descriptive only |

**Descriptive.** F0 median ε 0.0303 (3.47 kT); F1 0.0278 (3.56 kT). Group medians: F0 stocked 0.0317, wild 0.0299; F1 crosses 0.0272–0.0281.

**Reading.** The instrument repeats almost perfectly. In these libraries the reading still tracks the bisulfite conversion failure rate, so the stocked-vs-wild
difference cannot yet be attributed to the fish: as in brook charr, a library-quality term has to be removed before fish-level differences are read. The F1
offspring read lower error than F0 adults. The conversion-failure correction is the next development step (fit ε against conversion failure within run
batch and read the residual); it is development, so the pre-registered verdicts above stand as scored.
