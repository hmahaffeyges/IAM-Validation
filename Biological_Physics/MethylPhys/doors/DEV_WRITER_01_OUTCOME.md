# DEV-WRITER-01 — outcome (2026-10-10; development)

Scored by the rule sealed at 15:42 PDT (`DEV_WRITER_01.md`), with `data/DEV_WRITER_01/score_writer_01.py` (output `score_writer_01_output.txt`).
Both full texts were supplied by the author and read before classification.

| measurement | enzyme | method | D | by the rule |
|---|---|---|---|---|
| Adam et al. 2023, Nucleic Acids Res 51:6622 | full-length murine DNMT1 | competitive rates, 256 flanks | 80 (abstract average; 87 in Fig. 2B; flanks 29 to >300) | qualifies |
| Yokochi & Robertson 2002, J Biol Chem 277:11735, Table II | full-length human DNMT1 | steady-state k_cat/K_M^CG, 19.9 vs 0.42 | 47.4 | qualifies |
| Bashtrykov et al. 2012, Chem Biol 19:572 | full-length murine Dnmt1 | rates at ~1 µM DNA, above K_M | ~10 | excluded: a k_cat ratio |
| same, 40-mer with both sites on one molecule | | no unmethylated-site methylation seen | ≥ 60 | listed: a lower bound |

**Median D = 63.7 (67.2 with 87): above the sealed band 15–60.** By the sealed reading: the writer alone discriminates better than the cells
hold. Its single-step limit is ε = 0.0155 (E_hold = ln D = 4.15 k_BT); healthy cells hold ε = 0.032 (3.41 k_BT), about twice the writer's error.
Model A (the held copy error is the writer's single-step error) is **not confirmed**: cells lose marks by more than the writer's mistakes.
**What it does establish, as far as two measurements reach:** the writer's measured discrimination is a floor under the copy error that cells sit
above, by about 0.74 k_BT. The extra loss must come from routes other than the writer's choice (for example sites the writer does not reach in
time, or marks removed after writing); which, and whether its size follows from the law, is open. No constant changes.
**Limits.** Two qualifying measurements, on short oligonucleotide substrates; DNMT1's discrimination spans 29 to >300 by flanking sequence.
