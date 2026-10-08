# DEV-Q0-HEALTHY-01 — Stage Q0 on the healthy files behind P, and the limits (development, 2026-10-08)

**DEVELOPMENT - not commissioned.** Written after reading the three healthy Loyfer files and BEFORE any GSE128731 (test) file is read.

**Run.** Box job 8c7a87b8 (2026-10-08): `stage_q0_intake.intake` on the three whole hg19 Loyfer granulocyte files (the files P v2 was measured
on) and on the hg38 copy of GSM5652313 as a real wrong-build control. Records: `doors/data/DEV_Q0_HEALTHY_01/*.q0.json`.

| file | decision | lines | molecules | share with ≥ 6 CpG calls | lines outside hg19 ranges | autosomes |
|---|---|---|---|---|---|---|
| GSM5652313 (hg19) | proceed | 90,450,330 | 319,081,751 | 0.0647 | 0 | 22 |
| GSM5652314 (hg19) | proceed | 108,723,324 | 444,565,323 | 0.0665 | 0 | 22 |
| GSM5652315 (hg19) | proceed | 104,655,317 | 408,493,559 | 0.0704 | 0 | 22 |
| GSM5652313 (hg38 copy) | **stop: GENOME_BUILD_MISMATCH** | 104,999,003 | 391,140,087 | 0.0745 | 42,866,672 | 22 |

\measured The build rule separates the two builds of the same reads: 0 of 303.8 million hg19 lines out of range; 42.9 million of 105.0
million hg38 lines out of range.

**Limits, decided here before any test file:**
1. Q0.3 conversion: ≥ 98 % C-to-T (ENCODE WGBS data standard). The .pat format carries no non-CpG calls, so these files cannot set it.
2. Q0.5 read length: **recorded, no stop limit.** Three files from one laboratory give 0.0647-0.0704; a limit at that edge would refuse a
   laboratory for its read length, not for a property IAM-A depends on. What Stage Q needs from molecule length, enough qualifying
   opportunities, Stage Q enforces itself (≥ 100,000). Whether ε depends on read length is tested on GSE128731, whose runs differ in
   sequencer and read length.
3. Q0.6 duplicates: recorded, no stop limit (no published standard; Loyfer's own files carry no duplicate record).
