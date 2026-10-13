# DEV-IAMA-COMMISSION-01 — outcome (scored once under the seal, 2026-10-12; commissioning HELD under the stopping rule)

**Scored:** `data/DEV_IAMA_COMMISSION_01/scored_iama_commission_01.json`, from `epiqc_libraries_scored.csv` (Stage A 7 + Stage B 71 = the 78 sealed
libraries, matched to the seal's list exactly), with the seal and scorer unchanged. 25 libraries were excluded by the sealed intake rules
(conversion below 0.98 or a Stage Q0 refusal): MethylSeq 14, TruSeq 11.

| bar | verdict |
|---|---|
| C1 conversion (no correction at the 0.98 limit) | MET |
| C2 repeatability | MET in EMSeq LAB01, EMSeq LAB02, SPLAT, TrueMethylBS; TruSeq not assessed (no evaluable pair) |
| C3 across kits | MET in EMSeq, SPLAT, TrueMethylBS |
| C3 across laboratories (EMSeq, the one kit made in two labs) | **NOT MET**: 4 of 7 lines within 0.03; HG006 and HG007 differ by 0.109 and 0.162 |
| C5 a planted 1 % loss is seen | MET (53 libraries; smallest move 0.045) |
| C4 healthy reads Normal | not assessed (cell lines, not neutrophils; stopping rule) |

By the scorer's scope rule, SPLAT and TrueMethylBS meet every bar they are held to. **The note's stopping rule says commissioning stops when any bar
fails until the cause is found; C3 across laboratories failed, so nothing is declared commissioned.**

**Review (descriptive, after scoring; `why_iama_commission_01.py`, output `why_iama_commission_01_output.txt`).** The failure comes from three EMSeq
LAB02 libraries (HG006 and HG007 replicate 1, HG004 replicate 2) that read copy error about 15 % below every other EMSeq library (0.046-0.048
against 0.052-0.057). Conversion and duplicate fraction of the three are like the others, and input amount does not separate them (10 ng and 100 ng
libraries read normally in both labs). In the SRA records the three are the three most deeply sequenced EMSeq runs (283, 371 and 342 million
spots against 147-253 million for every other); the pipeline reads the first 12.2 million pairs of each file. This is a lead for the cause, not a
finding.

**Next (cause first, under the stopping rule):** read the three runs and two normal ones from other parts of their files (a development read,
not a re-score) to test whether the lower copy error belongs to the library or to the part of the file read. The scored result above stands as
scored.
**Not a milestone:** commissioning held under the stopping rule (development).
