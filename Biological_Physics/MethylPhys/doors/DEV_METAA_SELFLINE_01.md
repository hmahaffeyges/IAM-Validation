# DEV-METAA-SELFLINE-01: can Met-A read against its own loose share? (COM-METAA-039, development)

**In plain words.** Met-A, our commissioned array instrument, has to be compared with healthy cells from the same laboratory, because
every lab's machines read a little differently: untared, healthy neutrophils from other labs read 1.065 to 1.19 instead of 1.00. IAM-A
learned on 2026-10-15 that a healthy cell carries its own reference: its loose share. This test asks whether the same is true for Met-A,
whether a lab's offset also shows up in each array's own loose share, so a single line can take the offset out with no same-lab reference.
If it works, Met-A can be pointed at new labs, and new cell types, without building a reference in each one first.

**Rule, data and bars:** `data/DEV_METAA_SELFLINE_01/selfline.py` (docstring), written before any loose share of these arrays was computed.
Six series of healthy purified neutrophils (EPIC v1) whose Met-A readings are already committed; leave one series out; showcase bars.

**Synthetic test:** data/DEV_METAA_SELFLINE_01/selfline_synthetic.txt (MET: an offset the loose share carries is removed; one it does not carry is refused).

What it would take to succeed: each lab's Met-A offset predicted by its own arrays' loose share, from the other labs alone; finish line, not a promise.

**Amendment before scoring (2026-10-15).** Computing the loose shares showed that the six healthy GSE167998 neutrophil arrays are the six
GSE110554 reference arrays deposited again under new sample numbers: Met-A, loose share, bin count and probe count agree array by array
(GSM5121408 = GSM2998143, GSM5121424 = GSM2998021, GSM5121422 = GSM2998023, GSM5121419 = GSM2998057, GSM5121412 = GSM2998116,
GSM5121421 = GSM2998030; `data/DEV_METAA_SELFLINE_01/arrays_L_all.csv`). Kept, they would count one laboratory twice in every fit. They are
dropped (`arrays_L.csv`), leaving five independent series; the rule, the code and the bars are unchanged (the majority bar reads 4 of 5).
Finding for other records: wherever GSE110554 and GSE167998 healthy neutrophils are counted as two sets, they are one.
**Second amendment (2026-10-15).** The first scoring run returned no numbers: four arrays (GSM3462471, GSM7885055, GSM7885062,
GSM7885064) have no Met-A reading because the chain withheld them, and the line fit cannot take an empty value. Those four are left out, as
every Met-A read-out leaves out a withheld array; nothing else changed. The empty run printed "NOT MET" only because no bar could be
computed; it is not a result.

## Outcome (2026-10-15): NOT MET

Scored once on 68 arrays from five independent laboratories (`data/DEV_METAA_SELFLINE_01/selfline_series.csv`, `selfline_verdict.json`).

| series | arrays | raw offset | after the line | slope fitted on the other labs |
|---|---|---|---|---|
| GSE110554 (reference lab) | 6 | 0.001 | -0.130 | 5.49 |
| GSE118144 | 13 | 0.065 | -0.021 | 3.54 |
| GSE122244 | 4 | 0.075 | 0.053 | 6.18 |
| GSE247193 | 21 | 0.192 | 0.137 | 0.16 |
| GSE247195 | 24 | 0.069 | -0.100 | 8.17 |

All three bars unmet. The laboratory with the largest offset (GSE247193, 0.19) has a loose share (0.051) next to the reference lab's
(0.052), and the fitted slope swings from 0.16 to 8.2 with the laboratory left out: a laboratory's Met-A offset is not carried by its
arrays' own loose share. **What it means:** Met-A keeps its same-lab tare. The array offset is the two-channel chemistry of the array
(DEV-SELFTARE-01: background in each channel shifts a measured level), which moves Met-A's identity sites but barely moves a count of
open-sea stretches below 0.70, so the loose share cannot see it. Met-A's route away from a same-lab reference remains the array's own fixed
sites (DEV-SELFTARE-01/02), not the loose share. The self-reference stays IAM-A's advantage: on single molecules there is no channel offset.
