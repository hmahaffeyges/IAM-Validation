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
