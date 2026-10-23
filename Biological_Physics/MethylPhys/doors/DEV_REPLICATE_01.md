# DEV-REPLICATE-01: the earlier work, replayed through the chain (tuning)

**The instrument's job:** the chain, one command, reads every array the earlier instruments read, and we see whether the signals they found
are found again. Tuning, not commissioning: only data the earlier work already opened; every data set no instrument has opened stays unread.

## R1: the earlier CPG engine's 11 test arrays

The arrays its test-data manifest lists, fetched from GEO by accession (the IDAT fingerprints are checked against the author's local copies):
rheumatoid arthritis cases and controls (GSE42861, whole blood, 450K), three healthy bloods (GSE87571, 450K), a high-grade colon adenoma
(GSE288652, EPIC) and colorectal carcinoma at stages 1 and 4 (GSE166212, EPIC). Blood: specimen whole blood, the controls and healthy bloods
as references (`r1_blood_samples.csv`); tissue: specimen colon tissue (`r1_tissue_samples.csv`). Scorer `data/DEV_REPLICATE_01/r1_score.py`.

**What the earlier engine reported, written before the chain reads them:** epithelial cells read near zero in healthy blood and appear in
the colon tissues, rising from stage 1 (about 13 % of the sample) to stage 4 (about 25 %), with its disorder reading rising with stage;
one healthy blood (GSM2333905) carried a stem-cell-class elevation; RA cases departed from controls in their immune cells.

**What the chain should show if it agrees:** epithelial (colon) cells found only in the tissues and more at stage 4 than stage 1; the
whole-sample Met-A and the territories higher in the carcinomas than the adenoma, and higher at stage 4; the bloods near one another.
