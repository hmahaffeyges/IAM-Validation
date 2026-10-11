# DEV-TISSUE-COLON-01 — outcome (scored 2026-10-11, once, under the seal; development)

**Sealed:** `data/DEV_TISSUE_COLON_01/sealed_colon_tissue.json` (the score step checked the seal committed unchanged and the scorer's sha256).
**Scored:** `data/DEV_TISSUE_COLON_01/scored_colon_tissue.json`.

**Rule:** FINGERPRINT if >= 11 of 88 patients' colorectal tumours read Met-A(tumour) / Met-A(own adjacent normal, same slide) above
1.0921 at the composition-blind sites; else NO FINGERPRINT.

**Result: FINGERPRINT.** 28 of 88 pairs above the bar (binomial p 1.4e-15 at 0.05 per pair). Median R 1.0658; 80 of
88 tumours read above their own normal.

**Reviewed (descriptive, `data/DEV_TISSUE_COLON_01/why_colon_tissue.py`, output `why_colon_tissue_output.txt`):** pairs above the bar on all three
plates and in both stages (IIA, IIB); 5,923-6,000 of the 6,000 sealed sites read in every pair. A group-level fingerprint, not a per-patient test.
One laboratory and one series built the reading, so it extends no commissioned scope. Replication on data no version has analysed:
DEV-TISSUE-COLON-02 (GSE199057, another laboratory, EPIC v1).

**Prior reading:** GSE131013 was analysed earlier by the retired class-gauge method (PROC-TISSUE-01, processed betas, a different quantity); this
measurement had not been made on these arrays.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements.
