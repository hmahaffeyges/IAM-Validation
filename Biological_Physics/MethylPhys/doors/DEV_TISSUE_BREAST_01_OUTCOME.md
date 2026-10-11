# DEV-TISSUE-BREAST-01 — outcome (scored 2026-10-11, once, under the seal; development)

**Sealed:** `data/DEV_TISSUE_BREAST_01/sealed_breast_tissue.json` (re-seal after the guard fix; the score step checked the seal committed
unchanged and the scorer's sha256). **Scored:** `data/DEV_TISSUE_BREAST_01/scored_breast_tissue.json`.

**Rule:** FINGERPRINT if >= 6 of 53 patients' tumours read Met-A(tumour) / Met-A(own normal tissue, same 450K plate) above 1.0943 at the
composition-blind sites; else NO FINGERPRINT.

**Result: FINGERPRINT.** 23 of 53 pairs above the bar (binomial p 1.7e-16 at 0.05 per pair). Median R 1.065; 41 of 53 tumours read above their
own normal. 5,929-6,000 of the 6,000 sealed sites read in every pair.

**Reviewed (descriptive, `data/DEV_TISSUE_BREAST_01/why_breast_tissue.py`, output `why_breast_tissue_output.txt`):** pairs above the bar on
every plate (1-9 per plate); 12 tumours read below their own normal. A group-level fingerprint, not a per-patient test: 30 of 53 tumours stayed
below the per-pair bar. TCGA was used to build the reading, so this extends no commissioned scope; commissioning a breast-tissue Met-A would need
bars sealed on another laboratory's matched tumour and normal tissue (STATUS section 0, R4).

**Milestone:** listed in `Biological_Physics/README.md`, Advancements.
