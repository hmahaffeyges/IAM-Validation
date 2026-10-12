# DEV-TISSUE-LUNG-01 — outcome (scored 2026-10-12, once, under the seal; development)

**Sealed:** `data/DEV_TISSUE_LUNG_01/sealed_lung_01.json` (the score step checked the seal committed unchanged and the scorer's sha256).
**Scored:** `data/DEV_TISSUE_LUNG_01/scored_lung_tissue.json`.

**Rule:** FINGERPRINT if >= 6 of 28 patients' lung tumours read Met-A(tumour) / Met-A(own normal lung, same slide) above 1.1436 at the
composition-blind sites; else NO FINGERPRINT.

**Result: NO FINGERPRINT.** 4 of 28 tumours read above the bar; 6 were needed.

**Review (descriptive, after scoring; not a test;** `data/DEV_TISSUE_LUNG_01/why_lung_tissue.py`, output `why_lung_tissue_output.txt`).
20 of 28 tumours read above their own normal lung, but by little: median R 1.038. The pair bar came from two different patients' normal lungs on one
slide, and lung normals differ between people more than colon normals do (stand-in spread SD 0.0860, against 0.0502 for DEV-TISSUE-COLON-01), so
the bar is higher here. Tumours above the bar sit on three of five slides; asbestos exposure makes no visible difference. All 28 pairs read at
least 5,510 of the 6,000 sealed sites.

**What it means.** At this design's sensitivity, Met-A did not see lung cancer in this set as a group. It is not evidence that lung tumours have
no disorder: the reading leans up in most patients, and a bar set from between-person spread is wide for a within-person comparison. It does not
read liver or other cancers.

**Not a milestone:** no fingerprint by the sealed rule.
