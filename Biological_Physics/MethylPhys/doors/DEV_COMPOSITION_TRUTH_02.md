# DEV-COMPOSITION-TRUTH-02 - an adult, other-laboratory mixture truth set for composition, NILC and atlas_e (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Search (author decision H), done 2026-10-04 before reading any array.** GEO E-utilities over the EPIC platforms (GPL21145, GPL23976) for
mixture / reconstructed / flow cytometry / FACS / cell count / known proportions / artificial mixture / DNA mixture / titration, and over 450K
(GPL13534) for reconstructed / artificial mixtures of leukocytes. Every EPIC mixture or flow-counted series found comes from the laboratory that made
the purified references the chain uses (GSE110554, GSE112618, GSE110530, GSE167998, GSE182379, GSE181034, GSE180970). No adult EPIC mixture or
flow-counted set from another laboratory was found on GEO.
The one adult reconstructed-mixture set from another laboratory is **GSE77797** (450K: reconstructed mixtures of purified leukocytes with known
proportions, and whole bloods).

**Check (on GSE77797, 450K).** Stage 1 on its IDATs. Truth = the depositors' proportions in the series record. The three methods held to the
DEV-NILC-01 bars: Stage 2 NNLS8, NILC-e, atlas_e (`STAGE_A_PROPOSAL.patch`), each at the sites it finds on 450K.
- Before any truth is read: the number of the 963 composition markers present on the 450K array. Stage 2 requires >= 867. If fewer are present,
  NNLS8 is refused as specified and also run at the markers present, labelled as below its own requirement.
- Bars: neutrophil RMSE <= 0.02 and every other group in the truth RMSE <= 0.03 (groups the truth gives; T = CD4T + CD8T if the truth is not split).
- The platform differs from the templates' (EPIC templates on a 450K array). That is stated with every number.

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box job 5bd710b8 (Stage 1 on the 18 GSE77797 IDAT pairs, 450K) and scoring on the laptop (`data/DEV_COMPOSITION_TRUTH_02/gse77797_scores.csv`).

- \measured Composition markers present on the 450K array: **307 of 963** (Stage 2 requires 867). Stage 2 NNLS8 is refused as specified; its numbers at the
  307 markers are shown below, labelled.
- \observed The truth gives granulocytes, not neutrophils: the neutrophil bar (0.02) is applied to the sum NEU + EOS + BASO.

RMSE on the 12 reconstructed mixtures (bars: granulocytes <= 0.02, every other group <= 0.03):

| method | GRAN | MONO | B | NK | CD4T | CD8T |
|---|---|---|---|---|---|---|
| atlas_e | 0.041 | 0.012 | 0.011 | 0.019 | 0.017 | 0.005 |
| NILC-e | 0.061 | 0.020 | 0.019 | 0.012 | 0.036 | 0.017 |
| NNLS8 (307 markers, below its requirement) | 0.033 | 0.011 | 0.011 | 0.013 | 0.022 | 0.015 |

- \measured atlas_e meets every bar except granulocytes (0.041; bias -0.036). NILC-e: granulocytes 0.061 and CD4T 0.036 outside. NNLS8 at 307 markers:
  granulocytes 0.033 outside.
- \measured The 6 whole bloods (truth = the depositors' flow counts, which sum to 0.89-0.97): atlas_e 0.023 | 0.012 | 0.014 | 0.033 | 0.020 | 0.032; NNLS8 0.017 | 0.025 | 0.013 | 0.038 | 0.060 | 0.053 (same column order).
- \observed Every method reads granulocytes low on this other-laboratory set (bias -0.027 to -0.052), as on the cord-blood set (DEV-NILC-01). The rest of the
  composition is recovered within 0.03 by atlas_e on mixtures of another laboratory's purified cells.
- \openprob An EPIC mixture or flow-counted set from another laboratory still does not exist on GEO; a wet-lab mixture is the clean test.

**Wiring.** atlas_e and NILC-e stay behind `--dev-atlas-e` / `--dev-nilc`. Stage 2 stays NNLS8.
