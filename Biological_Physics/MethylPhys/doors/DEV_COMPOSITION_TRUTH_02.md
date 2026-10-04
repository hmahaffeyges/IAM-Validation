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
