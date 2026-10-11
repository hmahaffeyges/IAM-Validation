# DEV-COMPOSITION-TRUTH-04 — an independent whole-blood truth set for Met-A bar 7 (candidate, 2026-10-11; nothing read, no pass value written)

**Why.** Whole-blood Met-A leaves the commissioned scope until bar 7 (neutrophil fraction recovered within RMSE 0.02, DEV-NILC-01) is met on
whole bloods of known composition from a laboratory independent of the reference (STATUS R4a). The public search of 2026-10-11 (1,730 human EPIC
series with IDATs, 949 on 450K, Europe PMC, ArrayExpress; metadata only) found one such set.

**Candidate: GSE167202** (University of Colorado; GEO contributors share none with the Salas reference group or the Loyfer/Moss laboratory,
CANON/geo_contributors.csv). Whole blood on GPL29753, a custom EPIC on the v1 (B4) backbone with added custom probes; raw IDATs in the series
tar, already in S3 (`downloads/A_blood_immune/GSE167202/`); no record of any reading (CANON/data_spent.csv). Truth: the clinical blood count with
automated differential in each sample's GEO record (`data/DEV_COMPOSITION_TRUTH_04/gse167202_cbc.csv`, summary `cbc_summary.txt`, from
`cbc_summary.py`): 409 samples give neutrophil, lymphocyte and monocyte %; 387 give the full six-part differential; 374 of those sum to 100 ± 2.
Segmented neutrophil % has median 73.0 (10th-90th percentile 51.9-86.7), a wider range than healthy blood.

**What has to be decided before any test is written (author):**
1. Is GPL29753 inside the EPIC v1 scope? Checkable before reading: whether every Met-A identity site and composition site is on it, from the
   manifest alone.
2. Is a cohort of emergency-department patients (COVID-positive, COVID-negative, other infection) acceptable for bar 7? Bar 7 tests recovery of the
   neutrophil fraction, not health; the wide range of neutrophil % tests it across more of the scale.
3. How to treat error in the truth itself: an automated differential is not exact, and the paper does not state that the count and the DNA came
   from the same draw. Error in the truth adds to the measured RMSE, so a pass would be conservative and a failure would need reading.

**Next.** Answers to 1-3, then synthetic tests on the chain (composition recovery on constructed mixtures with truth error added), then the test
written and sealed, then the arrays read once. Search candidates and reasons for rejection: the 2026-10-11 search table (to be committed with the test).
