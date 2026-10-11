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

**Author decisions (2026-10-11), before any array is read:**
1. **Platform:** GPL29753 counts as EPIC v1 for bar 7 only if every site the commissioned chain reads is on it as a standard EPIC probe.
   Checked from the manifest alone (`data/DEV_COMPOSITION_TRUTH_04/platform_check.py`, output `platform_check_output.txt`): the 6,000 Met-A floor
   sites, 963 composition markers, 6,000 composition neutrophil sites and 48,528 noise sites are all present, none from the custom content:
   PASSED. Still to check: that chain Stage 1 reads a GPL29753 IDAT (its address count differs from the 1,051,815-address EPIC v1 manifest);
   if it does not, Stage 1 needs a GPL29753 manifest, a chain change recorded before use.
2. **Cohort:** accepted. Bar 7 tests recovery of the neutrophil fraction, not health; the record of any result states that the truth set is
   emergency-department patients (COVID-positive, COVID-negative, other infection) with no healthy group.
3. **Truth error:** bar 7 unchanged (RMSE <= 0.02). Error in the clinical count adds to the measured RMSE: a pass is conservative; a failure is
   recorded as not met, with no reinterpretation after reading.

**Next.** The Stage 1 read check, then synthetic tests on the chain (composition recovery on constructed mixtures with truth error added), then the test
written and sealed, then the arrays read once. Search candidates and reasons for rejection: the 2026-10-11 search table (to be committed with the test).

**Stage 1 read check, 2026-10-11** (`data/DEV_COMPOSITION_TRUTH_04/stage1_read_check.py`, output `stage1_read_check_output.txt`). The chain reads an
array by bead address through the standard EPIC v1 manifest, so the deciding test is address identity, not probe names. The platform check
above compared names only, and is kept as a record, not as the test. Every one of the 865,918 probes this chip shares with the EPIC v1 manifest
has the same bead address in the submitters' custom manifest. Every chain site's beads are on the chip. Chain Stage 1 calibrates one array of the
series (GSM5098953, which has no neutrophil count and is excluded from the bar-7 scoring set; only probe names and addresses were used, no value
read). On that array, 6 Met-A sites, 1 composition marker and 49 noise sites fall to the array's own quality check, as on any EPIC array.
**Passed.** Next: synthetic tests of bar 7 on this design, then the seal.

