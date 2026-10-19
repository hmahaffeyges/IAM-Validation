# Chain v4 showcase (2026-10-18)

Public samples picked from their GEO labels alone, by rules written before any file was opened
([development/TEST_PLAN_TRACKS_2026-10-17.md](../../../../../development/TEST_PLAN_TRACKS_2026-10-17.md), Parts 6 and 7; picks in
[selected.json](selected.json)), each read once through the one command on a clean install of the chain. A showcase of what the chain does:
nothing here is scored. Every report, bundle and log is in [run1_2/](run1_2/). Scripts: [showcase_v4.sh](showcase_v4.sh) and
[showcase_v4_run2.sh](showcase_v4_run2.sh).

## One person, read by every array instrument (scenarios 1 and 7)

The buffy coat of participant AK0005 (GSE350403, GSM10113783, EPIC v1), tared by the buffy coats of the next three participants of the same
study. Buffy coat is read on the whole-blood path and labeled as its own specimen, so every reading carries the preliminary line.

| instrument | reading |
|---|---|
| Met-A | A 1.053; tared A_rel 1.039: Normal (0.95-1.05) |
| C-Score | C 1.178; tared C_rel 1.067: not above the band (1.152) |
| IAM-Deconvolver | neutrophils 96.7 %, cd4 t cells 1.2 %, cd8 t cells 0.7 %, b cells 0.6 %, nk cells 0.5 %, monocytes 0.2 % |

## What the chain declines (scenario 6), each with its reason

| sample | reason printed |
|---|---|
| AK0005 PBMC, EPIC v1 (GSM10113784) | specimen 'peripheral blood mononuclear cells' refused at intake: PBMC (peripheral blood mononuclear cells) has no reference in the chain: the neutrophil reading needs whole blood or isolated neutrophils |
| AK0005 saliva, EPIC v1 (GSM10113785) | specimen 'saliva' refused at intake: 'saliva' is not a blood specimen the chain has a reference for (tissue, placenta and other specimens each need their own reference) |
| AK0005 buffy coat, EPIC v2 (GSM10113831) | EPIC v2 array (IDAT header): the chain reads EPIC v1 arrays only - no frozen neutrophil floor for EPIC v2, and Stage 1 (methylprep 1.7.1) does not calibrate EPI |
| AK0005 PBMC and saliva, EPIC v2 (GSM10113832-33) | refused at intake, as on EPIC v1 |
| a 450K test array (GSM2333901, the chain's test data) | quarantined at Stage 0: too few EPIC sites (hm450 coverage); nothing is scored |

## Known mixtures outside the Deconvolver's scope (scenario 3)

Three DNA mixtures of nine cell types (GSE281305, mixtures 1-3), including two cultured cell lines, fibroblasts and endothelial cells. The
IAM-Deconvolver is commissioned on blood-cell shares; each report says so at the top. With no reference for most of the cells in the tube, it
spreads them over the six blood cells it knows, and its raw total before normalizing is 1.13, 1.11 and
1.13 (1.00 when the reference explains the sample). The C-Score reads its sky against healthy neutrophils, the only
reference it has, and shows the departures across the whole sphere.

| mixture | shares reported |
|---|---|
| mix 1 | cd4 t cells 45.1 %, b cells 22.7 %, nk cells 15.6 %, neutrophils 13.9 %, monocytes 2.7 % |
| mix 2 | cd4 t cells 44.7 %, b cells 22.0 %, nk cells 15.8 %, neutrophils 14.9 %, monocytes 2.6 % |
| mix 3 | cd4 t cells 46.3 %, b cells 20.3 %, nk cells 15.3 %, neutrophils 15.1 %, monocytes 3.0 % |

## Healthy colon tissue read by IAM-Atlas v2.1 (scenario 4)

Three healthy colon biopsies (GSE151732, adenoma history "No", the first three by accession; EPIC v1), read by the atlas's commissioned colon
tissue use through its own scorer, unchanged: chain Stage 1, then the 6,000 sealed composition-blind sites of the colon tissue test,
where every colon cell type the atlas holds agrees, so a healthy tissue sample should match the atlas whatever its mix of cells. The reading
is the share of those sites within 0.05 of the atlas's colon-epithelium value; the atlas was commissioned on colon tissue at 0.90.
Tissue has no route through the one command yet, so this runs through the scorer directly
([scenario4_colon.py](scenario4_colon.py), [showcase_v4_s4.sh](showcase_v4_s4.sh); record [s4/scenario4_colon.json](s4/scenario4_colon.json)).

| sample | sites measured | share within 0.05 | median difference from the atlas |
|---|---|---|---|
| GSM4589141 | 5,997 of 6,000 | 0.911 | 0.011 |
| GSM4589144 | 5,997 of 6,000 | 0.941 | 0.010 |
| GSM4589145 | 5,998 of 6,000 | 0.943 | 0.010 |
| the three together | | 0.942 | |

Each healthy colon sample reads above the commissioned bar, and the three together read 0.942.

## Scenarios 2 and 5

Both are shown by the example reports in [REPORT_EXAMPLES_V4/final](../REPORT_EXAMPLES_V4/final): scenario 2 (Met-A on isolated neutrophils) by a
held-out healthy array, and scenario 5 (IAM-A) by a sample of its grand test, healthy CD8 T cells, read 0.9083 Normal. Scenario 5 has no
separate showcase sample because no fresh public whole-genome set of healthy resting CD8 T cells was found on GEO (searched 2026-10-18).
