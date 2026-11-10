# COMMISSION-MONO450-01 — outcome: MET (all three bars)

Sealed in `COMMISSION_MONO450_01_PREREG.md` (commit d34c9064) before any array was read; read by the one command on Google
(`series_google.sh`, tag m450_sorted, 4 shards, one shard first as a canary); scored by `commission_mono450_01_score.py`
(`data/DEV_METAA_HEALTHY_01/commission_mono450_01_summary.txt`, rows `commission_mono450_01_rows.csv`).

**Not a milestone:** the milestones are recorded in the commissioning notes (`COMMISSIONING_NOTE_METAA_ATLAS_SORTED.md`, `COMMISSIONING_NOTE_CSCORE_ATLAS_SORTED.md`).

**In plain words.** Healthy sorted monocytes on 450K arrays, from two laboratories the atlas never held for this cell, were read against the
atlas alone: the IAM-Deconvolver found monocytes as the main cell in every one, Met-A read Normal in 13 of the 14 it could give a state, and
the C-Score sat within its limit in 18 of 19. **What it would take** to finish both instruments: the same kind of sealed test
passed by neutrophils on EPIC, by lymphocytes, and by cells inside whole blood.

| Bar | Result | Met |
|---|---|---|
| 1. top cell found is a monocyte, 90 % of arrays | 19 of 19 | yes |
| 2. of arrays with a Met-A state, 85 % Normal | 13 of 14 (one above Normal, 1.048) | yes |
| 3. of arrays with a C-Score, 85 % within its limit | 18 of 19 | yes |

**All 19 arrays read as Infinium 450K by the chain.** Five GSE110607 arrays found the monocyte at 65 to 72.5 percent of the sample, below
the 73 percent at which the atlas's monocyte donors were measured, so Met-A gave no state, as the sealed rule says; they are counted, and
their C-Scores are scored.

**Readings by laboratory (Normal arrays, Met-A median).** GSE110607 1.0011 (8 arrays), GSE131627
1.0025 (5 arrays).

**Prior reading:** GSE110607's granulocytes were read in COMMISSION-METAA-SORTED-01; none of these 19 monocyte arrays was read before (`CANON/data_spent.csv`).
