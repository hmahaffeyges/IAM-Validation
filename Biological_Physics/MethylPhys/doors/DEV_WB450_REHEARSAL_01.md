# DEV-WB450-REHEARSAL-01: whole-sample Met-A on healthy 450K whole blood, tared on its own run (development, spent data only)

**The instrument's job:** a healthy 450K whole blood, tared on the other healthy bloods of its run, reads Normal on the gauge.

**Why a rehearsal:** plan 60.6 would spend GSE85210 (81 healthy never-smokers, unread) as the 450K whole-blood commissioning test. Before any
of it is read, the same reading is rehearsed on bloods already spent, so an unread set is not spent on a result whose cause is already in hand.

**Sources:** GSE67393 (public, GEO: the test half of its healthy 450K whole bloods, spent in DEV-CSCORE-LIMIT-01); IAM-Atlas v2.1 (this repository).

**Read:** the 57 bloods by the one command with `--full-chain --cell-panel atlas --dev-metaa-allsites` (series run 42321149732); each blood's
whole-sample Met-A against the median of its run's other bloods (`data/DEV_WB450_REHEARSAL_01/wb450_rehearsal.py`).

**Result** (`data/DEV_WB450_REHEARSAL_01/wb450_rehearsal_summary.txt`): tared on the run, median 0.9988, standard deviation 0.0499; a share of
0.596 within 0.95 to 1.05, short of the 80 % a commissioning bar would ask.

**Decision:** GSE85210 stays unread until the cause of this spread is found on spent data (plan 60.6).
