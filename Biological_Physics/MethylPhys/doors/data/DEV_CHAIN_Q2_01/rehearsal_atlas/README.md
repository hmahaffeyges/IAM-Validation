# Rehearsal of the PHY-084 scorer on the atlas laboratory's spent counts (2026-11-13, 1:30 PM Pacific; after the result)

Command (from DEV_CHAIN_Q2_01): python3 referee_score.py rehearsal_atlas/counts rehearsal_atlas/samples.csv forward_out rehearsal_atlas/out
Three libraries per type, so no bar is computable (the intercept's own uncertainty is undefined); the intercepts are what matter:
own-slope intercepts (out.csv), the same method that read the referee laboratory's B cells low (referee_out.csv).
Read along IAM-A's slope instead (celltype_vs_thermal.py), both laboratories' blood types sit near the thermal level.
