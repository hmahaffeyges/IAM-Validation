# DEV-WRITER-02 — outcome (2026-10-10; development)

Scored by the rule sealed 16:16 PDT with the strand amendment (`DEV_WRITER_02.md`), by `data/DEV_WRITER_02/score_writer_02.py` (one I/O fix after
sealing: the no-context bucket labelled "NA" was being read as a missing value; no rule changed). Reading: `context_eps.py` on the 153 healthy
Loyfer window files of PROC-CHANNEL-01, counts in `context_counts.csv`, per-sample fits in `writer_02_rows.csv`, output `score_writer_02_output.txt`.
**Check first:** summed over contexts, every sample reproduces PROC-CHANNEL-01's copy error and de novo rate exactly (largest difference 1e-16).
Counts per strand-pooled context class: at least 357,000 opportunities and 14,800 errors.

| statistic (median of 153 samples) | predicted | measured |
|---|---|---|
| copy-error slope on log ½[1/(1+D)+1/(1+D_rc)] | 1 | −0.240 |
| control (de novo) slope | 0 | +0.437 |
| **difference** | **1 (bar 0.5–1.5)** | **−0.688 (IQR −0.761 to −0.587)** |
Every one of the 153 samples has a negative difference and a negative copy-error slope.

**NOT MET.** Across flanking contexts, the copy error healthy cells hold does not follow DNMT1's discrimination measured outside the cell: it
goes slightly the other way (contexts the writer discriminates best carry no less copy error). Model A' is rejected. The agreement of the genome
average with twice the writer's single-step error (DEV-WRITER-01) is therefore not evidence for a writer-set copy error; at the genome mean it is
a coincidence of size. k from the forced fit (2.60) has no meaning once the slope fails and is not reported further.
**Recorded, not tested:** across contexts the copy error and the de novo rate are strongly anti-correlated (Spearman −0.82; `data/DEV_WRITER_02/describe_contexts.py`): a context-wide factor
moves the two channels in opposite directions, which this design cannot separate further.
**Consequence for D1.** The holding energy is not the writer's single-step discrimination. D1 stays open: whatever sets 3.41 k_BT acts after or
beside the writer's choice. No constant changes.
