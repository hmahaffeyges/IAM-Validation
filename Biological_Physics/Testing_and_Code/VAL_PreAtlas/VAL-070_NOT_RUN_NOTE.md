# VAL-070 NOT RUN — Lymphoid/Myeloid Operational Split (Pancreatic-Epic Preview)

**Status:** Not run, blocked on data dependency.

## Why this should run

Heath asked at the end of the pancreatic-epic VAL-066/067/068/069 sprint whether the immune-class A-score could be split into lymphoid-arm and myeloid-arm scores to capture the directional pattern more precisely. The framework rationale (CCL-027, OQ-2026-01) anticipates that bidirectional Stage 1 patterns may resolve into uniform-direction patterns within lineage-restricted CpG subsets (lymphocytes drop, neutrophils rise — the immune-cell-throttle pattern).

For pancreatic-epic specifically, Clark 2007 and the broader PDAC immunology literature predicts:
- Lymphoid arm (CD4 T effector, CD8 T effector, B cells): suppressed (β shifts consistent with transcriptional silencing of effector immune programs)
- Myeloid arm (MDSCs, M2 macrophages, neutrophils): expanded (β shifts consistent with myeloid suppressor expansion)
- Net pooled signal: cancellation, exactly what we observed across VAL-066/067/068.

## What's blocking

The Salas 2018 IDOL-Extended (IDOL-Ext) panel CpG list with per-CpG cell-lineage assignments (CD4T, CD8T, B, NK, monocyte, neutrophil) is required to do this honestly. The panel is published and openly available via the EpiDISH R package, but it is not currently staged in the project files for this Cookbook session. Constructing a heuristic lineage split based on Xu-538 CpG gene symbols would violate CCL-029 (NO-FABRICATION) — the gene-symbol-to-lineage map is not 1:1 and the resulting partition would be unreliable.

## What can be done with what we have

The 324-CpG VAL-069 directional panel built from GSE49149 already encodes per-CpG direction. We can hand the **172 positive-direction CpGs** and **152 negative-direction CpGs** to the immune-atlas card author as the input data for VAL-070 once the IDOL-Ext panel is staged. The lineage assignment will then come from joining the directional panel against IDOL-Ext.

The expected pattern, IF the immune-cell-throttle hypothesis is right, is:
- The 152 negative-direction CpGs should disproportionately map to lymphocyte-lineage IDOL-Ext markers
- The 172 positive-direction CpGs should disproportionately map to myeloid-lineage IDOL-Ext markers
- A 50/50 expected split would be the null hypothesis

## What goes into pancreatic-epic v0.1 in lieu of running VAL-070

The card's Stage 1 design block documents:
1. Pooled-entropy nulled across 3 cohorts (VAL-066/067/068)
2. Directional Xu-538 (324 CpGs, GSE49149-trained) recovers signal on TCGA-PAAD holdout (VAL-069 H2 PASS, d=+1.51, p=6.4e-05)
3. Directional fallback partial-fails on GSE74071 (VAL-069 H3 FAIL)
4. **Open question for v0.2+**: lymphoid-arm vs myeloid-arm split, blocked on IDOL-Ext panel staging. Logged as VAL-070-pending, contributing to OQ-2026-01.

## Action needed before VAL-070 can run

Stage the Salas IDOL-Ext panel CpG list with cell-type annotations. Source: EpiDISH R package, function `epidish()` reference matrix, or directly from Salas LA et al. Genome Biology 2018 (PMID 29945600) supplementary tables. Once staged, VAL-070 takes ~30 minutes to run on the existing GSE49149 + TCGA-PAAD + GSE74071 β data we already have downloaded.
