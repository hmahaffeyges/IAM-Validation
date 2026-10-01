# PROC-ENCODE-01 — pre-registration (written 2026-09-30, before any ENCODE read was counted)

**Purpose.** Three things the Loyfer-only results cannot show: (1) whether each architecture's error rate reproduces in an independent lab and
pipeline; (2) whether the IAM reading moves in the right direction when a cell becomes cancer, on single-molecule data; (3) how much of the
measured error is the instrument's own (sequencing + bisulfite chemistry).

**Data.** ENCODE WGBS alignments (GRCh38, gemBS pipeline), public on encode-public S3, streamed once each; see encode_samples.csv. The same 400
random 100-kb autosomal windows are read in every sample. Per read, CpG calls from the reference and the read's conversion tag (XB).

**Measures (same definitions as PROC-CHANNEL-01).** copy error = isolated unmethylated CpGs inside methylated reads (≥ 6 CpGs, ≥ 80 % methylated);
de novo error = isolated methylated CpGs inside unmethylated reads (≤ 20 %). Instrument floor per sample: non-CpG conversion failure (unconverted C
outside CpG) and per-substitution sequencing error (mismatches at reference bases the conversion cannot touch, divided by 3). Holding energy
E = kT·ln((1−ε)/ε) on the instrument-corrected ε; A on the physics floor ε₀ = 1/(1+e^(φM)), φ and M fixed from PROC-CHANNEL-01 (0.1629 / 0.2088; 20.94).

**Predictions.**
- P1 (architecture reproduces): the primary immune cells (CD14 monocyte, B, T, NK) carry lower copy error than every stromal/muscle sample (IMR-90,
  myoblast, aorta, psoas muscle, heart left ventricle) — the ordering seen in Loyfer. Pass: all four immune below the lowest of those.
- P2 (direction at transformation): the cancer line reads higher copy error (A_meth on the physics floor) than its normal counterpart in ≥ 4 of 5
  pairs: HepG2 vs liver; A549 vs lung; K562 vs CD34+ myeloid progenitor; GM12878 vs B cell; OCI-LY7 vs B cell.
- P3 (instrument floor is small): instrument-attributable error < 20 % of the measured copy error in every non-neural, non-stem sample.
Descriptive only (no prediction): pluripotent lines (H1, H9, HUES64), where non-CpG methylation is real and inflates the conversion proxy.
