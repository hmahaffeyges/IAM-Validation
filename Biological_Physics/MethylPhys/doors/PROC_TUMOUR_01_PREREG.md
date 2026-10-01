# PROC-TUMOUR-01 — pre-registration (written 2026-10-01, before any read of these datasets is aligned)

**Question.** Does the gate-error reading rise in a primary tumour against the same patient's adjacent normal tissue — no culture, no cell line,
no reference population? (PROC-ENCODE-01 showed it for cell lines, where culture itself raises error.)

**Data.**
- Primary: early-onset colorectal cancer, GSE284325 / PRJNA1198593 — 7 patients with tumour and adjacent normal (patients 1–7), WGBS paired-end.
- Secondary (descriptive): oral squamous cell carcinoma, GSE212634 / PRJNA876296 — 4 tumour/adjacent-normal pairs, WGBS and oxWGBS
  (oxidative bisulfite separates 5mC from 5hmC).

**Processing (fixed now).** First 20 M read-1 sequences per sample streamed from ENA; Trim Galore; Bismark (bowtie2) to GRCh38, library
directionality taken from a 1 M-read pilot (directional unless < 70 % of alignments land on the original strands); unique alignments; first and last
3 aligned bases ignored.

**Statistic** — identical to PROC-SALMON-01: qualifying molecule ≥ 6 CpG calls, ≥ 80 % methylated; isolated error = unmethylated interior CpG
between two methylated neighbours; ε = errors / opportunities; ε_corr = ε − s (A/T-reference mismatch rate); conversion failure reported;
per-patient genotype mask (site dropped if > 30 % of that patient's qualifying molecules covering it, ≥ 5, carry an error there).

**Predictions.**
- P1: ε_corr(tumour) > ε_corr(adjacent normal) in ≥ 6 of 7 early-onset CRC pairs.
- P2: median tumour/normal ratio of ε_corr ≥ 1.10.
- P3 (instrument): conversion failure differs by < 0.005 between tumour and normal in each pair (otherwise that pair is reported as instrument-limited).
Descriptive: oral SCC pairs on WGBS and on oxWGBS; whether the tumour excess survives on oxWGBS (5mC only).

**Stated limits now.** Adjacent normal can carry field effects (makes the test harder, not easier); tumour purity unknown; 7 pairs; one lab per set.
