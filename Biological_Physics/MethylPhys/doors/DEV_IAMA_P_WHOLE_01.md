# DEV-IAMA-P-WHOLE-01 — the neutrophil IAM-A position P measured on whole files (development, 2026-10-08)

**DEVELOPMENT - not commissioned.** Calibration of a development stage; no locked value and no result changes.

**Why.** P (the healthy neutrophil's distance from ε₀, the frozen denominator of IAM-A) was frozen in v1 on the first 60,000,000 bytes of
each Loyfer granulocyte file (one stretch of chr1). DEV-IAMA-REAL-01 then read the same three healthy files whole: the copy error over the
whole genome (0.0378-0.0392) is above the copy error over the first 60 MB (0.0355-0.0374), so the same healthy cells read 1.034-1.063 against
v1 P. A reading always covers the whole genome; P must be measured over the same sites the reading covers.

**Rule (unchanged from v1).** Per donor, P_d = H(mean ε of the other two donors) ÷ H(ε₀); P = mean of the three. ε₀ = 0.032 (canon).
Counts: isolated errors and opportunities from `doors/data/DEV_IAMA_REAL_01/*_whole_bundle.json` (hg19 files, loyfer_pat_v1).

| | v1 (first 60 MB) | v2 (whole files) |
|---|---|---|
| ε per file | 0.035562 / 0.037390 / 0.035511 | 0.038078 / 0.039228 / 0.037833 |
| P per donor | 1.105 / 1.084 / 1.106 (rule recomputed: 1.0981; frozen 1.099) | 1.1527 / 1.1396 / 1.1554 |
| P | 1.099 | **1.1492** (range 1.1396-1.1554) |
| the three files read whole | 1.0394 / 1.0632 / 1.0344 | 0.9940 / 1.0167 / 0.9892 |

\calibrated P = 1.1492 on the whole files of three donors from one laboratory. The three files are the files P was built on, so their
readings are in-sample; the test of P is another laboratory's healthy neutrophils (GSE128731, Box Run 2 session 2).

**Consequences.** `iama_positions_v2.json` is the frozen input; v1 is kept under `superseded` in `FROZEN_INPUTS_v3.json`. Stage Q refuses a
reading of part of a file (`--pat-max-bytes`) unless `--dev-allow-partial` is given for development. The book's Part VI does not print P.
