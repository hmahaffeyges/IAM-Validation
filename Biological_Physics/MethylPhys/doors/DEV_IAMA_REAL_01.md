# DEV-IAMA-REAL-01 - Stage Q (IAM-A) end to end on real single-molecule blood data (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Data.** Loyfer 2023 blood granulocyte .pat files (GSM5652313, GSM5652314, GSM5652315; GEO GSE186458), whole files from GEO. These are the
files the neutrophil position P was measured on (first 60,000,000 bytes of each); the rest of each file is genome the position never saw.
Search for another laboratory's healthy neutrophil or blood single-molecule data in a .pat-compatible form (2026-10-04): ENCODE has no neutrophil WGBS;
SRA neutrophil bisulfite / EM-seq runs found are surgical-patient series (SRP403075, SRP356148). No other-laboratory healthy set was found; an
independent check needs reads aligned and converted to .pat by the loyfer_pat_v1 pipeline, or P measured for another pipeline.

**Checks.**
1. `run_sample.py --pat <file>` runs end to end on each whole file: report and bundle written. Bar: 3 of 3.
2. The first 60,000,000 bytes through `--pat-max-bytes 60000000` reproduce `chain_tests/iama_floor_granulocytes.csv` (isolated errors and opportunities
   equal). Bar: 3 of 3 exact.
3. Whole-file IAM-A per file in Normal, and its two halves within 0.005 of each other. Bar: 3 of 3.
4. The IAM-A C-score (DEV-IAMA-CSCORE-01) for each file and half: recorded.

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box jobs 53dbbb3d (hg19 build: the files P was measured on, GEO names without a genome tag, 281-339 MB) and c491e31b (first run: it fetched the
`.hg38.pat.gz` files, which are another build - recorded as an observation below). Stage Q's .pat reader was changed to stream (the earlier
whole-file read joined every bgzip member into one buffer and does not finish on a whole file); on constructed multi-member and plain files it
returns the same lines as before (checked before the run). Records: `data/DEV_IAMA_REAL_01/` (bundles; `hg38_build/`).

| file | first 60 MB: IAM-A (eps) | whole file: IAM-A (eps) | opportunities (whole) | IAM-A C-score, whole (blocks of 1,000 sites) | hg38 build, whole |
|---|---|---|---|---|---|
| GSM5652313 | 0.9867 (0.035562) | 1.0394 (0.038078), halves 1.0392 / 1.0397 | 85,929,614 | 604 (halves 302 / 303); head 11.3 | 1.0772 |
| GSM5652314 | 1.0251 (0.037390) | 1.0632 (0.039228), halves 1.0633 / 1.0631 | 119,593,574 | 646 (halves 323 / 324); head 14.0 | 1.0979 |
| GSM5652315 | 0.9856 (0.035511) | 1.0344 (0.037833), halves 1.0344 / 1.0343 | 120,241,827 | 1047 (halves 525 / 523); head 16.7 | 1.0720 |

1. \measured `run_sample.py --pat` ran end to end on the three whole files: report and bundle written, **3 of 3**.
2. \measured The first 60,000,000 bytes reproduce `chain_tests/iama_floor_granulocytes.csv` exactly: isolated errors 585,119 / 737,728 / 725,421 and opportunities
   16,453,474 / 19,730,776 / 20,428,312, **3 of 3** (the job's summary table printed errors from the rounded eps; the bundles carry the exact counts).
3. \measured Whole files: IAM-A 1.039, 1.063, 1.034: **2 of 3** in Normal (bar 3 of 3); halves agree within 0.0005 on each file (bar 0.005).
4. \measured IAM-A C-score: 604-1,047 on the whole files, 11-17 on the 60 MB heads (independent errors give 1 within 0.013-0.03).
- \observed The copy error over the whole genome (0.0378-0.0392) is higher than over the first 60 MB (0.0355-0.0374), the stretch P was measured on. P as frozen
  holds for that stretch; over the whole genome the same healthy cells read 3-6 % higher. P depends on which sites the reading covers.
- \observed The same reads on the hg38 build read 1.072-1.098 (eps 0.0397-0.0409): the build is part of the pipeline, as the refusal rule says.
- \observed The C-score is far above 1 on healthy cells: copy errors are not independent along the genome; they cluster by region. C grows with the
  opportunities per block (whole file > head), as overdispersion does.
- \openprob Freeze P on whole files (or on a fixed site set) and say which; give C a block size set by opportunities rather than sites.
- \openprob No other-laboratory healthy neutrophil single-molecule data in a .pat-compatible form were found (search above).
