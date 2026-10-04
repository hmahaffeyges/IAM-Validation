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
