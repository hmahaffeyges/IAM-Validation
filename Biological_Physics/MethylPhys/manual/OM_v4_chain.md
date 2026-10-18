# Operations Manual: chain v4 (operator chapter)

The procedure (install, the one-command check, the commands, what is declined, outputs, frozen inputs) is the SOP, printed first in this
manual. This chapter is what an operator needs around it: before a run, reading the report, and what to do when the chain stops.

## Before a run

1. Run the release check once per installation and after every update: `python3 release_check_v4.py`. Every check must pass.
2. Arrays: the Illumina manifest must be in `$HOME/.methylprep_manifest_files/` (methylprep fetches it on first use). Read the first array of a batch alone, so the download is not raced.
3. Same-run references (arrays): read three or more healthy arrays from the same run first, each on its own. Put their Met-A reading (`A`) and C-score (`C`) from each report's bundle in a CSV with columns `id,A,C`, and pass it with `--ref-table` when reading the samples of that run. The tare divides by the references' median, so the laboratory's own habits cancel out. Nothing is fitted.
4. Sequencing: give the cell type with `--seq-cell`, and for a BAM the exact reference FASTA it was aligned to (`--ref-fasta`, indexed).
5. Declare the specimen exactly (`--specimen "whole blood"`, `"isolated neutrophils"`, `"buffy coat"` or `"constructed DNA mixture"`). Anything else is refused at intake with its reason.

## Reading the report

Each run writes one HTML page (a Researcher view, opened first, and a Plain view), the same report as two PDFs, the bundle (JSON) and one row
in the evidence ledger. The report reads top to bottom:

- **At a glance.** One line per instrument: its reading, its state, and whether the reading is inside that instrument's commissioned scope.
- **Every gauge has three marks:** 0, the physical floor; 1, healthy; and the saturation end, where every identity site is fully scrambled (Met-A 3.03; IAM-A 0.5 / ε_pred(L), the sample's own). The marks are derived, not fitted.
- **States.** Met-A, tared: Normal 0.95 to 1.05. The C-score: above its healthy band when above 1.152 (one-sided). IAM-A: Normal 0.7787 to 1.0551. Where these bands mean disease is for cancer researchers to find; the report prints readings and states, not diagnoses.
- **Commissioned scope.** A reading outside what an instrument was commissioned on is still shown, with the line *This score is preliminary until [instrument] is officially commissioned on [specimen or cell]*. Met-A is commissioned on isolated neutrophils, the C-score on whole blood, the IAM-Deconvolver on neutrophil, monocyte, CD4 T and CD8 T shares, IAM-A on healthy CD8 T cells.
- **Cell shares.** Each cell's share with its 95 % interval, read against IAM-Atlas v2.1. Before mixing anything, the tool checks the sample can be read; it refuses rather than report a wrong share.
- **The tare.** When `--ref-table` is given, the report prints the references' median and spread, the tared reading, and the smallest change this sample could have shown.
- **The noise gate.** On a noisy array Met-A's state is withheld and no gauge is drawn; the report says so.
- **The sky.** The sample laid over healthy neutrophils from IAM-Atlas v2.1 on one HEALPix sphere, every site in its genome place. Grey is where the sample matches healthy; color marks only departures beyond twice the sample's own spread. Scattered color is what healthy looks like; streaks along the genome are what the C-score calls clustered.
- **IAM-A** reads a sequencing sample against itself: the error with which its own cells hold their marks, against the healthy line of its declared cell type. If the sample's shape is outside that cell type's healthy shape, or too few molecules qualify, IAM-A declines and says why.
- **Technical record.** The chain's commit, every input's checksum, every frozen file read, and every caution (for example, Stage 0 did not run because the sample entered as a table of values).

## When the chain stops

| exit | printed | meaning | what to do |
|---|---|---|---|
| 0 | the report | read, or declined with a reason printed in the report (specimen, platform, sample shape) | read the report |
| 1 | an argument message | a command-line input is missing or malformed (for example a reference table without its columns) | correct the command |
| 2 | `QUARANTINE` and its code | Stage 0 failed the file (corrupt or truncated IDAT, missing channel, array type mismatch, low detection) | nothing is scored; check the file, rerun from the scanner output |
| 3 | `ENVIRONMENT_MISSING_MANIFEST` | the Illumina manifest could not be fetched | put the manifest in place and rerun; the array was not judged |

An EPIC v2 or 450K array is refused with one sentence (`PLATFORM_REFUSED`); the chain holds no Met-A reference for those platforms.

## Reproducing the commissioning

The commissioning tests and the headline results rebuild from public data with one command each: `reproduce/README.md` at the top of the
repository.
