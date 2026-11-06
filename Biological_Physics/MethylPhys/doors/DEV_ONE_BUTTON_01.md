# DEV-ONE-BUTTON-01: one folder, one button (design; plan 61.6)

**The instrument's job:** a researcher puts a sample's files in a folder and presses one button; the chain decides what the files are,
which ruler they read against and which readings apply, and says every choice it made. Nothing is guessed silently: every automatic
choice is printed with the evidence for it, and a choice the chain cannot make is a refusal with its reason.

## Today (read from `run_sample.py`)

| Choice | Today |
|---|---|
| What the files are | the researcher names them: `--grn`/`--red` (IDAT pair), `--betas`, `--pat`, `--bam` (needs `--ref-fasta`), `--site-table`, `--geo-pat GSM` |
| Array type | taken from the IDAT header when not given (Stage 0.1) |
| Specimen | `--specimen`, **default "whole blood"**: a run with no specimen is read as whole blood, not refused |
| Ruler | arrays: IAM-Atlas v2.1; sequencing: the IAM Sequencing Atlas only with `--dev-seq-ref auto` |
| Readings | the commissioned ones, plus every development reading with `--full-chain`; every cell with `--cell-panel atlas` |

## The design

1. **Find the files.** In the folder: a `_Grn.idat(.gz)` and `_Red.idat(.gz)` of one name is an array; a `.pat(.gz)` is molecules; a
   `.bam` is aligned reads; a `.beta` or a call-and-coverage bigWig pair or a per-site count table is sequencing levels. Two kinds in one
   folder, or none, is a refusal naming what was found.
2. **The ruler follows the platform.** Arrays read against IAM-Atlas v2.1; any sequencing input against the IAM Sequencing Atlas, without a
   setting (`--dev-seq-ref auto` becomes the default for sequencing).
3. **The specimen becomes optional.** Given, it is checked as today. Not given, the chain reads the sample's cell mix with every cell of the
   active atlas and names it from the mix, printed as the chain's own call:
   - one cell at 0.90 or more: a sorted or purified cell of that type;
   - the six blood cells summing to 0.90 or more: whole blood;
   - otherwise: a mixture named by its leading cells, read as tissue, flagged.
   The default of "whole blood" is removed: an unnamed specimen is never silently read as blood.
4. **Every reading that applies, every cell.** The one button is `--full-chain --cell-panel atlas`: the commissioned readings and every
   development reading, each with its flag, and a reading that cannot run listed with its reason.
5. **One record of every choice.** The report's first table lists each automatic choice (file kind, array type, ruler, specimen call and the
   shares behind it) beside what the researcher stated, if anything.

## Tests, written before any code

| Test | Bar |
|---|---|
| Synthetic: the same healthy inputs given with their specimen and without it | every reading identical; the specimen call matches the stated specimen on every input |
| Constructed mixtures with known shares (sorted, whole blood, tissue in blood) | the call is the right one of the three on each |
| A folder with two kinds of file, or none | refused, naming what was found |
| The readiness map (plan 45.1): one public file of each kind | every one read end to end by the button alone |

**When:** the code is written after the B and T and Johns Hopkins readings finish, so those readings stay on one chain version. Laptop work;
the readiness run is one GitHub job.
