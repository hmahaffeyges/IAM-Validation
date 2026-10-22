# The researcher package: MethylPhys chain v4

**Read first: [An Introduction to IAM and the Physics of Methylation](INTRODUCTION.md)**, how the instruments work, what each test showed, and what makes them different.

One command reads a DNA methylation sample through the commissioned IAM cell instruments and writes a report. This package is everything
needed to read the reports, run the chain on your own samples, and rebuild every headline result from public data. It is the
Biological_Physics folder of the repository at one commit, with the repository's license, README and citation: the cosmology, quantum and
semiconductor work is not needed to run or rebuild anything here.

**Book:** https://hmahaffeyges.github.io/IAM-Validation/ (Part VI, *Cellular Physics: Thermodynamics of the Methylome*).
**IAM-Atlas v2.1:** https://doi.org/10.5281/zenodo.23441999 (the atlas) and https://doi.org/10.5281/zenodo.23422688 (the fit's full outputs).
**Questions:** [Biological_Physics/MethylPhys/QUESTIONS_AND_ANSWERS.md](QUESTIONS_AND_ANSWERS.md), the hard questions first, every answer linked to its record; among them how each instrument accounts for sex and the X chromosome.

## Three ways in

### 1. Look (no install)

- Example reports from the chain, exactly as it wrote them: real healthy whole blood, real healthy CD8 T cells (IAM-A) and a labeled planted
  example, in [Biological_Physics/MethylPhys/doors/data/REPORT_EXAMPLES_V4/final/](doors/data/REPORT_EXAMPLES_V4/final)
  (each as a researcher PDF, a plain-language PDF, its figures and its result bundle; the HTML report, with the full technical
  record, is written alongside whenever you run the chain).
- What each instrument measures and how it was commissioned: [Biological_Physics/MethylPhys/INSTRUMENTS.md](INSTRUMENTS.md),
  [Biological_Physics/MethylPhys/COMMISSIONING_REPORT.md](COMMISSIONING_REPORT.md) and the commissioning notes in
  [Biological_Physics/MethylPhys/doors/](doors) (`COMMISSIONING_NOTE_*.md`).
- In the book: *Reading a report, step by step* and *The commissioning record, reproduced*.

### 2. Run the chain (about 30 minutes)

From GitHub, downloading only the biology folder (about 0.85 GB):

```
git clone --filter=blob:none --sparse https://github.com/hmahaffeyges/IAM-Validation.git && cd IAM-Validation
git sparse-checkout set Biological_Physics
```

Or from the Zenodo archive: unzip it, then in the folder run once (the release check lists the package's files with git):

```
git init && git add -A && git commit -qm archive
```

Then, with Python 3.11:

```
cd Biological_Physics/MethylPhys/chain
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python3 release_check_v4.py          # every check must pass on your machine first (about 6 minutes on 4 cores)
cd MethylPhys_Interface
python3 run_sample.py --grn S_Grn.idat --red S_Red.idat --specimen "whole blood" --id S001 --out S001.html      # an EPIC v1 array
python3 run_sample.py --pat S.pat.gz --seq-cell "cd8 t cells" --id S002 --out S002.html                         # whole-genome bisulfite
```

Two groups run in one series (Group-A): read each person with `run_sample.py` as above, list them in a CSV (`id`, `slide`, your group
column, optionally `healthy_reference`), then

```
cd ..
python3 stage_groupa.py collect MethylPhys_Interface/ samples.csv readings.csv    # one row per person from the report bundles
python3 stage_groupa.py compare readings.csv group patients controls             # the tared comparison and the call
```

The full procedure, with every option, what the chain declines and why, and every frozen file with its checksum:
[the operating procedure](sop/MethylPhys_CPG_SOP_v4.md) and
[the operations manual](manual/MethylPhys_CPG_Operations_Manual.pdf).

### 3. Rebuild a result from public data (one command each)

```
bash Biological_Physics/MethylPhys/reproduce/setup.sh                          # once: makes Biological_Physics/MethylPhys/reproduce/.venv
python3 Biological_Physics/MethylPhys/reproduce/reproduce.py --list            # every test, with its time, disk and memory
python3 Biological_Physics/MethylPhys/reproduce/reproduce.py cscore_band       # one test: downloads public data, runs it, compares every number with the committed result
```

Each test downloads only public files (GEO, ArrayExpress, ENCODE, the NCI Genomic Data Commons, ENA, Zenodo) and compares with the committed
result taken from git, so it runs from a clone or an unpacked archive after the `git init` step above. The tests, what each shows and its
fresh-clone result: [Biological_Physics/MethylPhys/reproduce/README.md](reproduce/README.md).

## What is where

| Folder or file | What it holds |
|---|---|
| `Biological_Physics/MethylPhys/chain/` | The chain: every stage, the frozen reference files it reads, the release check, the one command |
| `Biological_Physics/MethylPhys/atlas/v2.1/` | The compact copy of IAM-Atlas v2.1 the chain reads, with its checksums |
| `Biological_Physics/MethylPhys/doors/` | Every test's sealed rule and outcome record, the commissioning notes, and their data |
| `Biological_Physics/MethylPhys/reproduce/` | The one-command rebuilds and their verified results |
| `docs/book/` | The book's LaTeX source and `verify_book.py`, the script that checks every derivation and calculated number |
| `Cosmological_Physics/` | The cosmology chains and data behind Part II of the book |
| [MANIFEST.md](../../MANIFEST.md) | Every test, numbered, with its status and outcome |
| [Biological_Physics/MethylPhys/RUNS.md](RUNS.md) | Every rented-computer run, with the code it ran and what it cost |

## A result from your own samples

Run the chain on samples whose answer you know. Whether it agrees or not, the result belongs in the public record: open an issue with the
**Report a result** form (https://github.com/hmahaffeyges/IAM-Validation/issues/new/choose), attach the report and its bundle, and say what
you knew about the samples before you ran them.

These are research instruments, not a diagnostic test. The code is under the MIT License.
