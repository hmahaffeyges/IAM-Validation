# DEV-MINICELL-01: the mini cell, a molecule-level simulation of a cell and its sequencing meter

**Status:** design, written before any code (development). **A simulation of the model, not data.** It counts as evidence only where a
prediction is written down from it first and then compared with real data it was not tuned on.

**Not a milestone:** a development tool for testing the instruments and designing tests.

**Purpose.** A small system model of one cell type and one sequencing library, with every lever adjustable, so the chain can be run on
known truth: steady state (what each instrument reads for a given set of levers), sensitivity (which reading moves when one lever moves),
and contingencies (a lever removed or pushed: maintenance lost, the eraser over-active, the drive lowered, a laboratory with a bad end meter).

**The cell (inside).** Molecules spanning held and loose regions of the cell's identity sites, at the IAM-Atlas v2.1 site layout. Per
copy cycle: loss f and restore g at held sites (the canon's balance, eps = f/(f+g)); restore delayed after copying (copier lag); scheduled
writing and erasing at set rates; divisions. Levers enter only as these rates; nothing is fitted to a target reading.

**The meter (library and sequencer).** Each read molecule's calls pass through the library's molecule-end error per side, at that
library's measured reach along the molecule (DEV_ATLAS_V21_01.md: marked side the last call, unmarked side two to three calls), an interior
error per side, read depth, molecule length and duplicate reads. Output: a molecule file (.pat.gz) read by the one command, so every
instrument reads it exactly as it reads a real sample.

**Calibrated first (state estimation), on spent data only.** The three spent sorted libraries' measured end profiles and depth; the
hairpin genotypes (GSE116482: restores per loss with and without writers and erasers); the 153 healthy counts. A calibrated mini cell must
reproduce these before it is used for anything else.

**First uses.**
1. Planted errors shaped like the real fault (the unmarked side reaching two to three calls), for the seventh self-calibration form.
2. The missing input behind the sixth form's overcorrection on the B-cell library.
3. Whether copier lag separates the isolated-loss rate from permanent loss (PHY-060, PHY-044).

**What it would take to succeed:** the calibrated mini cell reproduces the three libraries' end profiles and the hairpin genotypes'
restores per loss within their counting error, before any other use.

**In plain words.** A practice cell and practice sequencer built from measured numbers, so the instruments can be tested on a sample whose
truth is known, the way an operator practices on a training simulator before acting on the live system.
