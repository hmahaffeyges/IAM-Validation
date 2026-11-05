# DEV-SEQ-INTAKE-01: the genome-coverage intake check (written and committed before its synthetic runs)

**The instrument's job:** the chain reads a sequenced sample only when its reads cover the whole genome evenly, and refuses, with the reason,
a sample that was read only in part.

**Why.** GSE98837's runs on SRA are stored in genome order. The B and T test read each from its first 140 million read pairs, which were
chromosomes 1 to 8; the chain read them anyway, Met-A refused them at its 90 % identity-site rule, and the cause took a week to find
(`DEV_METAA_ATLAS_CURVE_01.md`, the B and T second reading and its corrections). The author: the chain, when it first investigates what a sample
is and how it was prepared, must make sure this cannot happen again.

**The check** (`chain/stage_seq_values.py::genome_coverage`, called by `run_sample.py` on every sequenced input, .beta, bigWig and molecule
files alike, before any reading): for each autosome, the share of IAM-Atlas v2.1's sites on it that carry a value, against the median
autosome's share. Any autosome under half the median refuses the sample (`GENOME_COVERAGE_REFUSED`, exit 3, the bundle naming the short
chromosomes). It compares chromosomes with each other, so a uniformly thin library such as plasma passes. The sex chromosomes are not tested
(a woman has no chrY; chrX carries half the calls in a man). `--dev-allow-partial-genome` skips it for a synthetic that cuts a genome on
purpose, flagged.

**Rule (the same lesson for the workflows):** no reading takes the first N read pairs of a run; a run is read at full depth, or its order is
checked first.

**Synthetic** (`data/DEV_SEQ_INTAKE_01/genome_coverage_synthetic.py`, output `genome_coverage_synthetic.txt`): one spent healthy atlas input
written four ways and read by the one command. Bars: (A) whole, passes; (B) chromosomes 1 to 8 only, refused; (C) every CpG kept with
probability 0.05, passes; (D) chromosome 17 removed, refused. All four as stated, or not met.

**Synthetic run 1** (`data/DEV_SEQ_INTAKE_01/genome_coverage_synthetic_run1.txt`): all four cases passed or were refused as wanted, but case B
(chromosomes 1 to 8 only) was refused for the wrong reason: with 14 of 22 autosomes empty the median autosome is empty, so the check said no
autosome carries a value and named all 22. Amended (disclosed): the reference is the median share of the autosomes that carry any value, and
a bar is added before run 2: a refused case names exactly the chromosomes it lacks (B: chr9 to chr22; D: chr17).
