# PRED-TEMPERATURE-01: the temperature law (PHY-057), numbers sealed 2026-10-15 before any data is searched

**In plain words.** If IAM is right that a held DNA mark is a balance of energy against temperature, then a colder body holds its marks
more tightly by an exact amount, and a hotter body less tightly, with nothing adjusted. A hibernating animal's tagging mistakes should
drop by about a third when it cools to 5 C; a bird's, at 41 C, should sit slightly above ours.

**The rule.** ln[(1 - eps)/eps] = E/kT with E fixed, so the log-odds scale as 1/T. Basis: the measured 37 C line (firm 0.0212, loose 0.0497,
island 0.0124, development/loose_end_2026-10-15/loose_end_line.json). Code: `data/PRED_TEMPERATURE_01/temperature_law.py`; the numbers it
wrote are `data/PRED_TEMPERATURE_01/temperature_law.json` and are the prediction.

**The cleanest test: one species at two temperatures** (the within-species log-odds ratio is fixed by the two temperatures alone: 1.0343
for 18 C against 28 C; 1.115 for torpor at 5 C against active at 37 C). Across species the enzymes differ, so cross-species numbers are
read as a guide, not a test.

**What a miss means.** If the error does not move with temperature, or moves by a clearly different amount on the same species, the
Boltzmann premise does not hold for held marks there, and the gauge's use of kT has to be rethought.

**Data:** searched 2026-10-15 (stage T1, metadata only; see below). Salmon (PROC-SALMON-01) and every other fish set already scored cannot be the test.

**Synthetic test:** PRED_TEMPERATURE_01/temperature_T2_synthetic.py, PRED_TEMPERATURE_01/temperature_T2_synthetic.json

The chain synthetic (stage T2, 2026-10-17, below) supersedes the earlier measurement synthetic (temperature_synthetic.py, run outside the chain on 2026-10-15; kept as history).

The measurement synthetic (molecules planted at two temperatures, counted with the committed PROC-CHANNEL-01 counter) ran twice on
2026-10-15. Run 1 read NOT MET (kept as temperature_synthetic_run1_NOTMET.txt): converting counts back to an error rate with the formula
eps(1 - eps)^2 read the 5 C against 37 C ratio low (1.105 against 1.115) even at 200,000 molecules, a systematic bias in the conversion, not
in the law. Run 2 converts through the counter's own response curve, measured on planted molecules with its own seed, and read MET: at
200,000 molecules per temperature the sealed ratios are recovered within 0.51 percent (18 C against 28 C) and 0.31 percent (5 C against
37 C), told apart in every repeat. At 2,000 molecules the 18 C against 28 C pair is not reliably told apart (4 of 5), so a real test needs
at least about 20,000 qualifying molecules per temperature. The predictions themselves did not change.

The formula self-check (temperature_law_synthetic.txt) also read MET: at 37 C the rule returns the measured values exactly, and the error rises with temperature.

What it would take to succeed: single-molecule methylation of one species read at two body or water temperatures, with the log-odds ratio landing on the sealed value; finish line, not a promise.

# NO CHAIN: a prediction note; no chain stage reads temperature


## Stage T1, data search (2026-10-15, metadata only; no read opened)

GEO searched for bisulfite or methylome series with two temperatures in one species (Arabidopsis; hibernators; any non-human species
with "temperature"); the full candidate list was screened by title and summary.

**Primary: GSE54292** (Dubin et al., Arabidopsis thaliana, Swedish accessions; whole-genome bisulfite, whole rosettes at the 9-true-leaf
stage, two plants pooled per sample). Grown their whole lives at **10 C (157 samples) or 16 C (127)**; **111 accessions were grown at both**,
so each genotype is its own control. Reads on SRA: paired-end 2 x 90 bp, median 10.0 million pairs per sample (range 1.7-22.2 million),
about 13x coverage of the genome (`data/PRED_TEMPERATURE_01/t1_gse54292_design.json`, `t1_gse54292_runinfo_part.csv`). For a plant the
body temperature is the growth temperature, held for its whole life, so the marks are held at one temperature from start to finish.
**The number, already fixed by the law:** colder over warmer log-odds = (273.15 + 16) / (273.15 + 10) = 1.0212 (`data/PRED_TEMPERATURE_01/t1_plant_number.py`). Small: about a 2 percent
change in log-odds, so the paired design and the molecule count decide whether it can be seen; stage T2 measures that on synthetic data
first.

**Second: GSE199815** (Syrian hamster liver, whole-genome bisulfite; 3 animals each at late torpor, early arousal and euthermic).
Different animals per stage, torpor lasts days within weeks of euthermia, and body temperatures are not recorded in the series, so the
temperature at which the marks were held is not known. Kept as a second, weaker test.

**Not used:** the heat-stress Arabidopsis series (acute stress of hours to days, not a temperature held for life); salmon and other fish
already scored.

**Next, stage T2:** a synthetic on planted plant-like molecules with the committed counter, to find how many molecules per accession and how
many pairs it takes to see 1.0212 against the plants' own spread; then the sample list and the reading rule sealed before any read is
opened.

## Stage T2, synthetic through the chain (2026-10-17)
**The instrument's job:** a plant grown colder reads a lower held error, by the amount the law fixes from the two temperatures alone.
Chain stage TL (`chain/stage_temperature.py`) counts with the chain's own counter on the nuclear chromosomes, reads conversion on the chloroplast (unmethylated; in plants the non-CpG contexts are methylated, so the human check does not apply), and corrects for the sequencer's own error through the counter's response curve: a held site reads lost if it was lost or misread, independently, p = eps + s(1 - eps); p is read from the counter's statistic through its response to the sample's own calls-per-molecule mix, then eps = (p - s)/(1 - s).

Run 1 (NOT MET, kept in the log): the plain subtraction used for the tumor pairs over-corrected the planted held error by about 9 percent, more than the effect sought (about 7 percent lower held error at 10 C than at 16 C); and a bar requiring the 95 percent interval to contain 1.0212 was tighter than the job, failing on a 0.3 percent bias. Run 2 (MET, `PRED_TEMPERATURE_01/temperature_T2_synthetic.py`, kept as `temperature_T2_synthetic_run2.json`): planted per-site held error 0.030 read 0.0305 after the conversion correction was added (stage T3); conversion 0.0067 against 0.006 planted; 111 pairs with the two temperatures on different sequencer error: law true met 40 of 40, no temperature effect met 0 of 40, 20,000 molecules per plant enough. An uncorrected batch drift of 3 percent or more breaks it (met 0 of 40), so the sealed rule compares pairs within flowcells shared by both temperatures.

**Bars to seal (at most three):** B1 the paired median log-odds ratio, colder over warmer, lands within a quarter of the law's shift (1.0159 to 1.0265); B2 its 95 percent bootstrap interval excludes 1. Gates per sample: at least 20,000 qualifying molecules; chloroplast conversion failure at most 1 percent; flowcell read from the first read header before any methylation is counted.

## Stage T3: the sealed reading on the plants (sealed 2026-10-17, before any plant read is counted)

**The instrument's job:** a plant grown colder reads a lower held error, by the amount the law fixes from the two temperatures alone.

**Data:** GSE54292 (BioProject PRJNA236110), Arabidopsis accessions grown at 10 C and 16 C. 111 accessions have both; 233 samples, 394 runs, 584.7 GB of FASTQ from ENA (`PRED_TEMPERATURE_01/t3_pairs_runs.csv`, built by `build_t3_pairs.py`, sha256 5bb99b79b8c2c9559829ddf7e7739bb286381c9365de2a56294cf1bcaf87a248). Only one FASTQ header was opened before sealing; the public files carry no flowcell or lane (`@SRR1138710.1 1/1`), so the flowcell pairing named in stage T2 cannot be used. It is replaced by the conversion correction below and an information-only check on sequencer error.

**Sources:** all public. GEO GSE54292 / SRA PRJNA236110 (Dubin et al. 2015): the plants' bisulfite reads and their growth temperature; ENA: the FASTQ files; Ensembl Plants release 60: the TAIR10 genome with the chloroplast; Bismark, Bowtie 2, Trim Galore: the published aligners.

**Alignment** (`PRED_TEMPERATURE_01/box/align_plants.sh`): Trim Galore (HiSeq 2000, no poly-G trim), Bismark with Bowtie 2 on TAIR10 (Ensembl Plants release 60, with the chloroplast), deduplicated, sorted.

**Count and correction:** chain stage TL (`chain/stage_temperature.py`, sha256 943cda56bc674dd6f675bb6a4c7df7c2a7e983323aba9ae08e44da4e9ea21e48) on the five nuclear chromosomes. The held error is read through the counter's response curve, then corrected for the sequencer's own error s and the chloroplast conversion failure c: eps = (p - s)/(1 - c - s).

**Gates per sample:** at least 20,000 qualifying molecules; chloroplast conversion failure at most 1 %; corrected held error above zero. A sample failing a gate is left out with its reason. Replicates at one temperature are averaged.

**Bars** (scorer `PRED_TEMPERATURE_01/score_temperature.py`, sha256 a5a85be9341e4806e9a012f884ac8378163c91d0e16b93c2baa97f4d807c84b3, run once):
- B1: the paired median log-odds ratio, 10 C over 16 C, lies within a quarter of the law's shift, 1.0159 to 1.0265 (law 1.0212; nothing fitted).
- B2: its 95 % bootstrap interval excludes 1.

MET when both hold. **Information only:** the cold/warm ratios of sequencer error and conversion failure, and the reading repeated on pairs whose sequencer errors differ by less than 20 %.

**Scorer test (planted counts):** `PRED_TEMPERATURE_01/score_temperature_planted.py`, `score_temperature_planted.json`: law true read MET (median 1.0217); no temperature effect read NOT MET (median 1.0014); a planted gate failure was left out.

## Amendment, 2026-10-18, before any plant result was scored

The counter's correction step (calls per molecule -> per-site rate) was weighted by the length mix of every molecule, including unmethylated ones
that never qualify (found by DEV-CFDNA-SHORT-01; LESSONS C23). Computed exactly on a planted genome with 10 % unmethylated islands and the law exactly true, that bias moves the cold/warm ratio from 1.0212 to 1.0179-1.0195 (PRED_TEMPERATURE_01/counter_bias_check.json): inside the B1 window, but 10-15 % of the effect being measured, and larger in a genome with more unmethylated DNA. (A first, larger estimate came partly from the noise of the old simulated curve and is withdrawn; development log 2026-10-18.) Stage T3 run 3 was stopped at 159 of 233 samples so the verdict is read on a correct counter. No pair was formed, no
ratio was computed and no sample's value was read; the 159 counts were moved, unread, to results/PRED_TEMPERATURE_01/T3/superseded_run3_counter_bias/.
The fix is in the chain (qualifying-molecule length mix; exact response curve). The rule, the bars and the scorer are unchanged. T2 passes on
the fixed counter (run 3, `temperature_T2_synthetic.json`: planted 0.030 read 0.0303). All 233 samples are counted again (run 4).
