# DEV-METHIONINE-01 — methionine depletion as a second SAM lever (development; step 1 written 2026-10-10, nothing downloaded)

**Candidate.** Yokogami et al. 2022, BMC Cancer 22:1351 (doi 10.1186/s12885-022-10280-5); reads PRJDB12471 (DDBJ, public): three glioma-initiating
cell lines (MZGC1-3), complete vs methionine-deficient medium, RRBS 150 bp paired (HiSeq X), one library per condition. The paper reports
intracellular SAM lowered by depletion (Fig. 2B), depletion for 48-72 h, and markedly decreased proliferation with cell death.
Public search 2026-10-10 found no other methionine-restriction sequencing set (GEO, SRA); GSE225944 (homocysteine for methionine) is arrays only.

**Step 1, calculation (`development/sims/methionine_01.py`, output `methionine_01_output.txt`).** Copy error is written only when DNA is copied.
In 48-72 h only a share of the cells copy their DNA, and each copied duplex carries one new strand. Mapped through Stage Q's measured response on
RRBS molecules (wild-type mouse liver stand-in), IAM-A_rel is >= 1.08 in 6 of 27 cells of the grid, all needing a 10-fold SAM fall or most cells
copying in the window; with half the cells copying once and a 4-fold fall, IAM-A_rel is 1.03-1.05.

**Decision: parked.** With three pairs and one library each, a rise of 1.03-1.05 cannot be told from library-to-library spread (0.02-0.04).
The set is underpowered for the reason the physics gives: too few copies were made during the depletion. A decisive second lever needs restriction
long enough for many divisions (weeks in culture, or diet in vivo); none is public as of this date.

**In plain words.** We wanted to see whether starving cells of the raw material for methyl marks makes them hold their pattern
less well. Before downloading anything we worked out whether the only public experiment could show it, and it cannot: the cells were
starved for just two or three days and nearly stopped dividing, so too little could change to see over the normal spread between
samples. The test is parked until a longer starvation experiment is public.

**Correction note (2026-10-13).** Step 1 says copy error "is written only when DNA is copied." That wording predates the balance
derivation (CANON/foundation.json `balance_equation`, book section on the held state): the held error is the balance of restore and loss
over ANY interval, and non-dividing cells hold it too. What copying adds is a third loss route, missed restores on newly copied strands
(HYP-SATURATION-01, falsifier 7). The calculation's conclusion for this data set stands, because the starved cells nearly stop dividing
and so lose the copying route: a decisive methyl-supply test still needs weeks of restriction. The step 1 text is kept as written, as a
dated record.

**What it would take to finish:** a public methylation sequencing set from cells or animals kept on restricted methionine for weeks.

**Not a milestone:** parked before any data was read.

**Second use, sealed before any library is read (2026-11-12; PHY-083): energy or supply.** **Where IAM's Law is:** the hold's gain is the coin in thermal kicks. If the gain is the coin's ENERGY, a fall in SAM/SAH by a factor r lowers each methyl transfer by ln r of a kick and the held error at held sites rises only slightly (Einstein-relation route with the dyad's two coins, PHY-083); if the gain is the coin's SUPPLY, the held error rises by the factor r. Predicted rise of the held error, depleted over control, same line: r = 1.5: energy x1.018, supply x1.5 | r = 2: energy x1.032, supply x2 | r = 4: energy x1.066, supply x4 | r = 10: energy x1.114, supply x10. The earlier parking reason (too few copies in 48-72 h) predates the balance (rule 25: the held error is the writing/erasing balance over any interval, independent of division), so this question does not need divisions. **Inputs:** r from the paper's own SAM/SAH measurement (Fig. 2B, ELISA, MZGC1 and MZGC2 only; read from the figure by two readers independently, not by me alone); MZGC3 has no SAM measurement and is read as a third pair without r. **Reading:** the six RRBS libraries (PRJDB12471, DRR322659-DRR322664) aligned and counted by the chain (run_sample.py --bam with --ref-fasta), held error at held sites by the lone-loss counter, scored by the agreed bar method (pull at k = 2, library spread from the control libraries). **Which route wins:** the supply route is excluded if the rise is below the factor r at k = 2; the energy route is excluded if the rise exceeds its prediction at k = 2. **Known caveats, stated before reading:** tumor-derived glioma-initiating cells, not healthy; depletion increased cell death; the abstract reports global demethylation (a level change, not the held error read here); RRBS reads are short (150 bp), so lone losses are counted on few calls per molecule. Synthetic first: the counter on 150 bp molecules with a planted held error, before any library.
**Synthetic test:** `DEV_METHIONINE_01/methionine_synthetic.py`, `DEV_METHIONINE_01/methionine_synthetic_summary.txt`
RRBS-length molecules (3-15 CpGs,
assumed), loose 0.20, read by the one command (run_sample.py --dev-count-only): planted rises x1.066, x1.5, x4 read x1.053, x1.379, x2.662;
control replicates agree within 0.87 %. The counter compresses large rises, so each route's prediction is scored in the counter's own
reading, mapped through this synthetic re-run at the libraries' own molecule lengths before any library is scored.
**Reading job (written before any library is read):** each library (`data/DEV_METHIONINE_01/met_samples.csv`) by the committed GitHub pipeline
(`data/DEV_ERASE_01/erase_gh.sh` through the erase_test workflow): the same 20 spread blocks of 10,000,000 read pairs, Trim Galore 0.6.10
--rrbs --paired (new library mode "rrbs"; RRBS fragments are not duplicate-marked, since they share their cut sites by design), bwa-meth
0.2.0 on hg19 from the index release, wgbstools 0.1.0 bam2pat, then the one command's count (run_sample.py --dev-count-only). One library
first as a canary (MZGC1 control); the synthetic is re-run at that library's own molecule lengths before any depleted library is scored.
**The SAM/SAH falls, two independent readings of Fig. 2B (2026-11-12, before any depleted library is read):** the figure shows three
pairs (the legend names only MZGC1 and MZGC2, a discrepancy in the paper), each control normalized to 1.0. Reader one (the author): MZGC1
depleted 0.75, MZGC3 0.25-0.30, MZGC2 no bar drawn. Reader two (a separate vision reading given only the image): MZGC1 about 0.70, MZGC3
about 0.20-0.25, MZGC2 about 1.0. Computed by `DEV_METHIONINE_01/predictions_by_line.py` (`predictions_by_line_summary.txt`): MZGC1 r
1.33-1.43, energy route x1.013-1.016, supply route x1.33-1.43; MZGC3 r 3.33-5.00, energy route x1.057-1.077, supply route x3.33-5.00;
MZGC2 not scored (the readings disagree). Each prediction is mapped through the synthetic at the libraries' own molecule lengths before
scoring. **Open caveat:** SAM/SAH was measured at 72 h; the paper does not state when the RRBS libraries were taken.
**Predicted rise of the held error, depleted over control:** MZGC1 energy x1.013-1.016, supply x1.33-1.43; MZGC3 energy x1.057-1.077,
supply x3.3-5.0 (each mapped through the synthetic at the libraries' own molecule lengths before scoring). **Open caveat:** SAM/SAH was
measured at 72 h; the paper does not state when the RRBS libraries were taken.

**Read and scored as sealed (2026-11-12): neither route met; the held error fell in both lines.** (`DEV_METHIONINE_01/score_methionine.py`,
`DEV_METHIONINE_01/score_methionine_summary.txt`; counts in `DEV_METHIONINE_01/out/`, the one command through the erase_test pipeline;
the synthetic re-run at the two controls' own molecule lengths, `methionine_synthetic_lengths.py`, `methionine_synthetic_lengths_summary.txt`,
before any depleted library was opened). **Where IAM's Law is:** the work per methyl transfer, in coins of M kT, against the methyl
supply. MZGC1: held error depleted over control 0.936 (energy route 1.012-1.015, supply route 1.267-1.343). MZGC3: 0.929 (energy
1.053-1.070, supply 2.392-2.708), its loose share unchanged (0.645, 0.643). **What it shows:** the held error did not rise with the methyl
supply cut three- to five-fold; it fell about 7 % in both lines. The supply route (the hold set by the size of the methyl pool) is ruled out
by a wide margin. The energy route's predicted small rise was not seen either; something moves the held error down by about 7 %, the same
sign and size in both lines. Leading candidate, not tested: methionine restriction slows division, and the counter cannot tell a freshly
copied strand still awaiting its copier from a loss, so fewer copies in flight read as a lower held error. Open caveat as sealed: SAM/SAH
was measured at 72 h and the libraries' timing is not stated. One library per arm, so no replicate sets the noise.

