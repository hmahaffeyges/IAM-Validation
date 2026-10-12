# DEV-IAMA-NEURON-01 — sorted cortical neurons as a commissioning cell for IAM-A and Met-A (candidate data, 2026-10-11; nothing read, no bars)

**Why (author, 2026-10-11).** No public set of healthy sorted neutrophils from a second laboratory with 3 or more donors exists, so IAM-A test C4
cannot be met on neutrophils. Neurons have public healthy data from several laboratories, are the most ordered healthy cell measured
(PROC-CHANNEL-01: copy error 0.024, among the lowest of 56 cell types), and have stopped dividing, which tests whether healthy height depends on
divisions. Both instruments on the same cell type becomes possible.

**IAM-A (whole-genome bisulfite of sorted NeuN+ neurons).** Counts from `data/DEV_IAMA_NEURON_01/neuron_summary.py` (output `neuron_summary.txt`, table `neuron_check.csv`, search record `neuron_check_search_record.md`).
- Reference: GSE186458 (Loyfer; Hebrew University). 8 Cortex-Neuron samples, 6-8 donors (to confirm from the paper's supplementary table); public
  hg19 .pat files; already used to build references, so not a test.
- Test laboratory 1: GSE96612 (Feinberg laboratory, Johns Hopkins; shares no GEO contributor with Loyfer). 6 donors without neurological or
  psychiatric disease, 23 NeuN+ libraries across 4 regions, public FASTQ (SRP101887); BA9 alone 6 runs, 188 GB. Donor age, sex and post-mortem
  interval in the paper supplement only.
- Test laboratory 2: GSE107729 (Yi laboratory, Georgia Tech). 25 control NeuN+ donors (ages 25-80, post-mortem interval given for 20), public
  FASTQ; 3.0 TB. Controls of a schizophrenia study.
- Not usable: GSE330111 (103 donors, raw reads controlled access); GSE47966 (2 donors, hg18).

**Met-A (EPIC v1 arrays of sorted neurons, raw IDATs).**
- GSE306226 + GSE306227 (Pishva laboratory, Maastricht): 39 neurologically healthy neuron arrays, plus glia from the same donors.
- GSE112179 (Labrie laboratory): 33 control NeuN+ arrays from 28 donors.
- Exeter sets GSE234520 / GSE279509: healthy subset not in GEO; possible donor overlap with the Maastricht sets (UK brain banks).

**Missing.** Same-donor arrays and whole-genome bisulfite outside the reference laboratory (only the Moss pair, 2 EPIC donors, already read, the
reference laboratory's own). So the Met-A to IAM-A link test is not possible on neurons with independent data. No whole-genome bisulfite of
neuron-derived tumours is public (arrays only: central neurocytoma GSE288337, ganglioglioma GSE218542).

**Questions to answer before any bar is written.**
1. A post-mitotic cell makes no new copies: what IAM-A's copy error measures in neurons (errors written at the last divisions, plus maintenance
   since), derived and then tested on synthetic data.
2. The healthy reference built on Loyfer's 6-8 donors (donor count and depth from the paper's table).
3. Post-mortem interval: record per sample; test whether copy error follows it in the Yi controls before using them.

**Reference rule, written 2026-10-11 before the Loyfer donor table (Supplementary Table 1) is read.** The Loyfer neuron files are being read
whole by Stage Q (copy error and Stage Q0, `neuron_eps.csv` to be committed with its script); their readings have been seen, so they serve as
the neuron reference only and test nothing.
1. A reference is one cell type. Cortical neurons ("Cortex-Neuron") and cerebellar neurons ("Cerebellum-Neuron") are separate cell types and
   are not pooled. The sample titled "Neuron" with no region joins the cortical reference only if the donor table places it in cortex.
2. No donor is removed for its reading. A donor is excluded only for a reason recorded by the depositors: a diagnosed brain or psychiatric
   disease, or a Stage Q0 refusal. Age, sex, cause of death and post-mortem interval are recorded beside each reading, not used to adjust it.
3. The cortical reference's healthy spread is measured leave-one-donor-out. Whether healthy cortical neurons fit the Normal band
   (0.95-1.05) is a measured property of that spread, reported as found; the band is not changed to fit it.

**Donor table read, 2026-10-12** (Loyfer et al. 2023, Supplementary Table S1, fetched from the journal; `data/DEV_IAMA_NEURON_01/loyfer_neuron_donors.csv`).
Ten neuron samples from seven donors, all brain autopsies at Karolinska: cortex from five donors (one donor, aged 25, gave three samples), one
cerebellum, and one sample with no region (GSM5652224). Recorded causes of death: suicide for two donors, accident for one, blank for the rest; no
brain disease is recorded. Three samples are marked "not sorted". **Disclosure on rule 1 (one cell type):** the rule did not name purity. Loyfer's
own estimate puts Cortex-Neuron-Z000000TF at 0.478 neurons, so it is not one cell type and is left out of the reference; its copy error had been
seen before the donor table was read. Every other sample is estimated at 0.898 or more, so any purity cut above 0.478 and up to 0.898 gives
the same set. The unlabelled sample is not placed in cortex by the table and stays out (rule 1). The suicide donors stay in (rule 2); the reference
is also printed without them, descriptive only (author decision). Script: `neuron_reference_01.py`.

