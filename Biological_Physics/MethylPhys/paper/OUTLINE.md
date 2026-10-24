# The paper: outline and figure plan (plan step 50; draft, not for circulation)

Format: that of the published information-theory study of pediatric acute lymphoblastic leukemia (Koldobskiy et al., *Nature Biomedical
Engineering* 2021), whose structure this mirrors: an abstract of about 150 words; an introduction of about 500 words ending in what the
paper shows; Results under headings that each state one finding; a Discussion of about 1,500 words; Methods by subsection with the
equations; Data availability and Code availability; five or six main figures and a supplement.

**Rules for this draft.**
- No number is typed by hand in the paper. Every value is read from the canon (`CANON/manifest.json` rows and the test records named
  below) by the paper's generator, which is to be written; a value whose test has not run shows as `[pending: TEST]`. The values quoted
  in this outline are copied from `INTRODUCTION.md` for orientation only.
- Statistical and information-theoretic names only: a Bayesian hierarchical model sampled by Hamiltonian Monte Carlo (No-U-Turn),
  laboratory and platform terms estimated in a calibration stage and held fixed in the sampling (not sampled jointly: that was tested and
  not adopted), split-half null tests, an information-geometric embedding (Fisher-Rao distance), a
  thermodynamic lower bound (Landauer). How the methods were arrived at belongs to the book, not the paper.
- The framing builds on the published study: it shows what a derived bound adds to its approach, tested on its own data, and credits what
  was taken from it.
- Gate (plan step 49): nothing is circulated until the chain is complete, Met-A's healthy line is derived, and the study is replicated.

## Working title

*A thermodynamic bound on the fidelity of maintained DNA methylation reads epigenetic disorder in single samples without matched controls*

Alternative: *Reading methylation disorder against a physical floor: a control-free, platform-independent framework tested on pediatric
leukemia*

## Abstract (about 150 words; written last)

1. The problem: methylation disorder is central to cancer, but measuring it needs either matched normal controls or a model fitted inside
   each deeply sequenced sample, which excludes arrays, shallow sequencing and cell-free DNA.
2. What we introduce: a lower bound on the error of a maintained mark from the thermodynamics of information, and readings of each sample
   against that bound and against its own reference, with nothing fitted per sample.
3. The shared reference: a 79-cell-type methylation atlas built by a Bayesian hierarchical model with laboratory and platform terms
   calibrated across 21 sources, with full posteriors and convergence diagnostics at every site.
4. The test: the published leukemia whole-genome bisulfite data reread, `[pending: DEV-ALL-INFO-01]`; the UHRF1 knockout,
   `[pending: DEV-ERASE-01 UHRF1]`.
5. Reach: arrays and cell-free DNA, `[pending]`. All code, data and the atlas are public.

## Introduction (about 500 words; 4 paragraphs)

1. Methylation disorder in cancer: stochasticity as a hallmark; the information-theory study of pediatric leukemia (Ising model, normalized
   methylation entropy, Jensen-Shannon distance, UHRF1) and the earlier work on epigenetic stochasticity it builds on.
2. Two limits such measures share: the scale is set by controls (in that study, 5 normal samples), and the model is fitted inside each
   sample, so it needs deep sequencing.
3. What physics offers: a maintained mark is a stored bit; holding it against thermal noise has a minimum cost (Landauer), and that cost
   bounds the error a cell can reach. The bound gives a scale whose ends are derived, not learned.
4. This paper: the bound and the readings built on it; the reference atlas; a test on the published data, including the knockout of the
   maintenance factor UHRF1; reach beyond deep sequencing.

## Results (each heading states its finding once the test has run; until then it states the question)

**R1. A thermodynamic bound sets the floor and the ceiling of methylation fidelity.** The derivation: the cost of holding one mark
(M = 20.94 k_BT per ATP); the energy actually spent holding a mark, measured on single molecules in 56 healthy cell types (3.08 k_BT); the
floor and the ceiling; the read-count correction, derived exactly. A prediction made before the data: methyl-donor restriction in mice,
window 1.016 to 1.135, read 1.1239. *Sources: `doors/data/DEV_CHAIN_Q2_01/holding_energy_q2.json`; `doors/DEV_SAM_LEVER_01_OUTCOME.md`; the
ceiling and read-count records.* **Figure 1.**

**R2. A sample's own loosely held share sets its expected error.** IAM-A: held errors on held molecules against the sample's own loose
share; the self-reference line from healthy cell types; why this is not circular (synthetic tests); reproduction in a second laboratory
and in mouse. *Sources: IAM-A records; `doors/DEV_REPLICATE_01.md`.* **Figure 2.**

**R3. A jointly calibrated atlas lets any sample be read without a reference of its own.** IAM-Atlas v2.1: 21 sources, laboratory and platform
terms calibrated across them, posteriors and convergence diagnostics at every site, split-half null maps; deconvolution reproduced across laboratories; Met-A
and its healthy line derived laboratory by laboratory. *Sources: `atlas/v2.1/README.md`; `doors/DEV_METAA_HEALTHY_01.md` (bar 2), layer 1
(plan 14.7), `[pending]`.* **Figure 3.**

**R4. The published leukemia data reread.** Pass 1 (IAM-A at fixed depth, all 36 samples); pass 2 (full depth, region by region beside the
study's five findings: entropy gains, bivalent domains, promoters, discordant genes, the UHRF1 knockout). *Source:
`doors/DEV_ALL_INFO_01.md`, `[pending]`.* **Figure 4.**

**R5. Loss of a maintenance factor read against the bound.** The UHRF1 knockout read molecule by molecule: held error against loose share;
the entropy turnover in heterochromatic domains read as the distance to the ceiling. *Source: `doors/DEV_ERASE_01.md` (UHRF1),
`[pending]`.* **Figure 4 or 5.**

**R6. Beyond deep sequencing: arrays and cell-free DNA.** The same readings on arrays and on shallow plasma sequencing; the molecule-level
loose share that does not change under mixing; cell types the atlas lacks, flagged. *Source: `doors/DEV_ONE_DRAW_01.md`, `[pending]`.*
**Figure 5.**

**R7. Genome-wide departures in one picture.** The genome-wide projection and the information-geometric embedding of a sample's departures,
built for the study's own samples; each site's sensitivity to a small loss of holding (after the study's methylation sensitivity, derived
analytically here). *Source: plan 49.5, `[pending]`.* **Figure 6.**

## Discussion (about 1,500 words)

What a derived bound changes; what was taken from the published study (nearest-neighbor coupling, methylation sensitivity, the knockout as
a test) and how it was built in; what neither method yet does; limitations, stated plainly; the next study.

## Methods (subsections)

Data (all public; the accession list generated from `CANON/data_spent.csv`). The bound (equations). IAM-A: molecule counting, the loose
share (tile and molecule-level), held error, the self-reference line. The atlas: the hierarchical model, priors, sampler settings,
convergence, joint offsets, split-half nulls. Deconvolution. Met-A and its healthy line. The territories. The genome-wide projection and
the embedding. Alignment (bisulfite read alignment, fixed read sampling). Pre-registration: every test written and committed before it
ran. Statistics.

## Data availability and Code availability

All data are public (accessions listed). The chain, the atlas and every test record are public in the IAM-Validation repository; one
command reproduces every figure.

## Author contributions (draft)

Conception, derivation, software, analysis and writing; an independent reproducibility audit from a clean clone, following only the
written instructions (plan 50.4).

## Figures

| Figure | Panels | Status |
|---|---|---|
| 1 | the bound: cost of a mark, floor and ceiling, energy measured across 56 cell types, the mouse prediction | data in hand |
| 2 | IAM-A: held error against loose share, the line, second-laboratory and mouse reproduction | data in hand |
| 3 | the atlas: offsets, posterior widths, convergence, split-half null; Met-A's healthy line | the healthy line pending |
| 4 | the published leukemia data reread, beside the study's findings; the knockout | pending |
| 5 | arrays and cell-free DNA | partly pending |
| 6 | the genome-wide projection and embedding of the study's own samples; sensitivity | pending |
