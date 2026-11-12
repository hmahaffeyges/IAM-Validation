# DEV-ATLAS-V21-01 — IAM-Atlas v2.1: roster, build/held-out split and plan (design, 2026-10-12; no v2.1 data read)

**Author decision (2026-10-12):** build v2.1, run the joint fit and commission it, as one project; the rented machine is approved. Each version
gets its own DOI and re-passes commissioning before the chain switches; results scored on v2 stay on v2.

**Prep** (metadata, file listings and open-access papers only; nothing opened): `data/DEV_ATLAS_V21_01/atlas_v21_plan.md` (plan, split rule,
culture check, compute and download estimate, open questions) and `atlas_v21_roster.csv` (one row per series and cell type, with role, donor
lists, verified facts, entry-rule status, risks). Re-query scripts: `fetch_meta.py`, `fetch_papers.py`, `summarize_meta.py`. Source: the gold-mine
map (`data/DEV_ATLAS_COMMISSION_01/gold_mine_map.md`). Limits: papers of 7 series were not open access; donor counts where GEO has no donor field
count arrays; EGA, dbGaP, IHEC, ENA-only and CNCB were not searched; the compute estimate is a scaling argument, not a dry run.

**Split rule, fixed now (before any array is read):** whole series to one role by default; a series already read by any method builds and does
not test; a large unread series is split by donor (natural sort of GEO donor IDs; the first 12 odd ranks build, all others held out); all arrays of
one donor take the same role; at most 12 donors per source and cell type build; a set used to build cannot test, and a test set shares no GEO
contributor with any build source of the cell it tests. Donor lists are in the roster; any change is a new dated entry, never an edit.

**Decisions taken on the prep's recommendations** (author standing rule: the most defensible choice):
- granulocyte fractions enter as neutrophils only if marker-selected (CD15+ or CD16+) and passing the sample-level twin test against the atlas
  neutrophils;
- 12-donor cap adopted;
- 450K-only new cells allowed, with EPIC-only loci recorded as not measured (no filling);
- deferred to a later version: EPIC v2 thymocytes (until the Stage 1 EPIC v2 read check), sperm, thymus, cord-blood stem cells and placenta (no
  use needs them; each would add an assumed source term), adipocyte sequencing;
- cord-blood sets GSE68456 and GSE82084 build nothing, so GSE127824 stays a clean truth set;
- GSE306226 glia, GSE146552 ovarian and tubal epithelium, E-MTAB-4957 children's colon and ileum epithelium: conditional, each with the purity or
  identity check named in the plan, decided before the build;
- GSE132547 and GSE131989: used only after the independence checks named in the plan;
- sex inferred from X and Y probes at Stage 1, after the split is sealed;
- lung stroma: research-only in v2.1 (every public set is cultured); the culture check in the plan runs first;
- compute: NBLOCK 1,400 so 16 or more fits run at once; estimated 21-30 hours at 16 processes on the v2 machine type, entered in the ledger before
  launch.

**Firm build (prep count, provisional until Stage 1):** 428 new samples from 13 independent series; 6 new cell types; 80 cells in all.

**New cell types checked against the cancers and blood tests planned (author, 2026-10-12; gold-mine map and roster, metadata only).**
- Blood cancers: bone-marrow stem cells and promyelocytes (GSE49618) are in the firm build; cord-blood stem cells conditional.
- T-cell leukemia that starts in the thymus: thymocytes (GSE269608) are on EPIC v2, which Stage 1 does not read yet. An EPIC v2 read check
  would open them and other EPIC v2 sets; added to the job list.
- Myeloma (plasma cells) and B-cell subsets: GSE118255 is RRBS, which the build rule does not admit; held-out test only.
- Melanoma: melanocyte sets are cultured (GSE166069), too few donors (GSE103413) or of unverified access (GSE16368); the culture check decides
  whether the cultured set can be used, as for lung stroma.
- Placenta (GSE271697): matters for blood tests in pregnancy (placental DNA in the mother's blood), not for any test planned now; kept research
  until a pregnancy-relevant blood test is designed.
- Sperm, dura fibroblasts, nasal epithelium, myoblasts, tonsil NK intermediates: no planned test needs them.

**Culture check, step 1, run once 2026-10-12** (`data/DEV_ATLAS_V21_01/culture_drift_01.py` unchanged since its commit; outputs
`culture_drift_01_summary.txt`, `culture_drift_01_cpgs.csv`). GSE197512, 182 arrays, 7 lines: by the fixed rule, 385,407 of 860,381 CpGs
(44.8 %) drift with population doublings. **Disclosure, found on review:** in five lines the arrays were run on two dates and the later date
holds most of the later passages, so array batch is mixed into the trend there. **Review (descriptive, `culture_drift_01_batch_review.py`, output
`culture_drift_01_batch_review_output.txt`):** on the two lines run on one date (a fibroblast and a keratinocyte), 92,603 CpGs (11.1 %) drift in
both with the same sign. The step-1 set is used as the mask, as fixed: a mask can only remove sites, so a set enlarged by batch keeps the
check valid and costs coverage. Step 2 reports every result with both masks (the step-1 set; the batch-free set, descriptive).
Step 1 also names GSE179847 and GSE82234; they are read before step 2, and the drift set they give is reported beside this one.

**Culture check, step 1b, run once 2026-10-12** (`data/DEV_ATLAS_V21_01/culture_drift_02.py`, unchanged since its commit; output
`culture_drift_02_output.txt`). Columbia, GSE179847, 92 control arrays, three fibroblast lines: 282,723 of 826,716 CpGs (34.20 %) drift with
population doublings; 65.3 % of them are in the step-1 set. Of the batch-free core CpGs measured there, 54.7 % drift with the same sign in
Columbia, against 34.2 % of all CpGs (1.6 times chance). Aachen, GSE82234, three HUVEC donors (descriptive): at 38.1 % of the core CpGs all
three donors move in the core's direction, against 15.3 % at all CpGs (2.5 times). **Reading:** culture drift replicates in two more
laboratories and two more cell types, well above chance and far from completely: part of it is shared by dividing cells in a dish, part is
specific to a cell type or a laboratory. The step-1 mask stands as fixed; step 2 reports every result with it and with the batch-free core.

**Next:** the culture check, step 2 and the conditional checks (laptop); ledger entry; then the build run.

**Test data:** none (design)

**Neutrophil sources checked from GEO metadata, 2026-10-13 (no array read; a correction to the roster's test plan).**
All four series the prep named are already read by earlier work (`CANON/data_spent.csv`: DEV-METAA-450K-01, LOG 2026-10-09; GSE118144 and
GSE122244 also by DEV-BASE-CHAIN-01). Under the committed split rule a spent series builds only, so:
- **Build (unchanged):** GSE124565 (Sawalha and Coit, University of Pittsburgh; 12 healthy donors, 7 women and 5 men, ages 26-67, both IDATs,
450K) and GSE88824 (Florey Institute of Neuroscience; 8 healthy control donors, 450K). Neither shares a GEO contributor with the reference
groups (`CANON/independence_check.py`). GSE124565 alone ends the all-male reference.
- **Not valid as held-out tests:** GSE118144 (spent; its 13 controls are all female) and GSE122244 (spent). *Corrected 2026-10-13, the same day: an earlier line here said it holds no neutrophil samples and is on a chip Stage 1 cannot read; both were wrong. GEO lists 10 sorted neutrophil arrays with IDATs (5 healthy controls, C6-C10, and 5 Parkinson's patients) on GPL23976, and the chain already read this series' monocytes in DEV-NEWCELL-01. Being spent, its 5 healthy neutrophils build only, as the roster already lists them.*
- **Unread candidates for the held-out test, metadata only:** GSE72338 (Charite Berlin, 450K, 18 neutrophil arrays of healthy household
contacts, both sexes, ages 22-49, Sentrix positions published), GSE68456 (University of British Columbia, 450K, 7 granulocyte arrays sorted by
stringent FACS with erythroid exclusion, 5 women and 2 men) and GSE49618 (USC Norris, 450K, 6 granulocyte arrays, no donor fields). Each needs
its healthy status, donor overlap and purity confirmed before it is fixed as the test set.

**The three unread candidates read in detail (GEO series records, 2026-10-13; no array opened).**
- **GSE72338** (Charite Berlin, PMID 26374119): monocytes and granulocytes isolated from the blood of tuberculosis patients and their
household contacts, matched by age and sex; 18 arrays are labeled Neutrophil, all from household contacts, both sexes, ages 22-49, with the
Sentrix slide and position published. **Caveat to state in any result:** household contacts of tuberculosis patients are not screened healthy
donors; latent infection is possible, and the fraction read is granulocytes rather than sorted CD16+ neutrophils.
- **GSE68456** (University of British Columbia, PMID 26366232): **cord blood**, not adult blood; the series exists to show how red-cell
contamination distorts sorted cord-blood populations. The 7 stringent-FACS granulocyte arrays are clean of that problem but newborn, so they
cannot test an adult neutrophil reference. Kept for the newborn work instead.
- **GSE49618** (USC Norris, PMID 23634996): sorted **bone-marrow** cells from healthy volunteers, run as controls for a leukemia paper; 6
polymorphonuclear arrays, no donor fields. Marrow granulocytes are at a different stage of maturation than blood neutrophils.

**Choice:** GSE72338 is the held-out test set for the second-laboratory neutrophil reference, with both caveats above printed with the result;
GSE49618 is a second, weaker check (marrow, no donor fields); GSE68456 moves to the newborn track. If GSE72338's readings sit outside the
sealed band, the caveats are reported as the likely cause and the test is not counted as passed either way.

**In plain words: what the culture mask is.** Every reading compares a sample against healthy cells. Some healthy cell types
exist only as cells grown in a plastic dish, because nobody has published fresh ones. Growing in a dish changes a cell's
switch pattern, so those sites would make a healthy person look damaged. We measured which sites drift, using cells grown
for different lengths of time, and we ignore those sites. That list is the mask. Checked afterwards, part of what it flags
is the machine's measurement day rather than the dish, which is why we keep the larger list: a mask can only remove sites,
never add them, so being generous with it costs us sensitivity and never buys us a result. The drift reappeared in two more
laboratories and two more cell types, above chance but not completely, so the mask stays. **What it would take to do better.** Fresh, uncultured cells for the types we only have from a dish, which removes the guessing; and arrays measured on more than one day per cell line, which separates the dish's effect from the machine's measurement day. Both are searches we can run.

## The 15 open decisions, adopted (2026-10-13)

The author's standing instruction is to take whatever is most defensible, so every recommendation in `atlas_v21_plan.md` section 5
is adopted as written, with its own conditions kept. Decided before any v2.1 array is downloaded or read.

| # | Decision | Condition that still has to be met |
|---|---|---|
| 1 | Granulocytes enter as neutrophils only if marker-selected (CD15+ or CD16+) | each passes the sample-level twin test |
| 2 | At most 12 donors per source per cell | none |
| 3 | 450K-only new cells allowed; EPIC-only loci recorded NOT MEASURED, never filled | none |
| 4 | EPIC v2 thymocytes (GSE269608) deferred to v2.2 | unless the Stage 1 EPIC v2 read check is done first |
| 5 | Sperm, thymus, cord HSC and placenta left out of v2.1 | none |
| 6 | Spent sets may build, unspent sets test (standing rule) | GSE226298 is already in `CANON/data_spent.csv` |
| 7 | GSE306226 "astrocytes" enter flagged | kept only if closer to Tian astrocytes than to any other cell, cross-laboratory r at least 0.986; otherwise research |
| 8 | Nothing built from cord blood GSE68456 / GSE82084 | keeps GSE127824 a clean truth set |
| 9 | E-MTAB-4957 colon and ileum accepted, flagged pediatric | the "n"-suffix arrays identified before the split is sealed |
| 10 | GSE146552 OSE/FTE conditional | kept only if non-epithelial content is under the cut fixed in the plan |
| 11 | Adipocyte WGBS (GSE111445) deferred | none |
| 12 | Blocks of half the v2 size (NBLOCK 1,400) | see below |
| 13 | GSE132547 alveolar macrophages held out for U1/U2 | the paper confirms enrichment and that the authors are outside the Christensen/Salas group |
| 14 | GSE131989 (UCSF) | the paper confirms subjecttype 2 = control and a group separate from Wiencke's |
| 15 | Sex inferred from X/Y probes at Stage 1 and recorded per reading | after the split is sealed |

**Decision 12, settled from the record.** The README says 32 processes; `15_stageB_run_all.sh` runs 16 processes pinned to 8 cores
each, because 32 at once ran out of memory on 2026-09-28. The 0.829 s per locus (`records/14_stageB_timing_block.log`) is one block
measured alone, so v2's real run was the 16-process case (about 11.7 h), and v2.1's estimate is the 16-process column: **21-26 h**
for the firm roster. Halving the block size halves the memory per block; whether that allows 32 processes is measured on one v2.1
block on the box before the full run starts, not assumed.

**Lung and epithelium search, 2026-10-13 (before the split is sealed; metadata only).** GEO searched for sorted or enriched, uncultured
lung, airway and other epithelia on 450K / EPIC with raw files (`data/DEV_ATLAS_V21_01/search_lung_epithelia_2026-10-13.csv`, the query
and every hit). New and usable: **GSE137716** (Clifford, Nottingham), fresh airway epithelial cells by brush or pronase, entered as a
conditional builder for lung bronchus epithelium under the decision-10 rule, healthy never-smokers only. Not usable: GSE185556 (every
sorted sample from children with cystic fibrosis), GSE178808 (an HIV and COPD cohort), GSE161485 and GSE52074 (asthma cohorts, nasal),
GSE102726 and GSE40699 (cultured or cell lines), GSE134217 (pancreatic cancer). The lung tumor tests' blocker is lung stroma, not
epithelium: every public lung stroma set is cultured, so it waits for the frozen-base add-on test (readiness G9) and a labeled cultured
layer, not for v2.1.

**Pancreas search, 2026-10-13 (author: pancreas is his first priority; before the split is sealed; metadata only).** Pancreas is thin
in v2: duct 6 samples and acinar 6 (each Loyfer and Moss), beta 7, alpha 3, delta 3, endothelium 4, at most two laboratories each. The
search (`data/DEV_ATLAS_V21_01/search_pancreas_2026-10-13.csv`, query and every hit) found three unread sets: **GSE223153**, flow-sorted
fresh acinar (5) and ductal (5) cells by RRBS, entered as an independent held-out check at RRBS loci (RRBS covers too few array sites to
build under the coverage rule, which is not changed here); **GSE222147**, laser-captured normal acini (8), normal ducts (7), PanIN (14)
and cancer (6), entered as a flagged second check and as the pancreas precancer series for a breach design (manifest); **GSE143209**,
64 dispersed islet samples, research (a mixture, not a cell type). The rest were cancers, cultures, organoids or not methylation of
sorted cells.

**Thin-cell search, 2026-10-13 (GEO and ArrayExpress through OmicsDI; tissue stroma, tissue macrophages, minor blood cells; metadata
only; every hit in `data/DEV_ATLAS_V21_01/search_thin_cells_2026-10-13.csv`, pancreas OmicsDI hits in `search_pancreas_omicsdi_2026-10-13.csv`).**
682 hits, filtered in code to healthy, uncultured, sorted cells with raw files. One new usable set: **E-MTAB-2145** (Zilbauer, Cambridge;
ArrayExpress only, so earlier GEO searches could not see it), sorted neutrophils, monocytes, B, CD4 and CD8 cells on 450K, 12 arrays each,
all male, entered as an unread held-out laboratory (SDRF committed). The Salas extended blood reference (GSE167998) and its mixtures are
already v2's source for eosinophils, basophils and regulatory T cells and are spent, which is why those cells rest on one source. Not
usable: GSE140295 / GSE140078 (cultured commercial endothelial lots), GSE74877 (cultured primary cells), GSE71244 (no raw files), GSE50386
(histone marks, not DNA methylation). Decision: nothing else to add before the seal; the next search is due before the next build.

**Stage 1 on the box, run once (2026-10-13; readiness L4; `box/v21_stage1_box.py`, committed before it ran).** Reproduction gate first:
three of atlas v2's own Moss arrays recalibrated today matched v2's stored shards at every probe, maximum difference 0.0
(`reproduction_gate.json`), so v2.1's arrays and v2's are one Stage 1. Atlas v2's 292 calibrated shards, until now only on the box's
disk, were copied to S3 (`atlas_v2/stage1_shards/`). All 472 sealed build arrays then calibrated with no errors (211 on 450K, 261 on EPIC
v1); every shard, its metadata and the run manifest with IDAT and shard checksums are in S3 (`results/ATLAS_V21/stage1/`) and the manifest
is committed here (`stage1_v21_manifest.csv`). 37 arrays read below v2's quality rule (call rate 0.93) and leave the build by that rule,
as v2's did. The job's exit code 127 came from a log-copy step after the work (the box has no `aws` command); the work itself completed
and wrote its DONE marker. Next: sex (L5), genotype across build and held-out arrays (the same-donor check), twin tests (L6), source terms (L7).

**Identity job on the box, run once (2026-10-13; `box/v21_identity_box.py`, code published to S3 before any work).** The 472 build arrays
recalibrated with their SNP-probe values were identical to the Stage 1 shards on every array (the new option changes nothing); the 1,797
sealed held-out arrays were fetched, calibrated (none failed) and stored with checksums, and every raw IDAT is in S3
(`results/ATLAS_V21/raw/`); none was scored. The sex and same-donor check (`identity_check_v21.py`, committed before the results were read)
then gave: the genotype bar holds (different people reach at most 0.623 concordance at the 99.9th percentile, against the 0.90 bar);
**no held-out array is the same person as any build array**, so seal 2 stands with nothing removed; no build donor appears in two series.
Sex: 1,125 male, 1,087 female, 57 unresolved (flagged). The metadata comparison that rule c names was added after the run
(`identity_sex_metadata_v21.py`; rule c only flags): 1,457 of 1,477 agree, 20 disagree and are flagged, one of them a GSE146552 build
array (a female tissue called male), which is checked under that set's condition (decision 10).

**Conditions (readiness L8, 2026-10-14; `conditions_v21.json`).** Decision 1 for GSE103541: the paper GEO lists (PMC8011804) sorted its
granulocytes "based on FSC/SSC only", with no CD15 or CD16 antibody, so its 12 granulocyte arrays do not enter as neutrophils (research).
Decision 9: E-MTAB-4957's "n"-suffix arrays are the non-epithelial mucosal fraction of donor 39 and none is a build array; its 16 build arrays
are purified epithelial cells and enter flagged pediatric (this identification was due before seal 1 and was done before any build array was
read for a result). Pending, each on a named data step: D1 GSE224807 and D7 (twin output), D10 (deconvolution after the source terms), D13 and
D14 (held-out paper checks; they do not change the fit). **Seal 3**: while resolving L8 a rule bug was found - decision 7 had been applied to
the whole GSE306226 series, so its 30 firm neuron, microglia and oligodendrocyte arrays were marked conditional. Fixed to the astrocyte
fraction; a row-by-row comparison shows exactly those 30 rows changed and nothing else; the held-out list is unchanged; seals 1 and 2 are kept.

**Twin and source-term job 5115bd05, run once (2026-10-14): stopped by its own reproduction gate.** In v2 mode it matched v2's committed twin
table, split-half null and source terms (to 1e-12) exactly, but not the sample-level twin test (`10_twin_test_sample_level.csv`), so it stopped
before computing anything for v2.1 (`reproduction_gate_terms.json`). It did not save its own v2-mode table, so a diagnostic job runs that one
step and compares it row by row (`box/v21_twin10_diag_box.py`).

**Cause of job 5115bd05's gate failure, found (2026-10-14).** The diagnostic (job f27327c9, `diag10_v2mode_compare.csv`, `diag10.log`)
reproduced all 23 of v2's sample-level rows exactly and produced one extra: granulocytes against neutrophils, a sequencing-against-array pair
that v2 marks "not testable" and skips. Our copy carried a mode switch whose skip line was inverted, so in v2 mode it recorded the "not testable"
row and then computed the pair anyway. Fixed to skip every untestable pair in both modes, exactly as v2's script does (no new method); applied to
the diagnostic's own rows the fixed loop gives 23 rows identical to v2's record. The full job reruns with its gate unchanged.

**Decision 16 (author, 2026-10-14): the laboratory terms carry their uncertainty into the atlas.** In v2 each source's scale and offset
(a, b for sequencing, d for array laboratories, t the extra noise) entered the fit as fixed numbers, so their uncertainty never reached the
atlas's intervals. In v2.1 each block samples them, with a prior centered on the measured value (12b source terms) and as wide as its
bootstrap interval. Caveat stated in the methods: the prior was measured on all loci, so each block uses about 1/700 of its own data twice.
No next version is planned (a new one waits for new cells), so the change is made now. Gates before launch: a synthetic test with planted lab
terms (L17), and one real block fitted both ways (B4), recorded before the full fit. **Saved outputs widened** at the same time without
changing the model or the seeds (L18): 100 thinned draws of each cell's estimate (was 20), draws of the healthy spread, chain labels and
per-block sampler warnings. These serve 13 of the book's 79 cosmology tools (`atlas_tools79_needs`, development record).

**Decision 17 (2026-10-14, under the author's mandate: the best atlas and maps, using as many of the CMB's tools as hold up).** Besides the
main fit, the atlas is fitted twice more, each time on one of two sealed halves of every cell's donors: the CMB's half-mission split. Crossing
the halves gives map spectra with no noise bias, and half A minus half B is a pure noise map (the standard null). The split is sealed before
any fit (readiness L19); a cell with one donor can sit in only one half and is NOT MEASURED in the other. The halves run after the main fit
(about one more full fit of box time) so they do not delay it. The maps and their statistics (release checklist M01-M07) are built on the
fit's output; none is read on real data until the sphere construction passes its synthetic methylome and noise-map check (M01).

**Readiness L20, run once (job 30ad806b, rule committed first): the array-to-sequencing transfer is NOT a straight line.** On the 767,672
pairs of the 17 cells Loyfer and our arrays both measured (v2's own Deming line reproduced: a -0.090199, b 1.102649), 14 of the 42 bins with
at least 500 pairs miss the line by more than 0.02 (`transfer_shape_v21.csv`). Three regions: below x = 0.08 the line predicts negative
readings while 53-68% of real sequencing readings sit at or below 0.01 (residual up to +0.081); between x = 0.2 and 0.45 sequencing reads
lower than the line (about -0.04); above x = 0.97 sequencing levels off near 0.966 while the line keeps climbing (-0.03 to -0.06). The
shape is an S, not a line. This applies to atlas v2 as built: its Loyfer-scale estimates near both ends and in the 0.2-0.45 band carry
this error, now measured. By the rule, v2.1 does not launch on the straight line: a transfer that cannot leave [0, 1] is designed, then
tested on synthetic data, before launch. The form is chosen after seeing these residuals, on build (development) cells only; that is
disclosed, and the held-out arrays remain unopened.

**Decision 16, synthetic test run once (2026-10-14; `labterms_synthetic_v21.py`, bars and adoption rule committed first, amended only after
a smoke run): NOT ADOPTED. v2.1 keeps v2's fixed laboratory terms.** Both sampled arms met S1 (cell intervals covered 0.898 and 0.902
against 0.87-0.93; the sequencing-only cell 0.898 and 0.904) and S4 (no harm: median shift 0.0006-0.0011 against 0.002). Both missed S2
(the sequencing scale b covered in 15 of 20 replicates against at least 16) and S3 (up to 1.8% and 2.9% of cell-locus pairs at R-hat >= 1.01
against 1%; terms to 1.023 against 1.01). Reported, not a bar: the fixed arm, with the planted terms off its fixed values by their stated
uncertainty, still covered 0.899 (0.904 for the sequencing-only cell), so at their measured size the laboratory terms' uncertainty does
not measurably change the atlas's intervals. Readiness L17 was written as "bars met"; it now reads "settled by the committed rule", whose
third outcome (NONE) is this one. The release states that the terms' uncertainty is not carried, with this test as the measured reason.
The far larger error found tonight is the transfer's shape (L20, up to 0.08), which L21 addresses.

**Twin test and source terms, run 2 (job 178c64ed, 2026-10-14): reproduction gate passed, all four parts.** In v2 mode the job reproduced
v2's twin table, split-half null, sample-level twin test and source terms exactly, then computed v2.1 (`twins_v21.csv`,
`twin_sample_level_v21.csv`, `twin_v21.csv`, `source_terms_v21.json`). Profile level: 14 pairs above the twin threshold (v2: 7). Sample level,
2 twins: kidney tubular epithelium and podocyte (9 separating sites; held in v2 already) and **neutrophils and promyelocytes** (new in v2.1,
GSE49618; 7 separating sites). By v2's own rule a sample-level twin is HELD, so promyelocytes are held, as podocyte was. Source terms are
measured for every new source except GSE116754 (shares no cell with another array lab or with Loyfer: ASSUMED, offset 0, as v2 did for
GSE63409). Loyfer's terms with the v2.1 samples: a -0.0968, b 1.1130 (v2: -0.0902, 1.1026); this is still the straight-line transfer that
L20 found bends, and L21 replaces it before launch. Two offsets exceed the +/-0.01 seen so far: GSE118144 +0.011 (4 shared cells) and
GSE137716 -0.030 (via Loyfer, one shared cell); flagged for the L21 work, which re-derives every term on the bounded transfer.
In plain words: v2.1's new cells were checked for duplicates and each new laboratory's offset was measured. The check first reproduced v2
exactly. One new cell, promyelocytes, cannot be told apart from neutrophils sample by sample, so it is held, by the same rule v2 used.


**Readiness L21, run once (job 43e13497, rule committed first, 504ffb8a / 2980d106): the bounded transfer is ADOPTED.** A 3-parameter logistic in
logit space, tried first on L20's committed bins, missed 3 of 42 bins with a smooth wave and is not used. The adopted transfer is an empirical
monotone calibration curve (pool-adjacent-violators on L20's bins, linear between 42 knots; range 0.0089-0.9590, so it cannot leave [0, 1]).
It is instrument calibration, like v2's straight line, not physics. Synthetic test MET (recovery 0.0028; cross-halves 0.0020 / 0.0019). Real
check on L20's 767,672 pairs: built on 9 of the 17 shared cells and checked on the other 8, and the reverse: worst bin 0.0179 both ways, none
beyond +/-0.02. In the fit, g(m) replaces a + b m for Loyfer rows and its local slope replaces b in the variance; Tian2023 and ENCODE keep
their straight lines (L20 measured Loyfer only; recorded as a limit). Knots: `transfer_bounded_v21.json`.

**Readiness L8, the five open conditions, settled by rules committed before reading (`l8_conditions_v21.md`, f63d897d; box job 93d94090).**
- **D1, GSE224807 CD15+ as neutrophils: OUT (research).** Against the Salas neutrophils, r 0.9927 clears 0.985, but 198 sites separate
  every sample of one lab from every sample of the other, against fewer than 30. By v2's sample-level rule they are not one cell.
- **D7, GSE306226 astrocytes: OUT (research).** Their closest cell is Tian astrocytes, as required, but r 0.9575 is below 0.986.
- **D10, GSE146552 scraped epithelium: OUT (research).** EpiDISH 2.26.0 (RPC) reads a median 0.62 (ovary) and 0.59 (fallopian tube)
  non-epithelial share against the 0.20 cut; 0 of 8 and 1 of 7 arrays pass. Caveat: ovarian surface epithelium is mesothelial in origin
  and an epithelial reference may read part of it as fibroblast; the cut was fixed first, so the outcome stands.
- **D13, D14:** held-out only; carried to L15 as preconditions for scoring those two sets.
37 samples leave the build (781 remain: 401 in half A, 380 in half B). Ovary epithelium leaves v2.1; fallopian tube epithelium stays through
its other source. The sealed half split stands: the leaving samples drop out of their units, and no other unit changes half.

**Correction, same day.** The L8 record above omitted GSE137716 (lung bronchus epithelium, 12 conditional arrays), which decision 10
also covers. Its outcome is pending under the identical rule (addendum in `l8_conditions_v21.md`), and L8 is open again until it is read.

**D10, GSE137716, read once by the addendum rule (job 3cf4e084): IN.** 12 of 12 arrays under the 0.20 non-epithelial cut (median 0.000). L8 is settled.

**Readiness L10, planted test of the run script, run once (`planted_run_v21.py`, committed first): MET.** On a 40-locus, 5-cell synthetic
block laid out like the real sources: every output written with the right shapes; unmeasured pairs NaN; 95% interval coverage 0.947 and median
error 0.0115 on the planted means; v2 mode reproduces v2's own block code exactly (every summary column and all 20 draws); the 100-draw save
leaves every summary column identical to the 20-draw save. On the same curved truth v2's straight line misses by 0.0148 at planted values
below 0.1, against 0.0103 with the bounded transfer. R-hat p99 1.038 on this small block (no bar here; the fit's convergence bars are G0).

**Readiness L22, run once (job cb57b74b, rules committed first, fc501f01): NOT MET. The launch is stopped by the rule.** Twin recheck met
(3 cells changed, 7 pairs tested, no new twin). The bounded transfer, built on L20's 17 cells, was tested on the 5 Loyfer-shared cells of the
final table it had never seen (lung bronchus epithelium, oligodendrocytes, small intestine epithelium, t central memory cd4, t effector cell
cd8): 36 of 42 bins within 0.02, but between x 0.58 and 0.83 it reads 0.020-0.026 low. It passed cross-halves on v2's own cells at 0.018,
so cell-to-cell differences in the transfer were already close to the bar; new cells from new laboratories exceed it. Tian and ENCODE on
Loyfer's shape: not confirmed (worst 0.043, 0.039; their straight lines 0.106, 0.087). **Our design error:** the composed form alpha + beta g(m)
with a negative alpha predicts readings below 0 (to -0.03), breaking the bound L21 exists to keep; it should not have been committed. Next: a
diagnosis of the per-cell residuals (labeled as diagnosis), then a redesign committed before it is tested.

**Diagnosis after L22 (job e28be19b; nothing adopted; `transfer_diag_v21_cells.csv`).** Per cell, the mean sequencing reading against the
bounded transfer differs by cell, not by laboratory: removing the array laboratories' offsets moves no cell by more than about 0.008. The
spread is as wide among the cells L21 was built on as among the five it had not seen (x 0.45-0.85: cd8 t cells -0.090, lung alveolar
epithelium -0.035, pancreatic beta +0.052, naive cd8 t cells +0.047). Loyfer's donors and the array donors are different people, and
intermediate sites differ between people, so a per-cell residual mixes the instrument transfer with real between-cohort difference.
Pooled over many cells it averages out (L21 cross-halves 0.018 on 8-9 cells); over five it did not (0.026). Tian and ENCODE read against g
itself, without the alpha, beta layer, sit inside the same spread (cell means -0.012 to +0.003).

**Readiness L10, planted test run 2 (the model after the L22 second design; committed first): MET.** P1-P5 as before (coverage 0.959,
median error 0.0112; v2 mode still reproduces v2's code exactly). P6, a planted cell-level transfer bias of 0.03: on the cell seen only by
sequencing, 95% intervals held the truth in 0.963 of pairs with the scatter carried (bar 0.85), against 0.887 with it set to 0. Run 1's summary
is kept as `planted_run_v21_summary_run1.txt`.

**Readiness L22, second design, run once (job 4bb83f4e; run 1 6ea74736 failed before any bar): MET.** One bounded transfer on all 22
Loyfer-shared cells (978,886 pairs, 42 knots, range 0.0083-0.9589). Left one cell out at a time, the pooled bias is at most 0.0064 in every
bin (bar 0.02). Tian (microglia 0.014, oligodendrocytes 0.012) and ENCODE (embryonic stem cells 0.016) sit inside the Loyfer cells' own
leave-one-out spread (95th percentile 0.027) on the same curve, so all three sequencing sources use it with no extra terms. The cell-to-cell
scatter carried into the error bars: 0.004 below 0.15, 0.020 at 0.15-0.45, 0.039 at 0.45-0.85, 0.015 above 0.85. Every cell stays in.
Disclosed: the 22 cells had been seen in aggregate before this design was written; the test is leave-one-cell-out, not fresh data.

**Readiness L15, commissioning bars sealed before the atlas exists (`sealed_commission_v21.json`, scorers `commission_v21.py`).** U1: a held-out
sample's sites inside the 90 % posterior predictive interval at 0.80-0.97, every independent set, per cell and platform. U2a: the exact blind-site
lists the sealed cancer tests used, and U2b: the same rule on v2.1; each panel cell's held-out set mean within the test's tolerance of the epithelial
anchor at >= 0.90 of sites. U3: composition bars 1-2 unchanged. U4: development. Planted test of every scorer, run once: PASSED (a correct atlas reads
0.9261 coverage; intervals too narrow 0.7336, a shifted laboratory 0.5730 and too wide 1.0000 all FAILED as planted; a disagreeing panel cell 0.7172 FAILED).
**Original synthetic test (2026-10-14; superseded 2026-10-16 by the amended scorer's planted test below, kept as history):** DEV_ATLAS_V21_01/planted_commission_v21.py, DEV_ATLAS_V21_01/planted_commission_v21_summary.txt
**Prior reading:** the spent series named in this record (GSE103541, GSE118144, GSE122244, GSE167998, GSE226298, GSE271697) are BUILD sets of the atlas; none is a commissioning test set. The U1-U3 test sets are the plan's held-out assignments, unopened. GSE87197 (GSE87196's megakaryocytes) donors 1 and 2 are spent (the absorption test and the provisional column's test) and are read again to build the sequencing atlas's megakaryocyte column; donor 3 is unread. GSE223153 (outside healthy pancreas, sequencing) was read through the chain for the healthy test of Met-A on sequencing, before it is named here. GSE144691 (added 2026-11-05): its labels and read counts were read, and its 8 healthy libraries shared with GSE98837 are spent by the B and T test; its 5 other healthy libraries and its patients are unread.

## Amendment to the commissioning bars, sealed 2026-10-16 before the atlas exists

**Sealed 2026-10-16 by the author ("Seal the amendment").** The fit stood at 580 of 700 blocks; no held-out array had been opened. Stamped record:
`data/DEV_ATLAS_V21_01/amendment_v21b.json`. Scorer: chain Stage A21, `chain/stage_atlas_commission.py`; the original scorer
(`commission_v21.py`) and `sealed_commission_v21.json` are kept unchanged as history.

**The instrument's job:** a healthy cell from any laboratory reads inside the atlas's range for that cell, and a mixed blood sample splits into its cells correctly.

**Sources:** the held-out sets of atlas_v21_plan.md (U1, U2, U3 lists), all public on GEO or ArrayExpress: arrays with their laboratory, cell label and, for U3, FACS counts; sequencing GSE173787, public on GEO.

Why: the sealed bars passed or failed a whole cell on every set, and the U1 range left out the laboratory-to-laboratory shift the fit itself
measures, the trap DEV-SECONDLAB-02 fell into (lessons C19, KISS). The bars themselves are unchanged.

- **B1:** U1 reference profiles: at least 95 % of a cell's healthy held-out samples, all laboratories pooled, have coverage in 0.80-0.97; the array range includes the measured laboratory spread (SD of the fit's own array-laboratory offsets, 0.0065).
- **B2:** U2 blind sites: the epithelium is assessed and at least 95 % of the tissue's assessed panel cells reach share 0.90.
- **B3:** U3 composition: neutrophil mean absolute error at most 0.02 and at least 95 % of samples within 0.05.
- **Listed, not scored:** any sample or set the chain withholds (Stage 1 QC, the identity and sex check, the composition guard). Every set, cell and sample is still reported.

**Synthetic test:** DEV_ATLAS_V21_01/planted_commission_v21b.py, DEV_ATLAS_V21_01/planted_commission_v21b_summary.txt

**What it would take to succeed:** B1, B2 and B3 met by the real held-out sets once the atlas is released.

## U1 outcome (2026-10-16), scored once by the amended bar

Box job 85685c55 (code 77c31eb2): planted test PASSED first; then 1,319 held-out arrays scored, 159 listed (158 cells not in v2.1, 1 with no
array file). Results: `data/DEV_ATLAS_V21_01/u1_scored_v21.json`, `u1_samples_v21.csv` (stamped). Bar: a cell and platform pass when at least
95 % of its healthy held-out arrays have coverage 0.80-0.97 of the 90 % predictive interval.

**Commissioned on U1 (4 of 27 cell-platform pairs):** cortical neurons (450K), lung alveolar macrophages (EPIC), lung bronchus epithelium (450K),
naive CD4 T cells (450K). The other 23 did not reach 95 %. Median coverage over all 1,319 arrays is 0.891 against the interval's 0.90, so the
atlas's range is centered; the misses are a low tail (244 arrays below 0.80, 41 above 0.97), and 72 % of the low arrays come from three
laboratories (GSE59065, 113; GSE224807, 38; GSE247198, 24). These are reported, not reasons: the bar stands as sealed.

## U3 outcome (2026-10-16), scored once by the amended bar B3

Box job 50b9a3b9 (code 64bfaed4; run 1, 2fa39f71, crashed on its first sample on a column-name clash before anything was recorded). Planted
test PASSED first (planted blood: neutrophil error 0.001; a planted 0.06 bias caught). Tool: atlas_e built on the v2.1 means, 14 array-measured
blood cells (the mixture rule removed the three unsorted parent cells, b cells, cd4 t cells and cd8 t cells, as it did in v2). Truth rules fixed
before reading (score_u3_v21.py). Results: `data/DEV_ATLAS_V21_01/u3_scored_v21.json`, `u3_samples_v21.csv`.

| truth set | n | atlas v2.1 MAE | within 0.05 | current method (NNLS8) MAE | verdict |
|---|---|---|---|---|---|
| GSE127824, cord blood, flow counts | 24 | 0.126 (reads 0.124 low) | 0.17 | 0.093 | NOT MET |
| GSE167202, emergency-department bloods, clinical count | 195 | 0.084 | 0.40 | 0.080 | NOT MET |

**U3: FAILED.** The current commissioned composition step misses both sets almost as widely, so the misses are not particular to v2.1;
both sets differ from healthy adult blood (cord blood carries cord-blood granulocytes and nucleated red cells; these patients were acutely
ill). On adult bloods counted by flow, the same method read neutrophils within 0.016 (DEV-ATLAS-EPIC-01, development). Reported, not reasons:
the bar stands. The v2.1 composition step stays development.

## D13 and D14 recorded, and a correction to U1 (2026-10-16)

**Error found:** U1 scored GSE132547 and GSE131989 before their carried checks D13 and D14 were recorded, against the sealed rule
(sealed_commission_v21.json, carried_preconditions). Both checks are about the papers, not the arrays, and are recorded now by the plan's own wording
(atlas_v21_plan.md items 13 and 14):

- **D13, GSE132547 (alveolar macrophages): NOT MET.** The plan's condition: "the authors (Armstrong, Chen) are not in the Christensen/Salas group."
  The paper (Armstrong et al. 2019, ImmunoHorizons, PMID 31356157, doi 10.4049/immunohorizons.1900042) lists Brock C. Christensen as an author.
  GSE132547 is not used: its cell, lung alveolar macrophages, is NOT ASSESSED on U1 (no other held-out set).
- **D14, GSE131989 (sorted blood, rheumatoid arthritis study): MET.** Controls: GEO subjecttype 2 holds 30-31 arrays per sorted cell type and
  subjecttype 1 holds 61-63, matching the paper's 31 controls and 63 patients (Rhead et al. 2017, PMID 27723282); every held-out GSE131989
  array is subjecttype 2. Group: Barcellos (UC Berkeley) and Criswell (UCSF) with their co-authors; Wiencke is not among them.

**U1 corrected:** 3 of 26 assessed cell-platform pairs commissioned (cortical neurons 450K, lung bronchus epithelium 450K, naive CD4 T cells
450K); lung alveolar macrophages EPIC moves from COMMISSIONED to NOT ASSESSED. No number was recomputed and no bar changed; the GSE131989 pairs stand.

## U2a outcome (2026-10-16), scored once by the amended bar B2

Box job 8abbdbe9 (code 8737394c). Planted test PASSED first (arrays at the anchor: all four tissues pass; a 0.09 shift: all four fail). The
exact sealed blind sites of each tissue test; each panel as in its committed rule file; held-out arrays of each panel cell pooled; share of
blind sites within the test's tolerance of the v2.1 anchor. Result: `data/DEV_ATLAS_V21_01/u2_scored_v21.json`. GSE132547 was not in any panel.

| tissue | epithelium on held-out arrays | assessed panel cells (share) | verdict |
|---|---|---|---|
| colon | colon epithelium 0.999 (10 arrays) | neutrophils 0.971, T 0.963, B 0.981 | **COMMISSIONED** |
| lung | bronchus epithelium 0.980 (58) | neutrophils 0.960, monocytes 0.993, T 0.974, B 0.985 | **COMMISSIONED** |
| breast | none held out | T 0.836, B 0.914 | NOT ASSESSED |
| liver | none held out (no held-out hepatocytes) | neutrophils 0.461, monocytes 0.667, NK 0.636, T 0.573, B 0.694 | NOT ASSESSED |

Panel cells with no held-out set (fibroblasts, macrophages, smooth muscle, endothelium, adipocytes) are listed in the result file. The liver
blind sites (tolerance 0.03) do not hold for blood cells on outside arrays: reported here because the liver test's normals and tumors carry
blood, though its verdict is NOT ASSESSED without held-out hepatocytes.

## The IAM Sequencing Atlas, separate from v2.1 (plan 35.30)

**The author's decision:** the whole-genome atlas is built for sequenced samples only and kept apart from IAM-Atlas v2.1, which stays as
commissioned for arrays; an atlas that tries to do everything carries unforeseen consequences. **Its job:** a healthy sequenced sample reads
Normal on every instrument, against healthy cells of its own type measured from sequencing. **Its rule:** measure, not fit: every value is
counted from healthy donors' reads, and nothing is filled. Everything it carries is listed in `seq_atlas/SEQ_ATLAS_CHECKLIST.md`: each donor's
raw counts and measured disorder at every CpG, a healthy reference for every sequencing instrument, both spheres with their noise maps, the
CMB tools carried over (two noise splits, masks, spectra, cosmic variance, covariance, transfer function, nulls, cross-calibration, sealed
held-out sets) and, from healthy plasma only, fragment features.

**Scope (author):** build now from what is cheap: the 153 healthy sequencing inputs already read, plus open sources that add a new donor,
laboratory or cell type at little cost; every item of the checklist is built. The bigger atlas, with more donors and laboratories per cell,
comes through working with the Johns Hopkins group: their healthy sequenced data, help with controlled-access sets, and their computers.
**Cost, measured:** on a Google spot machine (n2-standard-4) one healthy input was read by the chain in 16.5 minutes for about one cent,
with readings identical to GitHub's; full-depth alignment from raw reads takes about 0.16 machine-hours per million read pairs.

## The IAM Sequencing Atlas, step 1: the counting tables (written and committed before the synthetic runs; plan 35.30)

**The instrument's job:** the atlas reports each cell's healthy level where donors were measured, leaves blank where they were not, and its
two noise maps carry noise only. **What is built:** `chain/stage_seq_atlas.py`, run by the one command (`run_sample.py --build seq-atlas
--build-samples CSV --build-block START:END --build-out PARQUET`). Each healthy donor's hg19 .beta file (methylated and total reads at every
one of the 28,217,448 CpGs) is read only over the block, from a folder or straight from GEO by byte range. Per cell and CpG: donors, reads,
the median of the donors' levels, the donors' spread, each donor's measured disorder (mean binary entropy), the 95 % range of the pooled level
from the counts (Jeffreys interval; exact, no sampling), the pooled level in each sealed donor half (IAM-Atlas v2.1 decision 17's split,
reused: all 153 inputs already hold a half, 79 A and 74 B) and in two random halves of every donor's reads (seed 20261029). Half A minus half
B is noise map 1; read half A minus read half B is noise map 2, defined for a cell with one donor. Blank (65535) where no donor of the cell
has a read; nothing is filled. Counts in the source .beta files are capped at 255; capped sites are counted in each block's summary.

**Prior reading:** none of the atlas's 153 sequencing inputs is read by this step's synthetic; they are already spent as atlas inputs.

**Synthetic test:** DEV_ATLAS_V21_01/seq_atlas/seq_atlas_synthetic.py, DEV_ATLAS_V21_01/seq_atlas/seq_atlas_synthetic.txt

**Synthetic bars** (both cells): 1. median |built level - planted donor level| at most 0.02; 2. both noise maps average within 0.005 of 0,
and the read-half map's spread is within 20 % of what counting predicts; 3. blank exactly where no reads, never where reads.
**Then:** one real block (CpGs 0 to 1,000,000) from the 153 on one Google spot machine, timed; then all 29 blocks.

**Step 1 synthetic, run 1: not met as written** (`seq_atlas/seq_atlas_synthetic_run1.txt`). Bars 2 and 3 met in both cells: both noise maps
average within 0.0001 of 0 and the read-half map's spread equals counting's prediction (0.0617 and 0.0616 against 0.0617); blanks fall
exactly where no reads. Bar 1 gave 0.0201 and 0.0202 against at most 0.02. The bar had not been derived from counting: a median of four donors
at about 20 reads scatters by about that much from counting alone. **Bar 1 amended, disclosed:** the level error must lie within 20 % of what
counting alone predicts (the same planted donors' reads drawn again independently), the form bar 2 already takes; run 2 uses a new seed
(20261030) so the amended bar is not checked on the data that prompted it. The builder is unchanged.

**Step 1 synthetic, run 2: met** (`seq_atlas/seq_atlas_synthetic.txt`, seed 20261030). Both cells: the level error equals counting's prediction
(0.0201 against 0.0201 and 0.0202); both noise maps average within 0.0002 of 0, and the read-half map's spread matches counting (0.0617
against 0.0617; 0.0614 against 0.0616); blanks fall exactly where no reads. **The real build's inputs:** `seq_atlas/seq_atlas_inputs.csv`,
the 153 healthy sequencing inputs with each one's GEO .beta file, cell and sealed half (IAM-Atlas v2.1 decision 17).

**Block 0, measured** (development; CpGs 0 to 1,000,000; chain 62b5a6e): built by the one command from the 153 inputs read from GEO by byte
range, on one Google spot machine (n2-standard-4, northamerica-northeast2): 203 s to build, 146 s to set up, 758 MB written, 7,493 donor-sites
at the source files' 255-read cap. The whole numbering (29 blocks) is about 3 machine-hours and about 22 GB. **Change before the other
blocks:** the two range columns (lo, hi) are replaced by the pooled methylated-read count (`meth`); with `reads`, the pooled level and its
exact 95 % range follow from the counts (Jeffreys interval), so nothing is lost and each block is smaller. The synthetic was run again on the
changed stage (seed 20261030): all bars met (`seq_atlas/seq_atlas_synthetic.txt`; the previous run kept as `seq_atlas_synthetic_run2.txt`).
Block 0 is built again with the other blocks.

**Step 1, the whole numbering, built** (development, not commissioned; chain 1cc8064). Every one of the 28,217,448 hg19 CpGs, from the 153
healthy inputs (56 cells), built by the one command in 113 pieces of up to 250,000 CpGs on 8 Google spot machines (n2-standard-4,
northamerica-northeast2), each piece uploaded the moment it was built (`seq_atlas/gcp_seqatlas.py`, launcher `seq_atlas/gcp_launch.py`). The
pieces cover the numbering exactly once, with no gaps or overlaps. 17.49 GB; 1.9 machine-hours (about 6 cents); about 15 minutes from launch to
the last machine's self-delete. No machine was reclaimed (a label in the first version of the shutdown script called every self-delete a
reclaim; corrected). Each cell has reads at 97.3 % to 98.4 % of all CpGs (`seq_atlas/seq_atlas_cell_coverage.csv`, with its donors; totals in `seq_atlas/seq_atlas_build_summary.json`); 605,451
donor-sites sit at the source files' 255-read cap and are counted. Three cells rest on one donor (naive CD4 T, naive CD8 T, effector memory CD8 T;
9 on two, 44 on three); for those three the donor-half noise map is blank and the
read-half map stands in. Stored in the Google bucket (northamerica-northeast2), seq_atlas/pieces/. **Next:** the healthy references for the
sequencing instruments, the spheres, spectra and masks, each with its synthetic first.

## The IAM Sequencing Atlas: a megakaryocyte column (written and committed before any of it runs; plan 35.30 and the provisional megakaryocyte column's branch)

**Why:** the 153 inputs hold no megakaryocytes, a large source of plasma DNA. The only open healthy megakaryocyte sequencing is GSE87197,
whose parent series GSE87196 (blood progenitors) holds the same three donors: 33, 19 and 19 single megakaryocytes. Donors 1 and 2 are spent
(the absorption test and the provisional column's test); donor 3 is unread and stays reserved. **Rule set here:** no donor-3 sample of any
progenitor type in GSE87196 goes into the atlas, so donor 3 is unseen when it is tested.

**The conversion (new in the chain):** these files are GRCh38 per-CpG read tables. `stage_seq_atlas.lift_to_hg19`, run by the one command
(`run_sample.py --build seq-lift`), pools one donor's cells on GRCh38, moves each distinct position to hg19 (UCSC hg38ToHg19 chain,
pyliftover; a position that does not map once is dropped and counted), and adds it to the hg19 CpG whose C or G it is (both strands to one
CpG), against the pipeline's own hg19 CpG dictionary (wgbstools init_genome, from the committed index release, 28,217,448 checked). It writes
an .lbeta (uint16 counts, no 255 cap), which the atlas build now reads beside .beta files.

**Synthetic test:** DEV_ATLAS_V21_01/seq_atlas/seq_lift_synthetic.py, DEV_ATLAS_V21_01/seq_atlas/seq_lift_synthetic.txt

**Synthetic bars** (known answer: the spent atlas input GSM5652316, moved to GRCh38 as test construction with 30 % of sites on the reverse
strand, then moved back by the one command): 1. at least 95 % of its CpGs with reads come back; 2. of those, at least 99 % with exactly the
original counts; 3. at most 0.1 % of the CpGs with reads afterwards had none before.

**Then, if met:** donors 1 and 2 converted, one .lbeta each (`seq_atlas/mk_build_files.csv`: 33 and 19 cells; each conversion's report
records how many positions moved and landed on hg19 CpGs), and the megakaryocyte column built by `--build seq-atlas` from the two (donor 1
in half A, donor 2 in half B), stored beside the 56 cells. **Its test on donor 3 is written and committed before donor 3 is read.**

**The GRCh38 round trip, read: met** (`seq_atlas/seq_lift_synthetic.txt`; one Google spot machine, the hg19 dictionary taken from the
committed index release with its stream checksum verified, 28,217,448 CpGs). Of the known answer's 27,446,949 CpGs with reads (93,566 did
not move to GRCh38 once and were left out of the construction), a share of 0.9959 came back (bar at least 0.95), 0.99957 of those with exactly the
original counts (bar at least 0.99), and 0.00024 of the CpGs with reads afterwards had none before (bar at most 0.001), with 30 % of sites
written on the reverse strand. **Donor 1 converted** (`seq_atlas/mk_donor1_lift.json`): 33 megakaryocytes, 23,871,358 GRCh38 positions,
23,484,305 moved to hg19, 23,453,776 landed on an hg19 CpG, 23,428,509 hg19 CpGs with reads.

**Donor 2 converted** (`seq_atlas/mk_donor2_lift.json`): 19 megakaryocytes, 19,246,527 GRCh38 positions, 18,950,618 moved, 18,925,280 landed,
18,909,575 hg19 CpGs with reads. The megakaryocyte column is built from donors 1 and 2 (donor 1 in half A, donor 2 in half B) by the one
command, in pieces stored beside the 56 cells' (seq_atlas/pieces_mk/).

**The megakaryocyte column's test on donor 3** (written and committed before any donor-3 file is read). **The instrument's job:** donor 3's
megakaryocytes read as megakaryocytes against the IAM Sequencing Atlas. **The reading:** donor 3's 19 megakaryocytes (`seq_atlas/mk_test_files.csv`)
are converted by the one command (`--build seq-lift`, the step whose synthetic met) and compared with every cell of the atlas by the one
command (`--build seq-compare`, new in the chain: at the CpGs where the sample has at least 5 reads and every one of the 57 cells has a
level, the mean distance between the sample's level and each cell's, and, for each cell, the mean distance between its two donor halves at
the same CpGs).

**Synthetic test:** DEV_ATLAS_V21_01/seq_atlas/seq_compare_synthetic.py, DEV_ATLAS_V21_01/seq_atlas/seq_compare_synthetic.txt

**Synthetic bars** (a constructed mini-atlas of three cells, and a new sparse donor of each): 1. each new donor's nearest cell is its own;
2. the comparison counts exactly the CpGs it should.

**Bars on donor 3:** 1. megakaryocytes is donor 3's nearest cell of the 57; 2. donor 3 sits from the megakaryocyte column at most 1.5 times as
far as the column's own two donors sit from each other at the same CpGs. **Then:** the column in plasma (planted megakaryocyte DNA picked up,
and the four cells that absorbed it stop rising) once the Deconvolver reads against the sequencing atlas.

**The identity comparison's synthetic, read: met** (`seq_atlas/seq_compare_synthetic.txt`): each new sparse donor's nearest cell was its own,
at 0.0780, 0.0780 and 0.0781 against 0.2070 or more for the others, and the comparison counted exactly the CpGs it should. **Donor 3's bar 2
withdrawn before any donor-3 file is read (disclosed):** the synthetic shows a sparser sample sits farther from its own cell than that cell's
two halves sit from each other from counting alone (0.0780 against 0.04987), so a bar on that ratio was not derived from counting and would
judge read depth, not the cell. Both distances are still reported. The test is bar 1: megakaryocytes is donor 3's nearest cell of the 57.


## The IAM Sequencing Atlas: each cell's held error, region by region (written and committed before it runs)

**Why (author):** IAM-A on plasma, after the Deconvolver has named what is in it. Each plasma fragment comes from one cell, so a plasma's
healthy held error is each cell's healthy held error weighted by its share; read per region, a tissue whose shed DNA holds its marks worse
can be seen on its own. **The instrument's job:** each healthy cell's held error is recovered region by region, as counting predicts.

**Build:** each of the 153 healthy sequencing inputs' molecule files (.pat, on GEO) counted by the one command (`run_sample.py --dev-count-only
--dev-count-tiles`, Stage Q2's own rules), then pooled by cell (`--build seq-held`), stored as counts with both noise splits (sealed donor
halves; halves of every donor's molecules), beside the atlas. Megakaryocytes have no molecule files (single-cell per-site tables only) and get
no column here; their share is still named by the Deconvolver.

**Synthetic test:** DEV_ATLAS_V21_01/seq_atlas/seq_held_synthetic.py, DEV_ATLAS_V21_01/seq_atlas/seq_held_synthetic.txt

**Synthetic bars** (two constructed cells, four donors each, planted held error per tile; counting's prediction by exact sum over every
molecule pattern under Stage Q2's rules): 1. median relative difference over a cell's tiles within 2 %; 2. at least 90 % of tiles within
2 counting standard deviations; 3. the halves add up exactly.\n

**The held-error synthetic, run 1: not met as written** (`seq_atlas/seq_held_synthetic_run1.txt`). Cell one met every bar (median relative
difference -0.0002); cell two met bars 2 and 3 but its median relative difference was -0.0201 against a bar of 2 %. That 2 % was not derived
from counting: the median of 40 tiles at these chances scatters by about 0.009 from counting alone, so -0.0201 is about two of its standard
deviations. **Bar 1 amended for run 2, before it runs (disclosed):** within 3 of the median's own counting standard deviations, computed from
each tile's chances; new seed 20261102. Bars 2 and 3 unchanged. The same lesson as the atlas's own synthetic run 1: a bar is derived from
counting before it is written.

**The held-error synthetic, run 2: met** (`seq_atlas/seq_held_synthetic.txt`): median relative differences -0.0116 and +0.0032 (limits
0.0295 and 0.0292), 0.975 of tiles within 2 counting standard deviations in both cells, and the halves add up exactly. **Then:** the 153
healthy inputs' molecule files counted by the one command on spot machines, and the reference built beside the atlas.

**The megakaryocyte column's test on donor 3, read: not met** (`seq_atlas/mk_donor3_compare.json`, `seq_atlas/mk_donor3_lift.json`). Donor 3's
19 megakaryocytes, converted by the one command (17,263,359 hg19 CpGs with reads) and compared at 582,215 CpGs, read nearest monocytes
(0.10776), then lung interstitial macrophages (0.10823); megakaryocytes ranked 10 of 57 (0.12175). Bar 1 not met. Two earlier attempts on the
spot machine read no atlas pieces (the job looked in the wrong folder; disclosed) and gave no comparison; this is the test's only result.
**What it shows:** the megakaryocyte column's own two donors sit 0.14419 apart at the same CpGs, against 0.05496 to 0.0748 for the other cells
with two halves: a column pooled from sparse single-cell reads carries far more counting scatter than the deep columns, and the comparison as
written treats every column as exact, so a noisy column sits far from everything, its own cell included. The synthetic gave every column the
same depth and could not see this. **Next:** a comparison that weighs each CpG by the counting scatter of both the sample and the column,
synthetic first with columns of unequal depth; donor 3 is spent, so reading it again with that comparison is development, not a test.

**The depth-weighted comparison** (written and committed before it runs). The comparison (`stage_seq_atlas.compare`, the one command's
`--build seq-compare`) now also gives each cell a depth-weighted score: at each compared CpG the squared difference between the sample's level
and the column's, over the scatter counting gives it (the sample's p(1 - p)/n, the column's p(1 - p)/reads, and the column's donors' spread
squared), averaged; it is about 1 when the sample is a healthy cell of that column. Cells are ranked by it (`weighted_ranked`) beside the plain
distance.

**Synthetic test:** DEV_ATLAS_V21_01/seq_atlas/seq_compare_synthetic2.py, DEV_ATLAS_V21_01/seq_atlas/seq_compare_synthetic2.txt

**Synthetic bars** (three deep cells and one shallow cell pooled from sparse reads, as the megakaryocyte column is): 1. by the depth-weighted
score each new donor's nearest cell is its own, the shallow cell's included; 2. the comparison counts exactly the CpGs it should.

**The comparison's synthetic 2, read: met** (`seq_atlas/seq_compare_synthetic2.txt`): by the depth-weighted score every new donor read nearest
its own cell, the shallow cell's included (own 0.945 to 0.949 for the deep cells, 4.465 for the shallow one, 27.255 or more for every other
cell). **What it does not show:** here the plain distance also put the shallow cell's donor nearest its own column (0.1314 against 0.2074),
so the synthetic does not reproduce donor 3's result, and a column's depth may not be the whole reason donor 3 read nearest monocytes.
**Next (development, donor 3 is spent):** donor 3 compared again with the depth-weighted score, reported, not scored.

**Donor 3 read again with the depth-weighted score (development, spent; reported, not scored)** (`seq_atlas/mk_donor3_compare_weighted.json`).
Megakaryocytes ranked 24 of 57 (3.8328); nearest were colon fibroblasts (2.002) and erythrocyte progenitors (2.0626). No cell reads near 1,
the value a healthy cell of that column gives, so donor 3 does not fit any column as counting predicts, megakaryocytes included. Neither the
plain distance nor the weighted score names donor 3's cell, so depth is not the explanation. The single-cell megakaryocyte libraries may carry
a protocol offset against the bulk sequencing behind the other columns, or the column may hold too few reads at the CpGs compared. **Next
(development, spent data):** whether the three megakaryocyte donors read nearer each other than any bulk column (donor 1 against donor 2 and
donor 3, each converted file compared directly), which separates a protocol offset from a weak column.

**The megakaryocyte donor-pair check** (written and committed before it runs; development, all three donors spent). **The question:** do the
three megakaryocyte donors read nearer each other than any bulk column, or do the single-cell pools carry a protocol offset of their own?
**The instrument** (`stage_seq_atlas.compare_pair`, the one command's `--build seq-pair`): two samples' summed squared level differences over
the squared difference counting predicts, at the CpGs where both have at least 5 reads; about 1 for the same cell read twice.
**Synthetic test:** DEV_ATLAS_V21_01/seq_atlas/seq_pair_synthetic.py, DEV_ATLAS_V21_01/seq_atlas/seq_pair_synthetic.txt
**Synthetic bars:** same cell read twice within 0.9 to 1.1; two cells differing at a tenth of sites at least 1.5. A first version averaged a
ratio at each site, which reads near 0 wherever both agree completely, as sparse reads near 0 or 1 nearly always do; it was changed to a ratio
of sums before any real sample was compared (disclosed). **Then:** donors 1, 2 and 3 compared pairwise; each donor compared with the two bulk
cells nearest donor 3 (monocytes and the next). Reported, not scored: if the donors read near each other (near 1 or below the bulk pairs),
the single-cell pools agree with each other and the offset is the protocol's, so the column needs a bulk megakaryocyte source; if not, the
pools themselves are too noisy at this depth.

**The pair comparison's synthetic: met** (`seq_atlas/seq_pair_synthetic.txt`). **The megakaryocyte donor-pair check, read (reported, not
scored)** (`seq_atlas/mk_pairs.csv`): the three donors against each other score 1.2826, 1.3001 and 1.3401; each donor against a monocyte
input 1.8726 to 2.0205 and against a lung interstitial macrophage input 1.9636 to 2.1122; two healthy monocyte donors against each other
1.9625 (over far more sites, where real differences between people show). By plain distance every pair reads alike (0.11 to 0.12), because
at a few reads a site counting scatter is most of the distance. **What it shows:** the three megakaryocyte donors agree with each other more
than with the cells donor 3 was named as, so the single-cell pools carry a shared megakaryocyte signal. **And a fault found:** the
comparison's depth-weighted score averaged a ratio at each site, the flaw the pair synthetic first exposed; it now sums both sides before
dividing. Synthetic 2 met again (`seq_atlas/seq_compare_synthetic2.txt`; the first run kept as `seq_compare_synthetic2_run1.txt`). Donor 3 was then read again with the corrected score (below); the earlier weighted re-read had used the faulty one.

**Donor 3 read again with the corrected depth-weighted score (development, spent; reported, not scored)**
(`seq_atlas/mk_donor3_compare_weighted2.json`, the one command's `--build seq-compare`, 582,215 sites): nearest megakaryocytes at 0.7069, 1st of
57; next colon fibroblasts 1.4571 and NK cells 1.4633. By plain distance the nearest is still monocytes. The committed test (nearest cell by
plain distance, on a donor unread when it was written) stays **not met**; this reading is development on a spent donor and cannot replace it.
**What it shows:** the column holds donor 3's cell once counting scatter is weighed correctly; plain distance is dominated by that scatter in
single-cell pools. **A fresh test** needs megakaryocyte sequencing the column has never seen, written before it is read.

**The held-error reference by region, built** (`seq_atlas/held_reference_v1.json`; the one command's `--build seq-held` on a Google spot
machine; in the chain as `chain/Runtime Matrices/Seq_Atlas/held_reference_v1.parquet`): all 153 healthy inputs counted, 14,133 regions of
2,000 CpGs, 56 cells, stored as counts with both noise splits (sealed donor halves and molecule halves). Each cell's whole-genome held error
ranges from 0.03327 (naive CD8 T cells) to 0.05487 (smooth muscle). **Next:** IAM-A on plasma, the plasma's expected held error built from
these columns and the Deconvolver's shares, synthetic first.

## The IAM-Deconvolver on sequencing, against the IAM Sequencing Atlas (development)

**Why:** on sequencing the Deconvolver read every sorted sample on the six-cell blood panel (a sorted pancreatic delta sample read 52 % CD4 T
cells), so every reading that uses the shares (Met-A for each cell and the whole sample, the C-Score for each cell) was wrong on sequencing.
**The instrument's job:** a mix of the atlas's own cells reads its planted shares. **The reference** (`stage_seq_atlas.deconv_reference`,
`run_sample.py --build seq-deconv-ref`; `seq_atlas/deconv_reference_v1.json`): each of the 56 cells' counted level and donor spread (at least
0.01) at the 813,866 array sites where every cell has a level; marker sites chosen on the sequencing atlas by the Deconvolver's own rule,
5,599 kept; in the chain (`Runtime Matrices/Seq_Atlas/deconv_reference_v1.parquet`). A sequenced sample read with `--dev-seq-ref` is split on
it, flagged. **Synthetic** (`seq_atlas/deconv_seq_synthetic.py`): 9 planted mixes at 20 reads per marker site, through the one command; bars
every planted share within 0.05, cells not planted at most 0.05.

**Run 1: not met** (`seq_atlas/deconv_seq_synthetic_run1.txt`; disclosed: run as a check of the script before this record was committed).
8 of 9 mixes met; the largest planted-share error was 0.0447. Mix 7 (tongue epithelium 0.60, effector memory CD4 T cells 0.30, gastric antrum
0.10) gave 0.0819 to cells not planted, spread over the planted cells' closest relatives (gastric fundus, tonsil, effector CD8 T cells), two
of them flagged by the chain as not resolved from their relatives. Even pure cells gave 0.006 to 0.045 to other cells at 20 reads per site.
**What it means for the purity check:** a sorted sample's share outside its own cell is read against this floor at its own depth, never
read as impurity on its own. **Next:** the same planted mixes at several depths, to measure that floor by depth, written before it runs.

**The share floor by depth, measured** (`seq_atlas/deconv_seq_floor.py`, `seq_atlas/deconv_seq_floor.txt`; written before it ran, measured,
not scored): a pure planted cell gives other cells, from counting alone, a median of 0.0152 at 5 reads per marker site, 0.0228 at 10, 0.0113
at 20 and 0.0112 at 40 (largest 0.0458). The floor hardly moves with depth; a healthy sorted sample whose share outside its own cell is well
above about 0.05 holds other cells, not counting scatter.

## The in-mix readings on the IAM Sequencing Atlas (written and committed before the synthetic runs; development)

**Why:** the healthy full-chain pass (DEV_METAA_ATLAS_CURVE_01, seq_atlas_01/healthy_fullchain_summary.txt) found Met-A for the whole sample and the territories not
reading Normal on healthy sequencing; both still mixed IAM-Atlas v2.1's array levels, put in sequencing units. **The change**
(`stage_seq_atlas.array_columns`, `run_sample.py --build seq-array-cols`): each sequencing atlas cell's counted level and donor spread at
every site of IAM-Atlas v2.1, in v2.1's own layout (`Runtime Matrices/Seq_Atlas/array_sites/`, built on one Google spot machine from the 113
atlas pieces). For a sequenced sample read with `--dev-seq-ref`, the chain now splits it on the sequencing atlas and mixes the sequencing
atlas's levels for every in-mix reading (Met-A for each cell and the whole sample, the territories, the C-Score); the bundle says which
atlas each part used (`sequencing_ruler`). Met-A's identity sites and the territory sets are still chosen on IAM-Atlas v2.1, so each
instrument keeps one set of sites. **Synthetic** (`seq_atlas/inmix_seq_synthetic.py`): 6 constructed healthy samples (pure, two-cell and
three-cell mixes; 20 and 8 reads per site), through the one command with the full chain; bars Met-A for the whole sample, islands and
domains each within 0.05 of 1 in every sample. **Then:** the healthy full-chain pass read again with these readings.

**The in-mix synthetic, read: met** (`seq_atlas/inmix_seq_synthetic.txt`): all 6 constructed healthy samples read Met-A for the whole
sample, islands and domains between 0.970 and 1.023, at 20 and at 8 reads per site, with the shares read within 0.03 of planted and the
bundle naming the IAM Sequencing Atlas for the in-mix levels. **Next:** the healthy full-chain pass read again (`metaa_hfc2`, same 137, same
scorer). Those samples are the sequencing atlas's own inputs, and the array-site columns hold both donor halves, so for the in-mix readings
this is a consistency check; the proof is healthy sequenced samples the atlas never held.

## The sky, NILC, the template split and B cells moved onto the atlases the repository holds (written before the synthetic runs; development)

**Why:** the healthy full-chain pass found these four never run: they read IAM-Atlas v2's single file, which the repository does not hold.
**The change** (`dev_stages.atlas_frame`, `dev_stages.sky_in_mix`): they now read the atlas the chain is using: IAM-Atlas v2.1 for an array
(its level, its interval as the uncertainty, its donor spread), or the IAM Sequencing Atlas at v2.1's sites for a sequenced sample (counted
levels, donor spread). The sky, when no v2 file is given, reads each site's departure from the sample's own mix of atlas cells (the
Deconvolver's shares), with the array's 0.02 noise term on arrays and the counting variance at each site's own reads on sequencing; its
block-shuffle null and look-elsewhere statistic are unchanged. NILC, the template split and B cells need named blood cells (neutrophils,
eosinophils, basophils, naive B cells, regulatory and memory CD4 T cells) that the sequencing atlas does not hold, so on a sequenced sample
the chain records that reason; on arrays they run on v2.1. The full chain turns all four on. **Synthetic** (`seq_atlas/sky_seq_synthetic.py`):
the 6 constructed healthy samples of the in-mix synthetic and one with chromosome 8 moved 0.15 toward full methylation, through the one
command with the full chain; bars: at most 1 of the 6 healthy shows structure beyond the null, and the planted sample does.

**The sky synthetic, runs 1 and 2: not met** (`seq_atlas/sky_seq_synthetic_run1.txt`, `seq_atlas/sky_seq_synthetic.txt`). Run 1 read no sky:
on the laptop the sphere library could not write its settings file in the home folder (sandbox); run 2 pointed it at a temporary folder.
NILC and the template split recorded, as intended, the blood cells the sequencing atlas does not hold; B cells recorded no stage 2
composition. Run 2: (1) 3 of the 6 healthy samples showed structure beyond the null (p 0.0099), all three of them mixes, neither pure cell;
(2) the planted sample did not (p 0.8218). Bar 2 was badly designed: the null shuffles blocks within each chromosome, so a change spread
over a whole chromosome is inside the null by construction. **Next:** a diagnostic through the one command (`seq_atlas/sky_seq_diag.py`; sample 3's mix, 8 reads): the sky with
the Deconvolver's own shares, with the planted shares declared (`--dev-shares`, a new development option for diagnosis: every share-based
reading uses the declared shares, the Deconvolver's own kept beside them), and with a regional plant (300 sites of chromosome 8).

**The sky diagnostic, run 1** (`seq_atlas/sky_seq_diag_run1.txt`, through the one command): with the Deconvolver's own shares the sample
showed structure (p 0.0099, one band unreadable); with the planted shares declared it did not (p 0.5644); the 300-site plant was not seen
(p 1.0). **Found:** the sky dropped every site where any cell given a share, however small, has no level (here monocytes, bladder and breast
at 0.5 to 1.4 %), leaving holes in the map. **Fixed:** each site now mixes the cells with a level there and is kept where they hold at least
0.99 of the shares (the territories' rule, MIX_COVER). A 300-site change is below what the sky sees at 8 reads; the plant is now 3,000 sites.
**Run 2** reads the same three ways.

**The sky diagnostic, run 2** (`seq_atlas/sky_seq_diag_run2.txt`, through the one command, the hole fix in): with the Deconvolver's own
shares no structure (p 1.0, every band read); planted shares declared, none (p 0.5644); 3,000 sites of chromosome 8 moved 0.15 at 8 reads
raised one band to 1.3674 times its null, not significant (p 0.1089): a measured limit of what the sky sees at that depth.
**Synthetic run 3** (amended before it ran, disclosed; `seq_atlas/sky_seq_synthetic.py`): the bar is the instrument's job only, at most 1 of
the 6 healthy samples showing structure; what the sky can see is measured, not scored, at 20 reads with 3,000 and 10,000 changed sites.

**The sky synthetic, run 3: met** (`seq_atlas/sky_seq_synthetic.txt`): none of the 6 constructed healthy samples showed structure beyond
the null (p 0.1386 to 0.9307), pure cells and mixes, at 20 and 8 reads, against the sample's own mix of the IAM Sequencing Atlas. **What
it sees, measured:** 3,000 and 10,000 contiguous sites of chromosome 8 moved 0.15 at 20 reads were not seen (p 0.9901 and 0.5743). This is
by construction: the null shuffles runs of 50 sites along each chromosome, so a moved region keeps its clustering and its band power; the
sky reads how departures are arranged beyond blocks, not regional level shifts, which the territories and the C-Score read. NILC, the
template split and B cells record, on sequencing, the blood cells the sequencing atlas does not hold.

## IAM-A in the sample's mix (written and committed before the synthetic runs; development)

**Why:** IAM-A reads a sorted cell against its own loose-share line; a mixed sample (plasma, tissue, whole blood) has no single cell, so it
was read on the self-reference line with the shape check off. **The reading** (`stage_seq_atlas.iama_in_mix`): in each tile of 2,000 CpGs
the healthy expectation is the sample's own held-error opportunities times its mix of the cells' healthy held-error rates, counted in the
IAM Sequencing Atlas's held-error reference (153 healthy inputs, by tile, overall and by sealed donor half); A is the sample's held errors
over the expected, so healthy reads 1. Tiles are read where every cell given a share has at least 50 opportunities and those cells hold at
least 0.99 of the shares. The rates are counted and the shares are the Deconvolver's: nothing is fitted. The one command reads it for any
molecule file read with `--dev-seq-ref` (`iam_a_mix` in the bundle), flagged. **Synthetic** (`seq_atlas/iama_mix_synthetic.py`): molecule
files built from the reference's own rates (b cells alone; a plasma-like mix; the mix against donor half A; the mix with every rate raised
20 %), shares declared, through the one command; bars: the first three within 0.02 of 1, the fourth within 0.03 of 1.2.

**IAM-A in the mix, synthetic run 1: not met** (`seq_atlas/iama_mix_synthetic_run1.txt`): b cells alone 0.99796, the plasma-like mix
0.99868, the mix with every rate raised 20 % 1.18894 (all met); the mix against donor half A was not read: half A holds no hepatocyte
donor, so no tile held every planted cell, and the reading said so. That was my test's error (the reference shows it). **Run 2** (amended
before it ran, new seed): the same four with the third against donor half B, which holds all five cells.

**IAM-A in the mix, synthetic run 2: met** (`seq_atlas/iama_mix_synthetic.txt`): b cells alone 0.9927, the plasma-like mix 0.99126, the
mix against donor half B 0.99343 (all within 0.02 of 1), every rate raised 20 % 1.1898 (within 0.03 of 1.2), about 14,000 tiles each.
**Next:** healthy sequenced inputs read against the other donor half, through the one command.

## The IAM Sequencing Atlas on the sky sphere: each cell's level and its 95 % range (written and committed before it runs)

**Why now:** the chain's readings on sequencing read healthy sequenced cells Normal (Met-A in each cell and the whole sample, the
territories, the C-Score, coupling-A, IAM-A), so the atlas's own checklist comes next, the spheres first (`seq_atlas/SEQ_ATLAS_CHECKLIST.md`,
C1 and D4). **The instrument's job:** each cell's map shows its healthy level at every site the atlas places, beside a map of how sure that
level is. **Derived, nothing fitted** (`stage_seq_atlas.range_of_average`, `sphere_columns`; `run_sample.py --build seq-sphere-cols`): the
level is the pooled count (methylated over all reads, as the readings use); the 95 % range of the cell's average is the donors' spread over
the square root of their number, times the t quantile for that many donors (two donors give a wide range, honestly); one donor, or a site
where the spread is blank, gives counting only (stated: it does not hold the person-to-person part). **Synthetic:**
`seq_atlas/sphere_range_synthetic.py`; bars for two and three donors: the stated variance within 0.9 to 1.1 of the true scatter, and the
range covering the truth 90 % to 99 % of the time. **Then:** the columns built, and the sphere drawn by the same builder as IAM-Atlas v2.1's
(`--build seq-atlas-sphere`), named as the sequencing atlas and without v2.1's cell-page check.

**The range synthetic: met; the columns built** (`seq_atlas/sphere_range_synthetic.txt`). Two donors: stated variance 1.005 of the true
scatter, coverage 0.948; three donors: 1.007 and 0.948. One donor, counting only (recorded, not scored): 0.613 and 0.851, as stated, so a
one-donor cell's map shows less doubt than it carries. Built by the one command (`--build seq-sphere-cols`;
`chain/Runtime Matrices/Seq_Atlas/sphere_cols/sphere_cols.json`): 56 cells, three of them one-donor (effector memory CD8 T cells, naive
CD4 T cells, naive CD8 T cells). The sphere is drawn next on a Google spot machine with the chain's own requirements.

**The IAM Sequencing Atlas on the sky sphere: drawn** (`run_sample.py --build seq-atlas-sphere` on a Google spot machine; 56 cells, the same
builder, layout and color scales as IAM-Atlas v2.1's sphere; 813951 sites, 1 unplaced). The level maps show the same large-scale
structure as v2.1's. The uncertainty maps (`seq_atlas/sphere_widths.py`, `seq_atlas/sphere_widths.txt`) show the ruler's depth honestly:
the median half-width of the 95 % range is 0.2056 for the 9 two-donor cells, past the 0.15 scale at 0.590 of their sites, and 0.0783 for
the 44 three-donor cells (0.265 past it); the 3 one-donor cells (0.0524) carry counting only. The scale is kept at v2.1's so the two atlases
read alike: the bright areas are where more healthy donors are needed.

**The IAM Sequencing Atlas's information globe (written and committed before it runs).** The same build as IAM-Atlas v2.1's
(`run_sample.py --build info-globe`, construction v5, the same site rule: measured in every cell, spread across cells above 0.08), now
given the atlas folder (`--build-input`, the sphere columns) and its name (`--build-name`). Its layout is written beside the picture, so the
chain's committed v2.1 layout is never replaced. **The job:** sites group by the cell family whose identity they carry, as on v2.1's globe.
Recorded beside it, not scored: the sites placed, and how clearly each family gathers (the concentration the build reports), against v2.1's.

**The information globe: built** (`seq_atlas/globe/`; `seq_atlas/globe_compare.py`, `seq_atlas/globe_compare.txt`; one Google spot
machine, both globes by the same command). The IAM Sequencing Atlas places 534617 sites (142527 clear of any second cell), IAM-Atlas
v2.1 191458 (41241), on the same rule: v2.1 has 79 cells, so more sites fall out where one of them is unmeasured. Each family gathers
about as tightly on the sequencing globe as on v2.1's (median concentration 0.42 against 0.35); embryonic stem cells are not in the
sequencing atlas. Recorded, not scored. Part of the extra sites may be the thin columns' own noise passing the spread rule (two-donor cells);
that is read when more donors are in.

**The two noise maps on the sky sphere (written and committed before it runs; checklist C3, D1 to D3).** The atlas stores both noise splits
at every CpG; the column builder now takes a field list (`stage_seq_atlas.array_columns`, `--build seq-noise-cols`: donor half A and B,
read half A and B at v2.1's sites; the committed columns are built as before), and `--build seq-noise-sphere` draws, per cell, map 1
(donor half A minus half B: donors plus counting) and map 2 (read half A minus half B: counting only) on the sky sphere, with each cell's
summary (`stage_seq_atlas.noise_maps`). **Known answer, the bar:** the committed Met-A reference holds the same four values at every
identity site from the same pieces, so the new columns must equal it there at least 0.999 of the time (`stage_seq_atlas.noise_check`).
Recorded, not scored: each map's mean (a healthy split has none), mean square, and the donor part.

**The two noise maps: drawn; the check met** (`seq_atlas/noise_sphere.json`, `seq_atlas/noise_sphere_cells.csv`; one Google spot machine).
The new columns equal the committed Met-A reference at every identity site (1238368 of 1238368 values). Map 2 (read halves, counting only)
carries no offset in any cell. Map 1 (donor halves) shows two healthy foregrounds, annotated, not subtracted (`seq_atlas/noise_by_chrom.py`,
`seq_atlas/noise_by_chrom.txt`): **the X chromosome**, where 16 of the 48 cells with both halves have donor halves of different sex (the X
chromosome's mean square up to 7.118 times the autosomes'; on the sphere, one diamond-shaped region); and **smooth muscle**, whose two
donor halves differ by 0.0614 across every chromosome (the median cell 0.0054), a whole-genome difference between its donors. **Next, read
before anything is changed:** whether the identity and territory sites the readings use include the X chromosome, and each healthy input's
sex against its reference half.

**Masks, the spectrum on the sphere and the null test (written and committed before it runs; checklist D5, D6 on the sphere, D10).**
Masks (`stage_seq_atlas.sky_mask_sites`): the sex chromosomes (the foreground the noise maps showed) and the 27117 probes that cannot be
placed uniquely in the genome (DEV-MAPPABILITY-01's list), with apodized edges (C2, 2 degrees). The test (`stage_seq_atlas.sky_null`,
`noise_null`; `run_sample.py --build seq-noise-null`): each map on the sphere at NSIDE 128, its band powers against 20 shuffles of its own
values (`sky_statistics`); a band shows structure when it is above all 20. **The instrument's job:** a map with no spatial structure
stays inside its null; real large-scale structure does not. **Synthetic first** (`seq_atlas/sky_null_synthetic.py`): plain noise at most
one band above its null in each of 5 maps; a planted change along the genome puts the lowest band above its null in all 5. **Then the
real maps, one bar:** the counting-only noise map (map 2) at most one band above its null in at least 90 % of cells. Recorded, not scored:
the donor-half map (map 1) of every cell.

**The null test: a setup failure before its first read.** The spot machine stopped at `import healpy`: the chain's requirements pinned
healpy 1.16.6, which imports a function scipy 1.17.1 (also pinned) no longer has, so the sky statistics could not run on a fresh install.
The pin is moved to healpy 1.17.3, checked with the chain's numpy and scipy (white noise reads 0.996 of its variance, as expected).
Nothing was read; the synthetic and its bars are unchanged.

**The null test: synthetic met; the counting map met; healthy people differ by region** (`seq_atlas/sky_null_synthetic.txt`,
`seq_atlas/noise_null.csv`, `seq_atlas/noise_null_summary.py`, `seq_atlas/noise_null_summary.txt`). Synthetic: plain noise had at most one
band above its null in each of 5 maps; the planted change lifted the lowest band in all 5. Real: the counting-only map (map 2) has at most
one band above its null in every one of the 56 cells (bar: 90 %), its power 0.982 to 0.985 of the null across the bands. The donor-half map (map 1)
lies above its null in 4 to 6 bands in all 48 cells with both halves: median 2.574 and 2.912 times the null at the largest scales, falling
to 0.988 at the finest. Healthy people differ from one another over large stretches of the genome, not site by site; strongest in
breast luminal epithelium, pancreatic acinar cells and cortical neurons. This is the person-to-person part the readings already carry
through their donor references; it is the measured case for more donors per cell.

**The bridge to IAM-Atlas v2.1 (written and committed before it runs; checklist C4, D11; a measurement, nothing scored).** For each of
the 56 cells both atlases hold, at the sites both measure (`stage_seq_atlas.bridge`, `run_sample.py --build seq-bridge`): the correlation,
the mean and median difference (sequencing minus v2.1), and how often v2.1's value sits inside the sequencing atlas's 95 % range. v2.1
drew in part on the same published sequencing atlas, so close agreement is expected and not independent; the offsets are the record.

**The bridge: measured** (`seq_atlas/bridge_v21.csv`, `seq_atlas/bridge_summary.txt`). Over the 56 cells the two atlases agree closely
(median r 0.9941; lowest 0.9542, effector memory CD8 T cells, a one-donor cell), with one consistent offset: the sequencing level sits
below v2.1's in all 56 cells (median -0.0316; median absolute difference 0.0267), so v2.1's value falls inside the sequencing atlas's
95 % range at 0.8048 of sites. This is the array-to-sequencing platform difference, now measured cell by cell; the readings on
sequencing use the sequencing atlas, so it does not enter them.

**The genome-distance spectrum (written and committed before it runs; checklist D6).** The same noise maps, read along the genome
(`stage_seq_atlas.genome_spectrum`, `noise_spectrum`; `run_sample.py --build seq-noise-spectrum`): the correlation of two sites' values
by the distance between them, in bands from under 1 kb to 10 to 100 Mb, at the autosome sites the sky masks keep. **The instrument's
job:** values with no regional structure read no correlation; values that share regions read one within the region that falls beyond it.
**Synthetic first** (`seq_atlas/genome_spectrum_synthetic.py`, constructed noise columns read by the one command): plain noise at most 0.01 in every band; planted 1 Mb blocks at least 0.1
under 100 kb, and beyond 10 Mb at most a quarter of the 10 to 100 kb band. **Then the real maps, one bar:** the counting-only map (map 2),
the median cell's correlation at most 0.02 in every band. Recorded, not scored: the donor-half map (map 1), the genomic size of the
regions over which healthy people differ.

**Three synthetics, read again through the one command (disclosed).** The range synthetic and the sky-null synthetic recorded above
called chain functions directly instead of running `run_sample.py`; the push gate refused the same shortcut in the genome-distance
synthetic. All three now build constructed inputs in the committed layout and read them with the one command (`--build seq-sphere-cols`,
`--build seq-noise-null`, `--build seq-noise-spectrum`); the seeds, truths and bars are unchanged. The column build now also writes each
value's standard error (`mu_se`) and takes a constructed folder and inputs list. The earlier results stand as recorded until the re-reads
land, which are recorded below whatever they show.

**A committed script was empty (mine, disclosed).** `seq_atlas/sphere_widths.py` was committed empty: a spelling fix opened the file for
writing before reading it. Its output (`sphere_widths.txt`, `sphere_widths.csv`) was written by the full script before that; the script is
restored and reproduces both files exactly.

**The range synthetic through the one command: met** (`seq_atlas/sphere_range_synthetic.txt`): two donors 1.005 and 0.948, three donors
1.007 and 0.948, the same as the first reading. The sphere columns are rebuilt with `mu_se`; the widths and the bridge reproduce unchanged.

**The genome-distance spectrum: synthetic met; the counting map met** (`seq_atlas/genome_spectrum_synthetic.txt`, `seq_atlas/noise_spectrum.csv`,
`seq_atlas/noise_spectrum_summary.py`, `seq_atlas/noise_spectrum_summary.txt`). Synthetic, through the one command: plain noise read no
correlation in any band; the planted 1 Mb blocks read 0.2 within a block (as built) and none beyond 10 Mb. Real: the counting-only map's
median correlation is at most 0.0002 in absolute value in every band (bar 0.02). The donor-half map gives the scale over which healthy
people differ: median correlation 0.1223 between sites under 1 kb apart, 0.0283 at 1 to 10 kb, 0.0120 at 10 to 100 kb, 0.0037 at 0.1 to
1 Mb, and none beyond. The differences are mostly local, among neighboring CpGs, with a weak tail to about 1 Mb that adds up to the
large-scale power on the sphere. **The sky-null synthetic through the one command: met** (`seq_atlas/sky_null_synthetic.txt`).

**Isotropy by chromosome (D14; a measurement, no bar; written and run after the noise maps)** (`seq_atlas/isotropy.py`,
`seq_atlas/isotropy.csv`, `seq_atlas/isotropy.txt`). Each autosome's mean of each noise map, per cell, against its own standard error and
the cell's median over its autosomes. The counting-only map: no autosome more than 4 standard errors from its cell's median in any of the
1232 cell-chromosome pairs (largest 3.79), so the read split is the same everywhere in the genome. The donor-half map: 409 of 1056 pairs
beyond 4 (largest 37.74, erythrocyte progenitors, chromosome 21). Its neighboring sites are correlated (the genome-distance spectrum), so
these standard errors are too small and the count overstates; it is recorded, not scored, as the person-to-person part again.

**The report globe and the information sphere on sequencing (written and committed before any reading).** Until now a sequenced
sample's globe, and the information sphere's distance from its own healthy self (checklist C2), were drawn against IAM-Atlas v2.1, an
array atlas, though the bridge measured a platform offset of 0.0316 between the two. A sequenced sample read with the sequencing
reference (`--dev-seq-ref`) now draws both against the IAM Sequencing Atlas's sphere columns (`report_globe_v4.use_atlas`), which carry
the same sites and now also each cell's donor spread (`donor_sd`, as stored in the array columns). The change rule, the layout and
everything else are unchanged. **The instrument's job:** a healthy sequenced sample shows little change on its globe. **Check, then one
bar on the next healthy pass:** the spent healthy B-cell input GSM5652316 read both ways (diagnostic); then, on the healthy inputs, the
median share of changed sites no higher against the sequencing atlas than against v2.1 (from pass 5's bundles).

**The check, and the bar moved before any scoring (disclosed)** (`seq_atlas/globe_bcell_check.txt`). The B-cell input read against the sequencing atlas: 0 changed sites
(62 against v2.1), 0 of 27 territories changed (1 against v2.1), Fisher-Rao distance from its own healthy self 0.11988 (0.20213);
sorted-cell Met-A unchanged (1.06673). But this input is one of the atlas's own donors, and the sphere columns pool every donor, so its
reference includes itself; the check is a diagnostic only. The one command now flags any sequenced sample that is an atlas input
(`reference_includes_this_sample`). **The bar moves to the healthy samples no build has used** (the 25 outside neurons and 10 pancreas
samples of the selection-free test): the median number of changed sites, and of changed territories, no higher against the sequencing
atlas than against v2.1, the same samples read both ways by the one command (`--dev-globe-v21` draws the v2.1 globe; every other reading is the same). Scorer, committed before the reads: `seq_atlas/globe_outside_score.py`.

**The range at levels of 0 or 1 (written and committed before it runs).** The outside globe reads (still landing) show about 800 changed
sites in every outside neuron against the sequencing atlas, and about 20 against v2.1, while changed territories and the distance from
the healthy self are lower. The cause, measured: 77361 neuron sites in the sequencing atlas have a width of exactly 0, all three donors
reading exactly 0 or 1, because counting's error sqrt(p (1 - p) / n) vanishes there, and 2729 more have a width under 0.005; over all
56 cells, 4351930 sites have zero width, while v2.1 has none (smallest 0.0033) (`seq_atlas/zero_widths.py`, `seq_atlas/zero_widths_before.txt`). The range synthetic did not see it: its bars covered only sites with a true level between 0.1 and 0.9. **The fix
(`stage_seq_atlas.range_of_average`):** counting's error from the level with half a read added each side, (m + 0.5) / (n + 1), so n
reads never pin a level finer than about one read, and the average of several donors never known better than its pooled counting allows
(the donors' t quantile where their spread is the larger part). Nothing fitted. **Synthetic version 2** (`seq_atlas/sphere_range_synthetic.py`):
the same bars, plus coverage of at least 0.90 at the sites outside 0.1 to 0.9, and no measured site with zero width. **Then:** the sphere
columns rebuilt and the outside samples read again against them; the globe test is scored on the first reading as committed, and again.

**Synthetic version 2: not met; version 3 written before it runs** (`seq_atlas/sphere_range_synthetic_v2.txt`). The floor met the two new
bars (coverage at the extreme sites 0.943 and 0.944 for two and three donors; no measured site with zero width) but broke the first:
taking the larger of the donors' error and counting's everywhere overstates the variance (1.251 and 1.166 of the true) and coverage fell
to 0.925 for two donors. Where donors differ, their spread already holds each donor's counting; only where they agree exactly does it say
nothing. **Version 3:** counting's error (from (m + 0.5) / (n + 1)) only where the donors' spread is exactly 0, and the donors' spread as
before everywhere else. The same synthetic and the same four bars.

**Synthetic version 3: met; the columns rebuilt** (`seq_atlas/sphere_range_synthetic_v3.txt`, `seq_atlas/zero_widths_after.txt`,
`seq_atlas/sphere_widths_v3.txt`). Two donors: stated over true variance 1.010, coverage 0.955; three: 1.007 and 0.948; at the extreme
sites 0.942 and 0.945; no measured site with zero width. The sphere columns are rebuilt with version 3: no site of any cell has zero width
now (smallest 0.0002), and the cell medians of the range's half-width are unchanged for two and three donors (0.2056, 0.0783). The outside
samples are read again against these columns once the first reading, on the old columns, has landed and been scored as committed.

**The globe on the outside healthy samples, first reading (old columns): not met** (`seq_atlas/globe_outside.csv`,
`seq_atlas/globe_outside.txt`). All 35 read both ways, none an atlas input. Changed sites: median 851 against the sequencing atlas, 68
against v2.1 (bar 1 not met); changed territories 1 and 1 (bar 2 met); the distance from the own healthy self lower against the
sequencing atlas in every group (median 0.17795 against 0.19809). The cause is the zero widths recorded above; the second reading, on
the rebuilt columns, is running.

**The globe's width on sequencing: the sample's own counting (written and committed before it runs).** The second reading (rebuilt
columns, still landing) shows more changed sites, not fewer: about 3000 in its first three neurons, because sites with no width before
were skipped and now count. The globe's width holds the person-to-person spread and the atlas's own uncertainty, not the sample's own
counting: on an array that noise is about the same at every site and the sample's own-spread scaling takes it out; on sequencing it
changes from site to site with the reads (one stray read in 20 is a change of 0.05). **The fix (`report_globe_v4.build`, `counts`):** for
a sequenced sample read against the sequencing atlas, each site's width also holds the sample's own counting at its expected level,
e (1 - e) / n with half a read added each side, as in the atlas's range. Nothing else changes. **Synthetic first**
(`seq_atlas/globe_seq_synthetic.py`): three healthy people drawn from the atlas's own B-cell column at a real sample's depth, and one with
a planted rise of 0.3 on chromosome 8 (100 to 115 Mb), read by the one command; bars: the healthy median at most 50 changed sites; the
planted one at least that median plus half its planted sites. **Then** the outside samples read a third time, and the committed bar
(changed sites, and changed territories, no higher than against v2.1) scored again.

**The globe synthetic, version 1: not met as written; two errors in its design (mine, disclosed)** (`seq_atlas/globe_seq_synthetic.txt`).
The three healthy people: 5, 1 and 3 changed (bar 1 met, median 3). The planted person: 215, against a bar of 1381 (not met). Two errors
of mine: (1) the globe's "sites_changed" counts blocks of 10 sites (it equals "blocks_changed"), so the bar compared blocks with sites;
the planted region holds 2756 measured sites, about 275 blocks; (2) a rise of 0.3 on mostly methylated B-cell sites was cut off at full
methylation: only 1097 of the 2756 sites got the whole 0.3. **Version 2, written before it runs** (`seq_atlas/globe_seq_synthetic_v2.py`):
a new seed; each planted site moved 0.3 toward the opposite end, so none is cut off; the bars in blocks (healthy median at most 50; the
planted person at least that median plus half the planted blocks).

**The globe synthetic version 2: met. The outside samples, second reading: not met** (`seq_atlas/globe_seq_synthetic_v2.txt`,
`seq_atlas/globe_outside_2.csv`, `seq_atlas/globe_outside_2.txt`). Synthetic, with the sample's own counting in the width: the healthy
three 2, 1 and 2 changed blocks (bar at most 50); the planted person 231 of its 275 planted blocks (bar at least 139.5). The second
reading of the outside samples (rebuilt columns, before the sample's counting was added): 2670 changed blocks against 68 on v2.1 (bar 1
not met), territories 0 against 1 (bar 2 met). The third reading, with the sample's counting in the width, is running.

**The outside samples, third reading (the sample's own counting in the width): not met; the globe's change count flagged**
(`seq_atlas/globe_outside_3.csv`, `seq_atlas/globe_outside_3.txt`). Changed blocks: median 156 against the sequencing atlas, 68 against
v2.1 (bar 1 not met). By cell: pancreatic acinar 156 against 158 and duct 267 against 275, now level with v2.1; cortical neurons 109
against 24. Changed territories 0 against 1 (bar 2 met); the distance from the own healthy self lower in every group. The neuron column
rests on three donors, and its donor-half noise map carries the strongest large-scale person-to-person structure of any cell, so three
donors understate how far healthy neurons from another laboratory differ, and the globe shows that as change; 109 of about 81,000 blocks.
That is a matter of more donors, not of the instrument. **Status:** the globe's change count on sequencing is read and flagged as not met
on healthy samples outside the atlas; the territories changed and the distance from the own healthy self are read as they are.

**Donor traits (A7; a table, no bar)** (`seq_atlas/donor_traits.py`, `seq_atlas/donor_traits.csv`, `seq_atlas/donor_traits.txt`). Each
input's recorded characteristics from its GEO sample record: tissue, cell type, the source recorded as "lab", age, sex and race. 92 female,
60 male, 1 not recorded; 23 of 56 cells have donors of both sexes, and in 29 cells the two sealed donor halves differ in sex makeup,
which is the X-chromosome foreground the donor-half noise maps show (annotated, not subtracted). Age is recorded for 152 inputs. The
"lab" field names six sources; six cells have donors from two of them (aorta endothelium, hepatocyte and four pancreatic cells), the only
cells where a source offset (A5) can be measured.

**Group-A on sequencing (B9; written and committed before any read).** **The instrument's job:** two groups of healthy people read as no
different after the tare, and a group shift planted into healthy samples is found (the arrays' commissioning, word for word). **Chain:**
`stage_groupa.py` now collects a sequencing reading from the one command's bundles (`collect ... sorted|whole|islands`; a list with no
slide column is tared within its series); the tare and the comparison are unchanged. **Set:** the 137 healthy inputs of healthy pass 5
(spent), `seq_atlas/groupa_seq/groupa_seq_samples.csv`. **Bar 1** (pass 5's committed bundles, sorted-cell Met-A): the sealed donor halves
A (76) and B (61) read as no different: under 2 % and under 2 standard errors. **Bar 2:** the 137 read again by the one command with pass
5's settings, every other sample (sorted by number, 68) with a planted 1 % loss of its own cell's identity pattern written into its read
counts (`groupa_seq/groupa_seq_plant.py`: the arrays' plant on counts, rounded without bias; the other 69 written back unchanged by the
same script): planted above unplanted by more than 2 standard errors, the unplanted the tare's references. Recorded, not scored: whole-sample
Met-A and the islands; the smallest difference each design could have found. Scorer: `groupa_seq/groupa_seq_score.py` (changed before any scoring so each bar is scored where its bundles are: bar 1 on GitHub
from pass 5's artifacts, bar 2 from the Google bucket, then joined by its `combine` step; the bars unchanged). **Disclosed:**
sizing the test, I read women against men on pass 5's sorted-cell Met-A (1.0029, 82; 1.0154, 55) and the healthy spread (relative SD
0.0728, against about 0.02 on arrays); the sexes are therefore not used for a bar. With this spread a 1 % planted loss may fall under what
137 samples can find; the result is recorded whatever it shows.

**The tumor gauge on sequencing: the distance to the entropy ceiling, counted (B8; written and committed before it runs).** The
ceiling reading (`stage_metaa_atlas.ceiling_reading`) was skipped on every sequenced sample's in-mix cells and missing from the sorted-cell
reading, the one that reads a sample against its cell of origin; the territories gave it, but with the expectation at exact levels while
their own reading is counted. **The change (`stage_metaa_atlas.ceiling_counted`):** on sequencing the ceiling is the reading the sample
would give if every site's true level were one half, read at the sample's own reads by the same expectation as the reading itself
(donor by donor at a common depth for the sorted-cell reading; the thinned counting for the counted islands; expected_H_counts elsewhere).
Derived, nothing fitted; arrays unchanged. **Synthetic first** (`seq_atlas/ceiling_seq_synthetic.py`, through the one command): B-cell
people drawn from the atlas at a real sample's reads, healthy, half way to one half, and at one half everywhere; bars on the sorted-cell
distance to the ceiling: healthy at least 0.9, the ceiling sample at most 0.1, half way between.

**The cell-count floor (D7; written and committed before it runs).** The cosmology map's row 3 (cosmic variance: a finite number of
independent modes) is, in a methylation sample, the finite number of copies of each cell behind a reading: a cell at share f read at n
reads per site has about f n copies of itself at each site, whatever the depth of the library. Plasma carries few genome copies and small
tissue shares, so a cell's in-mix reading there can rest on less than one copy per site. **The change (`stage_metaa_atlas.read_in_mix`, on
sequencing):** each cell's reading records copies_per_site (share times the median reads at its identity sites) and copies_behind_reading
(share times all those reads), and is flagged below one copy per site (`COPY_FLOOR`, stated). Recorded, not a bar on any sample.
**Synthetic first** (`seq_atlas/copies_seq_synthetic.py`, through the one command): B cells and monocytes at 0.9 and 0.1, every site at 4,
8, 20 or 60 reads; bars: copies within 10 % of the Deconvolver's share times n, and the flag on exactly the cells below one copy.

**The counted ceiling: synthetic met** (`seq_atlas/ceiling_seq_synthetic.txt`, `seq_atlas/ceiling_seq_synthetic.log`). Distance to the
ceiling on the sorted-cell reading: healthy 1.01136 (bar at least 0.9), half way 0.24266, at one half everywhere 0.03915 (bar at most 0.1);
the islands in the same order (each sample's readings in the log). Half way reads well below the middle because a site's disorder rises fastest near 0 and 1: the
gauge is in disorder, not in level. The tumor gauge now reads on sequencing against each sample's own cell of origin.

**The cell-count floor: synthetic met** (`seq_atlas/copies_seq_synthetic.txt`). Every cell's reported copies per site matched the
Deconvolver's share times the reads (bar 1 met), and the flag fell on exactly the cells below one copy (bar 2 met). Monocytes, planted at
0.1: 0.272 and 0.618 copies at 4 and 8 reads (flagged), 1.598 and 5.58 at 20 and 60 (not flagged); B cells 3.486 to 53.28, never flagged.
For plasma this is the measure of which cells a library can speak for.

**Group-A on sequencing: met on the sorted-cell reading** (`seq_atlas/groupa_seq/groupa_bar1.txt`, `groupa_bar2.txt`,
`groupa_seq_result.txt`; bar 1 scored on GitHub from pass 5's artifacts, bar 2 from the 137 reads on Google spot machines). **Bar 1:** the
sealed donor halves, -0.97 %, -0.772 standard errors: no difference found (met). **Bar 2:** the planted 1 % loss, +7.93 %, 6.493 standard
errors: found (met); the smallest difference the design could have found was 2.55 %. **Recorded, not scored:** whole-sample Met-A saw
the planted group at +1.10 % (1.274 standard errors, not found; the plant touched only each cell's own identity sites), and the halves at
-1.64 % (-1.958). **Flagged:** on the islands the two healthy halves read as different, +4.51 % at 2.231 standard errors, where healthy
groups should not differ. Group-A on the islands reading is not used on sequencing until that is understood.

**Laboratory offsets (A5) and covariance (D8).** The "lab" field in the donor traits names the tissue supplier; one laboratory sequenced
every input, so a laboratory offset cannot be measured inside the atlas. It is measured where it matters, by reading healthy samples from
other laboratories against the atlas (`seq_atlas/outside_data/`). The covariance between sites cannot be estimated pair by pair from one to
three donors per cell; as in a sky survey, it is taken as a function of distance, from the genome-distance spectrum already measured
(`seq_atlas/noise_spectrum_summary.txt`).

**Higher-order statistics: Minkowski functionals (D13; written and committed before it runs).** The power spectrum sees only pairs of
sites; the Minkowski functionals see the shape of a map at each level, so structure that pairs miss shows here. For each noise map on the
sphere (placed as the null test places it, the same masks) at 5 thresholds (-2 to 2 standard deviations): V0 the share of the sphere above,
V1 the length of the boundary, V2 the patches above minus the patches below (`stage_seq_atlas.minkowski`, `sky_minkowski`,
`noise_minkowski`; `run_sample.py --build seq-noise-minkowski`), each against 20 shuffles of the map's own pixels. **Synthetic first**
(`seq_atlas/minkowski_synthetic.py`, constructed noise columns through the one command): the plain maps at most 10 of their 50 tests
outside the shuffles; every planted map's boundary length at threshold 0 outside. **Then** every cell's two maps; recorded: the counting
map is expected to look like its shuffles (no structure, as in the null test); the donor map shows the shape of how healthy people differ.

**Minkowski synthetic, version 1: not met** (`seq_atlas/minkowski_synthetic.txt`). Bar 1: the plain maps fell outside their shuffles in 15
of 50 tests (at most 10); bar 2 met (every planted map's boundary length at threshold 0 outside). The plain maps' boundary length sat below
the shuffles at plus and minus 2 standard deviations in all five: each sphere pixel averages one to four sites, so pixels with fewer sites
spread wider, and those pixels cluster where the genome's sites are sparse; a shuffle over every pixel breaks that clustering. The test was
seeing the uneven spread of sites on the sphere, not the methylation. **Version 2 (written and committed before it runs;
`seq_atlas/minkowski_synthetic_v2.py`, seed 20261123):** each shuffle only among pixels that average the same number of sites, so the null
keeps the map's uneven spread; the bars unchanged.

**Minkowski functionals: synthetic version 2 met; the real maps not yet readable** (`seq_atlas/minkowski_synthetic_v2.txt`,
`seq_atlas/noise_minkowski.csv`, `seq_atlas/noise_minkowski_summary.txt`). Synthetic: the plain maps 3 of 50 tests outside their
shuffles (bar at most 10, met); every planted map found (met). The real maps: the donor-half map outside in 459 of 480 tests (0.956), and the
counting-only map in 455 of 560 (0.812). The counting map carries no spatial structure (it passes the null test on the power spectrum), so
its departure is its spread changing across the sphere with each site's reads and level, which a shuffle among pixels does not keep. On real
maps the test therefore cannot yet tell structure from spread. **Needed:** each site's noise value divided by its own counting error before
the map is drawn; the noise columns hold the half levels but not each half's reads, so the columns must carry those first. Until then the
Minkowski functionals are built and tested on constructed maps, and not read on the atlas.

## Healthy samples from another laboratory (written and committed before any is read)

**The instrument's job:** a healthy sample sequenced and processed by another laboratory reads Normal against the IAM Sequencing Atlas.
Every atlas input came from one laboratory, so this is the first test of the atlas as a ruler for anyone else's data (and the place a
laboratory offset, checklist A5, shows). **Samples** (`seq_atlas/outside_data/blueprint_healthy_test_samples.csv`): BLUEPRINT's public
healthy blood-cell bisulfite libraries, 49 adult venous blood and 10 cord blood, in 10 atlas cells, none read before (three donors also
gave libraries of other cells that were read in earlier work, flagged in the list). Each read by the one command from its GRCh38 call and
coverage bigWigs (`run_sample.py --dev-seq-beta CALL.bw,COV.bw`), `--specimen "sorted CELL"`, the study's settings, against every donor
of the atlas. **Bars** (`seq_atlas/outside_data/blueprint_healthy_score.py`, on the adult libraries): own-cell Met-A median within 0.95 to
1.05; at least 75 % of them within 0.95 to 1.05; whole-sample Met-A median within 0.95 to 1.05. Recorded: the territories, the C-Score,
coupling-A, the own-cell share, each cell; the cord blood (newborn cells against an adult atlas).

**Healthy samples from another laboratory: not met** (`seq_atlas/outside_data/blueprint_healthy_test.csv`,
`blueprint_healthy_test.txt`). 38 of 59 read; 21 stopped because fewer than 90 % of their cell's identity sites had a value (BLUEPRINT's
tracks leave out sites). On the 49 adult libraries: own-cell Met-A median 0.7134 (bar 1 not met), none within 0.95 to 1.05 (bar 2 not met),
whole-sample Met-A 0.9479 (bar 3 not met). Every cell reads low, the islands lowest (median 0.2924). **The cause, measured**
(`blueprint_islands_offset.py`, `.csv`, `.txt`; diagnostic, added after scoring): at the islands the libraries read 0.00653 less
methylation than the atlas reference (below it in 52 of 59), and the islands reading follows that offset (rho +0.811). Where the true
level is near 0, a fraction of a percent of methylated calls sets the disorder, and each laboratory's floor of such calls (its bisulfite
conversion, its calling pipeline) differs; the atlas's one laboratory sits higher than BLUEPRINT's. The instruments are reading the
laboratory's floor, not the cell. This is the laboratory offset (checklist A5), measured. **Next:** each sample's own floor, measured at the
atlas's floor sites (`stage_seq_atlas.library_floor`, already in the chain) and the reference donors' floors, carried into the
expectation (the sample against itself); synthetic first, with two laboratories' floors planted; then this test read again as development
(the libraries are now spent), and a fresh outside laboratory kept for the test.

**Each library's floor, measured** (`seq_atlas/outside_data/library_floors.py`, `library_floors.csv`, `library_floors.txt`; diagnostic,
added after the outside-laboratory test was scored). The floor is the share of methylated calls at the atlas's floor sites (46989 sites
unmarked in every cell), from each library's own calls (`stage_seq_atlas.library_floor`, already in every sequenced bundle). BLUEPRINT's 59
libraries: median 0.00154; the atlas's own inputs (69 unplanted reads of Group-A): median 0.00626 (0.00469 to 0.01041). Each BLUEPRINT
library's islands reading follows its own floor (rho +0.906), and so do the atlas's own inputs (rho +0.444).

**The common library floor (written and committed before it runs).** **The instrument's job:** a healthy sample from another laboratory
reads Normal. **The physics:** a library reads a truly unmethylated read as methylated with its own probability f (its floor), so it reads
f + (1 - f) p where the truth is p. Two libraries with different floors are read alike by raising the lower to the higher: each of its
unmethylated reads turned methylated with probability u = (F - f) / (1 - f). The expected disorder after that is exact
(`stage_metaa_atlas._Hj`: the sum over Binomial(t; m - j, u) of H((j + t) / m)), so nothing is fitted and nothing is removed from a
measurement. **The change** (`--dev-floor-equal`, development): the sample's own floor is measured before any reading; the sorted-cell
reading puts the sample and each reference donor at the highest of their floors (`donor_pair`); the islands put the sample and the
reference half's pooled reads at the higher of the sample's floor and that half's mean floor (`islands_counted`). Each atlas input's floor
is recorded by `run_sample.py --build seq-input-floors` (`chain/Runtime Matrices/Seq_Atlas/input_floors_v1.parquet`; first committed empty, then built). **Synthetic first**
(`seq_atlas/outside_data/floor_synthetic.py`): constructed reference donors at floor 0.006, nine constructed B-cell people in three
laboratories (floors 0.0015, 0.006, 0.012), each read with and without the common floor. Bars: each laboratory's median sorted-cell
Met-A, and median islands reading, within 0.03 of 1 with the common floor.

**The atlas inputs' floors: built** (`chain/Runtime Matrices/Seq_Atlas/input_floors_v1.parquet` and `.json`; one Google spot machine, the
137 cached inputs from the bucket and the other 16 from GEO, by `run_sample.py --build seq-input-floors`). All 153: median 0.00628 (0.00469
to 0.01041); half A 0.00666, half B 0.00646 on average. The 69 unplanted inputs read again for Group-A carry the same floor in their bundles
to six places, so the table and the chain's own measurement agree.

**The common floor: synthetic met** (`seq_atlas/outside_data/floor_synthetic.txt`, `floor_synthetic_readings.csv`). With the common
floor every laboratory reads Normal: sorted-cell Met-A medians 1.0108, 0.9860 and 0.9990 (floors 0.0015, 0.006 and 0.012), the islands
0.9973, 0.9904 and 0.9910 (both bars met). Without it the floor shows as the offset seen on BLUEPRINT: the low-floor laboratory 0.9388 and
0.6618, the high-floor laboratory 1.0928 and 1.4236. Each library's floor was measured by the chain to within 0.0003 of the floor planted.

**Healthy pass 6 and BLUEPRINT read again, with the common floor (written and committed before either is read).** **Pass 6:** the 137
healthy inputs (`seq_atlas_01/healthy_fullchain6_samples.csv`: pass 5's settings plus `--dev-floor-equal`), each read by the one command from
its cached .beta, scored by the committed pass scorer (`seq_atlas_01/healthy_fullchain_score.py`) with pass 5's bar: every reading Normal on
the healthy inputs. **BLUEPRINT read again** (`seq_atlas/outside_data/blueprint_reread_floor_samples.csv`; development, the libraries are
spent): the same 59 libraries with `--dev-floor-equal`, scored by the same committed scorer (`blueprint_healthy_score.py`); reported, not a
test. The real outside-laboratory test waits for a laboratory not yet read.

## The IAM Sequencing Atlas: a granulocyte column (written and committed before any of it is built)

**Why:** the atlas's 56 cells hold no granulocytes (neutrophils), the largest share of healthy blood and of healthy plasma DNA, so a
blood or plasma sample's granulocyte DNA is given to other cells. **Inputs** (`seq_atlas/gran_inputs.csv`): the atlas series' own three
healthy blood granulocyte samples (GSM5652313, GSM5652314, GSM5652315; the same laboratory as the other 153 inputs), spent in earlier
development work (DEV_METAA_HEALTHY_01, step 3), which does not bar a ruler's build; halves sealed by seed 20261125 (GSM5652314 and
GSM5652315 in half A, GSM5652313 in half B). **Build:** the column's pieces by the one command (`--build seq-atlas`, the 56 cells' builder)
into their own folder beside the 56 cells' (each machine downloads the three .beta files once, checks their size, and the builder reads
them as local files; first launch read GEO's ftp addresses, which the builder does not open: corrected to https, disclosed), then every derived table rebuilt from both folders as one 57-cell atlas: the Met-A identity
sites, the Deconvolver's reference, the array-site and noise columns, the floor sites, the per-donor and per-half counts, the held-error
reference and the input floors. **Test:** the 57-cell atlas's healthy pass (every input read against the other sealed half of its own cell,
as pass 6) with the same bar: every reading Normal. The granulocytes' own check: the two sealed halves' noise map and the bridge to
IAM-Atlas v2.1's neutrophils, recorded.

**Healthy pass 6 (the common floor): every reading Normal, but the floor overcorrects** (`seq_atlas/outside_data/healthy_fullchain6_summary.txt`, a copy of
`DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain6_summary.txt`; the committed pass scorer). All 137 read: C-Score 1.0003 (0.964 inside its limit), sorted-cell Met-A
1.0085, whole-sample Met-A 1.0055, the domains 1.0011, the islands 1.0403 (pass 5 1.0122). **Read against each input's floor**
(`seq_atlas_01/pass6_floor_diag.py`, `pass6_floor_diag.txt`; diagnostic, added after scoring): without the floor the islands rise with the
floor difference between sample and reference (rho +0.341); with it they fall with it (rho -0.555). The data carry 0.488 of the correction
the floor makes: within the atlas's laboratory, only about half of the rate measured at the floor sites reaches the islands, so the rate
there is not one false-call rate shared by every site. Sorted-cell Met-A does not follow the floor at all (rho -0.010), and the correction
adds a small tilt (-0.126). **So the common floor, applied whole, is not the physics of these libraries:** it stays a development flag, off by
default; pass 5's settings remain the healthy baseline. **What it leaves:** the floor sites measure something more than a uniform false
call (part of it is specific to the sites, the person or the cell), and the outside laboratory's shortfall at the islands (0.2924) is larger
than a floor of 0.0015 makes in the synthetic (0.6618), so more than the floor separates the two laboratories. BLUEPRINT read again with the
floor (development) is recorded below whatever it shows.

**The granulocyte column: pieces built; identity sites and intake.** The column's 113 pieces were built by the one command on four Google
spot machines (`seq_atlas/pieces_gran/` in the bucket): 3 donors (2 in half A, 1 in half B), median 87 reads per site, the same stored
fields as the 56 cells. IAM-Atlas v2.1 has no granulocyte column, and Met-A's identity sites are chosen on v2.1, so the granulocyte column
is read at v2.1's neutrophil identity sites (`stage_metaa_atlas.IDENTITY_OF`; blood granulocytes are mostly neutrophils; stated). Intake
accepts isolated or sorted granulocytes as a cell of the sequencing atlas (`stage_0_intake.SEQ_ATLAS_ONLY_CELLS`), read and flagged like any
atlas cell. The three inputs are added to `seq_atlas/seq_atlas_inputs.csv` (156 inputs, 57 cells). **The rebuild:** the tables that compare
cells (the Deconvolver's reference, the floor sites, the Met-A reference, the array-site and noise columns) are rebuilt by the same commands
from all the piece folders; the per-input tables (per-donor and per-half counts, island counts, held-error tiles) take the granulocytes'
rows counted the same way; the input floors are measured again on the new floor sites; then the sphere columns. **Test:** a healthy pass
on the 57-cell atlas with pass 5's settings and bar.

**BLUEPRINT read again with the common floor (development; the libraries are spent)** (`seq_atlas/outside_data/blueprint_reread_floor.txt`,
`blueprint_reread_floor.csv`; the committed scorer). The adult libraries' islands rise from 0.2924 to 0.8594 and own-cell Met-A from
0.7134 to 0.7983; the whole-sample reading (0.9479) and the domains (0.9211) do not move. The floor closes most of the islands' gap and a
small part of own-cell Met-A's. **What stays open:** the Deconvolver gives these sorted libraries a median own-cell share of 0.5508 (the
atlas's own inputs read 0.9817), so against the atlas they read as mixtures; own-cell Met-A is then read at sites the other cells do not
share. Next on this line: why a sorted library from another laboratory reads as half its own cell (the Deconvolver's markers on that
laboratory's floor, or the cells themselves), synthetic first, before any outside laboratory not yet read is used.

**The rebuild's first attempt (my error, disclosed).** The builders read one piece file at a time and take the cells in that file as
every cell; given the granulocytes' folder beside the 56 cells', the floor-site rule judged the granulocyte-only files alone (442223
sites, against 46989: a site "unmarked in every cell" there meant unmarked in granulocytes). The provisional megakaryocyte folder had
worked only because the builders leave that column out. The machine was stopped and none of its tables used. **Fix:**
`stage_seq_atlas.merge_pieces` (`run_sample.py --build seq-merge-pieces`) joins the folders block by block into one 57-cell folder (the
same CpG rows checked; a cell in two folders refused), and every builder reads that folder. Checked on one block: every value identical to
its source; the 57-cell floor sites a subset of the 56-cell ones (592 of 593); and the 56-cell rule on the same block gives v1's sites
exactly.

**The 57-cell atlas: built** (one Google spot machine, the one command throughout; `seq_atlas/atlas57_summary.txt`, `seq_atlas/atlas57_merge.json`,
`seq_atlas/atlas57_add_rows.py` and `.json`). The pieces merged into one 57-cell folder (113 pieces; kept in the bucket at
`seq_atlas/pieces57/`), then from it: the Deconvolver's reference (57 cells, 5699 markers, was 5599), the floor sites (46947, was 46989;
the 57-cell sites a subset of the 56-cell ones), the Met-A reference (57 cells; granulocytes 6000 identity sites, v2.1's neutrophil
sites; every other cell's rows unchanged), the array-site columns (every 56-cell value identical, checked on eight cell and field pairs;
the granulocyte column at all 813951 sites) and the noise columns (kept in the bucket, `seq_atlas/atlas57b/noise_cols.tgz`). The
granulocytes' per-input rows were counted from their own three inputs and added (per-donor counts 18000 rows, per-half counts 6000, island
counts 66986; the 56 cells' rows kept as they were, checked; a builder that writes a row for every cell gives zeros for cells it was not
given, checked before those rows were left out). Every input's floor measured again on the new floor sites (156; median 0.0062855; the
granulocytes 0.00899, 0.005889, 0.008653), and the held-error reference rebuilt from every input's tiles (57 cells; granulocytes: 3
donors, held error 0.03841). **Next:** the sphere columns, then the 57-cell healthy pass (`seq_atlas_01/healthy_fullchain7_samples.csv`:
pass 5's 137 inputs and settings, and the three granulocytes, each read against the other sealed half), the same scorer and bar.

**The 57-cell sphere columns** (`chain/Runtime Matrices/Seq_Atlas/sphere_cols/`, `sphere_cols_build.json`; `run_sample.py --build
seq-sphere-cols` on one Google spot machine from the committed 57-cell tables). The 57-cell healthy pass's machines installed these same
files from the bucket, each checked against its SHA-256 before any read.

**The 57-cell healthy pass: met** (`seq_atlas/outside_data/healthy_fullchain7_summary.txt`, a copy of
`DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain7_summary.txt`; `seq_atlas/outside_data/pass7_vs_pass5.txt`). All 140 inputs read
by the one command against the 57-cell atlas (five Google spot machines), scored by the committed pass scorer (given the 140-input list:
an optional third argument added before scoring, the default unchanged): C-Score 1.0003 (0.964 inside its limit), sorted-cell Met-A 1.0075,
whole-sample Met-A 1.0059, own-cell share 0.9812, the domains 1.0014, the islands 1.0118: every reading Normal, the bar met. **The 137
earlier inputs** read as they did on 56 cells (the largest change in any reading 0.0059, in an own-cell share; Met-A within 0.00012).
**The granulocytes**, each read against the other sealed half: own-cell share 0.9907 to 0.9929; sorted-cell Met-A 1.0516, 0.9346 and
0.9731; whole-sample Met-A 0.9917 to 1.0340. Their islands read 1.2600 (GSM5652313, the only donor of half B, floor 0.00899, read against
half A) and 0.7189 (GSM5652314, floor 0.005889, read against GSM5652313 alone): a mirror pair, the islands' known dependence on each
library's floor seen between two people. The granulocyte column rests on three donors, so that dependence is not averaged out.

**Where BLUEPRINT's own-cell share goes** (`seq_atlas/outside_data/blueprint_share_destinations.py`, `blueprint_share_destinations.txt`;
diagnostic, added after the test and its re-read were scored; nothing scored). The share the Deconvolver does not give a sorted library's own
cell goes almost entirely to that cell's own subtypes: BLUEPRINT's CD8-positive alpha-beta T cells (every CD8 T cell) read naive CD8 T cells
0.482 against the atlas's "cd8 t cells" 0.059; its CD4-positive T cells naive CD4 0.251 beside 0.380; memory B cells B cells 0.187 beside
0.657. Where BLUEPRINT's sorting matches an atlas cell's definition, the own share is high: naive B cells 0.877, CD56-dim NK cells 0.855.
Monocytes are the exception: 0.558 own, 0.315 lung interstitial macrophages (a related but different cell). **So the test's low own-cell
shares were mostly my mapping of BLUEPRINT's broad fractions onto the atlas's narrower cells** (the sample list's `atlas_cell` column),
not the instrument; own-cell Met-A then read a mix of subtypes against one subtype. **For the next outside laboratory, written before it
is read:** only libraries whose sorting definition matches an atlas cell's, and the whole-sample (in-mix) reading as the primary one.
What remains on BLUEPRINT that is not definition: the monocytes' share given to macrophages, and the islands' floor dependence.

**How each library's floor reaches each kind of site** (`seq_atlas/outside_data/floor_by_site_class.py`,
`floor_by_site_class_healthy.txt`, `floor_by_site_class_plasma.txt`; diagnostic, development, nothing scored). At sites each sample's own
mix expects unmarked, the share of reads called methylated, against the library's floor measured at the floor sites. **In the 30 healthy
plasmas the floor is one rate for the whole library:** the islands take 1.005 of it and other unmarked sites 1.061, and both follow each
library's floor (rho +0.749 and +0.885). **Inside the atlas's own laboratory it is not:** across the 137 healthy inputs the islands take
0.291 of the floor-site rate and other unmarked sites 0.393, and neither follows the library's floor (rho +0.039 and +0.026). So in those
libraries most of what the floor sites read is not a false-call rate the whole library shares, which is why the common floor overcorrected
in healthy pass 6 and closed the gap in plasma. **Next, written before it runs:** the floor read where a reading reads, at the unmarked
sites of the reading's own class rather than at the floor sites, synthetic first.

**Deposited on Zenodo (DOI 10.5281/zenodo.23934666)** (`deposit/zenodo_seq_atlas_upload.py`, `deposit/zenodo_seq_atlas_SHA256.txt`, `seq_atlas/zenodo_deposit_summary.txt`). The 57-cell atlas at every
CpG (113 merged pieces in 12 sets), the runtime tables the chain reads, the input lists and donor traits, and a SHA-256 manifest naming every
piece: 19 files, 18.43 GB, each file's MD5 checked as it arrived. The first upload stopped at Zenodo's 100-file limit for a record (the pieces
had been sent one by one; disclosed) and the pieces were sent again in sets. Replication needs only this record, the repository and the public
archives the inputs came from; the deposit manifest lists every file at one of those three.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements (results register SEQ-ATLAS-PUBLIC).

<!-- PLAIN:BEGIN (generated by CANON/results_to_tex.py from CANON/results_register.json; do not edit by hand) -->
**In plain words** (The IAM Sequencing Atlas is public, with a DOI): Our second reference, for sequenced samples, is deposited on Zenodo with a permanent DOI and an open license: every healthy cell type it holds, at every CpG of the genome, counted from healthy donors' public reads with nothing fitted, with a checksum for every piece. The repository's one command rebuilds it from the public files.

**In plain words** (Megakaryocytes join the IAM Sequencing Atlas and its globe): The cells that make platelets, a large source of the DNA in blood plasma, are now in our sequencing reference and on its globe, as a provisional 58th cell type. Platelet DNA the chain had never read comes out nearest them of every cell. The 57 cells before are unchanged.
<!-- PLAIN:END -->

**Are BLUEPRINT's tracks plain read counts? (diagnostic, written before it runs)** (`seq_atlas/outside_data/blueprint_track_counts.py`;
outputs `blueprint_track_counts.csv`, `blueprint_track_counts.txt`). With the common floor, BLUEPRINT's sorted-cell Met-A still reads 0.7983:
its libraries show less disorder at the identity sites than their reads allow. Every sequencing reading counts disorder from the methylated
read count, which the chain takes as level x coverage from BLUEPRINT's two tracks. If the public level is smoothed, shrunk or a model estimate,
those are not counts and the libraries would read too ordered. The check, on chromosomes 21 and 22 of the first 8 libraries of
`blueprint_healthy_test_samples.csv` (spent): the share of CpGs whose level x coverage is a whole number, and the number of distinct levels at
coverage 10 (at most 11 for counts). Nothing is scored; it decides whether BLUEPRINT's tracks can be read as counts at all.

**BLUEPRINT's tracks, first run, and the rounding test (written before the second run).** First run, 8 libraries (`blueprint_track_counts.txt`, Toronto spot machine): coverage is a whole number at every CpG (median
share 1.0000); at coverage 10 the levels take exactly 11 values, as counts do; but level x coverage is within 0.01 of a whole number at only
0.6025 of CpGs (0.4867 to 0.7211), lower the deeper the library. That is what a count written as a level rounded to a fixed number of decimals
looks like. Second run, the first 3 libraries: for 2, 3 and 4 decimals, the share of levels on that grid and the share where the nearest
whole count gives the level back. If one precision gives the level back at nearly every CpG, the counts are recoverable and the tracks are
counts; nothing is scored.

**BLUEPRINT's tracks are read counts (second run, the rounding test).** (`blueprint_track_counts.txt`, 3 libraries; the first run is kept as `blueprint_track_counts_run1.txt`.) Every level lies on the
3-decimal grid (median share 1.0000; 0.2634 on the 2-decimal grid), and the nearest whole count, divided by coverage and written to 3 decimals,
gives the level back at 0.9978 of CpGs (0.9977 to 0.9995). The public levels are the methylated read count over coverage, written to 3
decimals, so level x coverage is within 0.0005 x coverage of the count: under 0.03 reads at these libraries' depths. The low own-cell Met-A on
BLUEPRINT's libraries is not an artifact of their tracks; the laboratory floor stays the measured cause. The chain is unchanged.

**BLUEPRINT's low own-cell Met-A is not the depth trend (diagnostic on committed outputs; nothing scored).** (`seq_atlas/outside_data/blueprint_depth_check.py`, output `blueprint_depth_check.txt`; written after a first look at the same committed
tables in a notebook, disclosed.) Inside the atlas's laboratory, sorted-cell Met-A falls 0.0066 per read of depth above its reference donors
(pass 4, 125 samples, r −0.762). BLUEPRINT's libraries are deeper (median 44.5 calls per site over the 38 read), but within BLUEPRINT Met-A
rises with depth (rho +0.378), and its B cells and CD8 T cells, at 37 and 34 calls per site, about the atlas's own depth, read 0.591 and 0.593.
Depth does not explain the gap. With the tracks shown to be counts and the floor only half of it, what remains is the laboratory's libraries
holding fewer off-pattern reads at the identity sites than the atlas's donors of the same cells.

## A second outside laboratory's healthy CD8 T cells (development; written and committed before any is read)

**The instrument's job:** a healthy sample sequenced and processed by another laboratory reads Normal on the gauge.

**Why now.** The first outside laboratory (BLUEPRINT) read low (own-cell Met-A 0.7134). Its tracks are read counts, depth does not explain it, and
the common floor explains only part. A second laboratory tells whether that is BLUEPRINT's or every outside laboratory's. It is read as
development, so the cleaner healthy set found (GSE144691, healthy donors' CD8 subsets) stays unread for the real test.

**Data.** GSE245832 (`seq_atlas/outside_data/lab2_cd8_samples.csv`): 4 CMV-seropositive donors, each with naive CD8 T cells (read as naive CD8 T
cells), memory CD8 T cells (CCR7-negative, CD45RA-negative: read as effector memory CD8 T cells) and CMV-specific IFN-gamma-positive CD8 T cells,
sorted after a 4-hour peptide stimulation; whole-genome bisulfite, GRCh38 Bismark coverage files (counts). None read before.
**Prior reading:** none; the series' sample labels and file list only.

**Read by** `seq_atlas/outside_data/lab2_cd8_read.sh`: each file written as .lbeta on the hg19 numbering by the one command (`--build seq-table`,
bismark-cov, hg38), then read by it with the first outside test's flags (`--dev-seq-values 1 --full-chain --dev-seq-ref auto --dev-seq-level donors
--dev-terr-counted --dev-metaa-sites`, specimen "sorted CELL"). Scored by `seq_atlas/outside_data/lab2_cd8_score.py` (outputs `lab2_cd8_test.csv`,
`lab2_cd8_test.txt`).

**Bars (the first outside test's three, unchanged),** on the 8 naive and memory CD8 samples: 1. own-cell Met-A median within 0.95 to 1.05; 2. at
least 75 % of them within 0.95 to 1.05; 3. whole-sample Met-A median within 0.95 to 1.05. Recorded, not scored: the islands, the domains, the
C-Score, coupling-A, IAM-A is not read (no molecules), the own-cell share, each library's floor, and the 4 CMV-specific responders (cells answering
the stimulation). The stimulation lasted 4 hours, flagged with every reading.

**Synthetic test:** `DEV_ATLAS_V21_01/seq_atlas/seq_table_synthetic.py`, `DEV_ATLAS_V21_01/seq_atlas/seq_table_synthetic.txt` (the table conversion, met); the readings are the
chain's, each with its own synthetic on record.

**A second outside laboratory's healthy CD8 T cells, read: not met** (`seq_atlas/outside_data/lab2_cd8_test.txt`, `lab2_cd8_test.csv`). Read on 3 Toronto spot machines, about 10 minutes a library; 9 of 12
read (donor 1's three libraries cover 5,215 to 5,367 of the 6,000 identity sites, under the 90 % rule). Own-cell Met-A median
0.5740 over the 6 scored naive and memory CD8 libraries read (0.5257 to 0.6927; bar 1 not met; none
within 0.95 to 1.05, bar 2 not met); whole-sample Met-A median 1.0258 over the 8 scored (bar 3 met).
The CMV-specific responders (recorded) read 0.5754 to 0.6315. The Deconvolver gives the read naive CD8 libraries 0.54 to 0.63 of their own cell and
memory CD8 libraries 0.08 to 0.18 (the libraries read; their sort is broader than the atlas's effector-memory column). Like BLUEPRINT (0.7134), and lower:
own-cell Met-A on sequencing reads low on both outside laboratories, while the atlas's own laboratory's healthy donors, read against each
other, read 1.0077 (pass 5). It is not one laboratory's fault. **Next:** which side of the identity sites carries the gap (the marked sites'
unmethylated reads, or the unmarked sites' methylated reads), from the chain's per-site tables, beside each library's own error rates at the
sites every cell holds the same.

**Why own-cell Met-A reads low on the outside laboratory (diagnostic; nothing scored).** (`seq_atlas/outside_data/own_cell_gap.py` on the chain's own per-site tables `lab2_sites/`, output `own_cell_gap_lab2.txt`; written
after a first look at the same tables by band of reference level, disclosed.) At the sample's own cell's identity sites the outside libraries
read more extreme than the atlas's level on both sides: where the atlas's donors read the cell at 0.7 to 0.9, these libraries read it 0.069
higher; at 0.9 to 0.95, 0.027 higher; below 0.1, 0.002 lower. A share of other cells' DNA in the atlas's sorts would make the gap grow with how
far the other cells sit from this one at each site; it does not (slopes of the gap on that contrast are negative in every band, -0.02 to -0.10,
and at the 0.7 to 0.9 band the other cells are nearly as methylated as this one, contrast +0.014). Ruled out. What remains is a laboratory
difference tied to the site's level: sites the atlas's laboratory reads as partly unmethylated in this cell, another laboratory reads as nearly
fully methylated. Inside one laboratory it cancels (pass 5, 1.0077); across laboratories it does not. **What it means for commissioning:** own-cell
Met-A on sequencing from another laboratory needs each site's laboratory term, measured from healthy cells of the same type sequenced by other
laboratories, as IAM-Atlas v2.1 measured each source's offset in its calibration stage. The spent outside sets (BLUEPRINT, GSE245832) can
measure it; a fresh laboratory (GSE144691) then tests it. Whole-sample Met-A, which reads the sample's own mix at every atlas site, already reads
Normal on both outside laboratories (1.0258 here).

**BLUEPRINT read again for its per-site tables (development, spent; written before it runs).** To measure the laboratory term at the identity sites (above), the laboratories' own per-site tables are needed beside the atlas's. The first
BLUEPRINT read kept bundles only. The same 59 spent libraries (`blueprint_healthy_test_samples.csv`) are read again by the one command, each with
its row's own flags plus `--dev-metaa-sites` (`seq_atlas/outside_data/blueprint_sites_read.sh`, 7 Toronto spot machines); the readings must match
the first read exactly (a check of the chain), and the tables go to `blueprint_sites/`. Nothing is scored.

**The same-laboratory tare on sequencing, measured on spent readings (written and committed before it runs; development, nothing commissioned).** Three outside laboratories shift own-cell Met-A differently (BLUEPRINT low, the second laboratory low, GSE134664 high, much of it from
depth). A tare from the same laboratory's own healthy references (`stage_metaa_atlas.tare`, the rule the arrays already use) removes a
laboratory's offset only if the offset is one factor for that laboratory and cell. `seq_atlas/outside_data/same_lab_tare.py` measures (1) the
spread among one laboratory's healthy donors of a cell, in the atlas's laboratory, BLUEPRINT and the second laboratory, and (2) every sample
tared on 3, 5 or 8 others from its own laboratory and cell: the tared median, spread and share inside 0.95 to 1.05. Nothing is scored against
a bar: it sets the number of references the real test on GSE144691's five new healthy libraries will use, and that test's bar will be the
atlas laboratory's own tared reading.
**Prior reading:** GSE144691's labels and read counts only (its eight healthy libraries shared with GSE98837 are spent); none of its five new
healthy libraries has been read.

**The same-laboratory tare, measured.** (`seq_atlas/outside_data/same_lab_tare.txt`, `same_lab_tare.csv`.) (1) The spread among one laboratory's healthy donors of a cell is about
the same in every laboratory: the atlas's 0.0620 (42 cells, 126 samples), BLUEPRINT 0.0744 (4 cells, 32 samples), the second laboratory 0.0492
(2 cells, 9 samples), so each laboratory's offset is close to one factor for the laboratory and cell. (2) Tared on its own laboratory's healthy
references, the offset is gone: BLUEPRINT 0.7134 untared, tared median 1.0001 (3 references), 0.9999 (5), 0.9991 (8); the second laboratory
0.5891 untared, 0.9889 (3), 1.0008 (5, only 6 draws). The tared spread (0.076 to 0.099) is mostly the donors' own spread, so more references
change it little; one tared healthy sample falls inside 0.95 to 1.05 in 33 to 52 % of draws. **Learned:** on sequencing a single sample's
sorted-cell Met-A spreads 6 to 7 % even inside the atlas's laboratory, wider than the gauge's Normal band of 5 % either side, so the real test
on GSE144691's five new healthy libraries is judged on their tared median, as healthy pass 5 was, with the per-sample band from the atlas
laboratory's own tared spread. Development; nothing commissioned.

**The healthy test on the Johns Hopkins laboratory: CD4 T cells, tared on their own laboratory (written and committed before any is read).** The instrument's job: a healthy sample from a laboratory the IAM Sequencing Atlas has never seen, tared on that laboratory's own healthy
samples of the same cell, reads Normal. GSE86340 (Johns Hopkins, the laboratory of the leukemia study already read) holds whole-genome
bisulfite sequencing of CD4 T cells from six healthy donors aged 18 to 86 (`seq_atlas/outside_data/jhu_cd4_samples.csv`: 271 to 493 million
read pairs each, two runs each). Every read pair is aligned by the committed pipeline in chunks of about 25 million pairs on Google spot
machines (`jhu_cd4_google.sh`, the same steps as `split_align.yml`), merged and read once by the one command, after the genome-coverage
intake check. Each donor's own-cell Met-A is tared on the other five. Bars (`jhu_cd4_score.py`, outputs `jhu_cd4_test.csv`,
`jhu_cd4_test.txt`): 1. the tared own-cell Met-A median within 0.95 to 1.05; 2. the whole-sample Met-A median within 0.95 to 1.05. Bar 1 is
judged on the median, as healthy pass 5 was, because one healthy sample's sorted-cell Met-A spreads 6 to 7 % on sequencing (the
same-laboratory tare, above). If met, the same-laboratory tare is commissioned for own-cell Met-A on sequencing, for CD4 T cells.
**Prior reading:** none of GSE86340's samples has been read; only its sample labels and read counts.
**Synthetic test:** `DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/metaa_donors_synthetic.py`, `DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/metaa_donors_synthetic.txt` (the sorted-cell reading on sequencing, met). The tare itself was measured on the spent laboratories (`seq_atlas/outside_data/same_lab_tare.txt`, above).

**The sequencing atlas's information sphere carries IAM-Atlas v2.1's caption (found by the author; disclosed).** The picture (`seq_atlas/globe/seq_globe.html`) draws the IAM Sequencing Atlas's 534617 sites from 56 cell types (`seq_globe.json`),
but its "What you are looking at" panel says 191,458 sites and 79 cell types of IAM-Atlas v2.1: the one command's sphere builder writes that
panel as fixed text (`run_sample.py`, `_ISV_ABOUT`), whatever atlas it is given. The website also linked the sequencing atlas's box to this page as
"its information globe" without a choice of atlas, and gave IAM-Atlas v2.1's box no sphere link. Fixed on the site: one information sphere page
with a choice of atlas, each box linked to its own, the counts read from each build record, and a note on the panel's wording. To do (after the B
and T and Johns Hopkins readings, one chain version): the builder writes its panel from the atlas given, and the sphere is rebuilt with all 57
cell types (it was drawn before the granulocytes were added).

**Correction: the sequencing atlas's globe is not at every CpG (found while checking the book; disclosed).** The sky-sphere build above (`--build seq-atlas-sphere`) used IAM-Atlas v2.1's builder and layout: 813951 sites, the same places as the
array atlas (`website/art/out/seq_cells/index.json`: sites 813951, NSIDE 256). The checklist row C1 asks for every CpG of the genome on the
sphere at a resolution fine enough for the whole genome; that is not built. The atlas itself holds all 28,217,448 CpGs. Public text that said
the globe or its cell pictures show every CpG is corrected; the planned-against-built table marks C1 partly built. A whole-genome globe
(28.2 million CpGs need a finer HEALPix resolution; NSIDE 2048 has 50.3 million pixels) is plan item 60.23.

**The Johns Hopkins tumor-against-own-normal test (written and committed before any of its tissue samples is read).** The other half of the same-laboratory test. The instrument's job: a person's tumor, read against that same person's normal tissue from
the same laboratory, reads abnormal. Samples (`seq_atlas/outside_data/jhu_tissue_samples.csv`, from GSE86340's labels and ENA's run list):
five people with tumor and normal tissue (three liver, two lung), 165 to 299 million read pairs each, plus one lung tumor and one lung normal
whose labels do not say they are one person (recorded, not scored); 12 samples, 114 chunks, every read pair of every run, aligned on Google
(`DEV_ERASE_01/split_google.sh`, the same steps as `split_align.yml`), merged and read once by the one command after the genome-coverage check,
with the flags the CD4 test uses. Self-reference: each reading of a tumor divided by the same reading of the person's own normal tissue
(`stage_metaa_atlas.tare`'s rule with one reference). No direction is assumed. Bars (`jhu_tissue_score.py`, outputs `jhu_tissue_test.csv`,
`jhu_tissue_test_pairs.csv`, `jhu_tissue_test.txt`): 1. whole-sample Met-A outside 0.95 to 1.05 in at least 4 of the 5 labeled pairs; 2. the
islands outside 0.95 to 1.05 in at least 4 of the 5. Recorded, not scored: the domains, the C-Score, coupling-A and IAM-A against own normal, each
tissue's untared readings against the IAM Sequencing Atlas (normal tissue beside a tumor may carry a field change), the cell shares and the
unlabeled lung pair. No new chain code: every reading used here has its own synthetic on record. **Prior reading:** GSE86340's labels and run
sizes only; none of these 12 tissue samples has been read.

**The megakaryocyte column tested on platelet DNA (plan 60.24; written before the platelet file is opened).** The column (built from single-cell donors 1 and 2) was not met on donor 3 by plain distance and reads donor 3 nearest
megakaryocytes once counting noise is weighed, but donor 3 is spent, so the column needs a fresh test. GSE206818 is platelet whole-genome
bisulfite sequencing from 3 healthy people, published only as one merged level track on hg19 (`GSE206818_platelets_merged.bedGraph.gz`; the
series states the raw reads are not public); platelet DNA is mostly megakaryocyte DNA. Unread: only its column layout and GEO notes were read.

**The instrument's job:** platelet DNA reads as megakaryocytes against the atlas.

**Sources:** GSE206818 (public, GEO: the merged platelet level track, hg19); the IAM Sequencing Atlas (public, Zenodo 10.5281/zenodo.23934666);
the megakaryocyte column (pieces built from GSE87197 donors 1 and 2, public, GEO).

**Chain change (development):** the table converter takes a level-only track (`stage_seq_atlas.table_to_lbeta`, format `level-bedgraph`):
each position is written as LEVEL_N = 1000 reads (stated), so the comparison reads it as a deep sample with negligible counting scatter of its
own; flagged. **Synthetic first** (`seq_atlas/mk_platelet_synthetic.py`, run by `mk_platelet_synthetic.sh` on GitHub): a constructed atlas of 12
deep cells and one shallow cell (2 donors at about 2 reads, 40 % of CpGs, the megakaryocyte column's case); a level track pooled from 3 people of
the shallow cell, and one of a deep cell, each converted and compared by the one command (`--build seq-table`, `--build seq-compare`). Bars: each
sample nearest its own cell by the counting-weighted score.

**Bar (the test):** megakaryocytes is the platelets' nearest cell of the 57 by the counting-weighted score. Recorded: the plain-distance rank, the
next cells and their scores. **If met:** the column joins the in-mix readings, flagged provisional; plasma's megakaryocyte DNA is then read
against it instead of the four stand-in cells. **Replication:** the megakaryocyte pieces are deposited on Zenodo with the test.

**The platelet test's synthetic: met** (`seq_atlas/mk_platelet_synthetic.txt`, GitHub run 42334515979). The level-only track of the shallow cell came out nearest its own column by the counting-weighted score (0.8521; the next cell 17.7702), and the deep cell's track nearest its own (0.224); both also first by plain distance. The test on the platelet file runs next, on a Google machine (the megakaryocyte pieces are in the build bucket).

**The megakaryocyte column on platelet DNA: met.** (`seq_atlas/mk_platelet_compare.json`; Google machine iam-mk-platelet, chain []001070a; the platelet track
sha256 8f94ec5a6cbbd965..., 27,187,351 positions all on hg19 CpGs, 24,546,329 compared.) By the counting-weighted score the platelets are
nearest **megakaryocytes** of all 57 cells (1.1186), then nk cells (2.9281), aorta endothelium (3.3828), lung interstitial macrophages (3.6383) and
monocytes (3.651). Recorded: by plain distance megakaryocytes rank 34 of 57 (0.14724;
first monocytes, 0.10462), as the synthetic showed for a shallow column: plain distance counts the column's own counting scatter
as difference. **Next, as written:** the column joins the in-mix readings, flagged provisional, and plasma's megakaryocyte DNA is read against
it instead of the four stand-in cells; its pieces go to Zenodo with this test.

**The megakaryocyte column into the published atlas and globe (author's yes to publish both new versions).** The 57 published cells are not touched: the atlas's version 1.1 (Zenodo draft, reserved DOI 10.5281/zenodo.24029328) and the globe's new version (reserved DOI 10.5281/zenodo.24029332) carry every file of the version before, and add the column: its 29 pieces with a SHA256 manifest, its inputs (`seq_atlas/mk_inputs.csv`, the granulocyte column's form), build list, conversion reports and the platelet test; on the globe, its slices at NSIDE 2048, its Globe Viewer texture and both its noise maps with their nulls, built from those same pieces after checking them against the manifest. Job: `deposit/zenodo_mk_v11.sh` and `deposit/zenodo_mk_v11.py` on one Google machine, saving each output as it goes; the laptop publishes once every file is in each draft. READMEs: `deposit/zenodo_mk_v11_README_atlas.md`, `deposit/zenodo_mk_v11_README_globe.md`.

**Correction (disclosed): the platelet synthetic's shallow cell is not the megakaryocyte column's case.** The synthetic's record calls its shallow cell (2 donors at about 2 reads, 40 % of CpGs) "the megakaryocyte column's case". Measured from the column's 29 build reports (`piece_*.json` in `pieces_megakaryocytes_29.tar`): 25,220,009 of 28,217,448 CpGs hold reads (89.4 %), each piece's median 1 to 5 reads (median of pieces 4). The synthetic was harder than the real column, not easier, so its pass still covers the test; the wording was mine and wrong, and I repeated it to the author before checking it. The atlas README states the measured figures.

**The Johns Hopkins healthy CD4 test, read: not met** (`seq_atlas/outside_data/jhu_cd4_test.txt`, `jhu_cd4_test.csv`; all 6 donors read by the one command at full depth on Google, scored by the
committed scorer.) Bar 1 met: the tared own-cell Met-A median 1.0026. Bar 2 not met: the whole-sample Met-A median 1.1435 (untared, as the bar
was written). **What the rows show, recorded, not scored:** the laboratory's own-cell offset is small (untared median 0.9782, against 0.71 and
0.57 at the two outside laboratories read before), but no single donor's tared own-cell reading falls within 0.95 to 1.05: the three young
donors (18, 25 and 25 years) read 0.8952, 0.8843 and 0.9302, the three old donors (82, 82 and 86) 1.0877, 1.0751 and 1.1266. Every young donor
reads below every old one, in own-cell Met-A, whole-sample Met-A and the domains (Spearman with age 0.883 each, 6 donors). Tared on donors
whose ages straddle 18 to 86, each donor reads its age's distance from the middle; the median meeting bar 1 says nothing about any one donor.
The Deconvolver's CD4 T-cell share of these sorted CD4 cells is 0.000 to 0.408 (the rest falls to the atlas's other T-cell columns),
recorded. GSE86340's healthy CD4 libraries are spent. **Not concluded:** whether age is the cause (6 donors, two age groups).

**The Johns Hopkins CD4 donors' cell shares (development, after scoring; disclosed).** Read from each donor's bundle (the Deconvolver's shares, `cell_shares`). GSM2103004 (25 years), labeled CD4 T cells, reads 0.641 monocytes and 0.203 lung interstitial macrophages, with no T-cell column above 0.03: by its own methylation it is not a CD4 T-cell sample, and it is one of the three young readings. The other five read as CD4 T cells split across the atlas's CD4 columns; the three old donors hold less naive CD4 (0.000 to 0.104, against 0.179 and 0.197 in the two other young) and more effector-memory CD4 (0.109 to 0.194, against 0.031 and 0.081), the shift the immune system makes with age. So the spread the test found is at least partly what cells each sample holds, not the laboratory and not shown to be age itself. In the atlas laboratory's own healthy pass (pass 7, 123 donors of 41 cells, aged 17 to 83, `donor_traits.csv`), own-cell Met-A tared within each cell on its other donors shows no trend with age (Spearman +0.053, p 0.56); a weak check, as most donors are within 5 years of their cell's median age. **Next, each written before it runs:** GSM2103004's labels and quality; and each sample's own-cell reading against the CD4 subsets its own shares say it holds.

**A cross-laboratory test on the same cell (design, author: the CD4 test 'werent even comparing apples to apples'; the test itself is written after its rehearsal, before any fresh sample is read).** The Johns Hopkins CD4 test could not tell a laboratory from a cell difference: one sample is not CD4 T cells by
its own methylation, and the old donors hold other CD4 subsets than the young. The next test is built so it can.

**The instrument's job:** a healthy sample of one cell type, from a laboratory the atlas has never read, reads Normal against the atlas, person
by person.

1. **The same cell, checked by the chain:** at intake each sample's own Deconvolver shares must show the labeled cell (the share of its cell
   family at or above a level set on the rehearsal, before any fresh sample is read); a sample below it is refused with the reason, not read.
2. **A cell that does not change its make-up with age:** monocytes (CD14, one lineage, sorted the same way in every laboratory), not T cells.
3. **Age recorded beside every reading**, and each laboratory's donors compared with atlas donors of similar age.
4. **Normal's edges measured first:** how far healthy people of one cell differ on sequencing inside the atlas's own laboratory; the bar is per
   person, never a median.
5. **Each library's instrument terms measured from its own data** (the false-call rate at the library-floor sites, conversion, copies, depth):
   the laboratory is corrected from what its library shows, never by a fitted laboratory constant.

**Sources (labels and file layouts only, nothing read):** BLUEPRINT (`outside_data/blueprint_healthy_candidates.csv`): 22 monocytes never read
(16 monocytes, 6 classical monocytes; ages 0-5 to 70-75); GSE173787 / PRJNA727170: 10 healthy CD14 monocytes (HC-1 to HC-10; also 12 from
multiple sclerosis); PRJNA574550: 51 monocyte runs, labels to check. **Rehearsal (spent data, answer known):** BLUEPRINT's 10 spent mature
neutrophils against the atlas's granulocyte column, with steps 1 to 5, before the test is written.

**What the chain itself reported on every outside laboratory read so far (development, after the fact; nothing read again, nothing scored).** (`outside_data/crosslab_chain_table.csv`, `crosslab_chain_table.py`; 89 samples from 5 laboratories, each already read
by the one command.) The scoring of those tests read each sample against the cell on its label; the chain's own Deconvolver reports otherwise:
the cell it finds most of holds only 24 to 80 % of a "sorted" sample, the rest in neighboring subsets, and one Johns Hopkins "CD4" sample is mostly
monocytes. Per laboratory (medians): BLUEPRINT false-call rate 0.0016 (0.25 times the atlas laboratory's 0.0063, `input_floors_v1.parquet`),
41 calls a site, Met-A of the cell found 0.71; Johns Hopkins 0.0051, 17, 0.94; GSE245832 0.0059, 3, 0.73; GSE134664 0.0092, 8, 1.29; GSE98837
0.0158, 15, 0.87. No single library term orders the laboratories (false-call rate and depth each partial). **Next (each written before it runs, synthetic first):** read each sample
against the atlas in the shares the chain measured, with each library's own false-call rate and depth carried into its expectation, and test it
on these 89 (spent, answer known, the per-person bar); only then a fresh laboratory, compared only with samples the Deconvolver matches.

**The instrument's job:** a healthy sample from any laboratory reads Normal against the atlas in its own measured cell shares, person by person.

**Sources:** the 89 bundles the one command wrote (GEO GSE98837, GSE134664, GSE245832, GSE86340 and BLUEPRINT, all public); the IAM Sequencing Atlas (public, Zenodo).

**Range of the data:** real libraries: false-call rate 0.001 to 0.033, 2 to 71 calls a site, top-cell share 0.24 to 0.80 (`crosslab_chain_table.csv`);
synthetic to try: false-call rate 0.0005 to 0.05, 1 to 100 calls a site, top-cell share 0.2 to 1.0.

**Where a label got into the chain, and the fix (the Deconvolver decides; author: "find where the breakdown happened and make sure it can NEVER happen again using code").** Found in the chain's own code, not only in the scorers. (1) `run_sample.py` set the
development sorted-cell Met-A's cell from the specimen's words (from 2026-10-22); (2) Stage Q2 took IAM-A's and coupling-A's cell from the
specimen (from 10-25); (3) a 'sorted X' sample's mix was overridden to 100 % X for the whole-sample reference, the departures and the C-Score
(from 10-28); (4) `run_series.py` let a sample list's `atlas_cell` column set the cell (from 10-25); and the cross-laboratory scorers read
(1) and compared samples by the sample list's label. The chain's gate checked that every reading came from the one command, but not who chose
the cell. **The fix (chain code):** `stage_0_intake.LABEL_RULE`, `cells_found`, `chain_cell`, `label_check`; the four paths removed; the
Deconvolver runs first on every sequencing file (its values are made first); every single-cell reading takes the cell the Deconvolver found
most of; a label it disagrees with is written to the bundle (`label_check`), and a cell set by hand is written as the operator's
(`operator_cell`). **So it cannot return:** `CANON/chain_gate.py` rule 23b-h refuses chain code that sets a cell or a mix from the specimen's
words and any scorer that reads the label's cell without `label_check` (control cases C17 to C21; the old `run_sample.py` and the Johns Hopkins
and B and T scorers are refused by it). The lesson is written in the chain itself: the Decoder's section "The Deconvolver decides what is in
a sample, in every instrument", `stage_0_intake.py`'s header and the workflow's boxes A2 and E1.

**The instrument's job:** a sample is read as what the Deconvolver finds in it, whatever its label says.

**Synthetic test:** `DEV_ATLAS_V21_01/label_rule/label_synthetic.py` (four constructed arrays, each labeled "sorted cd4 t cells", through the
one command: all CD4; 50 % CD4 and 50 % monocytes; 24 % CD4, 40 % monocytes and 36 % NK cells; all monocytes). Bars: 1. the monocyte sample is
read as monocytes and the label's disagreement recorded; 2. the CD4 sample is read as CD4; 3. in both mixes every cell found at 1 % or more has
its own Met-A, and the single-cell reading takes the cell found most of.

**Sources:** IAM-Atlas v2.1's blood columns (public, Zenodo), through the chain's own reference (`stage_deconv_v21.reference`).

**Range of the data:** real outside "sorted" libraries: the labeled cell's measured share 0.24 to 0.80, and 0.00 in the Johns Hopkins sample
read as monocytes (`seq_atlas/outside_data/crosslab_chain_table.csv`); synthetic: the labeled cell at 1.00, 0.50, 0.24 and 0.00.

**Cross-laboratory: what separates the laboratories, from the chain's own numbers (development, diagnostic; nothing scored, nothing re-read).** Read from what the one command already wrote for the 89 outside samples
(`seq_atlas/outside_data/crosslab_chain_table.csv`) and, for the 18 that wrote per-site tables, by the committed diagnostic
`floor_by_site_class.py` (`crosslab_floor_by_class.txt`). **The islands follow each library's own floor:** across all 89 the islands
reading against the library's false-call rate at the floor sites, rho +0.954 (BLUEPRINT, floor 0.0016: 0.31; GSE98837, floor 0.0158: 1.37).
In the outside laboratories the islands take the whole floor (median share 1.001, rho +0.957, 18 libraries), as healthy plasma did (1.005);
inside the atlas's own laboratory they take 0.291 (11-03). So an outside library's false calls reach the islands at its own measured rate,
and the atlas's reference there holds its own laboratory's smaller share: the islands read the difference of two laboratories' chemistry,
not of two people. This is the step written on 11-03 and never run: the floor read where a reading reads, at the unmarked sites of the
reading's own class. **The whole sample and the domains do not follow the floor** (whole rho +0.41, domains against depth -0.09): by
laboratory, whole-sample Met-A in the measured mix reads 0.945, 1.016, 1.101, 1.143 and 1.328, within 0.95 to 1.05 for 30 % of the 89.
The domains are not unmarked sites; the two laboratories reading high there (Johns Hopkins CD4, whose old donors hold more memory CD4, and
GSE134664 B) are the ones whose Deconvolver mixes differ most from the atlas's donors, so the domains are read next against age and the
measured subsets, before anything is called a laboratory term.

**The floor counted at the islands read (written and committed before it runs; plan 60.3 step 2).** The step written on 11-03. **The instrument's job:** a healthy sample from another laboratory reads
Normal at the islands. **The physics:** a library calls an unmethylated read methylated with its own probability; at the islands a reading
reads, the sample's rate and the reference's are each counted where the sample's own measured mix expects the islands unmarked (pooled
reference level at most 1 %), and both are read at the higher of the two by the exact floor raise of 11-02 (`floor_raise`, `_Hj`). Nothing is
fitted: both floors are counts at the very sites read. **The change** (`--dev-floor-class`, `stage_metaa_atlas.FLOOR_CLASS`, development, off
by default; the Johns Hopkins tissue readings are pinned to the chain before it). **Synthetic test:**
`seq_atlas/outside_data/floor_class_synthetic.py` (one command, Google, `floor_class_synthetic.sh`): the reference passes 0.29 of its floor
to the islands, as the atlas laboratory does; five outside laboratories of three people at floor and share (0.0015, 1.0), (0.005, 1.05),
(0.006, 1.0), (0.009, 1.05), (0.016, 0.75), plus the atlas laboratory's own case (0.006, 0.29); each read with no floor, the common floor
and the class floor. Bars, class floor: 1. every laboratory's median islands within 0.03 of 1; 2. the atlas laboratory's own case within 0.03 of 1.

**Sources:** the IAM Sequencing Atlas (public, Zenodo 10.5281/zenodo.24029328) and its spent healthy input GSM5652316 (public, GEO) as the base.

**Range of the data:** real: library floors 0.0016 to 0.0158 and the share of the floor at the islands 0.29 (atlas laboratory) and 0.75 to
1.05 (outside), `crosslab_chain_table.csv`, `crosslab_floor_by_class.txt`; synthetic: floors 0.0015 to 0.016, shares 0.29 and 0.75 to 1.05.

**The Deconvolver decides: synthetic met** (`label_rule/label_synthetic.txt`, GitHub run 42358846755, the one command on the fixed chain). Four arrays
labeled "sorted cd4 t cells": built as all CD4, found CD4 0.9925, read as CD4, the label agrees; 50 % CD4 and 50 % monocytes, found 0.5017 and
0.4981, both read, the sorted-cell reading on monocytes, the label flagged; 24 % CD4, 40 % monocytes and 36 % NK cells, found 0.2338, 0.4003
and 0.3580, all three read; all monocytes, found 0.9965, read as monocytes, the label flagged. All three bars met.

**Cross-laboratory: the domains read the naive or memory state of the cells (development, diagnostic, from the chain's own readings; nothing scored).** Inside BLUEPRINT (59 libraries, one laboratory, so no laboratory
term), the domains do not follow age (rho +0.145, p 0.27) but the cell's history: naive-type CD8 0.665, naive B 0.759 to 0.782, naive-type CD4
0.772, central and effector memory CD4 0.955 to 0.993, effector memory CD8 0.998, memory B 0.984, class-switched memory B 1.155, classical
monocytes 1.111; the cord-blood donors (0 to 5 years) read 0.759. Naive and memory cells are different cell states, each with its own
healthy level in the domains, so a sample whose naive-to-memory make-up differs from its reference's reads off 1 there: a mismatch of
cells, not of laboratories (no mechanism claimed; the held balance is independent of division, rule 25). Against
the Deconvolver's own measured memory share (memory over memory plus naive columns, 84 samples): rho +0.254 (p 0.02) for the domains and
+0.378 (p 0.0004) for the whole sample; within the Johns Hopkins CD4 donors +0.943, within BLUEPRINT +0.260, within GSE245832 -0.717. The
Sequencing Atlas's subset columns rest on one or two donors (naive CD4 1, naive CD8 1, effector memory CD8 1, memory B 2), and the
Deconvolver gives most BLUEPRINT memory libraries almost no memory share, so the split of naive from memory is the open part. **Next:** the
Deconvolver's naive and memory split tested on constructed mixes of those columns (synthetic first, over the real share range), then more
donors for the subset columns from the held leads; until then the domains and the whole sample of lymphocytes are reported with the
measured memory share beside them, never called a laboratory term.

**Can the Deconvolver tell naive from memory CD4 on sequencing? (written and committed before it runs).** The open part of the domains finding above.

**The instrument's job:** the Deconvolver reports the naive and memory CD4 shares a sample holds.

**Synthetic test:** `DEV_ATLAS_V21_01/label_rule/subset_synthetic.py` (it runs the one command on each constructed sample; one grid row per GitHub job through cov_read.yml; 30 rows, its grid
listed in the script's folder): constructed sequenced samples from the IAM Sequencing Atlas's own columns at IAM-Atlas v2.1's sites, CD4 at 0.3 or 0.7 of
the sample split between naive and memory CD4 at memory fraction 0 to 1, each site varied as a person, at 3, 15 or 40 reads a site. Bars
(`subset_synthetic_score.py`): 1. at 15 and 40 reads, the measured memory fraction within 0.15 of the planted in every row; 2. the measured
total CD4 within 0.1. Three reads a site recorded, where it breaks.

**Sources:** the IAM Sequencing Atlas (public, Zenodo 10.5281/zenodo.24029328), its columns as committed in the chain's runtime tables.

**Range of the data:** real: outside libraries 3 to 71 calls a site, the labeled cell 0.24 to 0.80 of the sample
(`crosslab_chain_table.csv`); synthetic: 3, 15 and 40 reads a site, CD4 at 0.3 and 0.7, memory fraction 0 to 1.

**The Johns Hopkins tumor-against-own-normal test: specimen corrected, all 12 merged again (disclosed; my error).** The sample list passed each sample's GEO
title as the specimen ("liver tumor", "lung normal"). The chain's report refuses to print a status word ("tumor"), so all five tumor merges
stopped at the report (exit 35) after their reads were counted; the dispatcher dropped each after three tries, and its log did not say so.
Under the label rule the specimen names only the route, never the status, so every sample is now read with its organ only ("liver tissue",
"lung tissue"), tumor and normal alike, the chain still pinned to the commit before the label change; tumor and normal stay in the list for
the scorer only. The five normals read before with the status in the specimen are kept aside in the bucket (out_specimen_label/) and read
again the same way as the tumors, so both arms are read identically. Each merge now saves its merged reads to the bucket the moment they
exist and takes them back after a reclaim, and keeps the chain's own log; the dispatcher logs every job it gives up on and every sample
stuck with all chunks aligned, and the machine check prints both as failures. First result: GSM2103009 (liver tumor) read, exit 0.

**Naive and memory CD4 on sequencing: synthetic met** (`label_rule/subset_synthetic.txt`; 30 of 30 rows read by the one command on Google, one row a
machine). At 15 and 40 reads a site the measured memory fraction is within 0.15 of the planted in every row (largest gap 0.068) and the total
CD4 within 0.1 (largest gap 0.043); at 3 reads a site most rows still hold (largest gaps 0.25 and 0.118, recorded). So on constructed samples
built from the atlas's own subset columns the Deconvolver separates naive from memory CD4. What it cannot settle is whether those columns,
one donor each, stand for other people's naive and memory cells: the BLUEPRINT memory libraries read almost no memory share. That is a
question about the columns' donors, read next on the spent outside samples before any fresh one.

**The Johns Hopkins tumor-against-own-normal test, read: not met** (`seq_atlas/outside_data/jhu_tissue_test.txt`, `jhu_tissue_test.csv`, `jhu_tissue_test_pairs.csv`; all 12 read by the
one command at full depth on Google, every sample's specimen its organ only, scored by the committed scorer.) Bar 1 not met: whole-sample Met-A,
tumor against own normal, outside 0.95 to 1.05 in 2 of the 5 people (0.8422, 0.9983, 0.9407, 0.9963, 1.0443). Bar 2 met: the islands outside the
band in 4 of 5 (1.1388, 2.0477, 1.1871, 1.4060; the fifth 1.0278). **Recorded, not scored:** IAM-A against own normal below 0.95 in 4 of 5
(0.9284, 0.8843, 1.0130, 0.8678, 0.9001); the domains below 0.95 in the three liver pairs (0.7688, 0.9133, 0.8888); the C-Score within 0.99 to
1.01 in all five; the unlabeled lung pair whole 1.1268, islands 1.1747. The tissues' untared whole-sample Met-A against the IAM Sequencing
Atlas reads 1.20 to 1.36 in 11 of 12, normal and tumor alike; indiv1's liver tumor (GSM2103009) reads 1.0416 (recorded; see the correction below). So, read against the same
person's own normal tissue from the same laboratory, the islands and IAM-A move in most tumors and the whole-sample Met-A does not.

**Correction to the tumor-against-own-normal test: the "normal" is not healthy tissue (author; disclosed).** GSE86340 labels these samples only "normal" from the same individual as
the tumor (for example "indiv1-liver-normal"; "cancer / normal: normal"); where in the organ they were taken is not stated, and the sample
list's "adjacent normal" was my wording, not the series'. They are a cancer patient's non-tumor tissue, not healthy tissue, so the test above
read each tumor against the same person's non-tumor tissue, a weaker reference than healthy: any change the non-tumor tissue already
carries narrows the gap. Its result stands as recorded under that name. The earlier line that the atlas has no liver or lung column was wrong:
the IAM Sequencing Atlas holds hepatocytes (3 donors), lung alveolar epithelium, bronchus epithelium and alveolar endothelium (3 each) and two
lung macrophage types, and the chain read every tissue against them in the Deconvolver's measured mix; against that healthy reference the tumors
and their non-tumor tissue read 1.20 to 1.36 (11 of 12; indiv1's liver tumor 1.0416), which is this laboratory's offset (its healthy CD4 donors read 1.14 whole-sample) or a
change in the non-tumor tissue, not yet told apart. **Next (its design, synthetic and range of the data recorded first):** both tissues of each person read
against the IAM Sequencing Atlas's healthy liver and lung cells only (author: the healthy reference for every comparison comes from the
atlases, never from a sample labeled healthy or normal); each library's own chemistry measured from its own reads at the sites read (the
class floor), never from another laboratory's or person's samples, so the non-tumor side is tested for a change rather than assumed healthy.

**The class-floor synthetic: run in 18 parts (disclosed).** Its first run went to one machine, which stopped at its first step for six hours (a variable the machine never
defines; fixed). Relaunched the same way, the 54 runs of the one command (6 laboratory settings, 3 people, 3 floor settings) in series would take
about as long as the machine's time limit and saved nothing until the end. They now run in 18 parts, one laboratory setting and person per Google
machine, each part's rows saved to the bucket as it finishes (`floor_class_part.sh`, `floor_class_synthetic.py ... PART`), and the parts are
scored together by the same bars (`floor_class_synthetic.py combine`). Each part draws its own people from its own seed (20261107 + 1000 + part);
the bars are unchanged.

**The class-floor synthetic, read: not met** (`seq_atlas/outside_data/floor_class/floor_class_synthetic.txt`, `floor_class_readings.csv`; 18 parts on Google, the one
command, combined by the committed scorer). With the class floor every laboratory setting reads the islands at 0.72 to 0.85 (bar 1 not met),
the atlas laboratory's own case at 0.74 (bar 2 not met). Without any correction the atlas laboratory's own case already reads 1.235, where the
construction meant 1. **Cause, from the floors the chain recorded:** at the 47,789 island sites the sample's mix expects unmarked, the chain
measured the atlas-laboratory sample at 0.0064 and the reference at 0.00014, though the construction planted only 0.006 x 0.29 = 0.0017 of
false calls there. What the class floor counts at the islands is mostly the islands' own low methylation in the constructed people (their
island levels are drawn with a person-to-person spread the atlas's pooled reference does not carry), not the library's false calls; the same
mismatch puts the uncorrected reading at 1.235. So neither the construction nor the measure isolates the chemistry. **Next (its design first):**
the false-call share that reaches the islands measured where truth is known to be unmarked for everyone (the library-floor sites that lie in
islands), and a construction whose people carry the atlas's own island levels. The class floor stays development, off by default.

**The class floor, version 2 (written and committed before it runs).** The chain now counts the floor only at the library-floor sites that lie inside the islands read (32,570 of the 46,925
floor sites; every healthy cell of the atlas unmarked there), so a methylated call there is a false call, never the islands' own low methylation
(`stage_metaa_atlas._floor_probes`, behind `--dev-floor-class`, off by default).

**The instrument's job:** a healthy sample from any laboratory reads Normal on the islands.

**Sources:** GEO GSM5652316 (public: the base sample's reads, spent); the IAM Sequencing Atlas (public, Zenodo: island levels and floor sites).

**Reference:** the IAM Sequencing Atlas only; each library's false-call rate measured from its own reads at the floor sites inside the islands.

**Synthetic test:** `DEV_ATLAS_V21_01/seq_atlas/outside_data/floor_class_synthetic_v2.py` (the one command on constructed samples; 18 parts on
Google, scored together by its combine step). Constructed people and donor halves carry the atlas's own island levels; floor sites inside islands
take the island share of the false-call rate, as the other island sites do.

**Range of the data:** the 89 outside libraries' floors 0.0015 to 0.016 and the island shares 0.29 (atlas laboratory) and 0.75 to 1.05 (outside
laboratories), measured by `floor_by_site_class.py`; the synthetic's six settings span 0.0015 to 0.016 and shares 0.29 to 1.05.

Bars (scorer), version 1's unchanged: 1. with the class floor, each laboratory setting's median islands within 0.03 of 1; 2. the atlas
laboratory's own case within 0.03 of 1 (no overcorrection); each setting is read on three constructed people.

**Prior reading:** GSM5652316 is spent (version 1 and earlier floor work); nothing else is read.

**The class floor, version 2, read: met** (`seq_atlas/outside_data/floor_class_v2/floor_class_v2_synthetic.txt`, `floor_class_v2_readings.csv`; 18 parts on Google,
the one command, combined by the committed scorer). With the floor counted at the library-floor sites inside the islands, every laboratory
setting reads the islands at 0.989 to 1.004 (bar 1 met); the atlas laboratory's own case 0.9895 (bar 2 met). Uncorrected the same settings read
0.968 to 2.109. **Next (written first):** the 89 spent outside samples read again with `--dev-floor-class`, the islands' bar per person; the
domains, which do not follow the floor, are read against the naive and memory make-up the Deconvolver measures.

**The cross-laboratory rehearsal on spent data with the class floor (written and committed before any of its readings exists).** The 89 outside healthy samples already read (BLUEPRINT 59, GSE245832 CD8 12, GSE98837 CD8 8, Johns Hopkins CD4 6,
GSE134664 B 4) read again by the one command with `--dev-floor-class`: B, T and Johns Hopkins from their saved chunks (merge only), BLUEPRINT and
the second laboratory from their public files (`xl_class/xl_class_job.sh`, each sample saved as it is read; Google).

**The instrument's job:** a healthy sample from any laboratory reads Normal on the islands.

**Sources:** the five series' public files as first read (GEO GSE86340, GSE98837, GSE134664, GSE245832; BLUEPRINT's public tracks); the IAM
Sequencing Atlas (public, Zenodo).

**Reference:** the IAM Sequencing Atlas only, each sample against the atlas in the mix the IAM-Deconvolver measures in it; each library's
false-call rate from its own reads at the library-floor sites inside the islands; no tare.

Bars (scorer `xl_class/xl_class_score.py`): 1. at least 80 of the 89 samples each read the islands inside Normal, 0.95 to 1.05; 2. in each of the
five laboratories at least 75 % of its samples each do. Recorded, not scored: whole sample and domains, each sample's floors and main cell.

**Prior reading:** all 89 are spent (read before without the class floor); this is the rehearsal before a fresh laboratory (the BLUEPRINT
monocytes, unread).

**The cross-laboratory rehearsal: the second laboratory's first downloads failed (my error, disclosed).** The job wrapper rewrites each machine's rows with
Python's csv module, which ends lines with a carriage return and newline; the second laboratory's reader takes its URL as the last field, so every
URL carried a carriage return and every download failed. BLUEPRINT's reader strips the line and was not affected. The lists are now written
with plain newlines; the three second-laboratory machines are relaunched on the fixed job. No reading was made.

**The cross-laboratory rehearsal with the class floor, read: not met** (`xl_class/xl_class_summary.txt`, `xl_class_readings.csv`; the committed scorer.) 76 of the 89 read; 13 refused because
their islands record holds no class floor (the 12 second-laboratory samples, read at about 3 reads a site, and one BLUEPRINT sample); why is open.
Islands inside Normal: 8 of 76. Laboratory medians, islands: BLUEPRINT 0.9171, GSE134664 B 0.8498, GSE98837 CD8 0.8585, Johns Hopkins CD4 0.9377.
**What it shows, recorded:** the class floor pulled the laboratories together (their islands medians had spread from 0.30 to 1.37 in the first
reading, `crosslab_chain_table.csv`) but every laboratory now reads below Normal, whether its own floor was below the reference's (BLUEPRINT,
Johns Hopkins) or above it (GSE134664, GSE98837). The synthetic (met) did not show this; its constructed people carried no person spread at
the islands (version 2), which real people do. Next: the same 76 read against the synthetic's terms one at a time, written first, to find which
real condition the synthetic left out. The class floor stays off by default.

**Who the IAM Sequencing Atlas's donors were (metadata only; nothing read).** `seq_atlas/seq_atlas_donor_sources.csv`: each of the atlas's 156 donor files with its cell, provider, tissue,
age and sex from GEO (GSE186458), and the provider's donor source from the source publication's methods (the published sequencing atlas of
normal human cell types; GEO gives one shared protocol paragraph for every sample, so the provider is the per-sample key). 119 files (45 cells)
are tissue remnants from clinically indicated surgery at one hospital, normal tissue distant from any known pathology, confirmed with the
surgeons or pathologists before removal; 20 are cadaveric organ donors or islet programs (pancreas cells, hepatocytes); 9 come from a
Karolinska collaboration (adipocytes at cosmetic surgery after weight loss; neurons and oligodendrocytes from brain autopsies); 8 are
commercial primary cells. Ages 3 to 83 (median 52); 94 female, 61 male. Every cell type was purified by sorting and its purity checked. So the
atlas's 'healthy' is normal tissue from people, chosen and verified as normal, mostly people who were having surgery for something else; it is
not a cohort of people free of disease. IAM-Atlas v2.1's 21 sources get the same table next.

**Who IAM-Atlas v2.1's new build sources' donors were (metadata only; nothing read).** `atlas_v21_donor_sources.csv`: the 15 series the v2.1 build added (472 arrays entering, firm or conditional), each with
what its record or paper says about its donors (from the build roster, `atlas_v21_roster.csv`). 150 arrays from healthy volunteers or donors (5
series; the brain cells from post-mortem donors described as neurologically healthy); 133 from the healthy controls of disease studies (5
series: lupus, multiple sclerosis, Alzheimer's disease, Parkinson's disease, antiphospholipid syndrome; only their controls were used); 16 normal
ovarian and tubal tissue from surgical patients without cancer; 16 children's normal intestinal mucosa; 9 embryonic stem cell lines (not
donors); 148 from two series whose GEO records do not state donor health, now read from their papers: GSE103541 (60) a population twin
birth cohort, all aged 19, not selected for disease; GSE252045 (88) blood from leukapheresis chambers of CMV-positive donors, their health not
otherwise described. The cells v2.1 carries
from IAM-Atlas v2 (the sequencing atlas above and the earlier array references) are tabled next, so both atlases are covered in full.

**Who the donors of the groups IAM-Atlas v2.1 carries from v2 were (metadata only).** `atlas_v2_groups_donor_sources.csv`, from each source's paper: the two sorted blood reference libraries are healthy
volunteers through commercial suppliers (37 donors, screened for no history of heart, lung or kidney disease, asthma, blood disorders,
autoimmune disease, cancer or diabetes; and 56 donors, all anonymous healthy donors); the array cell reference for plasma DNA is mostly normal
tissue from surgery, from the same hospital group as the sequencing atlas; the brain single-nucleus series is three adult male post-mortem
brains, their health not described in the text read. With `seq_atlas_donor_sources.csv` and `atlas_v21_donor_sources.csv` this covers every
group both atlases are built from.

**After the class-floor rehearsal: what is shared by every laboratory (development; diagnostic of committed readings; nothing scored).** From `xl_class/xl_class_readings.csv` and the first reading's `crosslab_chain_table.csv`: the class floor moved every
laboratory toward Normal from both sides (islands medians, first reading to class floor: BLUEPRINT 0.30 to 0.92, Johns Hopkins CD4 0.68 to
0.94, GSE134664 B 1.08 to 0.85, GSE98837 CD8 1.37 to 0.86), and each stopped short by about the same amount, whether its floor at the floor sites
inside the islands was below the atlas laboratory's (BLUEPRINT 0.0013, Johns Hopkins 0.0049, against 0.0068 to 0.0070) or above it (0.0091,
0.0120). A deficit shared by every laboratory is a term every outside sample differs in from the reference, not the floor. Two candidates fit
what is measured, to be separated by a test written first: (1) the cell mix: purer samples read lower (Spearman rho -0.32 with the share of
the cell found most of, p 0.004), as they would if the Deconvolver spreads part of a sample onto neighboring subsets (it moved 7 to 24 % so on
constructed arrays, `DEV_METAA_HEALTHY_01.md`), making the expected disorder too high; (2) person spread at the islands, which version 2's
constructed people did not carry. The class floor stays off by default.

**The class floor, version 3: real atlas donors against the other sealed half (written and committed before it runs).** Version 2 built both its reference
and its people from the atlas's average island levels, so they agreed by construction, and planted 0.29 of the floor at the floor sites
inside the islands, where the real outside libraries carry their whole floor (`xl_class/xl_class_readings.csv`). Version 3 constructs only
the planted false calls: each person is a real atlas B-cell donor's own reads at every CpG, read against the sealed donor half it is not in,
with the chain's own committed island and donor counts (`floor_class_synthetic_v3.py`, one part per machine by `floor_class_v3_part.sh`,
each part saved after every reading).

**The instrument's job:** a healthy sample from another laboratory reads Normal at the islands.

**Sources:** GSE186458 (public, GEO: three B-cell donors' .beta files, GSM5652316 and GSM5652318 in half A, GSM5652317 in half B); the IAM
Sequencing Atlas (public, Zenodo: its sealed halves' island and donor counts).

**Reference:** IAM Sequencing Atlas only (the donor half the person is not in); no tare.

**Range of the data:** outside library floors 0.0016 to 0.0158 and island shares 0.75 to 1.05 (`crosslab_chain_table.csv`,
`crosslab_floor_by_class.txt`); at the floor sites inside the islands the outside floor equals the library floor. Synthetic version 3: floors
0.0015 to 0.016, shares 0.75 to 1.05, the whole floor at every library-floor site, real donors and the real reference.

**Bars,** with the class floor: 1. each laboratory's median islands reading within 0.03 of 1; 2. the atlas laboratory's own case within
0.03 of 1. Recorded, not scored: the readings with no floor and with the common floor, and each floor the chain measured.

**What it separates:** if the three real donors read Normal here and the 89 still do not, the shared deficit is not the floor's planting
and not person spread at the islands (real people carry their own); the cell mix is then next.

**The class floor, version 3: first launch refused by the chain (my error, disclosed; no reading made).** The script wrote each person's counts as 8-bit numbers where the chain's
`.lbeta` input holds two 16-bit counts a CpG (as version 2 wrote them), so the chain read half the CpGs and refused every reading ("14,108,724
CpGs, not the 28,217,448 of the hg19 numbering"). The script still exited 0, so each part marked itself read; the parts' rows and markers are
moved aside in the bucket (`failed_8bit/`), not deleted. Fixed: 16-bit counts, and a part with any reading the chain refused or left empty now
fails and is never marked read. Bars, people and planting unchanged.

**The class floor, version 3, read: not met** (`seq_atlas/outside_data/floor_class_v3/floor_class_v3_synthetic.txt`, `floor_class_v3_readings.csv`; 18 parts
on Google, the one command, all 54 readings made.) Bar 1 not met: with the class floor every planted laboratory read high, 1.05 to 1.37 (the
four at an island share near 1, 1.25 to 1.37). Bar 2 not met by 0.021: the atlas laboratory's own case, nothing planted, median 1.0507 (each
1.0545, 1.0213, 1.0507). **What it shows, recorded, not scored:** (1) with the class floor the reading no longer follows the planted floor
(floors 0.0015 to 0.009 all read 1.25 to 1.37), so the floor itself is counted; (2) the planted laboratories read high where the real outside
libraries read low with the same correction (`xl_class_readings.csv`: 0.85 to 0.94), so this planting does not reproduce the real libraries:
the share of the floor placed at every island site (0.75 to 1.05 of the floor, from `crosslab_floor_by_class.txt`) is more than the real
outside libraries carry there, and that share measure is not confirmed; (3) real healthy donors of the atlas's own laboratory, read with no
floor against the other sealed half, spread 0.82 to 1.14 at the islands, wider than a 0.95 to 1.05 band. The class floor stays off by
default. Next, before anything new is planted: the real outside libraries' false-call rate at the island sites outside the floor sites,
measured from the per-site tables already saved (18 samples), and the cell-mix candidate, each by a test written first.

**The library floor by molecule position: the detector-split null test on sequencing (development; synthetic written before it runs).** The record of
11-01 found sorted-cell Met-A follows each library's own disorder per read at its identity sites (rank correlation 0.974), shallower
libraries holding more off-pattern reads, not counting. Whether those reads are biology or library chemistry is open. On arrays the probe
designs split the instrument from the biology; on sequencing the split is where a call sits on its molecule: biology does not depend on it,
chemistry at the read ends does (end repair's M-bias). `stage_seq_atlas.floor_by_position` (`run_sample.py --build seq-floor-position`):
at the floor sites, opposite calls by distance from the molecule's nearer end, and beside the gap between a pair's reads.
**Synthetic** (`seq_atlas/floor_position_synthetic.py`): three constructed libraries of 400,000 molecules, no end effect, ten times at
the ends, and a plasma-level floor twice at the ends. Bars: no end effect reads end over interior 0.8 to 1.25; the planted ratios
recovered within 10 %; inner distances read the interior rate within 15 %.
**Range of the data:** library floors measured by the chain run from 0.00154 (BLUEPRINT) and 0.00628 (the atlas's inputs) to 0.022 to
0.037 (plasmas); the synthetic's interior floors 0.005 and 0.03 and its end floors 0.005 to 0.06 span them. The end-over-interior ratio
of real libraries is what the test measures; the synthetic covers none (1), large (10) and modest (2).
**Then, spent and already aligned (no alignment):** the molecule files in the build bucket, Johns Hopkins CD4 (85 pieces), the outside B
and CD8 T cells (192) and the Johns Hopkins tissues (126), spread four pieces at a time over the free machines (`floor_position_real.sh`).
**Prior reading:** all three were read before (`CANON/data_spent.csv`); nothing here is scored.

**The library floor by molecule position: synthetic met; the real libraries read.** Synthetic (GitHub run 42591860537): no end
effect read 0.90, the planted 10 and 2 read 10.30 and 1.99, inner distances at the interior rate. On 30 spent, already aligned
libraries (`seq_atlas/floor_position_real_summary.txt`, `floor_position_real.csv`; no alignment, about 15 minutes): the opposite calls
sit at the molecules' ends. The outside CD8 T laboratory (GSE98837) reads its first and last CpGs at 7.5 to 12.5 times its interior
floor; the outside B laboratory (GSE134664) 3.1 to 4.9 times; the Johns Hopkins CD4 libraries 0.89 to 21, library by library (about half
near 1, the rest 4 to 21). The interior floors are close across all three (0.0013 to 0.0067), while the pooled floor the chain uses
(`library_floor`) carries the ends: GSE98837's 0.014 to 0.021 is mostly end calls over an interior of 0.003 to 0.005. So most of what
differs between these laboratories' floors is library chemistry at the read ends, not the cells. On the 18 libraries with a whole-sample
bundle, the end rate alone does not track whole-sample Met-A (rank correlation -0.27), so the next
test removes the end calls from the reading itself (each molecule's outer CpGs left out, as end trimming does) and reads the same spent
libraries again, from these molecule files. Development; nothing scored.

**The floor by molecule position: a confound in the first measure, and what trimming would cost (disclosed before the next run).** Real molecules hold few
CpG calls (median 2 to 3 in GSE98837, Johns Hopkins and GSE134664 files), so the 'interior' calls (5 or more CpGs from either end) come
only from long, CpG-dense molecules, while end calls come from everywhere: part of each end-over-interior ratio above can be the mix of
sites, not chemistry (the differences between libraries over the same genome, Johns Hopkins 0.9 to 21, stand). The measure now also
compares end calls with inner calls at the same sites (Mantel-Haenszel over sites holding both, each side; offline, a planted 10 read
10.19), and the synthetic gains that bar, written before its run.
**Range of the data:** the first measure read 0.89 to 20.97 across the 30 real libraries (`floor_position_real.csv`), interior floors 0.0013
to 0.0067 and end floors 0.0019 to 0.0415; the synthetic plants end-over-inner 1, 10 and 2 with floors 0.005 to 0.06, spanning them. Trimming by CpG count costs too much: leaving out 1, 2 or 4 calls at each
end keeps 38 to 49 %, 17 to 23 % and 5 to 6 % of the calls, and read-end chemistry is set in bases, which the molecule files no longer
hold. `--dev-seq-trim-ends N` (`stage_seq_values.trim_pat`) is built for testing and stays off; the next step is decided by the
site-matched result.

**The floor by molecule position, site-matched: synthetic met; the end effect is real and differs by library.** Synthetic (GitHub run 42600923336):
site-matched 0.9836, 10.0144 and 2.0076 (`seq_atlas/floor_position_synthetic.txt`) for the planted 1, 10 and 2; all four bars met. On the 30 spent libraries (400 of 403 pieces, 8 Google
machines; `seq_atlas/floor_position_matched.csv`), end calls against inner calls at the same sites: GSE98837 CD8 4.6 to 8.1 (the first
measure's 7.5 to 12.5 was part site mix), GSE134664 B 3.1 to 4.2, Johns Hopkins CD4 0.86 to 12.4 (ten libraries 0.86 to 1.17, eight 1.9 to
12.4). So library chemistry at the molecule ends is real and is a property of each library, not of the laboratory alone. On the 18
libraries with a whole-sample reading it does not track whole-sample Met-A (rank correlation -0.10), so it is
not by itself the laboratory term in Met-A; it does inflate the pooled library floor the chain measures. Development; nothing scored.

**The molecule-position split at each library's identity sites (development, spent, already-aligned files; nothing scored).** The floor
split found a real read-end effect that differs by library but does not follow whole-sample Met-A. Met-A reads the identity sites, so the same
split is made there: `run_sample.py --build seq-floor-position --build-cell CELL` (`stage_seq_atlas.identity_side_table`), CELL the main cell
the IAM-Deconvolver found in each library's own earlier bundle (`seq_atlas/identity_position_cells.csv`; Johns Hopkins GSM2103004, labeled
CD4, is read at monocytes, where its Deconvolver put it). Checked offline on constructed molecules with a planted tenfold end rate: read 9.54 (`seq_atlas/identity_position_offline_check.txt`).
Run over the 18 sorted libraries' already-aligned pieces on 12 machines (`floor_position_identity.sh`). If the end calls carry the libraries'
extra off-pattern reads at the identity sites, each library can be calibrated on its own molecules, with nothing from other samples.
**Prior reading:** all 18 were read before (`CANON/data_spent.csv`).

**The IAM-Deconvolver weighted by each site's read count on sequencing (chain option, development; its synthetic written before it runs).**
Auditing what was built alongside the IAM Sequencing Atlas against what reading a sample uses: on sequencing the Deconvolver received each
site's methylation fraction but not the read count it rests on, so a site seen in 2 molecules counted as much as one seen in 60. A CMB fit
weights each pixel by its own noise; `--dev-seq-noise-weight` does the same (`stage_deconv_v21._solve`: the counting noise p(1-p)/n from the
model's prediction joins each site's weight). Offline on constructed mixes from the atlas's marker levels, with depth varying site to site:
share error 0.164 to 0.103 and false cells 0.103 to 0.056 at 2 reads, 0.064 to 0.038 and 0.048 to 0.020 at 40, better at every depth.
**Synthetic** (`seq_noise_weight_synthetic.py`, `seq_noise_weight_synthetic_score.py`): six constructed samples (three blood-like, three
plasma-like) from the atlas's levels at IAM-Atlas v2.1's sites, read as counts by the one command with and without the option; bars: closer
to the truth in 5 of 6, the share on cells not in the sample no larger in 5 of 6, at depth 20 no worse by more than 0.01.
**Range of the data:** the real libraries read at a median 6 to 26 calls per site (the 18 sorted libraries) and the real healthy plasmas at
a median 6 with 251,945 of 813,951 sites read (GSM7426156); the synthetic reads depth 6 and 20 with site-to-site variation and, for plasma,
a third of the sites read.

**The molecule-position split at the identity sites, read: the laboratories differ at molecule ends, not inside.** On the 18 sorted
libraries (`seq_atlas/floor_position_identity_summary.txt`), each library's Met-A follows its opposite-call rate at molecule ends (rank
correlation 0.88) and not inside molecules (-0.20): end rates GSE134664 B 0.087 to 0.089, Johns Hopkins CD4 0.061 to 0.075,
GSE98837 CD8 0.031 to 0.059, while interior rates overlap (0.004 to 0.015). The site-matched ratio does not follow Met-A (0.15); the
interior calls come from fewer, CpG-denser sites, so the next step reads Met-A itself without the outer calls: the 18 libraries through the
one command with `--dev-seq-trim-ends` 0, 1 and 2 (`trim_read.sh`, `trim_read_samples.csv`), one library read first and checked before the
rest are sent. **Prior reading:** all 18 were read before (`CANON/data_spent.csv`).

**The read-count weighting synthetic, read: not met (bar 2).** (`seq_atlas/seq_noise_weight_synthetic_summary.txt`; the
first send found no input, my relative path, and read nothing.) Bar 1 met, 6 of 6 closer to the truth (blood depth 6: 0.1396 to 0.0882;
plasma: 0.3621 to 0.2003); bar 3 met at depth 20 (0.0768 to 0.0631). Bar 2 not met, 3 of 6: the share on cells not in the sample rose in
two bloods (0.0448 to 0.0515 at depth 20, 0.0555 to 0.0721 at depth 6) and one plasma (0.0819 to 0.1413). A design error of mine confounds
bar 2 on the plasmas: two of their planted tissues (liver hepatocytes, vascular endothelial cells) are not cell types of the IAM Sequencing
Atlas, so their DNA must be read as other cells; the bloods' rises are not explained by it. Weighting brings the shares closer overall and
spreads small shares over more cells at moderate depth. `--dev-seq-noise-weight` stays built and off.

**The trim reading, read: one end call out cuts the laboratories' spread sevenfold; the reference must take the same path.** The 18 sorted libraries through the one
command with 0, 1 and 2 calls left out at each molecule end (`trim_read.sh`, `seq_atlas/trim_read_summary.txt`, all 54 readings; 27 bundles first read with no Met-A, a chain bug fixed in
`_cover_den`, were set aside and read again). Spread of the laboratories' median whole-sample Met-A: 0.2404 untrimmed, 0.0338 with one call
out, 0.1887 with two; of the per-cell Met-A: 0.3147, 0.0877, 0.2544. With one call out every laboratory reads a little below 1 (0.94 to
0.97), as it should when the sample is trimmed and the atlas it is read against is not: the reference must be processed as the sample is.
The atlas was built from per-CpG count files, which hold no molecules, so its 156 donors are counted again from their published molecule
files with the same trim (`stage_seq_values.pat_to_lbeta`, `run_sample.py --build seq-pat-lbeta`; offline synthetic: the count file equals
trim_pat then values_from_pat at every site, for 0, 1 and 2 calls out), and the atlas's own build then runs unchanged on those counts
(`atlas_trim_donors.sh`; one donor first, its untrimmed count checked against the published .beta). **Prior reading:** all 18 libraries
were read before (`CANON/data_spent.csv`).

**The trimmed donors' canary counted nothing: an empty download read as an empty donor (chain bug, fixed).** The first donor's count came back with 0 molecules: GEO sent the Google
machine an empty file with no error, and neither the fetch (`geo_pat`) nor the counter checked. The fetch now compares each download with the
size GEO states and fetches again when short; the counter refuses a file with no molecules. Checked locally (an empty file refused; an empty
then a short body refetched until whole). The empty canary output was set aside (`atlas_trim_void/`) and the canary sent again.

**The trimmed donors' canary: the count file equals GEO's published file; one call out keeps 32 % of molecules.** Donor GSM5652176, counted by the one command
from its molecules (`--geo-pat`, the size check on): untrimmed, the count file equals GEO's published .beta at all 28,217,448 CpGs below
that file's cap of 255 (0 differing). With one call left out at each end, 122,743,125 of 388,137,463 molecules keep a call (molecules with one
or two CpGs keep none) and 350,898,464 of 954,416,086 calls are kept (37 %): the trimmed atlas rests on about a third of the depth, as does
a trimmed sample. The other 155 donors are being counted on Google.

**The trimmed atlas: build and test written before they run (development).** The instrument's job: a sorted healthy library read with one molecule-end call
left out at each end reads Normal against the atlas built the same way, from any laboratory. The 156 donors of
`seq_atlas_inputs.csv` (57 cells, the 3 granulocyte donors among them) are counted with one call out (`atlas_trim_donors.sh`), the atlas is
built from those counts by the one command exactly as the published one, every cell in one folder (`trim_atlas_build.sh`: 113 pieces, then the
in-mix columns at IAM-Atlas v2.1's sites and the Deconvolver's reference), and the 18 sorted libraries of three laboratories are read
again with one call out against it (`run_sample.py --dev-seq-atlas-dir`, development: the bundle and the reference door name the rebuilt atlas
as not door-checked). Measured, not scored: the laboratories' spread of whole-sample and per-cell Met-A, and how far each laboratory's median
sits from 1, beside the same libraries read untrimmed against the published atlas (0.2404 and 0.3147) and trimmed against it (0.0338 and
0.0877). Only the in-mix columns and the Deconvolver's reference are rebuilt for this test; the floor sites and the per-input tables stay the
published ones, as the bundle says.

**Count corrected (my error).** The build was first written for 159 donors, taking the 3 granulocyte donors of `gran_inputs.csv` to be outside `seq_atlas_inputs.csv`; they are inside it (156 donors, 57 cells). The build's own count check refused to build on all 12 machines; it now expects 156 and drops duplicates.

**The trimmed atlas's first piece, checked against the published one before the rest were built.** (`seq_atlas/trim_atlas_canary_piece.csv`; CpGs 0 to 250,000,
all 57 cells.) Same CpG rows and fields; per cell, 0.921 to 0.958 of the sites keep a level, the
reads at those sites are 0.537 to 0.592 of the published depth, and the level moves by a median of
0 (median absolute change 0.0102 to 0.0261, the counting noise of the smaller depth).
The other 112 pieces were then built. A first launch of all 12 build machines before this check refused to build on a donor count I had
written wrong (above), and the launcher left those machines idle instead of deleting them; the launcher now deletes a machine however its
job ends.

**The first trimmed library against the trimmed atlas: refused before reading (the megakaryocyte column).** The one-library canary stopped because the default
chain reads sequencing with the provisional megakaryocyte column, a separate Deconvolver reference not in the rebuilt atlas. With
`--dev-seq-atlas-dir` the chain now reads without that column and the bundle's reference block says so; the comparison readings (trimmed
against the published atlas) carried it, so each library's megakaryocyte share there is checked before the two are compared.

**Was the atlas source's end bias already trimmed? No (a check, before reading anything into the trimmed atlas).** The end bias of
bisulfite reads is well known in sequencing work (often called M-bias), and pipelines can trim it before publishing; if the atlas source had,
trimming its donors again would cut twice. On the first 600,000 molecule lines of donor GSM5652176, at the floor sites (unmarked in every
cell), a molecule's end calls read marked 2.33 times as often as its inner calls (0.00619 against 0.00266; 3,713 and 5,265 calls at 115
floor sites, so indicative only). The source's files keep an end effect, smaller than most of the sample laboratories' (0.86 to 12.4 times),
so the atlas is trimmed as the samples are, not twice. In public text the end bias is credited as known; what this work adds is putting the
atlas through the same processing as the sample.

**The trimmed libraries against the trimmed atlas: not met.** (`seq_atlas/trim_atlas_read_summary.txt`, `trim_atlas_read_rows.csv`; all 18.) Putting
the atlas through the same trim widened the laboratories' spread instead of closing it: whole-sample Met-A 0.0338 against the
published atlas, 0.1003 against the trimmed one; per-cell 0.0878 and 0.2632. The change is laboratory by laboratory:
GSE98837's CD8 cells moved to 1.0569 (from 0.839), the other two hardly moved. So the
trimmed sample read against the published atlas stays the better reading, and the rebuilt atlas stays a development folder, not a new
version. Why one laboratory moves and the others do not is the next thing to read (the depth each library keeps after the trim, and the
atlas cells' own end effect).

**Sequencing step 1, per-library self-calibration: the atlas's own error profile (development; spent data).** The instrument's job: a healthy
library from any laboratory reads 1 against the sequencing atlas, with nothing borrowed from other samples. The trims showed why cutting data is
not the answer: one end call left out of the sample alone tightened the laboratories (whole-sample spread 0.034 against the published atlas) but
put every laboratory below 1 (0.94 to 0.97; per cell 0.75 to 0.84), and trimming the atlas too widened them (`trim_atlas_read_summary.txt`);
the atlas's own donors keep an end effect (2.33 times their inner rate on one donor). The route the CMB missions took is to forward-model each
instrument's measured response onto the expectation: each library's own error at the molecules' ends and inside, measured at the floor sites
(every healthy cell unmarked) and its cell's held identity sites, both sides (`floor_by_position` now reports each side), and the atlas's own,
built into the expected value. The atlas's side needs no new reading: each donor's published counts and its one-call-trimmed counts (the trimmed
atlas build) differ by exactly its end calls, so its end and inner rates come from those two count files. The expectation and its synthetic are
written before any library is read with it.
**Range of the data:** in the real libraries, end calls 0.86 to 12.4 times the inner rate at the same sites, library by library (30 spent libraries, site-matched,
`floor_position_matched.csv`); the atlas's own donor 2.33 times (one donor). The synthetic plants end profiles across 0.8 to 13 times.

**Each laboratory's end signature has its own side (development; spent, already-aligned data; nothing read again).** The 277 saved pieces of the 18 sorted libraries
(`floor_position_identity_by_side.csv`, pooled from each piece's site-matched end-over-inner ratio per side, weighted by sites): at the identity sites
the cell holds marked, the B-cell laboratory (GSE134664) reads its end calls 1.6 to 3.2 times as often opposite as its inner calls at the same
sites, the Johns Hopkins CD4 laboratory 1.2 to 1.35 times, the CD8 laboratory (GSE98837) 0.85 to 1.2 times; at the sites held unmarked the CD8
laboratory reads 2.9 to 6.8 times, the others 0.67 to 2.4. Met-A follows the marked side (rank 0.84; unmarked -0.48): B cells 1.26 to 1.31, CD8
0.71 to 0.96, CD4 0.91 to 0.99. A false T at a marked site near a molecule's end is what end repair with unmarked C writes; a false C at an
unmarked site is a different chemistry. Each library carries its own signature, per side, measurable from its own molecules: the input the
per-library forward model needs (absolute rates per side and distance from the end: the pieces are being read again with the per-side counts).

**Sequencing step 1, each library calibrated on its own calls: built, synthetic sealed (development).** The absolute per-side rates, read again from the
277 pieces by the one command (`floor_by_position`, per side): end calls are 59 to 76 % of the calls at marked identity sites and 25 to 48 % at
unmarked ones; at marked sites the end calls read opposite 3.4 to 12.4 % against 2.9 to 7.4 % inside. The atlas's own end calls, measured
from its 156 donors (published counts minus the one-call-trimmed counts, `metaa_inner_reference_v1`; 787 of 905,266 donor-site rows had fewer
published than trimmed calls and are left out): across the 57 cells the atlas's end calls read opposite more often than its inner calls, a
median 0.43 points more at marked sites and 0.95 at unmarked ones (B, CD4, CD8 T cells and monocytes 1.1 to 1.6 points at marked sites).
So the atlas and every library carry an end error of their own. The route (`stage_metaa_atlas.self_calibrated`, `run_sample.py
--dev-seq-self-cal`): the healthy level is the atlas's inner level; each library's own end excess per side (end rate minus inner rate at the
found cell's identity sites) moves it at each site by that site's own share of end calls; nothing cut, nothing borrowed from other samples.
**Synthetic (sealed before any planted library is read, `selfcal_synthetic_score.py`):** one spent library from each of three laboratories,
read five ways (`selfcal_synthetic_samples.csv`): unplanted; end C to T 5 %; end T to C 3 %; both 8 % and 3 %; a uniform C to T 2 % at every
call (a change in the cell). Bar 1: with the calibration every end plant moves Met-A by at most 0.01 (all nine). Bar 2: without it the median
move is at least 0.02 (else the synthetic cannot see its plant). Bar 3: the uniform change is kept, its move with the calibration at least
80 % of its move without (all three).
**Range of the data:** in the real libraries, end calls at marked identity sites read opposite 3.4 to 12.4 % and inner calls 2.9 to 7.4 %
(18 spent libraries, `floor_position_identity_by_side.csv`); the synthetic plants end errors of 3 to 8 % on top of each library's own.

**The self-calibration synthetic: not met (bar 1, 1 of 9); bar 2 informative; bar 3 met (3 of 3).** (`DEV_ATLAS_V21_01/seq_atlas/selfcal_synthetic_summary.txt`; 15 libraries
read by the one command with --dev-seq-self-cal, one per row, scored by the sealed scorer.) Each library measures its own planted end error
(marked-side end rate up by 0.046 to 0.071 for a planted 0.05 to 0.08), so the measurement holds; the shortfall is in how it enters Met-A.
Planted end error removed (change in Met-A from the unplanted reading, without and with the calibration): B cells (found 89 %) 92 % and 97 %
on the marked side and both sides, but on the unmarked side the plant moved Met-A +0.026 and the calibration -0.024 (overshot); CD4 T cells
(found 36 %) 91 %, 95 % and the unmarked side within 0.01; CD8 T cells (found 32 %) 73 %, 88 % and 87 %. A uniform 2 % change was kept in
all three (bar 3). The unplanted B library read 1.29 without and 1.00 with the calibration; CD4 0.954 and 0.942; CD8 0.798 and 0.766.
What the pattern says: the found share does not explain the shortfall (CD4 at 36 % is absorbed nearly as well as B at 89 %); the library
whose reading sits farthest from 1 (CD8, 0.77) is absorbed least. Met-A is a ratio of disorders, H of the library over H of the expectation,
and an error added to both moves them by the slope of H at each one's own level; only when the library reads 1 do the two moves cancel.
Correcting the expectation alone therefore cannot leave Met-A unchanged when the library departs from healthy, by construction. Next (a new
synthetic, sealed before reading): remove the library's measured end error from its own calls (the measured response taken out of the
measured data, against the atlas's inner level, which carries no end calls), and test it on the same 15 rows' bundles' conditions.

**The self-calibration synthetic, second form (sealed before reading; development).** The same 15 rows, the same bars and scorer form
(`selfcal_synthetic2.sh`, `selfcal_synthetic2_score.py`), with Met-A read on each library's own calls with its measured end error taken out
(run_sample.py --dev-seq-self-cal-data), against the atlas's inner level. Each read also records the first form's reading, so the two are
compared on the same molecules. One row first as a canary.
**Range of the data:** in the real libraries, marked-side end calls read opposite 4.4 to 11.5 % and inner calls 3.7 to 6.0 % (the first
synthetic's unplanted rows); the plants add 3 to 8 % at the ends, as before.

**The second form's design does not remove a counted error (found on its first 11 rows; my design error, recorded before its last rows land).** A sequencing site's level is a few reads; an end error flips a whole read at a few sites rather than
lowering every site a little. Subtracting each site's average expected error from its observed level is clipped away where no error fell and
moves a hit site by about a hundredth where the error moved it by a third, so the disorder the errors add stays in (unplanted B cells: 1.29
without, 1.28 with). The first form, the error put into the counting expectation, handles the reads exactly (B cells 1.29 to 1.00).
**The self-calibration synthetic, third form (sealed before reading; development).** The same 15 rows, bars and scorer form
(`selfcal_synthetic3.sh`, `selfcal_synthetic3_score.py`): Met-A as the observed disorder minus the disorder the library's measured end error
adds, in expectation under counting, read against the atlas's inner level (run_sample.py --dev-seq-self-cal-disorder). Each read records all
four forms on the same molecules. One row first as a canary.
**Range of the data:** in the real libraries, marked-side end calls read opposite 4.4 to 11.5 % and inner calls 3.7 to 6.0 % (the first
synthetic's unplanted rows); the plants add 3 to 8 % at the ends, as before.

**The third form, on its first 11 rows, and why: not met; the fault is the correction's model (development).** Comparing the forms on the same molecules (`selfcal_synthetic3_score.py`
when its last rows land): the both-sides plant cancels in CD8 T cells (-0.003) while each side alone does not (+0.082 marked, -0.004
unmarked), and B cells (89 % of their sample) do better than CD8 and CD4 (about a third). The first three forms corrected each site by a
step rule (marked or unmarked side, split at one half) on the found cell's own atlas column; an end error flips calls in proportion to each
state's share, on every molecule in the mix.
**The self-calibration synthetic, fourth form (sealed before reading; development).** The same 15 rows, bars and scorer form
(`selfcal_synthetic4.sh`, `selfcal_synthetic4_score.py`): every found cell's inner level mixed in the sample's shares, the library's
measured end error flipping calls in proportion to each state's share at each site (`stage_metaa_atlas.mix_forward`,
run_sample.py --dev-seq-self-cal-mix). Each read records all five forms on the same molecules. One row first as a canary.
**Range of the data:** in the real libraries, marked-side end calls read opposite 4.4 to 11.5 % and inner calls 3.7 to 6.0 % (the first
synthetic's unplanted rows); the plants add 3 to 8 % at the ends, as before.

**The self-calibration synthetic, fourth form, read: not met (13 of 15 rows; two planted CD4 rows stopped unread).** Bar 1 (planted end errors absorbed, Met-A moves at most 0.01): CD8 T cells (32 % of
their sample) marked side +0.107, unmarked +0.012, both +0.119; B cells (89 %) unmarked -0.018, both -0.011. Without the calibration the same
plants moved Met-A +0.375, +0.120, +0.529 (CD8) and +0.026, +0.441 (B). Bar 3 (a uniform 2 % change kept): 88 % (CD8) and 84 % (B), met.
CD4 T cells (36 %): unmarked side +0.014 (without the calibration +0.058); the uniform change kept 100 %. Correction: the CD4 rows were slow,
not hung. Two finished; the marked-side and both-sides CD4 rows were stopped by hand after 80 minutes, before they finished, which was a
mistake in reading their logs, and they were not read. The verdict does not depend on them. **What all four forms share:** each removes most of the planted error (about 75 to 98 %), none removes enough, and
the remainder is largest where the found cell is a minority of its sample and on the marked side. Modelling the mix (fourth form) did not
change that, and converting the measured excess to a per-call rate accounts for about 4 %. Next: compare, site by site, the change each plant
made with the change the correction predicts, on molecules already aligned, before any fifth form.

**The library-noise rule (2026-11-10; chain change, development).** `stage_q2_selfref.meter_end_damage`, recorded in every count (end_damage, meter): each library's
lone-loss rate near the molecule ends minus farther in. Synthetic (offline design check, 60,000 planted molecules, two draws each): no end
damage reads -0.0011 to -0.0003 (ok); 0.003 planted reads 0.0019 to 0.0029; 0.006 planted reads 0.0036 to 0.0047 (flagged). On the 18 spent
sorted libraries: GSE98837's CD8 libraries flagged in 6 of 8; GSE134664's B cells and the Johns Hopkins CD4 libraries all ok.
**Range of the data:** in the real libraries, end damage -0.0034 to 0.0079 (PHY-050 records); the synthetic planted 0 to 0.006.

**Why the first four forms failed (found 2026-11-11, from the fourth form's bundles).** Each form measured the library's end error on sides set by
the read cell's own level at its identity sites. Those are the sites where the cell differs from the rest of the sample, so for a cell that is a
third of its sample the 'marked' side holds mostly unmarked molecules of the other cells, and the measured rate is a blend of both errors. The
measurement of each planted error itself was right (CD8 library: marked-side excess up 0.046 for a planted 0.05, unmarked up 0.029 for 0.03).
It matches the failure pattern of all four forms: worst for a minority cell, on the marked side; nearly right for the B-cell library (89 %).

**The self-calibration synthetic, fifth form (sealed before reading; development).** The same 15 rows, bars and scorer form
(`selfcal_synthetic5.sh`, `selfcal_synthetic5_score.py`): Met-A against every found cell's inner level mixed in the sample's shares, the
library's two end-error rates measured where every found cell agrees (stage_metaa_atlas.agree_rates; run_sample.py --dev-seq-self-cal-agree),
put into the mixed forward model (mix_forward). Checked before sealing on a simulated library on the atlas's own columns (cd8, cd4 and
b cells): planted 0.05 and 0 recovered as 0.0505 and 0.0001; planted 0.08 and 0.03 as 0.0811 and 0.0304. One row first as a canary.
**Range of the data:** in the real libraries, marked-side end calls read opposite 4.4 to 11.5 % and inner calls 3.7 to 6.0 % (the first
synthetic's unplanted rows); the plants add 3 to 8 % at the ends, as before.

**The fifth form's canary: no agreement sites (rule changed before the other rows run).** The CD8 both-sides row found 12 cells, among them the provisional
megakaryocytes (1.8 percent), whose column is missing at most array sites; the rule required every found cell to hold a value, so no site
qualified and the reading fell back to the earlier form (recorded in its bundle). The rule now asks that the cells with a value hold at least
98 percent of the sample's DNA and all agree. Checked on that row's own 12 cells and shares with simulated counts at the real sites before
the next canary: 30,588 marked and 101,805 unmarked agreement sites; planted 0.08 and 0.03 recovered as 0.0807 and 0.0307, planted 0.007 and
0.035 as 0.0066 and 0.0356. The bars and rows are unchanged.

**The self-calibration synthetic, fifth form, read: not met** (`selfcal_synthetic5_summary.txt`, the sealed scorer unchanged). Bar 1, end errors absorbed: 3 of 9.
Bar 2, the plant visible without the calibration: median 0.3249. Bar 3, a uniform change kept: 2 of 3. Two CD8 rows found no agreement
sites (cells without an atlas array column held more than 2 percent of the sample) and their reading is the first form's, counted as written.
**What it showed (exploration on the same bundles, after the score).** The agreement sites measure a planted error exactly: on the CD8
both-sides row the rates read 0.0888 (marked) and 0.0597 (unmarked), the library's own 0.0087 and 0.0335 plus the planted 0.08 and 0.03,
yet Met-A still moved +0.1043; so the remainder is in how the rates enter the mixed expectation, not in measuring them. And the library's own
end error is not the same at every CpG: B cells (89 percent of their sample) read 0.0542 at their identity sites and 0.1216 at the sites where
every cell is marked; CD4 read 0.0214 and 0.0038 on the unmarked side; CD8 agrees on both sides. The fifth form's premise, one rate per
library, fails in two of the three laboratories. Next, a sixth form, synthetic first: the two rates measured on the read cell's own identity
sites, only where the mixed expectation is near 0 or 1, so the molecules' state is known and the site set is the one read.

**The self-calibration synthetic, sixth form (sealed before reading; development).** The same 15 rows, bars and scorer form (`selfcal_synthetic6.sh`,
`selfcal_synthetic6_score.py`): Met-A as the observed disorder minus the disorder the library's end error adds, computed at the library's own
level at each of the cell's identity sites, with the end error measured at those sites in ten bins of the mixed atlas inner level
(stage_metaa_atlas.own_level_added; run_sample.py --dev-seq-self-cal-own), read against the atlas's inner level mixed in the sample's shares.
Checked before sealing offline on simulated libraries with the same sites under every condition (the design check, not the test): at 89
percent share, planted end errors moved the reading by 0.0009 to 0.0039 (without the correction 0.0144 to 0.0922, the largest with an end error
twice as high at the cell's marked sites); a real 2 percent change was kept in full. One row first as a canary.
**Range of the data:** in the real libraries, marked-side end calls read opposite 4.4 to 11.5 % and inner calls 3.7 to 6.0 % (the first
synthetic's unplanted rows); the plants add 3 to 8 % at the ends, as before.

**Diagnosing the sixth form: two sources per library, and every site's inputs (development; nothing scored).** Each library's molecule-end error is read by two
independent sources: at the read cell's identity sites, and at the sites where every found cell is fully marked or fully unmarked. In the
sixth form's bundles they agree in the CD8 library on both sides (where every correction has worked), and disagree in the B-cell library on the
marked side (0.054 against 0.122) and in the CD4 library on the unmarked side (0.021 against 0.004), where the corrections failed; a planted
error is seen alike by both (+0.047 and +0.046). The sixth form also takes out too much on the B-cell library's unmarked plant (-0.058), where
its two sources agree, and an offline model with independent reads gives a fifth of that, so the model lacks an input. Duplicate reads are not it:
the three libraries carry the same, about 1.8 reads per molecule line. run_sample.py --dev-seq-self-cal-dump now writes every site's inputs,
read on the same rows as the sixth form (`selfcal_dump.sh`), for the offline model.
**Range of the data:** in the real libraries, marked-side end calls read opposite 4.4 to 11.5 % and inner calls 3.7 to 6.0 % (the first
synthetic's unplanted rows); the plants add 3 to 8 % at the ends, as before.

**The self-calibration synthetic, sixth form, read: not met** (`selfcal_synthetic6_summary.txt`, the sealed scorer unchanged). Bar 1, end errors absorbed: 1 of 9.
Bar 2, the plant visible without the calibration: median 0.3249. Bar 3, a uniform change kept: 3 of 3. It removes the marked-side plants
(CD8 +0.019, B cells +0.005, against +0.375 and +0.325 without) and takes out too much elsewhere (B cells, unmarked plant, -0.058).
**Where the error sits along the molecule (exploration, after the score; the three spent merged molecule files, molecules of exactly 12 calls,
each call judged by the other 11).** Marked side: a fault at the last call only, flat from the second (B cells 0.229 against 0.039 inside; CD8
0.059 against 0.040; CD4 0.100 against 0.040). Unmarked side: it reaches two to three calls in (B cells 0.041, 0.016, 0.011, then 0.007). Inside
the molecule the three laboratories read alike (about 0.040 marked, 0.007 to 0.010 unmarked). Every form so far measured the end as the last
call and the inner level as every other call, so on the unmarked side the inner reference carried end error. Next: each library's own reach per
side from its own profile, the reference taken beyond it, offline first on the per-site dumps (`selfcal_dump.sh`).

**Per-side reach at the identity sites (exploration; `seq_atlas/perside_reach_summary.txt`).** On the three self-calibration libraries, each library's calls
counted beyond its own per-side reach (marked side past the last call, unmarked side past the third): on the unmarked side the three laboratories
read 0.0041, 0.0032 and 0.0030 opposite (all calls: 0.0089, 0.0179, 0.0032), 0.588 to 0.911 of the atlas, which is itself built with its donors'
end calls; on the marked side they still differ beyond the end (0.0666, 0.0366, 0.0574; 1.127, 0.535 and 0.878 of the atlas), so that
difference is not a molecule-end fault: it is biology (donors, subtypes) or a meter error of another kind, which one library per laboratory
cannot separate. The 54 Loyfer counts (PHY-079) show no end fault in the counter's held error, which is mostly marked-side sites; the atlas's own B-cell donor GSM5652316, read by run_sample.py --build seq-floor-position at the B-cell identity sites, does carry one on the unmarked side (last call 0.0189 opposite, inside 0.0036; site-matched end over inner 5.3086) and a weak one on the marked side (1.1093), so the atlas's unmarked side carries end error too (`seq_atlas/atlas_donor_perside_summary.txt`). Next: the seventh form counts the sample past
its own per-side reach; whether the atlas's unmarked side also needs it is checked on its donors' counts first.

**The atlas donors' own per-side profile (exploration; `seq_atlas/atlas_donor_perside_summary.txt`).** Nine of the atlas's donors (three each of B, CD8 and CD4), read by
the one command's seq-floor-position build at their own cell's identity sites: every donor carries the end fault, unmarked side site-matched
2.9781 to 5.3086, marked side 1.1093 to 1.2259. Past each side's reach the sample and the atlas compare like for like: marked side, CD4 library
0.0574 against its atlas donors' median 0.0582, B 0.0666 against 0.0545, CD8 0.0366 against 0.0619 (the CD8 library reads more marked: biology
or a meter error not at the ends); unmarked side, the three libraries 0.0030 to 0.0041 against the donors' 0.0058 to 0.0062 (a laboratory-level
interior difference, not an end fault). **The seventh form, from this:** the same per-side reach for the sample and the atlas (the atlas rebuilt
from its donors counted past the same reach), the interior difference left to the agreement-site calibration, each library's reach from its
own profile; one library read first as a canary.
