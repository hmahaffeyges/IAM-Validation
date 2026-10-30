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
**Prior reading:** the spent series named in this record (GSE103541, GSE118144, GSE122244, GSE167998, GSE226298, GSE271697) are BUILD sets of the atlas; none is a commissioning test set. The U1-U3 test sets are the plan's held-out assignments, unopened. GSE87197 (GSE87196's megakaryocytes) donors 1 and 2 are spent (the absorption test and the provisional column's test) and are read again to build the sequencing atlas's megakaryocyte column; donor 3 is unread.

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
