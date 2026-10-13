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

