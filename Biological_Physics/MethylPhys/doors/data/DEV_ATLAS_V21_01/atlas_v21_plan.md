# IAM-Atlas v2.1: roster, build/held-out split, culture check, compute and download estimate (prep only)

Prepared 2026-10-12. Metadata, file listings and papers only. No IDAT, beta table, read, BAM or .pat file was downloaded or opened.
Counts come from GEO SOFT (series and per-sample records, `view=brief`), GEO `suppl/filelist.txt`, the ENA read-run report, and the ArrayExpress
SDRF/FTP listing, all queried 2026-10-12, plus the open-access methods of the linked papers (Europe PMC full text). Each row of
[atlas_v21_roster.csv](atlas_v21_roster.csv) names its count source. Paper PMIDs are taken from each GEO series' `Series_pubmed_id` field, and full text from Europe PMC. Treat the counts as provisional until the parent re-queries them with a committed script (the
fetch scripts used here are saved beside this file).

## 1. Roster summary

- **Series verified:** 76 GEO series (all 63 primary class A/B/C GEO series, 5 secondary-B series GSE153459, GSE247198, GSE343621, GSE59163 and GSE128734, and 8 lung-stroma and culture-series records) and 2 ArrayExpress studies (E-MTAB-4957, E-MTAB-3709).
- **v2 baseline (from the repository roster):** 74 cells IN, 314 samples in the fit (296 from `roster_samples.csv` with qc True and cell IN, plus 18 GSE63409 normal arrays at call rate >= 0.93), 7 sources.
- **v2.1 build, firm (passes the entry rule on metadata; twin test still to run):** 428 new samples from 13 series, 6 new cell types; 80 cells. Every new source is independent of the v2 groups by GEO contributor list (no shared surname with GSE186458, GSE110554, GSE167998, GSE122126, GSE215353 in `CANON/geo_contributors.csv`).
- **v2.1 build, conditional:** 118 more samples (32 granulocyte fractions waiting on the D6 decision; 86 from OSE/FTE scraping, sperm, thymocytes on EPIC v2, cord-blood HSC, adipocyte WGBS). With all of them: 91 cells, 860 samples.
- **Held-out:** 1,663 arrays or libraries across 40 series (1,458 arrays, 205 sequencing libraries), plus 2 truth sets (320 arrays) and 6 culture-check series (GSE197512, GSE179847, GSE82234, GSE123995, GSE223153, E-MTAB-4957 organoids).

### What v2.1 adds, by cell (build rows only)

| cell | v2_sources | v2_samples | new_sources | new_samples_firm | new_samples_cond |
|---|---|---|---|---|---|
| HSC (bone marrow) | GSE63409 | 3 | GSE49618 | 3 | 0 |
| NEW: HSC (cord blood) | - | 0 | GSE273074 | 0 | 12 |
| NEW: cd56bright nk cells | - | 0 | GSE252045 | 9 | 0 |
| NEW: cd8 central memory t cells | - | 0 | GSE252045 | 16 | 0 |
| NEW: cd8 temra t cells | - | 0 | GSE252045 | 6 | 0 |
| NEW: central memory th1 cells | - | 0 | GSE75405 | 5 | 0 |
| NEW: cytotoxic cd4 t cells (cd28-) | - | 0 | GSE75405 | 5 | 0 |
| NEW: promyelocytes | - | 0 | GSE49618 | 3 | 0 |
| NEW: sperm | - | 0 | GSE114753 | 0 | 12 |
| NEW: thymocyte DN1 | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte DN2 | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte DN3 | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte DPE | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte DPL | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte ISP | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte SP4 | - | 0 | GSE269608 | 0 | 5 |
| NEW: thymocyte SP8 | - | 0 | GSE269608 | 0 | 5 |
| adipocytes | Loyfer2023,Moss2018 | 6 | GSE111445 | 0 | 6 |
| astrocytes | Tian2023 | 1 | GSE306226 | 10 | 0 |
| b cells | Loyfer2023,Salas2018 | 9 | GSE103541, GSE118144, GSE122244, GSE224807, GSE88824 | 49 | 0 |
| cd4 t cells | Loyfer2023,Salas2018 | 8 | GSE103541, GSE118144, GSE224807, GSE88824 | 43 | 0 |
| cd8 t cells | Loyfer2023,Salas2018 | 9 | GSE103541, GSE118144, GSE224807, GSE88824 | 43 | 0 |
| colon epithelium | Loyfer2023,Moss2018 | 8 | E-MTAB-4957 | 8 | 0 |
| cortical neurons | Loyfer2023,Moss2018 | 11 | GSE306226 | 10 | 0 |
| effector memory cd8 t cells | Loyfer2023,Salas2022 | 5 | GSE252045 | 7 | 0 |
| embryonic stem cells | ENCODE | 2 | GSE116754 | 9 | 0 |
| fallopien tubes epithelium | Loyfer2023 | 3 | GSE146552 | 0 | 8 |
| memory b cells | Loyfer2023,Salas2022 | 8 | GSE252045 | 8 | 0 |
| microglia | Tian2023 | 1 | GSE306226 | 10 | 0 |
| monocytes | Loyfer2023,Salas2018,Salas2022 | 14 | GSE103541, GSE122244, GSE224807, GSE226298, GSE252045, GSE88824 | 55 | 0 |
| naive b cells | Salas2022 | 4 | GSE252045 | 9 | 0 |
| naive cd4 t cells | Loyfer2023,Salas2022 | 6 | GSE252045, GSE75405 | 13 | 0 |
| naive cd8 t cells | Loyfer2023,Salas2022 | 6 | GSE252045, GSE75405 | 20 | 0 |
| neutrophils | Salas2018,Salas2022 | 12 | GSE103541, GSE118144, GSE122244, GSE124565, GSE224807, GSE226298, GSE88824 | 37 | 32 |
| nk cells | Loyfer2023,Salas2018,Salas2022 | 13 | GSE224807, GSE88824 | 19 | 0 |
| oligodendrocytes | Loyfer2023,Tian2023 | 5 | GSE306226 | 10 | 0 |
| ovary epithelium |  | 0 | GSE146552 | 0 | 8 |
| small intestine epithelium | Loyfer2023 | 3 | E-MTAB-4957 | 8 | 0 |
| t central memory cd4 | Loyfer2023 | 3 | GSE252045 | 9 | 0 |
| t effector cell cd8 | Loyfer2023 | 3 | GSE75405 | 4 | 0 |

Main gains: neutrophils get four independent laboratories on arrays (GSE88824, GSE124565, GSE118144, GSE122244: 37 firm samples), with up to 32 more if granulocyte fractions are admitted. Microglia, astrocytes and oligodendrocytes get real donors (GSE306226, 10 each) beside the pooled Tian profiles. Small intestine epithelium, t central memory cd4, t effector cell cd8 and naive B get a second laboratory. Colon epithelium gets a third laboratory (E-MTAB-4957), the first purified colon epithelium independent of the atlas groups (v2 has Loyfer and Moss). Ovary epithelium (WAIT) can be filled only conditionally (GSE146552, scraping-enriched). No candidate was found for the other 15 waiting types.

### Corrections to the gold mine map found in this pass

1. **GSE123995 hepatocytes are cultured:** the paper calls them "primary hepatocyte cultures" from non-transplantable livers (PMC6877651). They move from build to culture check.
2. **Five build candidates are RRBS/ERRBS, not whole-genome:** GSE104405 (ERRBS), GSE223153 (reduced representation, GEO design), GSE118255, GSE343621 and GSE159474 (RRBS). D9 (whole cell, >=10 reads at >=90 % of array CpGs) fails for any reduced-representation library, so they can test only at the loci they cover.
3. **E-MTAB-4957 is 450K only** (array design A-MEXP-2255 for all 134 arrays), not 450K plus EPIC. Its 5 normal sigmoid arrays are the same barcodes as E-MTAB-3709's 5 normal arrays. The two studies are one source.
4. **GSE81224's 20 arrays are all inside GSE146552.** GSE201872 contains GSE85566's 112 arrays. GSE157651's 23 healthy arrays are all inside GSE111396's 26 healthy arrays. None of these pairs can serve as build and test.
5. **GSE16368 (Roadmap):** the ENA run report lists 1,037 runs and no fastq files, so reads are controlled. Its bisulfite-seq breast cells have 2 (luminal) and 3 (myoepithelial) donors, and its keratinocyte, fibroblast and melanocyte are cultured, with 1 donor each. Not used.
6. **GSE224807:** 74 never-smoker donors (69 with sorted 450K arrays; 2 with sorted cells on custom EPIC GPL29753). Sorted by antibody-coated magnetic beads (paper PMC10211291).
7. **GSE116754** is the series v2 already used to measure the ENCODE term (atlas v2 LESSONS 11). Its stem cells are 2 lines (WA09 7 arrays, WA07 2), not donors.
8. **Register gap:** GSE226298 was read by DEV-NEWLAB-GRAN-01 (Met-A bars 10-11) but has no row in `CANON/data_spent.csv`.
9. **GSE271697 placenta:** the paper reports epithelial composition of 80-85 % for included trophoblast fractions. Its own syncytiotrophoblast samples failed its inclusion criteria. Research only.
10. **GSE306226 astrocytes** are the NeuN-/SOX10-/IRF8- "other glia" fraction (paper), not an astrocyte-marker sort. Its co-authors include the Exeter group (Lunnon, Smith, Imm), and GSE234520 is Exeter (Hannon, Mill), so the two are not independent of each other.
11. **Name collision, not overlap:** GSE96612's "K D Hansen" is Kasper Hansen (JHU), not Helen M. Hansen of the Salas group.

## 2. Build/held-out split rule (written before any data is read)

1. **Whole series to one role by default.**
2. **A spent series builds and does not test.** This covers any series already read by any method (`CANON/data_spent.csv` plus repository search). Spent series: GSE88824, GSE124565, GSE118144, GSE122244, GSE103541, GSE226298, GSE224807 (CD15 never-smokers, DEV-METAA-450K-01) and GSE271697. For spent sets with more than 12 donors, the first 12 donor IDs in natural sort order build. The remainder is not used, except for U1 with a one-line **Prior reading:** disclosure.
3. **A large unspent series is split by donor with one fixed rule:** natural-sort the donor IDs as GEO gives them (the field is named per series in the roster). Donors at odd ranks 1, 3, ..., 23 (the first 12 odd ranks) build; every other donor is held-out. The rule applies to GSE224807, GSE306226, E-MTAB-4957, GSE114753 and GSE273074.
4. **The split is at donor level.** All cell types and all arrays of one donor take the same role.
5. **Cap of 12 donors per (source, cell type) in the build.** This keeps any one laboratory from dominating a cell's donor spread, keeps compute near 2x v2, and leaves test halves. (v2's largest cell entries hold 12-14 samples.)
6. **A set used to build cannot test.** A test set must share no GEO contributor with any build source of the cell it tests. Same-laboratory pairs are kept to one role: GSE59065/GSE71955, GSE130029/GSE130030, GSE189255/GSE189256, GSE68456/GSE82084/GSE127824, GSE118255/GSE343621, GSE85566/GSE201872/GSE146376, GSE111396/GSE157651, GSE146552/GSE81224, GSE103541/GSE234520.
7. **The donor lists are written now**, in the `donor_ids` and `samples` columns of the roster. Any change is a new dated entry, never an edit.

### Held-out assignments by use

- **Commissioned Met-A neutrophil reading (EPIC v1 scope):** GSE184269 granulocytes (10 donors, FACS, unspent) and GSE247198 (Vilnius, 48 neutrophil arrays from 2 donors, time series, unspent). GSE118144 and GSE122244 are already spent by the commissioning, so they build instead. GSE224807's held-out CD15 half (54 donors, 450K) serves the 450K route only, with a Prior reading line.
- **U1 profiles:** GSE184269, the GSE224807 held-out half, GSE59065 (99 CD4 and 100 CD8; young and old arms), GSE87640, GSE87095, GSE131989 (female; UCSF check first), and smaller sets: GSE130029, GSE130030, GSE121192, GSE112696, GSE189255, GSE189256, GSE188573, GSE85647, GSE100561, GSE104287, GSE153459. Sequencing: GSE173787, plus GSE118255 and GSE159474 at RRBS loci. Brain: the GSE306226 held-out half, GSE112179, GSE66351, GSE98203, GSE191200. HSC: GSE104405 at ERRBS loci.
- **U2 blind sites (tissue tests):** the colon epithelium held-out half (E-MTAB-4957, 8 donors). Lung bronchus epithelium from GSE104471 (12 normal brushings), GSE85566 (41 controls) and the non-overlapping GSE201872 EPIC arrays. Lung alveolar macrophages from GSE132547 (12 donors x 2 lobes; purity and Dartmouth check first). Blood types as for U1.
- **U3 blood composition (truth sets):** GSE127824 (24 cord bloods, FACS counts, 7 fractions) and GSE167202 (296 COVID-negative ED whole bloods with clinical CBC differentials; provisional, patients not healthy, custom EPIC). Building nothing from the UBC laboratory keeps GSE127824 independent.
- **U4 identity sites:** the held-out halves of GSE224807 and GSE306226, plus GSE184269 and GSE59065. These are per-cell readings on donors that no atlas source contains.
- **IAM-A:** GSE107729 and GSE96612 (the two neuron test laboratories named in DEV-IAMA-NEURON-01), GSE128734 (same DNA on arrays and sequencing, 2 donors), and GSE59163 (neutrophil RRBS; the C-score null fails on RRBS, lesson T8).

## 3. Lung stroma and the culture check

**Facts (GEO).** Every public healthy lung stroma set found is cultured:
- GSE146376 (Chicago): primary cultured airway smooth muscle, 54 non-asthmatic donors in the vehicle arm, EPIC v1.
- GSE111396 (Nottingham): lung fibroblasts at passage 4, 26 healthy donors (18 parenchymal, 8 airway), 450K. GSE157651's 23 healthy arrays are a subset of these.
- GSE107226: 4 normal cultured lung fibroblasts, 450K.
- GSE135097: 2 normal untreated cultured lung fibroblasts, 450K.

No uncultured, sorted lung fibroblast or smooth-muscle set exists in GEO or ArrayExpress (gold mine map, 12 ArrayExpress queries). These sets also all sit at one passage, so no within-set passage trend can be measured.

**Proposed culture check (fixed before any array is read):**
1. **Division-drift CpGs.** Take the serial-passage series and, within each line, compute the Spearman correlation of beta with population doublings (or passage):
   - GSE197512 (Van Andel, EPIC v1): 182 baseline arrays from 7 Coriell lines with population doublings, including vascular smooth muscle (19 arrays), vascular endothelium (31), and skin, foreskin and fetal fibroblasts.
   - GSE179847 (Columbia, EPIC v1): 92 control arrays at 21 % oxygen from 6 normal fibroblast lines, with passages.
   - GSE82234 (Aachen, 450K): HUVEC at passage 4 vs late passage, 3 donors.

   A CpG is division-drift if it is monotone with doublings (|rho| >= a cut fixed now, for example 0.5) with the same sign in at least 2 lines of 2 cell types. These sets become reference data for the check, not tests.
2. **Size the culture effect on cells that exist both cultured and uncultured:**
   - (a) E-MTAB-4957: purified epithelium vs organoids, same laboratory, juvenile sigmoid 18 and ileum 18 organoid arrays from 9 donors each. Only 1 donor is shared with the purified fractions. The paper reports no DMPs between low- and high-passage pediatric organoids.
   - (b) GSE123995: cultured hepatocytes (56 donors) vs atlas hepatocytes (Loyfer, Moss). The laboratory offset is confounded with culture here, so the source term must be fitted first.
   - (c) GSE223153: fresh-sorted vs organoid acinar cells, same laboratory, RRBS loci only.

   For each, report the share of CpGs where the cultured profile falls outside the atlas 90 % interval, before and after masking the drift set.
3. **Pass rule for a cultured cell type to enter:** after the mask, cultured profiles of cell types with an uncultured reference fall inside the atlas interval at the U1 bar (80-97 % of sites) in all of (a)-(c), and what is left outside is not enriched for cell-identity CpGs.
4. **Only then compare lung stroma.** After the mask, compare GSE111396 fibroblasts with Loyfer colon and heart fibroblasts, and GSE146376 smooth muscle with Loyfer smooth muscle (5 samples). Whether Loyfer's stromal samples were themselves cultured must be read from Loyfer 2023 Supplementary Table 1 first; it is not in the files read here.

**Recommendation: lung stroma stays research-only in v2.1.** Even a passing mask cannot separate organ identity from culture for a cell type with no uncultured profile. The lung test would then rest on an untested stand-in, the failure mode of lesson T16. Use cultured lung stroma only as a named sensitivity panel when choosing U2 blind sites for lung tests, and say so in each lung seal. Revisit if an uncultured sorted lung-stroma methylome appears; controlled-access and single-cell atlases were not searched.

## 4. Compute and download estimate (nothing launched)

**Cost model.** Stage B fits one NUTS run per block of loci. The cost per gradient grows with the observations in the block (samples x loci) and the parameters (2 x cells + 1 per locus). The assumed scaling is

    s/locus(v2.1) = 0.829 s x (N_obs_eff + 2 N_cells + 1) / (N_obs_eff,v2 + 2 x 74 + 1) x f_tree

where 450K arrays count 0.55 of an observation (they cover about 450k of the ~814k loci), and f_tree = 1.0 (central) to 1.15 (upper; a larger model may take more leapfrog steps under max_tree_depth 8). v2: 314 samples, effective 301.

| case | cells | samples | effective obs | s/locus (central - upper) | wall, 16 processes | wall, 32 processes |
|---|---|---|---|---|---|---|
| v2 (measured block 0) | 74 | 314 | 301 | 0.83 | 11.7 h | 5.9 h |
| v2.1 firm | 80 | 742 | 652 | 1.5 - 1.86 | 21.2 - 26.3 h | 10.6 - 13.1 h |
| v2.1 with all conditional | 91 | 860 | 747 | 1.71 - 2.15 | 24.1 - 30.3 h | 12.1 - 15.2 h |

The machine is c7a.32xlarge (128 vCPU, 246 GB), and the loci stay those of v2: every array CpG with Loyfer depth >= 10 in >= 90 % of samples, ~814k.

**Discrepancy to resolve.** The README says 32 processes on 128 cores. `15_stageB_run_all.sh` defaults to 16 processes of 8 cores, because 32 at once (~420 GB at ~13 GB per block) were killed out of memory on 2026-09-28. Memory per block scales about like the cost (1.8-2.3x), so 1,163-locus blocks would peak at ~23-30 GB, and only ~8 would fit on 246 GB at once. Keep 16 or more in parallel by splitting into NBLOCK = 1,400 (~580 loci per block, ~12-15 GB), or rent a larger-memory 128-vCPU box. Plan on **about 21-30 h on the v2 box at 16 processes**, or 10.5-15 h if 32 fit. Add the upstream steps: Stage 1 on 546 new build arrays (time not recorded in the v2 files read), the closed-form source terms (step 12), and the twin test (step 10).

**Source terms.** Each new array source needs a Deming term from a cell type it shares with an existing source (step 12). The firm sources all share at least one cell. Sperm, thymocytes and cord-blood HSC share none, so their terms would be ASSUMED (D8), one more reason they are conditional. OSE can borrow only from FTE in the same series.

**Download (selected samples only; per-sample files, not whole RAW.tar):**

| item | GB | source |
|---|---|---|
| build arrays, firm (428) | 5.4 | GEO filelist.txt sizes; E-MTAB at 2 x 7.7 MB per array |
| build arrays, conditional | 1.4 | same |
| build WGBS raw reads, GSE111445 (6 libraries; conditional) | 215.1 | ENA fastq_bytes |
| held-out arrays (1,458) | 14.5 | GEO filelist.txt |
| held-out sequencing, GEO processed files (where present) | ~24 | GEO filelist.txt (GSE107729 5.6, GSE173787 11.4, GSE128734 4.6, RRBS sets ~2) |
| held-out sequencing raw reads (all) | 10343.0 | ENA: GSE107729 7468.7, GSE128734 1061.9, GSE96612 938.5, GSE173787 716.8 |
| truth sets | 4.6 | GEO filelist.txt |
| culture check (arrays; GSE223153 RRBS raw 46.2) | 5.5 | GEO / AE |
| lung stroma (research) | 1.3 | GEO filelist.txt |

The array work for the build is about 7 GB. Everything sequencing-based is large: the IAM-A neuron reads alone are 8.4 TB. Fetch them by series when the IAM-A test is designed, not for the atlas build.

## 5. Open questions for the author (each with a recommendation)

1. **Granulocyte fractions as neutrophils (D6 put "granulocytes" OUT as a mixture).** The fractions are GSE224807 CD15+ beads, GSE103541 FACS granulocytes, GSE226298 and GSE184269 granulocytes. *Recommendation:* admit only marker-selected fractions (CD15+ or CD16+), and only if each passes the sample-level twin test against the Salas neutrophils and its eosinophil share (estimated with the atlas) is under a cut fixed now. *Reason:* CD15 selects neutrophils plus eosinophils. D6 was written for unsorted granulocyte labels, and the chain already treated GSE224807 CD15 as neutrophils in DEV-METAA-450K-01.
2. **Cap of 12 donors per source per cell.** *Recommendation:* adopt it. *Reason:* it matches v2's largest entries, holds compute near 2x, and leaves test halves in the large series.
3. **450K-only new cells (promyelocytes, OSE, sperm, cord HSC).** *Recommendation:* allow them under the GSE63409 precedent, with EPIC-only loci NOT MEASURED (`n_obs` = 0). *Reason:* D9 forbids filling. It does not forbid a whole-platform measurement, and v2 already holds six 450K-only bone-marrow cells.
4. **EPIC v2 thymocytes (GSE269608).** *Recommendation:* defer to v2.2 unless the Stage 1 EPIC v2 read check is done first. *Reason:* the chain refuses EPIC v2 today, eight adjacent subsets from one laboratory risk twin failures, and the term would be ASSUMED.
5. **New cells no use needs (sperm, thymus, cord HSC, placenta).** *Recommendation:* leave them out of v2.1. *Reason:* no U1-U4 use or tissue test needs them, each adds an ASSUMED source term and compute, and the build stays focused on the cells the commissioning plan reads.
6. **Spent sets as builders, unspent sets as tests.** *Recommendation:* adopt as the standing rule, and add GSE226298 to `CANON/data_spent.csv` first. *Reason:* a set already read cannot give a sealed test; building from it costs nothing.
7. **GSE306226 "astrocytes" (triple-negative glia).** *Recommendation:* enter flagged. Keep them only if, at sample level, they are closer to Tian astrocytes than to any other atlas cell, with cross-laboratory r at least that of oligodendrocytes (Tian vs Loyfer 0.986); otherwise research. *Reason:* the fraction also holds endothelium, pericytes and other non-labeled nuclei.
8. **Cord blood (UBC).** *Recommendation:* build nothing from GSE68456/GSE82084. *Reason:* building from them would make GSE127824, the only FACS-counted truth set outside the Salas group, in-sample.
9. **Colon and ileum epithelium from children (E-MTAB-4957).** *Recommendation:* accept, flagged pediatric (6-15 y), and confirm what the "n" suffix arrays (39ACn, 39SCn, 39TIn, all in the held-out half) are before the split is sealed. *Reason:* it is the only purified colon epithelium independent of the atlas groups. The suffix may mark a different fraction; if it is a negative fraction, drop those 3 arrays.
10. **GSE146552 OSE/FTE (scraping, not sorting).** *Recommendation:* conditional entry. After the source term, deconvolve each array with the atlas and keep the set only if non-epithelial content is under a cut fixed now; otherwise research. *Reason:* this is the only candidate for a WAIT type, and purity is not stated.
11. **Adipocyte WGBS (GSE111445).** *Recommendation:* defer. *Reason:* adipocytes already have 2 sources; the set needs ~215 GB of alignment and a depot decision (the paper reports distinct subcutaneous and visceral methylomes).
12. **Compute layout.** *Recommendation:* confirm whether the 0.83 s/locus came from 16 x 8-core or 32 x 4-core processes, and run v2.1 with NBLOCK = 1,400. *Reason:* the README and the run script disagree, and v2.1 blocks at the v2 size would not fit 16 at once.
13. **GSE132547 alveolar macrophages.** *Recommendation:* use as held-out U1/U2 only after the paper confirms how BAL macrophages were enriched and that the authors (Armstrong, Chen) are not in the Christensen/Salas group. *Reason:* the Dartmouth institute is shared, and purity is not in GEO.
14. **GSE131989 (UCSF).** *Recommendation:* confirm from the paper that subjecttype 2 means control, and that the group is separate from Wiencke's, before using it. *Reason:* the institute is shared with an atlas reference group.
15. **GSE224807 sex.** *Recommendation:* infer sex from X/Y probes at Stage 1 (a data step after the split is sealed) and record it beside each reading. *Reason:* GEO gives no sex field for GSE224807, GSE103541 or GSE88824 (GSE88824's paper gives 5 F / 3 M overall, not per array).

## 6. Limits of this prep

- Nothing was downloaded or opened beyond metadata, file listings and paper text. Purity, sex (where GEO gives none), coverage and twin status are data steps and remain open.
- **Papers not read** (not open access, or no paper linked): GSE124565 (30471352), GSE132547 (31356157), GSE96612 (30643296), GSE114753 (28950428), GSE269608 (40753575), GSE49618 (23634996), and GSE184269's first paper (34706222). GSE88824's paper (PMID 30571772, PMC6301777) is open access and its methods were read: 8 subjects, 5 females and 3 males, mean age 39; the sorting method is not in the extracted text. No paper is linked for GSE75405, GSE226298, GSE122244, GSE191200, GSE111396 or GSE157651. For these, purification and health facts come from GEO only.
- Health status is not explicit in GEO for GSE103541 or GSE252045. GSE87640's "HC" group includes symptomatic non-IBD ("HL") donors.
- Counts by rule: donor IDs were parsed from GEO fields named per series (titles, source names, characteristics). Where a series gives no donor field, each array counts as a donor (marked in the roster); those donor counts are provisional.
- **Not re-verified:** GSE41826 and GSE87196 (no public raw data, per the gold mine map). No new search was run beyond the gold mine inputs: EGA, dbGaP, IHEC, ENA-only and CNCB were not searched, so absence statements carry over as provisional.
- **Compute:** the estimate is a scaling argument from one measured block (block 0: 1,163 loci, 74 cells, 964 s). It is not a dry run; a 1,000-locus dry run of the v2.1 roster on the box should replace it.
