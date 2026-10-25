# DEV-REPLICATE-01: the earlier work, replayed through the chain (tuning)

**The instrument's job:** the chain, one command, reads every array the earlier instruments read, and we see whether the signals they found
are found again. Tuning, not commissioning: only data the earlier work already opened; every data set no instrument has opened stays unread.

## R1: the earlier CPG engine's 11 test arrays

The arrays its test-data manifest lists, fetched from GEO by accession (the IDAT fingerprints are checked against the author's local copies):
rheumatoid arthritis cases and controls (GSE42861, whole blood, 450K), three healthy bloods (GSE87571, 450K), a high-grade colon adenoma
(GSE288652, EPIC) and colorectal carcinoma at stages 1 and 4 (GSE166212, EPIC). Blood: specimen whole blood, the controls and healthy bloods
as references (`r1_blood_samples.csv`); tissue: specimen colon tissue (`r1_tissue_samples.csv`). Scorer `data/DEV_REPLICATE_01/r1_score.py`.

**What the earlier engine reported, written before the chain reads them:** epithelial cells read near zero in healthy blood and appear in
the colon tissues, rising from stage 1 (about 13 % of the sample) to stage 4 (about 25 %), with its disorder reading rising with stage;
one healthy blood (GSM2333905) carried a stem-cell-class elevation; RA cases departed from controls in their immune cells.

**What the chain should show if it agrees:** epithelial (colon) cells found only in the tissues and more at stage 4 than stage 1; the
whole-sample Met-A and the territories higher in the carcinomas than the adenoma, and higher at stage 4; the bloods near one another.

**R1, tissue (series run 40027190291, chain commit 7dc86ca4; `data/DEV_REPLICATE_01/r1_tissue/r1_tissue_summary.txt`).** The IDAT fingerprints
match the author's copies. Stage 0 stopped the stage 1 carcinoma GSM5065990 at intake: bisulfite conversion 0.749, below 0.80
(`r1_tissue/GSM5065990_intake.txt`); the earlier engine had read it. On the other three, untared, development:

| sample | cells found (3 % or more) | whole-sample Met-A | islands | domains |
|---|---|---|---|---|
| adenoma GSM8772491 | colon epithelium 29 %, CD8 TEMRA T 18 %, embryonic stem 10 %, vascular endothelium 9 %, pancreatic acinar 8 %, oligodendrocytes 6 %, regulatory T 5 %, prostate epithelium 4 %, memory B 3 % | 1.14022 | 1.5826 | 1.07795 |
| adenoma GSM8772492 | colon epithelium 42 %, enteroendocrine 13 %, vascular endothelium 11 %, lung bronchus 9 %, CD8 TEMRA T 8 %, memory B 6 %, oligodendrocytes 4 % | 1.21579 | 1.44186 | 1.19155 |
| stage 4 carcinoma GSM5065985 | enteroendocrine 27 %, CD8 TEMRA T 20 %, prostate epithelium 15 %, memory B 11 %, colon epithelium 8 %, pancreatic acinar 8 %, lung bronchus 6 %, smooth muscle 3 % | 1.08983 | 1.14323 | 1.14104 |

Epithelium is found in every tissue, as the earlier engine found; the stage comparison waits on the refused stage 1 array. Cells no healthy colon
holds (stem, pancreas, prostate, lung, brain support cells) are how the IAM-Deconvolver explains a tumor that matches no healthy cell. The islands
read highest in the two adenomas; the domains do not follow stage across these three. Met-A for each cell was withheld on both adenomas (50 to
82 % of identity sites measured, under the 90 % rule); from here each cell is read when at least half its sites are measured, flagged below 90 %.

**Three faults R1 found in the chain, fixed before the rerun.** (1) The globe drew a healthy reference only for blood and isolated cells; for
a tissue it looked for a cell named after the specimen. It now uses the person's own mix as the IAM-Deconvolver found it for every mixed
specimen (tissue, plasma, unnamed), mixed over the cells with an atlas value at each site (kept at 99 % of the shares, read_in_mix's rule).
(2) The information sphere skipped without a word when the globe was not drawn; it now records why. (3) Each cell's Met-A was withheld below
90 % of its identity sites measured; it is now read at half or more and flagged below 90 %. Also: the report's new sections first carried a
word its claim check refuses, so the one command stopped at the report; reworded. Checked on the healthy blood GSM7508429 before the rerun: globe
drawn, 45 of 77,355 blocks changed as before; the sphere read 27 territories.

**R1, blood (series run 40027130915, chain commit 7dc86ca4; `data/DEV_REPLICATE_01/r1_blood/r1_blood_summary.txt`).** Seven 450K whole bloods,
all read, flagged; the commissioned Met-A and C-Score refuse 450K (no frozen 450K neutrophil reference), as they should. The GSE87571 files are the
author's files; the GSE42861 files are recompressed copies whose decompressed contents are the same (`r1_blood/GSE42861_content_check.txt`). Untared:

| array | group | whole-sample Met-A | islands | domains |
|---|---|---|---|---|
| GSM1051525 | rheumatoid arthritis | 1.03598 | 0.99028 | 1.05523 |
| GSM1051526 | rheumatoid arthritis | 1.0268 | 0.96407 | 1.05287 |
| GSM1051533 | control | 1.01161 | 0.97932 | 1.03786 |
| GSM1051534 | control | 1.02966 | 0.99519 | 1.06715 |
| GSM2333901 | healthy | 0.96081 | 0.96839 | 0.94645 |
| GSM2333905 | healthy | 0.96259 | 0.97307 | 0.95173 |
| GSM2333950 | healthy | 0.97892 | 0.98739 | 0.94809 |

The largest difference is between the two series, cases and controls alike: the per-array offset the atlas ruler is to remove. Two cases
against two controls show no difference. The earlier engine's stem-cell elevation in GSM2333905: the IAM-Deconvolver finds bone-marrow stem
cells (HSC) at 6.6 % there and 6.4 % in GSM2333901, and 450K bloods elsewhere (EPIC-Italy) also carry progenitor shares, so it reads as the 450K
platform absorbed into a progenitor share rather than something of that person.

**Speed (DEV-SPEED-01).** Profile 1 (explore run 40030041288): 585 of 837 s in the globe, 519 s of it in one site-name lookup (np.isin, 52 calls).
A hashed lookup gives the same count (260 of the test blocks) in 0.04 s instead of 83 s; a whole blood through the one command took 29 s on the
laptop afterwards.

**R1 tissue rerun, with the three fixes (series run 40032094322, chain commit 1369dd0f; `data/DEV_REPLICATE_01/r1_tissue_b/r1_tissue_b_summary.txt`).**
All three fixes hold on real tissue: the globe is drawn on each (changed blocks rise from the two adenomas to the stage 4 carcinoma), the
sphere reads 26 or 27 territories, and Met-A reads 75 to 79 cells on each. The sphere lights no territory even on the carcinoma: a change
spread over the whole sample is scaled away by its own territories, the blind spot the plan already carries.

## R2, smoking, and R3, aging (written and committed before any of these arrays is read by the chain)

The earlier engine's smoking and aging tests (CPG-VAL-022, CPG-VAL-017) used GSE50660 and GSE40279, which post no IDATs on GEO (their
file lists hold only the array manifest, signal intensities and processed values), so the chain, which starts from each array's own control
probes, cannot read them. The same two questions are asked on two series with IDATs, both already opened (spent, for tuning):

- **R2, smoking: GSE42861** (450K, 689 whole bloods; 354 rheumatoid arthritis and 335 controls; smoking recorded: current, former, occasional,
  never; `data/DEV_REPLICATE_01/r2_smoking_labels.csv`). Earlier result to compare: CPG-VAL-022 on GSE50660, current against never smokers,
  d +0.11, p 0.65 (22 current smokers), NULL.
- **R3, aging: GSE87571** (450K, 732 healthy whole bloods, ages 14 to 94; `r3_aging_labels.csv`). Earlier result to compare: CPG-VAL-017 on
  Hannum, the immune reading falling with age, r -0.197.

**The reading.** Every array by `run_sample.py` (`--cell-panel atlas --dev-metaa-allsites`), pass 1, untared, through `series.yml` in two
batches each; the labels are not used until every reading is in. **Compared, once every array is read:** for R2, current against never
smokers (all, and the controls alone), and for R3, each reading against age (rank correlation), on: the cell shares (neutrophils and the
lymphocytes first); Met-A for the whole sample and each cell; the territories; the distance to the ceiling. **Positive control, R2:** the
chain's own calibrated value at cg05575921 (the AHRR site that smoking lowers most strongly in blood), current against never; if the chain
does not show it, its intake or calibration is wrong. R3 has its own: cg16867657 (ELOVL2), which rises with age in blood. Both are recorded by `run_sample.py` as `control_sites` (values only, no reading uses them). The scorer, `data/DEV_REPLICATE_01/r23_score.py`, is committed before any reading exists.

## R4, Alzheimer's disease: AIBL (written and committed before any of these arrays is read by the chain)

**Data.** GSE153712 (AIBL, EPIC v1, raw IDATs): 726 whole bloods, 471 healthy controls, 161 Alzheimer's disease, 94 mild cognitive
impairment (`data/DEV_REPLICATE_01/r4_aibl_labels.csv`; GEO records no age). Spent: the earlier engine read it (CPG-VAL-008, 009, 012).
**Earlier results to compare** (`r4_earlier_results.txt`): its per-cell reading lower in Alzheimer's for most cell types, strongest in
eosinophils, neutrophils, monocytes and B cells; its distance from healthy higher in Alzheimer's. **The reading:** every array by
`run_sample.py` (`--cell-panel atlas --dev-metaa-allsites`), pass 1, two batches. **Compared** by `r23_score.py r4`: Alzheimer's and mild
cognitive impairment each against healthy controls, on the cell shares, Met-A for the whole sample and each cell, the territories and the
distance to the ceiling. EPIC v1 whole blood is the platform and specimen the chain is commissioned on, so here the readings are as close to
the commissioned ones as any replay gets.

## R5 to R9: the rest of the earlier engine's series with IDATs (written and committed before any of these arrays is read by the chain)

Each array by `run_sample.py` (`--cell-panel atlas --dev-metaa-allsites`), pass 1, through `series.yml`; specimens named as the chain names
them, and a sorted or cultured cell is named "unknown" so the IAM-Deconvolver must say what it is. All spent (the earlier engine read them).
Earlier results: `data/DEV_REPLICATE_01/r5_r9_earlier_results.txt`. Scorer: `replay_score.py`, committed now.

| Run | Series | Arrays (with IDATs) | Specimen | Compared |
|---|---|---|---|---|
| R5 blood | GSE87650 | 384 whole bloods | whole blood | Crohn's (CD), ulcerative colitis (UC) against healthy (HC; HS and HL as the series labels them) |
| R5 sorted | GSE87650 | 240 (monocytes, CD4, CD8, whole blood of the sorted sub-experiment) | unknown | each cell type: what the Deconvolver finds; CD and UC against HC within each |
| R6 | GSE69138 | 185 stroke bloods | whole blood | the three stroke subtypes against each other (no controls with IDATs) |
| R7 | GSE269244 | 238 prostate tissues | tissue | tumor against normal |
| R8 | GSE49149, GSE60274, GSE74071, GSE84274 | 325 pancreas, brain and aorta tissues | tissue | tumor against adjacent or normal; aortic disease against normal |
| R9 | GSE180683 | 76 bloods of glioma patients | whole blood | glioma groups; time points |
| R9 | GSE84395 | 39 cultured pulmonary endothelial cells | unknown | PAH against control |

GSE256092 (VAL-097) posts no IDATs for its samples on GEO, so it is not read. **What each tissue run shows first is the Deconvolver's
answer:** a prostate, a pancreas, a brain and an aorta should each read mostly as that tissue's cells in the atlas, and a sorted monocyte as
monocytes. That is the plainest check of the chain on specimens it is not commissioned on.

**R2 smoking, read** (`data/DEV_REPLICATE_01/r2/r2_summary.txt`; all 689 arrays, pass 1, untared; series runs in `r23_runs.json`'s R2 ids,
commit c66d07a1). **Positive control met:** at AHRR (cg05575921) current smokers read 0.595 against never-smokers' 0.853 (d -2.83,
p 6.5e-57), and at F2RL3 (cg03636183) 0.622 against 0.739 (d -2.04), the known direction and size; the chain's intake and calibration read
another laboratory's 450K arrays correctly. **Met-A:** no clear smoking effect (whole sample d -0.17, p 0.19; neutrophils d -0.17;
territories d -0.11 and -0.22), as the earlier engine found on GSE50660 (d +0.11, NULL). Among the controls alone, neutrophils' Met-A
reads slightly lower in current smokers (d -0.42, p 0.023, one of many readings, uncorrected). **In plain words:** smoking moves a few
sites strongly, which the chain sees; it does not make blood's held marks more disordered.

**R3 aging, read** (`data/DEV_REPLICATE_01/r3/r3_summary.txt`; 731 of 732 arrays, pass 1, untared; ages 14 to 94). **Positive control
met:** ELOVL2 (cg16867657) rises with age, rho +0.956, the strongest known age site, read correctly. **The cell mix ages as expected in
part:** CD4 T cells fall (rho -0.537), NK and CD8 T cells fall, monocytes rise (+0.363); B cells rise (+0.243), recorded as the instrument
gave it. **Met-A rises with age:** whole sample rho +0.209 (p 1.2e-08), most in the sites outside the identity sites, islands and domains
("the rest", +0.314); the islands do not move (-0.007). Neutrophils' own Met-A rises (+0.143). The earlier engine's immune reading fell with
age on Hannum (r -0.197); the direction differs because that reading was not Met-A (higher meant healthier there), so the two are not
compared as like for like. **In plain words:** held marks become more disordered with age across blood, as the physics expects for marks
copied over more divisions, and the chain measures it in one person against the atlas with no cohort.

**R4 Alzheimer's, read** (`data/DEV_REPLICATE_01/r4/r4_summary.txt`; all 726 AIBL arrays, EPIC, pass 1, untared). In that file the
columns labeled "current" and "never" (the scorer's R2 wording, reused) mean the named group and the healthy controls. **The cell mix
moves:** Alzheimer's bloods carry more neutrophils (median 0.609 against 0.575, d +0.30, p 0.0011), the direction the literature reports
(a raised neutrophil share in Alzheimer's blood). **Met-A does not:** whole sample d +0.09 (p 0.38), every cell d +0.04 to +0.15, none
significant; mild cognitive impairment the same (islands d +0.24, p 0.034, one of many readings, uncorrected). The earlier engine reported
a per-cell reading lower in Alzheimer's for 106 of 115 cell types and a distance from the healthy centroid of d +0.200; the chain finds the
mix change and no disorder change once the mix is taken out. The shares of the five panel names (CD4, CD8, B, NK) read 0 here because the
79-cell atlas names these cells more finely; the finer shares are in each bundle. **In plain words:** in Alzheimer's blood the chain sees
more neutrophils, not more disordered marks.

**R5 Crohn's and colitis, whole blood, read** (`data/DEV_REPLICATE_01/r5/r5_ibd_blood_summary.txt`; 382 of 384 arrays of GSE87650, 450K, pass 1,
untared; groups as GEO labels them: CD Crohn's disease 103, UC ulcerative colitis 101, HL healthy 103, HS symptomatic controls without
inflammatory bowel disease 75). **Earlier engine:** its test on this series (VAL-128) used only the sorted-cell sub-experiment, so the whole
blood has no earlier result to compare; R5's sorted cells are still running.

- **Met-A, the territories and the whole-genome C-Score do not separate the groups** (Crohn's against healthy: whole-sample Met-A
  1.0301 against 1.0310, d +0.04; islands 1.0801 against 1.0902; C-Score 1.0092 against 1.0085).
- **The smoking site does:** AHRR cg05575921 reads lower in Crohn's than in healthy (0.8080 against 0.8620, d -0.52, p 0.00019) and than in
  colitis (d -0.57), while colitis and healthy match (d -0.05). Smoking is an established risk factor for Crohn's disease and is protective
  in colitis, so this is the known biology appearing through the chain's own calibrated values.
- **The IAM-Deconvolver's largest shares:** neutrophils 0.57 in Crohn's and 0.52 in colitis against 0.42 in healthy, the raised neutrophil
  share of active inflammation. The 450K bone-marrow progenitor shares (GMP, HSC) appear again, as on every 450K series, and are flagged.

**R5 Crohn's and colitis, sorted cells, read** (`data/DEV_REPLICATE_01/r5/r5_ibd_sorted_summary.txt`; 172 of 240 arrays of GSE87650, pass 1,
untared): the Deconvolver finds each sorted cell type as its largest share (CD4 T cells 0.73 to 0.82, CD8 T cells 0.59 to 0.70, monocytes 0.81
to 0.86). Crohn's or colitis against healthy within the same cell type (8 comparisons, 11 to 20 arrays a group): no reading reaches p 0.05.

**R6 stroke, read** (`data/DEV_REPLICATE_01/r6/r6_stroke_summary.txt`; 154 of 185 arrays of GSE69138): the three stroke subtypes against each
other; no Met-A, territory or C-Score reading reaches p 0.05 (closest: the islands, cardioembolic against small-vessel disease, p 0.055). The
series has no controls with IDATs.

**R7 prostate, read** (`data/DEV_REPLICATE_01/r7/r7_prostate_summary.txt`; 169 of 238 tissues of GSE269244, pass 1, untared): tumor against
adjacent normal. **The islands territory reads higher in tumors** (1.4546 against 1.3568, p 5.1e-06), the early gain at islands the territory
was built to read. Also: Met-A monocytes 0.9958 against 1.0231 (p 0.00036) and the whole-genome C-Score 0.9966 against 0.9998 (p 0.00019),
both lower in tumors. Whole-sample Met-A does not separate them (p 0.98). The tissue mixes differ (normal: vascular endothelium 0.35, prostate
epithelium 0.30; tumor: prostate epithelium 0.31, vascular endothelium 0.23), and blood-cell readings in a tissue are flagged, not commissioned.

**R8 tissues, shard 3 re-sent:** shard 3 of the 18 (every 18th array from the fourth, `data/DEV_REPLICATE_01/r8_tissues_shard3_samples.csv`) ran out of the machine's time limit; the same arrays are read again with the same settings (tissue, `--cell-panel atlas --dev-metaa-allsites`, pass 1), split over 10 machines; nothing else changes.

**R9 glioma blood, shards 0 and 9 re-sent:** two of its 10 shards ran out of the machine's time limit; their arrays (`data/DEV_REPLICATE_01/r9_glioma_blood_shards0_9_samples.csv`) are read again with the same settings (whole blood, `--cell-panel atlas --dev-metaa-allsites`, pass 1), split over 10 machines; nothing else changes.
