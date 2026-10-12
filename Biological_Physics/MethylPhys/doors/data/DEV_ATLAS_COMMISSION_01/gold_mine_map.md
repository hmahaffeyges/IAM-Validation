# IAM-Atlas v2: purified healthy cell candidates, classified and mapped (2026-10-12)

Metadata only. No IDAT, beta table, read, BAM or .pat file was downloaded or opened. Every count below comes from GEO SOFT metadata (series + per-sample, `view=brief`, queried 2026-10-12 in this run), ArrayExpress SDRF/BioStudies metadata, the 2025 compendium `metadata.parquet` label list (HuggingFace ylab/methyl-classification), or input 1; each CSV row names its count source.

## Scope

- Rows classified: **169** series. 130 unique accessions from input 1 (99 array, 27 sequencing and 6 mixture rows; GSE41826 and GSE128734 each appear in two lists and are merged into one row each), 15 from the compendium re-check, 18 from five targeted GEO priority searches, and 6 from ArrayExpress.
- Atlas cell types: input 2 lists **87** types, not the 74 given in the task. All 87 are mapped below, and the count discrepancy is noted as a deviation.
- Compendium re-check: the per-sample label list (16,959 GSMs, 107 original labels, 274 series) was re-read. Purified-cell labels missing from input 1: **sperm** (8 series) and **embryonic stem cell** (GSE86648, GSE95761; GSE116754 had been dropped). All were added. Leukocyte (whole blood) and PBMC labels (about 26 series) are not purified and were not added.

## Counts by primary use class

| class | meaning | primary | as secondary |
|---|---|---|---|
| A | BUILD | 28 | 5 |
| B | HELD-OUT TEST | 35 | 14 |
| C | TRUTH SET | 2 | 1 |
| D | RESEARCH | 17 | 57 |
| E | NOT USABLE | 87 | 0 |

Not-usable reasons, primary E (counted from the `not_usable_reason` column):

- cultured, no uncultured comparison / cell line: 28
- not purified: 18
- no raw data / controlled access: 13
- same group as atlas / ENCODE: 10
- not healthy: 9
- too few donors: 6
- redundant / duplicate: 2
- platform not supported: 1

## Corrections to input 1 found in this pass

- GSE87196 (BLUEPRINT progenitors): raw reads are at EGA under a data-access agreement (GEO design text), not SRA. Moved to E (controlled access); per-sample processed calls are public.
- GSE104700 (NIEHS monocyte RRBS) and GSE324619 (Tulane monocyte WGBS) have no public reads: the first is withheld under NIH GDS policy, the second is going to a controlled NIH biobank.
- GSE166844: RAW.tar contains only the EPIC manifest (GEO filelist.txt), so there are no IDATs.
- GSE43976: CD14 monocytes come from MS patients (GEO summary), not healthy donors.
- GSE213532: the "non-NEC" group is NEC patients after healing, so no healthy controls.
- GSE111396: lung (parenchymal/airway) fibroblasts. Input 1 mapped it to colon/heart fibroblasts.
- GSE147430: 128 of 132 donor IDs repeat GSE224807, so it is redundant.
- GSE122148: GEO summary says "isolated cultured cells", so it is cultured.

## Priority actions (most valuable first)

1. **lung stroma (fibroblasts, smooth muscle)**: Culture-check the only lung-stroma sets, then decide build vs research: mask division-drift CpGs from GSE197512 (population-doubling series incl. smooth muscle and fibroblasts) and GSE82234/GSE179847, then test GSE146376 vehicle arrays (airway smooth muscle, 70 donors) and GSE111396/GSE157651 (lung parenchymal/airway fibroblasts, 26 + 23 healthy donors). *Why:* No uncultured, sorted, healthy lung fibroblast or smooth-muscle set was found on GEO or ArrayExpress; smooth muscle is Loyfer-only and lung fibroblast is NEW; all three lung-stroma sets are cultured with no uncultured comparison (class E until culture-checked).
2. **independent neutrophils (arrays)**: Build neutrophils from GSE124565 (12 healthy, IDAT); hold out GSE88824 (8 donors with all six blood types + WBC), GSE118144 (13 controls, CD16+), GSE122244 (5) for the commissioned Met-A neutrophil reading. *Why:* Atlas neutrophils come only from the Salas group; these four are the largest independent healthy purified-neutrophil array sets with IDATs (the others are GSE72338, 10 TB household contacts; GSE247198, 3 donors; GSE83944, 1 donor). A set used for testing cannot also build.
3. **independent neutrophils (sequencing, IAM-A)**: Read IAM-A on GSE59163 (13 healthy neutrophil RRBS, SRA) and GSE128734 (same DNA on arrays and WGBS); file a BLUEPRINT/EGA access request only if WGBS neutrophils are required. *Why:* Only other-lab healthy neutrophil sequencing found; GSE87196 raw reads are EGA controlled access (GEO design text), not SRA as input 1 stated.
4. **blood: second/third lab for all common types**: Split GSE224807 never-smoker donors (about 67-71 per type) into a build half and a test half; add GSE103541 (29 donors), GSE184269 (24 donors; naive B second lab), GSE252045 (16 donors; central/effector-memory T), GSE59065 (101 donors CD4/CD8). *Why:* Large, healthy, IDAT-public, independent labs; GSE147430 and GSE318669 repeat GSE224807 donors and must not be used as independent tests.
5. **flow-counted adult whole blood (truth set)**: Ask NIEHS (Bell) for CBC differentials on the 99 GSE224807 whole-blood arrays; meanwhile use GSE167202 (525 whole blood with clinical differentials, ED patients) as a provisional truth set and GSE127824 (24 cord bloods, FACS) for cord blood. *Why:* No independent healthy adult whole-blood set with flow counts was found; all constructed leukocyte mixtures are Salas/Koestler-group. Clinical differentials (about 1-3 points error) cannot certify neutrophil RMSE <= 0.02.
6. **colon / small-intestine epithelium**: Add the Cambridge CD326+ sorted epithelium from ArrayExpress: E-MTAB-3709 (5 healthy children x 2 colon sites) and E-MTAB-4957 (16 juvenile + 7 fetal donors, with paired organoids). Confirm disease status and donor overlap first. *Why:* First independent purified colon epithelium (input 1 found none); also a second lab for Loyfer-only small intestine epithelium and a built-in culture check.
7. **WAITING types**: Fill ovary epithelium with GSE146552 (8 OSE + 8 FTE from non-malignant patients); verify raw-read access and UCSF independence for GSE16368 Roadmap (keratinocyte, dermal fibroblast, breast); treat GSE140295/GSE142224 liver endothelium as culture-conditional. *Why:* Only 3 of 16 waiting types have any build candidate; the other 13 have none.
8. **brain: second lab for astrocytes and microglia**: Build from GSE306226 (20 healthy donors x 4 types) and GSE234520 (FANS, 42 donors); hold out GSE191200, GSE112179, GSE66351 controls; check Exeter/BDR donor overlap. *Why:* Astrocytes and microglia are Tian-only (single pooled source).
9. **neurons on both instruments**: Request 450K IDATs from the Dracheva lab for GSE50853 (same nuclei on 450K and RRBS); otherwise pair GSE96612 WGBS with GSE306226 arrays (different donors). *Why:* No public set has both IDAT arrays and sequencing on the same neuronal nuclei.
10. **hepatocyte third lab**: Confirm in PMID 31798965 whether GSE123995 hepatocytes were cultured before DNA extraction. *Why:* Only independent hepatocyte source (56 donors).
11. **lung alveolar macrophages second lab**: Resolve Dartmouth independence for GSE132547 (Armstrong/Chen vs Christensen/Salas group) before building. *Why:* 12 healthy donors x 2 lobes; Loyfer-only type; institute shared with an atlas group.
12. **HSPC types (atlas plans GSE63409)**: Use GSE104405 (HSC ERRBS, SRA) and GSE49618 (BM sorted, 3 donors) for an independent HSC check; GSE273074 for cord-blood HSC; GSE87196 only via EGA. *Why:* Second source for the stem/progenitor types.
13. **sex balance (female donors)**: Use GSE131989 (31 female control donors x 4 types; UCSF check), GSE153459 (35 CD4, women) and GSE118144 controls to test female performance; record that GSE224807, GSE103541, GSE88824 do not report sex in GEO. *Why:* Several of the largest blood sets give no sex field.
14. **NEW cell types with build-grade data**: Placenta (GSE271697: cytotrophoblast, syncytiotrophoblast, Hofbauer, fibroblast; 18 subjects), thymocyte subsets (GSE269608, 5 human donors, EPIC v2), sperm (GSE114753 build, GSE126017 test), cord nRBC (GSE68456). *Why:* Healthy, purified, uncultured, IDAT-public, independent.
15. **controlled-access and unsearched sources**: Decide which of the 51 uncovered types are needed, then apply to BLUEPRINT (EGA), DEEP, IHEC/CEEHRC, dbGaP (MESA) and the NIH Aging Research Biobank (GSE324619 monocytes). *Why:* These repositories were not searched; several known purified sets sit behind access agreements.

## Map by atlas cell type (87 types in input 2)

A = build, B = held-out test, C = truth, D = research, E = not usable (reasons are in the CSV); "sec A/B" = secondary use. WAIT = waiting type (one source).

| type | atlas_sources | A_build | B_test | secondary_AB | D_research | E_not_usable |
|---|---|---|---|---|---|---|
| ad-dermal fibroblasts (WAIT) | Loyfer2023 | GSE16368 | - | - | GSE179847, GSE197512, GSE137134, GSE16368, GSE52025, GSE77135 | GSE137134, GSE163839, GSE196148, GSE52025, GSE77135, GSE89925 |
| adipocytes | Loyfer2023+Moss2018 | GSE111445 | - | - | - | - |
| aorta endothelium | Loyfer2023 | - | - | - | - | - |
| astrocytes | Tian2023 | GSE306226 | - | - | GSE306226 | - |
| b cells | Loyfer2023+Salas2018 | GSE103541, GSE224807, GSE49618, GSE68456 | GSE118255, GSE122244, GSE131989, GSE173787, GSE87095, GSE88824, GSE118144 | GSE118255, GSE87095, GSE87196, GSE88824, GSE103541, GSE224807, GSE68456 | GSE318669, GSE110607, GSE118255, GSE131989, GSE173787, GSE35069, GSE88824, GSE118144 | GSE110607, GSE166844, GSE35069, GSE59250, GSE87196, GSE137634, GSE137594, E-MTAB-13654 |
| basophils | Salas2022 | - | - | - | - | - |
| bladder epithelium | Loyfer2023 | - | - | - | - | - |
| brain endothelium (WAIT) | Tian2023 | - | - | - | - | - |
| breast basal epithelium | Loyfer2023 | GSE16368 | - | - | GSE16368 | - |
| breast luminal epithelium | Loyfer2023 | GSE16368 | - | - | GSE153309, GSE160233, GSE16368 | GSE153309, GSE160233, GSE86732 |
| cardiomyocyte | Loyfer2023 | - | - | - | - | - |
| cd4 t cells | Loyfer2023+Salas2018 | GSE103541, GSE224807, GSE59065, GSE68456 | GSE112696, GSE130029, GSE173787, GSE189255, GSE71955, GSE87640, GSE88824, GSE118144 | GSE87196, GSE88824, GSE103541, GSE153459, GSE224807, GSE68456 | GSE128734, GSE153459, GSE318669, GSE110607, GSE173787, GSE35069, GSE59065, GSE88824, GSE118144 | GSE110607, GSE166844, GSE35069, GSE56581, GSE59250, GSE87196, GSE137634, E-MTAB-13654 |
| cd8 t cells | Loyfer2023+Salas2018 | GSE103541, GSE224807, GSE59065, GSE68456 | GSE130030, GSE173787, GSE71955, GSE87640, GSE88824, GSE118144 | GSE87196, GSE88824, GSE103541, GSE224807, GSE68456 | GSE318669, GSE173787, GSE35069, GSE59065, GSE88824, GSE118144 | GSE147430, GSE166844, GSE35069, GSE87196 |
| colon enteroendocrine | Loyfer2023 | - | - | - | - | - |
| colon epithelium | Loyfer2023+Moss2018 | E-MTAB-3709, E-MTAB-4957 | - | E-MTAB-3709 | GSE213532, E-MTAB-4957, E-MTAB-11545 | GSE213532, E-MTAB-11545 |
| colon fibroblasts | Loyfer2023 | - | - | - | - | - |
| colon macrophages | Loyfer2023 | - | - | - | - | - |
| cortical neurons | Loyfer2023+Moss2018 | GSE107729, GSE234520, GSE306226, GSE96612 | GSE112179, GSE66351, GSE98203 | GSE107729 | GSE289184, GSE50853, GSE234520, GSE306226, GSE96612 | GSE41826 |
| effector memory cd8 t cells | Loyfer2023+Salas2022 | GSE252045 | - | GSE252045 | - | - |
| embryonic stem cells | ENCODE | GSE116754 | - | - | - | GSE86648, GSE95761 |
| endometrium epithelium | Loyfer2023 | - | - | - | - | GSE122148 |
| eosinophils | Salas2022 | - | - | - | GSE35069 | GSE35069 |
| erythrocyte progenitors | Loyfer2023 | - | - | GSE87196 | GSE128267 | GSE87196 |
| esophagus epithelium (WAIT) | Loyfer2023 | - | - | - | - | GSE127324 |
| fallopien tubes epithelium | Loyfer2023 | GSE146552 | GSE81224 | - | GSE146552 | - |
| gallbladder epithelium (WAIT) | Loyfer2023 | - | - | - | - | GSE262275 |
| gastric antrum epithelium | Loyfer2023 | - | - | - | - | - |
| gastric body epithelium | Loyfer2023 | - | - | - | - | - |
| gastric enteroendocrine (WAIT) | Loyfer2023 | - | - | - | - | - |
| gastric fundus epithelium | Loyfer2023 | - | - | - | - | - |
| granulocytes | Loyfer2023 | GSE103541, GSE184269, GSE224807, GSE49618, GSE68456 | GSE226298, GSE82084 | GSE103541, GSE184269, GSE224807, GSE68456 | GSE318669, GSE110607, GSE226298, GSE35069, GSE82084 | GSE110607, GSE166844, GSE185556, GSE35069 |
| heart fibroblasts | Loyfer2023 | - | - | - | - | - |
| hepatocyte | Loyfer2023+Moss2018 | GSE123995 | - | - | - | GSE127297 |
| jejunum enteroendocrine (WAIT) | Loyfer2023 | - | - | - | - | - |
| jejunum epithelium (WAIT) | Loyfer2023 | - | - | - | - | - |
| keratinocyte (WAIT) | Loyfer2023 | GSE16368 | - | - | GSE197512, GSE16368, GSE198972 | GSE198972, GSE262275 |
| kidney glomerular endothelium | Loyfer2023 | - | - | - | - | - |
| kidney glomerular epithelium | Loyfer2023 | - | - | - | - | - |
| kidney tubular endothelium | Loyfer2023 | - | - | - | - | - |
| kidney tubular epithelium | Loyfer2023 | - | - | - | - | GSE145745 |
| larynx epithelium (WAIT) | Loyfer2023 | - | - | - | - | - |
| liver endothelium (WAIT) | Loyfer2023 | - | - | - | GSE140295, GSE142224 | GSE140295, GSE142224, GSE262275 |
| liver macrophages (WAIT) | Loyfer2023 | - | - | - | - | GSE262275 |
| lung alveolar endothelium | Loyfer2023 | - | - | - | GSE140295, GSE84395 | GSE140295, GSE84395 |
| lung alveolar epithelium | Loyfer2023+Moss2018 | - | - | - | - | GSE150527, GSE185556 |
| lung alveolar macrophages | Loyfer2023 | GSE132547 | - | GSE132547 | - | GSE185556 |
| lung bronchus epithelium | Loyfer2023 | - | GSE104471, GSE201872, GSE85566 | - | GSE104471, GSE137716, GSE201872, GSE85566 | GSE137716 |
| lung interstitial macrophages | Loyfer2023 | - | - | - | - | - |
| lung pleural epithelium (WAIT) | Loyfer2023 | - | - | - | - | - |
| memory b cells | Loyfer2023+Salas2022 | GSE252045 | GSE118255 | GSE118255, GSE252045 | GSE118255 | - |
| memory cd4 t cells | Salas2022 | - | GSE121192, GSE131989 | - | GSE131989, GSE71825 | GSE59250, GSE71825 |
| microglia | Tian2023 | GSE234520, GSE306226 | GSE191200 | GSE191200 | GSE234520, GSE306226 | - |
| monocytes | Loyfer2023+Salas2018+Salas2022 | GSE103541, GSE184269, GSE224807, GSE252045, GSE49618, GSE68456 | GSE100561, GSE121192, GSE122244, GSE131989, GSE159474, GSE173787, GSE188573, GSE189256, GSE226298, GSE82084, GSE85647, GSE87640, GSE88824, GSE72338 | GSE87196, GSE88824, GSE103541, GSE184269, GSE224807, GSE252045, GSE343621, GSE68456 | GSE118696, GSE318669, GSE343621, GSE100561, GSE110607, GSE131989, GSE173787, GSE226298, GSE35069, GSE82084, GSE88824 | GSE104700, GSE110607, GSE166844, GSE324619, GSE35069, GSE43976, GSE56046, GSE59250, GSE87196 |
| naive b cells | Salas2022 | GSE184269, GSE252045 | GSE118255 | GSE118255, GSE184269, GSE252045 | GSE118255 | - |
| naive cd4 t cells | Loyfer2023+Salas2022 | GSE16368, GSE184269, GSE252045, GSE75405 | GSE121192, GSE131989 | GSE184269, GSE252045 | GSE131989, GSE16368, GSE71825 | GSE59250, GSE71825 |
| naive cd8 t cells | Loyfer2023+Salas2022 | GSE16368, GSE184269, GSE252045, GSE75405 | - | GSE184269, GSE252045 | GSE16368 | - |
| neutrophils | Salas2018+Salas2022 | GSE124565 | GSE122244, GSE88824, GSE118144, GSE72338 | GSE87196, GSE88824, GSE124565, GSE247198, GSE59163 | GSE128734, GSE247198, GSE59163, GSE35069, GSE83944, GSE88824, GSE118144 | GSE35069, GSE65097, GSE83944, GSE87196, GSE63499, GSE92749, GSE50797 |
| nk cells | Loyfer2023+Salas2018+Salas2022 | GSE184269, GSE224807, GSE252045, GSE68456 | GSE104287, GSE88824 | GSE87196, GSE88824, GSE184269, GSE224807, GSE252045, GSE343621, GSE68456 | GSE318669, GSE343621, GSE104287, GSE35069, GSE88824 | GSE35069, GSE87196 |
| oligodendrocyte precursors | Tian2023 | - | - | - | - | - |
| oligodendrocytes | Loyfer2023+Tian2023 | GSE107729, GSE234520, GSE306226, GSE96612 | GSE66351 | GSE107729 | GSE50853, GSE234520, GSE306226, GSE96612 | GSE41826 |
| osteoblasts (WAIT) | Loyfer2023 | - | - | - | GSE202067 | GSE202067 |
| ovary epithelium (WAIT) | Loyfer2023 | GSE146552 | GSE81224 | - | GSE146552 | - |
| pancreas endothelium | Loyfer2023 | - | - | - | - | - |
| pancreatic acinar | Loyfer2023+Moss2018 | GSE223153 | - | - | GSE223153 | - |
| pancreatic alpha | Loyfer2023 | - | - | - | GSE143209 | GSE143209 |
| pancreatic beta | Loyfer2023+Moss2018 | - | - | - | GSE143209 | GSE143209 |
| pancreatic delta | Loyfer2023 | - | - | - | - | - |
| pancreatic duct | Loyfer2023+Moss2018 | GSE223153 | - | - | GSE223153 | - |
| pericytes (WAIT) | Tian2023 | - | - | - | - | - |
| pharynx epithelium (WAIT) | Loyfer2023 | - | - | - | - | - |
| podocyte | Loyfer2023 | - | - | - | - | - |
| prostate epithelium | Loyfer2023 | - | - | - | - | - |
| regulatory t cells | Salas2022 | - | - | - | - | GSE59250 |
| small intestine epithelium | Loyfer2023 | E-MTAB-4957 | - | - | GSE213532, E-MTAB-4957, E-MTAB-11545 | GSE213532, E-MTAB-11545 |
| smooth muscle | Loyfer2023 | - | - | - | GSE197512, GSE146376 | GSE113061, GSE146376, E-MTAB-15426 |
| striated muscle | Loyfer2023 | - | - | - | GSE166652, GSE221540 | GSE166652, GSE166787, GSE221540 |
| t cells cd3 | Loyfer2023 | GSE156792, GSE49618, GSE68456 | GSE122244, GSE67170, GSE72354, GSE82084 | GSE156792, GSE68456 | GSE72354, GSE82084 | GSE103911, GSE117050 |
| t central memory cd4 | Loyfer2023 | GSE252045 | - | GSE252045 | - | - |
| t effector cell cd8 | Loyfer2023 | GSE75405 | - | - | - | - |
| t effector memory cd4 | Loyfer2023 | - | - | - | - | - |
| thyroid epithelium | Loyfer2023 | - | - | - | - | - |
| tongue epithelium | Loyfer2023 | - | - | - | - | - |
| tonsil palatine epithelium | Loyfer2023 | - | - | - | - | - |
| tonsil pharyngeal epithelium | Loyfer2023 | - | - | - | - | - |
| vascular endothelium | Moss2018 | - | - | - | GSE82234, GSE197512, GSE106099, GSE140295, E-MTAB-8952 | GSE106099, GSE140295, E-MTAB-8952 |
| vascular leptomeningeal cells | Tian2023 | - | - | - | - | - |
| vascular saphenous endothelium | Loyfer2023 | - | - | - | - | - |


## NEW cell types (not in the atlas)

| cell_type | A_build | B_test | secondary_AB | D_research | E_not_usable |
|---|---|---|---|---|---|
| NEW: B-cell subsets (transitional, DN2, activated naive) | - | GSE118255 | GSE118255 | GSE118255 | - |
| NEW: CD56bright NK | GSE252045 | - | GSE252045 | - | - |
| NEW: CD8 TEMRA | GSE252045 | - | GSE252045 | - | - |
| NEW: CD8 central memory T | GSE252045 | - | GSE252045 | - | - |
| NEW: Hofbauer cell | GSE271697 | - | - | GSE271697 | - |
| NEW: SATB2+ excitatory neurons | GSE234520 | - | - | GSE234520 | - |
| NEW: airway smooth muscle | - | - | - | GSE146376 | GSE146376 |
| NEW: antibody-secreting cell | - | GSE118255 | GSE118255 | GSE118255 | - |
| NEW: breast progenitor | GSE16368 | - | - | GSE16368 | - |
| NEW: central memory Th1 | GSE75405 | - | - | - | - |
| NEW: common lymphoid progenitor | - | - | GSE87196 | - | GSE87196 |
| NEW: common myeloid progenitor | - | - | GSE87196 | - | GSE87196, GSE50797 |
| NEW: cytotoxic CD4 T (CD28-) | GSE75405 | - | - | - | - |
| NEW: dura mater fibroblast | - | - | - | GSE77135 | GSE77135 |
| NEW: fetal intestinal epithelium | E-MTAB-4957 | - | - | E-MTAB-4957 | - |
| NEW: granulocyte-macrophage progenitor | - | - | GSE87196 | - | GSE87196, GSE50797 |
| NEW: hematopoietic stem cell | GSE104405, GSE273074, GSE49618 | - | GSE87196 | GSE104405, GSE273074 | GSE87196 |
| NEW: low-density granulocytes | - | - | - | - | GSE65097 |
| NEW: lung fibroblast | - | - | - | GSE107226, GSE111396, GSE157651 | GSE107226, GSE111396, GSE135097, GSE157651 |
| NEW: lymphatic endothelium | - | - | - | E-MTAB-8952 | E-MTAB-8952 |
| NEW: megakaryocyte | - | - | GSE87196 | - | GSE87196 |
| NEW: megakaryocyte-erythroid progenitor | - | - | GSE87196 | - | GSE87196 |
| NEW: melanocyte | GSE16368 | - | - | GSE103413, GSE16368, GSE166069 | GSE103413, GSE166069 |
| NEW: monocyte-derived macrophage (in vitro) | - | - | - | GSE118696 | - |
| NEW: multi-lymphoid progenitor | - | - | GSE87196 | - | GSE87196 |
| NEW: multipotent progenitor | - | - | GSE87196 | - | GSE87196 |
| NEW: myoblast (cultured) | - | - | - | GSE166652, GSE221540 | GSE166652, GSE221540 |
| NEW: nasal epithelium | - | GSE104471 | - | GSE101641, GSE104471, GSE109446, GSE65163 | GSE101641, GSE109446, GSE65163 |
| NEW: nucleated red blood cell (cord) | GSE68456 | GSE82084 | GSE68456 | GSE82084 | - |
| NEW: placental cytotrophoblast | GSE271697 | - | - | GSE271697 | - |
| NEW: placental fibroblast | GSE271697 | - | - | GSE271697 | - |
| NEW: promyelocytes | GSE49618 | - | - | - | - |
| NEW: sperm | GSE114753 | GSE102970, GSE126017 | - | GSE108058, GSE246935, GSE64096, GSE102970 | GSE115920, GSE139307 |
| NEW: syncytiotrophoblast | GSE271697 | - | - | GSE271697 | - |
| NEW: thymocyte subsets (DN1-3, ISP, DP, SP4, SP8) | GSE269608 | - | - | - | - |
| NEW: tonsil NK developmental intermediates | - | - | - | GSE318453 | - |


## Gaps that remain, and where else to look

- 51 of the 87 atlas cell types in input 2 have no independent BUILD or TEST candidate after this mapping: aorta endothelium, basophils, bladder epithelium, brain endothelium, cardiomyocyte, colon enteroendocrine, colon fibroblasts, colon macrophages, endometrium epithelium, eosinophils, esophagus epithelium, gallbladder epithelium, gastric antrum epithelium, gastric body epithelium, gastric enteroendocrine, gastric fundus epithelium, heart fibroblasts, jejunum enteroendocrine, jejunum epithelium, kidney glomerular endothelium, kidney glomerular epithelium, kidney tubular endothelium, kidney tubular epithelium, larynx epithelium, liver endothelium, liver macrophages, lung alveolar endothelium, lung alveolar epithelium, lung interstitial macrophages, lung pleural epithelium, oligodendrocyte precursors, osteoblasts, pancreas endothelium, pancreatic alpha, pancreatic beta, pancreatic delta, pericytes, pharynx epithelium, podocyte, prostate epithelium, regulatory t cells, smooth muscle, striated muscle, t effector memory cd4, thyroid epithelium, tongue epithelium, tonsil palatine epithelium, tonsil pharyngeal epithelium, vascular endothelium, vascular leptomeningeal cells, vascular saphenous endothelium.
- 13 of 16 WAITING types have no build candidate: brain endothelium, esophagus epithelium, gallbladder epithelium, gastric enteroendocrine, jejunum enteroendocrine, jejunum epithelium, larynx epithelium, liver endothelium, liver macrophages, lung pleural epithelium, osteoblasts, pericytes, pharynx epithelium. Filled only conditionally: ad-dermal fibroblasts, keratinocyte, ovary epithelium (ad-dermal fibroblasts and keratinocyte only through cultured Roadmap foreskin cells in GSE16368, access unverified). Leads that are cultured only: liver endothelium (GSE140295/GSE142224), osteoblasts (GSE202067), keratinocyte (GSE198972).
- Lung stroma: no uncultured sorted lung fibroblast or smooth muscle set found on GEO or ArrayExpress (12 ArrayExpress queries).
- No independent healthy adult whole blood with flow counts; neutrophil-fraction RMSE <= 0.02 cannot yet be certified on an independent truth set.
- No purified healthy EPIC v2 (GPL33022) set for any atlas type; the only EPIC v2 purified sets are thymocytes (GSE269608) and NK developmental intermediates (GSE318453).
- Basophils, eosinophils and regulatory T cells: only Salas-group or IDAT-less sets (GSE35069, GSE59250).
- NOT SEARCHED: EGA (BLUEPRINT, other controlled sets), DEEP, IHEC data portal, CEEHRC, dbGaP, NIH Aging Research Biobank, Human Cell Atlas, ENCODE4 beyond the atlas, ENA-only studies, CNCB/GSA (China), JGA (Japan).
- PARTLY SEARCHED: ArrayExpress (12 keyword queries, metadata only, 6 studies read); GEO targeted priority queries (5) on top of input 1's ~70 term sets.

## Independence flags

- NOT independent (atlas groups): GSE77797, GSE180970, GSE112618, GSE231417 (Salas/Koestler/Christensen/Kelsey/Wiencke); GSE139307 (Kelsey Lab, Brown); GSE239605 (Glaser); GSE262275 (Dor/Kaplan collaboration, per input 1); ENCODE GSE127297, GSE86732, GSE127324.
- CHECK (shared institute, no contributor-name match): GSE132547 (Dartmouth), GSE131989 and GSE16368 (UCSF, the Wiencke institute). GSE49618: Triche is also a contributor on Glaser-group GSE239605, which is not an atlas build series.
- Within-candidate overlaps: GSE103541/GSE166844 (same donors); GSE224807/GSE147430/GSE318669 (NIEHS Bell, shared donor IDs); GSE111396/GSE157651 (Nottingham); GSE68456/GSE82084/GSE127824 (UBC); E-MTAB-3709/E-MTAB-4957 (Cambridge); GSE306226 and the Exeter brain sets (BDR/Exeter); GSE247198/GSE83944 (possible shared donor).

## Deviations carried forward from input 1

- The search used GEO keyword queries (esearch over ~70 cell-type term sets) and was not an exhaustive census. ArrayExpress, ENA-only and EGA/dbGaP sets were not queried systematically, so BLUEPRINT/DEEP/IHEC controlled-access data are not listed.
- Many Loyfer-only cell types (gastric, tongue, tonsil, thyroid, kidney glomerular, podocyte, cardiomyocyte, sorted pancreatic alpha/beta/delta, saphenous/aortic endothelium) got only one keyword pass each. 'No candidate' for these is provisional.
- Purification method, healthy status and donor counts come from GEO text and sample characteristics. Papers were read only for the compendium (bioRxiv version; the Cell Genomics/PMC13030963 version was not opened), the LaBarge breast work and the GSE41826 mixture design. Donor counts derived from title prefixes (GSE224807 141, GSE318669 13) and the 'likely 1 donor' call for GSE83944 are provisional.
- IDAT availability is taken from the GEO esummary supplementary-file type field. RAW.tar contents were not listed (GSE166844 is marked unverified). For Roadmap GSE16368, raw-read access was not verified.
- The 2025 compendium's purified-cell series came from its metadata.parquet labels (CL terms plus a few epithelial UBERON terms). The 29 compendium series added from that list (rank 3) got metadata-level review only.
- 'Newer than 2023-06' uses the GEO public date (pdat), not the submission date.