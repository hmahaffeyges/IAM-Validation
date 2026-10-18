# Commissioned chain v4: showcase runs and the cfDNA program (draft, 2026-10-17)

**The chain's job:** one sample goes in; every commissioned instrument reads it inside its commissioned arena, reads outside it with a plain note, or declines with its reason.

Every set below is public, carries raw files (IDATs or reads), and is not in our used-data record (CANON/data_spent.csv). The samples are chosen by a written rule before any file is opened, run once through the one-command chain, and shown whatever they read.

## Part 1: the showcase (7 scenarios)

| # | Scenario | Public set | Instrument in its arena | Every other instrument |
|---|---|---|---|---|
| 1 | One person, three specimens, two array versions | GSE350403: 16 healthy adults; buffy coat, PBMC, saliva; each on EPIC v1 and v2 (96 arrays) | **C-Score** on buffy coat (whole blood), EPIC v1; Deconvolver shares | PBMC and saliva read with the note; EPIC v2 arrays: whatever the chain's intake rule does, shown as is |
| 2 | Isolated neutrophils from healthy people | GSE122244: healthy controls' sorted neutrophils, EPIC (with their B, T and monocyte fractions) | **Met-A**; Deconvolver should read about 100 % neutrophils | C-Score reads with the note (not whole blood) |
| 3 | Known mixtures of 9 cell types | GSE281305: in-vitro mixtures, EPIC, with the true proportions published | **IAM-Deconvolver** recovering the planted shares | Met-A and C-Score read with the note |
| 4 | Healthy colon tissue | GSE199057: 68 non-cancer colon mucosa, EPIC | **IAM-Atlas** (commissioned on colon tissue) | the rest read with the note |
| 5 | Healthy CD8 T cells, whole-genome sequencing | GSE245832 naive CD8 T cells (3 donors; raw reads in SRA to confirm), or another unopened healthy CD8 set | **IAM-A** on its own reference | array instruments sit out (no array) |
| 6 | Declines | arrays that fail controls, and a sample the shape check does not know (e.g. a cell line) | the chain refusing, with its reason | |
| 7 | All together | scenario 1 (same person, three specimens) | every array instrument on one person | IAM-A needs sequencing of the same person; none public found yet |

Still to find: a healthy **lung** tissue set with IDATs (GSE234014 has processed values only).

## Part 2: cfDNA (after the showcase)

1. **Synthetic first:** cfDNA pieces are short (about 167 letters), so fewer of them carry 6 or more CpG calls. Before any real plasma, a planted test answers whether IAM-A's held-error count holds on fragments this short. The atlas and Deconvolver (which tissue the cfDNA came from) need no change.
2. **Lung:** GSE277078, plasma cfDNA of 185 people with non-small-cell lung cancer, EPIC, IDATs, unused.
3. **Tissue injury, known source:** GSE287137, pediatric ARDS plasma cfDNA (lung injury), EPIC, IDATs, unused. This checks whether the Deconvolver sees lung DNA rising in the blood.
4. **Colorectal and pancreas (GI cancers):** GSE149438 (cfDNA, several GI cancers including colorectal and pancreatic, sequencing; processed files only, raw reads to confirm) and GSE186381 (colorectal serum cfDNA pools, EPIC, processed values only).

## Priority by lives (US yearly deaths, rounded; exact figures confirmed before any public use)
1. Lung, about 125,000: atlas commissioned on lung tissue; lung cfDNA set in hand
2. Colorectal, about 53,000, rising fastest under 50: atlas commissioned on colon tissue; healthy and tumor colon in hand
3. Pancreas, about 51,000: the author's priority; the atlas's pancreas cells are thin, so atlas work first

## Part 3: the tracks after the showcase (author, 2026-10-17)

The goal behind every track: whether these tests could one day stand in for routine screening (colonoscopy, Pap smear, mammogram) by reading an organ's normal cells, and the immune system, years before a tumor forms. That is for doctors, clinical studies and regulators to decide; these tracks gather the evidence it would take.

1. **Field (FLD-002):** normal tissue from patients against healthy people on the same slides, ages known. The breast gradient (DEV-FIELD-01) read both breasts of each patient about 18 % above healthy donors, confounded by slide, collection and age; this settles it.
2. **Precancer:** polyps and adenomas (colon), CIN lesions (cervix), DCIS (breast) against healthy tissue.
3. **Before diagnosis, in blood:** fresh pre-diagnosis sets to replicate the EPIC-Italy breast read (GRP-008).
4. **Immune track, organ by organ:** whole blood, patients against healthy people in the same lab series (Group-A); each instrument reads its part (Deconvolver shares, Met-A neutrophils, C-Score clustering, Atlas non-blood DNA). Question: does the pattern point to the organ under strain? With plasma cfDNA (which organ's cells are shedding), two readings from one blood draw. Sets in hand: breast GSE243529 and GSE283951, lung GSE275371 (blood immune cells, EPIC); colon and pancreas to scout. The class-era immune signals are leads to replicate with the commissioned instruments.
5. **One person over time:** IAM-A's self-reference read on the same person's cells across years.
6. **Collection routes:** plasma cfDNA, stool (colon lining), sputum (airways), urine, cervical swab; each needs a short-fragment synthetic first where the DNA is broken.

## Part 4: salmon track (author, 2026-10-17; gated on the temperature result)
IAM-A needs no atlas or other specimens, so it can read fish. Synthetic first; the Mahaffey number scales with water temperature by the law itself (M(T) = M(37 C) x 310.15 K / T); then healthy fish read Normal on their own reference; then one fisheries question on public data. Manifest row in the TIS section.

## Part 5: showcase sources checked against the record (2026-10-17, evening)

Each draft source was checked against CANON/data_spent.csv, MASTER_INVENTORY.csv and every repo mention. Four of the first picks cannot be used as drafted:

| Scenario | Draft pick | Finding | Use instead |
|---|---|---|---|
| 2 Met-A, neutrophils | GSE122244 | **spent**: an atlas build source and read in three commissionings | GSE83944 (circadian neutrophils, 450K, IDATs, unspent) if Met-A's platform scope covers 450K; otherwise keep searching EPIC |
| 4 Atlas, colon tissue | GSE199057 | normals fetched and calibrated in DEV-TISSUE-COLON-02 (recorded not suitable); not clean | a fresh colon set; GSE235414 adjacent-normal lung (EPIC, IDATs) is held for the field test (DEV-FINGERPRINT-03), so not used here |
| 5 IAM-A, CD8 | GSE245832 | every sample **stimulated** 4 hours with peptide; IAM-A is commissioned on healthy resting CD8 T cells | GSE173787 healthy-control CD8 T cells (bisulfite sequencing) is reserved as the atlas's held-out sequencing set; a fresh healthy CD8 WGBS set still to find |
| 1, 3 | GSE350403, GSE281305 | unspent, IDATs present, labels and (GSE281305) pure references for the mixtures | keep |

Immune track: GSE275371 is PBMCs from 35 people with lung adenocarcinoma and 50 without (EPIC, IDATs), confirmed.
