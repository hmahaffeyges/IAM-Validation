# DEV-ATLAS-COMMISSION-01 — commissioning IAM-Atlas v2, use by use (composition bars written 2026-10-09; plan for every use 2026-10-12)

**Why it matters.** Whole-blood Met-A needs the neutrophil fraction first: composition decides how much of the blood signal is neutrophil.
Today that is `blood_composition_EPIC_v1.json` (8 purified groups). If atlas v2 measures composition better, it replaces it.

**What is already measured (development, both truth sets seen before these bars):**

| truth set | neutrophils, atlas_e | neutrophils, current | other groups |
|---|---|---|---|
| GSE112618, 6 bloods with FACS counts | mean error **0.016**, max 0.028 | 0.031, max 0.046 | atlas_e better on CD4 T, monocytes; worse on CD8 T, NK (DEV-ATLAS-EPIC-01) |
| GSE182379, 12 constructed mixtures | RMSE **0.014** | 0.019 | atlas_e 6 of 8 groups ≤ bar, current 5 of 8 (DEV-ATLAS-EPIC-01) |

**Independence, checked against the atlas v2 roster.** Atlas v2's blood cells come from Salas 2018 (GSE110554), Salas 2022 (GSE167998) and
Loyfer 2023. GSE182379's mixtures were built by the Salas 2022 laboratory from purified cells and may share donors with GSE167998, so they
are not an independent test of either method (the current method is built from GSE110554 too). GSE112618 (separate bloods, FACS counted)
is independent of the reference cells but small (6).

**Bars (on truth sets not used to build atlas v2; fixed now).**
1. Neutrophil fraction: mean absolute error ≤ 0.02 and every sample within 0.05, on each independent truth set (≥ 2 sets, ≥ 20 samples).
2. Not worse than the current method on neutrophils on any independent set.
3. For use beyond neutrophils (later, per-cell readings): every one of the 8 groups RMSE ≤ 0.03 (DEV-NILC-01 bars).
Bars 1–2 commission atlas_e as the composition step for whole-blood Met-A; then whole-blood Met-A bars are rerun with it and Met-A is
re-commissioned on it. Bar 3 is a separate, later commissioning.

**Independent truth sets to read (to obtain):** GSE122126 in-vitro genomic DNA mixes (EPIC; the Dor laboratory, which also made the Loyfer atlas that atlas v2 is built on: independent of the Salas reference, not of the atlas (rule 9, 2026-10-11); compositions in the
paper's supplementary table, to be extracted); EPIC whole bloods with flow counts from other laboratories (GEO search to do).

## Simulation — lavage truth sets against the composition bars (2026-10-09, before any download)
Atlas v2 posterior draws, 60 random blocks (66,954 loci; the 8,000 most cell-informative used), an 11-cell lavage panel (alveolar and
interstitial macrophages, monocytes, CD4, CD8, B, NK, neutrophils, eosinophils, alveolar and bronchus epithelium). Truth: macrophages
~70–95 %, lymphocytes 4–20 %, neutrophils 0.5–12 %, eosinophils 0–2 %; each 'person' one posterior draw plus spread; counted fraction
from a 400-cell differential. Solver: non-negative least squares on posterior means (a stand-in; the real reading uses atlas_e).
| conditions | neutrophil MAE vs truth | MAE vs the count | within 0.05 of the count |
|---|---|---|---|
| array SD 0.02 | 0.004 | 0.010 | 300/300 |
| array 0.04, person 0.02 | 0.005 | 0.011 | 299/300 |
| array 0.06, person 0.04 | 0.006 | 0.011 | 299/300 |
| + random per-site lab offset SD 0.06 | 0.010 | 0.013 | 299/300 |
| + methylated-channel shift 10 % | 0.005 | 0.011 | 299/300 |
Reproduce: `development/sims/atlas_sims_01.py` (60 atlas blocks, checksums in `atlas_blocks_60.json`).
The count's own error (binomial, 400 cells) is most of the 0.010; bar 1 (MAE ≤ 0.02, every sample within 0.05) is reachable. Simulation is
optimistic: EPIC probe overlap with the atlas loci, atlas_e's own solver, and cell states in disease (MS, beryllium disease) not modelled.
**How it counts (most defensible):** lavage is lung, not blood. A pass is independent-laboratory evidence that atlas v2 measures
neutrophil fraction against a macrophage background; it does not by itself commission the whole-blood composition step, which still
needs a whole-blood truth set from a laboratory independent of both the Salas reference and the atlas (the Moss mixes,
GSE122126, are the atlas laboratory's own; the best public candidate is GSE167202, DEV-COMPOSITION-TRUTH-04). **Decision:** download GSE133062 first (healthy; Karolinska), then
GSE206709 (National Jewish), read with atlas_e unchanged, bars 1–2 as written. Differential cell count method and cells counted to be
taken from each paper before scoring.

## Correction to the truth-set list (2026-10-09, late; checked against the code and the papers)
1. **atlas_e holds the 12 circulating blood cells only** (chain/dev_stages.py ATLAS_E_CELLS: no macrophages, no epithelium). Lavage samples
   (~85 % macrophages) cannot be read by atlas_e unchanged; the lavage simulation above used an 11-cell lung panel, a different method.
   Lavage sets therefore test atlas v2's templates in the lung, not atlas_e, and do not count toward bars 1–2.
2. **GSE122126 (Moss 2018) is not a neutrophil truth set.** Its 9 genomic-DNA mixes put liver, lung, neuron or colon DNA at 0–10 % into one
   healthy donor's leukocyte DNA (paper, Fig. 3; Supplementary Data 1): the known quantity is the tissue fraction; the leukocytes' own
   neutrophil share is not given, and atlas_e has no tissue cells. Removed from the list of independent sets for bars 1–2.
**Still needed for bars 1–2:** EPIC whole bloods with flow or differential counts, or physical mixes of purified blood cells, from a
laboratory other than Salas (Dartmouth). Sample-field search of GEO (986 EPIC/450K samples with count fields) found none outside that
group. Next: the same search on ArrayExpress and on EPIC v2 (GPL33022) series, and published papers' supplements.

**Prior reading:** GSE110554 and GSE167998 are the composition reference; GSE112618, GSE182379 and GSE122126 were read by development tests of this chain (BOX RUN 1 job E, DEV-ATLAS-EPIC-01, DEV-CSCORE-MOSS-01) and are from the reference or the atlas laboratory.

## Commissioning plan for every use of the atlas (2026-10-12; DRAFT bars, no test data read)

The chain uses atlas v2 in four ways. Each is commissioned on its own, on purified healthy cells from laboratories that did not build the
atlas (rule 9: no shared GEO contributors with Loyfer, Salas, Moss, Tian or ENCODE). A use that is not commissioned is labeled development
wherever it appears. Bars below are drafts: each is fixed only after a synthetic test on the atlas's own posterior shows it can be met and can fail.

| use | what it does in the chain | already measured (atlas's own data) | commissioning test on independent cells (draft bar) |
|---|---|---|---|
| U1 reference profiles | each cell type's methylation per CpG with an uncertainty | held-out coverage PASSED (PROC-V5-HELDOUT; within the six sources) | purified cells from new laboratories fall inside the 90 % interval at 80-97 % of sites, per cell type and platform; reported by cell |
| U2 blind sites for tissue tests | picks sites where every panel cell agrees, so a change in tissue mix cannot move Met-A (breast, colon, liver, lung) | used by four sealed tissue tests; checked only on atlas means | on independent purified cells of every panel type, the blind sites stay within the test's own tolerance of the epithelial value at >= 90 % of sites |
| U3 composition (atlas_e) | blood cell fractions for whole-blood Met-A | neutrophils better than the current method (DEV-ATLAS-EPIC-01, in-sample); A5 FAILED on CD4 and CD8 (PROC-DECONV-V2-01) | bars 1-2 above (neutrophils), unchanged; CD4/CD8 stay development until A5 is met |
| U4 identity loci (per-cell Met-A) | the sites each cell type is read on | B1 and B2 FAILED: 71 % of held-out readings NORMAL (PROC-V12-IDENTITY-02) | stays development; not used by any commissioned reading |

**Order.** U2 first: it is what the cancer fingerprint results rest on, and passing it makes their blind-site property independent evidence,
not only an atlas property. Then U1, then U3 when an independent whole-blood truth set exists.

**Same cell, different laboratory.** Wherever two or more independent laboratories give the same cell type, their profiles must agree within
the atlas's stated uncertainty (reported as the share of sites whose difference is inside the combined 90 % interval).

**Test data.** The first search for independent purified cells (2026-10-12) is running; its candidates go into a table here, by use.

**Updating the atlas (author rule 2026-10-12).** A new search for purified cells at least every 6 months. Before any new set is used, its samples
are split in writing: held-out test data and build data (a set cannot both build and test the atlas). Each addition is a new atlas version
with its own DOI; a result scored under one version's seal stays on that version, and a new version is used by the chain only after it re-passes U1-U2.

**U2 in practice, 2026-10-12 (DEV-TISSUE-LUNG-03).** On the squamous-lung normals of that test the blind sites were not blind: stroma and immune blends moved Met-A past
a tight bar, and the atlas has no lung stromal cells to choose sites against. First priority for new atlas data: purified healthy lung
fibroblasts and lung smooth muscle.

**Lung stroma candidates, 2026-10-12 (metadata only; `data/DEV_ATLAS_COMMISSION_01/lung_stroma_candidates.py`, output
`lung_stroma_candidates_output.txt`).** GSE146376 (University of Chicago): primary cultured airway smooth muscle on EPIC v1, raw IDATs; the
untreated arrays of donors without asthma are one per donor. GSE107226: cultured lung fibroblasts on 450K, raw IDATs, including normal lungs.
Neither laboratory built the atlas. Both are cultured cells, and culture moves methylation (the MCF 10A lesson); before either enters the
atlas or protects a tissue test, its profile is compared with the same cell type in uncultured lung, and each set is split into build and
held-out data before any array is read.

**First search for independent purified cells, 2026-10-12** (metadata only; `data/DEV_ATLAS_COMMISSION_01/candidates_search_2026-10-12.json`,
summary `candidates_summary.py` -> `candidates_summary_output.txt`, which also lists the search's declared limits: GEO keywords only, a single pass
for some cell types, donor counts partly provisional). Leads, each to be verified from its files before any use:
- GSE224807 (NIEHS): six purified blood cell types and whole blood from the same donors, including granulocytes (CD15), on the custom EPIC
  chip; public 2023, after the atlas was built. Candidate held-out set for U1 and for the neutrophil reading.
- GSE127824: cord blood with matched flow counts, independent of the reference group. Candidate for whole-blood bar 7 (DEV-COMPOSITION-TRUTH-04),
  with the cord-blood caveat (nucleated red cells, newborn composition) to be assessed first.
- GSE123995: primary hepatocytes on EPIC, the only independent hepatocyte set found.
No independent set was found for adult colon epithelium, lung alveolar epithelium or breast epithelium on arrays.

**The neutrophil reading on the atlas (U4, first case).** The commissioned Met-A reference is six purified EPIC arrays (one laboratory, five
male and one female). The atlas pools neutrophils from more sources, but holds each cell's mean with array noise removed; Met-A compares a
specimen's single-array entropy with single-array entropy, so a reference taken from atlas means would read every specimen slightly high. The
route: derive the neutrophil sites and healthy reference from the atlas's per-sample spread and draws, then pass the same bars the EPIC reference
passed on held-out arrays (GSE224807 granulocytes a candidate) before re-commissioning. Until then the commissioned reference is unchanged.

**U5, the atlas on the sky map (author, 2026-10-12).** HEALPix pixelization of the genome is built and is not in the chain (book tool board,
row 15). Rule: nothing computed on the sky map reaches a reading, a report or a public claim until its null test passes on held-out healthy
cells; it is then commissioned like U1-U4, with its bars written before its data is read. Until then every document says "built; statistics
withheld until the null test passes". Checked 2026-10-12: the Met-A C-score reads the residuals in genome order (conductor_v3.stage_mc_cscore,
blocks of consecutive sites) and does not read the sky map; status fact healpix_not_in_chain refuses a claim that the chain uses HEALPix.

**In plain words.** Our map of healthy cell types gets used for several different jobs: choosing which sites to read, working
out what mixture of cells a blood sample holds, and naming the cell a sample came from. A map can be good enough for one job
and not another, so we commission the uses one at a time instead of blessing the map as a whole. **What it would take.** For
each use, purified healthy cells from laboratories that did not build the map, with the bar written before we look. The first
use being commissioned is the choice of sites a cancer test reads.
