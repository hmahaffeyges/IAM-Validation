# DEV-COMPOSITION-TRUTH-03 — whole-blood truth from the laboratory's own sorted cells (written 2026-10-09, before download or reading)

**Why.** Bars 1–2 of DEV-ATLAS-COMMISSION-01 need whole bloods of known composition from a laboratory other than Salas (Dartmouth),
whose arrays built atlas_e's blood templates. No such counted bloods were found on GEO or ArrayExpress. Another way: GSE224807 holds,
for the same people, whole blood and six sorted cell types (CD14, CD15, CD19, CD4, CD56, CD8): 30 people on 450K, 4 on EPIC (+custom).

**Truth (independent of the atlas).** For each platform, the laboratory's own template = the mean of its sorted arrays of each cell
type over all its people (self-tared betas, the chain's Stage 1 calibration). Each whole blood is fitted to that template (non-negative
least squares, fractions summed to 1, the most cell-informative 6,000 sites of the template). The laboratory's offsets are shared by
truth and blood. Simulated before download (atlas v2 draws as people, person spread 0.02, sort purity 93–99 %, array noise 0.02–0.06):
truth error for the neutrophil fraction 0.003–0.005 mean, 0.013 at most (`development/sims/atlas_sims_01.py`) — four times tighter than bar 1. (Using each person's own six
arrays instead gives 0.018 at array noise 0.04, biased low by noise in the templates; not used.)

**Reading.** atlas_e unchanged (chain/dev_stages.py, IAMAtlas_v2.parquet) on each whole blood, 450K sites present; and the current
blood_composition_EPIC_v1 for bar 2. Neutrophil fraction = CD15 (granulocytes; this laboratory sorts CD15, so eosinophils sit inside the
truth's 'neutrophil'; atlas_e's neutrophils + eosinophils + basophils are compared with it).

**Bars (DEV-ATLAS-COMMISSION-01, as written):** neutrophil MAE ≤ 0.02 and every blood within 0.05; not worse than the current method.
**How it counts:** 450K (30 people) is a different platform from the EPIC composition step, so it is evidence for atlas_e's templates on
an unseen laboratory, recorded with its platform; the 4 EPIC people are reported individually. Neither alone commissions the EPIC step
(bar 1 needs ≥ 2 sets, ≥ 20 samples on EPIC). If 450K fails bar 1, the reason is recorded before anything else is read.
