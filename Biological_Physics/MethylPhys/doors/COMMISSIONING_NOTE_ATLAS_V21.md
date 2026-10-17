# IAM-Atlas v2.1: commissioned for three cell references and two tissue tests' blind sites, 2026-10-16

**Status: COMMISSIONED 2026-10-16** for the uses below, each on its own bar sealed before the atlas existed (`data/DEV_ATLAS_V21_01/sealed_commission_v21.json`,
amended `amendment_v21b.json`); every other use stays development. The sealed rule: "A use that is not commissioned is labeled development
everywhere it appears."

**What it is.** A map of 79 healthy human cell types at 813,951 array sites, fitted as one Bayesian model from seven public sources on arrays
and sequencing, so every site of every cell carries its value and its range of error (record `DEV_ATLAS_V21_01.md`). The fit converged
(worst R-hat and effective sample size within the sealed limits) and every pair of cells passed the distinctness gate.

**Commissioned uses** (held-out healthy cells from laboratories that share no GEO contributor with any build source; scored once):

| use | what passed | bar |
|---|---|---|
| U1 reference profile, cortical neurons, 450K | 97.8 % of 45 arrays, 2 laboratories, inside the atlas's 90 % range at 0.80-0.97 of sites | >= 95 % of arrays |
| U1 reference profile, lung bronchus epithelium, 450K | 100 % of 53 arrays, 2 laboratories | >= 95 % |
| U1 reference profile, naive CD4 T cells, 450K | 100 % of 35 arrays, 2 laboratories | >= 95 % |
| U2a blind sites of the sealed colon tissue test | colon epithelium 0.999; neutrophils 0.971, T 0.963, B 0.981 of sites within tolerance | >= 0.90 for every assessed panel cell |
| U2a blind sites of the sealed lung tissue test | bronchus epithelium 0.980; neutrophils 0.960, monocytes 0.993, T 0.974, B 0.985 | >= 0.90 |

**Development (bars not met, or not assessed):** the other 23 assessed cell-platform reference profiles (U1: centered ranges, median coverage
close to nominal, but a low tail from three laboratories; diagnosis: ranges too narrow for between-laboratory spread at some sites, not a shift);
lung alveolar macrophages (not assessed: their only held-out set failed carried check D13); blood composition (U3 FAILED on cord blood and on
emergency-department bloods, where the current method also misses); the breast and liver blind sites (no held-out epithelium).

**Head-to-head against the published array reference atlas of human cell types** (DEV-ATLAS-USE-01, sealed 2026-10-15): on 731 held-out arrays, v2.1 met all three bars:
markers confirmed 0.424 against 0.398; blood mixtures median error 0.0224 against 0.0305; with brain cells the published array reference does not hold,
0.0228 against 0.0523, the outside share recovered in 87 % of mixtures.

**Reference sets:** the atlas was built from 21 sources (E-MTAB-4957, ENCODE, GSE103541, GSE116754, GSE118144, GSE122244, GSE124565, GSE137716, GSE224807, GSE226298, GSE252045, GSE306226, GSE49618, GSE63409, GSE75405, GSE88824, Loyfer2023, Moss2018, Salas2018, Salas2022, Tian2023; `data/DEV_ATLAS_V21_01/final_samples_v21.csv`). Held out and read once: U1 29 series (E-MTAB-2145, E-MTAB-4957, GSE100561, GSE104287, GSE104471, GSE112179, GSE112696, GSE121192, GSE130029, GSE130030, GSE131989, GSE132547, GSE153459, GSE184269, GSE188573, GSE189255, GSE189256, GSE191200, GSE201872, GSE224807, GSE247198, GSE306226, GSE59065, GSE66351, GSE85566, GSE85647, GSE87095, GSE87640, GSE98203), GSE132547 removed by D13; U2a the same held-out arrays at each tissue test's sealed sites; U3 GSE127824 and GSE167202. Three held-out sets are the held-out halves of build series, named as held-out halves in the plan's held-out assignments (atlas_v21_plan.md) (E-MTAB-4957 colon epithelium, used for the colon blind sites; GSE224807; GSE306226); the commissioned cell references rest only on laboratories that built nothing.

**Prior reading:** GSE103541, GSE118144, GSE122244 and GSE226298 are build sources, read to build the atlas; GSE184269 was read in the C-score band test (CANON/data_spent.csv), never against the atlas. None of the commissioned uses rests on them.

**Synthetic test:** chain/stage_atlas_u1_planted.py, chain/stage_atlas_u2.py, chain/stage_atlas_u3.py, DEV_ATLAS_V21_01/planted_u1_v21_summary.txt, DEV_ATLAS_V21_01/planted_u2_v21_summary.txt, DEV_ATLAS_V21_01/planted_u3_v21_summary.txt

Each use's planted test passed before its real scoring (box jobs 85685c55, 8abbdbe9, 50b9a3b9).

**Code and outputs (chain only):** `chain/stage_atlas_u1.py` with `chain/stage_atlas_u1_planted.py` (U1), `chain/stage_atlas_u2.py` (U2a),
`chain/stage_atlas_u3.py` (U3), each with its planted mode; the bars and the per-use status in `chain/stage_atlas_commission.py`
(`ATLAS_V21_COMMISSIONING`, `atlas_v21_use_status`). The scorers first ran from the test folder; their chain copies (logic unchanged) were
rerun from the chain and reproduce every committed result exactly (`data/DEV_ATLAS_V21_01/repro_chain_copies_v21.json`). Results
`u1_scored_v21.json`, `u2_scored_v21.json`, `u3_scored_v21.json`; the compact copy `Biological_Physics/MethylPhys/atlas/v2.1/`.

**What it would take to go further.** Each further cell reference: a held-out set from an independent laboratory with >= 95 % of arrays in range,
or the next version's range carrying a measured laboratory-by-site spread, re-sealed and tested on laboratories never read. Composition: U3 met on an
adult healthy truth set. Breast and liver blind sites: held-out epithelium.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements.

**In plain words** (IAM-Atlas v2.1 commissioned for three cell references and the colon and lung tissue tests): healthy cells from laboratories that never helped build the atlas were read against it, by bars fixed before it existed. Three cell types landed inside its predicted range, and the sites the colon and lung cancer tests rely on held steady in every cell type that could be checked. The other uses stay in development.

<!-- PLAIN:BEGIN (generated by CANON/results_to_tex.py from CANON/results_register.json; do not edit by hand) -->
**In plain words** (IAM-Atlas v2.1 commissioned for three cell references and the colon and lung tissue tests' blind sites, each on a bar sealed before the atlas existed): Commissioned means a use of the atlas is allowed to report results. Healthy cells from laboratories that never helped build the atlas were read against it, by bars fixed before it existed. Three cell types, brain neurons, airway lining and naive helper T cells, landed inside the atlas's predicted range, and the sites the colon and lung cancer tests depend on held steady in every cell type that could be checked. Its blood-composition use and its other cell references stay in development.
<!-- PLAIN:END -->
