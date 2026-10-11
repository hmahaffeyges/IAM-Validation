# DEV-TISSUE-BREAST-01 — Met-A on breast tumour against the same patient's normal tissue (development; SEALED 2026-10-11)

**Data.** TCGA-BRCA open-access 450K IDATs (NCI GDC), listed by `data/DEV_TISSUE_BREAST_01/fetch_tcga_brca.py list`
(`tcga_brca_450k_files.csv`). Only the 'Solid Tissue Normal' files are fetched; tumour files are fetched only after the seal is committed.
GSE69914 (breast, healthy donors, adjacent normal, cancer) was set aside: it ships submitter betas, not IDATs, and no patient IDs.

**Synthetic test 1, composition** (`development/sims/breast_tissue_01.py`, output `breast_tissue_01_output.txt`; atlas posterior draws only).
A tumour holds far more epithelium than its normal tissue. With no methylation change, that alone moves plain Met-A by -0.0756 per pair and hides a
planted 2 % change (-0.0047, 4.4 % of tumours above the healthy 95th percentile). Met-A read against each sample's own composition-expected pattern
moves by -0.0067 from composition alone (away from a fingerprint) and finds the 2 % change in every tumour (+0.0716). **The test uses the
composition-aware reading.**

**Synthetic test 2, template mismatch** (`development/sims/breast_tissue_02.py`, output `breast_tissue_02_output.txt`). True tissue built with
heart fibroblasts only and a hidden smooth-muscle component (0-10 %, absent from the panel). Composition-aware Met-A: composition alone -0.0050,
1.1 % of tumours above the healthy 95th percentile with no change; planted 2 % change +0.0691, found in every tumour. Plain Met-A again hides it.

**Real-normal step 3** (`data/DEV_TISSUE_BREAST_01/null_normals.py`, outputs `null_normals_output.csv`, `null_normals_summary.txt`; 97 TCGA
normal-tissue arrays, Stage 1, full atlas posteriors, each read against the other 96). The real spread is far wider than simulated: plain SD
0.0830, composition-aware SD 0.0651 (simulated 0.0115 and 0.0039). The fitted composition gives fibroblasts 0 and endothelium 0.168: the panel
(no breast fibroblast in the atlas) does not describe real breast stroma. **Synthetic tests 1-2 used a noise model 16x too small; no seal until
the source of the spread is found and the simulation is rebuilt on it.**

**Spread sources** (`spread_sources.py`, `spread_sources_output.txt`): the 450K processing plate dominates (12 plates, Kruskal-Wallis p 2.50e-11;
plate medians 0.923-1.155); endothelium share Spearman +0.440. **Same-plate tare** (`plate_tare_check.py`, `plate_tare_check_output.txt`):
composition-aware SD 0.0286, 95th percentile 1.0563. 56 of 91 patients have tumour and normal on one plate; 212 of 791 tumour aliquots sit on
plates holding >= 3 normals.

**Synthetic test 4 on the real normals: FAILED** (`plant_real_normals.py`, `plant_real_normals_output.csv`). A real normal blended toward
epithelium (atlas luminal mean, 30-60 %) with no methylation change reads 1.061 and crosses the healthy bar in 0.617 of cases (no change: 0.053).
The composition-aware reading does not hold its null on real tissue: either the atlas panel does not describe breast tissue (no breast fibroblast;
fitted endothelium 0.168) or an atlas template mixed into a 450K array does not behave like real epithelium. **The test is not sealed in this
form.** Designs to evaluate next (synthetic first): a reading on sites whose healthy value is the same in every breast tissue cell type
(composition-blind by construction); within-patient same-plate pairs only; normals with a measured epithelial share to replace the template plant.

**Synthetic test 5, composition-blind sites** (`blind_sites_test.py`; sites whose atlas value is within 0.05 of the luminal value in every panel
cell; plate tare). Composition shift alone crosses the healthy bar at 0.266 with the atlas template (`blind_sites_test_output.csv`), 0.032 with
a real 450K primary mammary epithelial array as template (ENCSR000ABL, `blind_sites_test_ENCSR000ABL_output.csv`; planted 5 % found 0.915), and
0.638 / 0.681 with the two MCF 10A 450K arrays (`..._ENCSR000ABG_output.csv`, `..._ENCSR000ACO_output.csv`). MCF 10A is an immortalised line
with its own culture-acquired gain (DEV-FINGERPRINT-02 breast_mcf10a), so blending it adds abnormal methylation: not a clean null. The
composition-aware reading fails with the real template too (0.245, `plant_real_normals_hmec450k_output.csv`) and is dropped.
**Synthetic test 6, no plant** (`blind_vs_composition.py`, `blind_vs_composition_output.txt`): across 94 real normals (epithelial share
0.002-0.737) the plate-tared blind reading does not follow composition (Spearman -0.142, p 0.17): passed. The shares come from the same
imperfect panel, so this is supporting, not decisive.

**Before any seal, still to run (synthetic, normal tissue only):** planted change inside a hidden component; null on the real TCGA normals (leave-one-out against the other normals); array
batch (plate) structure of the normals; power at 91 pairs with the spread measured on the real normals.

## Sealed 2026-10-11T16:53:43Z before any tumour array is downloaded or read
**Sealed:** `data/DEV_TISSUE_BREAST_01/sealed_breast_tissue.json` (scorer `score_breast_tissue.py`, sha256 in the seal; the score step refuses
unless the seal is committed unchanged). Reading: Met-A at the composition-blind sites (3,000 + 3,000, listed in the seal), each tumour against
its own patient's normal on the same 450K plate. Pairs: the 53 patients whose tumour and normal share a plate holding >= 3 normals.
**Rule:** FINGERPRINT if >= 6 of 53 pairs read Met-A(tumour)/Met-A(own normal) above 1.0943 at the sealed blind sites; else NO FINGERPRINT.
Power (`power_pairs_output.csv`, stand-in pairs of two different same-plate normals): a planted 2 % change crosses the pair bar in 0.6 of pairs and
the group call in every draw; no change and composition shift alone give group false calls of 0.017 and 0.004.
**Synthetic test:** DEV_TISSUE_BREAST_01/power_pairs.py, DEV_TISSUE_BREAST_01/power_pairs_output.csv, DEV_TISSUE_BREAST_01/blind_sites_test_ENCSR000ABL_output.csv, DEV_TISSUE_BREAST_01/blind_vs_composition_output.txt
