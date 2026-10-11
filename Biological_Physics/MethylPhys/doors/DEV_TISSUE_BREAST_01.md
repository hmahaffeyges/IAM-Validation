# DEV-TISSUE-BREAST-01 — Met-A on breast tumour against the same patient's normal tissue (development; design, 2026-10-11; NOT sealed)

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

**Before any seal, still to run (synthetic, normal tissue only):** planted change inside a hidden component; null on the real TCGA normals (leave-one-out against the other normals); array
batch (plate) structure of the normals; power at 91 pairs with the spread measured on the real normals.
