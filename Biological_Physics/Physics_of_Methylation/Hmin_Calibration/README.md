# H_min calibration — the code that set the eight methylation floors and the 32 substrate floors

**Restored to HEAD 2026-09-20** (author's decision). These files were removed from the public tree on 2026-04-19 (commit 538667d, "remove commercial calibration layer") but remained in public git history (commit 22749f0) and in the author's CC-BY-4.0 Zenodo deposit of 2026-04-17 (DOI 10.5281/zenodo.19633499, which carries `gape_mcmc_g002.py`). They are restored here unchanged, byte-for-byte from 22749f0, so that Paper 1 and Issue 003 can cite them at a live path.

| file | what |
|---|---|
| `gape_mcmc_g002.py` | **G-002** — the eight methylation H_min values. emcee ensemble MCMC, 32 walkers × 5 chains × 5,000 production steps, on 37 published reference cell methylomes (4–6 per class; Roadmap/ENCODE/Lister; sources and DOIs in the header). Every chain R-hat < 1.001. |
| `gape_mcmc_g003b.py` | **G-003b** — the 32 non-methylation floors (nucleosome occupancy, fuzziness, WPS, fragment size × 8 classes). |
| `gape_bootstrap_comparison.py` / `bootstrap_vs_mcmc_comparison.tsv` | the April bootstrap cross-check of the **32 G-003b floors**: 0.168 % mean relative difference, 24/32 within the bootstrap 95 % CI. **Contains no methylation rows.** |
| `methyl_bootstrap_PROC-HMIN-BOOT-01.json` | the methylation cross-check, run 2026-09-20 with the same reference data and bootstrap function: **8/8 frozen values in CI, 0.060 % mean, 0.095 % max.** Record: `Testing_and_Code/PROC_data/PROC-HMIN-BOOT-01/`. |
| `gape_mcmc_g008.py`, `gape_mcmc_e_a_bio.py`, `gape_mcmc_nbio_ordering.py` | later MCMC studies from the same period (G-008 TCGA ordering; E_A,bio; n_bio ordering), restored for completeness. |

**Scale.** G-002 was run on GenomicStudio-normalised Roadmap β. The author's April evidence report already stated that a systematic offset to other pipelines was expected and that absolute thresholds would need cross-pipeline validation; that offset was measured in September (+0.066 on the identity loci for noob-normalised 450K IDATs) and is removed by the pipeline map (LESSON-SCALE-01, PHASE 1c).

The frozen values consumed by the engine are in `CPG_Engine/Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json`; they byte-match the G-002 posteriors.
