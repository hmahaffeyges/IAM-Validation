# IAMAtlas v0.1 — MCMC Build Lessons (READ BEFORE ANY REBUILD)

**Audience:** the next person or AI who adds cell types / atlases to a class and re-runs the MCMC.
**Date:** 2026-05-25. **Author context:** written immediately after the v0.1 build so the fix isn't lost.

## TL;DR — the one rule

**For any rebuild, use `iamatlas_v0_1_mcmc_batched_FIXED.py`. Do NOT use the older `iamatlas_v0_1_mcmc_batched.py`.** The FIXED script contains a two-part model-specification fix that the old one lacks. The old script "worked" for 7 of 8 classes only because their data was strong enough to mask a latent bug. A rebuild may bring in marginal data that exposes it, exactly as stromal did.

## What the two scripts are

| Script | Status | Use when |
|---|---|---|
| `iamatlas_v0_1_mcmc_batched_FIXED.py` | **Canonical. Use this.** | Always, for any build or rebuild. |
| `iamatlas_v0_1_mcmc_batched.py` (older) | Historical. The 7 non-stromal v0.1 classes were built with it. | Never again. Kept only for provenance. |

## The bug (two compounding problems)

During the v0.1 build, the stromal class would not converge under the old script — divergences in the thousands at low `target_accept`, ESS collapse at high. Four wrong diagnoses were chased (cell count, a catch-all bucket, sampler config, treating each failure as the same) before the real cause was found by reading the model code and testing in isolation. It was two problems in `build_class_model`:

1. **The swamp.** The latent `z` was shaped `(n_cpg, n_ct)` — a parameter for every cell type at every CpG, even where that cell was never measured. Those unmeasured pairs are unconstrained by the likelihood (pure prior), forming a high-dimensional region NUTS cannot traverse. For stromal at 9 cells with 200:1 coverage imbalance, ~63% of parameters were unconstrained.
   **Fix:** define parameters only over observed (cpg, celltype) pairs (`n_pairs`, `obs_idx_pair`), never the full grid.

2. **The ridge (the real killer).** The old parameterization `mu_logit = mu_class_logit + sigma_class_logit * z` let `sigma_class_logit` (spread of the per-CpG means) trade off against `log_kappa` (per-atlas observation precision). The data identifies only their combination, creating a posterior ridge. ESS collapsed on exactly those two parameters (~8) while everything else mixed fine.
   **Fix:** model each pair's logit-mean directly — `mu_logit = pm.Normal("mu_logit", mu=mu_class_logit, sigma=3.0, shape=n_pairs)` — removing `sigma_class_logit` so `log_kappa` has nothing to trade against.

## Why the other 7 classes are still valid (and were NOT re-run)

The 7 non-stromal classes (immune, cycling, secretory, terminal, progenitor, stem_pluri, stem_adult) were built with the OLD script and converged cleanly (R-hat < ~1.05, good ESS, ~0 divergences). The bug was present in their model too, but their data was strong/balanced enough to overpower the ridge — they crossed the convergence threshold anyway.

This was verified, not assumed: the FIXED model was shown to reproduce the OLD model's posterior means on balanced data at correlation **0.99996**. Because the two models agree on well-behaved data, re-running the 7 good classes would cost ~18 days of compute to reproduce essentially identical numbers. So they were deliberately left as-is. They are valid.

**Implication for rebuilds:** if you re-run one of those 7 classes (to add cells/atlases), use the FIXED script. The result will match the old one where data is strong and will be *more robust* where any newly added data is marginal. There is no downside to the FIXED script and a real downside to the old one.

## Stromal-specific notes

- Sealed as **5 cells**: adipocyte, endothelial, fibroblast, smooth_muscle, stromal_other.
- Convergence (FIXED script): 0 divergences across all 16 batches, Pearson 0.934, MAE 0.078.
- ESS is soft on smooth_muscle and stromal_other (each only 2 atlases) — wide credible intervals, honest uncertainty, not error. Means (which drive scoring) are fine.
- **4 cells dropped from v0.1** for being single-atlas / too sparse: placenta, astrocyte, stellate, pericyte. Flagged for reintroduction in v0.2 when better-supported atlases arrive (e.g. a dedicated brain/CSF atlas would properly support astrocyte).

## Rebuild recipe (future)

1. Add new observation rows to `iamatlas_mcmc_inputs.csv` (long format: one row per cpg x celltype x atlas). The inputs CSV is in the repo at `iamatlas_production_data/inputs/`.
2. Run **`iamatlas_v0_1_mcmc_batched_FIXED.py`** for the affected class(es) only. Example:
   `python iamatlas_v0_1_mcmc_batched_FIXED.py --classes <class> --batch_size 1500 --tune 1000 --draws 1000 --chains 4 --target_accept 0.95 --inputs <inputs.csv> --out_dir <out>`
3. Check the per-batch convergence; 0 divergences is the key signal the geometry is healthy. ESS softness on thin-data cells is acceptable for v0.x (means are what drive scoring).
4. Re-run `merge_iamatlas_v0_1.py` to rebuild `IAMAtlas.csv` from all 8 per-class outputs.
5. Version the new atlas in the vault with a new SHA; bump the version folder (IAMAtlas_v0_2/ etc).

## Environment note

- The FIXED script uses `az.summary(..., stat_focus="mean")`, which requires a newer arviz (present on the build machine). Older arviz lacks that argument — if a future environment errors on it, remove the `stat_focus="mean"` argument (it only affects which statistic the convergence summary centers on, not the model or the saved posteriors).
