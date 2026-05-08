# Gasparoni 2018 Terminal-Class Addition — Staging Bundle

**Date staged:** 2026-05-06
**Decision authority:** Heath W. Mahaffey
**Status:** STAGED, NOT APPENDED. Production matrix (`iamatlas_mcmc_inputs.csv`) unmodified.

## Purpose

Adds Gasparoni 2018 (GSE66351, occipital cortex FANS-sorted brain methylation atlas)
to terminal class. Closes the data-imbalance between terminal class (~31K rows
before this addition) and the rest of the architecture-class pool. After append:
terminal class will grow to ~988K rows, comparable to secretory/cycling/progenitor.

## Source

- **File:** `gasparoni_2018_GSE66351_brain_celltype_atlas.csv` (extracted 2026-05-04)
- **Citation:** Gasparoni et al., Clinical Epigenetics (2018), PMID 30045751
- **GEO accession:** GSE66351
- **Platform:** Illumina HumanMethylation450 (HM450)
- **Donors:** n=16, occipital cortex, all CTRL condition
- **Source rows:** 957,638 (478,819 CpGs × 2 cell types)

## Decisions made (Heath, 2026-05-06)

1. **One IAMAtlas, deepens — not multiple atlases.** Gasparoni feeds into the
   single IAMAtlas as additional terminal-class input. No customer-facing
   provenance exposure.
2. **Cell-type label mapping:**
   - `cortical_neuron` (Gasparoni) → `Cortical_neurons` (matches Loyfer's
     existing label exactly; allows the model to pool donors at the per-CpG
     level for the tightest possible posterior — astro-genetic statistical
     efficiency by design)
   - `cortical_glia` (Gasparoni) → `Glia` (new terminal-class cell type;
     biologically honest because Gasparoni's NeuN-negative population mixes
     oligodendrocytes, astrocytes, and microglia; class-level hyperprior
     pools across cell types within terminal regardless)
3. **No researcher-name labels.** Atlas source `gasparoni_2018` is internal
   provenance metadata only. Customer-facing outputs do not expose it.
4. **Beta clamping at load time** matches the loader behavior in
   `iamatlas_v0_1_mcmc_batched.py` (lines 110-111): β ≤ 0 → 1e-4,
   β ≥ 1 → 1 - 1e-4.

## Files in this bundle

| File | SHA-256 | Size | Purpose |
|---|---|---|---|
| `stage_gasparoni_for_terminal.py` | `a496e29bf3fa3078ab8f9fce0eaeac6422207eef558fdff938e56ad007306cc3` | 3.5 KB | Script that produced the staged file |
| `gasparoni_terminal_addition.csv` | `3294b49880e5248b4cf85fc6bff830a0d6c9363c600fa0f89b16c55f78854f30` | 57.5 MB | 957,638 rows ready to append |

## Append procedure (DO NOT RUN UNTIL MAIN MCMC RUN COMPLETES)

The current MCMC run (started 2026-05-04) is processing all eight classes
in one invocation. The script loaded `iamatlas_mcmc_inputs.csv` into memory
at startup and will NOT re-read it during the run. **Modifying the file
on disk during the run is safe** in the strict sense (the script doesn't
re-open the file), but the discipline is to wait until the run completes
to keep the audit trail clean.

When the run completes (estimated 8-10 days from 2026-05-06):

```bash
# 1. Backup the production matrix
cp iamatlas_mcmc_inputs.csv iamatlas_mcmc_inputs.csv.backup_pre_gasparoni

# 2. Append (skip the header line of the staging file)
tail -n +2 gasparoni_terminal_addition.csv >> iamatlas_mcmc_inputs.csv

# 3. Verify row count increased by 957,637
wc -l iamatlas_mcmc_inputs.csv
# Expected: 11,896,300 (was 10,938,663 including header)

# 4. Re-run terminal class only against expanded pool
python iamatlas_v0_1_mcmc_batched.py --classes terminal \
       --batch_size 5000 --chains 4 --tune 1000 --draws 1000 \
       --target_accept 0.95 --out_dir iamatlas_v0_1_output

# 5. Compare new terminal posterior to existing
#    Existing: R-hat 1.01, ESS 1493, Pearson 0.799, MAE 0.356
#    Expected: ESS up substantially (~32× more rows), Pearson improved,
#              H_min posterior tighter and centered closer to 0.7728 anchor
```

## What this bundle does NOT do

- Does not modify `iamatlas_mcmc_inputs.csv`
- Does not change any of the other 7 classes
- Does not change the Loyfer or other existing terminal-class atlases
- Does not introduce researcher names into customer-facing outputs
- Does not advance the IAMAtlas version number — same atlas, deepens

## Expected impact on terminal class

| Metric | Before Gasparoni | After Gasparoni (estimated) |
|---|---|---|
| Terminal rows | 30,895 | 988,533 |
| Terminal atlases | 7 | 8 |
| Terminal cell types | 11 | 12 (adds Glia) |
| Cortical_neurons rows | 6,105 | 484,924 |
| ESS (per-CpG, terminal) | 1493 | likely 5,000+ |
| Pearson (held-out) | 0.799 | likely 0.85+ |
| H_min posterior centered | 0.7728 anchor | unchanged anchor, tighter posterior |
