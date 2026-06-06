# RETIRED PRE-BUILD REFERENCE — Immune class materials

This folder holds the pre-build (multi-atlas era) immune class drafts that were used as conceptual reference when building the IAMAtlas-native immune card. All materials here are **RETIRED** — they used Xu-538 / Loyfer / Moss / Salas / EpiSCORE / Caggiano / UniLIFE references that we no longer use at runtime. The current chain is IAMAtlas-only.

These files are preserved because they encode genuine operational insight from the pre-build era — pattern taxonomies, cross-cancer immune-class comparisons, per-immune-cell-type clinical reference pages, the immune-intelligence audit matrix, and so on. The next AI session can read these for *conceptual* reference when extending the IAMAtlas-native immune card. **Do not import the methods (Xu-538 etc.) directly — they are obsolete by the BUILD_SPEC §3.7 rule #2 "Single IAMAtlas, only IAMAtlas at runtime."**

## Subfolders

- `Immune_Atlas_PreBuild_RETIRED/` — the v0.3.2 immune-atlas "Rosetta Reference Card" pre-build, plus VAL-082 heme/AML, the Immune Class split test materials, Stage 1 cross-cancer comparison, retroactive Phase 1 commitment audit.
- `Immune_Class_Reference_PreBuild_RETIRED/` — the draft v1.0 immune card JSON, cross-reference module spec, immune-intelligence audit matrix v2, immune-cell-page audit coverage doc, planned website page outline, and individual cell pages (B-cells, CD4 T, CD8 T, NK, basophils, dendritic, eosinophils, Kupffer, macrophages, memory B, memory T, microglia, monocytes, naive B, naive CD4, naive CD8, neutrophils, plasma cells, regulatory T).

## What's live in the repo for the actual immune work

The IAMAtlas-native immune card is being built fresh under `Biological_Physics/atlas_vault/walther_clinical_runtime/DISEASE_MAPS_CARDS/Immune_Atlas/` (when created). It uses only IAMAtlas REBUILD + the Walther deconvolver + the A-scoring marker artifact + the Mahalanobis healthy hull. The pre-build conceptual content above is read for *what the card needs to express*; the implementation is rebuilt from scratch on IAMAtlas per the SOP v1.2 chain.
