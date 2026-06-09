# IAMAtlas HEALPix Mapping — Stage 4.6 grid contract

**Date:** 2026-06-06
**Module:** `generate_cpg_healpix_mapping.py`
**Output:** `iamatlas_cpg_to_healpix_nside128.npy` (to be generated; see Production Run below)
**Reference:** CPG Plate 1 — `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Methylome_Background.png`

## What this folder contains

The canonical CpG-to-HEALPix mapping that Stage 4.6 uses to project per-patient z-score departures onto the same HEALPix grid as Plate 1. This is **the binding contract** between every patient's personal Cosmic Methylome Background and the framework's reference visualization.

The mapping is generated **once** at IAMAtlas build time and cached as a `.npy` file. Stage 4.6's `patient_brightness_comparison.py` loads it at session startup with `np.load("iamatlas_cpg_to_healpix_nside128.npy")`.

## Plate 1's binding conventions

| Convention | Value |
|---|---|
| HEALPix NSIDE | 128 |
| HEALPix npix | 196,608 |
| Projection | Mollweide (equal-area, full-sky) |
| CpG ordering | by chromosome (chr1 → chr2 → ... → chr22 → chrX → chrY), then by MAPINFO within chromosome |
| Pixel assignment | sequential — i-th CpG in genomic order → pixel `floor(i × npix / n_cpgs)` |
| Multi-CpG per pixel | averaged per pixel (mean of CpG values mapped to that pixel) |
| Mean CpGs per pixel | 483,093 / 196,608 ≈ 2.46 |

The mapping is **deterministic** — given the same atlas CpG list and the same Illumina manifest, the generator produces byte-identical output.

## Production run

To generate the production mapping (run once at IAMAtlas build time):

```bash
# Acquire the Illumina EPIC v1 B4 manifest (or v2 equivalent) — public download:
# https://support.illumina.com/array/array_kits/infinium-methylationepic-beadchip-kit/downloads.html
# Cache it at:
mkdir -p ../external_manifests/
cp ~/Downloads/MethylationEPIC_v1_B4.csv ../external_manifests/

# Run the generator:
python generate_cpg_healpix_mapping.py \
    --atlas-csv ../IAMAtlasREBUILD.csv \
    --manifest-csv ../external_manifests/MethylationEPIC_v1_B4.csv \
    --output iamatlas_cpg_to_healpix_nside128.npy
```

Output:
- `iamatlas_cpg_to_healpix_nside128.npy` — np.ndarray shape (483,093,) dtype int32; pixel index per CpG in atlas row order
- `iamatlas_cpg_to_healpix_nside128.provenance.json` — provenance metadata (inputs, npix, n_cpgs_annotated, sentinel pixel for unannotated CpGs)

Expected runtime: 10–60 seconds depending on disk speed (most of the time is reading the EPIC manifest CSV).

## Smoke test (no external manifest needed)

The generator includes a smoke-test mode that builds a mini-mapping using the in-repo breast residual map's chr_annotated CSV (7,115 CpGs already carrying CHR + MAPINFO):

```bash
python generate_cpg_healpix_mapping.py --smoke-test
```

Expected output:
```
SMOKE TEST: Build CpG→HEALPix mapping from breast residual map
Source: ...breast_epic_residual_map_chr_annotated.csv
Loaded 7,114 CpGs with CHR + MAPINFO
Annotated: 7,114, Unannotated: 0
Pixel range: 0 → 196,580 (npix=196,608)
Smoke test PASS — the generator pipeline works end-to-end.
```

The smoke test confirms the generator pipeline works without requiring the full Illumina manifest download.

## Why this lives at IAMAtlas build time, not patient runtime

The CpG-to-pixel assignment is a function of the atlas's CpG list + the genomic annotation — both fixed at IAMAtlas build time. There is no reason to recompute it per patient. The patient runtime just looks up `pixel = mapping[cpg_index]` from the cached `.npy` array.

**This decouples Stage 4.6's per-patient cost from any genomic-annotation lookup.** Once the `.npy` file is generated and committed to the repo, every patient projection uses the same grid for free.

## Unannotated CpG handling

Some atlas CpGs may not appear in the Illumina manifest (rare — typically <0.5% of EPIC CpGs). These are assigned to the **sentinel pixel** (last pixel index, `npix - 1` = 196,607). Stage 4.6 masks this pixel as "no annotation available" — it renders BLACK in the patient Mollweide, parallel to the stromal galactic mask convention.

## Versioning

The `.npy` mapping file is versioned by the IAMAtlas it was built against:
- Current: `iamatlas_cpg_to_healpix_nside128.npy` (built against IAMAtlas REBUILD v0_2, canonical SHA `41b7c16f...`)
- When the IAMAtlas is rebuilt with a different CpG list (e.g., EPIC v2 → EPIC v3 transition), the mapping is regenerated.

The `provenance.json` file records the atlas SHA + manifest version used to generate the mapping; the engine verifies these at session startup.

## Cross-references

- Stage 4.6 module: `Biological_Physics/atlas_vault/walther_clinical_runtime/Brightness_Comparison/patient_brightness_comparison.py`
- Plate 1 reference: `Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Methylome_Background.png`
- BUILD_SPEC: `Biological_Physics/atlas_vault/walther_clinical_runtime/walther_clinical_BUILD_SPEC_v1_3.md` §3.5b + §5 Stage 4.6
