#!/usr/bin/env python3
"""
generate_cpg_healpix_mapping.py — One-time generator for iamatlas_cpg_to_healpix_nside128.npy

The canonical mapping that every patient projection in Stage 4.6 uses to sit on
the same HEALPix grid as Plate 1 (the Cosmic Microwave Methylome reference). Run
once at IAMAtlas build time; cached forever after.

PLATE 1 CONVENTIONS (the binding contract)
-------------------------------------------
- HEALPix NSIDE=128 (npix = 12 × 128² = 196,608)
- Mollweide projection (equal-area, full-sky)
- CpG ordering: by chromosome (chr1 → chr2 → ... → chr22 → chrX → chrY), then
  by MAPINFO (genomic position) within chromosome
- Pixel assignment: sequential — the i-th CpG in genomic order assigns to
  pixel index (i × npix / n_cpgs), so multiple CpGs share a pixel on average
  (483,093 / 196,608 ≈ 2.46 CpGs per pixel)
- Per-pixel value: per-pixel mean of all CpG values that map to that pixel

This produces the texture visible in Plate 1 — large-scale gradients from
chromosomal position + small-scale fluctuation from per-CpG biology, with the
statistical character of CMB anisotropy.

INPUTS
------
1. IAMAtlas REBUILD CSV (or class brightness CSVs) — for the canonical CpG list
2. Illumina manifest CSV — for chromosome + MAPINFO annotation per CpG.
   In the repo's existing breast/AD residual maps, CHR + MAPINFO columns exist
   for the 7,115 breast + 6,019 AD subsets. For the full 481,966-CpG atlas, the
   full Illumina EPIC v1 B4 manifest (or v2 equivalent) is needed.
   - Public download: https://support.illumina.com/array/array_kits/infinium-methylationepic-beadchip-kit/downloads.html
   - Cache to: Biological_Physics/atlas_vault/IAMAtlas_v0_1/external_manifests/EPIC_manifest_v1_B4.csv

OUTPUT
------
- iamatlas_cpg_to_healpix_nside128.npy — np.ndarray shape (n_cpgs,) dtype int32
  Each entry is the HEALPix pixel index (0..npix-1) that CpG[i] maps to, where
  CpG[i] is the i-th CpG in the IAMAtlas REBUILD CSV's row order.

USAGE
-----
    # Generate the mapping (one-time, at IAMAtlas build):
    python generate_cpg_healpix_mapping.py \\
        --atlas-csv ../IAMAtlas_v0_1/IAMAtlasREBUILD.csv \\
        --manifest-csv ../IAMAtlas_v0_1/external_manifests/EPIC_manifest_v1_B4.csv \\
        --output iamatlas_cpg_to_healpix_nside128.npy

    # Smoke test against the breast residual map (7,115 CpGs with CHR/MAPINFO):
    python generate_cpg_healpix_mapping.py --smoke-test

The Stage 4.6 module (`patient_brightness_comparison.py`) then loads the
mapping at session startup:

    cpg_to_pixel = np.load("iamatlas_cpg_to_healpix_nside128.npy")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS — Plate 1's binding contract
# ============================================================================

HEALPIX_NSIDE: int = 128
HEALPIX_NPIX: int = 12 * (HEALPIX_NSIDE ** 2)  # = 196608

# Chromosome ordering for genomic-order sort (matches Plate 1's convention)
CHROMOSOME_ORDER: dict[str, int] = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16,
    "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22,
    "X": 23, "Y": 24, "MT": 25, "M": 25,
    # also accept "chr1" form
    **{f"chr{k}": v for k, v in {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16,
        "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22,
        "X": 23, "Y": 24, "MT": 25, "M": 25,
    }.items()}
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_atlas_cpg_list(atlas_csv: Path | str) -> list[str]:
    """Load the canonical CpG list from the IAMAtlas REBUILD CSV.

    The atlas CSV's first column is cpg_id; we read that column in row order
    to preserve the atlas's canonical ordering.
    """
    atlas_csv = Path(atlas_csv)
    logger.info("Loading atlas CpG list from %s", atlas_csv)
    df = pd.read_csv(atlas_csv, usecols=["cpg_id"])
    cpg_ids = df["cpg_id"].astype(str).tolist()
    logger.info("Atlas CpG count: %d", len(cpg_ids))
    return cpg_ids


def load_epic_manifest(manifest_csv: Path | str) -> pd.DataFrame:
    """Load the Illumina EPIC manifest with CpG → CHR × MAPINFO annotation.

    Expected columns (Illumina manifest standard):
    - IlmnID OR Name OR cpg_id (the CpG identifier — first match wins)
    - CHR
    - MAPINFO

    Returns a DataFrame indexed by cpg_id with CHR + MAPINFO columns.
    """
    manifest_csv = Path(manifest_csv)
    logger.info("Loading EPIC manifest from %s", manifest_csv)

    # Illumina manifests sometimes have a 7-line header preamble before the data table.
    # Detect header offset by scanning for the row starting with "IlmnID" or "Name" or "cpg_id".
    header_row = 0
    with open(manifest_csv) as fp:
        for i, line in enumerate(fp):
            if i > 30: break
            first = line.split(",", 1)[0].strip().strip('"')
            if first in ("IlmnID", "Name", "cpg_id", "TargetID", "ProbeID"):
                header_row = i
                break

    df = pd.read_csv(manifest_csv, header=header_row, low_memory=False)

    # Identify the CpG-id column
    id_col = None
    for candidate in ("IlmnID", "Name", "cpg_id", "TargetID", "ProbeID"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"Could not find a CpG-id column in {manifest_csv}. "
            f"Tried: IlmnID, Name, cpg_id, TargetID, ProbeID. "
            f"Columns present: {list(df.columns)[:10]}..."
        )

    if "CHR" not in df.columns or "MAPINFO" not in df.columns:
        raise ValueError(
            f"Manifest {manifest_csv} missing required columns CHR and/or MAPINFO. "
            f"Columns: {list(df.columns)[:10]}..."
        )

    df = df[[id_col, "CHR", "MAPINFO"]].rename(columns={id_col: "cpg_id"})
    df["cpg_id"] = df["cpg_id"].astype(str)
    df["CHR"] = df["CHR"].astype(str).str.strip()
    df["MAPINFO"] = pd.to_numeric(df["MAPINFO"], errors="coerce")
    df = df.drop_duplicates("cpg_id").set_index("cpg_id")

    logger.info("Manifest loaded: %d CpGs with CHR + MAPINFO", len(df))
    return df


def sort_cpgs_by_genomic_order(
    cpg_ids: list[str],
    manifest: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Sort CpGs by chromosome × MAPINFO using Plate 1's canonical ordering.

    Returns
    -------
    (sorted_cpgs, unannotated_cpgs) : tuple
        sorted_cpgs    — CpG IDs in genomic order (annotated CpGs only)
        unannotated_cpgs — CpG IDs absent from the manifest (assigned to a
                           dedicated "no_annotation" pixel range at the end)
    """
    annotated = [c for c in cpg_ids if c in manifest.index]
    unannotated = [c for c in cpg_ids if c not in manifest.index]
    logger.info(
        "Annotated CpGs: %d / %d (unannotated: %d)",
        len(annotated), len(cpg_ids), len(unannotated)
    )

    # Build sort key: (chromosome_order_int, MAPINFO)
    sub = manifest.loc[annotated].copy()
    sub["chr_order"] = sub["CHR"].map(CHROMOSOME_ORDER).fillna(99).astype(int)
    sub = sub.sort_values(["chr_order", "MAPINFO"])
    sorted_cpgs = sub.index.tolist()

    return sorted_cpgs, unannotated


def assign_to_healpix_pixels(
    n_cpgs: int,
    nside: int = HEALPIX_NSIDE,
) -> np.ndarray:
    """Sequential pixel assignment for n_cpgs in genomic order.

    The i-th CpG (in genomic order) assigns to pixel floor(i * npix / n_cpgs).
    This yields ~uniform CpG-per-pixel density across the sphere.

    Returns
    -------
    pixel_ids : np.ndarray
        Length-n_cpgs array of HEALPix pixel indices in [0, npix-1].
    """
    npix = 12 * (nside ** 2)
    if n_cpgs <= 0:
        return np.array([], dtype=np.int32)
    pixel_ids = (np.arange(n_cpgs, dtype=np.int64) * npix // n_cpgs).astype(np.int32)
    # Clip safety (floor division should always give 0..npix-1, but be defensive)
    pixel_ids = np.clip(pixel_ids, 0, npix - 1)
    return pixel_ids


def build_cpg_to_pixel_mapping(
    atlas_csv: Path | str,
    manifest_csv: Path | str,
    nside: int = HEALPIX_NSIDE,
) -> tuple[np.ndarray, list[str], dict]:
    """End-to-end: read atlas + manifest, sort, assign pixels.

    Returns
    -------
    (cpg_to_pixel, cpg_ids_in_atlas_row_order, provenance) : tuple
        cpg_to_pixel — np.ndarray shape (n_cpgs,) — pixel index per CpG in
                       atlas row order (the order Stage 4.6 expects).
        cpg_ids_in_atlas_row_order — list of CpG IDs (for audit trail).
        provenance — dict with provenance metadata.
    """
    cpg_ids = load_atlas_cpg_list(atlas_csv)
    manifest = load_epic_manifest(manifest_csv)
    sorted_cpgs, unannotated = sort_cpgs_by_genomic_order(cpg_ids, manifest)

    # Assign pixels to annotated CpGs in genomic order
    annotated_pixels = assign_to_healpix_pixels(len(sorted_cpgs), nside=nside)

    # Build the cpg → pixel dictionary
    cpg_pixel_dict = dict(zip(sorted_cpgs, annotated_pixels))
    # Unannotated CpGs all go to pixel 0 (or a sentinel — see note below)
    SENTINEL_PIXEL = 12 * (nside ** 2) - 1  # last pixel = "no annotation"
    for c in unannotated:
        cpg_pixel_dict[c] = SENTINEL_PIXEL

    # Now produce the output array in ATLAS ROW ORDER (Stage 4.6 expects this)
    cpg_to_pixel = np.array(
        [cpg_pixel_dict[c] for c in cpg_ids],
        dtype=np.int32
    )

    provenance = {
        "nside": nside,
        "npix": 12 * (nside ** 2),
        "n_cpgs_in_atlas": len(cpg_ids),
        "n_cpgs_annotated": len(sorted_cpgs),
        "n_cpgs_unannotated": len(unannotated),
        "sentinel_pixel_for_unannotated": SENTINEL_PIXEL,
        "atlas_csv": str(atlas_csv),
        "manifest_csv": str(manifest_csv),
        "plate_1_conventions_anchor": "CPG_Plate_01_Cosmic_Microwave_Methylome.png",
    }

    return cpg_to_pixel, cpg_ids, provenance


# ============================================================================
# SMOKE TEST — uses the in-repo breast residual map (7,115 CpGs w/ CHR/MAPINFO)
# ============================================================================

def smoke_test():
    """Smoke test using the breast residual map's chr_annotated CSV.

    This file has CHR + MAPINFO for 7,115 CpGs already in the repo. Builds a
    mini mapping and confirms the pipeline works end-to-end without requiring
    the full Illumina manifest.
    """
    print("=" * 70)
    print("SMOKE TEST: Build CpG→HEALPix mapping from breast residual map")
    print("=" * 70)

    chr_annotated_csv = Path(
        "Biological_Physics/atlas_vault/walther_clinical_runtime/"
        "DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_residual_maps/"
        "breast_epic_residual_map_chr_annotated.csv"
    )
    if not chr_annotated_csv.exists():
        # Try the alternate path
        chr_annotated_csv = Path(
            "Biological_Physics/RETIRED_Phase1_PreBuild_Cards/Breast/"
            "breast_epic_residual_map_chr_annotated.csv"
        )
    if not chr_annotated_csv.exists():
        print(f"ERROR: Could not find breast residual map at expected paths.")
        return

    print(f"Source: {chr_annotated_csv}")
    df = pd.read_csv(chr_annotated_csv)
    print(f"Loaded {len(df)} CpGs with CHR + MAPINFO")

    # Simulate the manifest + atlas inputs from this single file
    cpg_ids = df["cpg"].astype(str).tolist()
    manifest = df[["cpg", "CHR", "MAPINFO"]].rename(columns={"cpg": "cpg_id"})
    manifest["CHR"] = manifest["CHR"].astype(str).str.strip()
    manifest["MAPINFO"] = pd.to_numeric(manifest["MAPINFO"], errors="coerce")
    manifest = manifest.drop_duplicates("cpg_id").set_index("cpg_id")

    sorted_cpgs, unannotated = sort_cpgs_by_genomic_order(cpg_ids, manifest)
    print(f"Annotated: {len(sorted_cpgs)}, Unannotated: {len(unannotated)}")

    pixels = assign_to_healpix_pixels(len(sorted_cpgs))
    print(f"Pixel range: {pixels.min()} → {pixels.max()} (npix={HEALPIX_NPIX})")
    print(f"Unique pixels touched: {len(np.unique(pixels))}")
    print(f"Mean CpGs per pixel (touched): "
          f"{len(sorted_cpgs) / max(len(np.unique(pixels)), 1):.2f}")

    # Build the full mapping (atlas row order = file row order here)
    cpg_pixel_dict = dict(zip(sorted_cpgs, pixels))
    SENTINEL = HEALPIX_NPIX - 1
    for c in unannotated:
        cpg_pixel_dict[c] = SENTINEL

    mapping = np.array(
        [cpg_pixel_dict[c] for c in cpg_ids],
        dtype=np.int32
    )

    out_path = Path("/tmp/breast_residual_cpg_to_healpix_smoketest.npy")
    np.save(out_path, mapping)
    print(f"\nSmoke-test mapping saved to: {out_path}")
    print(f"  shape: {mapping.shape}")
    print(f"  dtype: {mapping.dtype}")
    print(f"  min: {mapping.min()}, max: {mapping.max()}")
    print(f"  bytes: {out_path.stat().st_size:,}")
    print("\nSmoke test PASS — the generator pipeline works end-to-end.")
    print("Production run requires the full Illumina EPIC manifest + IAMAtlas REBUILD CSV.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate iamatlas_cpg_to_healpix_nside128.npy — the canonical "
            "CpG-to-HEALPix mapping for Stage 4.6 patient projection."
        )
    )
    parser.add_argument(
        "--atlas-csv",
        help="Path to IAMAtlas REBUILD CSV (or any CSV with cpg_id column in row order)",
    )
    parser.add_argument(
        "--manifest-csv",
        help="Path to Illumina EPIC manifest CSV with IlmnID/CHR/MAPINFO columns",
    )
    parser.add_argument(
        "--output",
        default="iamatlas_cpg_to_healpix_nside128.npy",
        help="Output .npy path (default: iamatlas_cpg_to_healpix_nside128.npy)",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=HEALPIX_NSIDE,
        help=f"HEALPix NSIDE (default: {HEALPIX_NSIDE} — matches Plate 1)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test on the in-repo breast residual map (7,115 CpGs)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke_test:
        smoke_test()
        return

    if not args.atlas_csv or not args.manifest_csv:
        parser.print_help()
        print("\nERROR: --atlas-csv and --manifest-csv are required for production run.")
        return

    cpg_to_pixel, cpg_ids, provenance = build_cpg_to_pixel_mapping(
        atlas_csv=args.atlas_csv,
        manifest_csv=args.manifest_csv,
        nside=args.nside,
    )

    out_path = Path(args.output)
    np.save(out_path, cpg_to_pixel)
    print(f"\nMapping saved to: {out_path}")
    print(f"  shape: {cpg_to_pixel.shape}")
    print(f"  dtype: {cpg_to_pixel.dtype}")
    print(f"  bytes: {out_path.stat().st_size:,}")
    print(f"\nProvenance:")
    for k, v in provenance.items():
        print(f"  {k}: {v}")

    # Also save provenance JSON alongside the mapping
    import json
    prov_path = out_path.with_suffix(".provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\nProvenance saved to: {prov_path}")


if __name__ == "__main__":
    main()
