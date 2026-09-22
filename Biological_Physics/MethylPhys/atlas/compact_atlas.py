#!/usr/bin/env python3
"""
compact_atlas.py
================
Compacts the rebuild outputs into repo-ready archives:

  IAMAtlasREBUILD.csv.xz             — the merged canonical atlas (xz max level)
  iamatlas_class_archives/
    stem_pluri_v0_1_REBUILD.tar.xz   — 3 files: per_celltype, brightness, result.json
    stem_adult_v0_1_REBUILD.tar.xz       (one per class, 8 total)
    progenitor_v0_1_REBUILD.tar.xz
    stromal_v0_1_REBUILD.tar.xz
    cycling_v0_1_REBUILD.tar.xz
    secretory_v0_1_REBUILD.tar.xz
    immune_v0_1_REBUILD.tar.xz
    terminal_v0_1_REBUILD.tar.xz

The per-class tarballs use the RECONCILED per_celltype (so re-runs start
from the corrected version). The original raw per_celltype is also included
in the tarball under a /raw/ subfolder for full provenance.

Total size on disk: typically 30-60 MB for the .csv.xz atlas + 60-120 MB total
for the 8 class archives, depending on compression.

Run AFTER merge_iamatlas_v0_1_REBUILD.py.
"""

import argparse, lzma, shutil, tarfile, hashlib
from pathlib import Path

CLASSES = ["stem_pluri", "stem_adult", "progenitor", "stromal",
           "cycling", "secretory", "immune", "terminal"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_atlas(in_path, out_path):
    print(f"Compacting {in_path} -> {out_path} (this takes a minute)...")
    with open(in_path, "rb") as src, lzma.open(out_path, "wb", preset=9 | lzma.PRESET_EXTREME) as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    src_size = in_path.stat().st_size / 1024 / 1024
    dst_size = out_path.stat().st_size / 1024 / 1024
    sha = sha256_of(out_path)
    print(f"  {src_size:.1f} MB -> {dst_size:.1f} MB ({100*dst_size/src_size:.1f}% of original)")
    print(f"  SHA-256: {sha}")
    return {"in_mb": src_size, "out_mb": dst_size, "sha256": sha}


def compact_class(in_dir, cls, out_dir):
    """Tar+xz a class's three files (RECONCILED per_celltype + brightness + result.json)
    plus the raw per_celltype under /raw/ for full provenance."""
    pct_recon = in_dir / f"iamatlas_v0_1_{cls}_per_celltype.RECONCILED.csv"
    pct_raw   = in_dir / f"iamatlas_v0_1_{cls}_per_celltype.csv"
    bright    = in_dir / f"iamatlas_v0_1_{cls}_brightness.csv"
    result    = in_dir / f"iamatlas_v0_1_{cls}_result.json"

    out_path = out_dir / f"{cls}_v0_1_REBUILD.tar.xz"
    with tarfile.open(out_path, "w:xz", preset=9 | lzma.PRESET_EXTREME) as tar:
        # Canonical (RECONCILED if exists, else raw)
        canonical = pct_recon if pct_recon.exists() else pct_raw
        tar.add(canonical, arcname=f"{cls}/iamatlas_v0_1_{cls}_per_celltype.csv")
        tar.add(bright, arcname=f"{cls}/iamatlas_v0_1_{cls}_brightness.csv")
        tar.add(result, arcname=f"{cls}/iamatlas_v0_1_{cls}_result.json")
        # Raw (always include for provenance)
        if pct_raw.exists() and pct_raw != canonical:
            tar.add(pct_raw, arcname=f"{cls}/raw/iamatlas_v0_1_{cls}_per_celltype.csv")
    sz = out_path.stat().st_size / 1024 / 1024
    sha = sha256_of(out_path)
    print(f"  {cls:12s}  {sz:6.1f} MB  sha256={sha[:16]}...")
    return {"path": str(out_path), "size_mb": round(sz, 2), "sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="iamatlas_v0_1_output_REBUILD")
    ap.add_argument("--atlas_csv", default="IAMAtlasREBUILD.csv")
    ap.add_argument("--out_atlas", default="IAMAtlasREBUILD.csv.xz")
    ap.add_argument("--class_archive_dir", default="iamatlas_class_archives")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    atlas_csv = Path(args.atlas_csv)
    out_atlas = Path(args.out_atlas)
    class_dir = Path(args.class_archive_dir)
    class_dir.mkdir(exist_ok=True)

    if not atlas_csv.exists():
        print(f"ERROR: atlas CSV not found: {atlas_csv}")
        print("Run merge_iamatlas_v0_1_REBUILD.py first.")
        return

    # Compact the main atlas
    print("=" * 60)
    print("Compacting main atlas to .csv.xz")
    print("=" * 60)
    atlas_stats = compact_atlas(atlas_csv, out_atlas)

    # Compact each class
    print()
    print("=" * 60)
    print("Compacting per-class archives (8 classes, 3-4 files each)")
    print("=" * 60)
    class_stats = {}
    for cls in CLASSES:
        class_stats[cls] = compact_class(in_dir, cls, class_dir)

    print()
    print("=" * 60)
    print("Compaction complete.")
    print("=" * 60)
    print(f"Atlas: {out_atlas} ({atlas_stats['out_mb']:.1f} MB)")
    print(f"Class archives: {class_dir}/  ({sum(c['size_mb'] for c in class_stats.values()):.1f} MB total)")
    print()
    print("Next: upload IAMAtlasREBUILD.csv.xz and the class archives to chat,")
    print("then provide a GitHub token to push to the repo.")


if __name__ == "__main__":
    main()
