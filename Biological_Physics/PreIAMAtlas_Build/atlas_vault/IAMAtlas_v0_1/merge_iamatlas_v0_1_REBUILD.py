#!/usr/bin/env python3
"""
merge_iamatlas_v0_1_REBUILD.py
================================
Patched version of merge_iamatlas_v0_1.py that reads the RECONCILED per_celltype
files (post-reconcile_duplicates.py) and produces:

  IAMAtlasREBUILD.csv                 — canonical matrix (same format as v0.1)
  IAMAtlasREBUILD_celltype_to_class.json
  IAMAtlasREBUILD_provenance.json     — rebuild metadata for future reference

Matrix format (rows = CpGs, columns):
  cpg_id,
  ──── architecture-class brightness (8 classes × 4 = 32 columns) ────
    <class>_mean, <class>_sd, <class>_ci_lo, <class>_ci_hi  (for each class)
  ──── per-cell-type posterior means/sds (variable, from RECONCILED files) ────
    <celltype>_mean, <celltype>_sd
  ──── meta ────
  n_classes_with_data

Differences from original merge:
  - Reads .RECONCILED.csv inputs (not the raw per_celltype.csv)
  - Builds CpG universe from union of all class CpGs (no separate file needed)
  - Writes provenance JSON
"""

import argparse, csv, json
from pathlib import Path
from datetime import datetime

CLASSES = ["stem_pluri", "stem_adult", "progenitor", "stromal",
           "cycling", "secretory", "immune", "terminal"]


def load_class_brightness(path):
    """Load class-level brightness CSV. Returns: {cpg_id: (mean, sd, ci_lo, ci_hi)}"""
    if not path.exists():
        return {}
    d = {}
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            d[row["cpg_id"]] = (row["mean"], row["sd"], row["ci_lo"], row["ci_hi"])
    return d


def load_per_celltype(path):
    """Load per-cell-type CSV. Returns: (celltype_list, {cpg_id: {celltype: (mean, sd)}})"""
    if not path.exists():
        return [], {}
    with open(path) as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        celltypes = []
        for c in fieldnames:
            if c.endswith("_mean") and c != "cpg_id":
                celltypes.append(c[:-len("_mean")])
        d = {}
        for row in reader:
            cpg = row["cpg_id"]
            cell_data = {}
            for ct in celltypes:
                m = row.get(f"{ct}_mean")
                s = row.get(f"{ct}_sd")
                if m not in (None, "", "NA"):
                    try:
                        cell_data[ct] = (float(m), float(s) if s not in (None, "", "NA") else 0.0)
                    except ValueError:
                        pass
            if cell_data:
                d[cpg] = cell_data
    return celltypes, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="iamatlas_v0_1_output_REBUILD")
    ap.add_argument("--output", default="IAMAtlasREBUILD.csv")
    ap.add_argument("--map_output", default="IAMAtlasREBUILD_celltype_to_class.json")
    ap.add_argument("--provenance_output", default="IAMAtlasREBUILD_provenance.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.output)

    # Build CpG universe from union of all class brightness files
    universe_set = set()
    class_brightness = {}
    for cls in CLASSES:
        f = in_dir / f"iamatlas_v0_1_{cls}_brightness.csv"
        cb = load_class_brightness(f)
        class_brightness[cls] = cb
        universe_set.update(cb.keys())
        if cb:
            print(f"  class brightness {cls}: {len(cb)} CpGs")
        else:
            print(f"  [WARN] class brightness missing: {f}")
    universe = sorted(universe_set)
    print(f"\nUniverse (union of class CpGs): {len(universe):,} CpGs")

    # Load per-cell-type from RECONCILED files
    celltype_data = {}        # cpg -> {celltype: (mean, sd)}
    celltype_to_class = {}    # celltype -> class
    all_celltypes_ordered = []

    for cls in CLASSES:
        f_ct = in_dir / f"iamatlas_v0_1_{cls}_per_celltype.RECONCILED.csv"
        celltypes, ct_data = load_per_celltype(f_ct)
        if celltypes:
            print(f"  per-cell-type {cls}: {len(celltypes)} cells, {len(ct_data):,} CpGs (reconciled)")
            for ct in celltypes:
                if ct not in celltype_to_class:
                    celltype_to_class[ct] = cls
                    all_celltypes_ordered.append(ct)
            for cpg, cd in ct_data.items():
                celltype_data.setdefault(cpg, {}).update(cd)
        else:
            print(f"  [WARN] per-celltype RECONCILED file missing for {cls}: {f_ct}")

    print(f"\nTotal cell types across all classes: {len(all_celltypes_ordered)}")

    # Build header
    fieldnames = ["cpg_id"]
    for cls in CLASSES:
        fieldnames += [f"{cls}_mean", f"{cls}_sd", f"{cls}_ci_lo", f"{cls}_ci_hi"]
    for ct in all_celltypes_ordered:
        fieldnames += [f"{ct}_mean", f"{ct}_sd"]
    fieldnames.append("n_classes_with_data")

    # Write merged matrix
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for cpg in universe:
            row = [cpg]
            n_present = 0
            for cls in CLASSES:
                if cpg in class_brightness[cls]:
                    mean, sd, lo, hi = class_brightness[cls][cpg]
                    row += [mean, sd, lo, hi]
                    n_present += 1
                else:
                    row += ["NA", "NA", "NA", "NA"]
            ct_data = celltype_data.get(cpg, {})
            for ct in all_celltypes_ordered:
                if ct in ct_data:
                    m, s = ct_data[ct]
                    row += [f"{m:.6f}", f"{s:.6f}"]
                else:
                    row += ["NA", "NA"]
            row.append(str(n_present))
            w.writerow(row)

    # Write celltype -> class mapping
    with open(args.map_output, "w") as f:
        json.dump(celltype_to_class, f, indent=2, sort_keys=True)

    # Write provenance
    import os
    provenance = {
        "atlas_version": "IAMAtlasREBUILD",
        "build_date": datetime.now().isoformat(),
        "predecessor": "IAMAtlas.csv.xz (collapsed; flatness bug)",
        "build_pipeline": [
            "iamatlas_v0_1_mcmc_batched_FIXED.py (per-class MCMC, batch_size=5000 except immune=1500)",
            "reconcile_duplicates.py (immune a-prefix merge, lowercase thin-duplicate drop)",
            "merge_iamatlas_v0_1_REBUILD.py (this step)",
        ],
        "classes": list(CLASSES),
        "n_classes": len(CLASSES),
        "n_cell_types_total": len(all_celltypes_ordered),
        "n_cpgs": len(universe),
        "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 1),
        "h_min_values_frozen_2026_04_06": {
            "terminal": 0.7728,
            "immune": 0.838889,
            "secretory": 0.8433,
            "progenitor": 0.8522,
            "cycling": 0.8561,
            "stromal": 0.8630,
            "stem_adult": 0.8737,
            "stem_pluri": 0.9822,
        },
        "distinctness_test_passed": {
            "stem_pluri":  "single-cell, signal/noise ~41x",
            "stem_adult":  "single-cell, signal/noise ~41x",
            "progenitor":  "11 cells DISTINCT, median pairwise |d|=0.043",
            "cycling":     "20 cells DISTINCT, median pairwise |d|=0.10 (after drop)",
            "secretory":   "20 cells DISTINCT, median pairwise |d|=0.115 (after drop)",
            "terminal":    "12 cells DISTINCT, median pairwise |d|=0.31 (after drop)",
            "stromal":     "FIXED-script original run, already good (5 cells)",
            "immune":      "63 cells DISTINCT post-reconciliation (12 a-prefix duplicates merged)",
        },
        "merge_inputs": str(in_dir),
        "celltype_to_class_file": args.map_output,
        "lessons_doc": "IAMAtlas_FLATNESS_LESSON.md",
        "reconciliation_report": "RECONCILIATION_REPORT.json (in --in_dir)",
    }
    with open(args.provenance_output, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nMerged:        {out_path}")
    print(f"Cell map:      {args.map_output}")
    print(f"Provenance:    {args.provenance_output}")
    print(f"\n  Size: {provenance['size_mb']} MB (uncompressed)")
    print(f"  Rows: {len(universe):,} CpGs")
    print(f"  Class-level columns:     32 (8 classes × 4)")
    print(f"  Per-cell-type columns:   {len(all_celltypes_ordered) * 2} ({len(all_celltypes_ordered)} cells × 2)")
    print(f"  Total columns:           {len(fieldnames)}")
    print(f"\nNext: python compact_atlas.py")


if __name__ == "__main__":
    main()
