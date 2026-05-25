#!/usr/bin/env python3
"""
IAMAtlas — Merge per-class outputs into canonical IAMAtlas matrix
==================================================================

Input:  8 per-class brightness CSVs + 8 per-cell-type CSVs from the
        production run, in iamatlas_v0_1_output/

Output:
  IAMAtlas.csv                       — the canonical matrix
  IAMAtlas_celltype_to_class.json    — celltype → architecture-class mapping

Matrix format (rows = CpGs, columns):
  cpg_id,
  ──── architecture-class brightness (8 × 4 = 32 columns) ────
  stem_pluri_mean, stem_pluri_sd, stem_pluri_ci_lo, stem_pluri_ci_hi,
  stem_adult_mean, ... ,  immune_..., terminal_...,
  ──── per-cell-type brightness (≥80 cell types × 2 = ≥160 columns) ────
  CD4_T-cells_mean, CD4_T-cells_sd,
  CD8_T-cells_mean, CD8_T-cells_sd,
  Hepatocytes_mean, Hepatocytes_sd,
  ... etc for every cell type that had MCMC posterior data ...
  ──── meta ────
  n_classes_with_data

Date: 2026-05-04
"""

import argparse
import csv
import json
from pathlib import Path


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
    """
    Load per-cell-type CSV from one class.
    
    Format from MCMC output:
      cpg_id, <ct1>_mean, <ct2>_mean, ..., <ct1>_sd, <ct2>_sd, ...
    
    Returns: (celltype_list, {cpg_id: {celltype: (mean, sd)}})
    """
    if not path.exists():
        return [], {}
    
    with open(path) as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        # Detect cell types: columns ending in _mean (excluding cpg_id)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="iamatlas_v0_1_output",
                        help="Directory with per-class outputs")
    parser.add_argument("--universe", default="iamatlas_cpg_universe.csv",
                        help="CpG universe (one column: cpg_id)")
    parser.add_argument("--output", default="IAMAtlas.csv",
                        help="Output canonical matrix")
    parser.add_argument("--map_output", default="IAMAtlas_celltype_to_class.json",
                        help="Output cell-type → class mapping")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.output)

    # Load universe
    universe = []
    with open(args.universe) as f:
        next(f)
        for line in f:
            cpg = line.strip()
            if cpg.startswith("cg"):
                universe.append(cpg)
    print(f"Universe: {len(universe)} CpGs")

    # Load each class — both class-level and per-cell-type
    class_brightness = {}
    celltype_brightness = {}        # cpg → {celltype: (mean, sd)}
    celltype_to_class = {}          # celltype → class
    all_celltypes_ordered = []
    
    for cls in CLASSES:
        # Class-level
        f_cls = in_dir / f"iamatlas_v0_1_{cls}_brightness.csv"
        cb = load_class_brightness(f_cls)
        class_brightness[cls] = cb
        if cb:
            print(f"  Class brightness {cls}: {len(cb)} CpGs")
        else:
            print(f"  [WARN] class brightness missing for {cls}: {f_cls}")
        
        # Per-cell-type
        f_ct = in_dir / f"iamatlas_v0_1_{cls}_per_celltype.csv"
        celltypes, ct_data = load_per_celltype(f_ct)
        if celltypes:
            print(f"    Per-cell-type {cls}: {len(celltypes)} cell types, {len(ct_data)} CpGs")
            for ct in celltypes:
                if ct not in celltype_to_class:
                    celltype_to_class[ct] = cls
                    all_celltypes_ordered.append(ct)
            for cpg, cd in ct_data.items():
                celltype_brightness.setdefault(cpg, {}).update(cd)

    print(f"\nTotal cell types across all classes: {len(all_celltypes_ordered)}")

    # Build header
    fieldnames = ["cpg_id"]
    # Class-level brightness columns
    for cls in CLASSES:
        fieldnames += [f"{cls}_mean", f"{cls}_sd", f"{cls}_ci_lo", f"{cls}_ci_hi"]
    # Per-cell-type brightness columns
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
            # Class-level
            for cls in CLASSES:
                if cpg in class_brightness[cls]:
                    mean, sd, lo, hi = class_brightness[cls][cpg]
                    row += [mean, sd, lo, hi]
                    n_present += 1
                else:
                    row += ["NA", "NA", "NA", "NA"]
            # Per-cell-type
            ct_data = celltype_brightness.get(cpg, {})
            for ct in all_celltypes_ordered:
                if ct in ct_data:
                    m, s = ct_data[ct]
                    row += [f"{m:.6f}", f"{s:.6f}"]
                else:
                    row += ["NA", "NA"]
            row.append(str(n_present))
            w.writerow(row)

    # Write celltype → class mapping
    with open(args.map_output, "w") as f:
        json.dump(celltype_to_class, f, indent=2, sort_keys=True)

    print(f"\nMerged: {out_path}")
    print(f"Cell-type → class map: {args.map_output}")
    
    # Summary
    import os
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  Size: {size_mb:.1f} MB")
    print(f"  Rows: {len(universe)} CpGs")
    print(f"  Class-level columns: {8 * 4} = 32")
    print(f"  Per-cell-type columns: {len(all_celltypes_ordered) * 2}")
    print(f"  Total columns: {len(fieldnames)}")


if __name__ == "__main__":
    main()
