"""
harmonize_stromal_labels.py
============================

Stromal class label harmonization for IAMAtlas v0.1 production MCMC re-run.

Purpose: collapse atlas-specific naming variants of the same biological cell type
into canonical labels, so the model can pool information across atlases instead of
fitting near-duplicate parameters that compete for the same signal.

The original stromal run failed (R-hat 3.67, ESS 4) because 17 cell-type labels
were spread across 17 atlases with most cell types supported by only a single
atlas. Many of those 17 labels are different names for the same biological cell
type (EC, Endo, Vascular_endothelial_cells, endothelial all = endothelial cells).

This script:
  1. Reads the production MCMC inputs file
  2. For arch_class == 'stromal' rows ONLY, applies the canonical label mapping
  3. All other arch_class rows pass through unchanged
  4. Writes a new harmonized inputs file
  5. Prints before/after summary so you can verify the mapping

Usage:
    python harmonize_stromal_labels.py [input_csv] [output_csv]

Defaults:
    input_csv  = iamatlas_mcmc_inputs.csv
    output_csv = iamatlas_mcmc_inputs_stromal_harmonized.csv

The original input file is NEVER modified.
"""

import csv
import sys
from collections import defaultdict, Counter

# ------------------------------------------------------------------
# CANONICAL LABEL MAPPING
# ------------------------------------------------------------------
# Each (original_label, atlas_source) -> canonical_label
# Only stromal-class rows are mapped. Mapping is intentionally conservative:
# only consolidates labels that are unambiguously the same biological cell type.
# Single-atlas, single-source labels (Peri, Stellate, Astro, placenta) keep their
# canonical names so they remain identifiable in the output.

STROMAL_LABEL_MAP = {
    # Endothelial cells: 13 sources collapse to one canonical label
    'EC': 'endothelial',
    'Endo': 'endothelial',
    'Vascular_endothelial_cells': 'endothelial',
    'endothelial': 'endothelial',

    # Fibroblasts: 11 sources collapse to one canonical label
    'FB': 'fibroblast',
    'Fib': 'fibroblast',
    'fibroblast': 'fibroblast',

    # Adipocytes: 4 sources collapse to one canonical label
    'Adipocytes': 'adipocyte',
    'Fat': 'adipocyte',
    'adipose': 'adipocyte',

    # Smooth muscle: 2 sources collapse to one canonical label
    'SM': 'smooth_muscle',
    'SMC': 'smooth_muscle',

    # Single-source cell types: keep canonical names but standardize casing
    'Peri': 'pericyte',
    'Stellate': 'stellate',
    'Astro': 'astrocyte',
    'placenta': 'placenta',

    # Catch-all "Stromal" buckets from atlases that didn't subclassify further
    'Stromal': 'stromal_other',
}


def harmonize(input_path: str, output_path: str) -> dict:
    """
    Read input CSV, apply stromal label mapping, write output CSV.
    Returns a summary dict for reporting.
    """
    # Stats for before/after reporting
    stats = {
        'total_rows': 0,
        'stromal_rows': 0,
        'mapped_rows': 0,
        'unmapped_stromal_labels': Counter(),
        'before': defaultdict(set),  # canonical -> set of (orig_label, atlas)
        'after': defaultdict(set),   # canonical -> set of atlases
    }

    with open(input_path, 'r', newline='') as fin, \
         open(output_path, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"Could not read header from {input_path}")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            stats['total_rows'] += 1

            if row['arch_class'] == 'stromal':
                stats['stromal_rows'] += 1
                orig_label = row['cell_type']
                atlas = row['atlas_source']

                if orig_label in STROMAL_LABEL_MAP:
                    canonical = STROMAL_LABEL_MAP[orig_label]
                    stats['before'][canonical].add((orig_label, atlas))
                    stats['after'][canonical].add(atlas)
                    if canonical != orig_label:
                        stats['mapped_rows'] += 1
                    row['cell_type'] = canonical
                else:
                    # Unmapped stromal label - this is a sign the mapping needs
                    # an entry added. Log it but pass through unchanged.
                    stats['unmapped_stromal_labels'][orig_label] += 1
                    stats['before'][orig_label].add((orig_label, atlas))
                    stats['after'][orig_label].add(atlas)

            writer.writerow(row)

    return stats


def print_summary(stats: dict, input_path: str, output_path: str) -> None:
    print()
    print("=" * 70)
    print("STROMAL LABEL HARMONIZATION SUMMARY")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total rows processed: {stats['total_rows']:,}")
    print(f"Stromal-class rows:   {stats['stromal_rows']:,}")
    print(f"Stromal rows mapped:  {stats['mapped_rows']:,}")
    print()

    if stats['unmapped_stromal_labels']:
        print("WARNING: stromal labels with no canonical mapping (passed through unchanged):")
        for label, count in stats['unmapped_stromal_labels'].most_common():
            print(f"  {label}: {count:,} rows")
        print("  --> Add these to STROMAL_LABEL_MAP and re-run if needed.")
        print()

    print("AFTER HARMONIZATION — atlas support per canonical cell type:")
    print(f"{'canonical':<20} {'n_atlases':>10}  source_atlases")
    print("-" * 70)
    for canonical in sorted(stats['after'].keys()):
        atlases = sorted(stats['after'][canonical])
        print(f"{canonical:<20} {len(atlases):>10}  {', '.join(atlases)}")
    print()

    # Sanity: compare to original 17-label structure
    print("BEFORE -> AFTER consolidation:")
    consolidations = []
    for canonical in sorted(stats['before'].keys()):
        orig_pairs = stats['before'][canonical]
        orig_labels = sorted({pair[0] for pair in orig_pairs})
        if len(orig_labels) > 1:
            consolidations.append((canonical, orig_labels, len(orig_pairs)))
    if consolidations:
        for canonical, orig_labels, n in consolidations:
            joined = ' + '.join(orig_labels)
            print(f"  {joined}  ->  {canonical}  ({n} atlas-source rows merged)")
    else:
        print("  (no consolidations applied — check mapping)")
    print()
    print("Harmonization complete.")
    print("=" * 70)


def main() -> int:
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'iamatlas_mcmc_inputs.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'iamatlas_mcmc_inputs_stromal_harmonized.csv'

    stats = harmonize(input_path, output_path)
    print_summary(stats, input_path, output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
