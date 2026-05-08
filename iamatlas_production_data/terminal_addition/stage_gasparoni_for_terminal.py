#!/usr/bin/env python3
"""
Stage Gasparoni 2018 (GSE66351) brain methylation atlas for terminal-class
addition to iamatlas_mcmc_inputs.csv.

Source: gasparoni_2018_GSE66351_brain_celltype_atlas.csv
        Schema: cpg_id, cell_type, beta_mean, beta_sd, n_donors, atlas_source,
                platform, reference, region, condition

Target: iamatlas_mcmc_inputs.csv production schema
        Schema: cpg_id, atlas_source, cell_type, arch_class, beta_observed,
                n_donors, weight

Cell-type label mapping (Heath's decision, 2026-05-06):
  - 'cortical_neuron' (Gasparoni)  ->  'Cortical_neurons'  (matches Loyfer label)
  - 'cortical_glia' (Gasparoni)    ->  'Glia'              (new terminal cell type)

Atlas source: 'gasparoni_2018' (internal provenance, not customer-facing)

Architecture class: terminal (all rows)

Beta clamping: matches loader behavior in iamatlas_v0_1_mcmc_batched.py
  beta <= 0  ->  1e-4
  beta >= 1  ->  1 - 1e-4

Weight: 1.0 (matches default behavior of every other atlas in the matrix)

Output: gasparoni_terminal_addition.csv
        Ready to append to iamatlas_mcmc_inputs.csv when main MCMC run finishes
        and terminal class is ready to re-run against expanded pool.
"""
import csv
import os
import sys
from pathlib import Path

SOURCE_CSV = Path("/home/claude/atlases_pollard_track/gasparoni_2018_GSE66351_brain_celltype_atlas.csv")
OUTPUT_CSV = Path("/home/claude/iamatlas_brightness_pilot/gasparoni_terminal_addition.csv")

CELL_TYPE_MAP = {
    "cortical_neuron": "Cortical_neurons",
    "cortical_glia": "Glia",
}

ARCH_CLASS = "terminal"
ATLAS_SOURCE = "gasparoni_2018"
WEIGHT = 1.0

PRODUCTION_HEADER = ["cpg_id", "atlas_source", "cell_type", "arch_class",
                     "beta_observed", "n_donors", "weight"]


def main():
    if not SOURCE_CSV.exists():
        print(f"ERROR: source file not found: {SOURCE_CSV}")
        sys.exit(1)

    n_in = 0
    n_out = 0
    n_clamped_low = 0
    n_clamped_high = 0
    n_skipped_label = 0
    cell_type_counts = {"Cortical_neurons": 0, "Glia": 0}

    with open(SOURCE_CSV, newline="") as fin, \
         open(OUTPUT_CSV, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(PRODUCTION_HEADER)

        for row in reader:
            n_in += 1

            src_ct = row["cell_type"].strip()
            if src_ct not in CELL_TYPE_MAP:
                n_skipped_label += 1
                continue
            target_ct = CELL_TYPE_MAP[src_ct]

            try:
                beta = float(row["beta_mean"])
            except (ValueError, KeyError):
                continue

            if beta <= 0:
                beta = 1e-4
                n_clamped_low += 1
            elif beta >= 1:
                beta = 1 - 1e-4
                n_clamped_high += 1

            try:
                n_donors = int(row.get("n_donors") or 1)
            except ValueError:
                n_donors = 1

            cpg = row["cpg_id"].strip()

            writer.writerow([
                cpg,
                ATLAS_SOURCE,
                target_ct,
                ARCH_CLASS,
                f"{beta:.6f}",
                n_donors,
                f"{WEIGHT:.4f}",
            ])
            n_out += 1
            cell_type_counts[target_ct] += 1

    out_size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)

    print(f"=== Gasparoni terminal-class staging complete ===")
    print(f"Source rows read:      {n_in:,}")
    print(f"Skipped (unknown label): {n_skipped_label:,}")
    print(f"Output rows written:   {n_out:,}")
    print(f"  Cortical_neurons:    {cell_type_counts['Cortical_neurons']:,}")
    print(f"  Glia:                {cell_type_counts['Glia']:,}")
    print(f"Beta clamped low (<=0):  {n_clamped_low:,}")
    print(f"Beta clamped high (>=1): {n_clamped_high:,}")
    print(f"Output file:          {OUTPUT_CSV}")
    print(f"Output size:          {out_size_mb:.1f} MB")
    print()
    print(f"Next step (after main MCMC run finishes):")
    print(f"  1. Backup production matrix:")
    print(f"     cp iamatlas_mcmc_inputs.csv iamatlas_mcmc_inputs.csv.backup")
    print(f"  2. Append (skip header from staging file):")
    print(f"     tail -n +2 gasparoni_terminal_addition.csv >> iamatlas_mcmc_inputs.csv")
    print(f"  3. Verify row count increased by ~957K:")
    print(f"     wc -l iamatlas_mcmc_inputs.csv")
    print(f"  4. Re-run terminal class only:")
    print(f"     python iamatlas_v0_1_mcmc_batched.py --classes terminal \\")
    print(f"            --batch_size 5000 --chains 4 --tune 1000 --draws 1000 \\")
    print(f"            --target_accept 0.95 --out_dir iamatlas_v0_1_output")


if __name__ == "__main__":
    main()
