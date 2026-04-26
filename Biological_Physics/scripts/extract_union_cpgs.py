"""
Streaming extractor — pulls the union of UniLIFE + EpiSCORE + Loyfer + Salas
target CpGs from the full GSE51057 + GSE51032 series matrix .gz files in
single passes. Output: cohort_betas_union.csv files reusable across VALs.
"""

import gzip
import os
import sys
import time
from pathlib import Path


def extract_target_cpgs(gz_path, target_cpgs, out_csv, log_every=50000):
    """Stream a GEO series matrix .txt.gz file, write only rows whose ID_REF
    is in target_cpgs to out_csv. Header (sample IDs) preserved verbatim."""
    print(f"\n>>> {gz_path}")
    print(f"    Target: {len(target_cpgs)} CpGs")
    print(f"    Output: {out_csv}")

    start = time.time()
    n_lines = 0
    n_kept = 0
    in_data = False
    header_written = False

    with gzip.open(gz_path, 'rt') as fin, open(out_csv, 'w') as fout:
        for line in fin:
            n_lines += 1
            if n_lines % log_every == 0:
                elapsed = time.time() - start
                print(f"    line {n_lines:,}  kept {n_kept:,}  ({elapsed:.0f}s)")

            line = line.rstrip("\n")

            if not in_data:
                if line.startswith("!series_matrix_table_begin"):
                    in_data = True
                    continue
                continue

            # In data section
            if line.startswith("!series_matrix_table_end"):
                break

            # First line in data section is the header (ID_REF + GSM IDs)
            if not header_written:
                # Header: starts with "ID_REF" — rewrite as 'CpG_ID' for clarity
                # The original is tab-separated with quoted strings
                fout.write(line.replace('"ID_REF"', '"CpG_ID"').replace('\t', ',') + "\n")
                header_written = True
                continue

            # Data row: first field is the CpG ID (quoted)
            tab_idx = line.find("\t")
            if tab_idx < 0:
                continue
            cpg_id = line[:tab_idx].strip().strip('"')
            if cpg_id in target_cpgs:
                fout.write(line.replace('\t', ',') + "\n")
                n_kept += 1

    elapsed = time.time() - start
    print(f"    DONE: {n_lines:,} total lines, {n_kept:,} CpGs kept ({elapsed:.0f}s)")
    return n_kept


def main():
    # Load target CpGs
    target_path = "/home/claude/run_everything/extract_cpgs_target.txt"
    target = set()
    with open(target_path) as f:
        for line in f:
            cpg = line.strip()
            if cpg:
                target.add(cpg)
    print(f"Loaded {len(target):,} target CpGs from {target_path}")

    out_57 = "/home/claude/run_everything/GSE51057_betas_union.csv"
    out_32 = "/home/claude/run_everything/GSE51032_betas_union.csv"

    n57 = extract_target_cpgs("/home/claude/GSE51057_series_matrix.txt.gz",
                               target, out_57)
    n32 = extract_target_cpgs("/home/claude/GSE51032_series_matrix.txt.gz",
                               target, out_32)

    print(f"\n=== Summary ===")
    print(f"GSE51057: {n57} / {len(target)} CpGs ({100*n57/len(target):.1f}% of union)")
    print(f"GSE51032: {n32} / {len(target)} CpGs ({100*n32/len(target):.1f}% of union)")
    for f in [out_57, out_32]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            print(f"  {f}: {sz:,} bytes ({sz/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
