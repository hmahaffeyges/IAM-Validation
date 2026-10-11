#!/usr/bin/env python3
"""Pre-run read-length check (LESSONS B1, 2026-10-10). bwa mem keeps an alignment only if its score >= -T; with bwa-meth's scoring a read
scores at most its length. A run is refused before alignment when fewer than MIN_SHARE of the first N reads of a (trimmed) fastq are at least
T + MARGIN bases long. Usage: preflight_readlen.py FASTQ[.gz] T [N=200000] [MIN_SHARE=0.8] [MARGIN=2]   exit 0 = go, 3 = refuse"""
import sys, gzip
f, T = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8; MARGIN = int(sys.argv[5]) if len(sys.argv) > 5 else 2
op = gzip.open if f.endswith(".gz") else open; n = ok = 0; L = []
with op(f, "rt") as h:
    for i, l in enumerate(h):
        if i % 4 == 1:
            k = len(l.rstrip("\n")); n += 1; ok += k >= T + MARGIN; L.append(k)
            if n >= N: break
share = ok / max(n, 1); L.sort(); med = L[len(L) // 2] if L else 0
print(f"preflight: {n} reads, median length {med}, share >= {T + MARGIN} bases {share:.4f} (need {MIN})")
sys.exit(0 if share >= MIN else 3)
