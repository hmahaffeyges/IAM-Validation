"""The IAM Sequencing Atlas's Zenodo record (DOI 10.5281/zenodo.23934666), summarized from the record's own SHA-256 list
(deposit/zenodo_seq_atlas_SHA256.txt, written into the record by deposit/zenodo_seq_atlas_upload.py). Usage: python3 zenodo_deposit_summary.py OUT.txt"""
import os, sys
H = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(H, "..", "..", "..", "..", "deposit", "zenodo_seq_atlas_SHA256.txt")
rows = [l.rstrip("\n").split("  ") for l in open(M, encoding="utf-8") if l.strip()]
top = [(n, int(b)) for _, n, b in rows if "/" not in n]; pieces = [(n, int(b)) for _, n, b in rows if "/" in n]
L = ["IAM Sequencing Atlas on Zenodo, DOI 10.5281/zenodo.23934666",
     f"files in the record: {len(top) + 1} (the {len(top)} listed below and the SHA-256 manifest itself); {sum(b for _, b in top) / 1e9:.2f} GB",
     f"merged 57-cell pieces: {len(pieces)}, {sum(b for _, b in pieces) / 1e9:.2f} GB, in {sum(1 for n, _ in top if n.startswith('pieces57_set'))} sets"] + [f"  {n}: {b:,} bytes" for n, b in top]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L[:3]))
