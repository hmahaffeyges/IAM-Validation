"""Group-A on sequencing (checklist B9), scored by the chain's own Group-A (stage_groupa.py collect and compare), written before any read.
Bar 1 (spent healthy pass 5 bundles, sorted-cell Met-A): the sealed donor halves A and B read as no different: |difference| under 2 % and
under 2 standard errors. Bar 2 (the same 137 read again by the one command, half with the planted 1 % loss, groupa_seq_plant.py): planted
above unplanted by more than 2 standard errors, the unplanted the tare's references. Recorded, not scored: the same on whole-sample Met-A
and the islands; the smallest difference each design could have found.
Usage: python3 groupa_seq_score.py PASS5_BUNDLES PLANT_BUNDLES OUT.txt"""
import os, sys, subprocess, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(H, "..", "..", "..", "..", "..", "chain", "stage_groupa.py"); W = os.path.dirname(os.path.abspath(sys.argv[3]))
S = pd.read_csv(os.path.join(H, "groupa_seq_samples.csv")); S["half"] = S.half.astype(str).str.strip(); L = []
def run(*a): return subprocess.run([sys.executable, G, *a], capture_output=True, text=True).stdout.strip()
for bundles, col, A, B, tag in ((sys.argv[1], "half", "A", "B", "bar 1"), (sys.argv[2], "planted", "1", "0", "bar 2")):
    s = S.copy(); s["healthy_reference"] = True if col == "half" else s.planted.eq(0)
    f = os.path.join(W, f"groupa_{col}.csv"); s.to_csv(f, index=False)
    for rd in ("sorted", "whole", "islands"):
        o = os.path.join(W, f"groupa_{col}_{rd}.csv"); run("collect", bundles, f, o, rd); out = run("compare", o, col, A, B)
        L.append(f"== {tag} ({col} {A} against {B}), {rd}" + (" (scored)" if rd == "sorted" else " (recorded)")); L.append(out)
def val(block, key):
    import re
    m = re.search(r"difference ([+-]?[0-9.]+) %, ([+-]?[0-9.]+) standard errors", block); return (float(m.group(1)), float(m.group(2))) if m else (None, None)
b1 = val(L[1], ""); b2 = val(L[7], "")
ok1 = b1[0] is not None and abs(b1[0]) < 2 and abs(b1[1]) < 2; ok2 = b2[1] is not None and b2[1] > 2
L += [f"bar 1 halves no different (|diff| < 2 %, |t| < 2): {b1}: {'met' if ok1 else 'not met'}", f"bar 2 planted found (t > 2): {b2}: {'met' if ok2 else 'not met'}",
      f"Group-A on sequencing: {'met' if ok1 and ok2 else 'not met'}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
