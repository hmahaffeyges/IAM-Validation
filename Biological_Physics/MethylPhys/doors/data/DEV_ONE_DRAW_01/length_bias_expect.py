"""The held error the qualifying rule leaves at each molecule length, derived exactly (stage_q2_selfref.qual_iso_expect), per opportunity, at
a per-call loss of 0.035 (synthetic tests cfDNA 1 and 3). Usage: python3 length_bias_expect.py OUT.txt"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "chain")); import stage_q2_selfref as Q2
E = 0.035; L = [f"per-call loss {E}; with no qualifying rule {E * (1 - E) ** 2:.5f} per opportunity"]
for k in (6, 8, 12, 20): L.append(f"{k} calls: expected held error {Q2.qual_iso_expect(k, E) / (k - 2):.5f} per opportunity")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
