"""The 57-cell IAM Sequencing Atlas's tables, summarized from the chain's own files (Runtime Matrices/Seq_Atlas). Usage: python3 atlas57_summary.py OUT.txt"""
import os, sys, json, pandas as pd
RM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain", "Runtime Matrices", "Seq_Atlas"))
F = pd.read_parquet(os.path.join(RM, "input_floors_v1.parquet")); H = json.load(open(os.path.join(RM, "held_reference_v1.json")))
D = json.load(open(os.path.join(RM, "deconv_reference_v1.json"))); S = json.load(open(os.path.join(RM, "floor_sites_v1.json"))); M = json.load(open(os.path.join(RM, "metaa_seq_reference_v1.json")))
L = [f"Deconvolver reference: {D['cells']} cells, {D['markers']} markers", f"floor sites: {S['sites']} ({S['cells']} cells)", f"Met-A reference: {M['cells']} cells, granulocytes {M['per_cell'].get('granulocytes')} sites",
     f"input floors: {len(F)} inputs, median {F.floor.median():.7g}; " + ", ".join(f"{r.gsm} {r.floor:.6g}" for r in F[F.cell == 'granulocytes'].itertuples()),
     f"held-error reference: {len(H['cells'])} cells; granulocytes {H['cells'].get('granulocytes')}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
