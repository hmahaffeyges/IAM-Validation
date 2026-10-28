"""DEV-ALL-INFO-01: the IAM-A reference values the record cites, copied from their committed sources (nothing computed).
Usage: python3 iama_reference_values.py OUT.txt"""
import sys, os, json
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6)
B = json.load(open(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_IAMA_GRAND_01/grand_rule.json")))
C = {c["cell"]: c for c in json.load(open(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/line_v1_cells.json")))["cells"]}
band = B.get("band") or B
L = [f"IAM-A Normal band v1 (Biological_Physics/MethylPhys/doors/data/DEV_IAMA_GRAND_01/grand_rule.json): {json.dumps(band)[:300]}"]
for c in ("naive cd4 t cells", "naive cd8 t cells", "b cells"):
    L.append(f"{c} (line_v1_cells.json): loose share {C[c]['L']:.4f}, held error {C[c]['eps']:.4f}, samples {C[c]['n']}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
