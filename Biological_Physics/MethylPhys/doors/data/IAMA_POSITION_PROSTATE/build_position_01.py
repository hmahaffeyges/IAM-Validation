"""IAM-A position P of one cell type from Loyfer 2023 (GSE186458) hg19 .pat files, WHOLE files, by the definition in
iama_positions_v2.json: per donor P_i = H(mean ε of the other donors) / H(ε0), ε0 from the canon (eps0_meth); P = mean of the P_i.
ε by chain/pat_eps_fast.py (Stage Q's whole-file rule, streamed; identical to Stage Q on GSM5652279). Usage: build_position_01.py CELL OUT.json PAT..."""
import os, sys, json, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
e0 = json.load(open(os.path.join(MP, "chain/Runtime Matrices/IAM_A_Positions/iama_positions_v2.json")))["eps0"]
e0 = e0["value"] if isinstance(e0, dict) else e0
cell, out, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
C = {}
for p in pats:
    r = subprocess.run([sys.executable, os.path.join(MP, "chain/pat_eps_fast.py"), p], capture_output=True, text=True).stdout.strip().split("\n")[-1]
    C[os.path.basename(p).split("_")[0]] = json.loads(r); print(os.path.basename(p), C[os.path.basename(p).split("_")[0]], flush=True)
eps = {g: c["errors"] / c["opportunities"] for g, c in C.items()}
Pi = {g: H(sum(v for k, v in eps.items() if k != g) / (len(eps) - 1)) / H(e0) for g in eps}
P = sum(Pi.values()) / len(Pi); m = sum(Pi.values()) / len(Pi); sd = (sum((x - m) ** 2 for x in Pi.values()) / (len(Pi) - 1)) ** 0.5
rec = dict(cell=cell, P=round(P, 4), P_range=[round(min(Pi.values()), 4), round(max(Pi.values()), 4)], cv_across_donors=round(sd / m, 4), eps0=e0,
           eps={g: round(v, 6) for g, v in eps.items()}, counts=C, n_donors=len(eps), pipeline="loyfer_pat_v1", genome="hg19", coverage="whole file")
json.dump(rec, open(out, "w"), indent=1); print(json.dumps({k: rec[k] for k in ("cell", "P", "P_range", "cv_across_donors", "eps")}))
