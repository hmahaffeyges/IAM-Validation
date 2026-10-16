"""DEV-CHAIN-Q2-01 step 2: the self-reference line built BY THE CHAIN from the chain's own stamped counts (a reproduction of the development line,
not a new test), then the ENCODE five read against it by the chain. Every input count's stamp is verified first; any edited or foreign count stops it.
 out/GSM*.json   Loyfer's 153 samples, chain Stage Q2 count_pat (box i-0f3380b0fa6d2bd0a and, for 13 GEO refused with 503, the same chain run locally)
 out/ENCFF*.json the ENCODE five, chain Stage Q2 count_bam
Writes: chain/Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json (stamped), line_v1_cells.json, line_v1_vs_development.json, encode_vs_line_v1.json"""
import os, sys, json, glob, re
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
import pandas as pd
_CV = json.load(open(os.path.join(R, "CANON/chain_versions.json")))
KNOWN = {v["chain_sha"] for v in _CV["versions"]}   # every chain version the checkpoint registered at a commit
def known_chain(sha): return sha in KNOWN
def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(H, "out", pat))):
        o = json.load(open(f)); st = o.get("chain_stamp") or {}
        assert Q2.payload_sha(o) == st.get("payload_sha256"), f"{os.path.basename(f)}: edited after the chain wrote it"
        assert known_chain(st.get("chain_sha")), f"{os.path.basename(f)}: written by a chain version that is not committed"
        out.append(o)
    return out
L = load("GSM*.json"); E = load("ENCFF*.json")
assert len(L) == 153 and len({o["sample"] for o in L}) == 153, len(L)
P = pd.DataFrame(L)[["sample", "cell", "loose_share", "held_err", "held_opp"]]
J, C = Q2.build_line(P)
J["version"] = 1; J["built_from"] = "Loyfer 2023 (GSE186458), 153 samples, 56 healthy cell types, counted by chain Stage Q2 count_pat; DEV-CHAIN-Q2-01"
os.makedirs(Q2.LINE_DIR, exist_ok=True)
Q2.write_json(J, os.path.join(Q2.LINE_DIR, "selfref_line_v1.json"), inputs=[dict(name=os.path.basename(o["sample"]) + ".json", sha256=o["chain_stamp"]["payload_sha256"]) for o in L])
Q2.write_json({"cells": C.to_dict(orient="records")}, os.path.join(H, "line_v1_cells.json"))
# the development line, read from its own files (its constants are on the scaled axis: divide by its SCALE)
dv = os.path.join(R, "development/loose_share_2026-10-15"); DJ = json.load(open(os.path.join(dv, "opensea_line.json")))["own_reads_every_CpG"]
SC = float(re.search(r"^SCALE\s*=\s*([0-9.]+)", open(os.path.join(dv, "opensea_line.py")).read(), re.M).group(1))
dev = dict(firm=DJ["firm"] / SC, slope=(DJ["loose"] - DJ["firm"]) / SC, cells=DJ["cells"], printed_to="4 decimals on the scaled axis")
tol = 0.00005 / SC * 2
cmp_ = dict(chain=dict(firm=J["firm"], slope=J["slope"], cells=J["cells"]), development_raw_axis=dev,
            firm_diff=J["firm"] - dev["firm"], slope_diff=J["slope"] - dev["slope"], tolerance=tol,
            agree=bool(abs(J["firm"] - dev["firm"]) <= tol and abs(J["slope"] - dev["slope"]) <= 2 * tol and J["cells"] == dev["cells"]))
Q2.write_json(cmp_, os.path.join(H, "line_v1_vs_development.json"))
rows = [dict(cell=o["cell"], file=o["sample"], **Q2.residual(o, version=1)) for o in E]
Q2.write_json({"rows": rows, "within_0_003": sum(abs(r["residual"]) <= 0.003 for r in rows), "of": len(rows)}, os.path.join(H, "encode_vs_line_v1.json"))
print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in J.items() if k in ("firm", "slope", "cells", "samples", "residual_abs_median", "within_0_003")}))
print("vs development:", "AGREE" if cmp_["agree"] else "DIFFER", {k: round(cmp_[k], 7) for k in ("firm_diff", "slope_diff", "tolerance")})
for r in rows: print(f"  {r['cell']:<28} L {r['loose_share']:.3f} held {r['held_err']:.4f} predicted {r['predicted']:.4f} residual {r['residual']:+.4f}")
