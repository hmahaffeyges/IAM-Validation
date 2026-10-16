"""DEV-CHAIN-Q2-01 step 3b: the sample-shape reference (COM-IAMA-053, sealed 2026-10-15), built BY THE CHAIN from the stamped counts in
shape_out/ (Loyfer 153: 146 on box i-0f54ab117054ee79f, 7 GEO-refused counted locally by the same committed chain d7f64ca700df; ENCODE 5).
Checks every stamp and that held error, loose share and opportunities equal the step-2 counts in out/ exactly, then writes the stamped
reference chain/Runtime Matrices/IAM_A_SelfRef/shape_reference_v1.json and applies it to the ENCODE five (encode_shape_v1.json)."""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
KNOWN = {v["chain_sha"] for v in json.load(open(os.path.join(R, "CANON/chain_versions.json")))["versions"]}
def load(d, pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(H, d, pat))):
        o = json.load(open(f)); st = o["chain_stamp"]
        assert Q2.payload_sha(o) == st["payload_sha256"], f"{f}: edited after the chain wrote it"
        assert st["chain_sha"] in KNOWN, f"{f}: chain version not committed"
        out[os.path.basename(f)] = o
    return out
N, O = load("shape_out", "*.json"), load("out", "*.json")
L = {k: v for k, v in N.items() if k.startswith("GSM")}; E = {k: v for k, v in N.items() if k.startswith("ENCFF")}
assert len(L) == 153 and len(E) == 5, (len(L), len(E))
for k, v in N.items():
    for x in ("held_err", "loose_share", "held_opp"): assert v[x] == O[k][x], (k, x, v[x], O[k][x])
ref = Q2.build_shape_reference(list(L.values())); ref.update(version=1, built_from="Loyfer 2023 (GSE186458), 153 healthy samples, 56 cell types, chain Stage Q2 counts with tile percentiles")
Q2.write_json(ref, os.path.join(Q2.LINE_DIR, "shape_reference_v1.json"), inputs=[dict(name=k, sha256=v["chain_stamp"]["payload_sha256"]) for k, v in sorted(L.items())])
rows = [dict(Q2.shape_check(e, e.get("loyfer_cell") or e["cell"], 1), file=e["sample"], encode_cell=e["cell"], reference_cell=e.get("loyfer_cell") or e["cell"]) for e in E.values()]
Q2.write_json({"rows": rows, "read": sum(r["read"] for r in rows), "of": len(rows)}, os.path.join(H, "encode_shape_v1.json"))
print("margin", round(ref["margin"], 4), "|", ref["margin_basis"], "| reproduce step 2: identical on", len(N))
for r in rows: print(f"  {r['encode_cell']:<26} {r['reason']}")
