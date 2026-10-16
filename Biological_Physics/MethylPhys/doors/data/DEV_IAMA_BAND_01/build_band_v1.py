"""COM-IAMA-054: IAM-A's healthy band v1, built BY THE CHAIN after the Hansen counts exist, by the rule sealed 2026-10-16 (DEV_IAMA_BAND_01.md).
Inputs, every stamp verified: Loyfer's 153 counts (DEV_CHAIN_Q2_01/out/, lab 'Loyfer 2023') and the Hansen donors (DEV_SECONDLAB_02/out/, lab
'Hansen GSE49627') that pass the shape check; a refused donor is not pooled. Writes chain/Runtime Matrices/IAM_A_SelfRef/band_v1.json (stamped)."""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
KNOWN = {v["chain_sha"] for v in json.load(open(os.path.join(R, "CANON/chain_versions.json")))["versions"]}
def load(pat, lab):
    out = []
    for f in sorted(glob.glob(os.path.join(R, "Biological_Physics/MethylPhys/doors/data", pat))):
        o = json.load(open(f)); st = o["chain_stamp"]
        assert Q2.payload_sha(o) == st["payload_sha256"] and st["chain_sha"] in KNOWN, f
        o["lab"] = lab; out.append(o)
    return out
L = load("DEV_CHAIN_Q2_01/out/GSM*.json", "Loyfer 2023"); Hn = load("DEV_SECONDLAB_02/out/SRR*.json", "Hansen GSE49627")
assert len(L) == 153 and len(Hn) == 3, (len(L), len(Hn))
Hp = [o for o in Hn if Q2.shape_check(o, "b cells", 1)["read"]]
B = Q2.build_band(L + Hp, version=1); B.update(version=1, hansen_pooled=[o["run"] for o in Hp], hansen_refused=[o["run"] for o in Hn if o not in Hp])
Q2.write_json(B, os.path.join(Q2.LINE_DIR, "band_v1.json"), inputs=[dict(name=os.path.basename(o["sample"]), sha256=o["chain_stamp"]["payload_sha256"]) for o in L + Hp],
              manifest=["COM-IAMA-054"]); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in B.items()}))
