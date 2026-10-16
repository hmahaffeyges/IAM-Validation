"""DEV-IAMA-GRAND-01 scored ONCE by the sealed rule (DEV_IAMA_GRAND_01.md), through chain Stage Q2 grand_verdict only: every count's stamp
verified; shape check first against the sample's Loyfer type; each read sample's IAM-A reading against band v1; MET if at least 75 percent of
the read samples are inside Normal and at least 8 were read. Writes grand_scored.json (stamped, with its verdict)."""
import os, sys, json, glob, csv
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
KNOWN = {v["chain_sha"] for v in json.load(open(os.path.join(R, "CANON/chain_versions.json")))["versions"]}
S = {r["runs"]: r for r in csv.DictReader(open(os.path.join(H, "samples.csv")))}; C = []
for f in sorted(glob.glob(os.path.join(H, "out", "SRR*.json"))):
    o = json.load(open(f)); st = o["chain_stamp"]
    assert Q2.payload_sha(o) == st["payload_sha256"], f"{f}: edited after the chain wrote it"; assert st["chain_sha"] in KNOWN, f"{f}: chain version not committed"
    C.append((o, S[o["run"]]["shape_type"]))
missing = sorted(set(S) - {o["run"] for o, _ in C})
V = Q2.grand_verdict(C); V.update(missing=missing, rule="DEV_IAMA_GRAND_01.md (sealed 2026-10-16)")
for r in V["rows"]: r["subset"] = S[r["sample"] if r["sample"] in S else next(k for k, v in S.items() if v["gsm"] == r["sample"])]["subset"]
Q2.write_json(V, os.path.join(H, "grand_scored.json"), manifest=["COM-IAMA-052"])
print(V["verdict"], "| read", V["read"], "of", V["of"], "| inside Normal", V["inside"], "| missing", missing)
for r in V["rows"]: print(" ", r["sample"], r["subset"], "|", r["shape"], "|", ("A %.3f %s" % (r["A"], "inside" if r["inside"] else "outside")) if r["read"] else "")
