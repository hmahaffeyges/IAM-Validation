"""Reproduction check (2026-10-16): the chain copies of the atlas v2.1 use scorers (chain/stage_atlas_u1.py, stage_atlas_u1_planted.py,
stage_atlas_u2.py, stage_atlas_u3.py) must give the committed results exactly. Compares each rerun output (in <dir>) with the committed one:
JSON payloads with the chain stamp removed, CSV contents, planted summaries as text. Writes repro_chain_copies_v21.json (stamped).
Usage: python repro_chain_copies_v21.py <dir with rerun outputs>"""
import os, sys, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../..")); sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain"))
import stage_q2_selfref as Q2
d = sys.argv[1]; out = {}
for f in ("u1_scored_v21.json", "u2_scored_v21.json", "u3_scored_v21.json"):
    a = {k: v for k, v in json.load(open(os.path.join(H, f))).items() if k != "chain_stamp"}; b = {k: v for k, v in json.load(open(os.path.join(d, f))).items() if k != "chain_stamp"}
    out[f] = Q2._canon(a) == Q2._canon(b)
for f in ("u1_samples_v21.csv", "u3_samples_v21.csv"):
    out[f] = pd.read_csv(os.path.join(H, f), comment="#").equals(pd.read_csv(os.path.join(d, f), comment="#"))
for f in ("planted_u1_v21_summary.txt", "planted_u2_v21_summary.txt", "planted_u3_v21_summary.txt"):
    out[f] = open(os.path.join(H, f)).read() == open(os.path.join(d, f)).read()
V = dict(identical=out, all_identical=all(out.values()))
Q2.write_json(V, os.path.join(d, "repro_chain_copies_v21.json"), manifest=["ATL-005"]); print(json.dumps(V))
