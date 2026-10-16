"""COM-IAMA-054: band v1 as the record quotes it, from the chain's stamped band_v1.json and each Hansen donor's reading (chain Stage Q2 reading()).
Writes band_v1_summary.json (stamped)."""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
B = json.load(open(os.path.join(Q2.LINE_DIR, "band_v1.json"))); assert Q2.payload_sha(B) == B["chain_stamp"]["payload_sha256"]
Hn = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_SECONDLAB_02/out/SRR*.json")))]
J = dict(lo=round(B["lo"], 4), hi=round(B["hi"], 4), median=round(B["median"], 4), n=B["n"], labs=B["labs"],
         hansen_readings={o["donor"]: round(Q2.reading(o, 1)["A"], 3) for o in Hn})
Q2.write_json(J, os.path.join(H, "band_v1_summary.json"), manifest=["COM-IAMA-054"]); print(json.dumps(J))
