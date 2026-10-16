"""DEV-SECONDLAB-02 diagnosis (reported, not scored; PHY-050): each Hansen donor's held error with its own measured sequencing error and conversion
failure taken out, (held - sequencing error) / (1 - conversion failure), from the stamped counts. Writes meter_diagnosis.json (stamped)."""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
rows = []
for f in sorted(glob.glob(os.path.join(H, "out", "SRR*.json"))):
    o = json.load(open(f)); assert Q2.payload_sha(o) == o["chain_stamp"]["payload_sha256"]
    rows.append(dict(donor=o["donor"], conv_fail=round(o["conv_fail"], 4), seq_err=round(o["sub_err"], 4), held=round(o["held_err"], 4),
                     corrected=round((o["held_err"] - o["sub_err"]) / (1 - o["conv_fail"]), 4)))
Q2.write_json(dict(rows=rows, corrected_min=min(r["corrected"] for r in rows), corrected_max=max(r["corrected"] for r in rows), scored=False), os.path.join(H, "meter_diagnosis.json"), manifest=["PHY-050"])
print(json.dumps(rows))
