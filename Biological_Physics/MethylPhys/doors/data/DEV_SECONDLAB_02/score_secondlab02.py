"""DEV-SECONDLAB-02 scored ONCE by the sealed rule (DEV_SECONDLAB_02.md, step 2), through chain Stage Q2 only.
 1. every count's stamp verified (payload hash, committed chain version)
 2. the sample-shape check (COM-IAMA-053, shape_reference_v1) applied first, reference type 'b cells'; a refused donor is not scored
 3. each read donor's residual against the chain's line (version 1); S1 = at least 2 of the 3 donors within the stamped bar
    (bar_secondlab02.json); fewer than 2 donors read -> NOT ASSESSED
 4. reported, not scored: the lab shift (each donor's residual minus Loyfer's B-cell mean residual), sequencing error, read length
Writes secondlab02_scored.json (stamped, with its verdict, so the manifest records it automatically)."""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
KNOWN = {v["chain_sha"] for v in json.load(open(os.path.join(R, "CANON/chain_versions.json")))["versions"]}
C = []
for f in sorted(glob.glob(os.path.join(H, "out", "SRR*.json"))):
    o = json.load(open(f)); st = o["chain_stamp"]
    assert Q2.payload_sha(o) == st["payload_sha256"], f"{f}: edited after the chain wrote it"
    assert st["chain_sha"] in KNOWN, f"{f}: chain version not committed"
    C.append(o)
assert len(C) == 3, f"expected 3 donors, found {len(C)}"
BAR = json.load(open(os.path.join(H, "bar_secondlab02.json"))); assert Q2.payload_sha(BAR) == BAR["chain_stamp"]["payload_sha256"]
bar = BAR["bar"]
LB = [json.load(open(f)) for f in glob.glob(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/out/GSM*.json"))]
lb = [Q2.residual(o, version=1)["residual"] for o in LB if o["cell"] == "b cells"]; lb_mean = sum(lb) / len(lb)
rows = []
for o in C:
    sh = Q2.shape_check(o, "b cells", 1); r = dict(donor=o["donor"], sample=o["sample"], run=o["run"], shape=sh, read=sh["read"],
        loose_share=o["loose_share"], held_err=o["held_err"], sub_err=o["sub_err"], conv_fail=o["conv_fail"], read_length=o["read_length_mode"])
    if sh["read"]:
        q = Q2.residual(o, version=1); r.update(predicted=q["predicted"], residual=q["residual"], within=abs(q["residual"]) <= bar, lab_shift=q["residual"] - lb_mean)
    rows.append(r)
n_read = sum(r["read"] for r in rows); n_in = sum(bool(r.get("within")) for r in rows)
verdict = "NOT ASSESSED" if n_read < 2 else ("MET" if n_in >= 2 else "NOT MET")
out = dict(test="DEV-SECONDLAB-02 S1", verdict=verdict, donors_read=n_read, donors_within=n_in, need=2, bar=bar, line_version=1, loyfer_b_mean_residual=lb_mean, rows=rows)
Q2.write_json(out, os.path.join(H, "secondlab02_scored.json"), manifest=["PHY-047", "COM-IAMA-053"])
print(verdict, "| read", n_read, "of 3 | within the bar", n_in, f"| bar {bar:.4f}")
for r in rows: print(" ", r["donor"], r["run"], "| shape", "read" if r["read"] else r["shape"]["reason"], "| L %.3f held %.4f" % (r["loose_share"], r["held_err"]),
                  ("| residual %+.4f %s | lab shift %+.4f" % (r["residual"], "within" if r["within"] else "outside", r["lab_shift"])) if r["read"] else "", "| seq err %.4f | read length %s" % (r["sub_err"], r["read_length"]))
