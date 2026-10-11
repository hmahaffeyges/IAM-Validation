"""DEV-FINGERPRINT-02 outcome table, from the committed scored_<pair>.json files only (rule 02b). Writes outcome_02_output.txt and prints the
markdown body of DEV_FINGERPRINT_02_OUTCOME.md. Usage: python3 outcome_02.py"""
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); rows = []
for f in sorted(glob.glob(os.path.join(HERE, "scored_*.json"))):
    o = json.load(open(f)); p = o["pair"]; S = json.load(open(os.path.join(HERE, f"sealed_{p}.json")))
    if "02b" not in S.get("rule", ""): continue
    ex = list(o["cancer_excess"].values())
    rows.append(f"| {p} | {S['cancer_cell']} vs {S['normal_cell']} | {min(ex):.4f} to {max(ex):.4f} | {o['armB_bar']:.4f} | {o['ratio_min_excess_to_donor']:.2f} | **{o['VERDICT']}** | "
                f"IAM-A_rel {o['IAMA_rel']:.4f}, Met-A_rel {o['MetA_rel']:.4f}, curve B {o['curveB']:.4f}, z {o['z']:.2f} ({o['armA']}) | share read {o['share_read']:.2f} |")
hdr = ("| pair | cancer vs normal | Arm B excess per cancer file | bar | min excess / donor excess | verdict (Arm B) | Arm A, reported only | readable share, cancer / normal |\n"
       "|---|---|---|---|---|---|---|---|")
body = hdr + "\n" + "\n".join(rows); open(os.path.join(HERE, "outcome_02_output.txt"), "w").write(body + "\n"); print(body)
