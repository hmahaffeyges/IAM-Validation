"""PHY-086 P4, after the fact (exploration; the sealed verdict, NOT MET, is unchanged; 2026-11-13). Two readings:
 (1) the knockout over control ratio read at zero loose along IAM-A's slope (b = 0.034625), since the knockouts carry about twice the loose share;
 (2) what a zero eraser share would do to PHY-084's B-cell prediction: the sealed prediction (referee_out.csv, eraser ratio 0.8642 in the
     grid) times 0.8642, against the measured B-cell intercept and its combined sigma. Mouse cells set against human; a lead only.
Writes p4_after_summary.txt."""
import os, json, glob, csv, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); b = 0.034625
C = {json.load(open(f))["gsm"]: json.load(open(f)) for f in glob.glob(os.path.join(H, "counts", "*_count.json"))}
z = lambda gs: st.mean(C[g]["held_err_length"] - b * C[g]["loose_share"] for g in gs)
wt, ko = z(("GSM4907469", "GSM4907470")), z(("GSM4907471", "GSM4907472"))
R = {r["cell"]: r for r in csv.DictReader(open(os.path.join(H, "..", "..", "DEV_CHAIN_Q2_01", "referee_out.csv")))}["b cells"]
pred, meas, u = float(R["pred_mid"]), float(R["intercept"]), float(R["u_combined"])
p0 = pred * 0.8642; sig = u * pred
L = [f"(1) at zero loose: control {wt:.5f}, knockout {ko:.5f}, ratio {ko / wt:.3f}",
     f"(2) PHY-084 B cells: measured {meas:.4f}; sealed prediction {pred:.4f} (pull {(meas - pred) / sig:+.2f}); with a zero eraser share {p0:.4f}"
     f" (pull {(meas - p0) / sig:+.2f}, same sigma). After the fact; a lead for A8, not a result."]
open(os.path.join(H, "p4_after_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
