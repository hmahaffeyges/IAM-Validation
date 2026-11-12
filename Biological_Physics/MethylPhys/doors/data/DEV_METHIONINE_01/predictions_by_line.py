"""DEV-METHIONINE-01: the sealed predictions per cell line (PHY-083; written before any depleted library is read).
Where IAM's Law is: the hold's gain is the coin in thermal kicks, M (canon M_cell). If the gain is the coin's ENERGY, a fall in SAM/SAH by
a factor r lowers each methyl transfer by ln r of a kick (Einstein-relation route with the dyad's two coins: odds(W) with W = 2(M - ln r) kT);
if it is the coin's SUPPLY, the held error rises by the factor r. The falls r come from two independent readings of the paper's Fig. 2B
(each control normalized to 1.0): reader one (the author) and reader two (a separate vision reading given only the image). A line is scored
only where the two readings agree. Writes predictions_by_line_summary.txt."""
import os, json, math
H = os.path.dirname(os.path.abspath(__file__))
M = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
odds = lambda W: (W / 2) / (1 - (1 - math.exp(-W)) / W)
e0 = 1 / (1 + odds(2 * M)); energy = lambda r: (1 / (1 + odds(2 * (M - math.log(r))))) / e0
READ = {"MZGC1": {"reader one": (0.75, 0.75), "reader two": (0.70, 0.70)},
        "MZGC3": {"reader one": (0.25, 0.30), "reader two": (0.20, 0.25)},
        "MZGC2": {"reader one": None, "reader two": (1.0, 1.0)}}   # reader one: no bar drawn; reader two: about 1.0
L = [f"M = {M}; control held error (two coins) {e0:.5f}", "line | depleted SAM/SAH (both readings) | r range | energy route | supply route"]
for line, d in READ.items():
    if d["reader one"] is None or d["reader two"] is None or abs(min(d["reader one"]) - min(d["reader two"])) > 0.15:
        L.append(f"{line} | readings disagree ({d}) | not scored"); continue
    lo_h, hi_h = min(min(v) for v in d.values()), max(max(v) for v in d.values())
    r_lo, r_hi = 1 / hi_h, 1 / lo_h
    L.append(f"{line} | {lo_h:.2f}-{hi_h:.2f} | {r_lo:.2f}-{r_hi:.2f} | x{energy(r_lo):.3f}-{energy(r_hi):.3f} | x{r_lo:.2f}-{r_hi:.2f}")
t = "\n".join(L) + "\n"; open(os.path.join(H, "predictions_by_line_summary.txt"), "w").write(t); print(t)
