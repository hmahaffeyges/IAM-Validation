"""Register A7, the within-sample slope from the derived hold (exploration on spent tile counts; 2026-11-14). Where IAM's Law is: the hold
is the derived pump, eps = 1/(1 + odds(2 M f)), odds(w) = (w/2)/(1 - (1 - e^-w)/w), M = dG_ATP/(R T) (CANON). Hypothesis, written before the
measured within-sample slope was re-read (private register, 5:51 AM Pacific): maintenance is cooperative, so the push a held mark receives
scales with the held share of its neighbourhood, f = 1 - x (x = the tile's looseness). Prediction then: the counter's held error rises with
x at d eps/dx; at x = 0 that is 0.0419 per site, 0.0369-0.0398 through the counter (lone loss 0.88-0.95 of per site); fail if the median
measured slope is outside 0.5-2 times, or of the other sign. After the fact: the same form at each sample's own mean looseness. Reads
slope_read_summary.txt (the eleven samples' within-sample slopes); writes slope_pump_summary.txt."""
import os, math, statistics as S
H = os.path.dirname(os.path.abspath(__file__))
M = 54000 / (8.314462618 * 310.15)
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
e = lambda x: 1 / (1 + odds(2 * M * (1 - x)))
sl = lambda x: (e(x + 0.005) - e(x - 0.005)) / 0.01
rows = [l.split() for l in open(os.path.join(H, "slope_read_summary.txt")) if l.strip().startswith("GSM")]
meas = [float(r[-2]) for r in rows]; s0 = sl(0.0)
L = [f"M {M:.2f}; predicted slope at x = 0: {s0:.4f} per site, {0.88 * s0:.4f}-{0.95 * s0:.4f} through the counter",
     f"measured within-sample slope, eleven samples: median {S.median(meas):.4f} (range {min(meas):.4f}-{max(meas):.4f}); "
     f"median / prediction {S.median(meas) / (0.915 * s0):.2f}: {'inside' if 0.5 <= S.median(meas) / (0.915 * s0) <= 2 else 'OUTSIDE'} 0.5-2 (written before reading)",
     "after the fact, at each sample's own mean looseness x (ratio = measured / (0.915 x predicted per-site slope)):"]
rat = []
for r in rows:
    x, wb = float(r[-4]), float(r[-2]); cell = " ".join(r[1:-7]); q = wb / (0.915 * sl(x))
    if "erythrocyte" not in cell: rat.append(q)
    L.append(f"  {r[0]} {cell:28} x {x:.3f} measured {wb:.4f} predicted {0.88 * sl(x):.4f}-{0.95 * sl(x):.4f} ratio {q:.2f}")
L.append(f"the ten that passed PHY-087's null test: median ratio {S.median(rat):.2f} (range {min(rat):.2f}-{max(rat):.2f}); erythrocyte "
         "progenitors (failed the null test; loose share 0.957) far below")
L.append("Exploration on spent counts; the one assumption is the linear crew share f = 1 - x; nothing fitted.")
open(os.path.join(H, "slope_pump_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
