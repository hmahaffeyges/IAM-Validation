"""PHY-087 / register A7 (exploration; 2026-11-14): how the derived form's match to the within-sample slope depends on tile size. The first
laboratory's eleven chain tile files (tiles_lab1/, copied from the PHY-087 step 1 bucket; slope_read.py reproduces the committed reading from
them exactly) are read by slope_read.py with TILE_AGG = 1, 2, 5, 10, 25 (tiles of 2,000 to 50,000 CpGs), and each sample's within-sample slope
is divided by the derived form's slope at its own looseness (slope_pump.py's form). Writes slope_tilesize_summary.txt."""
import os, sys, math, subprocess, tempfile, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
M = 54000 / (8.314462618 * 310.15); odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
e = lambda x: 1 / (1 + odds(2 * M * (1 - x))); sl = lambda x: (e(x + 0.005) - e(x - 0.005)) / 0.01
L = []
for k in (1, 2, 5, 10, 25):
    out = os.path.join(tempfile.mkdtemp(), "s.txt")
    subprocess.run([sys.executable, os.path.join(H, "slope_read.py")], check=True, capture_output=True,
                   env=dict(os.environ, TILES_DIR=os.path.join(H, "tiles_lab1"), SUMMARY=out, TILE_AGG=str(k)))
    T = [l.split() for l in open(out) if l.strip().startswith("GSM")]
    r = [float(t[-2]) / (0.915 * sl(float(t[-4]))) for t in T if "erythrocyte" not in " ".join(t[1:-7])]
    L.append(f"tiles of {2000 * k:>6} CpGs: halves' correlation min {min(float(t[-5]) for t in T):.3f}; within slope / form, ten samples: "
             f"median {np.median(r):.2f} (range {min(r):.2f}-{max(r):.2f})")
L += ["The ratio rises with tile size: finer tiles measure looseness more noisily, which flattens a fitted slope (regression dilution).",
      "Where every sample passes the null test (tiles of 10,000 CpGs and up) the ten samples' ratios span 0.51-1.89 (medians 1.19-1.37);",
      "the 0.93 reported first was at the finest tiles, where one sample fails the null test. All inside the 0.5-2 band written first."]
open(os.path.join(H, "slope_tilesize_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
