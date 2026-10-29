"""DEV-METAA-ATLAS-CURVE-01, the healthy Met-A ratio at one common depth (written and committed before any thinned reading). Each of the
atlas's healthy sequencing inputs is read by the one command thinned to about 30,000,000 held-error chances (run_sample.py --dev-thin, the
fraction in healthy_seq_thin.csv; samples already below are read whole and left out of the bar). The ratio: Met-A over the median of the
other healthy samples of the same cell, all at the common depth. The test's job: once every sample has the same depth, the ratio no longer
follows the sample's original depth. Bar: |Spearman rho| of the ratio against the original depth at most 0.2. Also reported: the ratio's
median and 2.5th to 97.5th percentile. Usage: python3 thin_score.py BUNDLE_DIR OUT.txt"""
import sys, os, glob, json, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "healthy_seq_thin.csv"))
R0 = pd.read_csv(os.path.join(H, "..", "band_01", "ratio_depth_readings.csv"))[["gsm", "depth"]]
rows = []
for r in S.itertuples():
    f = glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True)
    if not f: continue
    m = (json.load(open(f[0])).get("development") or {}).get("metaa_atlas") or {}
    if m.get("A") is not None: rows.append((r.gsm, r.cell, float(m["A"]), float(r.thin)))
d = pd.DataFrame(rows, columns=["gsm", "cell", "A", "thin"]).merge(R0, on="gsm")
d["ratio"] = [x.A / d[(d.cell == x.cell) & (d.gsm != x.gsm)].A.median() if ((d.cell == x.cell) & (d.gsm != x.gsm)).any() else float("nan") for x in d.itertuples()]
t = d[(d.thin < 1) & d.ratio.notna()]; rho, p = spearmanr(t.ratio, t.depth)
L = [f"read {len(d)} of {len(S)}; thinned and with another sample of their cell {len(t)}",
     f"ratio at the common depth: median {t.ratio.median():.4f}, 2.5th to 97.5th percentile {t.ratio.quantile(.025):.4f} to {t.ratio.quantile(.975):.4f}",
     f"ratio against original depth: rho {rho:+.3f} (p {p:.2g}); bar (|rho| at most 0.2): {'met' if abs(rho) <= 0.2 else 'not met'}"]
d.to_csv(os.path.join(H, "thin_readings.csv"), index=False); open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
