"""The library floor's test (written and committed before any real sample's floor is read). The 137 healthy inputs of the Met-A test
(metaa_seqatlas_healthy.csv) read again by the one command, the floor recorded on each (development.library_floor) beside Met-A against the
other donor half. Depth: as in the Met-A test (seq_term_01/band_01/ratio_depth_readings.csv). Reported: the floor's median and range, its
Spearman rho with depth, Met-A's rho with the floor. Bar (both): the floor follows depth (|rho| at least 0.3, the same sign as Met-A's, which
was negative) and Met-A follows the floor (rho at least 0.3). Usage: python3 floor_score.py BUNDLE_DIR OUT.txt"""
import sys, os, glob, json, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "metaa_seqatlas_healthy.csv"))
R0 = pd.read_csv(os.path.join(H, "..", "seq_term_01", "band_01", "ratio_depth_readings.csv"))[["gsm", "depth"]]
rows = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True))
    if not f: continue
    D = json.load(open(f[-1])).get("development") or {}; m = D.get("metaa_atlas") or {}; fl = D.get("library_floor") or {}
    rows.append(dict(gsm=r.gsm, cell=r.cell, A=m.get("A"), floor=fl.get("rate"), floor_unmarked=(fl.get("unmarked") or {}).get("rate"), floor_calls=(fl.get("unmarked") or {}).get("calls")))
d = pd.DataFrame(rows).merge(R0, on="gsm", how="left"); t = d[d.floor.notna() & d.A.notna()]
r1 = spearmanr(t.floor, t.depth).correlation; r2 = spearmanr(t.A, t.floor).correlation; ok = r1 <= -0.3 and r2 >= 0.3
L = ["The library floor's test: the 137 healthy inputs read again by the one command, floor and Met-A (against the other donor half) on each",
     f"read {len(d)} of {len(S)}; with floor and Met-A {len(t)}",
     f"floor: median {t.floor.median():.5f}, 2.5th to 97.5th percentile {t.floor.quantile(.025):.5f} to {t.floor.quantile(.975):.5f}",
     f"floor against depth: rho {r1:+.3f} (at most -0.3 needed)", f"Met-A against the floor: rho {r2:+.3f} (at least 0.3 needed)", f"bar: {'met' if ok else 'not met'}"]
d.to_csv(os.path.join(H, "floor_readings.csv"), index=False); open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
