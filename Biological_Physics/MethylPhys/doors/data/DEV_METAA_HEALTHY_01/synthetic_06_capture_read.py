"""DEV-METAA-HEALTHY-01, the capture after synthetic test 6, read (development; nothing scored). Reads the capture's saved tables (cap_low.parquet,
cap_center.parquet: the values the healthy line was measured on, xs, the array's own values, x_input, the mix es and weights w) and writes the
stretch of xs against the input, and A / A_pred with the healthy prediction on the array's own values (stage_metaa_atlas.predicted_healthy,
censored line). Usage: python3 synthetic_06_capture_read.py FOLDER OUT.txt"""
import sys, os, numpy as np, pandas as pd
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_metaa_atlas as SA
fo, out = sys.argv[1:3]; SA.LINE_CENSORED = True; des, _, _ = SA._ruler(); L = []
for nm in ("low", "center"):
    C = pd.read_parquet(os.path.join(fo, f"cap_{nm}.parquet")); d = des.reindex(C.index)
    for k in ("I", "II"):
        m = (d == k).to_numpy(); x, y = C["x_input"][m].to_numpy(), C["xs"][m].to_numpy(); ok = np.isfinite(x) & np.isfinite(y); b, a = np.polyfit(x[ok], y[ok], 1)
        L.append(f"{nm} design {k}: {int(ok.sum())} sites; the values the line was measured on = {a:+.4f} + {b:.4f} x the array's own values")
    xr, es, w = C["x_input"], C["es"], C["w"]; hp = SA.predicted_healthy(xr, es, w); Ar = (w * SA._H(xr.to_numpy())).sum() / (w * SA._H(es.to_numpy())).sum()
    L.append(f"{nm}: on the array's own values A / A_pred {Ar / hp['A_pred']:.5f}; line " + "; ".join(f"design {k} a {v['a']:+.4f} b {v['b']:.4f}" for k, v in hp["blur"].items()))
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
