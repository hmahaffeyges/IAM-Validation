"""Group-A, reported line (2026-10-20, after scoring; no bar): the planted loss is scattered, so it should move Met-A and leave the C-Score
alone. Per array, on the same array, the planted reading against its own unplanted reading (both from the raw files' calibrated values:
the raw-file route for the unplanted reading, the values route for the planted one). Reads the committed rows; writes tandem_groupa_summary.txt."""
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
I = pd.read_csv(os.path.join(H, "scored_groupa_01_rows_idat.csv")); V = pd.read_csv(os.path.join(H, "scored_groupa_01_rows_values.csv"))
m = I.merge(V, on="gsm", suffixes=("_raw", "_planted")); p = m[m.half_raw == "planted"]; u = m[m.half_raw == "unplanted"]
L = [f"unplanted half, same array by both routes ({len(u)} arrays): largest difference in Met-A {(u.A_raw - u.A_planted).abs().max():.4f}, in the C-Score {(u.C_raw - u.C_planted).abs().max():.4f}",
     f"planted half ({len(p)} arrays), planted against the array's own unplanted reading: Met-A moved by up to {(p.A_planted - p.A_raw).abs().max():.4f} (mean {(p.A_planted - p.A_raw).mean():+.4f}); "
     f"the C-Score moved by up to {(p.C_planted - p.C_raw).abs().max():.4f} (mean {(p.C_planted - p.C_raw).mean():+.4f})",
     f"healthy arrays above the C-Score's upper limit 1.152, slide-or-series tare, raw-file route: {int((I.C_rel > 1.152).sum())} of {int(I.C_rel.notna().sum())}"]
open(os.path.join(H, "tandem_groupa_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
