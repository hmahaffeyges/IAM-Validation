"""The two cell-mix readers compared (to-do step 34, 2026-10-22; a reported line, not a bar). Stage A (conductor_v3.stage_a_composition: 8
blood groups by non-negative least squares on 963 markers) and Stage D21 (the IAM-Deconvolver against IAM-Atlas v2.1: 6 blood cells) each give
the neutrophil share of a whole blood. Here both, as the chain recorded them in Group-A's committed reading (scored_groupa_01_rows_idat.csv:
the 278 healthy controls of GSE318173, each array's own unplanted IDAT reading; f_neu is Stage A's, share_neutrophils Stage D21's).
Usage: python3 two_readers_01.py   (writes two_readers_01_summary.txt beside itself)"""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(H, "scored_groupa_01_rows_idat.csv"))
R = R[(R.planted.astype(str).str.lower().isin(["false", "0", "no", "nan"])) | R.planted.isna()] if "planted" in R else R
x = R.f_neu.astype(float); y = R.share_neutrophils.astype(float); ok = x.notna() & y.notna(); x, y = x[ok], y[ok]; d = y - x
r = float(np.corrcoef(x, y)[0, 1]); sd = float(d.std(ddof=1))
L = [f"arrays with both readings: {ok.sum()} of {len(R)}",
     f"Stage A neutrophil share: median {x.median():.3f} (range {x.min():.3f} to {x.max():.3f})",
     f"Deconvolver neutrophil share: median {y.median():.3f} (range {y.min():.3f} to {y.max():.3f})",
     f"Deconvolver minus Stage A: mean {d.mean():+.4f}, SD {sd:.4f}; 95 % of arrays within {d.mean() - 1.96 * sd:+.4f} to {d.mean() + 1.96 * sd:+.4f}",
     f"correlation {r:.3f}; largest disagreement {d.abs().max():.4f}"]
open(os.path.join(H, "two_readers_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
