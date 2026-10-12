"""DEV-TISSUE-COLON-02 layout (metadata only, 2026-10-11): GSE199057 sample titles, tissue type and Sentrix slide from the GEO sample
records and supplementary file NAMES (no array opened). Pairs tumour [T<id>] with the same patient's normal [N<id>/<id>]; plate controls
excluded. Reads gse199057_samples.csv; writes layout_02_output.txt. Usage: python3 layout_02.py"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(HERE, "gse199057_samples.csv"), dtype={"pid": str, "slide": str})
S["grp"] = np.where(S.kind.str.startswith("colon tumor"), "T", np.where(S.kind.str.contains("non-CRC"), "H", "N"))
T = S[S.grp == "T"]; N = S[(S.grp == "N") & ~S.ctrl]; P = T.merge(N, on="pid", suffixes=("_t", "_n"))
x = S.groupby("slide").grp.agg(lambda v: "".join(sorted(set(v))))
out = [f"arrays {len(S)} (EPIC v1): tumours {len(T)}, patients' normals {len(N)}, non-CRC normals {int((S.grp == 'H').sum())}, plate controls {int(S.ctrl.sum())}",
       f"tumour-normal pairs by patient number: {len(P)} ({P.pid.nunique()} patient numbers); pairs on the same slide: {int((P.slide_t == P.slide_n).sum())}",
       f"slides {S.slide.nunique()} by content: " + ", ".join(f"{k} {v}" for k, v in x.value_counts().items()) + " (T tumour, N patient normal, H non-CRC normal)"]
open(os.path.join(HERE, "layout_02_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
