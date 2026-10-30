"""R12 within each slide (added after R12 was compared, disclosed; reference only, nothing scored): for the slides holding both groups, each
slide's median reading of later breast cancers minus its median of no-cancer bloods, then the median of those slide differences and a
Wilcoxon signed-rank test across slides, so a slide's own offset cannot make a group difference. Reads the chain's rows table.
Usage: python3 r12_within_slide.py ROWS.csv OUT.txt"""
import os, sys, pandas as pd
from scipy.stats import wilcoxon
H = os.path.dirname(os.path.abspath(__file__)); L = pd.read_csv(os.path.join(H, "r12_italy_labels.csv"), dtype=str).set_index("gsm")
R = pd.read_csv(sys.argv[1], dtype={"gsm": str}); R["key"] = R.reading.astype(str) + " | " + R.place.astype(str)
want = ["Met-A each cell | neutrophils", "Met-A each cell | cd4 t cells", "Met-A each cell | cd8 t cells", "Met-A each cell | b cells", "Met-A each cell | monocytes",
        "Met-A whole sample | whole sample", "Met-A whole sample, by place | islands", "Met-A whole sample, by place | domains"]
X = R[R.key.isin(want)].pivot_table(index="gsm", columns="key", values="A", aggfunc="first").join(L[["slide", "status"]], how="inner")
X = X[X.status.isin(["later breast cancer", "no cancer"])]; out = [f"R12 within each slide: {len(X)} arrays, later breast cancer against no cancer, reading | slides | median slide difference | Wilcoxon p"]
for k in want:
    if k not in X: continue
    d = []
    for s, g in X.groupby("slide"):
        a, b = g[g.status == "later breast cancer"][k].dropna(), g[g.status == "no cancer"][k].dropna()
        if len(a) and len(b): d.append(a.median() - b.median())
    p = wilcoxon(d).pvalue if len(d) > 5 else float("nan"); out.append(f"{k} | {len(d)} | {pd.Series(d).median():+.4f} | {p:.2g}")
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print("\n".join(out))
