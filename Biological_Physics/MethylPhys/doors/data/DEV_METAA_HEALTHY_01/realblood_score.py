"""The four-laboratory spent test of the measured middle (DEV_METAA_HEALTHY_01, sealed 0bd173f2), its scorer, committed before any of its
readings is scored. Normal for each of the six blood cells, frozen before: the sorted donors' table (chain/Runtime Matrices/Met_A_Widths/
donor_widths_v1.json) where the cell holds a share its sorted donors held, else the in-blood table (realblood_inblood_normal.json: the
atlas's donors mixed into 200 healthy bloods, the rotation test) where it holds a share it held there; else no Normal. Gauge: Met-A divided
by the measured middle (measured healthy reads 1); Normal: 1 plus or minus k x width / middle. Bars: 1. each laboratory at least 90 % of the
readings with a Normal inside; 2. each cell at least 85 % across the four; 3. each laboratory's median gauge for each cell (5 or more
readings) within 2 widths of 1. Usage: realblood_score.py OUT.txt LAB=DIR_or_CSV ... (a DIR holds bundles; a CSV has gsm, cell, share, met_a)."""
import sys, os, json, glob
import pandas as pd, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]
TS = json.load(open(os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "Met_A_Widths", "donor_widths_v1.json")))["cells"]
TB = json.load(open(os.path.join(H, "realblood_inblood_normal.json")))["cells"]
def normal(cell, share):
    s, b = TS.get(cell), TB.get(cell)
    if s and share >= s["share_min"]: return s, "sorted"
    if b and share >= b["share_min"]: return b, "in a blood"
    return None, None
out, lines = [], []
for arg in sys.argv[2:]:
    lab, src = arg.split("=", 1)
    if src.endswith(".csv"): R = pd.read_csv(src)[["gsm", "cell", "share", "met_a"]]
    else:
        rows = []
        for f in glob.glob(os.path.join(src, "*.json")):
            try: B = json.load(open(f))
            except Exception: continue
            for c in (B.get("cells_found") or {}).get("cells") or []: rows.append(dict(gsm=os.path.basename(f).split("_")[0].split(".")[0], cell=c.get("cell"), share=c.get("share"), met_a=c.get("met_a")))
        R = pd.DataFrame(rows)
    R = R[R.cell.isin(SIX)].dropna(subset=["met_a", "share"])
    for _, r in R.iterrows():
        w, t = normal(r.cell, r.share)
        if not w: out.append(dict(lab=lab, gsm=r.gsm, cell=r.cell, share=r.share, met_a=r.met_a, table=None, gauge=None, widths=None, inside=None)); continue
        g = r.met_a / w["median"]; d = (r.met_a - w["median"]) / w["width"]
        out.append(dict(lab=lab, gsm=r.gsm, cell=r.cell, share=r.share, met_a=r.met_a, table=t, gauge=round(g, 5), widths=round(d, 3), inside=bool(abs(d) <= w["k"])))
X = pd.DataFrame(out); Xs = X.dropna(subset=["inside"]); X.to_csv(sys.argv[1].replace(".txt", ".csv"), index=False)
lines.append(f"{len(X)} readings of the six cells; {len(Xs)} with a Normal ({Xs.table.value_counts().to_dict()})")
L1 = Xs.groupby("lab").inside.agg(["sum", "size"]); b1 = bool(((L1["sum"] / L1["size"]) >= 0.90).all())
lines.append("bar 1, each laboratory >= 90 %: " + ("met" if b1 else "not met") + " | " + ", ".join(f"{l} {int(v['sum'])} of {int(v['size'])}" for l, v in L1.iterrows()))
C2 = Xs.groupby("cell").inside.agg(["sum", "size"]).reindex(SIX).dropna(); b2 = bool(((C2["sum"] / C2["size"]) >= 0.85).all())
lines.append("bar 2, each cell >= 85 %: " + ("met" if b2 else "not met") + " | " + ", ".join(f"{c} {int(v['sum'])} of {int(v['size'])}" for c, v in C2.iterrows()))
M3 = Xs.groupby(["lab", "cell"]).agg(n=("widths", "size"), med=("widths", "median")); M3 = M3[M3.n >= 5]; b3 = bool((M3.med.abs() <= 2).all())
lines.append("bar 3, each laboratory's median within 2 widths of 1: " + ("met" if b3 else "not met") + " | off: " + (", ".join(f"{l} {c} {v.med:+.2f}" for (l, c), v in M3[M3.med.abs() > 2].iterrows()) or "none"))
lines.append("overall: " + ("MET" if b1 and b2 and b3 else "NOT MET"))
lines.append("median gauge by laboratory and cell:\n" + Xs.pivot_table(index="cell", columns="lab", values="gauge", aggfunc="median").reindex(SIX).round(4).to_string())
open(sys.argv[1], "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
