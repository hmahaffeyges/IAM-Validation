"""DEV-REPLICATE-01, R5 to R9 scorer (written and committed 2026-10-23 before any of their readings exists). It only summarizes what
run_sample.py wrote (pass-1 bundles, untared) against the labels committed beside it (TAG_labels.csv: gsm, series, group). For each series
and each pair of its groups (every group against the first named as REFERENCE, or all pairs when none is named): each reading's medians,
the difference, Cohen's d (pooled SD) and a two-sided Mann-Whitney p. Also each group's three largest cell shares (median), so a tissue's
make-up is reported as the IAM-Deconvolver found it. Readings as in r23_score.py. Nothing fitted.
Usage: python3 replay_score.py TAG FOLDER_OF_BUNDLES [REFERENCE_GROUP] [OUT.txt]"""
import sys, os, glob, json, itertools, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); tag, src = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None; out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(src, f"{tag}_summary.txt")
L = pd.read_csv(os.path.join(H, f"{tag}_labels.csv"), dtype=str).set_index("gsm")
EC = ("neutrophils", "cd4 t cells", "cd8 t cells", "b cells", "monocytes"); rows, shares, seen = [], {}, set()
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json")): continue
    g = os.path.basename(f).split(".")[0].split("_")[0]
    if g in seen or g not in L.index: continue
    seen.add(g); B = json.load(open(f)); r = {"gsm": g}
    sh = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or []}; shares[g] = sh
    for c, v in ((B.get("control_sites") or {}).get("values") or {}).items(): r["site " + c] = v
    w = B.get("met_a_whole") or {}; r["Met-A whole"] = w.get("A"); r["ceiling distance"] = (w.get("ceiling") or {}).get("distance_to_ceiling")
    for k, v in (w.get("by_place") or {}).items(): r["Met-A whole, " + k] = (v or {}).get("A")
    ec = {x["cell"]: x.get("A") for x in (B.get("met_a_each_cell") or {}).get("cells", [])}
    for c in ec: r["Met-A " + c] = ec.get(c)   # 2026-10-30 (author: the full chain is read on ALL the cells in the sample): every cell the chain read
    for x in (B.get("cell_shares") or {}).get("cells") or []: r["share " + x["cell"]] = x["share"]
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): r["territory " + k] = v.get("A")
    r["C-Score whole genome"] = ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"); rows.append(r)
X = pd.DataFrame(rows).set_index("gsm").join(L); feats = [c for c in pd.DataFrame(rows).columns if c != "gsm"]
T = [f"DEV-REPLICATE-01 {tag}: {len(X)} of {len(L)} arrays read (pass 1, untared)"]
for s, M in X.groupby("series"):
    gs = sorted(M.group.dropna().unique()); T.append(f"== {s}: " + ", ".join(f"{g} {int((M.group == g).sum())}" for g in gs))
    for g in gs:
        S = pd.DataFrame([shares[i] for i in M[M.group == g].index]).median().sort_values(ascending=False).head(3)
        T.append(f"   {g}: largest shares " + ", ".join(f"{c} {v:.2f}" for c, v in S.items()))
    pairs = [(g, ref) for g in gs if g != ref] if ref in gs else list(itertools.combinations(gs, 2))
    for a_, b_ in pairs:
        a, b = M[M.group == a_], M[M.group == b_]; T.append(f"-- {a_} ({len(a)}) against {b_} ({len(b)})")
        for c in feats:
            x, y = pd.to_numeric(a[c], errors="coerce").dropna(), pd.to_numeric(b[c], errors="coerce").dropna()
            if len(x) < 3 or len(y) < 3: continue
            sd = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2)); d = (x.mean() - y.mean()) / sd if sd > 0 else float("nan")
            T.append(f"   {c:42} {x.median():.4f} against {y.median():.4f} | d {d:+.2f} | p {stats.mannwhitneyu(x, y).pvalue:.2g}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T[:30]))
