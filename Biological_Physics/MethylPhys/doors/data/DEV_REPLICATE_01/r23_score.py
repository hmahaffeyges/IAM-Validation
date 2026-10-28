"""DEV-REPLICATE-01 R2 (smoking, GSE42861) and R3 (aging, GSE87571) scorer, written and committed before any of their readings exists. It only
summarizes what run_sample.py wrote (pass-1 bundles, untared) against the labels committed beside it; nothing is fitted or chosen. R2: current
against never smokers, all and the controls alone: each reading's medians, the difference, Cohen's d (pooled SD) and a two-sided Mann-Whitney
p. R3: each reading against age, Spearman's rho and its p. Readings: the control sites (recorded values), the cell shares of neutrophils, CD4,
CD8 and B cells, NK cells and monocytes, Met-A for the whole sample and each place, Met-A for neutrophils, CD4 and CD8 T cells, B cells and
monocytes, the two territories, and the whole sample's distance to the ceiling.
R4 (added before any R4 reading exists): AIBL GSE153712, Alzheimer's disease and mild cognitive impairment each against healthy controls, by
the same statistics as R2. R5 (added before any R5 reading exists): GSE281691, metabolic liver cancer (HCC) cases against controls, blood leukocytes, by the same
statistics as R2. Usage: python3 r23_score.py r2|r3|r4|r5 FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); which, src = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(src, f"{which}_summary.txt")
L = pd.read_csv(os.path.join(H, {"r2": "r2_smoking_labels.csv", "r3": "r3_aging_labels.csv", "r4": "r4_aibl_labels.csv", "r5": "r5_liver_labels.csv"}[which]), dtype=str).set_index("gsm")
SH = ("neutrophils", "cd4 t cells", "cd8 t cells", "b cells", "nk cells", "monocytes"); EC = ("neutrophils", "cd4 t cells", "cd8 t cells", "b cells", "monocytes")
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json")): continue
    g = os.path.basename(f).split(".")[0].split("_")[0]
    if g in seen or g not in L.index: continue
    seen.add(g); B = json.load(open(f)); r = {"gsm": g}
    for c, v in ((B.get("control_sites") or {}).get("values") or {}).items(): r["site " + c] = v
    sh = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or []}
    for c in SH: r["share " + c] = sh.get(c)
    w = B.get("met_a_whole") or {}; r["Met-A whole"] = w.get("A"); r["ceiling distance"] = (w.get("ceiling") or {}).get("distance_to_ceiling")
    for k, v in (w.get("by_place") or {}).items(): r["Met-A whole, " + k] = (v or {}).get("A")
    ec = {x["cell"]: x.get("A") for x in (B.get("met_a_each_cell") or {}).get("cells", [])}
    for c in EC: r["Met-A " + c] = ec.get(c)
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): r["territory " + k] = v.get("A")
    rows.append(r)
X = pd.DataFrame(rows).set_index("gsm").join(L); feats = [c for c in X.columns if c not in L.columns]
T = [f"DEV-REPLICATE-01 {which}: {len(X)} of {len(L)} arrays read (pass 1, untared)"]
if which in ("r2", "r4", "r5"):
    pairs = ((("all", X), "current", "never", "smoking_status"), (("controls only", X[X.disease_state.str.lower() == "normal"] if which == "r2" else X), "current", "never", "smoking_status")) if which == "r2" else \
            (((("Alzheimer's", X), "Alzheimer's disease", "healthy control", "status"), (("mild cognitive impairment", X), "Mild Cognitive Impairment", "healthy control", "status")) if which == "r4" else
             ((("all", X), "metabolic HCC case", "control", "status"),))
    for (name, M), ga, gb, col in pairs:
        a, b = M[M[col] == ga], M[M[col] == gb]; T.append(f"-- {ga} ({len(a)}) against {gb} ({len(b)}), {name}")
        for c in feats:
            x, y = pd.to_numeric(a[c], errors="coerce").dropna(), pd.to_numeric(b[c], errors="coerce").dropna()
            if len(x) < 5 or len(y) < 5: continue
            sd = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2)); d = (x.mean() - y.mean()) / sd if sd > 0 else float("nan")
            T.append(f"{c:42} {'case' if which == 'r5' else 'current'} {x.median():.4f} {'control' if which == 'r5' else 'never'} {y.median():.4f} | d {d:+.2f} | p {stats.mannwhitneyu(x, y).pvalue:.2g}")
else:
    age = pd.to_numeric(X.age, errors="coerce")
    for c in feats:
        v = pd.to_numeric(X[c], errors="coerce"); ok = v.notna() & age.notna()
        if ok.sum() < 10: continue
        rho, p = stats.spearmanr(age[ok], v[ok]); T.append(f"{c:42} rho {rho:+.3f} | p {p:.2g} | n {int(ok.sum())}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T))
