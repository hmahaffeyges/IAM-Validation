"""DEV-FIRES-01 scorer (written and committed 2026-10-23 before any of the 845 readings is opened). It only summarizes what run_sample.py wrote
(pass-1 bundles) and compares by Group-A's own rule, the chain's stage_groupa.compare, unchanged: each reading tared against the cancer-free
references (ref = yes) on the same chip where it holds at least 3, else the series; then cases against the cancer-free, Welch difference in
standard errors; |t| >= 2 'groups differ'. Every result is preliminary (450K, and readings beyond Met-A are development). Groups: breast (C50)
and colorectal (C18, C19, C20); windows of years from the blood draw to diagnosis: 0-2, 2-5, 5-10, over 10, and all. Readings, in the record's
order: (1) the cell shares (neutrophils, lymphocytes as CD4 + CD8 + B + NK families, monocytes); (2) Met-A for the whole sample, each of its
places, and each cell (neutrophils, CD4 T, CD8 T, B cells, monocytes); (3) the two territories; and the whole sample's distance to the ceiling.
Added after the first reading (2026-10-23, disclosed in DEV_FIRES_01.md): --sex F|M keeps only that sex in both groups and among the tare
references, because the first reading compared breast cases (233 women, 2 men) with cancer-free people of both sexes (340 women, 84 men).
Usage: python3 fires_score.py FOLDER_OF_BUNDLES [OUT.txt] [--sex F|M]"""
import sys, os, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_groupa as GA
SEX = sys.argv[sys.argv.index("--sex") + 1] if "--sex" in sys.argv else None; args = [a for i, a in enumerate(sys.argv) if a != "--sex" and (i == 0 or sys.argv[i - 1] != "--sex")]
src = args[1]; out = args[2] if len(args) > 2 else os.path.join(src, "fires_summary.txt" if not SEX else f"fires_summary_sex{SEX}.txt")
S = pd.read_csv(os.path.join(H, "gse51032_samples.csv"), dtype={"slide": str}).set_index("gsm")
LYMPH = lambda c: any(k in c for k in ("t cell", "t cells", "th1", "treg", "b cell", "nk cell", "lymph"))
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json")): continue
    g = os.path.basename(f).split(".")[0].split("_")[0]
    if g in seen or g not in S.index: continue
    seen.add(g); B = json.load(open(f)); r = {"gsm": g}
    sh = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or []}
    r["share neutrophils"] = sh.get("neutrophils"); r["share lymphocytes"] = sum(v for c, v in sh.items() if LYMPH(c)) if sh else None; r["share monocytes"] = sh.get("monocytes")
    w = B.get("met_a_whole") or {}; r["Met-A whole"] = w.get("A"); r["ceiling distance"] = (w.get("ceiling") or {}).get("distance_to_ceiling")
    for k, v in (w.get("by_place") or {}).items(): r["Met-A whole, " + k] = (v or {}).get("A")
    ec = {x["cell"]: x.get("A") for x in (B.get("met_a_each_cell") or {}).get("cells", [])}
    for c in ("neutrophils", "cd4 t cells", "cd8 t cells", "b cells", "monocytes"): r["Met-A " + c] = ec.get(c)
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): r["territory " + k] = v.get("A")
    rows.append(r)
X = pd.DataFrame(rows).set_index("gsm").join(S); feats = [c for c in pd.DataFrame(rows).columns if c != "gsm"]
if SEX: X = X[X.sex.astype(str).str.upper() == SEX]
X["group"] = np.where(X.cancer == "none", "cancer-free", np.where(X.cancer == "C50", "breast", np.where(X.cancer.isin(["C18", "C19", "C20"]), "colorectal", "other")))
win = pd.cut(pd.to_numeric(X.ttd, errors="coerce"), [0, 2, 5, 10, 100], labels=["0-2 y", "2-5 y", "5-10 y", "over 10 y"])
healthy = (X.ref.astype(str).str.lower() == "yes")
T = [f"DEV-FIRES-01: {len(X)} of {len(S)} arrays read (pass 1){', sex ' + SEX + ' only' if SEX else ''}; cases against the cancer-free by Group-A (stage_groupa.compare), preliminary"]
for site in ("breast", "colorectal"):
    for wn in ["all", "0-2 y", "2-5 y", "5-10 y", "over 10 y"]:
        keep = (X.group == "cancer-free") | ((X.group == site) & ((win == wn) if wn != "all" else True))
        n = int(((X.group == site) & keep).sum()); T.append(f"-- {site}, {wn} ({n} cases, {int((X.group == 'cancer-free').sum())} cancer-free)")
        for c in feats:
            R = X[keep].assign(A=pd.to_numeric(X[keep][c], errors="coerce")).reset_index()[["gsm", "A", "slide", "group"]]
            try: o = GA.compare(R, "group", site, "cancer-free", healthy=healthy.reindex(R.gsm).reset_index(drop=True), platform="450K")
            except ZeroDivisionError: o = {"call": None, "reason": "no spread in this reading (every value equal)"}
            T.append(f"   {c:42} " + (f"diff {100 * o['rel_difference']:+.2f} % | t {o['t']:+.2f} | {o['call']}" if o.get("call") else "no call: " + str(o.get("reason"))[:80]))
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T[:40]))
