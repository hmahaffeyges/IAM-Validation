"""DEV-METAA-HEALTHY-01, the per-person rehearsal read again with --dev-pred-raw (written and committed 2026-11-07 before any of its readings exist;
series run wb100_raw): the healthy prediction and its A on each array's own values, not on the atlas ruler. Refuses a bundle whose prediction
was not read on the array's own values. Below, the first rehearsal's description and bars, unchanged:
The per-person rehearsal on spent data (written and committed before any of its readings exist). The 100 spent healthy
whole bloods of bar 2 (bar2_labels.csv: 20 from each of five laboratories), each read again by run_sample.py on every atlas cell (series run
wb100_atlas). Reference: IAM-Atlas v2.1 only, in each blood's own Deconvolver mix; no same-run tare, no controls. Each person is read on its own.
Bars (Met-A, the healthy line): 1. at least 90 of the 100 read A / A_pred inside Normal, 0.95 to 1.05; 2. in each of the five laboratories at
least 16 of its 20 do. Recorded, not scored: untared A against the atlas, each blood's measured line, the cell shares, and the C-Score's
whole-genome reading against the atlas with its derived limit (its own synthetic comes first, plan 60.29.2).
Usage: python3 wb100_score.py FOLDER_OF_BUNDLES OUT.csv OUT.txt"""
import sys, os, glob, json, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); src, oc, ot = sys.argv[1:4]
L = pd.read_csv(os.path.join(H, "bar2_labels.csv"), dtype=str).set_index("gsm"); rows, seen = [], set(); REF = []
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    g = os.path.basename(f).split(".")[0].split("_")[0]
    if g in seen or g not in L.index or f.endswith(("_values.json", "_planted.json")): continue
    seen.add(g); B = json.load(open(f)); w = B.get("met_a_whole") or {}; h = w.get("healthy_prediction") or {}
    if not str(h.get("values") or "").startswith("the array's own values"): REF.append(g); continue
    c = ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}); lc = B.get("label_check") or {}
    rows.append(dict(gsm=g, A=w.get("A"), A_pred=h.get("A_pred"), ratio=h.get("A_over_pred"), blur=json.dumps(h.get("blur")),
                     C=c.get("C"), C_limit=c.get("limit"), C_within=c.get("within"), chain_cell=lc.get("chain_cell"),
                     refusal=B.get("refusal")))
X = pd.DataFrame(rows, columns=["gsm", "A", "A_pred", "ratio", "blur", "C", "C_limit", "C_within", "chain_cell", "refusal"]).set_index("gsm").join(L, how="right")
X["normal"] = (pd.to_numeric(X.ratio, errors="coerce") - 1).abs() <= 0.05
X.to_csv(oc)
n = int(X.normal.sum()); per = X.groupby("lab").normal.sum().astype(int)
T = [f"DEV-METAA-HEALTHY-01 per-person rehearsal: {int(X.ratio.notna().sum())} of {len(X)} healthy bloods read (reference IAM-Atlas v2.1 only)",
     f"Met-A A / A_pred inside 0.95 to 1.05: {n} of {len(X)}"] + [f"  {lab}: {k} of {int((X.lab == lab).sum())}" for lab, k in per.items()]
a = pd.to_numeric(X.A, errors="coerce"); T.append(f"recorded: untared A inside 0.95 to 1.05: {int(((a - 1).abs() <= 0.05).sum())} of {int(a.notna().sum())}")
cw = X.C_within.dropna(); T.append(f"recorded: C-Score whole genome within its derived limit: {int((cw == True).sum())} of {len(cw)}")
b1 = n >= 90; b2 = len(per) == 5 and (per >= 16).all()
T += [f"bar 1 (at least 90 of 100 inside Normal): {'met' if b1 else 'not met'}", f"bar 2 (at least 16 of 20 in each laboratory): {'met' if b2 else 'not met'}",
      "rehearsal: " + ("met" if b1 and b2 else "not met")]
T.insert(1, f"refused (prediction not on the array's own values): {len(REF)}" + (": " + ", ".join(sorted(REF)[:10]) if REF else ""))
open(ot, "w").write("\n".join(T) + "\n"); print("\n".join(T))
