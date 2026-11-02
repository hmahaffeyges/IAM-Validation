"""The Johns Hopkins leukemia study read by the full working chain on sequencing (written and committed before any of it is read).
The 36 samples (hopkins_fullchain_samples.csv: 31 B-cell leukemias, 3 healthy cord-blood B-cell samples, the Reh line and its UHRF1 knockout),
each merged from its saved alignment pieces and read by the one command with the healthy passes' sequencing settings (the rsflags column),
against the IAM Sequencing Atlas: the Deconvolver's shares, Met-A in the B-cell column and for the whole sample, the territories, the
C-Score, coupling-A, IAM-A, the globe and the information sphere. No bar: the readings are reported as the chain gives them, with the
healthy band where the chain draws one; experts in the disease set the lines. Per sample and per group (genotype): every reading.
Usage: python3 hopkins_fullchain_table.py LANDED_DIR OUT.csv OUT.txt"""
import sys, os, glob, json, collections, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "hopkins_fullchain_samples.csv")); rows, nr = [], collections.Counter()
for r in S.itertuples():
    f = sorted(x for x in glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True))
    o = dict(gsm=r.gsm, title=r.title, genotype=r.genotype)
    if not f: rows.append(o); continue
    B = json.load(open(f[-1])); D = B.get("development") or {}; cs = B.get("cell_shares") or {}; sh = sorted(((x["share"], x["cell"]) for x in cs.get("cells") or []), reverse=True)
    for k in ((B.get("full_chain") or {}).get("not_run") or {}): nr[k] += 1
    def put(name, v):
        if isinstance(v, dict) and v.get("A") is not None:
            o[name] = v["A"]; b = v.get("band") if isinstance(v.get("band"), dict) else {}
            if b: o[name + " inside band"] = bool((b.get("lo") is None or v["A"] >= b["lo"]) and (b.get("hi") is None or v["A"] <= b["hi"]))
    put("IAM-A", B.get("iam_a")); put("coupling-A", B.get("coupling_a")); put("Met-A B cells", D.get("metaa_atlas")); put("Met-A whole sample", B.get("met_a_whole"))
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): put("territory " + k, v)
    w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if w.get("C") is not None: o["C-Score"] = w["C"]; o["C-Score inside limit"] = bool(w.get("within"))
    g = B.get("globe") or {}; s = B.get("info_sphere") or {}
    o.update({"globe sites changed": g.get("sites_changed"), "territories changed": s.get("territories_changed"), "Fisher-Rao from own healthy self": s.get("fisher_rao_rms_rad"),
              "B-cell share": dict((c, x) for x, c in sh).get("b cells"), "largest share": f"{sh[0][1]} {sh[0][0]:.3f}" if sh else None, "reference": cs.get("reference")})
    rows.append(o)
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
num = [c for c in d.columns if c not in ("gsm", "title", "genotype", "largest share", "reference") and not c.endswith(("inside band", "inside limit"))]
L = [f"Johns Hopkins leukemia study, full working chain on sequencing: {d[num].notna().any(axis=1).sum()} of {len(S)} read", "median by group (n):"]
for k, g in d.groupby("genotype"):
    L.append(f"  {k} ({len(g)}): " + "; ".join(f"{c} {pd.to_numeric(g[c], errors='coerce').median():.4f}" for c in num if pd.to_numeric(g[c], errors="coerce").notna().any()))
L.append("not run (samples): " + (", ".join(f"{k} {v}" for k, v in sorted(nr.items())) or "none"))
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
