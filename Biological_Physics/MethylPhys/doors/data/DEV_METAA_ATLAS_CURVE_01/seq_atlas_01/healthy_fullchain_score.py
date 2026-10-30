"""The healthy full-chain pass on sequencing (written and committed before any of it is read). The 137 healthy sequenced inputs
(healthy_fullchain_samples.csv: each read against the other sealed donor half) read by the one command with the full chain, the Deconvolver
splitting each on the IAM Sequencing Atlas. The instrument's job, for every reading: a healthy sample reads Normal. Reported per reading, not
scored against one bar: how many were read, the median, the share inside the band the chain drew (where it drew one), and the reason for every
reading not run. The purity check: each sample's own-cell share against its depth and Met-A against the share outside its own cell (Spearman).
Usage: python3 healthy_fullchain_score.py LANDED_DIR OUT.txt"""
import sys, os, glob, json, collections, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "healthy_fullchain_samples.csv"))
dep = pd.read_csv(os.path.join(H, "metaa_seqatlas_readings.csv")).set_index("gsm").depth; rows, nr = [], collections.Counter()
def band(v):
    b = v.get("band") if isinstance(v, dict) else None
    return (b.get("lo"), b.get("hi")) if isinstance(b, dict) else (None, None)
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True))
    if not f: rows.append(dict(gsm=r.gsm, cell=r.cell, reading="(not read)", A=None)); continue
    B = json.load(open(f[-1])); D = B.get("development") or {}; cs = B.get("cell_shares") or {}; sh = {x["cell"]: x["share"] for x in cs.get("cells") or []}
    for k in ((B.get("full_chain") or {}).get("not_run") or {}): nr[k] += 1
    def add(name, v):
        if isinstance(v, dict) and v.get("A") is not None: lo, hi = band(v); rows.append(dict(gsm=r.gsm, cell=r.cell, reading=name, A=v["A"], lo=lo, hi=hi))
    add("IAM-A", B.get("iam_a")); add("coupling-A", B.get("coupling_a")); add("Met-A sorted cell", D.get("metaa_atlas")); add("Met-A whole sample", B.get("met_a_whole"))
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): add("territory " + k, v)
    w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if w.get("C") is not None: rows.append(dict(gsm=r.gsm, cell=r.cell, reading="C-Score whole genome", A=w["C"], lo=None, hi=w.get("limit")))
    rows.append(dict(gsm=r.gsm, cell=r.cell, reading="own-cell share", A=sh.get(r.cell), lo=None, hi=None, reference=cs.get("reference")))
d = pd.DataFrame(rows); d.to_csv(os.path.join(sys.argv[1], "healthy_fullchain_readings.csv"), index=False)
L = [f"Healthy full-chain pass on sequencing: {d[d.reading != '(not read)'].gsm.nunique()} of {len(S)} read", "reading | read | median | inside the band drawn"]
for k, g in d[d.reading != "(not read)"].groupby("reading"):
    a = pd.to_numeric(g.A, errors="coerce"); lo = pd.to_numeric(g.get("lo"), errors="coerce"); hi = pd.to_numeric(g.get("hi"), errors="coerce")
    inb = ((lo.isna() | (a >= lo)) & (hi.isna() | (a <= hi)))[lo.notna() | hi.notna()]
    L.append(f"{k} | {a.notna().sum()} | {a.median():.4f} | " + (f"{inb.mean():.3f} of {len(inb)}" if len(inb) else "no band drawn"))
o = d[d.reading == "own-cell share"].set_index("gsm"); m = d[d.reading == "Met-A sorted cell"].set_index("gsm")
o = o.join(dep, how="left"); x = o.join(m.A.rename("metaa"), how="inner")
if len(x) > 2:
    L.append(f"purity: own-cell share median {pd.to_numeric(o.A).median():.4f}; own share against depth rho {spearmanr(o.A, o.depth, nan_policy='omit').correlation:+.3f}; "
             f"Met-A against the share outside its own cell rho {spearmanr(1 - x.A, x.metaa, nan_policy='omit').correlation:+.3f} over {len(x)}")
L.append("not run (samples): " + (", ".join(f"{k} {v}" for k, v in sorted(nr.items())) or "none"))
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
