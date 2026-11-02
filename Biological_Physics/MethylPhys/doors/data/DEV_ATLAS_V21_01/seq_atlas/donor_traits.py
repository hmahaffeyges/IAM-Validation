"""The IAM Sequencing Atlas's donor traits (checklist A7): each input's recorded characteristics from its GEO sample record (sex, age and
any other trait the submitters recorded), with the cell and sealed donor half; foregrounds are annotated, never subtracted.
Usage: python3 donor_traits.py OUT.csv OUT.txt"""
import os, sys, time, collections, requests, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); I = pd.read_csv(os.path.join(H, "seq_atlas_inputs.csv")); rows = []
for r in I.itertuples():
    for t in range(5):
        try:
            x = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": r.gsm.strip(), "targ": "self", "form": "text", "view": "brief"}, timeout=60)
            if x.ok and "!Sample_characteristics" in x.text: break
        except Exception: pass
        time.sleep(5 * (t + 1))
    ch = {}
    for l in x.text.splitlines():
        if l.startswith("!Sample_characteristics_ch1") and ":" in l:
            k, v = l.split("=", 1)[1].split(":", 1); ch[k.strip().lower()] = v.strip()
    rows.append(dict(gsm=r.gsm.strip(), title=r.title, cell=r.cell.strip(), half=r.half.strip(), **ch)); time.sleep(0.4)
D = pd.DataFrame(rows); D.to_csv(sys.argv[1], index=False)
sx = next((c for c in D.columns if c in ("sex", "gender")), None); ag = next((c for c in D.columns if c.startswith("age")), None)
L = [f"{len(D)} inputs; recorded traits: {', '.join(c for c in D.columns if c not in ('gsm', 'title', 'cell', 'half'))}"]
if sx:
    L.append("sex: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(D[sx].fillna("not recorded")).most_common()))
    m = D.groupby("cell")[sx].apply(lambda s: s.dropna().nunique() > 1); L.append(f"cells whose donors are of both sexes: {int(m.sum())} of {D.cell.nunique()}")
    hh = D.groupby(["cell", "half"])[sx].apply(lambda s: "/".join(sorted(set(s.dropna())))).unstack()
    if "A" in hh and "B" in hh: L.append(f"cells whose two donor halves differ in sex makeup: {int((hh.A.fillna('') != hh.B.fillna('')).sum())}")
if ag:
    a = pd.to_numeric(D[ag].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"); L.append(f"age: {int(a.notna().sum())} recorded, median {a.median():g}, range {a.min():g} to {a.max():g}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
