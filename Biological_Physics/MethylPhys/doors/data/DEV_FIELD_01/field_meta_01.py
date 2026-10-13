"""DEV-FIELD-01: GSE287331 sample list, tissue counts and slide balance, from GEO METADATA ONLY (no IDAT, beta or matrix opened).
Each array's slide (Sentrix barcode) is read from its IDAT file NAME on GEO. Writes gse287331_samples.csv and field_meta_01_output.txt.

# NO CHAIN: metadata.
Usage: python3 field_meta_01.py
"""
import os, re, collections, requests
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE287331", "targ": "gsm", "form": "text", "view": "brief"},
                 timeout=300).text.replace("\r", "")
R = []
for b in t.split("^SAMPLE = ")[1:]:
    get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
    ch = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
    idat = [u for u in get("!Sample_supplementary_file") if ".idat" in u.lower()]
    m = re.search(r"_(\d{9,12})_(R\d\dC\d\d)_", idat[0]) if idat else None
    R.append(dict(gsm=b.split("\n")[0].strip(), title=(get("!Sample_title") or [""])[0], source=(get("!Sample_source_name_ch1") or [""])[0], platform=get("!Sample_platform_id")[0], idat_files=len(idat),
                  slide=m.group(1) if m else "", position=m.group(2) if m else "", **ch))
T = pd.DataFrame(R); T.to_csv(os.path.join(HERE, "gse287331_samples.csv"), index=False)
L = [f"arrays {len(T)} | platforms {sorted(T.platform.unique())} | arrays with both IDATs {int((T.idat_files >= 2).sum())}",
     "tissue counts: " + ", ".join(f"{k} {v}" for k, v in T.tissue.value_counts().items())]
S = T.groupby("slide").tissue.agg(lambda x: tuple(sorted(set(x))))
L.append(f"slides {T.slide.nunique()} | slides holding one tissue only: {int((S.map(len) == 1).sum())} | holding HDB and any patient tissue: "
         f"{int(S.map(lambda x: 'HDB' in x and len(x) > 1).sum())}")
for tis, g in T.groupby("tissue"):
    sl = set(g.slide); shared = sum(1 for s in sl if len(S[s]) > 1)
    L.append(f"  {tis}: {len(g)} arrays on {len(sl)} slides; {shared} of those slides also carry another tissue")
pid = T.title.str.extract(r"(\d+)")[0]; T["patient"] = pid
P = T[T.tissue.isin(["CUB", "OQ", "TU", "AN"])].groupby("patient")
L.append(f"patient ids parsed from titles: {T.patient.notna().sum()} of {len(T)} | example titles: {list(T.title.head(4))}")
L.append(f"patients with CUB+OQ+TU: {sum(1 for _, g in P if {'CUB','OQ','TU'} <= set(g.tissue))} | with AN too: {sum(1 for _, g in P if {'CUB','OQ','TU','AN'} <= set(g.tissue))}"
         f" | of CUB+OQ+TU patients, all three on ONE slide: {sum(1 for _, g in P if {'CUB','OQ','TU'} <= set(g.tissue) and g[g.tissue.isin(['CUB','OQ','TU'])].slide.nunique() == 1)}")
T.to_csv(os.path.join(HERE, "gse287331_samples.csv"), index=False)
L.append("other sample fields: " + ", ".join(c for c in T.columns if c not in ("gsm", "platform", "idat_files", "slide", "position", "tissue")))
open(os.path.join(HERE, "field_meta_01_output.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
