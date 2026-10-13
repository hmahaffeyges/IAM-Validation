"""DEV-BREACH-01: GSE281307 (DCIS, progressed vs not) sample list, groups and slide balance, from GEO METADATA ONLY (no array opened).
Slide = Sentrix barcode in each IDAT file name. Writes gse281307_samples.csv and breach_meta_01_output.txt.

# NO CHAIN: metadata.
"""
import os, re, requests
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE281307", "targ": "gsm", "form": "text", "view": "brief"},
                 timeout=300).text.replace("\r", "")
R = []
for b in t.split("^SAMPLE = ")[1:]:
    get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
    ch = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
    idat = [u for u in get("!Sample_supplementary_file") if ".idat" in u.lower()]
    m = re.search(r"_(\d{9,12})_(R\d\dC\d\d)_", idat[0]) if idat else None
    R.append(dict(gsm=b.split("\n")[0].strip(), title=(get("!Sample_title") or [""])[0], platform=get("!Sample_platform_id")[0],
                  idat_files=len(idat), slide=m.group(1) if m else "", position=m.group(2) if m else "", **ch))
T = pd.DataFrame(R); T.to_csv(os.path.join(HERE, "gse281307_samples.csv"), index=False)
L = [f"arrays {len(T)} | platform {sorted(T.platform.unique())} | with both IDATs {int((T.idat_files >= 2).sum())}",
     "fields: " + ", ".join(c for c in T.columns if c not in ("gsm", "title", "platform", "idat_files", "slide", "position")),
     "classification: " + str(T.get("classification", pd.Series(dtype=str)).value_counts().to_dict())]
if "classification" in T:
    S = T.groupby("slide").classification.agg(lambda x: tuple(sorted(set(x))))
    L.append(f"slides {T.slide.nunique()} | slides with both groups {int(S.map(len).gt(1).sum())} | one group only {int(S.map(len).eq(1).sum())}")
    for c, g in T.groupby("classification"): L.append(f"  {c}: {len(g)} arrays on {g.slide.nunique()} slides")
for c in T.columns:
    if c not in ("gsm", "title", "slide", "position", "idat_files", "platform", "classification") and T[c].nunique() < 15:
        L.append(f"{c}: {T[c].value_counts().to_dict()}")
if {"radiationtreatment", "hormonetreatment", "classification"} <= set(T.columns):
    U = T[(T.radiationtreatment == "Untreated") & (T.hormonetreatment == "Untreated")]
    L.append(f"untreated by both radiation and hormones: {U.classification.value_counts().to_dict()} on {U.slide.nunique()} slides; "
             f"slides holding both outcome groups {int(U.groupby('slide').classification.agg(lambda x: {'Progressor', 'Non-progressor'} <= set(x)).sum())}")
open(os.path.join(HERE, "breach_meta_01_output.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
