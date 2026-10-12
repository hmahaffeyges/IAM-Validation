"""DEV-TISSUE-LUNG-02: GSE83842 sample table from GEO metadata only (no IDAT opened). Patient id and kind from the title (e.g. 'S1T' / 'S1N'),
slide and position from the IDAT file names, stage and smoking from the characteristics. Writes gse83842_samples.csv and layout_lung_02_output.txt.
Usage: python3 samples_lung_02.py"""
import os, re, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE83842", "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text.replace("\r", "")
rows = []
for b in t.split("^SAMPLE = ")[1:]:
    g = b.split("\n")[0].strip(); get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
    sup = [u for u in get("!Sample_supplementary_file") if "idat" in u.lower()]; sx = re.search(r"_(\d{10,12})_(R\d\dC\d\d)_", sup[0]) if sup else None
    ch = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
    title = get("!Sample_title")[0]; m = re.search(r"\((\w+?)([TN])\)", title)
    rows.append(dict(gsm=g, title=title, patient=m.group(1), kind={"T": "tumor", "N": "normal"}[m.group(2)], platform=get("!Sample_platform_id")[0],
                     slide=sx.group(1) if sx else "", pos=sx.group(2) if sx else "", stage=ch.get("Stage", ""), smoker=title.split("_")[1].split(" ")[0] if "_" in title else "",
                     age=ch.get("age (yrs)", ""), sex=ch.get("gender", "")))
S = pd.DataFrame(rows).sort_values(["kind", "patient"]); S.to_csv(os.path.join(HERE, "gse83842_samples.csv"), index=False)
T = S[S.kind == "tumor"].set_index("patient"); N = S[S.kind == "normal"].set_index("patient"); P = T.index.intersection(N.index)
same = [p for p in P if T.loc[p, "slide"] == N.loc[p, "slide"]]
out = [f"samples {len(S)}: " + ", ".join(f"{k} {v}" for k, v in S.kind.value_counts().items()) + f"; platform {', '.join(S.platform.unique())}",
       f"tumor/normal pairs {len(P)}; same slide {len(same)}; slides {S.slide.nunique()}; tumor stage " + ", ".join(f"{k} {v}" for k, v in T.stage.value_counts().items()),
       "per slide (normal, tumor): " + "; ".join(f"{s} {int((g.kind == 'normal').sum())}/{int((g.kind == 'tumor').sum())}" for s, g in S.groupby("slide"))]
open(os.path.join(HERE, "layout_lung_02_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
