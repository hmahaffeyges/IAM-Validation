"""DEV-TISSUE-LUNG-01: GSE94785 sample table from GEO metadata only (no IDAT opened). Patient id from the title (e.g. 4T / 4N), kind from the
source field (tumor, normal, control), slide and position from the IDAT file names. Writes gse94785_samples.csv and layout_lung_01_output.txt.
Usage: python3 samples_lung_01.py"""
import os, re, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE94785", "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text.replace("\r", "")
rows = []
for b in t.split("^SAMPLE = ")[1:]:
    g = b.split("\n")[0].strip(); get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
    sup = [u for u in get("!Sample_supplementary_file") if "idat" in u.lower()]; sx = re.search(r"_(\d{10,12})_(R\d\dC\d\d)_", sup[0]) if sup else None
    ch = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
    title = get("!Sample_title")[0]
    rows.append(dict(gsm=g, title=title, patient=re.match(r"^(\d+)", title).group(1), kind=get("!Sample_source_name_ch1")[0].strip().lower(),
                     platform=get("!Sample_platform_id")[0], slide=sx.group(1) if sx else "", pos=sx.group(2) if sx else "",
                     asbestos=ch.get("asbestos exposed", ""), age=ch.get("age (years)", ""), sex=ch.get("gender", "")))
S = pd.DataFrame(rows).sort_values(["kind", "patient"]); S.to_csv(os.path.join(HERE, "gse94785_samples.csv"), index=False)
T = S[S.kind == "tumor"].set_index("patient"); N = S[S.kind == "normal"].set_index("patient"); P = T.index.intersection(N.index)
same = [p for p in P if T.loc[p, "slide"] == N.loc[p, "slide"]]
out = [f"samples {len(S)}: " + ", ".join(f"{k} {v}" for k, v in S.kind.value_counts().items()) + f"; platform {', '.join(S.platform.unique())}",
       f"tumour/normal pairs {len(P)}; same slide {len(same)}; slides {S.slide.nunique()}",
       "per slide (control, normal, tumor): " + "; ".join(f"{s} {int((g.kind == 'control').sum())}/{int((g.kind == 'normal').sum())}/{int((g.kind == 'tumor').sum())}" for s, g in S.groupby("slide"))]
open(os.path.join(HERE, "layout_lung_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
