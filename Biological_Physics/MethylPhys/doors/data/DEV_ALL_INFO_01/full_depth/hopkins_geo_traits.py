"""The Johns Hopkins study's recorded traits (GSE116229): each sample's title, tissue, age and cytogenetics from its GEO sample record, for the
A-score database's cancer type column. Usage: python3 hopkins_geo_traits.py OUT.csv"""
import sys, requests, pandas as pd
t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": "GSE116229", "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text
R = []
for r in t.split("^SAMPLE = ")[1:]:
    L = r.splitlines(); ch = dict(l.split(" = ", 1)[1].split(": ", 1) for l in L if l.startswith("!Sample_characteristics_ch1") and ": " in l)
    R.append(dict(gsm=L[0].strip(), title=next(l.split(" = ", 1)[1] for l in L if l.startswith("!Sample_title")), tissue=ch.get("tissue", ch.get("tissue/cell type", "")),
                  age=ch.get("age (years)", ""), cytogenetics=ch.get("cytogenetics", "")))
pd.DataFrame(R).to_csv(sys.argv[1], index=False); print(len(R), "samples")
