"""The six physical arrays of the Met-A neutrophil healthy reference (metA_floors_v1_3.json 'refs'): donor record from GEO (GSE110554 sample
characteristics). Writes reference_donors.csv (sex, age, FACS purity, smoker). Usage: python3 reference_donors.py"""
import os, re, json, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
F = json.load(open(os.path.join(HERE, "../../../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
rows = []
for r in F["refs"]:
    g = re.search(r"GSM\d+", r).group(0)
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "self", "form": "text", "view": "brief"}, timeout=60).text
    ch = {l.split("=", 1)[1].split(":")[0].strip(): l.split(":", 1)[1].strip() for l in t.split("\n") if l.startswith("!Sample_characteristics") and ":" in l}
    rows.append(dict(gsm=g, array=r.split("_", 1)[1], cell=ch.get("cell type"), sex=ch.get("Sex"), age=int(ch.get("age")), purity_pct=int(ch.get("purity")), smoker=ch.get("smoker"))); time.sleep(0.3)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "reference_donors.csv"), index=False); print(R.to_string(index=False))
print(f"sex {R.sex.value_counts().to_dict()} | age {R.age.min()}-{R.age.max()} | purity {R.purity_pct.min()}-{R.purity_pct.max()} % | smokers {int((R.smoker == 'Yes').sum())}")
