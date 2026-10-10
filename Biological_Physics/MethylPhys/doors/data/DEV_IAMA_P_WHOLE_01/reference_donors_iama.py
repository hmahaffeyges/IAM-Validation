"""The three healthy granulocyte donors behind IAM-A's neutrophil position P (iama_positions_v2.json; chain_tests/iama_floor_granulocytes.csv):
donor record from GEO (GSE186458 sample characteristics). Writes reference_donors_iama.csv. Usage: python3 reference_donors_iama.py"""
import os, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
G = pd.read_csv(os.path.join(MP, "chain_tests/iama_floor_granulocytes.csv")).gsm.tolist(); rows = []
for g in G:
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "self", "form": "text", "view": "brief"}, timeout=60).text
    ch = {l.split("=", 1)[1].split(":")[0].strip(): l.split(":", 1)[1].strip() for l in t.split("\n") if l.startswith("!Sample_characteristics") and ":" in l}
    rows.append(dict(gsm=g, cell=ch.get("cell type"), lab=ch.get("lab"), sex=ch.get("Sex"), age=int(ch.get("age")))); time.sleep(0.3)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "reference_donors_iama.csv"), index=False); print(R.to_string(index=False))
print(f"sex {R.sex.value_counts().to_dict()} | age {R.age.min()}-{R.age.max()} | labs {sorted(R.lab.unique())}")
