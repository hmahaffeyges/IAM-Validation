"""PRJNA929650's clinical table (the publication's Table S1, 10.3389/fonc.2023.1127086, from Europe PMC's open-access supplementary files):
each sample's age, tumor stage, tumor type, number and diameter of tumors and molecular subtype, joined to prjna929650_labels.csv by
SampleID, for the A-score database's cancer type and stage columns. Nothing is typed by hand; "NA" in the table is kept as "not recorded".
Usage: python3 prjna929650_clinical.py OUT.csv"""
import os, sys, io, re, zipfile, requests, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
z = zipfile.ZipFile(io.BytesIO(requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10200909/supplementaryFiles", timeout=300).content))
x = zipfile.ZipFile(io.BytesIO(z.read("Table_1.docx"))).read("word/document.xml").decode()
R = [[re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)] for r in re.findall(r"<w:tr[ >].*?</w:tr>", x, re.S)]
C = ["no", "group", "label", "SampleID", "age", "sex", "tumor_stage", "tumor_type", "metastasis", "tumors", "tumor_diameter", "subtype"]
T = pd.DataFrame([r for r in R if len(r) == 12 and r[0].isdigit()], columns=C)
L = pd.read_csv(os.path.join(H, "prjna929650_labels.csv")); D = L.merge(T.drop(columns=["group", "label"]), on="SampleID", how="left")
for c in ("tumor_stage", "subtype", "tumor_type"): D[c] = D[c].replace({"NA": "not recorded", "": "not recorded"})
D.to_csv(sys.argv[1], index=False)
print(len(T), "table rows;", int(D.tumor_stage.notna().sum()), "of", len(D), "runs matched;", D[D.Label == "Breast cancer"].tumor_stage.value_counts().to_dict())
