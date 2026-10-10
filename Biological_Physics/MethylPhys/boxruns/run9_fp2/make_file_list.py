"""Box Run 9 (DEV-FINGERPRINT-02) file list: released ENCODE HAIB RRBS fastq files for a set of cell types, with md5.
Usage: python3 make_file_list.py normal|cancer   -> normal_rrbs_files.csv | cancer_rrbs_files.csv"""
import os, sys, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {"normal": ["epithelial cell of prostate", "hepatocyte", "epithelial cell of alveolus of lung", "bronchial epithelial cell", "mammary epithelial cell", "MCF 10A"],
        "cancer": ["LNCaP clone FGC", "HepG2", "A549", "MCF-7", "T47D"]}
k = sys.argv[1]; rows = []
g = requests.get("https://www.encodeproject.org/search/", params={"type": "Experiment", "assay_title": "RRBS", "lab.title": "Richard Myers, HAIB", "format": "json", "limit": "all",
    "biosample_ontology.term_name": SETS[k], "field": ["accession", "biosample_ontology.term_name", "files.accession", "files.file_format", "files.status", "files.md5sum",
    "files.href", "files.read_length", "files.file_size", "files.biological_replicates"]}, headers={"Accept": "application/json"}, timeout=120).json()["@graph"]
for x in g:
    for f in x.get("files", []):
        if f.get("file_format") == "fastq" and f.get("status") == "released":
            rows.append(dict(file=f["accession"], experiment=x["accession"], cell=x["biosample_ontology"]["term_name"], rep=(f.get("biological_replicates") or [None])[0],
                             read_length=f.get("read_length"), md5=f["md5sum"], url="https://www.encodeproject.org" + f["href"], GB=round(f["file_size"] / 1e9, 2)))
pd.DataFrame(rows).sort_values(["cell", "experiment", "rep"]).to_csv(os.path.join(HERE, f"{k}_rrbs_files.csv"), index=False); print(len(rows))
