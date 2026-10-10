"""DEV-FINGERPRINT-02: ENCODE HAIB 450K/EPIC IDAT list for the cancer and normal cells (released files only). Writes encode_arrays.csv
(file, experiment, cell, channel, platform, md5, url), sorted. Usage: python3 make_array_list.py"""
import os, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["LNCaP clone FGC", "epithelial cell of prostate", "MCF-7", "T47D", "mammary epithelial cell", "MCF 10A", "HepG2", "hepatocyte", "A549",
         "epithelial cell of alveolus of lung", "bronchial epithelial cell"]
g = requests.get("https://www.encodeproject.org/search/", params={"type": "Experiment", "assay_title": "DNAme array", "lab.title": "Richard Myers, HAIB", "format": "json",
    "limit": "all", "biosample_ontology.term_name": CELLS, "field": ["accession", "biosample_ontology.term_name", "files.accession", "files.file_format", "files.status",
    "files.href", "files.md5sum", "files.output_type", "files.platform.term_name"]}, headers={"Accept": "application/json"}, timeout=120).json()["@graph"]
rows = [dict(file=f["accession"], experiment=x["accession"], cell=x["biosample_ontology"]["term_name"], channel=f.get("output_type"),
             platform=(f.get("platform") or {}).get("term_name"), md5=f["md5sum"], url="https://www.encodeproject.org" + f["href"])
        for x in g for f in x.get("files", []) if f.get("file_format") == "idat" and f.get("status") == "released"]
pd.DataFrame(rows).sort_values(["cell", "experiment", "channel"]).to_csv(os.path.join(HERE, "encode_arrays.csv"), index=False); print(len(rows), "idat files")
