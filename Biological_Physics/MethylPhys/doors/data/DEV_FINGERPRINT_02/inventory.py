"""DEV-FINGERPRINT-02 inventory: ENCODE HAIB RRBS and 450K experiments for the cancer/normal pairs. Writes encode_inventory.csv.
Usage: python3 inventory.py"""
import os, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
cells = ["LNCaP clone FGC", "epithelial cell of prostate", "MCF-7", "T47D", "mammary epithelial cell", "MCF 10A", "HepG2", "hepatocyte", "A549",
         "epithelial cell of alveolus of lung", "bronchial epithelial cell"]
inv = []
for assay in ("RRBS", "DNAme array"):
    g = requests.get("https://www.encodeproject.org/search/", params={"type": "Experiment", "assay_title": assay, "lab.title": "Richard Myers, HAIB", "format": "json",
        "limit": "all", "biosample_ontology.term_name": cells, "field": ["accession", "biosample_ontology.term_name", "replicates.biological_replicate_number", "files.file_format",
        "files.file_size", "files.read_length", "files.run_type", "files.status"]}, headers={"Accept": "application/json"}, timeout=120).json()["@graph"]
    for x in g:
        fs = [f for f in x.get("files", []) if f.get("status") in ("released", None)]; fq = [f for f in fs if f.get("file_format") == "fastq"]
        inv.append(dict(assay=assay, acc=x["accession"], cell=x["biosample_ontology"]["term_name"], reps=len({r.get("biological_replicate_number") for r in x.get("replicates", [])}),
                        fastq=len(fq), GB=round(sum(f.get("file_size", 0) for f in fq) / 1e9, 1), readlen=sorted({f.get("read_length") for f in fq if f.get("read_length")}),
                        idat=len([f for f in fs if f.get("file_format") == "idat"])))
pd.DataFrame(inv).sort_values(["cell", "assay"]).to_csv(os.path.join(HERE, "encode_inventory.csv"), index=False); print(len(inv))
