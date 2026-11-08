"""Writes CANON/zenodo_files.json: the file list of each of our published Zenodo records that holds data a researcher rebuilds from (2026-11-08,
author: 'EVERYTHING is supposed to be on Zenodo for recreating it by a researcher'), and the folders inside the IAM Sequencing Atlas's runtime
tables archive. CANON/data_homes.py reads it (offline, on every push) to check that every atlas product a record names has a home.
Usage: python3 CANON/zenodo_files.py   (needs the network; run after any deposit)"""
import os, json, tarfile, io, requests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORDS = {"24051846": "IAM Sequencing Atlas v1.2 (with its globe and noise maps)", "24029332": "Sequencing Atlas Globe", "23441999": "IAM-Atlas v2.1",
           "23422688": "IAM-Atlas v2.1 fit outputs"}
RELEASES = {"hg19-bwameth-index-v1": "GitHub release: the hg19 bwa-meth index, its CpG dictionary references/hg19/CpG.bed.gz inside"}
out = dict(records={}, inside={}, releases=RELEASES)
for rid, what in RECORDS.items():
    r = requests.get(f"https://zenodo.org/api/records/{rid}", timeout=120).json()
    out["records"][rid] = dict(what=what, doi=r["doi"], files=sorted(f["key"] for f in r["files"]))
    for f in r["files"]:
        if f["key"] == "seq_atlas_runtime_tables.tgz":   # the folders inside, so a record naming one finds its home
            with requests.get(f["links"]["self"], stream=True, timeout=3600) as g:
                with tarfile.open(fileobj=io.BytesIO(g.content), mode="r:gz") as t:
                    out["inside"][f["key"]] = sorted({n.split("/")[1] if "/" in n else n for n in t.getnames()})
json.dump(out, open(os.path.join(ROOT, "CANON", "zenodo_files.json"), "w", encoding="utf-8"), indent=1); print("zenodo_files: written", {k: len(v["files"]) for k, v in out["records"].items()})
