"""DEV-FINGERPRINT-03 candidates (liver and lung), metadata only: no read, .pat, beta or IDAT opened. Re-queries ENA (runs, layout, read length,
FASTQ bytes) for the sequencing candidates and GEO (sample count, IDAT links) for the array candidates listed in candidates_search_2026-10-12.json.
Writes verify_candidates_output.txt. Usage: python3 verify_candidates.py"""
import os, io, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); out = []
SEQ = ["PRJNA762641", "PRJNA984754", "PRJNA287622", "PRJNA1198121", "PRJNA1103867", "PRJNA315696", "PRJNA649023"]
ARR = ["GSE235414", "GSE158422", "GSE94785", "GSE83842", "GSE114989", "GSE136319", "GSE89852", "GSE136380", "GSE77269", "GSE157341"]
for a in SEQ:
    try:
        t = requests.get("https://www.ebi.ac.uk/ena/portal/api/filereport", params={"accession": a, "result": "read_run", "format": "tsv",
                         "fields": "run_accession,library_layout,read_count,base_count,fastq_bytes"}, timeout=120).text
        d = pd.read_csv(io.StringIO(t), sep="\t"); tb = d.fastq_bytes.astype(str).str.split(";").apply(lambda x: sum(int(v) for v in x if v.isdigit())).sum() / 1e12
        out.append(f"{a}: {len(d)} runs, layout {d.library_layout.value_counts().to_dict()}, bases per read pair (median) {int((d.base_count / d.read_count).median())}, FASTQ {tb:.2f} TB")
    except Exception as e: out.append(f"{a}: ENA query failed ({str(e)[:60]})")
    time.sleep(1)
for g in ARR:
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "gsm", "form": "text", "view": "brief"}, timeout=120).text
    n = sum(l.startswith("!Sample_title") for l in t.split("\n")); idat = sum(l.startswith("!Sample_supplementary_file") and "idat" in l.lower() for l in t.split("\n"))
    out.append(f"{g}: {n} samples, {idat} IDAT links"); time.sleep(0.5)
open(os.path.join(HERE, "verify_candidates_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
