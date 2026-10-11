"""DEV-TISSUE-BREAST-01 inputs (2026-10-11): TCGA-BRCA open-access 450K IDATs from the NCI GDC (api.gdc.cancer.gov, no login).
list  -> tcga_brca_450k_files.csv: every open IDAT of TCGA-BRCA on 450K (file_id, file_name, md5sum, size, case, sample, sample_type), sorted.
fetch WORKDIR normal -> downloads ONLY sample_type 'Solid Tissue Normal' files into WORKDIR/idat, md5 checked. Tumour files are listed, never
fetched, until the test is sealed (fetch WORKDIR tumour refuses unless sealed_breast_tissue.json is committed).
Usage: python3 fetch_tcga_brca.py list | fetch WORKDIR normal|tumour"""
import os, sys, json, hashlib, subprocess, requests, pandas as pd
import time
def req(f, *a, **k):
    for i in range(6):
        try:
            r = f(*a, **k); r.raise_for_status(); return r
        except Exception as e:
            if i == 5: raise
            time.sleep(5 * (i + 1))
HERE = os.path.dirname(os.path.abspath(__file__)); LIST = os.path.join(HERE, "tcga_brca_450k_files.csv"); API = "https://api.gdc.cancer.gov"
if sys.argv[1] == "list":
    flt = {"op": "and", "content": [{"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
           {"op": "=", "content": {"field": "data_format", "value": "IDAT"}}, {"op": "=", "content": {"field": "access", "value": "open"}},
           {"op": "=", "content": {"field": "platform", "value": "Illumina Human Methylation 450"}}]}
    H = req(requests.post, API + "/files", json={"filters": flt, "fields": "file_id,file_name,md5sum,file_size,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type", "size": 10000, "format": "JSON"}, timeout=120).json()["data"]["hits"]
    T = pd.DataFrame([dict(file_id=h["file_id"], file_name=h["file_name"], md5sum=h["md5sum"], size=h["file_size"], case=h["cases"][0]["submitter_id"],
                           sample=h["cases"][0]["samples"][0]["submitter_id"], sample_type=h["cases"][0]["samples"][0]["sample_type"]) for h in H])
    T.sort_values(["sample_type", "case", "file_name"]).to_csv(LIST, index=False); print(T.groupby("sample_type").size().to_dict())
else:
    W, which = sys.argv[2], sys.argv[3]; T = pd.read_csv(LIST); os.makedirs(os.path.join(W, "idat"), exist_ok=True)
    if which == "tumour":
        seal = os.path.join(HERE, "sealed_breast_tissue.json"); rel = os.path.relpath(seal, subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip())
        assert os.path.exists(seal) and subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0, "tumour files are fetched only after the seal is committed"
        T = T[T.sample_type == "Primary Tumor"]
    else: T = T[T.sample_type == "Solid Tissue Normal"]
    bad = []
    for _, r in T.iterrows():
        p = os.path.join(W, "idat", r.file_name)
        for _ in range(3):
            if os.path.exists(p) and hashlib.md5(open(p, "rb").read()).hexdigest() == r.md5sum: break
            open(p, "wb").write(req(requests.get, f"{API}/data/{r.file_id}", timeout=300).content)
        else: bad.append(r.file_name)
    print(which, len(T), "files", "all md5 verified" if not bad else f"MD5 FAILED: {bad}")
