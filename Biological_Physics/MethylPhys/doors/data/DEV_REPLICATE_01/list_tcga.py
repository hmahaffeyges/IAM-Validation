"""DEV-REPLICATE-01 wave C (2026-11-07), metadata only: every open-access 450K IDAT pair of one TCGA project at the NCI GDC (api.gdc.cancer.gov,
no login), written as a samples list the series workflow reads (run_series.py fetches each pair from its grn_url and red_url and checks its
md5). One row per sample: gsm (the TCGA sample barcode, the array's id in the chain), slide, the two file URLs and md5s, case, and the GDC
sample type (Primary Tumor, Solid Tissue Normal, ...) kept as the label for scoring only. No IDAT is downloaded here.
Usage: python3 list_tcga.py TCGA-COAD OUT_SAMPLES.csv OUT_LABELS.csv"""
import sys, time, requests, pandas as pd
API = "https://api.gdc.cancer.gov"; proj, oc, ol = sys.argv[1:4]
def req(*a, **k):
    for i in range(8):
        try: r = requests.post(*a, **k); r.raise_for_status(); return r
        except Exception:
            if i == 7: raise
            time.sleep(min(120, 10 * 2 ** i))
flt = {"op": "and", "content": [{"op": "=", "content": {"field": "cases.project.project_id", "value": proj}},
       {"op": "=", "content": {"field": "data_format", "value": "IDAT"}}, {"op": "=", "content": {"field": "access", "value": "open"}},
       {"op": "=", "content": {"field": "platform", "value": "Illumina Human Methylation 450"}}]}
F = "file_id,file_name,md5sum,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"
H = req(API + "/files", json={"filters": flt, "fields": F, "size": 10000, "format": "JSON"}, timeout=120).json()["data"]["hits"]
rows = {}
for h in H:
    s = h["cases"][0]["samples"][0]; key = h["file_name"].rsplit("_", 1)[0]; ch = "grn" if h["file_name"].endswith("_Grn.idat") else "red"
    r = rows.setdefault(key, dict(gsm=s["submitter_id"], case=h["cases"][0]["submitter_id"], group=s["sample_type"], array=key))
    r[ch + "_url"] = f"{API}/data/{h['file_id']}"; r[ch + "_md5"] = h["md5sum"]
T = pd.DataFrame([r for r in rows.values() if "grn_url" in r and "red_url" in r]).sort_values(["group", "gsm"])
d = T.gsm.duplicated(keep=False); T.loc[d, "gsm"] = T.loc[d, "gsm"] + "_" + T.loc[d, "array"].str[:8]   # a sample run on two arrays keeps both
T["slide"] = "NA"
T[["gsm", "slide", "grn_url", "grn_md5", "red_url", "red_md5"]].to_csv(oc, index=False)
T[["gsm", "case", "group"]].to_csv(ol, index=False)
print(proj, len(T), "arrays |", T.group.value_counts().to_dict(), "| cases with tumor and normal:",
      len(set(T[T.group == "Primary Tumor"].case) & set(T[T.group == "Solid Tissue Normal"].case)))
