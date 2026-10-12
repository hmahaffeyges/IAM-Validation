"""DEV-TISSUE-LIVER-01 inputs (2026-10-12), metadata only: every open-access TCGA-LIHC 450K IDAT from the NCI GDC (api.gdc.cancer.gov), with
case, sample, sample type and the aliquot barcode, whose fourth field is the processing plate. Writes tcga_lihc_450k_files.csv and
layout_lihc_output.txt (patients with one tumor and one normal sample, and how many share a plate). No IDAT is downloaded.
Usage: python3 list_tcga_lihc.py"""
import os, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); API = "https://api.gdc.cancer.gov"
def req(*a, **k):
    for i in range(8):
        try: r = requests.post(*a, **k); r.raise_for_status(); return r
        except Exception:
            if i == 7: raise
            time.sleep(min(120, 10 * 2 ** i))
flt = {"op": "and", "content": [{"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LIHC"}},
       {"op": "=", "content": {"field": "data_format", "value": "IDAT"}}, {"op": "=", "content": {"field": "access", "value": "open"}},
       {"op": "=", "content": {"field": "platform", "value": "Illumina Human Methylation 450"}}]}
F = "file_id,file_name,md5sum,file_size,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type,cases.samples.portions.analytes.aliquots.submitter_id"
H = req(API + "/files", json={"filters": flt, "fields": F, "size": 10000, "format": "JSON"}, timeout=120).json()["data"]["hits"]
rows = []
for h in H:
    s = h["cases"][0]["samples"][0]; al = [a["submitter_id"] for p in s.get("portions", []) for an in p.get("analytes", []) for a in an.get("aliquots", [])]
    rows.append(dict(file_id=h["file_id"], file_name=h["file_name"], md5sum=h["md5sum"], size=h["file_size"], case=h["cases"][0]["submitter_id"],
                     sample=s["submitter_id"], sample_type=s["sample_type"], aliquot=al[0] if al else "", plate=al[0].split("-")[5] if al and len(al[0].split("-")) > 5 else ""))
T = pd.DataFrame(rows).sort_values(["sample_type", "case", "file_name"]); T.to_csv(os.path.join(HERE, "tcga_lihc_450k_files.csv"), index=False)
S = T.drop_duplicates("sample"); tu = S[S.sample_type == "Primary Tumor"]; no = S[S.sample_type == "Solid Tissue Normal"]
P = sorted(set(tu.case) & set(no.case)); one = [c for c in P if (tu.case == c).sum() == 1 and (no.case == c).sum() == 1]
same = [c for c in one if tu[tu.case == c].plate.iloc[0] == no[no.case == c].plate.iloc[0]]
out = [f"files {len(T)}; samples by type " + ", ".join(f"{k} {v}" for k, v in S.sample_type.value_counts().items()),
       f"cases with tumor and normal {len(P)}; with exactly one of each {len(one)}; same plate {len(same)}; plates {S.plate.nunique()}"]
open(os.path.join(HERE, "layout_lihc_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
