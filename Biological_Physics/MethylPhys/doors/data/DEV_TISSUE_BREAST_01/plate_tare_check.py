"""DEV-TISSUE-BREAST-01 (normal tissue values + tumour METADATA only; no tumour array read). (1) Spread of the normals after a same-plate tare:
each normal's comp and plain divided by the median of the OTHER normals on its plate (plates with >= 3 normals). (2) For the 91 patients with tumour
and normal, whether the two aliquots share a plate (GDC aliquot barcodes; file metadata only). Writes plate_tare_check_output.txt.
Usage: python3 plate_tare_check.py"""
import os, time, requests, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(HERE, "null_normals_output.csv")); out = []
for c in ("plain", "comp"):
    v = []
    for _, g in T.groupby("plate"):
        if len(g) < 3: continue
        x = g[c].values; v += [x[i] / np.median(np.delete(x, i)) for i in range(len(x))]
    v = np.array(v); out.append(f"{c} after same-plate tare: n {len(v)}, SD {v.std(ddof=1):.4f}, 95th percentile {np.quantile(v, .95):.4f}")
F = pd.read_csv(os.path.join(HERE, "tcga_brca_450k_files.csv")); F = F[F.sample_type == "Primary Tumor"]; ids = F.file_id.tolist(); al = {}
for k in range(0, len(ids), 500):
    for i in range(6):
        try:
            r = requests.post("https://api.gdc.cancer.gov/files", json={"filters": {"op": "in", "content": {"field": "file_id", "value": ids[k:k + 500]}},
                  "fields": "cases.submitter_id,cases.samples.portions.analytes.aliquots.submitter_id", "size": 1000, "format": "JSON"}, timeout=120); r.raise_for_status(); break
        except Exception: time.sleep(5 * (i + 1))
    for h in r.json()["data"]["hits"]: al.setdefault(h["cases"][0]["submitter_id"], set()).add(h["cases"][0]["samples"][0]["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"].split("-")[5])
T["case"] = T["sample"].str[:12]; P = T[T.case.isin(al)]
same = sum(r.plate in al[r.case] for _, r in P.iterrows())
out.append(f"patients with tumour and normal: {len(P)}; tumour on the same plate as its normal: {same}")
tp = pd.Series([p for s in al.values() for p in s]).value_counts()
out.append(f"tumour aliquots on plates that also hold >= 3 normals: {int(tp[tp.index.isin(T.plate.value_counts()[lambda s: s >= 3].index)].sum())} of {int(tp.sum())}")
open(os.path.join(HERE, "plate_tare_check_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
