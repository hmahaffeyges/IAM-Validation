"""DEV-TISSUE-BREAST-01 (descriptive; normal tissue only): what the spread of null_normals_output.csv tracks. Plate = TCGA aliquot barcode
field 6 (GDC cases.samples.portions.analytes.aliquots.submitter_id for each normal file); array quality = probes passing Stage 1; composition =
fitted fractions. Writes spread_sources_output.txt. Usage: python3 spread_sources.py BETADIR"""
import os, sys, time, requests, numpy as np, pandas as pd
from scipy.stats import spearmanr, kruskal
HERE = os.path.dirname(os.path.abspath(__file__)); BD = sys.argv[1]
T = pd.read_csv(os.path.join(HERE, "null_normals_output.csv")); F = pd.read_csv(os.path.join(HERE, "tcga_brca_450k_files.csv")); F = F[F.sample_type == "Solid Tissue Normal"]
for i in range(6):
    try:
        r = requests.post("https://api.gdc.cancer.gov/files", json={"filters": {"op": "in", "content": {"field": "file_id", "value": F.file_id.tolist()}},
              "fields": "file_id,cases.samples.submitter_id,cases.samples.portions.analytes.aliquots.submitter_id", "size": 1000, "format": "JSON"}, timeout=120); r.raise_for_status(); break
    except Exception:
        time.sleep(5 * (i + 1))
al = {}
for h in r.json()["data"]["hits"]:
    s = h["cases"][0]["samples"][0]; al[s["submitter_id"]] = s["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"]
T["aliquot"] = T["sample"].map(al); T["plate"] = T.aliquot.str.split("-").str[5]
T["probes"] = [int(pd.read_parquet(os.path.join(BD, s + ".parquet")).beta.notna().sum()) for s in T["sample"]]
out = [f"plates {T.plate.nunique()} (normals per plate: {T.plate.value_counts().to_dict()})"]
g = [x.comp_rel.values for _, x in T.groupby("plate") if len(x) >= 3]
out.append(f"comp_rel by plate (plates with >= 3 normals: {len(g)}): Kruskal-Wallis p = {kruskal(*g).pvalue:.2e}" if len(g) >= 2 else "too few plates with >= 3 normals")
out.append("plate medians of comp_rel: " + str(T.groupby("plate").comp_rel.median().round(3).to_dict()))
for c in ["probes"] + [c for c in T.columns if c.startswith("f_")]:
    rho, p = spearmanr(T[c], T.comp_rel); out.append(f"comp_rel vs {c}: Spearman {rho:+.3f} (p {p:.1e})")
open(os.path.join(HERE, "spread_sources_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
T.to_csv(os.path.join(HERE, "null_normals_output.csv"), index=False)
