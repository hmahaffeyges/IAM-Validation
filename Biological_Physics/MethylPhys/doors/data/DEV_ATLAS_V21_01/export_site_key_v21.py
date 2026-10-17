"""IAM-Atlas v2.1 site key in sequencing coordinates (author decision 2026-10-16: ship the atlas labeled by Loyfer's own CpG numbering).
For each atlas site (row order of atlas/v2.1/sites.parquet): the Illumina probe ID, the CpG index used in Loyfer et al. 2023's .beta files
(GEO GSE186458; wgbstools numbering, 1-based), and the Illumina manifest chromosome and position (GRCh37/hg19, as the manifest gives them).
Source: atlas/v2/inputs/probe_map_summary.csv.gz. The numbering was checked against Loyfer's own printed marker ranges (Supplementary
Table S12): all 931 of his array markers in the map fall inside his ranges (2026-10-16). Joining only; no value is changed.
Usage: python export_site_key_v21.py"""
import os, json, hashlib, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../.."))
A = os.path.join(R, "Biological_Physics/MethylPhys/atlas/v2.1")
S = pd.read_parquet(os.path.join(A, "sites.parquet"))
P = pd.read_csv(os.path.join(R, "Biological_Physics/MethylPhys/atlas/v2/inputs/probe_map_summary.csv.gz"), usecols=["IlmnID", "chr", "MAPINFO", "cpg_index"]).drop_duplicates("IlmnID")
K = S.merge(P, left_on="cpg", right_on="IlmnID", how="left").drop(columns="IlmnID").rename(columns={"cpg": "probe", "cpg_index": "loyfer_cpg_index", "chr": "chrom_hg19", "MAPINFO": "pos_hg19"})
assert len(K) == len(S) and (K.probe.values == S.cpg.values).all(), "row order must match sites.parquet"
K["loyfer_cpg_index"] = K.loyfer_cpg_index.astype("Int64"); K["pos_hg19"] = K.pos_hg19.astype("Int64")
out = os.path.join(A, "sites_sequencing_key.parquet"); K.to_parquet(out, index=False, compression="zstd")
n = int(K.loyfer_cpg_index.notna().sum()); dup = int(K.loyfer_cpg_index.dropna().duplicated().sum())
print(json.dumps(dict(sites=len(K), with_loyfer_index=n, without=len(K) - n, duplicate_index=dup, sha256=hashlib.sha256(open(out, "rb").read()).hexdigest(), bytes=os.path.getsize(out))))
