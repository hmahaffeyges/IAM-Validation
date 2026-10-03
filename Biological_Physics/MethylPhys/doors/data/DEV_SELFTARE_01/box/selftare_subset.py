#!/usr/bin/env python3
"""DEV-SELFTARE-01 step 1b (box): betas at the noise sites + identity sites + composition markers for every calibrated array,
the untared chain records, and the EPIC v1 probe design (GPL21145) for those sites."""
import json, os, gzip, io, sys
import pandas as pd
OUT = sys.argv[1]; RM = sys.argv[2]
ns = json.load(open(f"{RM}/noise_sites_EPIC_v1.json"))["sites"]; bc = json.load(open(f"{RM}/blood_composition_EPIC_v1.json"))
fl = json.load(open(f"{RM}/metA_floors_v1_3.json"))["platforms"]["EPIC"]["neutrophils"]["sites"]
sites = pd.Index(sorted(set(ns) | set(bc["neutrophil_sites"]) | set(bc["markers"]) | set(fl)))
jobs = pd.read_csv(os.path.join(OUT, "jobs.csv")); cols, recs, miss = {}, [], []
for g in jobs["gsm"].astype(str):
    fb = os.path.join(OUT, "betas", g + ".parquet"); fr = os.path.join(OUT, "rec", g + ".json")
    if os.path.exists(fb) and os.path.exists(fr):
        cols[g] = pd.read_parquet(fb)["beta"].reindex(sites).astype("float32"); recs.append(json.load(open(fr)))
    else: miss.append(g)
print("arrays", len(cols), "missing", miss[:10], flush=True)
M = pd.DataFrame(cols, index=sites); M.index.name = "site"; M.to_parquet("beta_subset.parquet")
pd.DataFrame(recs).to_csv("untared_records.csv", index=False)
p = "/home/ubuntu/data/G_chain_tests/GSE250556/idat/GPL21145_MethylationEPIC_15073387_v-1-0.csv.gz"
lines = gzip.open(p, "rt").read().splitlines()
i = [k for k, l in enumerate(lines) if l.startswith("IlmnID")][0]; j = [k for k, l in enumerate(lines) if l.startswith("[Controls]")][0]
man = pd.read_csv(io.StringIO("\n".join(lines[i:j])), usecols=["IlmnID", "Infinium_Design_Type", "Color_Channel"], low_memory=False).set_index("IlmnID")
man.reindex(sites).to_csv("probe_design.csv"); print(M.shape, len(recs))
