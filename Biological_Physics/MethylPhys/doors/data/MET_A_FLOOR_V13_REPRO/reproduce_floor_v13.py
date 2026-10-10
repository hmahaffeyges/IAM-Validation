"""Rebuilds the commissioned Met-A neutrophil matrices from public data with the committed builder, chain_tests/freeze_v13.py, unchanged except
input paths (2026-10-10). Inputs: the 6 GSE110554 neutrophil arrays through chain Stage 1 (betas from doors/data/DEV_NOISE_02/build_noise_sites_01.py
'calib'), metA_floors_v1_2 (its EPIC neutrophil block = metA_floors_v1_2_ALLCELLS_development.json), sites_ordered from neutrophil_reference_v1_2.
Compares metA_floors_v1_3.json and metA_floors_v1_3_loo.csv with the chain's files, and the block-10 clustering baseline with neutrophil_reference_v1_2.
Usage: reproduce_floor_v13.py BETAS_DIR WORKDIR"""
import os, sys, json, glob, shutil, subprocess, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
RM = os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors"); BET, W = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "shards"), exist_ok=True)
for f in glob.glob(os.path.join(BET, "GSM*.parquet")): shutil.copy(f, os.path.join(W, "shards", os.path.basename(f).replace(".parquet", "_s1.parquet")))
shutil.copy(os.path.join(RM, "metA_floors_v1_2_ALLCELLS_development.json"), os.path.join(W, "metA_floors_v1_2.json"))
shutil.copy(os.path.join(RM, "neutrophil_reference_v1_2.json"), os.path.join(W, "neutrophil_reference_v1.json"))
src = open(os.path.join(MP, "chain_tests/freeze_v13.py")).read().replace('D="/home/ubuntu/data/atlas_sources/blood/GSE110554/shards"', f'D="{os.path.join(W, "shards")}"')
assert "shards" in src and "/home/ubuntu" not in src.split("import")[1][:0] + src.split("D=")[1][:80]
open(os.path.join(W, "freeze_v13_paths.py"), "w").write(src)
print(subprocess.run([sys.executable, "freeze_v13_paths.py"], cwd=W, capture_output=True, text=True).stdout[-1500:])
a = json.load(open(os.path.join(W, "metA_floors_v1_3.json"))); b = json.load(open(os.path.join(RM, "metA_floors_v1_3.json")))
na, nb = a["platforms"]["EPIC"]["neutrophils"], b["platforms"]["EPIC"]["neutrophils"]
print("floor rebuilt %.8f | chain %.8f | diff %.2e" % (na["floor"], nb["floor"], abs(na["floor"] - nb["floor"])))
print("sites same:", na["sites"] == nb["sites"], "| refs same:", sorted(na["refs"]) == sorted(nb["refs"]), "| precision:", {k: round(v, 6) for k, v in na["precision_heldout"].items() if isinstance(v, float)}, "chain", {k: round(v, 6) for k, v in nb["precision_heldout"].items() if isinstance(v, float)})
la = pd.read_csv(os.path.join(W, "metA_floors_v1_3_loo.csv")); lb = pd.read_csv(os.path.join(RM, "metA_floors_v1_3_loo.csv"))
m = la.merge(lb, left_on="ref", right_on="ref", suffixes=("_new", "_chain")); print(m.round(6).to_string(index=False))
