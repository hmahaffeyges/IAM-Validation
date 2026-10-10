"""Rebuilds blood_composition_EPIC_v1.json with the committed builder chain_tests/blood_comp.py, unchanged except input paths (2026-10-10).
Inputs: the 91 Salas purified arrays (GSE110554, GSE167998) through chain Stage 1 (../DEV_NOISE_02/build_noise_sites_01.py calib), the atlas v2
sample roster, metA_floors_v1_2 (neutrophil sites). The 24-mixture test (needs the mixture arrays) is skipped; the build code is unchanged. Compares every key the chain reads. Usage: reproduce_blood_composition.py BETAS_DIR WORKDIR"""
import os, sys, json, glob, shutil, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
RM = os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors"); BET, W = sys.argv[1], sys.argv[2]; sh = os.path.join(W, "shards"); os.makedirs(sh, exist_ok=True)
for f in glob.glob(os.path.join(BET, "GSM*.parquet")): shutil.copy(f, os.path.join(sh, os.path.basename(f).replace(".parquet", "_s1.parquet")))
shutil.copy(os.path.join(RM, "metA_floors_v1_2_ALLCELLS_development.json"), os.path.join(W, "metA_floors_v1_2.json"))
shutil.copy(os.path.join(MP, "atlas/v2/inputs/roster_samples.csv"), os.path.join(W, "roster_samples.csv"))
src = open(os.path.join(MP, "chain_tests/blood_comp.py")).read()
a = 'D={"Salas2018":f"{R}/blood/GSE110554/shards","Salas2022":f"{R}/blood/GSE167998/shards"}'; assert src.count(a) == 1
src = src.replace(a, f'D={{"Salas2018":"{sh}","Salas2022":"{sh}"}}')
# The 24-mixture test runs before the matrix is written and needs the mixture arrays; it is skipped here (build code unchanged).
i = src.index('T=pd.read_csv("salas_mixture_truth.csv")'); j = src.index('FROZ={')
src = src[:i] + 'FULL=build(set(D)); print("markers per group (both studies):",FULL["n_markers"],"total",len(FULL["markers"]),flush=True)\n' + src[j:]; open(os.path.join(W, "blood_comp_paths.py"), "w").write(src)
r = subprocess.run([sys.executable, "blood_comp_paths.py"], cwd=W, capture_output=True, text=True); print(r.stdout[-1200:], r.stderr[-800:])
A = json.load(open(os.path.join(W, "blood_composition_EPIC_v1.json"))); B = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json")))
for k in ("groups", "neutrophil_sites", "group_of_cell", "rule"): print(k, "identical:", A.get(k) == B.get(k))
print("markers identical:", A["markers"] == B["markers"], "| per group:", {g: (len(A["markers"][g]), len(B["markers"][g]), A["markers"][g] == B["markers"][g]) for g in B["markers"]} if isinstance(B["markers"], dict) else (len(A["markers"]), len(B["markers"])))
def mx(a, b):
    if isinstance(b, dict): return max(mx(a[k], b[k]) for k in b)
    return float(np.nanmax(np.abs(np.array(a, dtype=float) - np.array(b, dtype=float))))
for k in ("mu_markers", "profiles_at_neutrophil_sites"): print(k, "max abs diff", mx(A[k], B[k]))
