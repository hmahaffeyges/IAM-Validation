"""DEV-FINGERPRINT-02 breast_hmec review (descriptive): per MCF-7 file, its ENCODE experiment, source and treatment (ENCODE API), with its sealed
Arm B excess and Arm G gain (scored_breast_hmec.json, scored_gain_breast_hmec.json). Writes mcf7_experiments_output.csv. Usage: python3 mcf7_experiments.py"""
import os, json, time, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); FP = os.path.join(HERE, "..")
F = pd.read_csv(os.path.join(FP, "../../../boxruns/run9_fp2/cancer_rrbs_files.csv")); F = F[F.cell == "MCF-7"]
B = json.load(open(os.path.join(FP, "scored_breast_hmec.json"))); G = json.load(open(os.path.join(FP, "scored_gain_breast_hmec.json")))
info = {}
for a in sorted(F.experiment.unique()):
    e = requests.get(f"https://www.encodeproject.org/experiments/{a}/", params={"format": "json"}, headers={"Accept": "application/json"}, timeout=60).json()
    bs = e["replicates"][0]["library"]["biosample"]; info[a] = (e.get("description", ""), "; ".join(t.get("treatment_term_name", "") for t in bs.get("treatments", []))); time.sleep(0.3)
F["description"] = F.experiment.map(lambda a: info[a][0]); F["treatment"] = F.experiment.map(lambda a: info[a][1])
F["armB_excess"] = F.file.map(B["cancer_excess"]); F["armG_gain"] = F.file.map(G["cancer_gain"])
F.drop(columns=["url", "md5"]).to_csv(os.path.join(HERE, "mcf7_experiments_output.csv"), index=False); print(F[["file", "experiment", "treatment", "armB_excess", "armG_gain"]].round(4).to_string(index=False))
