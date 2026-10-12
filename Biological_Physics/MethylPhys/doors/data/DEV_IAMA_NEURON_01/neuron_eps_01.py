"""DEV-IAMA-NEURON-01: Loyfer GSE186458 neuron .pat files (hg19) read whole by chain Stage Q (stage_q_iam_a.pat_site_table: copy error =
isolated errors / opportunities) and Stage Q0 (stage_q0_intake.intake; Q0.7 specimen not applied: neurons are not a commissioned specimen).
One row per file, appended as each finishes (resumable). Donor fields from loyfer_neuron_donors.csv (Loyfer et al. 2023, Supplementary Table S1).
Usage: python3 neuron_eps_01.py CHAIN_DIR PAT_DIR OUT.csv"""
import os, sys, glob, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, sys.argv[1])
import stage_q_iam_a as Q, stage_q0_intake as Q0
pat_dir, OUT = sys.argv[2], sys.argv[3]
Dn = pd.read_csv(os.path.join(HERE, "loyfer_neuron_donors.csv")).set_index("Sample name")
done = set(pd.read_csv(OUT)["sample"]) if os.path.exists(OUT) else set()
for f in sorted(glob.glob(os.path.join(pat_dir, "*.pat.gz"))):
    smp = os.path.basename(f).split(".")[0]
    if smp in done: continue
    T = Q.pat_site_table(f); o = float(T.opp_A.sum() + T.opp_B.sum()); e = float(T.err_A.sum() + T.err_B.sum()) / o
    I = Q0.intake(f, "isolated neutrophils"); bad = [c["step"] for c in I["checks"] if c["result"] == "REFUSED" and c["step"] != "Q0.7 specimen"]
    name = smp.split("_", 1)[1]; d = Dn.loc[name]
    row = dict(sample=smp, eps=round(e, 6), opportunities=int(o), q0=",".join(bad) or "PROCEED", donor=d["PatientID"], age=d["age"], sex=d["sex"],
               region=name.split("-Neuron")[0] if "-Neuron" in name else "unlabelled", sorted=("not sorted" not in str(d["Comments"])),
               purity=d["Estimated purity (Table S10)"], death=d["Diagnosis"] if isinstance(d["Diagnosis"], str) else "")
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False); print(row, flush=True)
