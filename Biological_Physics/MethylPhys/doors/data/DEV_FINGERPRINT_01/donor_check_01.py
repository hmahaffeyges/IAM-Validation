"""DEV-FINGERPRINT-01 against donor variation (2026-10-10; the sealed verdict is not changed). Reads the committed LNCaP rows of this test and the
DEV-FINGERPRINT-02 donor null; writes donor_check_01_output.json: LNCaP min/max Arm B excess, the largest healthy donor-to-donor Arm B excess, the
largest Met-A distance from curve B between the two healthy donors, and the ratio. Usage: python3 donor_check_01.py"""
import os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); N = json.load(open(os.path.join(HERE, "../DEV_FINGERPRINT_02/donor_null_02_output.json")))
R = pd.read_csv(os.path.join(HERE, "fingerprint_01_rows.csv")); L = R[R.cell == "LNCaP"].excess_run_loss
dmax = max(v for o in N.values() for v in o["cancer_excess"].values()); mdist = max(abs(o["MetA_rel"] - o["curveB"]) for o in N.values())
out = dict(lncap_excess_min=round(float(L.min()), 4), lncap_excess_max=round(float(L.max()), 4), donor_excess_max=round(dmax, 4),
           donor_metA_minus_curveB_max=round(mdist, 4), ratio_lncap_min_to_donor_max=round(float(L.min()) / dmax, 1))
json.dump(out, open(os.path.join(HERE, "donor_check_01_output.json"), "w"), indent=1); print(out)
