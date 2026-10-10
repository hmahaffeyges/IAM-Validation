"""DEV-LINK-IAMA-METAA-02, Met-A side: HCT116 decitabine dose series on EPIC v1 (GSE237553), read by the note's rule (2026-10-09).
Written and committed 2026-10-10 before any EM-seq (IAM-A) reading of the series exists.
Betas: chain Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta) on the GEO IDATs; sha256 of each IDAT printed.
Identity sites, chosen on the two vehicle arrays only: vehicle SD <= 0.05 and vehicle mean beta in 0.75-0.95 (high) or 0.05-0.25 (low);
up to 3,000 each, taken in order of smallest vehicle SD (ties by probe name). This ordering is the implementation of 'up to 3,000', fixed here.
Met-A_rel = mean H(beta) over the identity sites / the mean of the two vehicles. Allowance = the two vehicles' own readings around 1.
Also printed per array: identity-site mean beta (high and low sets): the note's rule tests the relation only where they stay on their side of 0.5.
Run: HOME=<dir with .methylprep_manifest_files> python3 metaa_dose_02.py IDAT_DIR OUT.csv"""
import os, sys, glob, hashlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../chain")))
from stage_1_idat_calibration import calibrate_idat_to_beta
IDAT, OUT = sys.argv[1], sys.argv[2]
# Doses from the GEO sample records (GSM7623726-31 titles HCT116_DMSO_1/2, HCT116_DAC30_1/2, HCT116_DAC300_1/2), not from the IDAT file names.
ARR = {"GSM7623726_Veh1": "vehicle", "GSM7623727_Veh2": "vehicle", "GSM7623728_DAC301": "DAC 30 nM", "GSM7623729_DAC302": "DAC 30 nM",
       "GSM7623730_DAC3001": "DAC 300 nM", "GSM7623731_DAC3002": "DAC 300 nM"}
def H(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
B = {}
for a in ARR:
    g = os.path.join(IDAT, f"{a}_Grn.idat"); r = g.replace("_Grn", "_Red")
    print(a, "sha256 Grn", hashlib.sha256(open(g, "rb").read()).hexdigest()[:16], "Red", hashlib.sha256(open(r, "rb").read()).hexdigest()[:16], flush=True)
    o = calibrate_idat_to_beta(g, r, verbose=False); B[a] = (o[0] if isinstance(o, tuple) else o).astype(float)
X = pd.DataFrame(B).dropna(); V = X[[a for a, t in ARR.items() if t == "vehicle"]]; mu, sd = V.mean(1), V.std(1, ddof=1)
def pick(mask): c = pd.DataFrame({"sd": sd[mask]}).rename_axis("p").reset_index().sort_values(["sd", "p"]); return c.p.head(3000).tolist()
hi = pick((sd <= 0.05) & mu.between(0.75, 0.95)); lo = pick((sd <= 0.05) & mu.between(0.05, 0.25)); S = hi + lo
ref = float(np.mean([H(X.loc[S, v]).mean() for v in V.columns]))
rows = [dict(array=a, treatment=t, MetA_rel=float(H(X.loc[S, a]).mean() / ref), hi_mean_beta=float(X.loc[hi, a].mean()),
             lo_mean_beta=float(X.loc[lo, a].mean()), n_hi=len(hi), n_lo=len(lo)) for a, t in ARR.items()]
R = pd.DataFrame(rows); R.to_csv(OUT, index=False); print(R.round(4).to_string(index=False))
print(R.groupby("treatment", sort=False)[["MetA_rel", "hi_mean_beta", "lo_mean_beta"]].mean().round(4).to_string())
