"""DEV-SELFTARE-03, the anchor range on real arrays (2026-10-21, exploration, not sealed). Healthy whole bloods of GSE318173 (Group-A's controls,
the arrays after the first 60 by slide and position), each calibrated by Stage 1 unchanged and read by the chain's conductor for Met-A and its
neutrophil share; then the same array with a loss of the neutrophils' pattern at every site of 2 % and 5 % (beta + f_neu x p x (0.5 - mu_NEU),
IAM-Atlas v2.1's healthy neutrophil). For each case the anchor range R = U_II - L_II (dev_stages.anchors). The scorer tares each R against the
healthy readings of the other arrays on its slide. Usage: python3 anchor_range_real_01.py OUTDIR [N]. Writes OUTDIR/anchor_range_real_01_rows.csv."""
import os, sys, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", "..")); CH = os.path.join(MP, "chain")
sys.path[:0] = [CH, os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING")]
import conductor_v3 as C, dev_stages as DV, stage_1_idat_calibration as S1, stage_deconv_v21 as D
from read_groupa import fetch
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 100; os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING", "groupa_samples.csv")).sort_values(["slide", "position"]).iloc[60:60 + N]
MU = D.reference(["neutrophils"])["mu"]["neutrophils"]; ST = DV._json("dev_selftare_typeII_EPIC_v1.json"); rows = []
for r in S.itertuples():
    try:
        g = fetch(r.gsm, r.grn, os.path.join(out, r.grn)); d = fetch(r.gsm, r.red, os.path.join(out, r.red))
        beta, _ = S1.calibrate_idat_to_beta(g, d, verbose=False); os.remove(g); os.remove(d)
        beta = pd.Series(beta, dtype="float64"); beta.index = beta.index.astype(str)
        o = C.run_neutrophil(beta, specimen="whole blood"); f = (o.get("composition") or {}).get("fractions", {}).get("NEU") or 0.0
        m = MU.reindex(beta.index)
        for p in (0.0, 0.02, 0.05):
            b = (beta + (f * p * (0.5 - m)).fillna(0.0)).clip(1e-6, 1 - 1e-6) if p else beta
            a = DV.anchor_range(b, f, None, ST); A = o["met_a"].get("A") if not p else C.run_neutrophil(b, specimen="whole blood")["met_a"].get("A")
            rows.append(dict(gsm=r.gsm, slide=str(r.slide), sex=r.sex, planted_pct=100 * p, f_neu=f, R=a.get("R"), A=A))
    except Exception as e:
        rows.append(dict(gsm=r.gsm, slide=str(r.slide), planted_pct=-1, R=None, A=None, error=str(e)[:200]))
    print(r.gsm, rows[-1].get("R"), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "anchor_range_real_01_rows.csv"), index=False); print("rows", len(rows))
