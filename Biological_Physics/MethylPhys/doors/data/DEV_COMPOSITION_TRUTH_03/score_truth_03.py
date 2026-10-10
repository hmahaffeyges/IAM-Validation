"""DEV-COMPOSITION-TRUTH-03 scoring, as written in doors/DEV_COMPOSITION_TRUTH_03.md before reading (2026-10-09).
Inputs: Stage 1 betas of the GSE224807 paired arrays (run_calibrate.py; manifest.csv), one parquet per array in BETAS (default
../../../../../../pair224807/betas, or s3://.../results/DEV_COMPOSITION_TRUTH_03/betas_<GPL>.parquet, sha256 in inputs_sha256.json);
the atlas v2 parquet (s3://.../atlas_v2/IAMAtlas_v2.parquet). Run from anywhere: python3 score_truth_03.py BETAS ATLAS_PARQUET OUT.csv"""
import os, sys, glob, json, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); CHAIN = os.path.abspath(os.path.join(HERE, "../../../chain")); sys.path.insert(0, CHAIN); os.chdir(CHAIN)
import dev_stages as DS, conductor_v3 as C3
BETAS, ATLAS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
M = pd.read_csv(os.path.join(HERE, "manifest.csv")); CELLS = ["CD15", "CD14", "CD19", "CD4", "CD56", "CD8"]; K = 6000
def beta(g):
    b = pd.read_parquet(os.path.join(BETAS, f"{g}.parquet")).beta; b.index = b.index.astype(str); return b
rows = []
for gpl, P in M.groupby("gpl"):
    B = {g: beta(g) for g in P.gsm}
    T_ = {}; tare = {}
    for g, b in B.items():
        bt, info = DS.selftare_map(b); T_[g] = bt if bt is not None else b
        tare[g] = {d: (m or {}).get("slope") for d, m in info.get("maps", {}).items()}
    sorted_ = P[P.cell != "WB"]; Tmpl = pd.DataFrame({c: pd.DataFrame({g: T_[g] for g in sorted_[sorted_.cell == c].gsm}).mean(1) for c in CELLS}).dropna()
    inf = Tmpl.std(1).sort_values(ascending=False).index[:K]; A = Tmpl.loc[inf]
    for _, r in P[P.cell == "WB"].iterrows():
        y = T_[r.gsm].reindex(inf); ok = y.notna(); f, _ = nnls(A[ok].values, y[ok].values); f = f / f.sum()
        truth = dict(zip(CELLS, f)); b = B[r.gsm]
        ae = DS.atlas_e(b, ATLAS); af = ae.get("fractions") or {}
        sa = C3.stage_a_composition(b); sf = sa.get("fractions") or {}
        gran = lambda d: sum(d.get(k, 0.0) for k in ("NEU", "EOS", "BASO")) if d else None
        rows.append(dict(gpl=gpl, person=r.person, gsm=r.gsm, truth_neu=truth["CD15"], truth_sum_sites=int(ok.sum()),
                         atlas_e_gran=gran(af), atlas_e_status=ae.get("status"), stageA_gran=gran(sf) if sf else None,
                         stageA_reason=sa.get("reason"), stageA_markers=sa.get("n_markers_used"), tare_I=tare[r.gsm].get("I"), tare_II=tare[r.gsm].get("II"),
                         **{f"truth_{c}": truth[c] for c in CELLS[1:]}))
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
for gpl, g in R.groupby("gpl"):
    for col in ("atlas_e_gran", "stageA_gran"):
        e = (g[col] - g.truth_neu).dropna()
        if len(e): print(f"{gpl} {col:13s} n {len(e):2d} | MAE {e.abs().mean():.4f} | max |err| {e.abs().max():.4f} | within 0.05 {int((e.abs() <= 0.05).sum())}/{len(e)} | bias {e.mean():+.4f}")
        else: print(f"{gpl} {col:13s} not read: {g.stageA_reason.dropna().iloc[0] if col == 'stageA_gran' and g.stageA_reason.notna().any() else 'no values'}")
print("bars: MAE <= 0.02, every blood within 0.05, not worse than the current method (Stage A)")
