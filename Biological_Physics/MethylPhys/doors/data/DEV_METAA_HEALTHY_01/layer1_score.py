"""DEV-METAA-HEALTHY-01 layer 1 scorer (written and committed before any file is counted). Reads only the bundles count_atlas_01.py kept and
IAM-Atlas v2.1. For each of the 56 cell types: the counted lone-error rates r (held_err on the loss side, gain_err on the gain side; pooled
over its samples by opportunities) turned into per-call rates by r = eps (1 - eps)^2; the predicted healthy levels, 1 - eps_loss where the
mark is held and eps_gain where it is held absent; and the atlas's median level for that cell (a) at the sites marked at 0.75 or above in all
79 cell types and (b) at the islands unmarked in all 79 (stage_metaa_atlas.territory_sets). The gap is atlas minus prediction.
Bar 2: |gap| <= 0.02 on the marked side in at least 45 of the 56 cell types. Bar 3: the same on the unmarked side.
Usage: python3 layer1_score.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "layer1_summary.txt")
S = pd.read_csv(os.path.join(H, "layer1_samples.csv")).set_index("gsm"); rows = []
for f in glob.glob(os.path.join(src, "**", "GSM*_bundle.json"), recursive=True):
    g = os.path.basename(f).split("_")[0]; c = (json.load(open(f)).get("iam_a_count_only") or {}).get("count") or {}
    if g in S.index and c.get("held_opp") and c.get("gain_opp"): rows.append(dict(gsm=g, cell=S.loc[g, "cell"], **{k: c[k] for k in ("held_err", "held_opp", "gain_err", "gain_opp")}))
X = pd.DataFrame(rows)
def per_call(r):   # r = eps (1 - eps)^2, the smaller root
    from scipy.optimize import brentq
    return brentq(lambda e: e * (1 - e) ** 2 - r, 0, 1 / 3) if 0 < r < 4 / 27 else np.nan
AT = os.path.join(H, "..", "..", "..", "atlas", "v2.1"); MU = []
for n in (1, 2, 3):
    t = pq.read_table(os.path.join(AT, f"mu_part{n}.parquet")); MU.append(pd.DataFrame({c: t.column(c).to_numpy() for c in t.column_names}))
ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist(); MU = pd.concat(MU, axis=1); MU.index = ids
MU = MU.where(MU != 65535) / 10000.0; full = MU.notna().all(1); marked = full & (MU.min(1) >= 0.75)
isl = pd.Index(SA.territory_sets()["islands"]).intersection(MU.index)
T = [f"DEV-METAA-HEALTHY-01 layer 1: {len(X)} of {len(S)} samples counted, {X.cell.nunique() if len(X) else 0} cell types | sites marked in all 79: {int(marked.sum())} | islands unmarked in all 79: {len(isl)}",
     "cell | samples | eps_loss | predicted marked 1-eps | atlas marked median | gap | eps_gain | predicted unmarked | atlas island median | gap"]
ok_m = ok_u = 0
for c, x in X.groupby("cell"):
    el = per_call((x.held_err * x.held_opp).sum() / x.held_opp.sum()); eg = per_call((x.gain_err * x.gain_opp).sum() / x.gain_opp.sum())
    am, ai = float(MU.loc[marked, c].median()), float(MU.loc[isl, c].median()); gm, gu = am - (1 - el), ai - eg; ok_m += abs(gm) <= 0.02; ok_u += abs(gu) <= 0.02
    T.append(f"{c} | {len(x)} | {el:.4f} | {1 - el:.4f} | {am:.4f} | {gm:+.4f} | {eg:.4f} | {eg:.4f} | {ai:.4f} | {gu:+.4f}")
n = X.cell.nunique() if len(X) else 0
T.append(f"bar 2 (marked side, |gap| <= 0.02 in at least 45 of 56): {ok_m} of {n}: {'met' if n == 56 and ok_m >= 45 else 'not met'}")
T.append(f"bar 3 (unmarked side, same): {ok_u} of {n}: {'met' if n == 56 and ok_u >= 45 else 'not met'}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T))
