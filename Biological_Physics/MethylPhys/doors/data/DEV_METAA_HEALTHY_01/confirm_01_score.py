"""DEV-METAA-HEALTHY-01 confirmation 1 (written and committed 2026-10-24 before any of its samples is counted). Layer 1 version 2 was
written after version 1 was seen, so it is confirmed here on samples that played no part: the three cord-blood CD19+ B-cell samples of
GSE116229 (another laboratory), counted by the leukemia replication's pass 1 (erase_gh.sh: 10 million read pairs through run_sample.py
--dev-count-only). Version 2's rule unchanged: pooled held errors, per-call rates, prediction 1 - eps_loss at the sites marked at 0.75 or above in
all 79 cell types and eps_gain at the islands unmarked in all 79, against IAM-Atlas v2.1's B-cell level on the sequencing scale. Stated before:
cord blood is newborn, the atlas's B cells adult; the count is at fixed depth (layer 1 read whole files).
Bars: marked side within 0.02; unmarked side within 0.02.  Usage: python3 confirm_01_score.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator as _P
_T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DEV_ATLAS_V21_01", "transfer_v21_final.json")))
G = _P(_T["knots_x"], _T["knots_y"])   # array scale -> sequencing scale
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "confirm_01_summary.txt")
S = pd.DataFrame({"gsm": ["GSM4914800", "GSM4914801", "GSM4914802"], "cell": ["b cells"] * 3}).set_index("gsm"); rows = []   # cord-blood CD19+ B cells, GSE116229
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
T = [f"DEV-METAA-HEALTHY-01 confirmation 1 (cord-blood B cells, GSE116229; atlas on the sequencing scale): {len(X)} of {len(S)} samples counted | sites marked in all 79: {int(marked.sum())} | islands unmarked in all 79: {len(isl)}",
     "cell | samples | eps_loss | predicted marked 1-eps | atlas marked median | gap | eps_gain | predicted unmarked | atlas island median | gap"]
ok_m = ok_u = 0
for c, x in X.groupby("cell"):
    el = per_call((x.held_err * x.held_opp).sum() / x.held_opp.sum()); eg = per_call((x.gain_err * x.gain_opp).sum() / x.gain_opp.sum())
    am, ai = float(G(MU.loc[marked, c].median())), float(G(MU.loc[isl, c].median())); gm, gu = am - (1 - el), ai - eg; ok_m += abs(gm) <= 0.02; ok_u += abs(gu) <= 0.02   # v2: sequencing scale (2026-10-24 fix: this comment had cut off the tally; disclosed)
    T.append(f"{c} | {len(x)} | {el:.4f} | {1 - el:.4f} | {am:.4f} | {gm:+.4f} | {eg:.4f} | {eg:.4f} | {ai:.4f} | {gu:+.4f}")
T.append(f"bar 1 (marked side, |gap| <= 0.02, the three pooled): {'met' if len(X) == 3 and ok_m == 1 else 'not met'}")
T.append(f"bar 2 (unmarked side, same): {'met' if len(X) == 3 and ok_u == 1 else 'not met'}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T))
