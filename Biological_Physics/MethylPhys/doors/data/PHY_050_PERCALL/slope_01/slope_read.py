"""PHY-087 step 1, the reading (exploration on spent samples; 2026-11-13). Question, written before counting (slope_tiles.py): is IAM-A's
baseline slope set by MIXING inside a sample (weakly held tiles lose more) or by CELL TYPE (flat inside samples, the slope only across them)?

Inputs: out tiles (<GSM>_tiles.csv.gz, chain Stage Q2 --dev-count-tiles, two independent halves per tile) for the 11 samples in
slope_samples.csv. The tile's looseness x = 1 - mean_call (the molecules' mean call in the tile). Null test first (the cosmologist's
cross-spectrum): across tiles with >= 200 opportunities per half, the two halves' held errors must correlate above 0.7 in every sample.
Then, in the same units of x:
  WITHIN: each sample's slope of tile held error on x, the held error read from half B against x read from all calls (half A is the
          replicate check), weighted by opportunities; the median over samples.
  ACROSS: the slope of each sample's held error (all its tiles pooled) on its mean x (opportunity-weighted), over the 11 samples.
MIXING predicts WITHIN close to ACROSS; CELL TYPE predicts WITHIN near zero. Writes slope_read_summary.txt."""
import os, glob, csv, numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
TD = os.environ.get("TILES_DIR", os.path.join(H, "out"))
# 2026-11-14: SAMPLES / SUMMARY / SKIP_MISSING let the same reading run on a second laboratory (ref_tiles.py); defaults unchanged
S = {r["gsm"]: r for r in csv.DictReader(open(os.environ.get("SAMPLES", os.path.join(H, "slope_samples.csv"))))}
rows, null, within = [], {}, {}
for g in S:
    if os.environ.get("SKIP_MISSING") and not os.path.exists(os.path.join(TD, f"{g}_tiles.csv.gz")): continue
    T = pd.read_csv(os.path.join(TD, f"{g}_tiles.csv.gz"), dtype={"chrom": str})
    AGG = int(os.environ.get("TILE_AGG", "1"))   # 2026-11-14: pool AGG consecutive tiles (for shallower libraries); 1 = as committed
    if AGG > 1:
        T["meth"] = T.calls * T.mean_call.fillna(0); T = T.assign(tile=T.tile // AGG).groupby(["chrom", "tile"], as_index=False)[["iso_a", "opp_a", "iso_b", "opp_b", "calls", "meth"]].sum()
        T["mean_call"] = (T.meth / T.calls).where(T.calls > 0)
    X = T[(T.opp_a >= 200) & (T.opp_b >= 200) & T.mean_call.notna()].copy()
    a, b = X.iso_a / X.opp_a, X.iso_b / X.opp_b
    null[g] = float(np.corrcoef(a, b)[0, 1])
    x = 1 - X.mean_call
    w = X.opp_b.values
    within[g] = float(np.polyfit(x, b, 1, w=np.sqrt(w))[0])
    within_a = float(np.polyfit(x, a, 1, w=np.sqrt(X.opp_a.values))[0])
    opp = X.opp_a + X.opp_b
    rows.append(dict(gsm=g, cell=S[g]["cell"], loose_share=float(S[g].get("loose_share") or "nan"), tiles=len(X), halves_r=round(null[g], 3),
                     x_mean=float((x * opp).sum() / opp.sum()), held=float((X.iso_a + X.iso_b).sum() / opp.sum()),
                     within_b=within[g], within_a=within_a))
D = pd.DataFrame(rows)
ok = bool((D.halves_r > 0.7).all())
across = float(np.polyfit(D.x_mean, D.held, 1)[0])
HAS_L = bool(D.loose_share.notna().all())
across_L = float(np.polyfit(D.loose_share, D.held, 1)[0]) if HAS_L else float("nan")
wmed, wlo, whi = float(D.within_b.median()), float(D.within_b.min()), float(D.within_b.max())
ratio = wmed / across
L = [f"null test (halves' correlation > 0.7 in every sample): {'PASSED' if ok else 'FAILED'} (min {D.halves_r.min():.3f})",
     f"WITHIN samples: median slope {wmed:.4f} (range {wlo:.4f} to {whi:.4f}); replicate half A median {D.within_a.median():.4f}",
     f"ACROSS samples: slope {across:.4f} per unit tile looseness (11 samples); on the counter's loose share: {across_L:.4f} (IAM-A's b 0.034625)",
     f"WITHIN / ACROSS = {ratio:.2f}  (MIXING predicts near 1; CELL TYPE near 0)", "", D.round(5).to_string(index=False)]
for lab, E in (((("all 11", D), ("the 10 that passed the null test (after the fact)", D[D.halves_r > 0.7]))) if HAS_L else ()):
    dx = float(np.polyfit(E.loose_share, E.x_mean, 1)[0]); bm = float(E.within_b.median()) * dx
    L.insert(4, f"MIXING-PREDICTED b, {lab}: within slope {E.within_b.median():.4f} x d(mean tile looseness)/d(loose share) {dx:.4f} = {bm:.4f},"
                f" {bm / 0.034625:.2f} of IAM-A's b; the rest is cell type (A8)")
open(os.environ.get("SUMMARY", os.path.join(H, "slope_read_summary.txt")), "w").write("\n".join(L) + "\n")
print("\n".join(L))
