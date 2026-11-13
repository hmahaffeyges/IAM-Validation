"""PHY-084, the referee test of the healthy baseline: the scorer, written and committed before any library is read.

Where IAM's Law is: the candidate (CANON eps0_candidate), the loss of a mark held against thermal kicks by two coins per restore, from M
alone; plus the eraser's measured share (a measured input the law does not set), between the two measured ratios: 0.8642 (all erasers,
stem cells, DEV-ERASE-01 recount) and 0.905 (TET2 in blood progenitors, PHY-085, mean of the two units).

The prediction is put through the meter, never the meter through the prediction: for each cell type, the counter's lone-loss reading
(held_err_length) that the forward model (forward_candidate.py, seq mode, read by the one command) gives at that cell type's own measured
clustering (median coupling of its libraries), interpolated linearly on the grid, at both ratios. The observation: the cell type's
zero-loose intercept of held_err_length against loose_share (Theil-Sen, 2,000 resamples of libraries, seed 20261113), as PHY-050.

Bar (GUM, k = 2), relative to the prediction's middle: between-cell-type spread of the healthy hold 8.8 % (celltype_net.csv, 55 types,
sd of log held error); the eraser band as a uniform distribution (half-width over sqrt 3); forward-model sampling, the grid's own
replicate spread; copier lag 0.4 % (lag_bound_blood.py); the intercept's own resampled standard error. Met if every cell type's
intercept lies within its bar; not met if any lies outside.

Usage: python3 referee_score.py COUNTS_DIR SAMPLES_CSV GRID_DIR OUT_PREFIX [REFUSED_CSV]
  COUNTS_DIR: one count JSON per library (erase_gh.sh's *_count.json form: gsm, held_err_length, loose_share, coupling)
  SAMPLES_CSV: columns gsm, cell
  GRID_DIR: forward_candidate.py outputs named pg_<ratio>_<coupling-param>.csv
"""
import os, sys, json, glob, re, math
import numpy as np, pandas as pd
from scipy.stats import theilslopes

H = os.path.dirname(os.path.abspath(__file__))
RATIOS = (0.8642, 0.905)
BETWEEN_TYPES = 0.0878   # celltype_net.csv: sd of log held error over 55 healthy types (computed 2026-11-12; recomputed below if present)
LAG = 0.004
SEED, NBOOT = 20261113, 2000


def between_types():
    p = os.path.join(H, "..", "PHY_050_PERCALL", "celltype_net.csv")
    if os.path.exists(p):
        T = pd.read_csv(p); return float(np.std(np.log(T.held0), ddof=1))
    return BETWEEN_TYPES


def grid(gdir):
    G = {}
    for f in glob.glob(os.path.join(gdir, "pg_*.csv")):
        m = re.match(r"pg_([\d.]+)_([\d.]+)\.csv$", os.path.basename(f))
        if not m: continue
        X = pd.read_csv(f); G.setdefault(float(m.group(1)), []).append((X.coupling.mean(), X.held_err_length.mean(), X.held_err_length.std(ddof=1) / X.held_err_length.mean()))
    out = {}
    for r, pts in G.items():
        sp = [p[2] for p in pts]
        if any(not np.isfinite(x) for x in sp): raise SystemExit(f"ratio {r}: a grid point has fewer than two replicates, so no sampling spread")
        pts.sort(); out[r] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), float(np.mean(sp)))
    missing = [r for r in RATIOS if r not in out]
    if missing: raise SystemExit(f"grid lacks ratios {missing}")
    return out


def predict(G, r, coupling):
    c, v, _ = G[r]
    if not (c.min() <= coupling <= c.max()): raise SystemExit(f"coupling {coupling:.3f} outside the grid ({c.min():.3f}-{c.max():.3f}) at ratio {r}: extend the grid, do not extrapolate")
    return float(np.interp(coupling, c, v))


REFUSED = None


def load(cdir, scsv):
    S = pd.read_csv(scsv); S["cell"] = S.cell.str.lower().str.strip(); R = []
    for f in sorted(glob.glob(os.path.join(cdir, "*_count.json"))):
        c = json.load(open(f)); R.append(dict(gsm=c["gsm"], obs=c["held_err_length"], loose=c["loose_share"], coupling=c["coupling"]))
    # 2026-11-13, amendment 3 (before scoring): libraries the chain refused at intake are reported, not read by another route (sealed rule);
    # they are named in REFUSED_CSV with their exit code and dropped from the list; any other missing count still stops the scorer
    if REFUSED is not None: S = S[~S.gsm.isin(REFUSED.gsm)]
    X = pd.DataFrame(R).merge(S, on="gsm", how="inner")
    if len(X) != len(S): raise SystemExit(f"{len(S) - len(X)} libraries in the sample list have no count")
    return X


def main(cdir, scsv, gdir, out):
    X = load(cdir, scsv); G = grid(gdir); bt = between_types(); rng = np.random.default_rng(SEED); rows = []
    for cell, g in X.groupby("cell"):
        s, b, _, _ = theilslopes(g.obs, g.loose)
        bb = np.array([theilslopes(g.obs.values[ix], g.loose.values[ix])[1] for ix in (rng.integers(0, len(g), len(g)) for _ in range(NBOOT))])
        cp = float(g.coupling.median()); p = [predict(G, r, cp) for r in RATIOS]
        mid, half = (p[0] + p[1]) / 2, abs(p[0] - p[1]) / 2
        u_er = half / mid / math.sqrt(3); u_fm = max(G[r][2] for r in RATIOS) / math.sqrt(3); u_ic = float(np.std(bb, ddof=1)) / mid
        u = math.sqrt(bt ** 2 + u_er ** 2 + u_fm ** 2 + LAG ** 2 + u_ic ** 2)
        rows.append(dict(cell=cell, n=len(g), coupling=round(cp, 3), intercept=round(b, 5), int_lo95=round(np.percentile(bb, 2.5), 5), int_hi95=round(np.percentile(bb, 97.5), 5),
                         pred_ratio_0864=round(p[0], 5), pred_ratio_0905=round(p[1], 5), pred_mid=round(mid, 5),
                         u_between=round(bt, 4), u_eraser=round(u_er, 4), u_forward=round(u_fm, 4), u_lag=LAG, u_intercept=round(u_ic, 4), u_combined=round(u, 4),
                         bar_lo=round(mid * (1 - 2 * u), 5), bar_hi=round(mid * (1 + 2 * u), 5), pull=round((b - mid) / (u * mid), 2), inside=bool(abs(b - mid) <= 2 * u * mid)))
    O = pd.DataFrame(rows); O.to_csv(out + ".csv", index=False)
    met = bool(O.inside.all())
    t = (f"PHY-084 referee test: {'MET' if met else 'NOT MET'} ({int(O.inside.sum())} of {len(O)} cell types inside their bars, k = 2)\n" + O.to_string(index=False) + "\n")
    open(out + "_summary.txt", "w").write(t); print(t); return met


if __name__ == "__main__":
    if len(sys.argv) > 5:
        REFUSED = pd.read_csv(sys.argv[5]); assert set(REFUSED.exit) == {35}, "only intake refusals (exit 35) may be dropped"
        print("refused at intake, reported, not read:", ", ".join(f"{g} ({c})" for g, c in zip(REFUSED.gsm, REFUSED.cell)))
    main(*sys.argv[1:5])
