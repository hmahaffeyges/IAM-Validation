"""DEV-METAA-HEALTHY-01 step 3 version 4, the scorer (written and committed 2026-10-24 before any reading exists). Built on version 3
(scored; its marked side matched the arrays, its whole gap was the unmarked side). At each neutrophil identity site, the array-against-
sequencing difference measured on OTHER cells (v4_read.py's readings): d = array mean - the sequencing mean put on the array scale by the
atlas's own inverse transfer, per cell, using only cells nearly unmarked there (sequencing mean at most 0.25) for the unmarked side and
nearly marked (at least 0.75) for the marked side; the site's d is the median over those cells (a site with none takes its side's median).
Version 4's level at each site = version 3's level for its side (the cell's own molecules, step3_03/step3_v3_summary.txt) + d, clipped to
(0, 1); the mark = the mean entropy over the sites. Nothing fitted; no neutrophil or granulocyte array is used. Bar 1 (version 1's): all six
commissioned arrays read Normal (0.95-1.05) against the derived mark.  Usage: python3 v4_score.py READINGS_DIR OUT.txt"""
import sys, os, re, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); MP = os.path.join(CH, "..")
sys.path.insert(0, CH); import stage_provisional as SPV
from scipy.interpolate import PchipInterpolator
T = json.load(open(SPV.SMOOTH)); g = PchipInterpolator(T["knots_x"], T["knots_y"]); gx = np.linspace(0, 1, 200001); gv = g(gx) + 1e-12 * gx; inv = lambda v: np.interp(np.clip(v, 0, 1), gv, gx)
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
import pyarrow.parquet as pq
AT = os.path.join(MP, "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
u = next(pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy() for n in (1, 2, 3) if "neutrophils" in pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet")).schema.names)
lv = pd.Series(u, index=ids); st = lv.reindex(F["sites"]); st = st[st.notna() & (st != 65535)] / 1e4; side = pd.Series(np.where(st >= 0.5, "held", "absent"), index=st.index)
def offsets(X, side):
    """Each site's array-against-sequencing difference from readings X (index site; columns value, gsm, platform, cell)."""
    M = X.groupby(["platform", "cell", X.index]).value.mean().unstack(level=[0, 1])
    cells = sorted({c for p, c in M.columns if (("array", c) in M.columns and ("sequencing", c) in M.columns)})
    D = pd.DataFrame({c: M[("array", c)] - inv(M[("sequencing", c)].to_numpy()) for c in cells}).reindex(side.index)
    S = pd.DataFrame({c: M[("sequencing", c)] for c in cells}).reindex(side.index)
    use = pd.DataFrame({c: np.where(side == "absent", S[c] <= 0.25, S[c] >= 0.75) for c in cells}, index=side.index)
    d = D.where(use).median(axis=1); return d.fillna(d.groupby(side).transform("median")), D, use, cells
if __name__ == "__main__":
  pass
if __name__ == "__main__" and len(sys.argv) > 2:
    X = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(sys.argv[1], "**", "*_v4.csv"), recursive=True)])
    d, D, use, cells = offsets(X, side)
    t = open(os.path.join(H, "step3_03", "step3_v3_summary.txt")).read(); m = re.search(r"array-scale level held ([0-9.]+), held absent ([0-9.]+)", t)
    base = side.map({"held": float(m.group(1)), "absent": float(m.group(2))}); lvl = (base + d).clip(1e-6, 1 - 1e-6)
    Hb = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b)); mark = float(Hb(lvl).mean())
    LOO = pd.read_csv(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3_loo.csv"))
    col = next(c for c in LOO.columns if "frozen" in c.lower() and "A" in c); ref = next(c for c in LOO.columns if c.lower() in ("ref", "gsm", "array", "sample"))
    L = [f"cells measured on both platforms: {', '.join(cells)} | arrays {int(X[X.platform == 'array'].gsm.nunique())}, sequencing samples {int(X[X.platform == 'sequencing'].gsm.nunique())}",
         f"identity sites with a measured difference: held {int(D.where(use)[side == 'held'].notna().any(axis=1).sum())} of {int((side == 'held').sum())}, absent {int(D.where(use)[side == 'absent'].notna().any(axis=1).sum())} of {int((side == 'absent').sum())}"]
    for s_ in ("held", "absent"):
        L.append(f"{s_}: median difference {d[side == s_].median():.4f} | version 3 level {base[side == s_].iloc[0]:.4f} -> version 4 mean level {lvl[side == s_].mean():.4f} (atlas {st[side == s_].mean():.4f}) | entropy {Hb(lvl[side == s_]).mean():.4f}")
    L += [f"derived mark {mark:.5f} bits | measured mark {F['floor']:.5f} bits (6 arrays) | derived / measured {mark / F['floor']:.4f}", "array | Met-A against the measured mark (frozen sites) | against the derived mark"]
    ok = 0
    for r in LOO.itertuples():
        a = getattr(r, col); v = a * F["floor"] / mark; ok += 0.95 <= v <= 1.05; L.append(f"{getattr(r, ref)} | {a:.4f} | {v:.4f}")
    L.append(f"bar 1 (all six Normal, 0.95-1.05, against the derived mark): {ok} of {len(LOO)}: {'met' if ok == len(LOO) == 6 else 'not met'}")
    open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
