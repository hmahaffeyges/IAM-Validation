"""DEV-BLOOD-SENSITIVITY-01, part 2 (written 2026-10-13, before it was run): the spread of the chain's whole-blood reading between
REAL healthy people - the number the simulation could not supply.

Why this is needed. Part 1 measured the LEVERAGE (a 1 % fidelity drop in the neutrophil component moves the reading +2.64 % at a
0.40 neutrophil share, +3.79 % at 0.55, +5.13 % at 0.70) but produced an unrealistically tight healthy band, because simulated people
were too alike. A detection limit is leverage divided by real spread, so the spread has to come from real arrays.

DECLARED IN ADVANCE: these arrays (GSE184269, read once as the C-score band's held-out test) are in CANON/data_spent.csv. This is
DEVELOPMENT use - a descriptive spread, not a sealed test - and it disqualifies them as a held-out test for any blood reading later.
No bar is set here and no verdict is taken; the output is a spread and a limit computed from it.

The chain, unchanged: conductor_v3.stage_a_composition (EPIC blood NNLS) then conductor_v3.stage_m_blood (whole-blood Met-A =
mean H(beta) / mean H(e) at the neutrophil sites). Both are the chain's own stages, so a researcher pressing the button gets this.
Untared values are used deliberately: the laboratory offset is common to one laboratory, so the WITHIN-laboratory spread is what a
tared reading would face.

Usage: python3 blood_spread_01.py BETA_DIR ROWS_CSV [OUTDIR]
Writes blood_spread_01_output.csv (one row per array) and blood_spread_01_summary.txt.
"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
import conductor_v3 as C
BD, ROWS = sys.argv[1], sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else HERE
R = pd.read_csv(ROWS)
wb = R[R.specimen.astype(str).str.contains("blood", case=False)]
rows = []
for _, r in wb.iterrows():
    p = glob.glob(os.path.join(BD, f"{r.gsm}*.parquet"))
    if not p: rows.append(dict(gsm=r.gsm, A=np.nan, reason="no calibrated beta")); continue
    b = pd.read_parquet(p[0]).iloc[:, 0].astype("float64"); b.index = b.index.astype(str)
    a = C.stage_a_composition(b)
    if not a.get("fractions"): rows.append(dict(gsm=r.gsm, A=np.nan, reason=a.get("reason", "composition refused"))); continue
    m, _resid = C.stage_m_blood(b, a)                        # the chain returns (record, residual map)
    A = m.get("A")
    rows.append(dict(gsm=r.gsm, A=(float(A) if A is not None else np.nan),
                     neutrophil_fraction=m.get("fraction"), n_sites=m.get("n_sites"),
                     shift_per_1pct_loss=m.get("shift_per_1pct_loss"),           # the CHAIN's own leverage, per sample
                     methylated_sites_mean_beta=m.get("methylated_sites_mean_beta"),
                     past_entropy_ceiling=m.get("past_entropy_ceiling"),
                     reason=(None if A is not None else str(m.get("reason", "no A"))[:90])))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "blood_spread_01_output.csv"), index=False)
ok = T.A.dropna()
L = ["DEV-BLOOD-SENSITIVITY-01 part 2: the spread of the chain's whole-blood reading between real healthy people", "",
     f"arrays attempted {len(T)} | read {len(ok)} | withheld {int(T.A.isna().sum())}"]
for _, r in T[T.A.isna()].iterrows(): L.append(f"  withheld {r.gsm}: {r.reason}")
if len(ok) >= 5:
    med, sd = float(ok.median()), float(ok.std())
    rng = (float(ok.min()), float(ok.max())); p5, p95 = float(np.percentile(ok, 5)), float(np.percentile(ok, 95))
    cv = sd / med
    L += ["", f"median {med:.4f} | SD {sd:.4f} | relative SD {cv * 100:.2f} % | range {rng[0]:.4f} to {rng[1]:.4f} "
              f"| 5th-95th {p5:.4f} to {p95:.4f}",
          f"the 95th percentile sits {(p95 / med - 1) * 100:+.2f} % above the median", ""]
    # the chain reports its own leverage per sample (shift_per_1pct_loss), so the limit uses the measured value, not the simulated one
    lev = T.shift_per_1pct_loss.dropna()
    if len(lev):
        lv = float(lev.median())
        L.append(f"the chain's own leverage, median over these arrays: {lv:.5f} in A per 1 % pattern loss "
                 f"({lv / med * 100:.2f} % of the median reading)")
        L.append(f"  so a loss of about {(p95 - med) / lv:.2f} % clears the healthy 95th percentile, and about "
                 f"{(ok.max() - med) / lv:.2f} % clears the highest healthy array read here")
        L.append(f"  (neutrophil share in these arrays: median {T.neutrophil_fraction.median():.3f}, "
                 f"range {T.neutrophil_fraction.min():.3f} to {T.neutrophil_fraction.max():.3f})")
    pc = T.past_entropy_ceiling.dropna()
    if len(pc): L.append(f"entropy-ceiling flag recorded on {len(pc)} arrays | past the ceiling: {int(pc.astype(str).str.lower().eq('true').sum())}")
    L += ["", "This is the WITHIN-laboratory spread of untared readings in one healthy set; a tared reading faces this, not the",
          "between-laboratory spread. It is a descriptive development measurement on arrays already spent as the C-score band's",
          "held-out test, so it sets no bar and disqualifies these arrays as a held-out test for a blood reading later."]
else:
    L += ["", "TOO FEW ARRAYS READ to report a spread; the withheld reasons are above."]
open(os.path.join(OUT, "blood_spread_01_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
