"""DEV-LEVERS-01 synthetic test 2b (written 2026-10-13, before it was run): the honest version of test 2.

Test 2 asked whether a pure Landauer floor can be told from pure thermal damage, and every design separated them 100/100,
because over 277-313 K the damage term changes by tens of times while the floor term changes by about 13 %. That is not the
real question. The real reading carries BOTH terms, the activation energy is not known exactly, and species differ from one
another for a hundred reasons that have nothing to do with temperature.

So: A(T) = a * [ (1 - f) * (T/310) + f * exp(-(Ea/k)(1/T - 1/310)) ], with
  f        the share of the reading that is thermal damage (truth: 0, 0.02, 0.05, 0.10, 0.25)
  Ea       75 kJ/mol in truth, FITTED by the reader over 40-140 kJ/mol (not given to it)
  species  a per-temperature multiplicative offset, lognormal sigma_s (0.03, 0.06, 0.10) - the confound that matters,
           since each temperature in a real series is a different animal
  donors   multiplicative lognormal sigma 0.06 within a species
Reader: fit a, f, Ea by least squares on the log reading (f bounded 0-1), 100 draws per cell.
Bars, written before the run: a design DETECTS a share f if the 5th percentile of the fitted f over 100 draws is above 0,
AND the false-positive rate at f = 0 (fitted f above the same threshold) is at most 5 of 100. The smallest detectable share
per species-spread is the answer this test exists to give.
Writes levers_synth_02b_output.csv and levers_synth_02b_summary.txt. Usage: python3 levers_synth_02b.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
KB = 8.617333262e-5; EVKJ = 1.0 / 96.485
TPOOL = np.array([277., 281., 285., 289., 293., 298., 303., 307., 310., 313.])
SIG_D = 0.06; EA_TRUE = 75. * EVKJ; NDRAW = 100
rg = np.random.default_rng(20261013)

def model(T, a, f, Ea):
    return a * ((1 - f) * (T / 310.0) + f * np.exp(-(Ea / KB) * (1.0 / T - 1.0 / 310.0)))

def fit(T, y):
    def res(p): return np.log(model(T, np.exp(p[0]), p[1], p[2] * EVKJ)) - np.log(y)
    best = None
    for f0 in (0.01, 0.1, 0.3):
        r = least_squares(res, [np.log(np.mean(y)), f0, 75.], bounds=([-10, 0, 40], [10, 1, 140]), max_nfev=400)
        if best is None or r.cost < best.cost: best = r
    return float(best.x[1]), float(best.x[2])

rows = []
for nT in (4, 6, 8):
    for nD in (5, 10, 20):
        for sig_s in (0.03, 0.06, 0.10):
            draws = {}
            for f_true in (0.0, 0.02, 0.05, 0.10, 0.25):
                fh = []
                for _ in range(NDRAW):
                    T = rg.choice(TPOOL, nT, replace=False)
                    off = rg.lognormal(0, sig_s, nT)                       # one offset per species
                    Ts = np.repeat(T, nD); o = np.repeat(off, nD)
                    y = model(Ts, 1.0, f_true, EA_TRUE) * o * rg.lognormal(0, SIG_D, Ts.size)
                    fh.append(fit(Ts, y)[0])
                draws[f_true] = np.array(fh)
            thr = float(np.percentile(draws[0.0], 95))                     # false-positive threshold from the f = 0 draws
            for f_true, fh in draws.items():
                if f_true == 0.0: continue
                rows.append(dict(n_temps=nT, n_donors=nD, sigma_species=sig_s, f_true=f_true,
                                 f_fit_median=float(np.median(fh)), f_fit_p05=float(np.percentile(fh, 5)),
                                 threshold_from_null=thr, detected=bool(np.percentile(fh, 5) > thr)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "levers_synth_02b_output.csv"), index=False)
L = ["DEV-LEVERS-01 synthetic test 2b: smallest share of thermal damage a real temperature series can detect",
     f"activation energy fitted (truth 75 kJ/mol) | donor spread {SIG_D} | {NDRAW} draws per cell", ""]
for (nT, nD, ss), g in T.groupby(["n_temps", "n_donors", "sigma_species"]):
    d = g[g.detected]
    sm = f"{d.f_true.min():.2f}" if len(d) else "none in this grid"
    L.append(f"{nT} temperatures x {nD} donors, species spread {ss:.2f}: smallest detectable damage share {sm}"
             f" | null threshold on fitted f {g.threshold_from_null.iloc[0]:.3f}")
any_ok = T.detected.any()
L += ["", "BAR: " + ("MET - some feasible series detects a damage share" if any_ok else "NOT MET - no series in this grid detects any damage share"),
      "Read this as the design requirement, not as a result about the world."]
open(os.path.join(OUT, "levers_synth_02b_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:3] + L[-3:]))
