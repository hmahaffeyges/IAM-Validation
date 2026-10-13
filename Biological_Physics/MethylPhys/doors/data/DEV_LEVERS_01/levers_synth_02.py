"""DEV-LEVERS-01 synthetic test 2 (written 2026-10-13, before it was run): how many temperatures and how many donors per
temperature does it take to tell the two clocks apart?

The question. Two terms depend on body temperature in different ways. The Landauer floor is proportional to T (absolute),
so a reading built on it changes slowly and almost linearly over the range life uses (277-315 K). A chemical damage rate
(5-methylcytosine deamination) follows Arrhenius, exp(-Ea/kT), so it changes by a large factor over the same range. If the
gauge is dominated by the floor, a species series must look near-linear in T; if damage dominates, it must look exponential.
Telling a line from an exponential over a 40 K window, with donor-to-donor spread, is the whole difficulty.

Design, fixed before running. For each (n_temperatures, n_donors, Ea, mix) the truth is one of two generators, each
normalized to the SAME value at 310 K (human body temperature) so the two cannot be separated by overall level, only by
shape:
  floor model : A(T) = a * (T / 310)
  damage model: A(T) = a * exp(-Ea/k * (1/T - 1/310))
Donor spread is multiplicative lognormal with sigma = 0.06, the healthy spread the committed neutrophil reference shows.
Temperatures are drawn from what is actually available: hibernator 278-310 K, fish 277-293 K, bird 313 K, human 310 K.
Reading: fit both models by least squares on the log reading, then pick the one with the lower residual sum. A design
"works" when the right model is chosen in at least 95 of 100 draws, for BOTH truths (no cheating by always choosing one).
Ea is swept over 60-110 kJ/mol, the range quoted for cytosine deamination chemistry, because the answer depends on it.
Optimistic in one way, stated here before the run: the damage shape is offered its TRUE activation energy, while a real
analysis knows that number only approximately. So a design that fails here is hopeless, and a design that works here still
has to survive a later variant in which the activation energy is fitted.
Writes levers_synth_02_output.csv and levers_synth_02_summary.txt. Usage: python3 levers_synth_02.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
KB = 8.617333262e-5            # eV/K
EV_PER_KJMOL = 1.0 / 96.485    # eV per kJ/mol
SIGMA = 0.06                   # donor spread, healthy neutrophil reference
TPOOL = np.array([277., 281., 285., 289., 293., 298., 303., 307., 310., 313.])
NDRAW = 100
rg = np.random.default_rng(20261013)

def floor_model(T, a): return a * (T / 310.0)
def damage_model(T, a, Ea_eV): return a * np.exp(-(Ea_eV / KB) * (1.0 / T - 1.0 / 310.0))

def fit_pick(T, y, Ea_eV):
    """Least squares on log reading for both shapes with the level free; return 'floor' or 'damage'."""
    ly = np.log(y)
    r = []
    for x in (np.log(T / 310.0), -(Ea_eV / KB) * (1.0 / T - 1.0 / 310.0)):
        lvl = float(np.mean(ly - x))                            # only the level is free; each shape is fixed, never fitted
        r.append(float(np.sum((ly - (lvl + x)) ** 2)))
    return "floor" if r[0] <= r[1] else "damage"

rows = []
for nT in (3, 4, 5, 6, 8, 10):
    for nD in (3, 5, 8, 12, 20):
        for Ea_kj in (60., 75., 90., 110.):
            Ea = Ea_kj * EV_PER_KJMOL
            hit = {"floor": 0, "damage": 0}
            for truth in ("floor", "damage"):
                for _ in range(NDRAW):
                    T = rg.choice(TPOOL, nT, replace=False)
                    Ts = np.repeat(T, nD)
                    mu = floor_model(Ts, 1.0) if truth == "floor" else damage_model(Ts, 1.0, Ea)
                    y = mu * rg.lognormal(0, SIGMA, Ts.size)
                    hit[truth] += (fit_pick(Ts, y, Ea) == truth)
            rows.append(dict(n_temps=nT, n_donors=nD, Ea_kJmol=Ea_kj,
                             floor_right=hit["floor"], damage_right=hit["damage"],
                             works=bool(hit["floor"] >= 95 and hit["damage"] >= 95)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "levers_synth_02_output.csv"), index=False)
ok = T[T.works]
L = ["DEV-LEVERS-01 synthetic test 2: telling the Landauer floor from thermal damage in a species temperature series",
     f"donor spread sigma {SIGMA} | draws {NDRAW} per truth | temperatures available {TPOOL.min():.0f}-{TPOOL.max():.0f} K", ""]
for Ea_kj, g in T.groupby("Ea_kJmol"):
    w = g[g.works]
    if len(w):
        best = w.sort_values(["n_temps", "n_donors"]).iloc[0]
        L.append(f"Ea {Ea_kj:5.0f} kJ/mol: smallest design that works = {int(best.n_temps)} temperatures x {int(best.n_donors)} donors "
                 f"(floor right {int(best.floor_right)}/100, damage right {int(best.damage_right)}/100)")
    else:
        L.append(f"Ea {Ea_kj:5.0f} kJ/mol: NO design in this grid separates the two (up to 10 temperatures x 20 donors)")
L += ["", f"designs that work: {len(ok)} of {len(T)}",
      "BAR: " + ("MET - a feasible series separates the two clocks" if len(ok) else "NOT MET - the two clocks cannot be separated with data we can get")]
open(os.path.join(OUT, "levers_synth_02_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
