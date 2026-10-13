"""DEV-BLOOD-SENSITIVITY-01 v2 (written 2026-10-13, before it was run): how small a fidelity drop in the neutrophil component can the
chain's whole-blood Met-A see, when neutrophils are only part of the sample AND healthy people differ from one another?

Why v2. v1 (kept as *_v1_DILUTION_ONLY.*) gave every simulated person the SAME cell-type profiles, so the only spread among healthy
draws was measurement noise and the healthy band came out at +-0.6 %, far tighter than real arrays. Its result still stands for the
narrower question it actually answered: dilution alone does not destroy the signal - a 1 % drop inside a 40 % component still moves
the reading by 2.7 %, because the mixture-predicted denominator absorbs the mixing. v2 adds the spread that sets a real detection
limit: each person gets their own profile for every cell type, drawn from the atlas posterior's own uncertainty (mean, sd), while the
reading is given the POPULATION MEAN profiles - which is the situation in life.

The question. An array averages all the cells in a tube together, so a mixture's disorder already contains the mixing itself. The
chain answers that by dividing by the disorder the estimated mixture predicts (dev_comp_methods.met_a: mean H(beta) / mean H(e),
e = sum f_g mu_g). Whether that leaves enough leverage to see a small change in one component has never been measured. This measures it.

Design, fixed before running:
  - Profiles: the atlas posteriors for the six cell types of whole blood (neutrophils, monocytes, CD4 T, CD8 T, B, NK).
  - Sites: the neutrophil sites, chosen by a rule written before any planting - neutrophil mean >= 0.90 or <= 0.10 with sd <= 0.05,
    the 3,000 with the smallest sd. Site choice never sees a planted array.
  - A person: each cell type has its OWN profile for that person, drawn from the atlas posterior (mean, sd) once per person, plus
    measurement noise (SD 0.02); the tube is the fraction-weighted sum. The reading is given the population means, not the person's.
    Fractions drawn around healthy whole blood, with the neutrophil share set per condition.
  - The plant: pattern LOSS in the neutrophil component only - its beta moved toward 0.5 by a factor d (d = 0.01 to 0.20).
    Nothing else changes, so the only difference between healthy and planted draws is the neutrophils' own fidelity.
  - Reading: the CHAIN's dev_comp_methods.met_a, given the TRUE fractions. Giving it the true fractions is deliberate - it is the
    best case. If the reading cannot see a drop with perfect composition knowledge, it certainly cannot with estimated composition.
  - Detected: at least 95 % of 200 planted draws read above the 95th percentile of 200 healthy draws at the same fractions.
  - WHAT IS BEING DECIDED, written now: the tissue cancers we have read show changes of roughly 2 to 7 % in this reading. If the
    smallest detectable drop at a realistic neutrophil share (0.55-0.70) is <= 2 %, whole blood on arrays can carry a signal of the
    size we already see in tissue, and the blood track can open on arrays. If it is > 5 %, it cannot, and blood needs sequencing,
    where molecules stay intact and can be assigned to a cell type. Either answer is written down.
Usage: python3 blood_sensitivity_01.py ATLAS_POSTERIOR_DIR [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
from dev_comp_methods import met_a                      # the chain's whole-blood reading, unchanged

POST, OUT = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else HERE)
CELLS = ["neutrophils", "monocytes", "cd4_t_cells", "cd8_t_cells", "b_cells", "nk_cells"]
P = {}
for c in CELLS:
    d = pd.read_parquet(os.path.join(POST, f"atlas_{c}_posterior.parquet")).set_index("cpg")
    P[c] = d
cpg = P["neutrophils"].index
for c in CELLS: cpg = cpg.intersection(P[c].index)
N = P["neutrophils"].loc[cpg]
pick = N[((N["mean"] >= 0.90) | (N["mean"] <= 0.10)) & (N["sd"] <= 0.05)].nsmallest(3000, "sd")
ns = list(pick.index)
MU = {c: P[c].loc[ns, "mean"] for c in CELLS}            # profiles the reading is given (population means)
SD = {c: P[c].loc[ns, "sd"] for c in CELLS}              # the atlas's own uncertainty: how much people differ at each site
rg = np.random.default_rng(20261013)
OTHER = dict(monocytes=(0.05, 0.10), cd4_t_cells=(0.10, 0.18), cd8_t_cells=(0.05, 0.10), b_cells=(0.03, 0.07), nk_cells=(0.03, 0.08))

def person(f_neu, d):
    f = {c: rg.uniform(*OTHER[c]) for c in OTHER}
    s = sum(f.values()); f = {c: v * (1 - f_neu) / s for c, v in f.items()}; f["neutrophils"] = f_neu
    beta = pd.Series(0.0, index=ns)
    for c in CELLS:
        own = np.clip(MU[c].values + rg.normal(0, 1, len(ns)) * SD[c].values, 1e-4, 1 - 1e-4)   # this person's own profile
        x = np.clip(own + rg.normal(0, 0.02, len(ns)), 1e-4, 1 - 1e-4)
        if c == "neutrophils" and d > 0: x = 0.5 + (x - 0.5) * (1 - d)       # pattern loss in neutrophils only
        beta += f[c] * x
    return beta, f

L = [f"DEV-BLOOD-SENSITIVITY-01 v2 (people differ): smallest neutrophil fidelity drop the chain's whole-blood Met-A can see ({len(ns)} neutrophil sites)", ""]
rows = []
for f_neu in (0.40, 0.55, 0.70):
    H0 = []
    for _ in range(200):
        b, f = person(f_neu, 0.0); r, n, tot = met_a(b, f, MU, ns); H0.append(r)
    thr = float(np.percentile(H0, 95)); smallest = None
    for d in (0.01, 0.02, 0.05, 0.10, 0.20):
        H1 = []
        for _ in range(200):
            b, f = person(f_neu, d); r, n, tot = met_a(b, f, MU, ns); H1.append(r)
        frac = float(np.mean(np.array(H1) > thr)); med = float(np.median(H1))
        hit = frac >= 0.95
        if hit and smallest is None: smallest = d
        rows.append((f_neu, d, med, frac, hit))
        L.append(f"neutrophils {f_neu:.2f} | planted drop {d * 100:4.0f} % | median reading {med:.4f} "
                 f"(healthy median {np.median(H0):.4f}, 95th {thr:.4f}) | above the healthy 95th: {frac * 100:5.1f} % "
                 f"{'DETECTED' if hit else ''}")
    L.append(f"  -> smallest detected at neutrophils {f_neu:.2f}: {'none of those tested' if smallest is None else f'{smallest * 100:.0f} %'}")
    L.append("")
real = [r for r in rows if r[0] >= 0.55 and r[4]]
sm = min((r[1] for r in real), default=None)
verdict = ("ARRAYS CAN CARRY IT: the smallest detectable drop at a realistic neutrophil share is "
           f"{sm * 100:.0f} %, within the 2-7 % range the tissue cancers show" if sm is not None and sm <= 0.02 else
           "ARRAYS CANNOT CARRY A TISSUE-SIZED SIGNAL IN WHOLE BLOOD: the smallest detectable drop at a realistic neutrophil share is "
           + (f"{sm * 100:.0f} %" if sm is not None else "larger than 20 %") + ", above the 2 % the tissue cancers show. Blood needs "
             "sequencing, where molecules stay intact and can be assigned to a cell type.")
L += [verdict, "", "Read with the TRUE fractions, which is the best case: with estimated composition the reading can only do worse."]
pd.DataFrame(rows, columns=["f_neutrophils", "planted_drop", "median_reading", "frac_above_healthy_95th", "detected"]).to_csv(
    os.path.join(OUT, "blood_sensitivity_01_output.csv"), index=False)
open(os.path.join(OUT, "blood_sensitivity_01_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
