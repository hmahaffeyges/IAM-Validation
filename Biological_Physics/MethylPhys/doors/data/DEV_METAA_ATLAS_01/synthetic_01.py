"""DEV-METAA-ATLAS-01, synthetic test 1 (written and committed 2026-10-21 before it is run; development, not sealed).
Runs the chain's own development stage, chain/stage_metaa_atlas.py, unchanged.

The instrument's job: a healthy sample of a cell, read against that cell in IAM-Atlas v2.1 and tared against healthy samples of the same
cell run the same way, reads Normal; a planted loss of the cell's pattern moves it by the shift the reading predicts.

Synthetic arrays, per panel cell (neutrophils, monocytes, B, CD4 T, CD8 T, NK) and per "laboratory": the atlas cell's best value at every
identity site, made softer or harder by the laboratory's offset d (beta + d x (0.5 - beta); d = -0.03, 0, +0.03, +0.06, the size of the
atlas-to-array gap measured in DEV-REFERENCES-01 and beyond it), plus each site's healthy person-to-person spread (normal, donor_sd), clipped
to (0, 1). 12 healthy arrays per laboratory (fixed seed). Each array is tared against the other 11 of its laboratory (leave one out);
its planted copy (a 1 % loss of the cell's pattern: every identity site moved 1 % of the way to 0.5) is tared against the same 11.

Bars (two):
  B1  healthy reads Normal: at least 95 % of healthy arrays, in every cell and laboratory, read A_rel in 0.95-1.05 after the tare.
  B2  the plant is found at its predicted size: in every cell, the mean change of A_rel from each array to its planted copy is within 25 % of
      the change the reading predicts (shift_per_1pct_loss / reference median).
Reported, not scored: untared A by laboratory (the offset the tare must cancel).
Usage: python3 synthetic_01.py   (writes synthetic_01_rows.csv, synthetic_01_summary.txt beside itself)"""
import os, sys, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H_, "..", "..", "..", "chain"))
import stage_metaa_atlas as SA
CELLS = ["neutrophils", "monocytes", "b cells", "cd4 t cells", "cd8 t cells", "nk cells"]; LABS = [-0.03, 0.0, 0.03, 0.06]; N = 12
rng = np.random.default_rng(20261021); rows = []
for cell in CELLS:
    S = SA.identity_sites(cell); mu = SA._col("mu", cell).reindex(S).to_numpy(); sd = SA._col("donor_sd", cell).reindex(S).fillna(0).to_numpy()
    for d in LABS:
        X = [np.clip(mu + d * (0.5 - mu) + rng.normal(0, 1, len(S)) * sd, 1e-4, 1 - 1e-4) for _ in range(N)]
        A = [SA.read(pd.Series(x, index=S), cell)["A"] for x in X]
        for i, x in enumerate(X):
            refs = A[:i] + A[i + 1:]; r = SA.read(pd.Series(x, index=S), cell); t = SA.tare(r["A"], refs)
            p = SA.read(pd.Series(np.clip(x + 0.01 * (0.5 - mu), 1e-4, 1 - 1e-4), index=S), cell); tp = SA.tare(p["A"], refs)
            rows.append(dict(cell=cell, lab=d, i=i, A=r["A"], A_rel=t["A_rel"], A_rel_planted=tp["A_rel"],
                             predicted_change=r["shift_per_1pct_loss"] / t["reference_median"], n_sites=r["n_sites"]))
R = pd.DataFrame(rows); R.to_csv(os.path.join(H_, "synthetic_01_rows.csv"), index=False)
R["normal"] = R.A_rel.between(0.95, 1.05); R["change"] = R.A_rel_planted - R.A_rel
b1 = R.groupby(["cell", "lab"]).normal.mean(); B1 = bool((b1 >= 0.95).all())
g = R.groupby("cell").agg(measured=("change", "mean"), predicted=("predicted_change", "mean")); g["ratio"] = g.measured / g.predicted
B2 = bool(((g.ratio - 1).abs() <= 0.25).all())
L = ["DEV-METAA-ATLAS-01 synthetic test 1 (chain/stage_metaa_atlas.py)", f"identity sites per cell: " + ", ".join(f"{c} {len(SA.identity_sites(c))}" for c in CELLS),
     "untared A by laboratory offset (the offset the tare must cancel): " + "; ".join(f"d {d:+.2f}: {R[R.lab == d].A.mean():.4f}" for d in LABS),
     f"B1 healthy Normal after the tare, lowest share over cell x laboratory: {b1.min():.3f} (bar >= 0.95) -> {'MET' if B1 else 'NOT MET'}",
     "B2 planted 1 % loss, measured / predicted change per cell: " + ", ".join(f"{c} {g.ratio[c]:.3f}" for c in CELLS) + f" (bar within 0.75-1.25) -> {'MET' if B2 else 'NOT MET'}",
     f"ALL BARS {'MET' if B1 and B2 else 'NOT MET'}"]
open(os.path.join(H_, "synthetic_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
