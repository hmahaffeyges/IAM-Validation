"""Register A8, exploration on spent counts (2026-11-13; nothing sealed). Each healthy atlas cell type's reading over the THERMAL level
(DEV_CHAIN_Q2_01/celltype_vs_thermal.csv: zero loose along IAM-A's slope, against the derived per-site loss pushed through the counter at
the type's own clustering; above 1 = an eraser share s = 1 - 1/ratio at the canon coin) set against how much the type divides and renews:
MKI67 (dividing), the restore crew DNMT1+UHRF1 (crew_vs_held.csv, Human Protein Atlas, nTPM) and lifespan (Sender and Milo 2021,
lifespan_vs_held.csv). The hypothesis from PHY-085/086: the erasers act where cells divide and change identity, and little in mature cells.
Null test: the rank correlation must have the same sign in both random halves of the types (1000 splits).
Writes types_vs_thermal_turnover_summary.txt."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__))
T = pd.read_csv(os.path.join(H, "..", "DEV_CHAIN_Q2_01", "celltype_vs_thermal.csv")); T = T[T.lab == "atlas"].set_index("cell")
C = pd.read_csv(os.path.join(H, "crew_vs_held.csv")); Lf = pd.read_csv(os.path.join(H, "lifespan_vs_held.csv"))
rows = []
for _, r in C.iterrows():
    names = [n.strip().lower() for n in str(r["cells"]).split(";")]
    hit = [n for n in names if n in T.index]
    if hit: rows.append(dict(group=r["hpa"], ratio=float(T.loc[hit, "ratio"].mean()), MKI67=r["MKI67"], crew=r["crew_DNMT1_UHRF1"], lifespan=r["lifespan_days"]))
X = pd.DataFrame(rows)
L = [f"{len(X)} Human Protein Atlas groups matched to atlas types; reading over thermal: median {X.ratio.median():.3f}"]
rng = np.random.default_rng(3)
for k in ("MKI67", "crew", "lifespan"):
    Y = X.dropna(subset=[k]); rho, p = spearmanr(Y.ratio, Y[k]); same = 0
    for _ in range(1000):
        i = rng.permutation(len(Y)); a, b = Y.iloc[i[: len(Y) // 2]], Y.iloc[i[len(Y) // 2:]]
        same += np.sign(spearmanr(a.ratio, a[k])[0]) == np.sign(rho) == np.sign(spearmanr(b.ratio, b[k])[0])
    L.append(f"reading/thermal vs {k}: rho {rho:+.2f} (p {p:.3f}, n {len(Y)}); same sign in both random halves {same / 10:.0f} % of 1000 splits")
L += ["", X.sort_values("ratio").round(3).to_string(index=False), "",
      "Exploration on spent counts; the hypothesis predicts reading/thermal rising with MKI67 and the crew, falling with lifespan."]
import math
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
Mc = 20.941; ed = lambda d: 1 / (1 + odds(2 * Mc * d))
r, c = X.ratio.to_numpy(), X.crew.to_numpy(); fitK = lambda r, c: max(0.0, float(np.sum((r - 1) / c) / np.sum(1 / c ** 2)))
ed_, e0_ = [], []
for i in range(len(r)):
    m = np.ones(len(r), bool); m[i] = False; K = fitK(r[m], c[m]); ed_.append(r[i] - (1 + K / c[i])); e0_.append(r[i] - r[m].mean())
L.insert(4, f"DUTY FORM (push on a fraction d = crew/(crew + K) of the time; reading/thermal = 1 + K/crew; exact at d 0.9: {ed(0.9) / ed(1.0):.4f} vs 1/d 1.1111):"
            f" K {fitK(r, c):.2f}; leave-one-out rms {np.sqrt(np.mean(np.square(ed_))):.4f} against a flat mean {np.sqrt(np.mean(np.square(e0_))):.4f}:"
            f" {'no better than flat, NOT supported' if np.sqrt(np.mean(np.square(ed_))) >= np.sqrt(np.mean(np.square(e0_))) else 'better than flat'}; below thermal (unreachable) {int((r < 1).sum())}")
open(os.path.join(H, "types_vs_thermal_turnover_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
