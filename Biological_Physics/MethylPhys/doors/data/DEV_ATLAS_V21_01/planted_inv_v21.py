"""Planted test of mode v21inv (converted readings; transfer_inverse_v21.md). Written 2026-10-15 BEFORE it runs. Synthetic only.
# NO CHAIN: tests the atlas block model (atlas_block_v21.py), not a chain stage
What the first planted test missed: truth in the flat stretches, and cells seen only by sequencing. This block puts both in.
Seed 20261015; 60 loci, 6 cells. Truth: 35 percent Uniform(0.003, 0.03), 35 percent Uniform(0.965, 0.997), 30 percent Uniform(0.03, 0.965).
Cells 0-1: sequencing only (Loyfer-like, 2 samples, coverage 10-40). Cells 2-5: the same sequencing plus two array laboratories (offsets
+0.006, -0.004; 2 arrays each). Reads: Binomial(cov, g_strict(m + donor noise 0.02) + a cell-level bias drawn from the real scatter s(m),
shared by the cell's samples). 10 percent of pairs unobserved. Fitted with atlas_block_v21.fit_block(mode="v21", K=20), seed 1007.
STRICT arm (the transfer named on the command line; design 2 = transfer_v21_anchored.json, since design 1 failed C2 before any fit) BARS, all must hold:
 Q1 convergence: R-hat p99 <= 1.01 and at most 1 percent of draws at the maximum tree depth.
 Q2 recovery: 95 percent intervals hold the planted value in >= 0.90 of measured pairs overall and >= 0.85 of the sequencing-only cells'
    pairs with truth below 0.03 or above 0.965; median |mu - truth| <= 0.03.
CONTROL arm (transfer_v21_final.json, the adopted flat curve), same data. VALIDITY: the control must show the failure (R-hat p99 > 1.01 or
more than 1 percent of draws at the maximum tree depth). If it does not, this block is too small to see the failure and the test reads
NOT VALID; the real-block check then decides alone. Writes planted_strict_v21_summary.txt."""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import atlas_block_v21 as AB
rng = np.random.default_rng(20261015); NL, NC = 60, 6; cells = [f"cell{i}" for i in range(NC)]; loci = np.array([f"cg{i:08d}" for i in range(NL)])
# design 2: the transfer is named on the command line (transfer_v21_anchored.json)
ST = json.load(open(os.path.join(H, sys.argv[1] if len(sys.argv) > 1 else "transfer_v21_strict.json"))); FT = json.load(open(os.path.join(H, "transfer_v21_final.json")))
g = lambda x: np.interp(x, ST["knots_x"], ST["knots_y"]); s = lambda x: np.interp(x, ST["scatter_mid"], ST["scatter_s"])
z = rng.random((NC, NL)); truth = np.where(z < 0.35, rng.uniform(0.003, 0.03, (NC, NL)), np.where(z < 0.70, rng.uniform(0.965, 0.997, (NC, NL)), rng.uniform(0.03, 0.965, (NC, NL))))
miss = rng.random((NC, NL)) < 0.10; bias = rng.normal(0, 1, (NC, NL)) * s(truth); obs = []
for c in range(NC):
    for l in range(NL):
        if miss[c, l]: continue
        m = truth[c, l]
        for _ in range(2):
            md = np.clip(m + rng.normal(0, 0.02), 0, 1); cov = int(rng.integers(10, 41))
            obs.append((cells[c], "LoyferSyn", "wgbs", l, rng.binomial(cov, float(np.clip(g(md) + bias[c, l], 0, 1))) / cov, float(cov)))
        if c >= 2:
            for lab, d in (("LabA", 0.006), ("LabB", -0.004)):
                for _ in range(2): obs.append((cells[c], lab, "array", l, float(np.clip(m + d + rng.normal(0, 0.02) + rng.normal(0, 0.01), 0, 1)), 0.0))
O = pd.DataFrame(obs, columns=["cell", "src", "kind", "l", "y", "cov"]); O["y"] = O.y.clip(1e-3, 1 - 1e-3)
AR = {"LabA": {"d": 0.006}, "LabB": {"d": -0.004}}
def arm(T):
    TT = {"knots_x": T["knots_x"], "knots_y": T["knots_y"], "knot_slopes": T["knot_slopes"], "scatter_mid": T["scatter_mid"], "scatter_s": T["scatter_s"], "array": AR}
    R = AB.fit_block(O, loci, TT, 1007, mode="v21inv", K=20); S = R["summary"] if isinstance(R["summary"], pd.DataFrame) else pd.DataFrame(R["summary"])
    W = R["warnings"]; ndraw = 4 * 400
    tv = truth[[cells.index(x) for x in S.cell], [int(np.where(loci == p)[0][0]) for p in S.cpg]]; mm = S.n_obs.values > 0
    inside = (S.mu_lo.values <= tv) & (tv <= S.mu_hi.values)
    so = np.isin(S.cell.values, ["cell0", "cell1"]) & ((tv < 0.03) | (tv > 0.965)) & mm
    return dict(rhat=float(R["rhat_p99"]), maxdepth=W["draws_at_max_tree_depth"] / ndraw, div=W["divergences"], wall=W["wall_s"],
                cover=float(inside[mm].mean()), cover_seq_ext=float(inside[so].mean()), n_seq_ext=int(so.sum()), err=float(np.median(np.abs(S.mu.values - tv)[mm])))
A = arm(ST)
I1 = A["cover"] >= 0.90 and A["cover_seq_ext"] >= 0.85 and A["err"] <= 0.03; I2 = A["maxdepth"] <= 0.01
fmt = lambda k, r: f"{k}: R-hat p99 {r['rhat']:.4f}, draws at max depth {100 * r['maxdepth']:.1f} percent, divergences {r['div']}, wall {r['wall']:.0f} s, coverage {r['cover']:.3f}, sequencing-only extremes {r['cover_seq_ext']:.3f} (n {r['n_seq_ext']}), median error {r['err']:.4f}"
L = [f"Planted test of mode v21inv (planted_inv_v21.py), {len(O)} observations", fmt("V21INV", A),
     f"I1 recovery: {'met' if I1 else 'NOT met'} | I2 sampler: {'met' if I2 else 'NOT met'}", "VERDICT: " + ("MET" if (I1 and I2) else "NOT MET")]
open(os.path.join(H, "planted_inv_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
