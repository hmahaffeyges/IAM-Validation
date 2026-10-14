"""Atlas v2.1 readiness L10 and L18: planted test of the run script (atlas_block_v21.py). Written and committed 2026-10-14 BEFORE it runs.
Synthetic block, seed 20261014: 40 loci, 5 cells; truth mu bimodal like methylation. Sources as in the real fit: a Loyfer-like WGBS source
(3 samples per cell, coverage 10-40, reads drawn as Binomial(cov, h(m)) with h the adopted bounded transfer), a pooled WGBS source on top
of it (alpha -0.01, beta 1.02), and two array laboratories (offsets +0.006 and -0.004, noise 0.01). Donor spread 0.03 on the logit-free
scale as v2 models it. 15 percent of (cell, locus) pairs get no observation at all.
BARS (all must hold; MET otherwise NOT MET):
 P1 every declared output written (<block>.parquet, _draws.npz with draws, sd_draws, chain, cells, cpg; _prior.parquet; _warnings.json);
    shapes: summary NC x NL rows with v2's columns; draws and sd_draws (100, NC, NL); chain (100,) in 0..3.
 P2 NOT MEASURED: every pair with no observation is NaN in the summary and in both draw arrays; no measured pair is NaN.
 P3 recovery (v2.1 mode): over measured pairs, the 95 percent interval holds the planted mu in at least 90 percent, and the median |mu - truth|
    is at most 0.03.
 P4 v2 reproduction: v2's own code (14_stageB_block.py from the model to the outputs, executed unchanged on the same block and terms) and
    atlas_block_v21.fit_block(mode="v2", K=20) give identical summary columns and identical 20 draws (exact, NaN where NaN).
 P5 widening changes nothing fitted: fit_block(mode="v2", K=100) gives summary columns identical to the K=20 save.
Reported, no bar: the v2 straight line's recovery error on the same curved truth, and divergences per mode.
Writes planted_run_v21_summary.txt."""
import os, sys, json, tempfile, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import atlas_block_v21 as AB
V2SRC = os.path.join(HERE, "..", "..", "..", "atlas", "v2", "scripts", "14_stageB_block.py")
rng = np.random.default_rng(20261014); NL, NC = 40, 5; cells = [f"cell{i}" for i in range(NC)]; loci = np.array([f"cg{i:08d}" for i in range(NL)])
TJ = json.load(open(os.path.join(HERE, "transfer_bounded_v21.json"))); kx, ky = np.array(TJ["knots_x"]), np.array(TJ["knots_y"]); g = lambda x: np.interp(x, kx, ky)
z = rng.random((NC, NL)); truth = np.where(z < 0.4, rng.beta(1.2, 14, (NC, NL)), np.where(z < 0.85, rng.beta(14, 1.3, (NC, NL)), rng.random((NC, NL))))
miss = rng.random((NC, NL)) < 0.15
obs = []
def put(c, src, kind, l, y, cov):
    if not miss[c, l]: obs.append((cells[c], src, kind, l, float(y), float(cov)))
for c in range(NC):
    for l in range(NL):
        m = truth[c, l]
        if c < 3:
            for _ in range(3):
                md = np.clip(m + rng.normal(0, 0.03), 0, 1); cov = rng.integers(10, 41); put(c, "LoyferSyn", "wgbs", l, rng.binomial(cov, g(md)) / cov, cov)
        if c == 3:
            md = np.clip(m + rng.normal(0, 0.03 / np.sqrt(3)), 0, 1); cov = 30; h = np.clip(-0.01 + 1.02 * g(md), 0, 1); put(c, "PoolSyn", "pooled", l, rng.binomial(cov, h) / cov, cov)
        for lab, d in (("LabA", 0.006), ("LabB", -0.004)):
            if c >= 1:
                for _ in range(2): put(c, lab, "array", l, np.clip(m + d + rng.normal(0, 0.03) + rng.normal(0, 0.01), 0, 1), 0.0)
O = pd.DataFrame(obs, columns=["cell", "src", "kind", "l", "y", "cov"]); O["y"] = O.y.clip(1e-3, 1 - 1e-3)
T21 = {"sequencing": {"LoyferSyn": {"form": "bounded", "knots_x": kx.tolist(), "knots_y": ky.tolist()}, "PoolSyn": {"form": "composed", "alpha": -0.01, "beta": 1.02}},
       "array": {"LabA": {"d": 0.006}, "LabB": {"d": -0.004}}}
T21["sequencing"]["Loyfer2023"] = T21["sequencing"]["LoyferSyn"]
T2 = {"LoyferSyn": {"a": -0.0902, "b": 1.1026, "d": 0.0, "t": 0.0}, "PoolSyn": {"a": -0.1038, "b": 1.1200, "d": 0.0, "t": 0.0},
      "LabA": {"a": 0.0, "b": 1.0, "d": 0.006, "t": 0.0}, "LabB": {"a": 0.0, "b": 1.0, "d": -0.004, "t": 0.0}}   # v2's own straight-line form
BI = 7; SEED = 1000 + BI; W = tempfile.mkdtemp()
COLS = ["mu", "mu_sd", "mu_lo", "mu_hi", "donor_sd", "rhat", "ess"]
same = lambda a, b: bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True))
# v2.1 mode
R = AB.fit_block(O, loci, T21, SEED, mode="v21", K=100); dst = os.path.join(W, "block_00007.parquet"); AB.write_block(R, dst)
fl = [dst, dst.replace(".parquet", "_draws.npz"), dst.replace(".parquet", "_prior.parquet"), dst.replace(".parquet", "_warnings.json")]
S = pd.read_parquet(dst); Z = np.load(fl[1], allow_pickle=False)
P1 = all(os.path.exists(f) for f in fl) and len(S) == NC * NL and all(c in S.columns for c in ["cell", "cpg", "n_obs"] + COLS) \
     and Z["draws"].shape == (100, NC, NL) and Z["sd_draws"].shape == (100, NC, NL) and Z["chain"].shape == (100,) and set(np.unique(Z["chain"])) <= {0, 1, 2, 3}
nob = S.n_obs.values.reshape(NC, NL) == 0
P2 = bool(S.loc[S.n_obs == 0, COLS].isna().all().all() and S.loc[S.n_obs > 0, COLS].notna().all().all()
          and np.isnan(Z["draws"][:, nob]).all() and np.isnan(Z["sd_draws"][:, nob]).all() and not np.isnan(Z["draws"][:, ~nob]).any())
tv = truth[[cells.index(c) for c in S.cell], [int(np.where(loci == p)[0][0]) for p in S.cpg]]; mm = S.n_obs.values > 0
cover = float(((S.mu_lo.values <= tv) & (tv <= S.mu_hi.values))[mm].mean()); err = float(np.median(np.abs(S.mu.values - tv)[mm]))
P3 = cover >= 0.90 and err <= 0.03
# v2's own code on the same block
src = open(V2SRC).read(); i = src.index("cells=sorted(O.cell.unique())"); v2dst = os.path.join(W, "v2_block_00007.parquet")
os.environ["WARM"], os.environ["DRAWS"] = "600", "400"
NS = dict(O=O.copy(), ST=T2, NL=NL, loci=loci, BI=BI, dst=v2dst, os=os, json=json, time=AB.time, np=np, pd=pd, jax=AB.jax, jnp=AB.jnp, numpyro=AB.numpyro, dist=AB.dist, ssp=AB.ssp,
          MCMC=AB.MCMC, NUTS=AB.NUTS, init_to_value=AB.init_to_value, summary=AB.summary)
exec(src[i:], NS)
V2S = pd.read_parquet(v2dst); V2D = np.load(v2dst.replace(".parquet", "_draws.npz"))["draws"]
R2 = AB.fit_block(O, loci, T2, SEED, mode="v2", K=20); R2b = AB.fit_block(O, loci, T2, SEED, mode="v2", K=100)
P4 = all(same(V2S[c].values, R2["summary"][c].values) for c in COLS) and same(V2D, R2["draws"])
P5 = all(same(R2["summary"][c].values, R2b["summary"][c].values) for c in COLS)
e2 = float(np.median(np.abs(R2["summary"].mu.values - tv)[mm])); low = mm & (tv < 0.1)
e2low = float(np.median(np.abs(R2["summary"].mu.values - tv)[low])); e21low = float(np.median(np.abs(S.mu.values - tv)[low]))
L = ["Atlas v2.1 L10/L18 planted test, run once (planted_run_v21.py)",
     f"block: {NL} loci x {NC} cells, {len(O)} observations, {int(nob.sum())} pairs with none",
     f"P1 outputs and shapes -> {'met' if P1 else 'NOT met'}", f"P2 NOT MEASURED pairs NaN, measured pairs filled -> {'met' if P2 else 'NOT met'}",
     f"P3 recovery (v2.1 mode): 95% interval coverage {cover:.3f} (bar 0.90), median |error| {err:.4f} (bar 0.03) -> {'met' if P3 else 'NOT met'}",
     f"P4 v2 mode equals v2's own code: summary columns and 20 draws identical -> {'met' if P4 else 'NOT met'}",
     f"P5 K=100 save leaves every summary column identical to K=20 -> {'met' if P5 else 'NOT met'}",
     f"reported: median |error| on the curved truth, v2 straight line {e2:.4f} vs v2.1 {err:.4f}; at planted mu < 0.1: v2 {e2low:.4f} vs v2.1 {e21low:.4f}",
     f"reported: divergences v2.1 {R['warnings']['divergences']}, v2 {R2['warnings']['divergences']}; R-hat p99 v2.1 {R['rhat_p99']:.4f}; wall v2.1 {R['warnings']['wall_s']} s",
     f"PLANTED: {'MET' if (P1 and P2 and P3 and P4 and P5) else 'NOT MET'}"]
open(os.path.join(HERE, "planted_run_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
