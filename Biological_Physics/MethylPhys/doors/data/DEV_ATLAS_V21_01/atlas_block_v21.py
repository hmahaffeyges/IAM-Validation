"""Atlas v2.1 stage B: the model and the saved outputs of one block (2026-10-14). v2's 14_stageB_block.py, line for line, except:
 1. Sequencing rows (Loyfer, Tian, ENCODE alike): mean h(m) = g(m), g the one bounded transfer (L22 second design, transfer_v21_final.json);
    the donor-spread slope is the centered difference (h(m + 0.01) - h(m - 0.01)) / 0.02 (continuous in m), replacing v2's constant b; the
    read-count term is h(1 - h) / coverage, the binomial variance at the value sequencing reads (v2's m(1 - m) came with its straight line);
    and the measured cell-to-cell scatter of the transfer is carried: + n * s(m)^2, n the source's samples for that cell and locus, s(m)
    interpolated between the range midpoints, so the source's mean carries s(m)^2 however many samples it has.
 2. Saved outputs widened (L18): K = 100 thinned draws of mu AND of the donor spread, the chain of every saved draw, and per-block sampler
    warnings (divergences per chain, mean acceptance, draws at the maximum tree depth).
 mode = "v2" reproduces v2 exactly (straight line a + b m, constant b, m(1 - m)); the planted test (planted_run_v21.py) checks that against
 v2's own code on the same synthetic block before the v2.1 mode is trusted.
Model, priors, sampler settings (NUTS, target 0.9, max tree depth 8, 600 warm-up, 400 draws, 4 chains), seeds (1000 + block) and the
NOT MEASURED rule are v2's."""
import os, json, time, numpy as np, pandas as pd
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist, scipy.special as ssp
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary
numpyro.set_host_device_count(4)
FLOOR = 0.01 ** 2; HSTEP = 0.01; MAXDEPTH = 8

def row_terms(O, TERMS, mode):
    """Per observation: array offset d, and for sequencing rows (alpha, beta) on g (v2.1) or (a, b) of the straight line (v2)."""
    if mode == "v2":
        a = O.src.map(lambda s: TERMS[s]["a"]).values; b = O.src.map(lambda s: TERMS[s]["b"]).values; d = O.src.map(lambda s: TERMS[s]["d"]).values
        return a, b, d
    AR = TERMS["array"]
    al = np.zeros(len(O)); be = np.ones(len(O))
    d = np.array([AR[s]["d"] if k == "array" else 0.0 for s, k in zip(O.src, O.kind)])
    return al, be, d

def fit_block(O, loci, TERMS, seed, mode="v21", K=100, warm=600, draws=400):
    """O: columns cell, src, kind (array | wgbs | pooled), l (locus index), y (clipped to [1e-3, 1 - 1e-3]), cov. Returns a dict of outputs."""
    NL = len(loci); cells = sorted(O.cell.unique()); ci = {c: i for i, c in enumerate(cells)}; NC = len(cells)
    O = O.copy(); O["c"] = O.cell.map(ci)
    if mode == "v21inv":
        # transfer_inverse_v21.py (2026-10-15): each sequencing reading y is converted ONCE to the array scale, x = g^-1(y), with g the smooth
        # transfer; its noise is divided by the local slope k = g'(x) (error propagation), k floored at 0.151, the smallest non-flat segment
        # slope of the adopted curve, so a reading on a plateau is down-weighted (at most 44x) instead of given infinite noise.
        from scipy.interpolate import PchipInterpolator
        _P = PchipInterpolator(TERMS["knots_x"], TERMS["knots_y"]); _gx = np.linspace(0, 1, 200001); _gv = _P(_gx) + 1e-12 * _gx
        sq = (O.kind != "array").values; yv = O.y.values.astype(float); xv = yv.copy()
        xv[sq] = np.interp(yv[sq], _gv, _gx); kv = np.ones(len(O)); kv[sq] = np.maximum(_P.derivative()(xv[sq]), 0.151)
        sv = np.zeros(len(O)); sv[sq] = np.interp(xv[sq], TERMS["scatter_mid"], TERMS["scatter_s"])
        O["y"] = np.clip(xv, 1e-3, 1 - 1e-3); INV_K = jnp.array(kv); INV_S = jnp.array(sv)
    p1, p2, d = row_terms(O, TERMS, mode)
    a = jnp.array(p1); b = jnp.array(p2); dd = jnp.array(d); t = jnp.zeros(len(O))
    c_ = jnp.array(O.c.values); l_ = jnp.array(O.l.values); y_ = jnp.array(O.y.values); cov_ = jnp.array(O["cov"].values)
    isarr = jnp.array((O.kind == "array").values); ispool = jnp.array((O.kind == "pooled").values)
    if mode != "v2":
        kx = jnp.array(TERMS["knots_x"]); ky = jnp.array(TERMS["knots_y"]); g = lambda x: jnp.interp(x, kx, ky)
        SMOOTH = "knot_slopes" in TERMS
        BINOM_M = bool(TERMS.get("diag_binom_on_m"))   # DIAGNOSTIC ONLY (diagnostic 5): read-count noise from the cell value m, as in v2
        if SMOOTH:
            # transfer_smooth_v21.py: monotone cubic (PCHIP) through the same knots, with its exact, continuous slope
            km = jnp.array(TERMS["knot_slopes"]); nk = len(TERMS["knots_x"])
            def _seg(x):
                x = jnp.clip(x, kx[0], kx[-1]); i = jnp.clip(jnp.searchsorted(kx, x) - 1, 0, nk - 2)
                hh = kx[i + 1] - kx[i]; u = (x - kx[i]) / hh
                return i, hh, u
            def g(x):
                i, hh, u = _seg(x)
                return ((2 * u ** 3 - 3 * u ** 2 + 1) * ky[i] + (u ** 3 - 2 * u ** 2 + u) * hh * km[i]
                        + (-2 * u ** 3 + 3 * u ** 2) * ky[i + 1] + (u ** 3 - u ** 2) * hh * km[i + 1])
            def gslope(x):
                i, hh, u = _seg(x)
                return ((6 * u ** 2 - 6 * u) * ky[i] / hh + (3 * u ** 2 - 4 * u + 1) * km[i]
                        + (-6 * u ** 2 + 6 * u) * ky[i + 1] / hh + (3 * u ** 2 - 2 * u) * km[i + 1])
        smid = jnp.array(TERMS["scatter_mid"]); sval = jnp.array(TERMS["scatter_s"]); sc = lambda x: jnp.interp(x, smid, sval)
        nrep = jnp.array(O.groupby(["cell", "src", "l"]).y.transform("size").values.astype(float))
    def model():
        nu = numpyro.sample("nu", dist.Normal(0., 2.).expand([NL])); tau = numpyro.sample("tau", dist.HalfNormal(2.))
        lmu = numpyro.sample("lmu", dist.Normal(nu[None, :], tau).expand([NC, NL])); mu = jax.nn.sigmoid(lmu)
        lsd = numpyro.sample("lsd", dist.Normal(jnp.log(0.03), 0.7).expand([NC, NL])); sd = jnp.exp(lsd)
        m = mu[c_, l_]; s_ = sd[c_, l_]
        if mode == "v21inv":
            loc = jnp.where(isarr, m + dd, m)
            var = jnp.where(isarr, s_ ** 2 + t ** 2 + FLOOR, jnp.where(ispool, s_ ** 2 / 3., s_ ** 2) + m * (1 - m) / (jnp.maximum(cov_, 1.) * INV_K ** 2)
                            + nrep * INV_S ** 2 / INV_K ** 2 + t ** 2 + FLOOR)
        elif mode == "v2":
            loc = jnp.where(isarr, m + dd, a + b * m)
            var = jnp.where(isarr, s_ ** 2 + t ** 2 + FLOOR, (b ** 2) * jnp.where(ispool, s_ ** 2 / 3., s_ ** 2) + m * (1 - m) / jnp.maximum(cov_, 1.) + t ** 2 + FLOOR)
        else:
            h = a + b * g(m)
            slope = b * gslope(m) if SMOOTH else b * (g(m + HSTEP) - g(m - HSTEP)) / (2 * HSTEP)
            loc = jnp.where(isarr, m + dd, h)
            var = jnp.where(isarr, s_ ** 2 + t ** 2 + FLOOR, (slope ** 2) * jnp.where(ispool, s_ ** 2 / 3., s_ ** 2) + (m * (1 - m) if BINOM_M else jnp.clip(h * (1 - h), 0., 0.25)) / jnp.maximum(cov_, 1.)
                            + nrep * sc(m) ** 2 + t ** 2 + FLOOR)
        numpyro.sample("y", dist.Normal(loc, jnp.sqrt(var)), obs=y_)
    lm = O.groupby("l").y.mean().reindex(range(NL)).fillna(0.5).clip(0.02, 0.98).values
    cm = O.groupby(["c", "l"]).y.mean().unstack().reindex(index=range(NC), columns=range(NL))
    l0 = ssp.logit(cm.clip(0.02, 0.98).fillna(pd.DataFrame(np.tile(lm, (NC, 1))))).values
    INIT = init_to_value(values={"nu": jnp.array(ssp.logit(lm)), "tau": jnp.array(1.0), "lmu": jnp.array(l0), "lsd": jnp.full((NC, NL), jnp.log(0.03))})
    t0 = time.time()
    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=INIT, max_tree_depth=MAXDEPTH), num_warmup=warm, num_samples=draws, num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "accept_prob", "num_steps")); Gs = mc.get_samples(group_by_chain=True); jax.block_until_ready(Gs); wall = time.time() - t0
    mu = jax.nn.sigmoid(Gs["lmu"]); sm = summary({"mu": mu, "lsd": Gs["lsd"]}, group_by_chain=True)
    flat = np.asarray(mu).reshape(-1, NC, NL); sdf = np.exp(np.asarray(Gs["lsd"]).reshape(-1, NC, NL))
    NOBS = O.groupby(["c", "l"]).size().unstack().reindex(index=range(NC), columns=range(NL)).fillna(0).astype(int).values
    rows = []
    for c in range(NC):
        rows.append(pd.DataFrame({"cell": cells[c], "cpg": loci, "n_obs": NOBS[c], "mu": flat[:, c, :].mean(0), "mu_sd": flat[:, c, :].std(0), "mu_lo": np.percentile(flat[:, c, :], 2.5, 0), "mu_hi": np.percentile(flat[:, c, :], 97.5, 0),
                                  "donor_sd": sdf[:, c, :].mean(0), "rhat": np.asarray(sm["mu"]["r_hat"])[c], "ess": np.asarray(sm["mu"]["n_eff"])[c]}))
    out = pd.concat(rows)
    for col in [c for c in out.columns if c not in ("cell", "cpg", "n_obs")]: out.loc[out.n_obs == 0, col] = np.nan
    idx = np.linspace(0, flat.shape[0] - 1, K).astype(int); DR = flat[idx].astype(np.float16); SDR = sdf[idx].astype(np.float16)
    DR[:, NOBS == 0] = np.nan; SDR[:, NOBS == 0] = np.nan
    chain = (idx // draws).astype(np.int8)
    ex = mc.get_extra_fields(group_by_chain=True); dv = np.asarray(ex["diverging"]); ns = np.asarray(ex["num_steps"])
    warn = {"divergences": int(dv.sum()), "divergences_per_chain": dv.sum(1).astype(int).tolist(), "accept_prob_mean": float(np.asarray(ex["accept_prob"]).mean()),
            "draws_at_max_tree_depth": int((ns >= 2 ** MAXDEPTH - 1).sum()), "wall_s": round(wall, 1), "loci": NL, "cells": NC, "seed": int(seed), "mode": mode, "K": K}
    prior = pd.DataFrame({"cpg": loci, "nu": np.asarray(Gs["nu"]).reshape(-1, NL).mean(0), "tau": float(np.asarray(Gs["tau"]).mean())})
    return dict(summary=out, draws=DR, sd_draws=SDR, chain=chain, cells=np.array(cells), cpg=np.array(loci), prior=prior, warnings=warn,
                rhat_p99=float(np.nanpercentile(out.rhat, 99)), ess_p01=float(np.nanpercentile(out.ess, 1)))

def write_block(R, dst):
    """Every declared output of one block, written atomically: <dst>.parquet, _draws.npz, _prior.parquet, _warnings.json."""
    base = dst[:-len(".parquet")]
    np.savez_compressed(base + "_draws.tmp.npz", draws=R["draws"], sd_draws=R["sd_draws"], chain=R["chain"], cells=R["cells"], cpg=R["cpg"]); os.replace(base + "_draws.tmp.npz", base + "_draws.npz")
    R["prior"].to_parquet(base + "_prior.parquet")
    json.dump(dict(R["warnings"], rhat_p99=R["rhat_p99"], ess_p01=R["ess_p01"]), open(base + "_warnings.json", "w"), indent=1)
    R["summary"].to_parquet(dst + ".tmp"); os.replace(dst + ".tmp", dst)
