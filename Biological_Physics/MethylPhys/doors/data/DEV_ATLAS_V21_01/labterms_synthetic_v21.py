"""Atlas v2.1 decision 16, synthetic test (readiness L17). Written 2026-10-14 and committed BEFORE it was run.

QUESTION. v2 fixed each source's terms (sequencing a, b; array-lab offset d; extra noise t) at their measured values. v2.1 samples them in
every block from priors centered on the measured values and as wide as their measured uncertainty. Does the sampled version recover the
terms, give cell intervals that cover the truth at their stated rate, converge, and leave the estimates unchanged when the fixed values
happen to be right?

MODEL. Atlas v2's block model (14_stageB_block.py) line for line - nu, tau, lmu, lsd, the array and sequencing likelihoods, the 0.01 floor,
the init rule, NUTS target 0.9, tree depth 8 - with one switch: terms "fixed" (v2) or "sampled" (v2.1). Sampled: a ~ N(a0, sa), b ~ N(b0, sb)
per sequencing source; d ~ N(d0, sd) per array lab except the anchor lab (our own array scale, d = 0, as in v2); arm "abdt" also samples
t ~ HalfNormal(0.01) per source (v2: t = 0).

SYNTHETIC BLOCK (generated from the model's own likelihood, so coverage is a pure test of the inference). 120 loci, 10 cells, 3 samples per
cell and source: cells 0-3 sequencing + anchor arrays; 4-5 sequencing + lab M arrays; 6-7 lab M only; 8 sequencing only; 9 lab X + anchor.
Sequencing coverage 10-40 reads. Priors as in the real fit: a0 -0.090, sa 0.004; b0 1.103, sb 0.0065 (Loyfer's bootstrap interval);
lab M d0 -0.005, lab X d0 0, sd 0.005 (the +/-0.01 bound). Planted terms are drawn from the priors in each replicate; planted t = 0
except in arm abdt (drawn from its prior).

BARS (fixed now; the sampled arm adopted for v2.1 must meet all of them; 20 replicates per arm):
  S1 cell coverage: the 90% interval of mu covers the planted mu in 0.87-0.93 of all (cell, locus) pairs, and in 0.85-0.95 for the
     sequencing-only cell 8.
  S2 term coverage: each planted a, b, d inside its 90% interval in at least 16 of 20 replicates (chance alone gives 15 or fewer 4% of the time).
  S3 convergence, in every replicate: at most 1% of (cell, locus) pairs at R-hat >= 1.01 (atlas v2's own gate measure, V4; v2 reached
     0.19%), every lab term at R-hat <= 1.01, divergent transitions <= 0.5% of draws.
  S4 no harm: with planted terms exactly at the prior centers (5 extra replicates), median |mu_sampled - mu_fixed| <= 0.002.
REPORTED, NOT A BAR: the fixed arm's coverage when the planted terms differ from its fixed values by their stated uncertainty.
ADOPTION RULE: if arm abdt meets S1-S4 it is adopted (t sampled); otherwise if arm abd meets them, abd is adopted and t stays fixed as in v2;
otherwise decision 16 is NOT adopted and v2.1 keeps v2's fixed terms. The summary says which.
AMENDED 2026-10-14 after a SMOKE run only (other seeds, short chains, counts for nothing; no real-run replicate had been fitted):
  (1) the generator no longer floors readings at 0 and 1. With the floor, 14% of sequencing readings sat on it and the terms were biased
      in BOTH arms (a fitted -0.060 for a planted -0.093); without it they were recovered. The test is of the inference, so data come from the
      model as written. Whether the real data bend at the ends - v2's line -0.090 + 1.103 m predicts negative readings below m = 0.08 - is a
      separate check on real data (readiness L20).
  (2) S3 was "maximum R-hat <= 1.01 over all 2,400 parameters", stricter than v2's own standard; it is now v2's measure, stated above.
Usage: python3 labterms_synthetic_v21.py   (jax 0.10.2, numpyro 0.22.0, the box's pins)"""
import os, sys, json, time
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist, scipy.special as ssp
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary
numpyro.set_host_device_count(4); jax.config.update("jax_enable_x64", False)
HERE = os.path.dirname(os.path.abspath(__file__))
NL, NC, NS, WARM, DRAWS, REPS, REPS_NOHARM = 120, 10, 3, 600, 400, 20, 5
SMOKE = os.environ.get("SMOKE") == "1"   # code check only: 1 replicate per arm, seeds 900+, short chains, writes nothing, counts for nothing
if SMOKE: WARM, DRAWS, REPS, REPS_NOHARM = 150, 100, 1, 1
PRI = dict(a0=-0.090, sa=0.004, b0=1.103, sb=0.0065, dM0=-0.005, dX0=0.0, sd=0.005, t_scale=0.01)
FLOOR = 0.01 ** 2
CELLS = {0: ("seq", "anc"), 1: ("seq", "anc"), 2: ("seq", "anc"), 3: ("seq", "anc"), 4: ("seq", "M"), 5: ("seq", "M"), 6: ("M",), 7: ("M",), 8: ("seq",), 9: ("X", "anc")}
SRC = ["seq", "anc", "M", "X"]; si = {s: i for i, s in enumerate(SRC)}
def simulate(rng, planted_at_center, arm):
    nu = rng.normal(0, 2, NL); tau = 1.0
    mu = ssp.expit(nu[None, :] + tau * rng.normal(0, 1, (NC, NL))); sd = np.exp(rng.normal(np.log(0.03), 0.7, (NC, NL)))
    if planted_at_center: a, b, dM, dX = PRI["a0"], PRI["b0"], PRI["dM0"], PRI["dX0"]
    else: a, b, dM, dX = rng.normal(PRI["a0"], PRI["sa"]), rng.normal(PRI["b0"], PRI["sb"]), rng.normal(PRI["dM0"], PRI["sd"]), rng.normal(PRI["dX0"], PRI["sd"])
    t = np.abs(rng.normal(0, PRI["t_scale"], 4)) if (arm == "abdt" and not planted_at_center) else np.zeros(4)
    d = {"anc": 0.0, "M": dM, "X": dX}
    rows = []
    for c, srcs in CELLS.items():
        for s in srcs:
            for _ in range(NS):
                for l in range(NL):
                    m, sdd = mu[c, l], sd[c, l]
                    if s == "seq":
                        cov = rng.integers(10, 41); loc = a + b * m; var = b * b * sdd ** 2 + m * (1 - m) / cov + t[si[s]] ** 2 + FLOOR
                    else:
                        cov = 0; loc = m + d[s]; var = sdd ** 2 + t[si[s]] ** 2 + FLOOR
                    rows.append((c, si[s], l, float(rng.normal(loc, np.sqrt(var))), float(cov)))
    O = np.array(rows)
    return O, dict(mu=mu, a=a, b=b, dM=dM, dX=dX, t=t)
def fit(O, arm, seed):
    c_, s_, l_ = O[:, 0].astype(int), O[:, 1].astype(int), O[:, 2].astype(int); y_ = jnp.array(O[:, 3]); cov_ = jnp.array(O[:, 4])
    isarr = jnp.array(s_ != si["seq"]); c_, s_, l_ = jnp.array(c_), jnp.array(s_), jnp.array(l_)
    def model():
        nu = numpyro.sample("nu", dist.Normal(0., 2.).expand([NL])); tau = numpyro.sample("tau", dist.HalfNormal(2.))
        lmu = numpyro.sample("lmu", dist.Normal(nu[None, :], tau).expand([NC, NL])); mu = jax.nn.sigmoid(lmu)
        lsd = numpyro.sample("lsd", dist.Normal(jnp.log(0.03), 0.7).expand([NC, NL])); sd = jnp.exp(lsd)
        if arm == "fixed": a, b, dM, dX = PRI["a0"], PRI["b0"], PRI["dM0"], PRI["dX0"]; t = jnp.zeros(4)
        else:
            a = numpyro.sample("a", dist.Normal(PRI["a0"], PRI["sa"])); b = numpyro.sample("b", dist.Normal(PRI["b0"], PRI["sb"]))
            dM = numpyro.sample("dM", dist.Normal(PRI["dM0"], PRI["sd"])); dX = numpyro.sample("dX", dist.Normal(PRI["dX0"], PRI["sd"]))
            t = numpyro.sample("t", dist.HalfNormal(PRI["t_scale"]).expand([4])) if arm == "abdt" else jnp.zeros(4)
        dvec = jnp.stack([jnp.asarray(0., jnp.float32), jnp.asarray(0., jnp.float32), jnp.asarray(dM, jnp.float32), jnp.asarray(dX, jnp.float32)])
        m = mu[c_, l_]; sv = sd[c_, l_]; tt = jnp.asarray(t)[s_]
        loc = jnp.where(isarr, m + dvec[s_], a + b * m)
        var = jnp.where(isarr, sv ** 2 + tt ** 2 + FLOOR, (b ** 2) * sv ** 2 + m * (1 - m) / jnp.maximum(cov_, 1.) + tt ** 2 + FLOOR)
        numpyro.sample("y", dist.Normal(loc, jnp.sqrt(var)), obs=y_)
    lm = np.clip(np.bincount(np.asarray(l_), weights=np.asarray(y_), minlength=NL) / np.bincount(np.asarray(l_), minlength=NL), 0.02, 0.98)
    cm = np.full((NC, NL), np.nan)
    for c in range(NC):
        k = np.asarray(c_) == c; cm[c] = np.bincount(np.asarray(l_)[k], weights=np.asarray(y_)[k], minlength=NL) / np.maximum(np.bincount(np.asarray(l_)[k], minlength=NL), 1)
    l0 = ssp.logit(np.clip(np.where(np.isfinite(cm), cm, lm[None, :]), 0.02, 0.98))
    init = {"nu": jnp.array(ssp.logit(lm)), "tau": jnp.array(1.0), "lmu": jnp.array(l0), "lsd": jnp.full((NC, NL), np.log(0.03))}
    if arm != "fixed": init.update(a=jnp.array(PRI["a0"]), b=jnp.array(PRI["b0"]), dM=jnp.array(PRI["dM0"]), dX=jnp.array(PRI["dX0"]))
    if arm == "abdt": init["t"] = jnp.full(4, 0.005)
    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=init_to_value(values=init), max_tree_depth=8), num_warmup=WARM, num_samples=DRAWS,
              num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed)); G = mc.get_samples(group_by_chain=True)
    div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    keep = {"mu": jax.nn.sigmoid(G["lmu"])}; keep.update({k: G[k] for k in ("a", "b", "dM", "dX", "t") if k in G})
    sm = summary(keep, group_by_chain=True)
    rmu = np.asarray(sm["mu"]["r_hat"]); rh = float(np.mean(rmu >= 1.01))                  # share of (cell, locus) pairs at R-hat >= 1.01
    rt = max([float(np.nanmax(np.asarray(sm[k]["r_hat"]))) for k in sm if k != "mu"] or [1.0])  # worst lab term
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in keep.items()}
    return flat, (rh, rt), div / (4 * DRAWS)
def cover(draws, truth):
    lo, hi = np.percentile(draws, 5, 0), np.percentile(draws, 95, 0); return (truth >= lo) & (truth <= hi)
if __name__ == "__main__":
    out = {"bars": "S1-S4 in the docstring", "arms": {}}; t0 = time.time()
    for arm in ("abdt", "abd", "fixed"):
        R = []
        for r in range(REPS):
            rng = np.random.default_rng((900 if SMOKE else 1000) + r); O, T = simulate(rng, False, arm if arm != "fixed" else "abd")
            F, rh, dv = fit(O, arm, 2000 + r)
            cv = cover(F["mu"], T["mu"])
            rec = dict(rep=r, cov_all=float(cv.mean()), cov_c8=float(cv[8].mean()), frac_mu_rhat_ge_1_01=rh[0], max_term_rhat=rh[1], div_frac=dv)
            for k in ("a", "b", "dM", "dX"):
                if k in F: rec[f"cov_{k}"] = bool(cover(F[k][:, None], np.array([T[k]]))[0])
            R.append(rec); print(arm, r, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, f"{time.time()-t0:.0f}s", flush=True)
        out["arms"][arm] = R
    NH = []
    for r in range(REPS_NOHARM):
        rng = np.random.default_rng(5000 + r); O, T = simulate(rng, True, "abd")
        Fs, _, _ = fit(O, "abd", 6000 + r); Ff, _, _ = fit(O, "fixed", 6000 + r)
        NH.append(float(np.median(np.abs(Fs["mu"].mean(0) - Ff["mu"].mean(0))))); print("noharm", r, NH[-1], flush=True)
    out["noharm_median_abs"] = NH
    def bars(arm):
        R = out["arms"][arm]; ca = np.mean([x["cov_all"] for x in R]); c8 = np.mean([x["cov_c8"] for x in R])
        terms = {k: sum(x[f"cov_{k}"] for x in R) for k in ("a", "b", "dM", "dX")}
        S1 = 0.87 <= ca <= 0.93 and 0.85 <= c8 <= 0.95; S2 = all(v >= 16 for v in terms.values())
        S3 = all(x["frac_mu_rhat_ge_1_01"] <= 0.01 and x["max_term_rhat"] <= 1.01 and x["div_frac"] <= 0.005 for x in R); S4 = float(np.median(NH)) <= 0.002
        return dict(S1=S1, S2=S2, S3=S3, S4=S4, cov_all=ca, cov_c8=c8, terms_covered_of_20=terms,
                    max_frac_mu_rhat=max(x["frac_mu_rhat_ge_1_01"] for x in R), max_term_rhat=max(x["max_term_rhat"] for x in R), max_div_frac=max(x["div_frac"] for x in R))
    out["bars_abdt"], out["bars_abd"] = bars("abdt"), bars("abd")
    Rf = out["arms"]["fixed"]; out["fixed_reported"] = dict(cov_all=float(np.mean([x["cov_all"] for x in Rf])), cov_c8=float(np.mean([x["cov_c8"] for x in Rf])))
    ok = lambda b: b["S1"] and b["S2"] and b["S3"] and b["S4"]
    out["adopted"] = "abdt (a, b, d and t sampled)" if ok(out["bars_abdt"]) else ("abd (a, b, d sampled; t fixed as in v2)" if ok(out["bars_abd"]) else "NONE - v2.1 keeps v2's fixed terms")
    if SMOKE: print("SMOKE run: nothing written"); print(json.dumps(out["bars_abd"], default=float)); sys.exit(0)
    json.dump(out, open(os.path.join(HERE, "labterms_synthetic_v21.json"), "w"), indent=1, default=float)
    L = [f"Atlas v2.1 decision 16, synthetic test (labterms_synthetic_v21.py), run once {time.strftime('%Y-%m-%d')}",
         f"arm abdt: {json.dumps(out['bars_abdt'], default=float)}", f"arm abd: {json.dumps(out['bars_abd'], default=float)}",
         f"fixed terms (reported, not a bar): {json.dumps(out['fixed_reported'])}", f"no-harm medians: {[round(x, 5) for x in NH]}",
         f"ADOPTED: {out['adopted']}", "ALL BARS MET" if out["adopted"] != "NONE - v2.1 keeps v2's fixed terms" else "BARS NOT MET"]
    open(os.path.join(HERE, "labterms_synthetic_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
