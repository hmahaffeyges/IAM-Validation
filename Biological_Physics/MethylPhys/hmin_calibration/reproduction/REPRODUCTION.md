# Reproduction check — 2026-09-22

`samplers/gape_mcmc_g002.py` was re-run unmodified. Every floor the live chain divides by came back inside its
own posterior standard deviation.

| class | floor in the running chain | re-run posterior mean | re-run SD | difference |
|---|---|---|---|---|
| cycling | 0.856100 | 0.855928 | 0.007685 | -0.000172 |
| immune | 0.838889 | 0.838855 | 0.006874 | -0.000034 |
| progenitor | 0.852200 | 0.852184 | 0.008593 | -0.000016 |
| secretory | 0.843300 | 0.843321 | 0.008435 | +0.000021 |
| stem_adult | 0.873700 | 0.873857 | 0.007796 | +0.000157 |
| stem_pluri | 0.982200 | 0.982189 | 0.008822 | -0.000011 |
| stromal | 0.863000 | 0.862755 | 0.008658 | -0.000245 |
| terminal | 0.772800 | 0.772950 | 0.006895 | +0.000150 |

Largest absolute difference **0.000245**, against posterior SDs of 0.0069 to 0.0088.
Gelman-Rubin R-hat was below 1.001 on all eight parameters across five chains. Total runtime 13 seconds.

## Read the script's own verdict carefully

The run prints **'All consistent: NO — see TENSION flags'** and shows the immune class at Δ = +6.37σ. That
comparison is against the *pre-calibration published* values that were current before this calibration, not
against the floors the chain uses now. The immune entry is the documented move from 0.795040 to 0.838889, which
happened when six immune cell types replaced neutrophils alone. Against the floors actually in use, shown in the
table above, there is no tension at all.

## What is and is not reproducible here

Each chain's *starting positions* are deterministic: `run_chain` seeds its generator with `chain_id * 42`. The
MCMC proposals are not seeded, so a re-run is **not bit-identical** — it reproduces the posterior, not the
samples. That is why this deposit ships a comparison of posterior means against their SDs rather than a hash.

## Environment of the verified run

Python 3.11.16, numpy 2.4.6, emcee 3.1.6. The versions used for the
original April 2026 run were not recorded; these are the versions under which the reproduction above succeeded.
