"""DEV-WRITER-02 step 1 (2026-10-10, before any per-context copy error is read): can the copy error healthy cells hold be predicted,
context by context, from the writer's discrimination measured outside the cell?
Prediction under test (Model A', nothing fitted): in flanking context c, eps_c = k / (1 + D_c), with D_c the DNMT1 HM/UM specificity of the
256 NNCGNN contexts (Adam et al. 2023, pairwise ratios of their k_NNCGNN; 29 to >300, average 87) and k the number of chances to lose the site
per copy. k = 1 is the writer's single step; the whole-genome numbers (DEV-WRITER-01: eps 0.032 against a single-step limit 0.0155) say k ~ 2.
Null: eps_c does not depend on D_c.
Test: ordinary least squares of log eps_c on log(1/(1+D_c)) across the 256 contexts, one fit per sample, then the 153 samples' slopes.
Slope 1 = the writer sets the copy error in every context; slope 0 = it does not. k is read off the intercept AFTER the slope is tested.
Noise modelled: (i) counting, binomial on the real per-context opportunity counts; (ii) a per-context nuisance (conversion, sequencing error,
mappability, CpG density) as a lognormal factor on eps, the same in every sample, size sigma_ctx; (iii) the part of that nuisance CORRELATED
with D, size rho, which is what could fake or mask the signal.
Counts: whole-file opportunities per sample 16-20 M (chain_tests/iama_floor_granulocytes.csv), spread over 256 contexts with the uneven
genomic frequency of NNCGNN (simulated as a lognormal with spread 0.8, so the rarest contexts hold a few thousand).
Usage: python3 writer_context_01.py"""
import numpy as np
rg = np.random.default_rng(20261010)
NC, NS, OPP = 256, 153, 18e6
D = np.exp(rg.uniform(np.log(29), np.log(330), NC))                      # replaced by Adam's 256 measured ratios at scoring
w = np.exp(rg.normal(0, 0.8, NC)); w /= w.sum(); n = np.maximum((OPP * w).astype(int), 50)
x = np.log(1 / (1 + D)); x = x - x.mean()
def one(slope_true, k, sigma_ctx, rho):
    """Returns the fitted slope per sample and its spread across samples."""
    u = rg.normal(0, 1, NC); u = u - u.mean()
    ctx = sigma_ctx * (rho * (x / x.std()) + np.sqrt(max(0.0, 1 - rho ** 2)) * u)   # nuisance, fixed across samples
    base = k / (1 + D) if slope_true else np.full(NC, k / (1 + np.exp(np.average(np.log(D), weights=n))))
    sl = []
    for _ in range(NS):
        eps = np.clip(base * np.exp(ctx + rg.normal(0, 0.05)), 1e-5, 0.5)           # per-sample level varies 5 %
        obs = rg.binomial(n, eps) / n; ok = obs > 0
        sl.append(np.polyfit(x[ok], np.log(obs[ok]), 1)[0])
    return np.array(sl)
print("truth            sigma_ctx  rho   slope mean   sd across samples   95 % band")
for lab, st_, k in (("writer sets eps", 1, 2.0), ("no dependence ", 0, 2.0)):
    for sigma_ctx, rho in ((0.0, 0.0), (0.3, 0.0), (0.3, 0.5), (0.6, 0.0), (0.6, 0.5), (0.6, 0.9)):
        s = one(st_, k, sigma_ctx, rho)
        print(f"{lab}   {sigma_ctx:8.1f}  {rho:4.1f}   {s.mean():9.3f}   {s.std():15.4f}   {np.percentile(s, 2.5):.3f}-{np.percentile(s, 97.5):.3f}")
print("\nhow big a D-correlated nuisance must be to fake slope 1 when the writer sets nothing:")
for rho in (0.9, 1.0):
    for sigma_ctx in (0.5, 1.0, 1.5, 2.0, 2.34):
        s = one(0, 2.0, sigma_ctx, rho); print(f"  rho {rho:.1f} sigma_ctx {sigma_ctx:.2f} -> slope {s.mean():.3f}")
# CONTROL ARM. A sequence-dependent technical artefact (conversion, sequencing error, mappability) acts on the CALLS, so it shows up in any
# territory. The writer's discrimination acts only where the writer is choosing: methylated (held) molecules. The control measures the same
# slope in UNMETHYLATED territory (molecules with no methylated calls, where the apparent "error" is a C read where T is expected). Under the
# prediction the control slope is 0; under a technical artefact both slopes move together.
print("\ncontrol arm (unmethylated territory), same nuisance, writer absent there:")
for sigma_ctx, rho in ((0.3, 0.0), (0.6, 0.5), (0.6, 0.9), (1.0, 0.9)):
    sm = one(1, 2.0, sigma_ctx, rho); sc = one(0, 2.0, sigma_ctx, rho)
    print(f"  sigma_ctx {sigma_ctx:.1f} rho {rho:.1f} -> methylated slope {sm.mean():.3f}, control slope {sc.mean():.3f}, difference {sm.mean() - sc.mean():.3f}")
print("\npredicted spread of eps across contexts (k=2):", " ".join(f"{2 / (1 + d):.4f}" for d in np.percentile(D, [0, 25, 50, 75, 100])))
