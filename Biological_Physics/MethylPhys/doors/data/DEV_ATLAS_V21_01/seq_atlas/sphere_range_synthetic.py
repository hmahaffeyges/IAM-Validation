"""The IAM Sequencing Atlas's 95 % range of each cell's average (stage_seq_atlas.range_of_average; checklist D4): synthetic, written and
committed before it runs. The instrument's job: the range a cell's average is given matches how far that average really scatters.
Known answer (seed 20261112): 20,000 sites, each with a true cell level (Beta(0.5, 0.5), the two-ended spread of methylation levels) and a
person-to-person spread around it (logit-normal, sd 0.3, stated); for k = 1, 2 and 3 donors, each counted at Poisson(33) reads, built 200
times; the cell's average is the pooled level, and its truth the mean of the persons' levels it could have drawn (the true cell level's
person average). The stated standard error and range from range_of_average (t quantile with k - 1 degrees of freedom). Bars, for k = 2 and 3, at sites
whose truth lies between 0.1 and 0.9: mean stated variance over the true variance of the pooled level about its truth, within 0.9 to 1.1;
and coverage of the truth by the stated range within 0.90 to 0.99. k = 1 is recorded, not scored: one donor's range is
counting only and does not hold the person-to-person part (stated in the sphere's record).
Usage: python3 sphere_range_synthetic.py OUT.txt"""
import os, sys, numpy as np
from scipy.special import expit, logit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"))); import stage_seq_atlas as Q
rng = np.random.default_rng(20261112); S = 20000; T = 200
mu = np.clip(rng.beta(0.5, 0.5, S), 0.002, 0.998); L = []; res = {}
truth = np.mean(expit(logit(mu)[None, :] + rng.normal(0, 0.3, (4000, S))), 0)   # the persons' average level at each site
for k in (1, 2, 3):
    est = np.empty((T, S)); var = np.empty((T, S)); cov = np.empty((T, S))
    for t in range(T):
        lv = expit(logit(mu)[None, :] + rng.normal(0, 0.3, (k, S))); n = rng.poisson(33, (k, S)).clip(1); m = rng.binomial(n, lv)
        sd = np.std(m / n, axis=0, ddof=1) if k >= 2 else np.full(S, np.nan)
        p, lo, hi, se = Q.range_of_average(m.sum(0), n.sum(0), sd, k); est[t] = p; var[t] = se ** 2; cov[t] = (lo <= truth) & (truth <= hi)
    inner = (truth > 0.1) & (truth < 0.9)   # where the range is not held at 0 or 1
    ratio = float(var[:, inner].mean() / ((est[:, inner] - truth[inner]) ** 2).mean()); c = float(cov[:, inner].mean()); res[k] = (ratio, c)
    L.append(f"k = {k} donors: stated over true variance {ratio:.3f}; coverage of the truth {c:.3f}")
ok = all(0.9 <= res[k][0] <= 1.1 and 0.90 <= res[k][1] <= 0.99 for k in (2, 3))
L = ["The IAM Sequencing Atlas's 95 % range of a cell's average: synthetic (seed 20261112, 20,000 sites, 200 builds, Poisson(33) reads a donor, person sd 0.3 logit)"] + L + \
    [f"bars (k = 2 and 3): variance ratio within 0.9 to 1.1 and coverage within 0.90 to 0.99: {'met' if ok else 'not met'}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
