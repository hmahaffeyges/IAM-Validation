"""The copy process of a sequencing library, measured from the sample itself (DEV-ONE-DRAW-01, development; candidate for the chain).
copy_distribution: at the library-floor sites (every healthy cell holds them unmarked) a false molecule is rare, so the calls at a site with
a false call are mostly one molecule's copies. Sites holding two false molecules are removed by deconvolution: with p the false-call rate and
each site's molecules n / mean copies, the share of sites with a false call that hold two is known (Poisson), and the observed counts are
(1 - pi2) q + pi2 (q * q); solved for q by iteration. Nothing fitted.
expected_H_copies: the expected disorder H(k / n) at a site of healthy level e with n calls, when the calls are copies drawn from q: each
site's molecules Poisson(n / mean(q)), each molecule methylated with probability e and copied by q; tabulated by e (steps of 0.01) and the
simulated call count, and taken as a ratio to the same simulation without copies times the exact binomial expectation (expected_H_counts),
so the tabulation's own binning cancels."""
import numpy as np

def Hb(x):
    x = np.clip(np.asarray(x, "float64"), 1e-12, 1 - 1e-12)
    return np.where((x <= 1e-9) | (x >= 1 - 1e-9), 0.0, -(x * np.log2(x) + (1 - x) * np.log2(1 - x)))

def copy_distribution(m, n, iters=6, CMAX=2000):
    m = np.asarray(m, np.int64); n = np.asarray(n, np.int64); hit = m >= 1
    if hit.sum() < 30: return None
    p = m.sum() / max(n.sum(), 1); g = np.bincount(np.minimum(m[hit], CMAX), minlength=CMAX + 1).astype("float64"); g /= g.sum(); q = g.copy(); pi2 = 0.0
    for _ in range(iters):
        Ec = float((np.arange(len(q)) * q).sum()); lam = p * n[hit] / max(Ec, 1.0)
        pi2 = float((np.exp(-lam) * lam ** 2 / 2).sum() / (1 - np.exp(-lam)).sum())
        qq = np.convolve(q, q)[: len(q)]; q = np.clip((g - pi2 * qq) / (1 - pi2), 0, None); q[0] = 0; q /= q.sum()
    return dict(q=q, mean=float((np.arange(len(q)) * q).sum()), sites=int(hit.sum()), two_molecule_share=round(pi2, 5), rate=float(p))

def _tab(es, n, qv, qp, rng, reps, NMAX, eb):
    S = np.zeros((101, NMAX + 1)); N = np.zeros((101, NMAX + 1)); Ec = float((qv * qp).sum())
    for _ in range(reps):
        m1 = rng.poisson(n / Ec); tot = int(m1.sum()); c = rng.choice(qv, tot, p=qp); meth = rng.random(tot) < np.repeat(es, m1)
        site = np.repeat(np.arange(len(n)), m1); ns = np.bincount(site, c, len(n)).astype(np.int64); ks = np.bincount(site, c * meth, len(n))
        g = (ns >= 1) & (ns <= NMAX); np.add.at(S, (eb[g], ns[g]), Hb(ks[g] / ns[g])); np.add.at(N, (eb[g], ns[g]), 1)
    return S, N

def expected_H_copies(es, n, cd, expected_H_counts, seed=20261106, reps=3, NMAX=400):
    es = np.asarray(es, "float64"); n = np.asarray(n, np.int64); q = cd["q"]; qv = np.flatnonzero(q); qp = q[qv] / q[qv].sum()
    eb = np.clip(np.rint(es * 100).astype(np.int64), 0, 100); nn = np.clip(n, 0, NMAX)
    S1, N1 = _tab(es, n, qv, qp, np.random.default_rng(seed), reps, NMAX, eb)
    S0, N0 = _tab(es, n, np.array([1]), np.array([1.0]), np.random.default_rng(seed + 1), reps, NMAX, eb)
    T1 = np.where(N1 >= 20, S1 / np.maximum(N1, 1), np.nan); T0 = np.where(N0 >= 20, S0 / np.maximum(N0, 1), np.nan)
    r = T1[eb, nn] / T0[eb, nn]; ex = np.asarray(expected_H_counts(es, n), "float64"); ok = np.isfinite(r) & (T0[eb, nn] > 0)
    out = np.where(ok, ex * r, np.nan)
    out[~ok] = np.asarray(expected_H_counts(es[~ok], np.maximum(1, np.rint(n[~ok] / cd["mean"])).astype(np.int64)), "float64"); return out
