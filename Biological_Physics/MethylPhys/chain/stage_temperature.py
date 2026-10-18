#!/usr/bin/env python3
"""Chain stage TL: the temperature law on single molecules (PRED-TEMPERATURE-01, PHY-057). Written 2026-10-17, before any read is opened.

The instrument's job: a plant grown colder reads a lower held error, by the amount the law fixes from the two temperatures alone.
  law          ln[(1 - eps)/eps] = E/kT with E fixed  ->  log-odds(cold) / log-odds(warm) = T_warm / T_cold   (nothing fitted)
  count        the chain's own counter (stage_q2_selfref.count_bam_region, Bismark XM), on the nuclear chromosomes only
  meter        a site reads lost if it was lost and converted (eps(1 - c), c = conversion failure, read on the chloroplast) or held and misread
               (s(1 - eps), s = the sequencer's own error on A/T bases), independently: p = eps(1 - c) + s(1 - eps).
               The counter's statistic is a known function g(p) of that per-site rate, set by the sample's own mix of calls per molecule
               (n_hist); p = g^-1(held_err) is read through that response curve, then eps = (p - s)/(1 - c - s). (T2 synthetic run 1, 2026-10-17:
               the plain subtraction held_err - s used for the tumor pairs over-corrects by about 9 % here, more than the effect sought.)
  conversion   read on the chloroplast, which carries no methylation: the share of its CpG calls read methylated is the conversion failure
               (in plants the non-CpG contexts are methylated, so the human non-CpG check does not apply)
  pairs        one ratio per accession grown at both temperatures; the reading is the paired median with a bootstrap 95 % interval
Nothing here is fitted; the predicted ratio comes from the two temperatures."""
import os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stage_q2_selfref as Q2

NUCLEAR = ["Chr1", "Chr2", "Chr3", "Chr4", "Chr5"]; PLASTID = "ChrC"   # TAIR10 names
NAMES = {"tair": (NUCLEAR, "ChrC"), "ensembl": (["1", "2", "3", "4", "5"], "Pt")}   # the same TAIR10 assembly as TAIR and Ensembl Plants name it
def names(refs, chroms=None):
    for k, (n, p) in NAMES.items():
        if all(c in refs for c in (chroms or n)) and (chroms is None or p in refs or all(c in n for c in chroms)): return n, p
    raise ValueError("reference carries neither TAIR (Chr1-5) nor Ensembl (1-5) chromosome names")
MIN_QUALIFYING, MAX_CONV_FAIL = 20_000, 0.01                           # per-sample gates (T2 synthetic: 20,000 molecules tell 18 C from 28 C)

def predicted_ratio(t_cold_c, t_warm_c): return (273.15 + t_warm_c) / (273.15 + t_cold_c)
def logodds(e): return float(np.log((1 - e) / e))

GRID = np.round(np.arange(0.002, 0.1201, 0.002), 4)
def response(nh, grid=GRID, m=None, seed=None):
    """The counter's held-error statistic g(p) for qualifying molecules with this sample's calls-per-molecule mix nh (counts of QUALIFYING
    molecules by number of calls), each call lost with probability p, by the counter's own rule (Q2: at least QUAL_MIN_CALLS calls, at least
    QUAL_MIN_METH of them held; isolated losses over interior calls). Exact (2026-10-18, LESSONS C23): with k losses among n calls placed at
    random, an interior call is an isolated loss with probability k(n-k)(n-k-1) / (n(n-1)(n-2)); summed over the qualifying k by the binomial.
    (m and seed are kept for callers of the earlier simulated curve and are not used.)"""
    from math import comb
    ns = np.array(sorted(k for k in nh if k >= Q2.QUAL_MIN_CALLS)); w = np.array([nh[k] for k in ns], float)
    if not len(ns): return grid, None
    g = []
    for p in grid:
        iso = opp = 0.0
        for n, wt in zip(ns, w):
            n = int(n); kmax = int(np.floor(n - Q2.QUAL_MIN_METH * n + 1e-9))
            k = np.arange(0, kmax + 1)
            pk = np.array([comb(n, int(x)) for x in k], float) * p ** k * (1 - p) ** (n - k); pq = pk.sum()
            if pq <= 0: continue
            e = (n - 2) * np.sum(pk * k * (n - k) * (n - k - 1)) / (n * (n - 1) * (n - 2)) / pq   # isolated losses per qualifying molecule
            iso += wt * e; opp += wt * (n - 2)
        g.append(iso / opp if opp else np.nan)
    return grid, np.array(g)
def invert(held_err, nh):
    grid, g = response(nh)
    if g is None: return None
    ok = np.isfinite(g); grid, g = grid[ok], g[ok]                              # points where no molecule qualifies carry no response
    d = np.flatnonzero(np.diff(g) <= 0); k = (d[0] + 1) if len(d) else len(g)   # the rising part: near p = 0.1 qualification flattens it
    grid, g = grid[:k], g[:k]
    if len(g) < 3 or not (g[0] <= held_err <= g[-1]): return None
    return float(np.interp(held_err, g, grid))

def plastid_conversion(bam, fa, chrom=PLASTID):
    """Conversion failure on the chloroplast: methylated CpG calls / all CpG calls, read with the chain's own parser."""
    import pysam
    B = pysam.AlignmentFile(bam, "rb"); F = pysam.FastaFile(fa); m = n = 0
    for r in B.fetch(chrom):
        if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate or r.mapping_quality < 10: continue
        ref0 = r.reference_start - 1; ref = F.fetch(chrom, max(ref0, 0), r.reference_end + 1).upper()
        if ref0 < 0: ref = "N" + ref
        c, _, _ = Q2._calls_of(r, ref, ref0, False); m += sum(c.values()); n += len(c)
    return dict(plastid_calls=int(n), conv_fail=(m / n) if n else None)

def count_plant(bam, fa, workers=4, chroms=None, span=10_000_000):
    """One sample: the chain counter on the nuclear chromosomes, the meter correction, and the chloroplast conversion check."""
    import pysam
    F = pysam.FastaFile(fa); nuc, plastid = names(F.references, chroms); chroms = chroms or nuc
    jobs = [(bam, fa, c, s, min(F.get_reference_length(c), s + span)) for c in chroms for s in range(0, F.get_reference_length(c), span)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as P: parts = P.map(Q2.count_bam_region, jobs)
    else: parts = [Q2.count_bam_region(j) for j in jobs]
    s = collections.Counter(); nh = collections.Counter()
    for p in parts:
        for x in ("e_iso", "e_opp", "molecules", "qualifying", "reads", "at_mm", "at_n"): s[x] += p[x]
        for a, b in p["n_hist"].items(): nh[int(a)] += b
    eps = s["e_iso"] / max(s["e_opp"], 1); sub = s["at_mm"] / max(s["at_n"], 1)
    p_site = invert(eps, nh)
    out = dict(held_err=eps, held_opp=int(s["e_opp"]), qualifying=int(s["qualifying"]), reads=int(s["reads"]), sub_err=sub, n_hist=dict(nh),
               p_site=p_site)
    out.update(plastid_conversion(bam, fa, plastid) if plastid in F.references else dict(plastid_calls=0, conv_fail=None))
    cf = out["conv_fail"] or 0.0
    out["eps_corr"] = (p_site - sub) / (1 - cf - sub) if p_site is not None else None
    why = []
    if out["qualifying"] < MIN_QUALIFYING: why.append(f"{out['qualifying']:,} qualifying molecules (at least {MIN_QUALIFYING:,})")
    if out["conv_fail"] is None or out["conv_fail"] > MAX_CONV_FAIL: why.append(f"chloroplast conversion failure {out['conv_fail']} (at most {MAX_CONV_FAIL})")
    if not (out["eps_corr"] or 0) > 0: why.append("corrected held error not above zero")
    out["usable"] = not why; out["reason"] = "; ".join(why)
    return out

def score(pairs, t_cold_c, t_warm_c, boot=10_000, seed=1, tol=0.25):
    """pairs: [(eps_corr_cold, eps_corr_warm), ...], one per accession grown at both. Bars (KISS, the instrument's job):
      B1 the paired median ratio lands within a quarter of the law's shift: 1 + (pred - 1) * [1 - tol, 1 + tol]
      B2 its 95 % bootstrap interval excludes 1: the colder plants read a lower held error, not the same."""
    R = np.array([logodds(c) / logodds(w) for c, w in pairs]); rng = np.random.default_rng(seed)
    med = float(np.median(R)); bs = np.median(R[rng.integers(0, len(R), (boot, len(R)))], axis=1); lo, hi = (float(x) for x in np.quantile(bs, [0.025, 0.975]))
    pred = predicted_ratio(t_cold_c, t_warm_c); win = [1 + (pred - 1) * (1 - tol), 1 + (pred - 1) * (1 + tol)]
    b1 = bool(win[0] <= med <= win[1]); b2 = bool(lo > 1.0 or hi < 1.0)
    return dict(pairs=len(R), median_ratio=med, ci95=[lo, hi], predicted=pred, window=win, B1_within_quarter_of_law=b1, B2_excludes_no_change=b2, met=b1 and b2)
