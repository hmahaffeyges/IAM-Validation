#!/usr/bin/env python3
"""stage_seq_values.py - a methylation value at each IAM-Atlas v2.1 site from sequencing molecules (DEVELOPMENT, written 2026-10-22, to-do step 40).

NOT COMMISSIONED. The array readers (the IAM-Deconvolver, Met-A against the atlas, Met-A for each cell, the globe) read one value per atlas site.
A sequencing run gives molecules instead. Here each molecule's calls are counted at the exact CpG each atlas site names (the frozen key
atlas/v2.1/sites_sequencing_key.parquet: an EPIC v1 probe name, its hg19 position and its CpG index in the .pat files' numbering), and the value
is methylated calls over calls, with the number of calls kept beside it. Every line of the .pat file counts here (IAM-A's own count keeps its line
split; this module does not touch it). A site with fewer than min_calls calls has no value: the caller sets min_calls, and the readings that use
these values report what depth they needed (author 2026-10-22: "let the instrument tell us...not our assumptions").

  key()                        the frozen site key, as arrays sorted by CpG index
  values_from_pat(path, min_calls=1, keep=None)
                               Series beta by probe, Series calls by probe, and a summary (sites with a value, calls per site); keep, if given,
                               is a function of the line number that says whether a line counts (for thinning tests)
"""
import os, gzip
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); KEY = os.path.join(HERE, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")
FLAG = "values from sequencing molecules at IAM-Atlas v2.1's sites; not commissioned by our standards (development, to-do step 40)"
_k = {}

def key():
    if not _k:
        K = pd.read_parquet(KEY).drop_duplicates("probe").sort_values("loyfer_cpg_index")
        _k.update(idx=K["loyfer_cpg_index"].to_numpy(np.int64), probe=K["probe"].to_numpy(str))
    return _k

def values_from_pat(path, min_calls=1, keep=None):
    k = key(); idx = k["idx"]; n = len(idx); M = np.zeros(n, np.int64); C = np.zeros(n, np.int64)
    op = gzip.open if str(path).endswith(".gz") else open; buf_p, buf_m, buf_w = [], [], []; lines = 0
    def flush():
        if not buf_p: return
        p = np.array(buf_p, np.int64); m = np.array(buf_m, bool); w = np.array(buf_w, np.int64); j = np.searchsorted(idx, p)
        ok = (j < n) & (idx[np.minimum(j, n - 1)] == p); np.add.at(C, j[ok], w[ok]); np.add.at(M, j[ok & m], w[ok & m])
        buf_p.clear(); buf_m.clear(); buf_w.clear()
    with op(path, "rt") as fh:
        for ln, line in enumerate(fh):
            if keep is not None and not keep(ln): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 4: continue
            i, pat, w = int(f[1]), f[2], int(f[3]); lines += 1
            for q, x in enumerate(pat):
                if x == "C" or x == "T": buf_p.append(i + q); buf_m.append(x == "C"); buf_w.append(w)
            if len(buf_p) > 5_000_000: flush()
    flush()
    has = C >= max(1, int(min_calls))
    beta = pd.Series(M[has] / C[has], index=k["probe"][has], dtype="float64"); calls = pd.Series(C[has], index=k["probe"][has], dtype="int64")
    cc = C[C > 0]
    summary = dict(flag=FLAG, source=os.path.basename(str(path)), lines_counted=lines, atlas_sites=int(n), min_calls=int(min_calls),
                   sites_with_any_call=int((C > 0).sum()), sites_with_value=int(has.sum()),
                   calls_per_site_median=float(np.median(cc)) if len(cc) else 0.0, calls_per_site_p10=float(np.percentile(cc, 10)) if len(cc) else 0.0)
    return beta, calls, summary

def thin_pat(path, frac, dest, seed=20261022):
    """TESTS ONLY: a thinned copy of a .pat(.gz) file, each molecule kept with probability frac (a line of count w keeps Binomial(w, frac)), seeded,
    so a deep sample stands in for a shallower one (to-do step 40: the lowest depth each instrument can read). Returns the path and the counts."""
    rng = np.random.default_rng(seed); op = gzip.open if str(path).endswith(".gz") else open; kin = kout = 0
    with op(path, "rt") as fh, gzip.open(dest, "wt") as o:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4: continue
            w = int(f[3]); k = int(rng.binomial(w, frac)); kin += w
            if k: f[3] = str(k); o.write("\t".join(f) + "\n"); kout += k
    return dest, dict(fraction=frac, seed=seed, molecules_in=kin, molecules_kept=kout)


HG19_CPGS = 28217448   # CpGs in the hg19 reference numbering the .pat and .beta files share (stated; a file of another length is refused)
def values_from_beta(path, min_calls=1):
    """A value at each atlas site from a .beta file (2026-10-22, to-do 40.10): the standard binary per-CpG count file (two numbers per CpG in the
    hg19 CpG numbering, methylated reads and all reads; uint8, or uint16 for .lbeta), read at the CpG index each atlas site names in the frozen
    key. Gives the same three things as values_from_pat. Per-site counts only: no molecules, so IAM-A cannot read it."""
    raw = (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")).read()
    dt = np.uint16 if ".lbeta" in str(path) else np.uint8; a = np.frombuffer(raw, dtype=dt)
    if a.size % 2 or a.size // 2 != HG19_CPGS:
        raise ValueError(f"{os.path.basename(path)}: {a.size // 2:,} CpGs, not the {HG19_CPGS:,} of the hg19 numbering the atlas key uses")
    a = a.reshape(-1, 2).astype(np.int64); k = key(); r = a[k["idx"] - 1]   # CpG index is 1-based
    M, C = r[:, 0], r[:, 1]; ok = C >= max(1, int(min_calls))
    beta = pd.Series(np.where(ok, M / np.maximum(C, 1), np.nan), index=k["probe"]).dropna()
    calls = pd.Series(C, index=k["probe"])[beta.index]
    return beta, calls, dict(source=os.path.basename(path), format=".beta (per-CpG counts)", atlas_sites=int(len(k["idx"])), sites_with_value=int(ok.sum()),
                             calls_per_site_median=float(np.median(C[ok])) if ok.any() else 0.0, min_calls=int(min_calls), flag=FLAG)
