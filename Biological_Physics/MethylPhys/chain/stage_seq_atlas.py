"""stage_seq_atlas.py - the IAM Sequencing Atlas, one block of the hg19 CpG numbering (DEVELOPMENT; plan 35.30; written 2026-10-29).

Separate from IAM-Atlas v2.1 (arrays), for sequenced samples only. Measured, not fitted: every value is counted from healthy donors' reads.
Input: each healthy donor's .beta file (wgbstools format: methylated and total reads, uint8, at every one of the 28,217,448 hg19 CpGs), read
only over the block, from a local folder or straight from GEO by byte range. Output: one parquet per block, one column per cell and quantity,
uint16 (level-type values x 10,000; 65535 = blank, where no donor of the cell has a read; nothing is filled).

Per cell, at every CpG of the block:
  donors     donors with at least one read (uint8 in a uint16 column)
  reads      all reads, summed over donors (capped at 65534)
  level      median of the donors' own levels (methylated / all, per donor)
  spread     standard deviation of the donors' levels (two or more donors)
  disorder   mean over donors of each donor's binary entropy at the site, in bits (each donor's measured disorder)
  meth       methylated reads, summed over donors (capped at 65534); with reads it gives the pooled level and its exact 95 % range from the
             counts alone (Jeffreys interval: beta quantiles of meth + 0.5 and reads - meth + 0.5; 2026-10-29: stored as the count, not the range)
  halfA, halfB   pooled level in each sealed donor half (IAM-Atlas v2.1 decision 17's split); halfA - halfB is noise map 1
  readA, readB   pooled level when every donor's reads are split in two at random (hypergeometric, seed stated); readA - readB is noise map 2,
                 defined even for a cell with one donor
"""
import os, io, numpy as np, pandas as pd
HG19_CPGS = 28217448; BLANK = 65535; SEED = 20261029
STATS = ("donors", "reads", "meth", "level", "spread", "disorder", "halfA", "halfB", "readA", "readB")

def read_block(src, start, end):
    """Methylated and total reads over CpGs [start, end) (0-based) of one .beta file: a local path, or an https URL read by byte range."""
    n = end - start
    if str(src).startswith("http"):
        import requests
        for t in range(5):
            try:
                r = requests.get(src, headers={"Range": f"bytes={2 * start}-{2 * end - 1}"}, timeout=300)
                if r.status_code == 206 and len(r.content) == 2 * n: break
            except Exception: pass
            import time; time.sleep(15 * (t + 1))
        else: raise IOError(f"{src}: byte range {2 * start}-{2 * end - 1} not returned")
        a = np.frombuffer(r.content, np.uint8)
    else:
        if os.path.getsize(src) != 2 * HG19_CPGS: raise ValueError(f"{src}: not the hg19 numbering ({os.path.getsize(src) // 2:,} CpGs)")
        a = np.fromfile(src, np.uint8, count=2 * n, offset=2 * start)
    a = a.reshape(n, 2); return a[:, 0].astype(np.int32), a[:, 1].astype(np.int32)

def _u16(x, scale=1e4):
    out = np.full(x.shape, BLANK, np.uint16); ok = np.isfinite(x); out[ok] = np.clip(np.round(x[ok] * scale), 0, BLANK - 1); return out

def _H(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where((p <= 0) | (p >= 1), 0.0, h)

def build_block(samples, start, end, source):
    """samples: DataFrame with gsm, cell, half (A or B) and file (local path or URL). Returns (table, summary dict)."""
    rng = np.random.default_rng([SEED, start]); cols = {}; summ = dict(start=start, end=end, cells={}, capped_sites=0)
    for cell, g in samples.groupby("cell", sort=True):
        M, N = [], []
        for f in g.file: m, n = read_block(f, start, end); M.append(m); N.append(n)
        M, N = np.array(M), np.array(N); summ["capped_sites"] += int((N >= 255).sum())
        has = N > 0; nd = has.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"): lv = np.where(has, M / N, np.nan)
        allm, alln = M.sum(0), N.sum(0)
        level = np.nanmedian(np.where(has, lv, np.nan), axis=0) if len(g) > 1 else lv[0]
        spread = np.where(nd >= 2, np.nanstd(lv, axis=0, ddof=1) if len(g) > 1 else np.nan, np.nan)
        dis = np.where(nd > 0, np.nansum(np.where(has, _H(np.nan_to_num(lv)), 0), 0) / np.maximum(nd, 1), np.nan)
        hA = g.half.to_numpy() == "A"
        def pooled(sel):
            mm, nn = M[sel].sum(0), N[sel].sum(0)
            with np.errstate(divide="ignore", invalid="ignore"): return np.where(nn > 0, mm / nn, np.nan)
        n1 = N // 2; m1 = np.where(N > 0, rng.hypergeometric(np.maximum(M, 0), np.maximum(N - M, 0), np.minimum(n1, N)), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rA = np.where(n1.sum(0) > 0, m1.sum(0) / n1.sum(0), np.nan); rB = np.where((N - n1).sum(0) > 0, (M - m1).sum(0) / (N - n1).sum(0), np.nan)
        blank = nd == 0
        vals = dict(level=level, spread=spread, disorder=dis, halfA=pooled(hA), halfB=pooled(~hA), readA=rA, readB=rB)
        for k, v in vals.items(): v = np.where(blank, np.nan, v); cols[f"{cell}|{k}"] = _u16(v)
        cols[f"{cell}|donors"] = np.where(blank, BLANK, nd).astype(np.uint16); cols[f"{cell}|reads"] = np.where(blank, BLANK, np.minimum(alln, BLANK - 1)).astype(np.uint16); cols[f"{cell}|meth"] = np.where(blank, BLANK, np.minimum(allm, BLANK - 1)).astype(np.uint16)
        summ["cells"][cell] = dict(donors=len(g), halfA=int(hA.sum()), halfB=int((~hA).sum()), sites_with_reads=int((~blank).sum()), median_reads=float(np.median(alln[~blank])) if (~blank).any() else 0)
    T = pd.DataFrame(cols); T.insert(0, "cpg", np.arange(start + 1, end + 1, dtype=np.int32))   # 1-based hg19 CpG index, as the .pat and .beta numbering
    return T, summ

def geo_url(gsm, title):
    """The hg19 .beta file of one published sequencing sample on GEO."""
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{gsm}_{title}.beta"
