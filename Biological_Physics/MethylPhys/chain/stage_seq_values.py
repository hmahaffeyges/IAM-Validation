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


def trim_pat(path, n, dest):
    """DEVELOPMENT (2026-11-09, DEV-ATLAS-V21-01): a copy of a .pat(.gz) file with the n outermost CpG calls at each end of every molecule
    left out ('.'), the calls where library chemistry at the read ends lands (the floor by molecule position: GSE98837's end calls read 9
    times its interior floor, and the effect reaches about 4 CpGs in). A molecule left with no call is dropped. Returns the path and counts."""
    op = gzip.open if str(path).endswith(".gz") else open; cin = cout = mol = kept = 0
    with op(path, "rt") as fh, gzip.open(dest, "wt") as o:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4: continue
            w = int(f[3]); p = list(f[2]); idx = [j for j, x in enumerate(p) if x in "CT"]; cin += w * len(idx); mol += w
            for j in idx[:n] + idx[-n:] if n > 0 else []: p[j] = "."
            left = sum(1 for x in p if x in "CT")
            if left: f[2] = "".join(p); o.write("\t".join(f) + "\n"); cout += w * left; kept += w
    return dest, dict(trim_each_end=int(n), molecules_in=mol, molecules_kept=kept, calls_in=cin, calls_kept=cout)


def pat_to_lbeta(path, out, n=0):
    """DEVELOPMENT (2026-11-09, DEV-ATLAS-V21-01): a molecule file (.pat(.gz), hg19 CpG numbering) counted into a per-CpG count file (.lbeta:
    uint16 methylated and all calls at every hg19 CpG, the layout stage_seq_atlas.read_block reads), with the n outermost calls at each end
    of every molecule left out, exactly as trim_pat leaves them out, in one pass. With n = 0 it is the plain count (a check: it must equal the
    published .beta of the same molecules, below that file's cap of 255). Used to count the IAM Sequencing Atlas's own donors the same way a
    trimmed sample is read, so sample and reference take the same path. Returns the path and counts."""
    M = np.zeros(HG19_CPGS, np.uint32); C = np.zeros(HG19_CPGS, np.uint32); op = gzip.open if str(path).endswith(".gz") else open
    bp, bm, bw = [], [], []; mol = cin = cout = 0
    def flush():
        if not bp: return
        p = np.array(bp, np.int64) - 1; m = np.array(bm, bool); w = np.array(bw, np.uint32); ok = (p >= 0) & (p < HG19_CPGS)
        np.add.at(C, p[ok], w[ok]); np.add.at(M, p[ok & m], w[ok & m]); bp.clear(); bm.clear(); bw.clear()
    with op(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4: continue
            i, s, w = int(f[1]), f[2], int(f[3]); idx = [j for j, x in enumerate(s) if x in "CT"]; mol += w; cin += w * len(idx)
            keep = idx[n:len(idx) - n] if n > 0 else idx
            for j in keep: bp.append(i + j); bm.append(s[j] == "C"); bw.append(w)
            cout += w * len(keep)
            if len(bp) > 5_000_000: flush()
    flush(); a = np.empty(2 * HG19_CPGS, np.uint16); a[0::2] = np.minimum(M, 65535); a[1::2] = np.minimum(C, 65535); a.tofile(out)
    return out, dict(trim_each_end=int(n), molecules=mol, calls_in=cin, calls_kept=cout, cpgs_with_calls=int((C > 0).sum()), source=os.path.basename(str(path)), flag=FLAG)

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


def construct_draw(blood, add, depth_buffy, depth_plasma, out_dir, seed=20261022, plant=None):
    """TESTS ONLY (2026-10-22, author: GSE186888 stays unread, "how are you going to explore without one?"): a CONSTRUCTED blood draw with a
    known answer, built from spent values. Buffy coat: the blood's own levels. Plasma: (1 - sum of add) x the blood's levels + each added atlas
    tissue's share x its IAM-Atlas v2.1 best value (where the tissue has none, the blood's). Each half is read as k of n reads at every atlas
    site, n = its depth and k ~ Binomial(n, level), seeded, and written as a .beta count file (hg19 CpG numbering) beside a truth record.
    blood: Series of levels by probe; add: {atlas tissue: share}. plant (to-do 40.11): (cell, loss), a loss of that cell's pattern planted in the
    blood's levels BEFORE the reads are counted, scaled by the cell's share as run_sample.py --dev-plant scales it (the IAM-Deconvolver's six-cell
    shares of the blood), so both halves carry it. Returns the truth record."""
    import json, sys
    sys.path.insert(0, HERE); import stage_metaa_atlas as SA
    prec = None
    if plant:
        import stage_v4 as V4
        b0 = pd.Series(blood, dtype="float64"); sh = {r["cell"]: r["share"] for r in (V4.stage_d21(b0).get("cells") or [])}
        if plant[0] not in sh: raise ValueError(f"plant cell {plant[0]!r} is not among the six blood cells {sorted(sh)}")
        bp = SA.plant(b0, plant[0], plant[1] * sh[plant[0]]); prec = dict(cell=plant[0], loss=plant[1], share=round(sh[plant[0]], 4),
                                                                          applied=round(plant[1] * sh[plant[0]], 6), sites_moved=int((bp != b0).sum()))
        blood = bp
    k = key(); lev = pd.Series(blood, dtype="float64").reindex(k["probe"]).to_numpy(); ok = np.isfinite(lev)
    tot = float(sum(add.values())); pl = lev * (1 - tot)
    for t, f in add.items():
        mu = SA._col("mu", t).reindex(k["probe"]).to_numpy(); pl = pl + f * np.where(np.isfinite(mu), mu, lev)
    rng = np.random.default_rng(seed); os.makedirs(out_dir, exist_ok=True); rec = dict(rule=construct_draw.__doc__.split("\n")[0], added=add, blood_share=round(1 - tot, 6),
                                                                                      depth=dict(buffy=depth_buffy, plasma=depth_plasma), seed=seed, sites=int(ok.sum()), planted=prec, flag="CONSTRUCTED for a test, not a person")
    for half, L, n in (("buffy", lev, depth_buffy), ("plasma", pl, depth_plasma)):
        a = np.zeros((HG19_CPGS, 2), np.uint8); m = rng.binomial(n, np.clip(L[ok], 0, 1))
        a[k["idx"][ok] - 1, 0] = m; a[k["idx"][ok] - 1, 1] = n
        with gzip.open(os.path.join(out_dir, f"{half}.beta.gz"), "wb", compresslevel=1) as f: f.write(a.tobytes())
    json.dump(rec, open(os.path.join(out_dir, "truth.json"), "w"), indent=1); return rec


KEY38 = os.path.join(HERE, "..", "atlas", "v2.1", "sites_key_hg38.parquet")

def values_from_bigwig(call_bw, cov_bw, min_calls=1):
    """A value at each atlas site from a pair of GRCh38 bigWig tracks (2026-10-24, DEV-METAA-HEALTHY-01 step 3): a per-CpG methylation level
    (0 to 1) and its read coverage, the form public consortia (e.g. BLUEPRINT) release whole-genome bisulfite calls in. Each atlas site is read
    at its CpG's GRCh38 position (atlas/v2.1/sites_key_hg38.parquet: the 0-based start of the C); where the track has no value there, at the
    next base (the G, where a caller reports the reverse strand). Gives the same three things as values_from_beta; no molecules, so IAM-A
    cannot read it. Needs pyBigWig (chain requirements)."""
    import pyBigWig
    K = pd.read_parquet(KEY38); bc, bv = pyBigWig.open(call_bw), pyBigWig.open(cov_bw); lv = np.full(len(K), np.nan); cv = np.zeros(len(K)); at_g = 0
    for ch, g in K.groupby("chrom_hg38").groups.items():
        if ch not in bc.chroms(): continue
        iv = bc.intervals(ch) or []; jv = bv.intervals(ch) or []
        L = dict((a, v) for a, b, v in iv); C = dict((a, v) for a, b, v in jv); pos = K.beg0_hg38.to_numpy()[g]
        for k, q in zip(g, pos):
            q = int(q); w = q if q in L else (q + 1 if (q + 1) in L else None)
            if w is None: continue
            at_g += w != q; lv[k] = L[w]; cv[k] = C.get(w, 0)
    ok = np.isfinite(lv) & (cv >= max(1, int(min_calls)))
    beta = pd.Series(np.where(ok, np.clip(lv, 0, 1), np.nan), index=K.probe.to_numpy()).dropna()
    calls = pd.Series(cv, index=K.probe.to_numpy())[beta.index]
    return beta, calls, dict(source=os.path.basename(call_bw) + " + " + os.path.basename(cov_bw), format="bigWig pair (GRCh38 per-CpG level and coverage)",
                             atlas_sites=int(len(K)), sites_with_value=int(ok.sum()), read_at_next_base=int(at_g),
                             calls_per_site_median=float(np.median(calls)) if len(calls) else 0.0, min_calls=int(min_calls))


GENOME_COVERAGE_RULE = dict(min_ratio=0.5)   # 2026-11-04: an autosome's share of sites with a value, against the median autosome's

def genome_coverage(values, key=None, rule=GENOME_COVERAGE_RULE):
    """Intake check for every sequenced sample (2026-11-04, author: 'the chain ... should make sure this cant happen again'). A whole-genome
    library carries reads on every autosome in proportion to its CpGs. A sample read only in part (the first N read pairs of a run stored in
    genome order, a truncated download, a lost chunk) carries reads on some chromosomes and not others: GSE98837's first 140 million pairs held
    chromosomes 1 to 8 only (DEV_METAA_ATLAS_CURVE_01.md). For each autosome, the share of IAM-Atlas v2.1's sites on it with a value, against
    the median share of the autosomes that carry any value; any autosome under rule['min_ratio'] of the median refuses the sample. A uniformly thin library (plasma at one
    read per site) passes, since the test compares chromosomes with each other, not with a depth. values: per-site levels indexed by probe
    (NaN where no value). Returns dict(status='OK' | 'REFUSED', by_chrom, short, reason)."""
    import os, pandas as pd
    key = key or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlas", "v2.1", "sites_sequencing_key.parquet")
    K = pd.read_parquet(key, columns=["probe", "chrom_hg19"]).drop_duplicates("probe").set_index("probe").chrom_hg19
    v = pd.Series(values).dropna(); v = v[v.index.isin(K.index)]   # sites with a value (callers may drop the rest)
    auto = [f"chr{i}" for i in range(1, 23)]
    share = (K.reindex(v.index).value_counts() / K.value_counts()).reindex(auto).fillna(0.0)   # over every key site on the chromosome
    med = float(share[share > 0].median()) if (share > 0).any() else 0.0   # run 1 (disclosed): the median over all 22 is 0 when most are empty
    ratio = (share / med) if med > 0 else share * 0
    short = [c for c in auto if ratio[c] < rule["min_ratio"]]
    out = dict(by_chrom={c: round(float(share[c]), 4) for c in auto}, median_share=round(med, 4), min_ratio=rule["min_ratio"], short=short)
    if med == 0: out.update(status="REFUSED", reason="no autosome carries a value")
    elif short: out.update(status="REFUSED", reason=f"reads cover the autosomes unevenly: {', '.join(short)} under {rule['min_ratio']} of the median autosome's "
                           f"share of sites with a value ({med:.3f}); the sample was read only in part (a run read from its first read pairs while "
                           f"stored in genome order, a truncated download or a lost chunk), so no reading is made")
    else: out.update(status="OK", reason=None)
    return out
