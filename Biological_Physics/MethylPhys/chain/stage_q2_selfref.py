#!/usr/bin/env python3
"""Stage Q2 - IAM-A self-reference (COMMISSIONED 2026-10-17 on healthy blood CD8 T cells, whole-genome bisulfite; IAMA_COMMISSIONING). Written 2026-10-15 to move the own-read loose share, the held
error and the self-reference line out of side scripts (development/loose_share_2026-10-15/, doors/data/DEV_SECONDLAB_01/) into the chain.
Author's rule (2026-10-15): side scripts are for exploration only; every logged test runs through the chain, and the gate checks the stamp.

ONE COUNTER, two readers of the same molecules:
  count_pat(lines, tiles)          wgbstools .pat lines (hg19 CpG-index tiles): an exact port of own_reads_loose.one()
  count_bam(bam, ref, ...)         aligned reads (gemBS tag XB, or Bismark tag XM), GRCh38 or hg19, read pairs merged: port of secondlab.py
Both: molecules alternate EVEN -> loose share, ODD -> held error (no molecule used twice).
  loose share = tiles of 2,000 consecutive CpGs whose molecules' mean call is below CUT (0.70), over tiles with >= MIN_TILE_CALLS calls
  held error  = on molecules with >= 6 CpG calls and >= 80 % methylated, isolated unmethylated interior calls / interior calls
THE LINE lives only here, in its runtime matrix (Runtime Matrices/IAM_A_SelfRef/selfref_line_v*.json), on the RAW held-error axis this counter
writes. There is no scale factor anywhere: the 0.7412824486 factor of the development line (opensea_line.py) is gone, so a raw value can never
be compared with a scaled line again (the DEV-SECONDLAB-01 error). predict() and residual() are the only way to use it.
STAMP: every output this stage writes carries chain_stamp = {stage, chain_sha (sha256 of the chain's .py files), payload_sha256, inputs,
utc}. CANON/repro_check.py rule 23 recomputes the payload hash and checks chain_sha against CANON/chain_versions.json."""
import os, sys, json, glob, gzip, hashlib, datetime, collections, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
LINE_DIR = os.path.join(HERE, "Runtime Matrices", "IAM_A_SelfRef")
CUT, TILE, MIN_TILE_CALLS = 0.70, 2000, 200
QUAL_MIN_CALLS, QUAL_MIN_METH = 6, 0.8
QUAL_MAX_METH = 0.2   # 2026-10-24 (DEV-METAA-HEALTHY-01 layer 1): the gain side's mirror of QUAL_MIN_METH, a molecule held unmarked (stated)
STAGE = "Q2 self-reference"
HG19_CPGS = [("chr1",2284470),("chr2",2164335),("chr3",1623646),("chr4",1473930),("chr5",1506454),("chr6",1475569),("chr7",1568666),("chr8",1309135),
             ("chr9",1226821),("chr10",1351291),("chr11",1289987),("chr12",1277218),("chr13",803708),("chr14",859154),("chr15",825117),("chr16",1047108),
             ("chr17",1138298),("chr18",650016),("chr19",1009963),("chr20",693212),("chr21",340133),("chr22",554124)]   # PROC-CHANNEL-01 make_windows.py

# ---------------------------------------------------------------- stamp
def chain_sha():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))): h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()
def _canon(obj): return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float).encode()
def payload_sha(obj): return hashlib.sha256(_canon({k: v for k, v in obj.items() if k != "chain_stamp"})).hexdigest()
def file_sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""): h.update(b)
    return h.hexdigest()
# Manifest link (author 2026-10-16: "when the chain runs, it should have a certain amount of updates that happen automatically from the chain
# script"). Every result names the Master Testing Manifest rows it belongs to, in its stamp (manifest=[...] or env IAM_MANIFEST="ID,ID").
# The stamp's manifest field is outside the payload hash, so it never changes a result. CANON/chain_events.py reads these stamps at every
# checkpoint and writes each row's chain outputs and verdicts into CANON/manifest.json; CANON/manifest_fresh.py refuses a push with a stamped
# result (written from 2026-10-16) that names no row.
def _rows(manifest):
    m = manifest if manifest is not None else [x for x in os.environ.get("IAM_MANIFEST", "").split(",") if x.strip()]
    return sorted({str(x).strip() for x in m})
def stamp(obj, inputs=(), manifest=None):
    """Attach the chain stamp to a result dict. inputs: [{'name':..., 'sha256':..., 'size':...}] (file_sha of each input where possible).
    manifest: the Master Testing Manifest row IDs this result belongs to."""
    obj = {k: v for k, v in obj.items() if k != "chain_stamp"}
    obj = json.loads(json.dumps(obj, default=float))   # hash what will be stored: JSON turns number keys into text (2026-10-16, found on the GSE49627 count, run 2)
    obj["chain_stamp"] = dict(stage=STAGE, chain_sha=chain_sha(), payload_sha256=payload_sha(obj), inputs=list(inputs), manifest=_rows(manifest),
                              utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return obj
def write_json(obj, path, inputs=(), manifest=None):
    """Write a stamped result. A result with a 'verdict' field (MET, NOT MET, INCONCLUSIVE, NOT ASSESSED, REPRODUCED, DIFFERS) is recorded
    as a run on each named manifest row by CANON/chain_events.py."""
    o = stamp(obj, inputs, manifest); json.dump(o, open(path, "w", encoding="utf-8"), indent=1, default=float)
    assert payload_sha(json.load(open(path, encoding="utf-8"))) == o["chain_stamp"]["payload_sha256"], f"{path}: stored result does not verify"
    return o
def write_csv(df, path, inputs=(), manifest=None):
    df.to_csv(path, index=False); s = dict(stage=STAGE, chain_sha=chain_sha(), payload_sha256=file_sha(path), inputs=list(inputs), manifest=_rows(manifest),
                                            utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    json.dump({"chain_stamp": s}, open(path + ".stamp.json", "w"), indent=1); return s

# ---------------------------------------------------------------- tiles and the .pat reader (exact port of own_reads_loose.one)
TILE_SOURCE = "table"   # 2026-10-25 (plan 14.9, development): "dictionary" builds the tiles from the genome dictionary's own chromosome ranges
def hg19_tiles():
    """{chrom: array of [start, end) CpG-index tiles}, consecutive 2,000-CpG tiles, partial last tile dropped (own_reads_genome.tiles).
    TILE_SOURCE "table" (the commissioned line's tiles): chromosome counts from HG19_CPGS, exact for chr1-chr13 and short for chr14-chr22, so
    tiles from chr15 on sit at shifted addresses and many chr21-chr22 molecules fall in no tile (plan 14.9). "dictionary" (development): each
    chromosome's range from Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json (wgbstools init_genome hg19), the ranges intake checks."""
    T = collections.defaultdict(list)
    if TILE_SOURCE == "dictionary":
        import json as _j
        R = _j.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")))["ranges"]
        for c, _ in HG19_CPGS:
            a0, b0 = R[c]
            for s in range(a0, b0 - TILE + 2, TILE): T[c].append((s - 1, s - 1 + TILE))
        return {c: np.array(v) for c, v in T.items()}
    off = 1
    for c, n in HG19_CPGS:
        for s in range(off, off + n - TILE + 1, TILE): T[c].append((s - 1, s - 1 + TILE))
        off += n
    return {c: np.array(v) for c, v in T.items()}
def _finish(tile_calls, e_iso, e_opp, extra=None, g_iso=None, g_opp=None):
    m = np.array([a[0] / a[1] for a in tile_calls.values() if a[1] >= MIN_TILE_CALLS])
    out = dict(loose_share=float((m < CUT).mean()) if len(m) else None, tiles=int(len(m)), held_err=e_iso / max(e_opp, 1), held_opp=int(e_opp),
               cut=CUT, tile_cpgs=TILE, min_tile_calls=MIN_TILE_CALLS,
               # the tile distribution's shape, for the sample-shape check (COM-IAMA-053, added 2026-10-15): percentiles of the tile means
               tile_p10=float(np.quantile(m, 0.10)) if len(m) else None, tile_p50=float(np.quantile(m, 0.50)) if len(m) else None,
               tile_p90=float(np.quantile(m, 0.90)) if len(m) else None)
    if g_opp is not None:   # 2026-10-24: the gain side, the mirror of held_err: a lone methylated call between two unmethylated ones, on molecules held unmarked
        out.update(gain_err=g_iso / max(g_opp, 1), gain_opp=int(g_opp))
    if extra: out.update(extra)
    return out
def _mol(M):
    """2026-10-24 (DEVELOPMENT, plan 40.11): the molecule-level loose shares, recorded beside the tile loose share and used by no reading yet.
    mol_low_share: of the molecules with at least QUAL_MIN_CALLS calls on the tile half, the share whose methylated fraction is below CUT (the
    tile rule applied to each molecule); mol_mid_share: the share held neither marked nor unmarked (between QUAL_MAX_METH and QUAL_MIN_METH).
    Both add linearly when cell types are mixed, so a mix of cells with the same shares reads the same share; the tile loose share does not
    (synthetic test cfDNA 1)."""
    return dict(mol_n=int(M[0]), mol_low_share=(M[1] / M[0] if M[0] else None), mol_mid_share=(M[2] / M[0] if M[0] else None))
def count_pat(lines, tiles):
    tile_calls = {}; e_iso = e_opp = g_iso = g_opp = 0; M = [0, 0, 0]   # 2026-10-24 (plan 40.11): molecules on the tile half: qualifying, below CUT, undecided
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        c, i, pat, n = p[0], int(p[1]), p[2], int(p[3]); s = [x for x in pat if x != "."]
        if not s: continue
        if k % 2 == 0:
            w = tiles.get(c)
            if w is None: continue
            j = np.searchsorted(w[:, 0], i, side="right") - 1
            if j < 0 or i > w[j, 1]: continue
            a = tile_calls.setdefault((c, j), [0, 0]); a[0] += n * s.count("C"); a[1] += n * len(s)
            if len(s) >= QUAL_MIN_CALLS:   # the loose share counted molecule by molecule, which adds linearly when cell types mix (the tile share does not)
                f = s.count("C") / len(s); M[0] += n; M[1] += n * (f < CUT); M[2] += n * (QUAL_MAX_METH < f < QUAL_MIN_METH)
        elif len(s) >= QUAL_MIN_CALLS and s.count("C") >= QUAL_MIN_METH * len(s):
            e_opp += n * (len(s) - 2); e_iso += n * sum(1 for q in range(1, len(s) - 1) if s[q] == "T" and s[q - 1] == "C" and s[q + 1] == "C")
        elif len(s) >= QUAL_MIN_CALLS and s.count("C") <= QUAL_MAX_METH * len(s):
            g_opp += n * (len(s) - 2); g_iso += n * sum(1 for q in range(1, len(s) - 1) if s[q] == "C" and s[q - 1] == "T" and s[q + 1] == "T")
    return _finish(tile_calls, e_iso, e_opp, dict(reader="pat", assembly="hg19", **_mol(M)), g_iso, g_opp)

def count_pat_tile_errors(lines, tile=TILE):
    """2026-10-25 (DEV-ERASE-01, the graded maintenance test; plan 14.1): the held error tile by tile, in two independent halves. Held
    molecules and lone errors by count_pat's own rules (at least QUAL_MIN_CALLS calls, at least QUAL_MIN_METH methylated; a lone unmethylated
    call between two methylated ones, over the len - 2 interior opportunities). A molecule belongs to the tile of its first CpG, tiles being
    blocks of `tile` consecutive CpG indices on each chromosome (index // tile), so the tiling needs no genome table and works on any genome.
    Half A takes the even lines, half B the odd lines, so the two halves share no molecule. Returns {(chrom, tile): [iso_A, opp_A, iso_B, opp_B]}."""
    T = {}
    for k, line in enumerate(lines):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        c, i, pat, n = p[0], int(p[1]), p[2], int(p[3]); s_ = [x for x in pat if x != "."]
        if len(s_) < QUAL_MIN_CALLS or s_.count("C") < QUAL_MIN_METH * len(s_): continue
        a = T.setdefault((c, i // tile), [0, 0, 0, 0]); h = 2 * (k % 2)
        a[h + 1] += n * (len(s_) - 2); a[h] += n * sum(1 for q in range(1, len(s_) - 1) if s_[q] == "T" and s_[q - 1] == "C" and s_[q + 1] == "C")
    return T

def count_pat_sites(lines, sites):
    """2026-10-24 (DEV-METAA-HEALTHY-01 step 3 version 2): the held error counted only at a set of sites (genome-wide CpG indices, the .pat
    numbering), by count_pat's own rules: the same split (odd lines), the same held and held-absent molecules, classed from ALL their calls
    (at least QUAL_MIN_CALLS; marked share at least QUAL_MIN_METH, or at most QUAL_MAX_METH), the same isolated error; only the calls
    counted, errors and opportunities, are those at the given sites. Returns the loss and gain sides with their opportunities."""
    S = set(int(x) for x in sites); e_iso = e_opp = g_iso = g_opp = 0
    for k, line in enumerate(lines):
        if k % 2 == 0: continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        i, pat, n = int(p[1]), p[2], int(p[3])
        pos = [i + q for q, x in enumerate(pat) if x != "."]; s_ = [x for x in pat if x != "."]
        if len(s_) < QUAL_MIN_CALLS: continue
        f = s_.count("C") / len(s_)
        for q in range(1, len(s_) - 1):
            if pos[q] not in S: continue
            if f >= QUAL_MIN_METH: e_opp += n; e_iso += n * (s_[q] == "T" and s_[q - 1] == "C" and s_[q + 1] == "C")
            elif f <= QUAL_MAX_METH: g_opp += n; g_iso += n * (s_[q] == "C" and s_[q - 1] == "T" and s_[q + 1] == "T")
    return dict(held_iso=int(e_iso), held_opp=int(e_opp), held_err=(e_iso / e_opp if e_opp else None),
                gain_iso=int(g_iso), gain_opp=int(g_opp), gain_err=(g_iso / g_opp if g_opp else None), n_sites=len(S))

def count_pat_site_classes(lines, groups):
    """2026-10-24 (DEV-METAA-HEALTHY-01 step 3 version 3): every call at given sites, by group (groups: site index -> label, e.g. the cell's
    held and held-absent identity sites), split by the molecule's class from ALL its calls, Stage Q2's own rules: held (marked share at least
    QUAL_MIN_METH), held absent (at most QUAL_MAX_METH), loose (in between), or short (fewer than QUAL_MIN_CALLS calls; counted in the level,
    not classed). Both halves of the file are read (the level uses every molecule). Returns, per group: calls and marked calls per class, the
    site level (marked calls over all calls), each class's share of the classed calls and its own level."""
    G = {int(k): v for k, v in groups.items()}; T = {}
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4: continue
        i, pat, n = int(p[1]), p[2], int(p[3]); pos = [i + q for q, x in enumerate(pat) if x != "."]; s_ = [x for x in pat if x != "."]
        hit = [(G[pp], x) for pp, x in zip(pos, s_) if pp in G]
        if not hit: continue
        f = s_.count("C") / len(s_)
        k = "short" if len(s_) < QUAL_MIN_CALLS else ("held" if f >= QUAL_MIN_METH else ("absent" if f <= QUAL_MAX_METH else "loose"))
        for g, x in hit:
            a = T.setdefault(g, {c: [0, 0] for c in ("held", "absent", "loose", "short")}); a[k][0] += n; a[k][1] += n * (x == "C")
    out = {}
    for g, a in T.items():
        tot = sum(v[0] for v in a.values()); cl = sum(a[c][0] for c in ("held", "absent", "loose"))
        out[g] = dict(calls=int(tot), level=(sum(v[1] for v in a.values()) / tot if tot else None),
                      **{f"{c}_calls": int(a[c][0]) for c in a}, **{f"{c}_share": (a[c][0] / cl if cl else None) for c in ("held", "absent", "loose")},
                      **{f"{c}_level": (a[c][1] / a[c][0] if a[c][0] else None) for c in a})
    return out

# ---------------------------------------------------------------- the aligned-read reader (port of secondlab.py, plus Bismark XM)
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
def _cpg_index(seq):
    b = np.frombuffer(seq.upper().encode(), dtype=np.uint8); return np.flatnonzero((b[:-1] == ord("C")) & (b[1:] == ord("G")))
def _calls_of(r, ref, ref0, sub):
    q = r.query_sequence; out = {}; nc = [0, 0]; at = [0, 0]
    if q is None: return out, nc, at
    xm = r.get_tag("XM") if r.has_tag("XM") else None; xb = r.get_tag("XB") if r.has_tag("XB") else None
    if xm is None and xb not in ("C", "G"): return out, nc, at
    top = (r.get_tag("XG") == "CT") if (xm is not None and r.has_tag("XG")) else (xb == "C")
    pairs = r.get_aligned_pairs(matches_only=True); L = len(pairs)
    for j, (qi, ri) in enumerate(pairs):
        if j < 3 or j >= L - 3: continue
        k = ri - ref0
        if k < 1 or k + 1 >= len(ref): continue
        rb, qb = ref[k], q[qi]
        if sub and rb in "AT": at[1] += 1; at[0] += qb != rb
        if xm is not None:
            z = xm[qi]
            if z in "Zz": out[ri if top else ri - 1] = z == "Z"
            elif z in "XxHh": nc[1] += 1; nc[0] += z in "XH"
            continue
        if top and rb == "C":
            if ref[k + 1] == "G":
                if qb in "CT": out[ri] = qb == "C"
            elif qb in "CT": nc[1] += 1; nc[0] += qb == "C"
        elif (not top) and rb == "G":
            if ref[k - 1] == "C":
                if qb in "GA": out[ri - 1] = qb == "G"
            elif qb in "GA": nc[1] += 1; nc[0] += qb == "G"
    return out, nc, at
def _merge(a, b):
    m = dict(a)
    for p, v in b.items():
        if p in m and m[p] != v: m[p] = None
        elif p not in m: m[p] = v
    return [(p, v) for p, v in sorted(m.items()) if v is not None]
def count_bam_region(args):
    """One region of one alignment file -> partial counts. args = (bam, ref_fasta, chrom, start, end)."""
    bam, fa, chrom, s0, e0 = args
    import pysam
    B = pysam.AlignmentFile(bam, "rb"); F = pysam.FastaFile(fa)
    cp = _cpg_index(F.fetch(chrom)); ntile = len(cp) // TILE
    T = collections.defaultdict(lambda: [0, 0]); st = collections.Counter(); buf = {}; k = [0]; nh = collections.Counter()
    def done(calls):
        if not calls: return
        st["molecules"] += 1; k[0] += 1
        if k[0] % 2 == 0:
            rk = int(np.searchsorted(cp, calls[0][0]))
            if rk < len(cp) and cp[rk] == calls[0][0] and rk // TILE < ntile:
                a = T[rk // TILE]; a[0] += sum(v for _, v in calls); a[1] += len(calls)
        else:
            n = len(calls); mm = sum(v for _, v in calls)
            if n >= QUAL_MIN_CALLS and mm >= QUAL_MIN_METH * n:
                nh[min(n, 200)] += 1   # calls per QUALIFYING molecule (stage TL's response curve). Until 2026-10-18 every molecule was counted here,
                                       # including unmethylated ones that never qualify, which biased the curve (LESSONS C23, DEV-CFDNA-SHORT-01)
                st["qualifying"] += 1; st["e_opp"] += n - 2
                st["e_iso"] += sum(1 for i in range(1, n - 1) if (not calls[i][1]) and calls[i - 1][1] and calls[i + 1][1])
            elif n >= QUAL_MIN_CALLS and mm <= QUAL_MAX_METH * n:   # 2026-10-24: the gain side
                st["g_opp"] += n - 2; st["g_iso"] += sum(1 for i in range(1, n - 1) if calls[i][1] and (not calls[i - 1][1]) and (not calls[i + 1][1]))
    for r in B.fetch(chrom, s0, e0):
        if r.reference_start < s0 or r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate or r.mapping_quality < 10: continue
        st["reads"] += 1
        ref0 = r.reference_start - 1; ref = F.fetch(chrom, max(ref0, 0), r.reference_end + 1).upper()
        if ref0 < 0: ref = "N" + ref
        c, n1, a1 = _calls_of(r, ref, ref0, st["reads"] % 20 == 0)
        st["nc_m"] += n1[0]; st["nc_t"] += n1[1]; st["at_mm"] += a1[0]; st["at_n"] += a1[1]
        if r.is_paired and not r.mate_is_unmapped and r.next_reference_name == chrom and r.next_reference_start >= s0:
            if r.query_name in buf: done(_merge(buf.pop(r.query_name), c)); continue
            if r.next_reference_start >= r.reference_start and r.next_reference_start < e0: buf[r.query_name] = c; continue
        done(sorted(c.items()))
    for c in buf.values(): done(sorted(c.items()))
    return dict(chrom=chrom, tiles={int(t): v for t, v in T.items()}, n_hist={int(a): int(b) for a, b in nh.items()}, **{x: int(st[x]) for x in ("e_iso", "e_opp", "g_iso", "g_opp", "molecules", "qualifying", "reads", "nc_m", "nc_t", "at_mm", "at_n")})
def combine_bam(parts, assembly):
    T = {}; s = collections.Counter()
    for p in parts:
        for t, v in p["tiles"].items():
            a = T.setdefault((p["chrom"], int(t)), [0, 0]); a[0] += v[0]; a[1] += v[1]
        for x in ("e_iso", "e_opp", "g_iso", "g_opp", "molecules", "qualifying", "reads", "nc_m", "nc_t", "at_mm", "at_n"): s[x] += p[x]
    return _finish(T, s["e_iso"], s["e_opp"], g_iso=s["g_iso"], g_opp=s["g_opp"], extra=dict(reader="aligned reads", assembly=assembly, sub_err=s["at_mm"] / max(s["at_n"], 1),
                   conv_fail=s["nc_m"] / max(s["nc_t"], 1), molecules=int(s["molecules"]), qualifying=int(s["qualifying"]), reads=int(s["reads"])))
def count_bam(bam, fa, workers, assembly, chroms=AUTOSOMES, span=10_000_000):
    import pysam, multiprocessing as mp
    F = pysam.FastaFile(fa); jobs = [(bam, fa, c, s, min(F.get_reference_length(c), s + span)) for c in chroms for s in range(0, F.get_reference_length(c), span)]
    with mp.get_context("fork").Pool(workers) as P: parts = P.map(count_bam_region, jobs, chunksize=1)
    return combine_bam(parts, assembly)

# ---------------------------------------------------------------- the line
def build_line(per_sample):
    """per_sample: DataFrame with cell, loose_share, held_err, held_opp (RAW counter output). Per cell: median loose share, opportunity-weighted
    held error; Theil-Sen line across cells (as opensea_line.py, without its scale factor)."""
    from scipy.stats import theilslopes
    g = per_sample.dropna(subset=["loose_share", "held_err"]).groupby("cell")
    eps = pd.Series({c: (x.held_err * x.held_opp).sum() / x.held_opp.sum() for c, x in g}).rename_axis("cell")   # same arithmetic as before, on any pandas (requirements pin 1.5.3; the clean room found include_groups needs 2.2, 2026-10-16)
    C = pd.DataFrame({"L": g["loose_share"].median(), "eps": eps, "n": g.size()})
    s, i, _, _ = theilslopes(C.eps.values, C.L.values); res = C.eps.values - (i + s * C.L.values)
    return dict(axis="raw counter held error (no scale factor)", firm=float(i), slope=float(s), cells=int(len(C)), samples=int(C.n.sum()),
                residual_abs_median=float(np.median(abs(res))), within_0_003=int((abs(res) <= 0.003).sum()), cut=CUT, tile_cpgs=TILE), C.reset_index()
def line(version=None):
    fs = sorted(glob.glob(os.path.join(LINE_DIR, "selfref_line_v*.json")))
    if not fs: raise RuntimeError("no self-reference line in the chain yet (Runtime Matrices/IAM_A_SelfRef): build it with build_line() first")
    f = [x for x in fs if x.endswith(f"_v{version}.json")][0] if version else fs[-1]; J = json.load(open(f))
    if payload_sha(J) != J["chain_stamp"]["payload_sha256"]: raise RuntimeError(f"{os.path.basename(f)} has been edited after the chain wrote it")
    return J
def predict(loose_share, version=None):
    J = line(version); return J["firm"] + J["slope"] * float(loose_share)
def residual(rec, version=None):
    """rec: a count from count_pat / count_bam (RAW). Returns the self-reference reading against the chain's line."""
    J = line(version); p = J["firm"] + J["slope"] * rec["loose_share"]
    return dict(loose_share=rec["loose_share"], held_err=rec["held_err"], predicted=p, residual=rec["held_err"] - p,
                line_version=J["version"], axis=J["axis"])

# ---- sample-shape check (COM-IAMA-053; author: "Seal as drafted", 2026-10-15) --------------------------------------------------------------
# A sample is read only if its tile 10th percentile is no more than MARGIN below the lowest healthy sample of its own cell type. MARGIN is the
# 90th percentile, over cell types with at least 2 healthy samples, of the within-type spread (max - min) of the tile 10th percentile; it is
# computed from the reference, never typed. A cell type with no healthy reference is not read.
SHAPE_Q = 0.90
def build_shape_reference(records):
    """records: chain counts (count_pat/count_bam output with cell and tile_p10). Returns the reference dict (write it with write_json)."""
    G = {}
    for r in records:
        if r.get("tile_p10") is not None: G.setdefault(r["cell"], []).append(float(r["tile_p10"]))
    spread = [max(v) - min(v) for v in G.values() if len(v) >= 2]
    return dict(kind="IAM-A sample-shape reference", margin=float(np.quantile(spread, SHAPE_Q)), margin_basis=f"{SHAPE_Q:.0%} quantile of within-type spread of tile_p10 over {len(spread)} cell types",
                cells={c: dict(min_p10=min(v), n=len(v)) for c, v in sorted(G.items())}, samples=int(sum(len(v) for v in G.values())))
def shape_reference(version=None):
    fs = sorted(glob.glob(os.path.join(LINE_DIR, "shape_reference_v*.json")))
    if not fs: raise RuntimeError("no shape reference in the chain yet: build it with build_shape_reference()")
    f = [x for x in fs if x.endswith(f"_v{version}.json")][0] if version else fs[-1]; J = json.load(open(f))
    if payload_sha(J) != J["chain_stamp"]["payload_sha256"]: raise RuntimeError(f"{os.path.basename(f)} has been edited after the chain wrote it")
    return J
def shape_check(rec, cell, version=None, ref=None):
    """Returns dict(read=bool, reason, ...). ref: a reference dict (for tests); else the chain's stored reference."""
    J = ref if ref is not None else shape_reference(version); c = J["cells"].get(cell)
    if c is None: return dict(read=False, reason=f"not read: no healthy reference of this type ({cell})", cell=cell)
    lo = c["min_p10"] - J["margin"]; p = rec.get("tile_p10")
    if p is None: return dict(read=False, reason="not read: no tile shape in this count", cell=cell)
    ok = p >= lo
    return dict(read=bool(ok), cell=cell, tile_p10=p, healthy_min_p10=c["min_p10"], margin=J["margin"], limit=lo, healthy_n=c["n"],
                reason="read" if ok else f"not read: tile 10th percentile {p:.3f} is below its healthy type's limit {lo:.3f} (sample shape outside healthy)")

# ---------------------------------------------------------------- the reading and its healthy band (COM-IAMA-054, sealed by the author 2026-10-16)
def reading(rec, version=None):
    """IAM-A self-reference reading: A = measured held error / what the sample's own loose share predicts on the line. 1 = on the healthy
    line; above 1 = more holes than its own line predicts (right on the gauge)."""
    q = residual(rec, version); return dict(A=rec["held_err"] / q["predicted"], **q)
def build_band(healthy, version=None):
    """healthy: counts of healthy samples from every laboratory, each already through the shape check (a refused sample is not passed in).
    Band = 2.5th to 97.5th percentile of their readings, pooled (sealed 2026-10-16: 'Seal as drafted'). Returns the band with its counts."""
    a = np.array([reading(o, version)["A"] for o in healthy], dtype=float); labs = collections.Counter(o.get("lab", "unknown") for o in healthy)
    return dict(lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)), median=float(np.median(a)), n=int(len(a)), labs=dict(labs),
                rule="2.5th to 97.5th percentile of pooled healthy readings, A = held error / line prediction (COM-IAMA-054)", line_version=version or line()["version"])

# ---------------------------------------------------------------- the grand sealed test (COM-IAMA-052), drafted 2026-10-16
GRAND_SHARE = 0.75; GRAND_MIN_READ = 8
CD8_SHAPE_TYPE = {"naive": "naive cd8 t cells", "tem": "effector memory cd8 t cells", "tcm": "cd8 t cells", "tscm": "cd8 t cells"}
def band(version=1):
    J = json.load(open(os.path.join(LINE_DIR, f"band_v{version}.json"), encoding="utf-8")); assert payload_sha(J) == J["chain_stamp"]["payload_sha256"]; return J
IAMA_COMMISSIONING = {"status": "COMMISSIONED", "date": "2026-10-17", "record": "Biological_Physics/MethylPhys/doors/COMMISSIONING_NOTE_IAMA.md",
    "specimens": ["healthy blood CD8 T cells"], "chemistry": ["whole-genome bisulfite"], "band": [0.7787, 1.0551]}
def grand_verdict(records, version=1, B=None, ref=None):
    """records: [(count, shape type)] for each healthy sample. Each is shape-checked first (refused = listed, not counted); each read sample's
    reading A must lie inside IAM-A's band v1. MET if at least GRAND_SHARE of the read samples are inside and at least GRAND_MIN_READ were read;
    NOT ASSESSED if fewer were read."""
    B = B or band(version); rows = []
    for rec, typ in records:
        s = shape_check(rec, typ, ref=ref); r = dict(sample=rec.get("sample") or rec.get("run"), shape_type=typ, shape=s["reason"], read=s["read"])
        if s["read"]: A = reading(rec)["A"]; r.update(A=A, inside=bool(B["lo"] <= A <= B["hi"]))
        rows.append(r)
    n = sum(r["read"] for r in rows); k = sum(r.get("inside", False) for r in rows)
    v = "NOT ASSESSED" if n < GRAND_MIN_READ else ("MET" if k >= GRAND_SHARE * n else "NOT MET")
    return dict(verdict=v, read=n, inside=k, of=len(rows), band=dict(lo=B["lo"], hi=B["hi"]), rows=rows)
