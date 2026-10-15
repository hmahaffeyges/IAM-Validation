"""DEV-SECONDLAB-01 (PHY-047 second laboratory, PHY-050 meter error), written 2026-10-15 before any ENCODE read is counted.
ENCODE (Bernstein lab, gemBS pipeline, GRCh38) sorted healthy blood cells: does IAM-A's self-reference line, set on Loyfer's 56 cell types
with the own-read counter (development/loose_share_2026-10-15/opensea_line.json, own_reads_every_CpG: firm 0.0298, loose 0.0555), hold in a
different laboratory, sequencer and pipeline? Same counter, re-implemented for gemBS alignments (no XM tag; the CpG call comes from the read
base against the reference on the read's conversion strand, tag XB: C = top strand, read C methylated / T not, at the CpG's C; G = bottom
strand, read G / A at the CpG's G):
 molecule   = a read pair (mates merged; overlapping calls that disagree are dropped); MAPQ >= 10; flagged duplicates, secondary and
              supplementary skipped; the first and last 3 aligned bases of each read ignored (as the tumor extractor)
 parity     = molecules alternate in fetch order: EVEN -> loose share, ODD -> held error, so no molecule is used twice (own_reads_loose.py)
 loose share= GRCh38 autosomes tiled into consecutive 2,000-CpG tiles (partial last tile dropped, own_reads_genome.py); a molecule goes to
              the tile of its first CpG; a tile is loose if its molecules' mean call is below 0.70; share over tiles with >= 200 calls
 held error = molecules with >= 6 CpG calls, >= 80 % methylated; every interior call is an opportunity; an unmethylated interior call with
              both neighbors methylated is an isolated error (PROC-CHANNEL-01)
 meter      = sequencing error: A/T-reference mismatches over A/T positions on 1 read in 20; conversion failure: non-CpG reference C (top)
              or G (bottom) read unconverted (as the tumor extractor)
Usage: python secondlab.py synthetic | run <bam> <ref.fa> <workers> <out.json> | score <dir with *.json>"""
import os, sys, json, re, collections, multiprocessing as mp, numpy as np
FIRM, LOOSE, CUT, TILE, MINC = 0.0298, 0.0555, 0.70, 2000, 200
AUTO = [f"chr{i}" for i in range(1, 23)]
def cpg_index(seq):
    b = np.frombuffer(seq.upper().encode(), dtype=np.uint8); return np.flatnonzero((b[:-1] == ord("C")) & (b[1:] == ord("G")))
def calls_of(r, ref, ref0, sub):
    """CpG calls {cpg C position: methylated} for one read, plus meter counts."""
    xb = r.get_tag("XB") if r.has_tag("XB") else None; q = r.query_sequence; out = {}; nc = [0, 0]; at = [0, 0]
    if xb not in ("C", "G") or q is None: return out, nc, at
    pairs = r.get_aligned_pairs(matches_only=True); L = len(pairs)
    for j, (qi, ri) in enumerate(pairs):
        if j < 3 or j >= L - 3: continue
        k = ri - ref0
        if k < 1 or k + 1 >= len(ref): continue
        rb, qb = ref[k], q[qi]
        if sub and rb in "AT": at[1] += 1; at[0] += qb != rb
        if xb == "C" and rb == "C":
            if ref[k + 1] == "G":
                if qb in "CT": out[ri] = qb == "C"
            elif qb in "CT": nc[1] += 1; nc[0] += qb == "C"
        elif xb == "G" and rb == "G":
            if ref[k - 1] == "C":
                if qb in "GA": out[ri - 1] = qb == "G"
            elif qb in "GA": nc[1] += 1; nc[0] += qb == "G"
    return out, nc, at
def merge(a, b):
    m = dict(a)
    for p, v in b.items():
        if p in m and m[p] != v: m[p] = None
        elif p not in m: m[p] = v
    return [(p, v) for p, v in sorted(m.items()) if v is not None]
def region(args):
    bam, fa, chrom, s0, e0 = args
    import pysam
    B = pysam.AlignmentFile(bam, "rb"); F = pysam.FastaFile(fa)
    cp = cpg_index(F.fetch(chrom)); ntile = len(cp) // TILE
    T = collections.defaultdict(lambda: [0, 0]); e_iso = e_opp = 0; nc = [0, 0]; at = [0, 0]; nmol = nq = 0; buf = {}; k = 0
    def done(calls):
        nonlocal e_iso, e_opp, nmol, nq, k
        if not calls: return
        nmol += 1; k += 1
        if k % 2 == 0:
            rk = int(np.searchsorted(cp, calls[0][0]))
            if rk < len(cp) and cp[rk] == calls[0][0] and rk // TILE < ntile:
                a = T[rk // TILE]; a[0] += sum(v for _, v in calls); a[1] += len(calls)
        else:
            n = len(calls); mm = sum(v for _, v in calls)
            if n >= 6 and mm >= 0.8 * n:
                nq += 1; e_opp += n - 2; e_iso += sum(1 for i in range(1, n - 1) if (not calls[i][1]) and calls[i - 1][1] and calls[i + 1][1])
    nread = 0
    for r in B.fetch(chrom, s0, e0):
        if r.reference_start < s0 or r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate or r.mapping_quality < 10: continue
        nread += 1
        ref0 = r.reference_start - 1; ref = F.fetch(chrom, max(ref0, 0), r.reference_end + 1).upper()
        if ref0 < 0: ref = "N" + ref
        c, n1, a1 = calls_of(r, ref, ref0, nread % 20 == 0); nc[0] += n1[0]; nc[1] += n1[1]; at[0] += a1[0]; at[1] += a1[1]
        if r.is_paired and not r.mate_is_unmapped and r.next_reference_name == chrom and r.next_reference_start >= s0:
            if r.query_name in buf: done(merge(buf.pop(r.query_name), c)); continue
            if r.next_reference_start >= r.reference_start and r.next_reference_start < e0: buf[r.query_name] = c; continue
        done(sorted(c.items()))
    for c in buf.values(): done(sorted(c.items()))
    return dict(chrom=chrom, tiles={int(t): v for t, v in T.items()}, e_iso=e_iso, e_opp=e_opp, nc=nc, at=at, molecules=nmol, qualifying=nq, reads=nread)
def summarize(parts):
    T = {}; s = collections.Counter()
    for p in parts:
        for t, v in p["tiles"].items():
            a = T.setdefault((p["chrom"], int(t)), [0, 0]); a[0] += v[0]; a[1] += v[1]
        for x in ("e_iso", "e_opp", "molecules", "qualifying", "reads"): s[x] += p[x]
        s["nc_m"] += p["nc"][0]; s["nc_t"] += p["nc"][1]; s["at_mm"] += p["at"][0]; s["at_n"] += p["at"][1]
    m = np.array([a / b for a, b in T.values() if b >= MINC])
    return dict(loose_share=float((m < CUT).mean()) if len(m) else None, tiles=int(len(m)), held_err=s["e_iso"] / max(s["e_opp"], 1), held_opp=int(s["e_opp"]),
                sub_err=s["at_mm"] / max(s["at_n"], 1), conv_fail=s["nc_m"] / max(s["nc_t"], 1), molecules=int(s["molecules"]), qualifying=int(s["qualifying"]), reads=int(s["reads"]))
def predicted(L): return FIRM + (LOOSE - FIRM) * L
def score(rows):
    """Bars (development, showcase; fixed 2026-10-15 before data): S1 at least 3 of the 5 cells' raw held error within 0.003 of the Loyfer
    own-read line at their own loose share (Loyfer's own cells: 41 of 56 within 0.003). Reported, not scored: residuals with the meter's
    sequencing error subtracted; each ENCODE cell beside the same Loyfer cell type."""
    R = []
    for r in rows:
        p = predicted(r["loose_share"]); R.append(dict(r, predicted=round(p, 5), residual=round(r["held_err"] - p, 5), within=bool(abs(r["held_err"] - p) <= 0.003),
                                                        held_err_corrected=round(r["held_err"] - r["sub_err"], 5)))
    k = sum(x["within"] for x in R); return R, dict(S1_within=k, of=len(R), verdict="MET" if (len(R) == 5 and k >= 3) else ("NOT MET" if len(R) == 5 else "INCOMPLETE"))
if __name__ == "__main__" and sys.argv[1] == "synthetic":
    import pysam, tempfile
    rng = np.random.default_rng(11); H = os.path.dirname(os.path.abspath(__file__)); W = tempfile.mkdtemp(); L = []
    seq = []
    for i in range(4000):                       # 4,000 blocks of 25 bases, each holding one CpG: 4,000 CpGs = 2 tiles of 2,000
        b = list(rng.choice(list("AT"), 25)); b[12], b[13] = "C", "G"; seq.append("".join(b))
    seq = "".join(seq); open(f"{W}/ref.fa", "w").write(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n"); pysam.faidx(f"{W}/ref.fa")
    cp = cpg_index(seq); assert len(cp) == 4000
    def run_case(beta_tile, eps, suberr, label, want_L, want_eps):
        hdr = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(seq)}]}; reads = []; TR = [0, 0]
        for n in range(6000):
            st = 25 * int(rng.integers(0, (len(seq) - 300) // 25)) + 13; tile = int(np.searchsorted(cp, st)) // TILE; tile = min(tile, 1)
            xb = "C" if n % 2 == 0 else "G"; meth = {}
            for p in cp[(cp >= st + 5) & (cp < st + 245)]: meth[int(p)] = rng.random() < beta_tile[tile]
            if eps is not None:                 # planted held error in mostly-methylated molecules
                meth = {p: (rng.random() >= eps) for p in meth}
            cl = [v for _, v in sorted(meth.items())]; n_ = len(cl)
            if n_ >= 6 and sum(cl) >= 0.8 * n_: TR[1] += n_ - 2; TR[0] += sum(1 for i in range(1, n_ - 1) if (not cl[i]) and cl[i - 1] and cl[i + 1])
            for mate, (a, ln) in enumerate(((st, 150), (st + 100, 150))):   # every CpG a read can see is planted
                s = list(seq[a:a + ln])
                for p, v in meth.items():
                    if xb == "C" and a <= p < a + ln: s[p - a] = "C" if v else "T"
                    if xb == "G" and a <= p + 1 < a + ln: s[p + 1 - a] = "G" if v else "A"
                for j in range(ln):
                    if xb == "C" and s[j] == "C" and not (a + j in meth): s[j] = "T"
                    if xb == "G" and s[j] == "G" and not (a + j - 1 in meth): s[j] = "A"
                    if s[j] in "AT" and seq[a + j] in "AT" and rng.random() < suberr: s[j] = "C" if s[j] == "A" else "G"
                r = pysam.AlignedSegment(); r.query_name = f"m{n}"; r.query_sequence = "".join(s); r.reference_id = 0; r.reference_start = a
                r.cigarstring = f"{ln}M"; r.mapping_quality = 40; r.flag = 1 | 2 | (64 if mate == 0 else 128) | (16 if mate else 32)
                r.next_reference_id = 0; r.next_reference_start = st + 100 if mate == 0 else st; r.query_qualities = pysam.qualitystring_to_array("I" * ln)
                r.set_tag("XB", xb); reads.append(r)
        reads.sort(key=lambda r: r.reference_start)
        with pysam.AlignmentFile(f"{W}/t.bam", "wb", header=hdr) as o:
            for r in reads: o.write(r)
        pysam.index(f"{W}/t.bam")
        S = summarize([region((f"{W}/t.bam", f"{W}/ref.fa", "chr1", 0, len(seq)))])
        if want_eps is not None: want_eps = TR[0] / TR[1]; se = (want_eps / max(S["held_opp"], 1)) ** 0.5   # truth: the counter's rule on the planted molecules (all of them)
        ok = (want_L is None or abs(S["loose_share"] - want_L) < 1e-9) and (want_eps is None or abs(S["held_err"] - want_eps) < 3 * se) and abs(S["sub_err"] - suberr) < 0.5 * suberr + 2e-4
        L.append(f"{label}: loose share {S['loose_share']} (want {want_L}), held {S['held_err']:.5f} (want {want_eps if want_eps is None else round(want_eps, 5)}), sub_err {S['sub_err']:.5f} (planted {suberr}), conv {S['conv_fail']:.4f} -> {'ok' if ok else 'WRONG'}")
        return ok
    ok = [run_case((0.40, 0.92), None, 0.002, "tile 1 loose, tile 2 firm, both strands", 0.5, None),
          run_case((0.92, 0.92), None, 0.002, "both firm", 0.0, None),
          run_case((1.0, 1.0), 0.03, 0.004, "planted held error 0.03 (truth: the rule on the planted molecules)", None, 0.0)]
    R, V = score([dict(cell="x", loose_share=0.2, held_err=predicted(0.2) + 0.001, sub_err=0.002)] * 3 + [dict(cell="y", loose_share=0.3, held_err=predicted(0.3) + 0.01, sub_err=0.002)] * 2)
    ok.append(V["verdict"] == "MET"); L.append(f"scorer, 3 of 5 on the line: {V['verdict']} (want MET)")
    R, V = score([dict(cell="y", loose_share=0.3, held_err=predicted(0.3) + 0.01, sub_err=0.002)] * 5); ok.append(V["verdict"] == "NOT MET"); L.append(f"scorer, 0 of 5 on the line: {V['verdict']} (want NOT MET)")
    L.append(f"SYNTHETIC: {'MET' if all(ok) else 'NOT MET'}"); open(os.path.join(H, "secondlab_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
elif __name__ == "__main__" and sys.argv[1] == "run":
    bam, fa, nw, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    import pysam
    F = pysam.FastaFile(fa); jobs = []
    for c in AUTO:
        n = F.get_reference_length(c)
        for s in range(0, n, 10_000_000): jobs.append((bam, fa, c, s, min(n, s + 10_000_000)))
    with mp.get_context("fork").Pool(nw) as P: parts = P.map(region, jobs, chunksize=1)
    S = summarize(parts); json.dump(S, open(out, "w"), indent=1); print(json.dumps(S), flush=True)
elif __name__ == "__main__" and sys.argv[1] == "score":
    import glob
    rows = [dict(cell=os.path.basename(f)[:-5], **json.load(open(f))) for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.json")))]
    R, V = score(rows); H = os.path.dirname(os.path.abspath(__file__))
    json.dump(dict(rows=R, verdict=V), open(os.path.join(H, "secondlab_scored.json"), "w"), indent=1); print(json.dumps(V)); [print(r) for r in R]
