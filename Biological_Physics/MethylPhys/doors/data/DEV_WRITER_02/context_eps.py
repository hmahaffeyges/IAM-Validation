"""DEV-WRITER-02 cell side (box). Per sample and per NNCGNN context: the copy-error and de novo (control) channels of PROC-CHANNEL-01, with
channel.py's definitions unchanged, split by the hg19 flanking sequence of each CpG.
Copy error: molecules with >= 6 covered CpGs and >= 80 % C; at each covered call, opportunity += count; error += count when the call is T and
both covered neighbours are C (channel.py: copy_err = iso_T / nC_m). Control: molecules with >= 6 covered CpGs and <= 20 % C; error = C with both
covered neighbours T (channel.py: denovo = iso_C / nC_u). Context = hg19 bases -2..+3 around the C of the CpG (NNCGNN, top strand), from
wgbstools references/hg19 CpG.bed.gz (chr, position of the C, CpG index) and genome.fa.gz.
Check built in: summed over contexts, each sample's copy_err and denovo must equal PROC-CHANNEL-01's (rerun_2026-10-10_channel_samples.csv).
Inputs: the 153 cached window files of PROC-CHANNEL-01 (399 windows, windows_hg19_cpgidx.bed). Output: context_counts.csv.
Usage: python3 context_eps.py WINDOWS.bed CACHE_DIR WGBS_HG19_REF_DIR OUT.csv [NPROC] [HG19_FASTA]
(HG19_FASTA defaults to REF/genome.fa.gz; any indexed UCSC hg19 FASTA gives the same contexts, checked by the CG test below.)"""
import sys, os, gzip, collections, multiprocessing as mp, pandas as pd, pysam
WINB, CACHE, REF, OUT = sys.argv[1:5]; NP = int(sys.argv[5]) if len(sys.argv) > 5 else 16
FASTA = sys.argv[6] if len(sys.argv) > 6 else os.path.join(REF, "genome.fa.gz")
WIN = [l.split() for l in open(WINB)]
def ctx_map():
    want = set()
    for c, st, en in WIN: want.update(range(int(st), int(en)))
    fa = pysam.FastaFile(FASTA); M = {}; bad = 0
    for l in gzip.open(os.path.join(REF, "CpG.bed.gz"), "rt"):
        ch, pos, idx = l.split()[:3]; idx = int(idx)
        if idx not in want: continue
        p = int(pos) - 1                                    # 0-based position of the C
        s = fa.fetch(ch, p - 2, p + 4).upper()
        if s[2:4] != "CG" or len(s) != 6 or any(b not in "ACGT" for b in s): bad += 1; continue
        M[idx] = s
    assert bad < 0.001 * (len(M) + bad), f"{bad} of {len(M) + bad} window CpGs are not CG in this FASTA: wrong genome or coordinates"
    return M, bad
def one(args):
    g, M = args; E = collections.Counter(); O = collections.Counter(); EC = collections.Counter(); OC = collections.Counter()
    for line in open(os.path.join(CACHE, f"{g}.pat.txt"), "rb"):
        p = line.rstrip(b"\n").split(b"\t")
        if len(p) < 4: continue
        start = int(p[1]); pat = p[2].decode(); cnt = int(p[3])
        cov = [(start + i, ch) for i, ch in enumerate(pat) if ch != "."]
        if len(cov) < 6: continue
        s_ = "".join(ch for _, ch in cov); fC = s_.count("C") / len(s_)
        if fC >= 0.8: tgt, bg, Eo, Oo = "T", "C", E, O
        elif fC <= 0.2: tgt, bg, Eo, Oo = "C", "T", EC, OC
        else: continue
        for j, (idx, ch) in enumerate(cov):
            k = M.get(idx, "NA"); Oo[k] += cnt
            if 0 < j < len(cov) - 1 and ch == tgt and s_[j - 1] == bg and s_[j + 1] == bg: Eo[k] += cnt
    return [dict(sample=g, ctx=k, meth_err=E[k], meth_opp=O[k], ctrl_err=EC[k], ctrl_opp=OC[k]) for k in set(O) | set(OC)]
if __name__ == "__main__":
    M, bad = ctx_map(); print("window CpGs with a context", len(M), "| skipped (N or not CG)", bad, flush=True)
    gs = sorted(f[:-8] for f in os.listdir(CACHE) if f.endswith(".pat.txt"))
    with mp.Pool(NP) as P: R = [r for rs in P.imap_unordered(one, [(g, M) for g in gs]) for r in rs]
    pd.DataFrame(R).to_csv(OUT, index=False); print("samples", len(gs), "rows", len(R))
