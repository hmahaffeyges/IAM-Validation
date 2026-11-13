"""Copier-lag check (DEV-METHIONINE-01, exploration; prediction in the record before reading). Builds each hg19 CpG's constitutive replication
timing (mean of the 15 ENCODE UW Repli-seq wavelet signals at the CpG), numbers CpGs as wgbstools does, checks that numbering against the
chain's per-chromosome index ranges (IAM_A_Positions/hg19_cpg_chrom_ranges.json), cuts timing into thirds by CpG, and splits each library's
molecule file by its first CpG into early / mid / late parts. Each part is then counted by the one command (run_sample.py --dev-count-only)."""
import os, sys, json, gzip, glob, numpy as np
import py2bit, pyBigWig
W = os.environ.get("IAM_WORK", os.getcwd()) + "/"   # the workspace holding iamrepo/ and rt/ (hg19.2bit and the 15 Repli-seq bigWigs)
RANGES = json.load(open(W + "iamrepo/Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json"))["ranges"]
order = sorted(RANGES, key=lambda c: RANGES[c][0])
tb = py2bit.open(W + "rt/hg19.2bit"); bws = [pyBigWig.open(f) for f in sorted(glob.glob(W + "rt/*WaveSignalRep1.bigWig"))]
print(len(bws), "timing tracks; chromosome order", order[:5], "...")
timing = {}; bad = []
for c in order:
    a, b = RANGES[c]
    seq = tb.sequence(c).upper(); arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    pos = np.nonzero((arr[:-1] == ord("C")) & (arr[1:] == ord("G")))[0]
    n_exp = b - a + 1
    if len(pos) != n_exp and len(pos) != b - a: bad.append((c, len(pos), n_exp))
    vals = []
    for bw in bws:
        if c not in bw.chroms(): continue
        v = np.array(bw.values(c, 0, len(seq), numpy=True), dtype=np.float32)[pos]; vals.append(v)
    timing[c] = (a, np.nanmean(np.vstack(vals), axis=0) if vals else np.full(len(pos), np.nan))
print("numbering check (chromosomes whose CpG count differs from the chain's range):", bad[:5])
if bad: sys.exit("CpG numbering does not match wgbstools; stop")
allv = np.concatenate([t for _, t in timing.values()]); allv = allv[~np.isnan(allv)]
q1, q2 = np.quantile(allv, [1 / 3, 2 / 3]); print("timing thirds cut at", round(float(q1), 2), round(float(q2), 2), "(higher = earlier)")
def third(c, i):
    a, t = timing.get(c, (None, None))
    if t is None: return None
    k = i - a
    if k < 0 or k >= len(t) or np.isnan(t[k]): return None
    return "late" if t[k] < q1 else ("mid" if t[k] < q2 else "early")
for f in sys.argv[1:]:
    g = os.path.basename(f).split(".")[0]; outs = {k: gzip.open(f"{os.path.dirname(f)}/{g}_{k}.pat.gz", "wt") for k in ("early", "mid", "late")}; n = dict(early=0, mid=0, late=0, none=0)
    for line in gzip.open(f, "rt"):
        c, i = line.split("\t", 2)[:2]; k = third(c, int(i))
        if k: outs[k].write(line); n[k] += 1
        else: n["none"] += 1
    for o in outs.values(): o.close()
    print(g, n)
