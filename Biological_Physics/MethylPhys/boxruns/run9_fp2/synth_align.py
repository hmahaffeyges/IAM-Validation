"""Box Run 9 synthetic alignment test (2026-10-10, before any rerun; LESSONS B1). Chooses bwa mem's minimum alignment score (-T, passed through
bwa-meth 0.2.0, which puts extra arguments after its own -T 40) for 30-36-base directional RRBS reads, by a rule fixed here before running.
Reads: 400,000 from hg19 chr1-chr22 windows holding >= 6 CpGs within the read, both original strands (OT, OB), length uniform 30-36 (after
trimming); every CpG methylated except planted copy errors at eps 0.03 (C -> T, independent); every non-CpG C converted; 0.5 % random
substitutions. Each read's name carries its true chromosome, start and strand.
Per T in 16, 18, 20, 22, 25, 28, 30, 40: share aligned with MAPQ >= 10 (wgbstools bam2pat's default filter), share of those placed within 3 bases
of the truth, and Stage Q's copy error on the .pat (eps_of.py) against eps_ref = the same Stage Q rule applied to the calls of the final simulated reads (sequencing errors included), at their true CpGs.
SELECTION RULE (fixed before running): the LARGEST T with unique share >= 0.80, misplaced <= 0.01 and |eps_T / eps_ref - 1| <= 0.05.
If none qualifies, the rerun does not start.
Usage: python3 synth_align.py gen REF.fa OUT.fq TRUTH.json | python3 synth_align.py place BAM OUT.json"""
import sys, json, random, gzip
def rc(s): return s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))
if sys.argv[1] == "gen":
    # Each chromosome is read once; every start whose 30-base window holds >= 6 CpGs (a read of any length 30-36 then holds >= 6) is a candidate;
    # reads are drawn uniformly over candidates, chromosomes in proportion to their candidate counts. (2026-10-10: replaces rejection sampling,
    # which spent the job's hour on random lookups; selection rule unchanged.)
    import pysam, numpy as np
    fa = pysam.FastaFile(sys.argv[2]); rg = random.Random(20261010); npr = np.random.default_rng(20261010); EPS = 0.03; ERR = 0.005; NREADS = 400000
    chroms = [f"chr{i}" for i in range(1, 23)]; SEQ = {}; CAND = {}
    for c in chroms:
        q = fa.fetch(c).upper(); SEQ[c] = q; a = np.frombuffer(q.encode(), np.uint8)
        cg = np.zeros(len(a), np.int32); cg[:-1] = (a[:-1] == 67) & (a[1:] == 71)
        w = np.convolve(cg, np.ones(29, np.int32), "valid")          # CpG starts in [st, st+29) -> a CpG fully inside 30 bases
        st = np.nonzero(w >= 6)[0]; st = st[(st > 1_000_000) & (st < len(a) - 1_000_000)]; CAND[c] = st
        print(c, len(st), flush=True)
    tot = sum(len(v) for v in CAND.values()); pick = npr.choice(len(chroms), NREADS, p=[len(CAND[c]) / tot for c in chroms])
    n = 0; nE = nO = 0
    with open(sys.argv[3], "w") as out:
        for ci in pick:
            c = chroms[ci]; st = int(CAND[c][npr.integers(len(CAND[c]))]); ln = rg.randint(30, 36); top = SEQ[c][st:st + ln + 1]
            if "N" in top: continue
            strand = rg.choice("+-"); s = top if strand == "+" else rc(SEQ[c][st - 1:st + ln])
            r = []
            for i in range(ln):
                b = s[i]
                if b == "C" and s[i + 1] == "G": r.append("C" if rg.random() >= EPS else "T")
                elif b == "C": r.append("T")
                else: r.append(b)
            cg = [i for i in range(ln) if s[i] == "C" and s[i + 1] == "G"]
            r = "".join(x if rg.random() >= ERR else rg.choice([y for y in "ACGT" if y != x]) for x in r)
            k = "".join(r[i] for i in cg if r[i] in "CT")     # calls as the final read shows them (sequencing errors included)
            if len(k) >= 6 and k.count("C") >= 0.8 * len(k):
                nO += len(k) - 2; nE += sum(1 for i in range(1, len(k) - 1) if k[i] == "T" and k[i - 1] == "C" and k[i + 1] == "C")
            out.write(f"@{c}:{st}:{strand}:{n}\n{r}\n+\n{'I' * ln}\n"); n += 1
    json.dump({"reads": n, "eps_ref_stageQ_rule": nE / nO, "planted_eps": EPS}, open(sys.argv[4], "w")); print("reads", n, "eps_ref", round(nE / nO, 5))
elif sys.argv[1] == "place":
    import pysam
    b = pysam.AlignmentFile(sys.argv[2]); tot = q = ok = 0
    for a in b.fetch(until_eof=True):
        if a.is_secondary or a.is_supplementary: continue
        tot += 1
        if a.is_unmapped or a.mapping_quality < 10: continue
        q += 1; c, st, strand, _ = a.query_name.split(":"); ok += (a.reference_name == c and abs(a.reference_start - int(st)) <= 3)
    json.dump({"reads": tot, "unique_share": q / tot, "misplaced": 1 - ok / max(q, 1)}, open(sys.argv[3], "w")); print(q / tot, 1 - ok / max(q, 1))
