#!/usr/bin/env python3
"""Alignment record for Stage Q0 (Box Run 2): bisulfite conversion and duplicate fraction from a bwa-meth BAM marked by Sambamba.

conversion_rate = converted / (converted + unconverted) cytosine calls in CHH context (H = A, C or T), where cytosines are unmethylated in
the source: the lambda spike-in when it carries >= MIN_CALLS CHH calls, otherwise CHH on human chr1 reads (CHH methylation is near
zero in blood cells, so this reads slightly low if any is present). CHH (not CHG) so that dcm sites (CCWGG) in lambda grown in dcm+ E. coli cannot count. Strand from
bwa-meth's YD tag: f = C->T strand (reference C, read T converted / C not), r = G->A strand (reference G, read A converted / G not).
duplicate_fraction = duplicates / mapped primary reads, from samtools flagstat on the Sambamba-marked BAM.
Usage: alignment_qc.py MARKED.bam REF.fa OUT.json [--max-reads N]"""
import sys, json, subprocess, re

MIN_CALLS = 1000
CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def load(fa, names):
    seqs, cur, keep = {}, None, False
    for l in open(fa):
        if l.startswith(">"):
            cur = l[1:].split()[0]; keep = cur in names
            if keep: seqs[cur] = []
        elif keep: seqs[cur].append(l.strip().upper())
    return {k: "".join(v) for k, v in seqs.items()}


def chh_calls(bam, contig, ref, max_reads):
    conv = unconv = n = 0
    p = subprocess.Popen(["samtools", "view", "-F", "3844", "-q", "10", bam, contig], stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        f = l.rstrip("\n").split("\t"); n += 1
        if n > max_reads: break
        yd = next((x[5:] for x in f[11:] if x.startswith("YD:Z:")), None)
        if yd not in ("f", "r"): continue
        pos = int(f[3]) - 1; seq = f[9]; i = 0
        for k, op in CIG.findall(f[5]):
            k = int(k)
            if op in "M=X":
                for j in range(k):
                    r = pos + j; b = seq[i + j]
                    if r + 2 >= len(ref) or r < 2: continue
                    if yd == "f" and ref[r] == "C" and ref[r + 1] != "G" and ref[r + 2] != "G":
                        conv += b == "T"; unconv += b == "C"
                    elif yd == "r" and ref[r] == "G" and ref[r - 1] != "C" and ref[r - 2] != "C":
                        conv += b == "A"; unconv += b == "G"
                pos += k; i += k
            elif op in "IS": i += k
            elif op in "DN": pos += k
    p.kill()
    return conv, unconv, n


def main():
    bam, fa, out = sys.argv[1:4]
    mx = int(sys.argv[sys.argv.index("--max-reads") + 1]) if "--max-reads" in sys.argv else 2_000_000
    rec = {"bam": bam, "context": "CHH", "strand_tag": "YD (bwa-meth)"}
    R = load(fa, {"lambda", "chr1"})
    c, u, n = chh_calls(bam, "lambda", R.get("lambda", ""), mx) if "lambda" in R else (0, 0, 0)
    src = "lambda spike-in"
    if c + u < MIN_CALLS:
        rec["lambda_chh_calls"] = c + u; src = "human chr1 CHH (no usable lambda spike-in)"
        c, u, n = chh_calls(bam, "chr1", R["chr1"], mx)
    rec.update(source=src, chh_converted=c, chh_unconverted=u, reads_scanned=n,
               conversion_rate=(round(c / (c + u), 5) if c + u >= MIN_CALLS else None))
    fs = subprocess.run(["samtools", "flagstat", bam], capture_output=True, text=True).stdout
    g = lambda pat: int(re.search(pat, fs).group(1)) if re.search(pat, fs) else None
    dup, prim = g(r"(\d+) \+ \d+ duplicates"), g(r"(\d+) \+ \d+ primary mapped")
    rec.update(duplicates=dup, primary_mapped=prim, duplicate_fraction=(round(dup / prim, 5) if dup is not None and prim else None),
               marked_by="Sambamba 0.6.5 markdup (Loyfer 2023 parameters)")
    json.dump(rec, open(out, "w"), indent=1); print(json.dumps(rec))


if __name__ == "__main__":
    main()
