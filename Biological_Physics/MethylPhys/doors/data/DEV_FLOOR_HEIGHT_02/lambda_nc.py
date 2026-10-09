"""Bisulfite/enzymatic non-conversion on the unmethylated lambda spike in a bwa-meth BAM: fraction of reference C (CpG and non-CpG) read as C.
Usage: lambda_nc.py REF.fa OUT.json BAM... (contig named lambda* in the hg19_lambda_puc19 index)."""
import sys, json, subprocess, os, re
fa = sys.argv[1]
names = [l.split("\t")[0] for l in open(fa + ".fai")]
lam = next(n for n in names if n.lower().startswith("lambda"))
ref = subprocess.run(["samtools", "faidx", fa, lam], capture_output=True, text=True).stdout.split("\n", 1)[1].replace("\n", "").upper()
out = {}
for bam in sys.argv[3:]:
    c = {"cpg_C": 0, "cpg_T": 0, "chh_C": 0, "chh_T": 0}; n = 0
    p = subprocess.Popen(["samtools", "view", "-F", "3844", "-q", "10", bam, lam], stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        f = l.split("\t"); n += 1; r = int(f[3]) - 1; q = 0; seq = f[9]; yd = next((x[5:].strip() for x in f[11:] if x.startswith("YD:Z:")), None)
        for ln, op in re.findall(r"(\d+)([MIDNSHP=X])", f[5]):
            ln = int(ln)
            if op in "M=X":
                for i in range(ln):
                    rp = r + i
                    if not 0 < rp < len(ref) - 1: continue
                    b = seq[q + i]
                    if yd == "f" and ref[rp] == "C":
                        k = "cpg" if ref[rp + 1] == "G" else "chh"; c[k + "_C"] += b == "C"; c[k + "_T"] += b == "T"
                    elif yd == "r" and ref[rp] == "G":
                        k = "cpg" if ref[rp - 1] == "C" else "chh"; c[k + "_C"] += b == "G"; c[k + "_T"] += b == "A"
                r += ln; q += ln
            elif op in "DN": r += ln
            elif op in "IS": q += ln
    nc = lambda a, b: a / (a + b) if a + b else None
    out[os.path.basename(bam)] = dict(reads=n, **c, nonconv_cpg=nc(c["cpg_C"], c["cpg_T"]), nonconv_chh=nc(c["chh_C"], c["chh_T"]))
    print(os.path.basename(bam), json.dumps(out[os.path.basename(bam)]), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
