"""CpG methylation calls on the pUC19 spike in a bwa-meth BAM (YD strand tag). Usage: puc19.py REF.fa OUT.json BAM..."""
import sys, json, subprocess, os, re
ref = subprocess.run(["samtools", "faidx", sys.argv[1], "pUC19"], capture_output=True, text=True).stdout.split("\n", 1)[1].replace("\n", "").upper()
out = {}
for bam in sys.argv[3:]:
    meth = unm = n = 0
    p = subprocess.Popen(["samtools", "view", "-F", "3844", "-q", "10", bam, "pUC19"], stdout=subprocess.PIPE, text=True)
    for l in p.stdout:
        f = l.rstrip("\n").split("\t"); n += 1; pos = int(f[3]) - 1; seq = f[9]; yd = next((x[5:] for x in f[11:] if x.startswith("YD:Z:")), None)
        ops = re.findall(r"(\d+)([MIDNSHP=X])", f[5]); r = pos; q = 0
        for ln, op in ops:
            ln = int(ln)
            if op in "M=X":
                for i in range(ln):
                    rp = r + i
                    if 0 < rp < len(ref) - 1:
                        if yd == "f" and ref[rp] == "C" and ref[rp + 1] == "G":
                            b = seq[q + i]; meth += b == "C"; unm += b == "T"
                        elif yd == "r" and ref[rp] == "G" and ref[rp - 1] == "C":
                            b = seq[q + i]; meth += b == "G"; unm += b == "A"
                r += ln; q += ln
            elif op in "DN": r += ln
            elif op in "IS": q += ln
    out[os.path.basename(bam)] = dict(reads=n, cpg_meth=meth, cpg_unmeth=unm, frac_meth=(meth / (meth + unm) if meth + unm else None))
    print(os.path.basename(bam), out[os.path.basename(bam)], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
