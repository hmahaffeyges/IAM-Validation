"""DEV-IAMA-TRIM-01: the trim diagnostic measured on REAL reads, position by position, with no alignment and no reference.

# NO CHAIN: this computes a property of the raw reads (the chemistry's own fingerprint), not a physics reading. Stage Q is not
# called here; the trim rule it feeds is in trim_rule_01.py and its planted test in planted_trim_01.py.

What is measured, per window of reads, straight from the read sequences:

    d = count(C not followed by G) / count((C or T) not followed by G)

In a bisulfite library nearly every cytosine outside a CpG is converted to thymine, so d sits near zero and is a property of the
CHEMISTRY. It knows nothing about IAM-A, which is the whole point: the trim must be decided by something measured independently of
the reading it corrects.

Windows are pulled by position out of the archive (fastq-dump -N/-X, SRA random access), exactly as the cause run staged its parts,
so a whole file is never downloaded. Default 20 windows spread evenly over each run, 50,000 read pairs each.

Writes head_diag_reads_01.csv (run, window, start_spot, d, n_sites, reads) and, for each run, the decision trim_rule_01 would make -
WITHOUT changing K_SD or MAX_TRIM, which were fixed before this ran.

Usage: python3 head_diag_reads_01.py OUTDIR FASTQ_DUMP [RUN ...]
"""
import os, re, subprocess, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from trim_rule_01 import decide_trim, K_SD, MAX_TRIM
OUT = sys.argv[1]; FQD = sys.argv[2]
RUNS = sys.argv[3:] or ["SRR13051128", "SRR13051256", "SRR13051268", "SRR13051263", "SRR13051136"]
NWIN, WIN = 20, 50000
NCG = re.compile(r"C(?!G)")

def spots(run):
    import urllib.request
    t = urllib.request.urlopen("https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=" + run, timeout=180).read().decode()
    row = [l for l in t.strip().split("\n") if l.startswith(run)]
    hdr = t.strip().split("\n")[0].split(",")
    return int(row[0].split(",")[hdr.index("spots")])

def d_of(path):
    """C among C+T at positions NOT followed by G, over the read sequences in a fastq file."""
    c = ct = 0
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1: continue
            s = line.strip()
            c += len(NCG.findall(s))
            ct += len(NCG.findall(s)) + len([1 for m in re.finditer(r"T(?!G)", s)])
    return (c / ct if ct else float("nan")), ct

rows = []
for run in RUNS:
    n = spots(run); step = max(1, (n - WIN) // (NWIN - 1))
    for w in range(NWIN):
        a = 1 + w * step
        tmp = os.path.join(OUT, f"{run}_w{w}")
        os.makedirs(tmp, exist_ok=True)
        r = subprocess.run([FQD, "--split-files", "-N", str(a), "-X", str(a + WIN - 1), "-O", tmp, run],
                           capture_output=True, text=True)
        f1 = os.path.join(tmp, run + "_1.fastq")
        if r.returncode or not os.path.isfile(f1):
            rows.append(dict(run=run, window=w, start_spot=a, d=float("nan"), n_sites=0, reads=0, note="fastq-dump failed")); continue
        d, ct = d_of(f1)
        rows.append(dict(run=run, window=w, start_spot=a, d=d, n_sites=ct, reads=WIN, note=""))
        subprocess.run(["rm", "-rf", tmp])
        print(run, w, f"{d:.5f}", flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "head_diag_reads_01.csv"), index=False)
dec = {}
for run, g in T.groupby("run"):
    g = g.sort_values("window")
    o = decide_trim(g.d.values)
    dec[run] = dict(decision=o["decision"], skip_windows=int(o["skip_windows"]), trimmed_fraction=float(o["trimmed_fraction"]),
                    reason=o["reason"], d_head=float(g.d.iloc[0]), d_later=float(np.nanmedian(g.d.values[len(g) // 2:])))
json.dump(dict(K_SD=K_SD, MAX_TRIM=MAX_TRIM, windows=NWIN, window_reads=WIN, runs=dec),
          open(os.path.join(OUT, "head_diag_reads_01_decision.json"), "w"), indent=1)
print(json.dumps(dec, indent=1))
