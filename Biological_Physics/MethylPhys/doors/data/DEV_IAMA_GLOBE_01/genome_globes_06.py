"""DEV-IAMA-GLOBE-01, step 6 (written 2026-10-22; development, pictures; the chain's way). Steps 2 to 5 called the globe code directly and are
withdrawn. Here every globe is drawn by the one command, chain/MethylPhys_Interface/run_sample.py --pat ... --dev-genome-globe, unchanged.
This file only prepares the input files (downloads; the constructed halves and plant) and runs the one command; it reads nothing itself.
  VIEW self           GSM5652282 against its own predicted held error
  VIEW twice          GSM5652282 read twice: its molecules split in two halves by a seeded coin (seed 20261022; each pair of lines, the unit
                      the counter reads, to one half), half 2 --pat against half 1 --prior-pat
  VIEW twice_planted  the same, with a CONSTRUCTED loss in half 2 only (held molecules in chr2's CpG 2,500,001 to 2,800,000; each interior
                      methylated call between two methylated ones turned unmethylated with probability 0.15)
  VIEW people         GSM5652282 against GSM5652283 (two healthy people)
Usage: python3 genome_globes_06.py VIEW OUTDIR   (writes OUTDIR/VIEW.json, the bundle; OUTDIR/VIEW_explorer.json; OUTDIR/VIEW.log)"""
import os, sys, gzip, subprocess, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
RS = os.path.join(MP, "chain", "MethylPhys_Interface", "run_sample.py"); view, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.join(MP, "chain")); import stage_q0_intake as Q0
def fetch(g):   # 2026-10-22 (lesson C34): the chain fetches the file, on the build it reads (hg19); never picked by name here
    return Q0.geo_pat(g, out)["path"]
PL = (2284470 + 2_500_001, 2284470 + 2_800_000)   # chr2's CpG 2,500,001 to 2,800,000 in the counter's genome-wide CpG index (chr1 holds 2,284,470)
def split(src, h, dst, plant=False):
    g = np.random.default_rng(20261022); gp = np.random.default_rng(1); side = None
    with gzip.open(src, "rt") as fi, gzip.open(dst, "wt") as fo:
        for k, line in enumerate(fi):
            if k % 2 == 0: side = int(g.random() < 0.5)
            if side != h: continue
            if plant and k % 2 == 1:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and PL[0] <= int(p[1]) <= PL[1]:
                    pat = list(p[2]); cl = [q for q, x in enumerate(pat) if x != "."]
                    for j in range(1, len(cl) - 1):
                        if pat[cl[j - 1]] == pat[cl[j]] == pat[cl[j + 1]] == "C" and gp.random() < 0.15: pat[cl[j]] = "T"
                    p[2] = "".join(pat); line = "\t".join(p) + "\n"
            fo.write(line)
    return dst
a = fetch("GSM5652282") if view in ("twice", "twice_planted") else None; cmd = ["--seq-cell", "cd8 t cells", "--dev-genome-globe", "--genome-region", "1000"]
if view == "self": args = ["--geo-pat", "GSM5652282", "--geo-dir", out, "--id", "GSM5652282"]
elif view in ("twice", "twice_planted"):
    h1 = split(a, 1, os.path.join(out, "half1.pat.gz")); h2 = split(a, 0, os.path.join(out, "half2.pat.gz"), plant=view == "twice_planted")
    args = ["--pat", h2, "--prior-pat", h1, "--prior-name", "the same sample's other half", "--id", "GSM5652282_" + view] + (["--banner", "PLANTED: a constructed loss in one stretch of half 2, not a person."] if view == "twice_planted" else [])
elif view == "people": args = ["--geo-pat", "GSM5652282", "--prior-geo-pat", "GSM5652283", "--geo-dir", out, "--prior-name", "another healthy person (GSM5652283)", "--id", "GSM5652282_vs_GSM5652283"]
else: sys.exit("VIEW is self, twice, twice_planted or people")
r = subprocess.run([sys.executable, RS] + args + cmd + ["--out", os.path.join(out, view + ".html"), "--bundle", os.path.join(out, view + ".json"),
                    "--explorer-spec", os.path.join(out, view + "_explorer.json")], capture_output=True, text=True)
open(os.path.join(out, view + ".log"), "w").write(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:]); print(r.stdout[-3000:]); sys.exit(r.returncode)
