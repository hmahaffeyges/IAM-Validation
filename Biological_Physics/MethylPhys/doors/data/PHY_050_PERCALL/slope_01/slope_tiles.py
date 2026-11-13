"""PHY-087, IAM-A's baseline slope, step 1 (exploration on spent samples; 2026-11-13). Where IAM's Law is: the baseline's form
eps_pred(L) = a + b L has a derived intercept (PHY-083) but a measured slope b (0.034625). Two physical readings of b:
  MIXING: inside one sample, molecules from weakly held tiles lose more; a sample with more loose tiles averages higher. Predicts a rise of
          tile held error as the tile's own marking falls, INSIDE each sample, of a size that reproduces b.
  CELL TYPE: types with more loose tiles hold worse everywhere. Predicts a flat relation inside samples; b appears only across samples.
Written before any tile is counted. For each sample in slope_samples.csv (spent Loyfer healthy, already counted for PHY-050, chosen to
span L): run_sample.py --geo-pat GSM --dev-count-only --dev-count-tiles (chain Stage Q2: held error per tile, two independent halves), and
from the same .pat the tile's mean call (molecules' mean call, the counter's own loose rule; tiles = CpG index // 2000). Writes
out/<GSM>_tiles.csv.gz (chrom, tile, iso_a, opp_a, iso_b, opp_b, calls, mean_call). Usage, from the repository root: python3 <this> WORKERS"""
import os, sys, csv, gzip, json, subprocess, tempfile, glob, collections, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); IF = os.path.join(HERE, "..", "..", "..", "..", "chain", "MethylPhys_Interface")
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
def one(row):
    g = row["gsm"]; dst = os.path.join(OUT, f"{g}_tiles.csv.gz")
    if os.path.exists(dst): return g, "exists"
    w = tempfile.mkdtemp(prefix=g); t = os.path.join(w, "t.csv.gz"); lg = os.path.join(OUT, f"{g}_log.txt")
    with open(lg, "w") as fh:
        rc = subprocess.run(["python3", "run_sample.py", "--geo-pat", g, "--geo-dir", w, "--specimen", "sorted cells", "--seq-cell", g, "--id", g,
                             "--dev-count-only", "--dev-count-tiles", t, "--out", os.path.join(w, "r.html"), "--bundle", os.path.join(w, "b.json")],
                            cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    pats = glob.glob(os.path.join(w, "*.pat.gz"))
    if rc != 0 or not os.path.exists(t) or not pats: subprocess.run(["rm", "-rf", w]); return g, f"FAILED rc {rc}"
    S = collections.defaultdict(lambda: [0, 0.0])   # (chrom, tile) -> [molecule calls, summed calls methylated]
    with gzip.open(pats[0], "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4: continue
            s_ = [x for x in p[2] if x != "."]
            if not s_: continue
            n = int(p[3]); a = S[(p[0], int(p[1]) // 2000)]; a[0] += n * len(s_); a[1] += n * s_.count("C")
    with gzip.open(t, "rt") as fi, gzip.open(dst, "wt") as fo:
        r = csv.reader(fi); h = next(r); fo.write(",".join(h + ["calls", "mean_call"]) + "\n")
        for x in r:
            c, m = S.get((x[0], int(x[1])), [0, 0.0]); fo.write(",".join(x + [str(c), f"{m / c:.5f}" if c else ""]) + "\n")
    subprocess.run(["rm", "-rf", w]); return g, "ok"
if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(HERE, "slope_samples.csv"))))
    with mp.get_context("fork").Pool(int(sys.argv[1])) as P:
        for g, st in P.imap_unordered(one, rows): print(g, st, flush=True)
