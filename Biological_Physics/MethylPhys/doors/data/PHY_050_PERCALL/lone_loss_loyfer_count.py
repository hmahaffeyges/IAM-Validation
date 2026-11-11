"""PHY-050, third test (2026-11-12; sealed in doors/PHY_050_LONE_LOSS_LOYFER_PREREG.md before any sample is counted): the same counting as the
per-call test (DEV_CHAIN_Q2_01/percall_count.py). Each healthy sorted Loyfer 2023 sample in lone_loss_loyfer_unseen_samples.csv is read by the one command, run_sample.py --geo-pat GSM --dev-count-only (the chain fetches the sample's
hg19 .pat.gz from GEO on the build it reads, and Stage Q2 counts it on the commissioned line's tiles, recording the per-call loss rate
held_e_call and held_err_length beside the counter's held error and the loose share). Run on GitHub's free machines in shards.
Usage, from the repository root: python3 <this> <shard> <nshards> <workers>   Output: xout/<GSM>.json (the count) and xout/<GSM>_log.txt"""
import os, sys, csv, json, subprocess, tempfile, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
IF = os.path.join(HERE, "..", "..", "..", "chain", "MethylPhys_Interface")
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)   # top level: the workflow uploads xout/*.json and xout/*.txt only
def one(row):
    g, cell = row; p = os.path.join(OUT, f"{g}.json")
    if os.path.exists(p): return g, "exists"
    w = tempfile.mkdtemp(prefix=g); b = os.path.join(w, "b.json"); lg = os.path.join(OUT, f"{g}_log.txt")
    for attempt in range(3):
        with open(lg, "w") as fh:
            rc = subprocess.run(["python3", "run_sample.py", "--geo-pat", g, "--geo-dir", w, "--specimen", "sorted cells", "--seq-cell", g, "--id", g,
                                 "--dev-count-only", "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0 and os.path.exists(b):
            c = json.load(open(b))["iam_a_count_only"]["count"]
            json.dump(dict(sample=g, cell=cell, counter="run_sample.py --geo-pat --dev-count-only (chain Stage Q2)", **c), open(p, "w"), indent=1)
            subprocess.run(["rm", "-rf", w]); return g, "ok"
    subprocess.run(["rm", "-rf", w]); return g, f"FAILED rc {rc}"
if __name__ == "__main__":
    k, n, wk = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rows = [(r["sample"], r["cell"]) for i, r in enumerate(csv.DictReader(open(os.path.join(HERE, "lone_loss_loyfer_unseen_samples.csv")))) if i % n == k]
    print("shard", k, "of", n, ":", len(rows), "samples", flush=True)
    with mp.get_context("fork").Pool(wk) as P:
        for g, st in P.imap_unordered(one, rows): print(g, st, flush=True)
