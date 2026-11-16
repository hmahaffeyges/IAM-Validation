#!/usr/bin/env python3
"""Synthetic for the run-length spectrum of losses (register CP1, 2026-11-16). Two planted .pat files of held molecules (14 calls each, 60,000
molecules): (1) every call lost independently with probability 0.05; (2) the same mean loss planted as PAIRS of adjacent losses. Read through the
one command (count only). Checks: (1) independent: loss_run_ratio within 0.97-1.03 and observed runs of length 1 and 2 within 5 % of expected;
(2) paired: loss_run_ratio below 0.75 and observed runs of length 2 above 1.5 x expected. Exit 1 on failure."""
import os, sys, json, gzip, random, subprocess, tempfile
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
W = tempfile.mkdtemp(); rng = random.Random(16)
def make(path, paired):
    with gzip.open(path, "wt") as f:
        for m in range(60000):
            if paired:
                s = ["C"] * 14
                for _ in range(14):
                    if rng.random() < 0.025:
                        q = rng.randrange(13); s[q] = s[q + 1] = "T"
            else:
                s = ["T" if rng.random() < 0.05 else "C" for _ in range(14)]
            f.write(f"chr1\t{1000 + 20 * m}\t{''.join(s)}\t1\n")
def run(tag, paired):
    pat = os.path.join(W, tag + ".pat.gz"); make(pat, paired); b = os.path.join(W, tag + ".json")
    p = subprocess.run([sys.executable, "run_sample.py", "--pat", pat, "--specimen", "cultured cells", "--seq-cell", "syn", "--id", "syn",
                        "--dev-count-only", "--dev-allow-partial-genome", "--dev-count-genome", "hg19",
                        "--out", os.path.join(W, tag + ".html"), "--bundle", b], cwd=IF, capture_output=True, text=True)
    if p.returncode != 0: print("::error::run failed " + tag + " " + p.stdout[-400:] + p.stderr[-400:]); sys.exit(1)
    return json.load(open(b))["iam_a_count_only"]["count"]
I = run("indep", False); P = run("paired", True)
oi, ei = I["loss_runs"]["obs"], I["loss_runs"]["exp"]; op, ep = P["loss_runs"]["obs"], P["loss_runs"]["exp"]
ok1 = 0.97 <= I["loss_run_ratio"] <= 1.03 and all(abs(oi[r] - ei[r]) <= 0.05 * ei[r] for r in ("1", "2"))
ok2 = P["loss_run_ratio"] < 0.75 and op["2"] > 1.5 * ep["2"]
msg = (f"run-length synthetic: independent ratio {I['loss_run_ratio']:.3f} obs {oi} exp {ei} -> {'met' if ok1 else 'NOT met'}; "
       f"paired ratio {P['loss_run_ratio']:.3f} obs {op} exp {ep} -> {'met' if ok2 else 'NOT met'}; coupling indep {I.get('coupling')} paired {P.get('coupling')}")
os.makedirs("xout", exist_ok=True); open("xout/loss_runs_synthetic_summary.txt", "w").write(msg + "\n"); print("::notice::" + msg[:1800])
sys.exit(0 if ok1 and ok2 else 1)
