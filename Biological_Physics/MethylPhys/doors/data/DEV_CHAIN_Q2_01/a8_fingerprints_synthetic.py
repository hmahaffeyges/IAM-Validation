#!/usr/bin/env python3
"""A8 fingerprints (register A8; exploration, 2026-11-16; author: 'Why cant we run the test on synthetic data and or use the simulator to test
out the 5hmC and traffic levers?'). Runs ON A GITHUB MACHINE. Planted .pat files of held molecules (14 calls, 60,000 molecules each), one per
A8 candidate, every candidate raising the TRUE loss from the base 0.040 to 0.050 (x1.25), read through the one command (run_sample.py, count only, as
the run-length synthetic). Question: which of the chain's readings tell the candidates apart, so the train reads what A8 needs.
  base       independent losses at 0.040
  crew       (a) slower restore crew: independent losses at 0.050 (also stands for (b) double loss and for uniform traffic: on one strand a
             lost pair reads as one T exactly like a lost mark, and an extra uniform loss rate is independent too; those need hairpin reads)
  traffic    (iv) traffic in stretches: base 0.040 plus events that clear a run of neighbouring CpGs (mean length 3), adding 0.010
  hmc        (c) transient 5hmC: true loss 0.050, of which 0.010 sits as 5hmC mid-erase, which bisulfite reads as C (marked)
  minority   a minority of cells much worse: 85 % of molecules at 0.040 and 15 % at 0.1067 (mean 0.050)
Readings from the chain: held_err (isolated losses per opportunity), coupling, loss_run_ratio, loss_runs. Read here from the same file (the chain
does not report it): per-molecule dispersion = variance of losses per molecule over its binomial variance, and the share of molecules with 3+
losses. Nothing fitted; seeds fixed."""
import os, sys, json, gzip, random, subprocess, tempfile, statistics
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"
W = tempfile.mkdtemp(); L = 14; N = 60000
def mol(g, kind):
    if kind == "base": return ["T" if g.random() < 0.040 else "C" for _ in range(L)]
    if kind == "crew": return ["T" if g.random() < 0.050 else "C" for _ in range(L)]
    if kind == "traffic":
        s = ["T" if g.random() < 0.040 else "C" for _ in range(L)]
        for q in range(L):                           # an event starts at q with rate r; it clears a geometric run (mean 3)
            if g.random() < 0.010 / 3.0:
                k = 1
                while g.random() < 2 / 3 and k < L: k += 1
                for j in range(q, min(L, q + k)): s[j] = "T"
        return s
    if kind == "hmc":                                # lost (T) at 0.040; 5hmC mid-erase at 0.010, read as C by bisulfite
        out = []
        for _ in range(L):
            u = g.random(); out.append("T" if u < 0.040 else "C")
        return out
    if kind == "minority":
        p = 0.1067 if g.random() < 0.15 else 0.040
        return ["T" if g.random() < p else "C" for _ in range(L)]
def make(path, kind, seed):
    g = random.Random(seed); per = []
    with gzip.open(path, "wt") as f:
        for m in range(N):
            s = mol(g, kind); per.append(s.count("T"))
            f.write(f"chr1\t{1000 + 20 * m}\t{''.join(s)}\t1\n")
    mu = statistics.mean(per); var = statistics.pvariance(per); p = mu / L
    return dict(true_loss=(0.050 if kind != "base" else 0.040), read_loss_per_call=round(mu / L, 5),
                dispersion=round(var / (L * p * (1 - p)), 3), share_3plus=round(sum(x >= 3 for x in per) / N, 4))
def run(kind, seed):
    pat = os.path.join(W, kind + ".pat.gz"); extra = make(pat, kind, seed)
    b = os.path.join(W, kind + ".json")
    q = subprocess.run([sys.executable, "run_sample.py", "--pat", pat, "--specimen", "cultured cells", "--seq-cell", "syn", "--id", "syn",
                        "--dev-count-only", "--dev-allow-partial-genome", "--dev-count-genome", "hg19",
                        "--out", os.path.join(W, kind + ".html"), "--bundle", b], cwd=IF, capture_output=True, text=True)   # the one command, as loss_runs_synthetic.py
    if q.returncode != 0: print("::error::run failed " + kind + " " + q.stdout[-400:] + q.stderr[-400:]); sys.exit(1)
    c = json.load(open(b))["iam_a_count_only"]["count"]
    r = dict(kind=kind, held_err=c.get("held_err"), coupling=c.get("coupling"), loss_run_ratio=c.get("loss_run_ratio"),
             runs_obs=c.get("loss_runs", {}).get("obs"), runs_exp=c.get("loss_runs", {}).get("exp"), **extra)
    print("::notice::A8 FINGERPRINT " + json.dumps(r)); return r
R = [run(k, 20261116 + i) for i, k in enumerate(("base", "crew", "traffic", "hmc", "minority"))]
B = R[0]
for r in R[1:]:
    print("::notice::A8 VS BASE " + json.dumps(dict(kind=r["kind"], held_err_x=round(r["held_err"] / B["held_err"], 3),
          coupling_minus_base=round((r["coupling"] or 0) - (B["coupling"] or 0), 3), run_ratio_minus_base=round((r["loss_run_ratio"] or 0) - (B["loss_run_ratio"] or 0), 3),
          dispersion=r["dispersion"], share_3plus_x=round(r["share_3plus"] / B["share_3plus"], 2))))
os.makedirs("xout", exist_ok=True); json.dump(R, open("xout/a8_fingerprints.json", "w"), indent=1)
