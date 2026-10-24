"""DEV-METAA-HEALTHY-01 layer 1, counter test 1 (synthetic; first written to call Stage Q2's counter directly, rewritten before it counted
anything real so every file goes through the one command, rule 23). Constructed molecules written as a .pat.gz file: half held marked, each
call lost independently with rate eps_loss; half held unmarked, each call gained with rate eps_gain; 12 calls each. Each file read by
run_sample.py --dev-count-only. The count must return held_err near eps_loss (1 - eps_loss)^2 and gain_err near eps_gain (1 - eps_gain)^2 (a lone
error between two correct calls; the qualifying filter shifts this slightly), within 5 %; and with eps_loss = eps_gain, the two within 5 % of
each other. Usage: python3 gain_counter_01.py [OUT.txt]"""
import sys, os, gzip, json, subprocess, tempfile, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "gain_counter_01_summary.txt"); g = np.random.default_rng(20261024); N, K = 200000, 12
T = tempfile.mkdtemp(); L = ["DEV-METAA-HEALTHY-01 counter test 1: constructed molecules, 12 calls, seed 20261024, each file read by run_sample.py --dev-count-only"]
for el, eg in ((0.045, 0.045), (0.045, 0.012), (0.03, 0.006)):
    f = os.path.join(T, f"c_{el}_{eg}.pat.gz")
    with gzip.open(f, "wt") as fh:
        for k in range(N):
            held = k % 4 < 2; r = g.random(K); s = "".join(("T" if x < el else "C") if held else ("C" if x < eg else "T") for x in r)
            fh.write(f"chr1\t{1000 + k}\t{s}\t1\n")
    b = os.path.join(T, "b.json")
    p = subprocess.run([sys.executable, RS, "--pat", f, "--specimen", "purified cells", "--seq-cell", "constructed", "--dev-count-only", "--id", "constructed",
                        "--out", os.path.join(T, "r.html"), "--bundle", b], capture_output=True, text=True)
    c = (json.load(open(b)).get("iam_a_count_only") or {}).get("count") if os.path.exists(b) else None
    if not c: sys.exit(f"run_sample.py exit {p.returncode}: {p.stdout[-600:]} {p.stderr[-600:]}")
    xl, xg = el * (1 - el) ** 2, eg * (1 - eg) ** 2
    L.append(f"built loss {el} gain {eg} | held_err {c['held_err']:.5f} (expected about {xl:.5f}, ratio {c['held_err'] / xl:.3f}) | gain_err {c['gain_err']:.5f} (expected about {xg:.5f}, ratio {c['gain_err'] / xg:.3f}) | gain/held {c['gain_err'] / c['held_err']:.3f}")
    print(L[-1], flush=True)
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
