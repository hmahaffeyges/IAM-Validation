"""Re-run at the libraries' own lengths (2026-11-12, as sealed: only the molecule lengths change, drawn from the two control
libraries' pooled distribution in control_lengths.csv, MZGC1 and MZGC3 controls, before any depleted library is scored). Synthetic for DEV-METHIONINE-01's energy-or-supply test (PHY-083; synthetic; development; written before any library is read).
Question: on short reduced-representation molecules (RRBS, 150 bp reads; 3-15 CpGs per molecule, an assumption stated before reading,
to be replaced by the libraries' own lengths), does the chain's lone-loss counter separate a held-error rise of x1.066 (the energy
route at a four-fold SAM/SAH fall) from x1.5 and x4 (the supply route), at the depth of a real library? Each synthetic sample: N molecules
on hg19-index autosome tiles, written as a .pat file and read by the one command, run_sample.py --pat FILE --dev-count-only, exactly as the
counter synthetic (DEV_CHAIN_Q2_01/counter_synthetic.py), whose molecule and placement functions are reused. Loose share 0.20 in every
sample. Writes methionine_synthetic_summary.txt (and a per-sample csv kept outside the repository)."""
import os, sys, json, gzip, random, subprocess, tempfile, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(H, "..", "DEV_CHAIN_Q2_01"); sys.path.insert(0, Q)
src = open(os.path.join(Q, "counter_synthetic.py")).read(); src = src[:src.index("rows = []")]   # functions only, not its run
ns = {"__file__": os.path.join(Q, "counter_synthetic.py")}; exec(compile(src, "counter_synthetic.py", "exec"), ns)
IF = ns["IF"]; place, molecule = ns["place"], ns["molecule"]
E0, N, LOOSE, SEED = 0.0466, 150000, 0.20, 20261113
_LW = pd.read_csv(os.path.join(H, 'control_lengths.csv')); LS = _LW.cpgs_per_molecule.tolist(); LW = _LW.share.tolist()
def sample(e, seed):
    g = random.Random(seed); lines = []
    for _ in range(N):
        L = g.choices(LS, LW)[0]; held = g.random() >= LOOSE; c, i = place(g, L)
        lines.append((i, f"{c}\t{i}\t{molecule(g, L, held, e, e)}\t1"))
    lines = [x for _, x in sorted(lines)]
    w = tempfile.mkdtemp(prefix="ms"); pat = os.path.join(w, "s.pat.gz"); b = os.path.join(w, "b.json")
    with gzip.open(pat, "wt") as f: f.write("\n".join(lines) + "\n")
    rc = subprocess.run(["python3", "run_sample.py", "--pat", pat, "--specimen", "sorted cells", "--id", f"met_syn_{seed}", "--dev-count-only",
                         "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, capture_output=True, text=True)
    if not os.path.exists(b): raise SystemExit(f"run_sample.py wrote no bundle (exit {rc.returncode}): {rc.stdout[-600:]}")
    B = json.load(open(b))
    if "iam_a_count_only" not in B: raise SystemExit(f"no count (seed {seed}): {B.get('reason') or B.get('status')} | {rc.stdout[-500:]}")
    return B["iam_a_count_only"]["count"]["held_err_length"]
rows = []
for f in (1.0, 1.0, 1.066, 1.066, 1.5, 1.5, 4.0, 4.0):
    k = len(rows); v = sample(E0 * f, SEED + 101 * k); rows.append(dict(planted_rise=f, rep=k % 2, held_err_length=v)); print(f, v, flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(os.environ.get("TMPDIR", "/tmp"), "methionine_synthetic.csv"), index=False)
c = R[R.planted_rise == 1.0].held_err_length; base = c.mean(); spread = abs(c.iloc[0] - c.iloc[1]) / base
L = ["Methionine synthetic at the controls' own lengths (control_lengths.csv, loose 0.20, N = %d per sample)" % N, R.round(5).to_string(index=False), "",
     f"control replicate spread {100 * spread:.2f} %"]
for f in (1.066, 1.5, 4.0):
    m = R[R.planted_rise == f].held_err_length.mean(); L.append(f"planted x{f}: read x{m / base:.3f}")
t = "\n".join(L) + "\n"; open(os.path.join(H, "methionine_synthetic_lengths_summary.txt"), "w").write(t); print(t)
