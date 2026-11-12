"""The held-error counters on molecules whose truth we set (PHY-079 follow-up; synthetic; development; nothing sealed).
Question: which of the chain's held errors (Stage Q2 count_pat: raw held_err, length-corrected held_err_length = e(1-e)^2, per-call
held_e_call = e) returns the planted per-call loss e, and does the zero-loose intercept across samples recover it, (a) with losses
independent and (b) with losses that cluster (a loss makes the next call's loss more likely)? Molecule lengths are drawn from a real
library's qualifying-molecule lengths (percall_out, the first sample); loose regions are methylated at random levels 0.3-0.9 per region.
Each synthetic sample: N molecules on hg19-index chr1 tiles, written as a .pat file and read by the one command,
chain/MethylPhys_Interface/run_sample.py --pat FILE --dev-count-only, exactly as the real samples were counted. Writes
counter_synthetic.csv and counter_synthetic_summary.txt."""
import os, sys, json, glob, gzip, random, subprocess, tempfile, numpy as np, pandas as pd
from scipy.stats import theilslopes
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface")
E_TRUE, N, SEED = 0.0466, 120000, 20261112   # e(1-e)^2 = 0.04234 at e = 0.0466 ... the planted per-call loss
REAL = json.load(open(sorted(glob.glob(os.path.join(H, "percall_out", "*.json")))[0]))
REAL = REAL[0] if isinstance(REAL, list) else REAL
LENS = {int(k): v[0] for k, v in REAL["held_by_length"].items() if int(k) <= 40}
Ls, Ws = list(LENS), list(LENS.values())
RANGES = json.load(open(os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")))["ranges"]
AUTO = [(f"chr{k}", RANGES[f"chr{k}"]) for k in range(1, 23)]   # each autosome's [first, last] CpG index (wgbstools init_genome hg19), as intake checks
WTS = [b - a for _, (a, b) in AUTO]
def place(g, L):
    """a molecule's chromosome and first CpG index, uniform over the autosomes' CpGs in the genome-wide index the .pat uses"""
    c, (a, b) = g.choices(AUTO, weights=WTS)[0]; return c, g.randrange(a, b - L)
def molecule(g, L, held, e, q_after):
    if held:
        s = []; prev = 0
        for _ in range(L):
            p = q_after if prev else e; x = g.random() < p; s.append("T" if x else "C"); prev = x
        return "".join(s)
    lv = g.uniform(0.3, 0.9); return "".join("C" if g.random() < lv else "T" for _ in range(L))
def sample(loose, e, q_after, seed):
    g = random.Random(seed); lines = []
    for k in range(N):
        L = g.choices(Ls, weights=Ws)[0]; held = g.random() >= loose
        c, i = place(g, L)
        lines.append((i, f"{c}\t{i}\t{molecule(g, L, held, e, q_after)}\t1"))
    lines = [x for _, x in sorted(lines)]   # genome order, as a .pat is stored
    w = tempfile.mkdtemp(prefix="cs"); pat = os.path.join(w, "s.pat.gz"); b = os.path.join(w, "b.json")
    with gzip.open(pat, "wt") as f: f.write("\n".join(lines) + "\n")
    rc = subprocess.run(["python3", "run_sample.py", "--pat", pat, "--specimen", "sorted cells", "--id", f"synthetic_{seed}", "--dev-count-only",
                         "--out", os.path.join(w, "r.html"), "--bundle", b], cwd=IF, capture_output=True, text=True)
    if not os.path.exists(b): raise SystemExit(f"run_sample.py wrote no bundle (exit {rc.returncode}): {rc.stdout[-600:]}")
    B = json.load(open(b))
    if "iam_a_count_only" not in B: raise SystemExit(f"no count (seed {seed}): {B.get('reason') or B.get('status')} | {rc.stdout[-500:]}")
    return B["iam_a_count_only"]["count"]
rows = []
for case, q_after in (("independent", E_TRUE), ("clustered (next loss x1.5)", 1.5 * E_TRUE), ("clustered (next loss x3)", min(0.5, 3 * E_TRUE))):
    for j, ls in [(j, ls) for j, ls in enumerate((0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)) for _ in (0, 1)]:
        rep = sum(1 for x in rows if x["case"] == case and x["loose_planted"] == ls)
        r = sample(ls, E_TRUE, q_after, SEED + 100 * j + 10000 * rep + {"independent": 0, "clustered (next loss x1.5)": 3}.get(case, 7))
        rows.append(dict(case=case, loose_planted=ls, replicate=rep, raw=r["held_err"], length=r["held_err_length"], per_call=r["held_e_call"], coupling=r.get("coupling")))
        print(case, ls, round(r["held_err"], 5), r["held_err_length"] and round(r["held_err_length"], 5), r["held_e_call"] and round(r["held_e_call"], 5), flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(H, "counter_synthetic.csv"), index=False)
truth = dict(per_call=E_TRUE, lone=E_TRUE * (1 - E_TRUE) ** 2)
L = [f"Counter synthetic: planted per-call loss e = {E_TRUE}; its lone-loss chance e(1-e)^2 = {truth['lone']:.5f}", R.round(5).to_string(index=False), ""]
for case, g in R.groupby("case"):
    for col in ("raw", "length", "per_call"):
        gg = g.dropna(subset=[col]); s, b, _, _ = theilslopes(gg[col], gg.loose_planted)
        ref = truth["per_call"] if col == "per_call" else truth["lone"]
        L.append(f"{case:26} {col:9} at zero loose {g[g.loose_planted == 0][col].mean():.5f}  intercept {b:.5f}  truth {ref:.5f}  off {100 * (b / ref - 1):+.1f} %")
t = "\n".join(L) + "\n"; open(os.path.join(H, "counter_synthetic_summary.txt"), "w").write(t); print(t)
