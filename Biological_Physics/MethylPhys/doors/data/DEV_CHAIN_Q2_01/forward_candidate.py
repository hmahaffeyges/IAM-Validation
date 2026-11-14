"""The baseline candidate pushed through our own meter (PHY-083; forward model; development; nothing sealed).
The candidate (CANON eps0_candidate) is a per-site loss. The healthy value it is compared with (CANON eps_healthy_pool) is the counter's
lone-loss reading at zero loose share. Rather than correct the meter, this writes held molecules whose per-site loss is the candidate,
with the real libraries' molecule lengths (percall_out, as counter_synthetic.py) and a stated clustering, reads them with the one command
(run_sample.py --pat FILE --dev-count-only), and reports what the counter reads: held_err_length, held_e_call and the counter's own
clustering measure (coupling), to be set beside the healthy samples' measured coupling (median 1.46). Two ways losses can cluster:
  seq X : after a loss, the next call's loss chance is X times the candidate (losses follow each other along the molecule)
  het K : each molecule's loss chance is the candidate times a Gamma(K, 1/K) draw (some molecules or regions simply lose more)
  ind   : independent losses.
Usage: python3 forward_candidate.py MODE PARAM REPS OUTPREFIX [ERASER_RATIO]   (writes OUTPREFIX.csv and OUTPREFIX_summary.txt)"""
import os, sys, json, glob, gzip, random, subprocess, tempfile
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface")
E = json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["eps0_candidate"]["value"]
MODE, PARAM, REPS, OUT = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
# 2026-11-13 (PHY-084): optional 5th argument, the eraser ratio r (held error with the eraser removed over the control, measured:
# DEV-ERASE-01 recount median 0.8642 in stem cells; PHY-085 TET2 in blood progenitors, mean of the two units). The planted per-site
# loss is then the candidate over r (the candidate holds against thermal kicks alone; the eraser adds its measured share on top).
RATIO = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
E = E / RATIO
N, SEED = 120000, 20261113
REAL = json.load(open(sorted(glob.glob(os.path.join(H, "percall_out", "*.json")))[0])); REAL = REAL[0] if isinstance(REAL, list) else REAL
LENS = {int(k): v[0] for k, v in REAL["held_by_length"].items() if int(k) <= 40}; Ls, Ws = list(LENS), list(LENS.values())
RANGES = json.load(open(os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")))["ranges"]
AUTO = [(f"chr{k}", RANGES[f"chr{k}"]) for k in range(1, 23)]; WTS = [b - a for _, (a, b) in AUTO]
def molecule(g, L):
    if MODE == "het": e = min(0.5, E * g.gammavariate(PARAM, 1 / PARAM)); q = e
    elif MODE == "seq": e = E; q = min(0.5, PARAM * E)
    else: e = q = E
    s, prev = [], False
    for _ in range(L):
        x = g.random() < (q if prev else e); s.append("T" if x else "C"); prev = x
    return "".join(s)
def sample(seed):
    g = random.Random(seed); lines = []
    for _ in range(N):
        L = g.choices(Ls, weights=Ws)[0]; c, (a, b) = g.choices(AUTO, weights=WTS)[0]; i = g.randrange(a, b - L)
        lines.append((c, i, f"{c}\t{i}\t{molecule(g, L)}\t1"))
    lines.sort(key=lambda x: (x[0], x[1]))
    w = tempfile.mkdtemp(prefix="fc"); pat = os.path.join(w, "s.pat.gz"); bj = os.path.join(w, "b.json")
    with gzip.open(pat, "wt") as f: f.write("\n".join(x[2] for x in lines) + "\n")
    rc = subprocess.run(["python3", "run_sample.py", "--pat", pat, "--specimen", "sorted cells", "--id", f"fc_{seed}", "--dev-count-only",
                         "--dev-allow-partial-genome", "--out", os.path.join(w, "r.html"), "--bundle", bj], cwd=IF, capture_output=True, text=True)
    if not os.path.exists(bj): raise SystemExit(f"run_sample.py wrote no bundle (exit {rc.returncode}): {rc.stdout[-600:]}")
    B = json.load(open(bj))
    if "iam_a_count_only" not in B: raise SystemExit(f"no count: {B.get('reason') or B.get('status')} | {rc.stdout[-500:]}")
    return B["iam_a_count_only"]["count"]
rows = []
for r in range(REPS):
    c = sample(SEED + 1000 * r + int(PARAM * 100) + {"ind": 0, "seq": 1, "het": 2}[MODE] * 100000 + (0 if RATIO == 1.0 else int(RATIO * 1e4) * 1000000))
    rows.append((r, c["held_err_length"], c["held_e_call"], c.get("coupling"), c.get("held_err"))); print(MODE, PARAM, r, rows[-1][1:], flush=True)
# 2026-11-13: held_err (isolated losses over interior calls, the quantity behind CANON eps0_meth / H_ref_cell) added as the last column,
# so the Met-A healthy level (register M2) can be compared like for like.
with open(OUT + ".csv", "w") as f: f.write("mode,param,replicate,held_err_length,held_e_call,coupling,held_err\n" + "".join(f"{MODE},{PARAM},{a},{b},{c},{d},{e}\n" for a, b, c, d, e in rows))
m = lambda k: sum(x[k] for x in rows) / len(rows)
open(OUT + "_summary.txt", "w").write(f"forward model of the candidate (per-site loss {E:.5f}, eraser ratio {RATIO}) through the counter, {MODE} {PARAM}, {REPS} samples of {N} held molecules: "
                                      f"held_err_length {m(1):.5f}, held_e_call {m(2):.5f}, coupling {m(3):.3f}, held_err {m(4):.5f}\n")
print(open(OUT + "_summary.txt").read())
