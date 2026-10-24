"""DEV-ONE-DRAW-01 synthetic test cfDNA 1 (written and committed before it runs; plan 40.11). Why IAM-A reads low on plasma: mixing of cell
types, or short fragments. Constructed molecules in the .pat form on every autosome (the first 40 tiles of 2,000 CpGs of each, hg19 CpG
indices as Stage Q2's tiles): two cell types with the same per-call loss rate EPS on molecules held marked; each holds its own 8 % of tiles
unmarked (molecules at most 1 call in 12 methylated). Read alone, mixed 70/30 and 50/50 by molecule, and with every molecule cut from 12
calls to 8, 6 and 5 (run 2, after run 1 stopped at 5 calls: no molecule that short qualifies). Each file through run_sample.py --pat --specimen plasma (IAM-A on the self-reference line, shape check not applied).
Usage: python3 synthetic_cfdna_01.py [OUT.txt]"""
import sys, os, gzip, json, subprocess, tempfile, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_q2_selfref as Q2
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "synthetic_cfdna_01_summary.txt")
g = np.random.default_rng(20261024); T = Q2.hg19_tiles(); AUT = [f"chr{i}" for i in range(1, 23)]; NT, MOL, EPS = 40, 70, 0.035
RG = json.load(open(os.path.join(CH, "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")))["ranges"]
# only tiles inside each chromosome's index range in the genome dictionary (Q0.2): Stage Q2's tiles for chr15-chr22 are offset from it (plan 14.9)
tiles = [(c, int(a)) for c in AUT for a, b in [t for t in T[c] if t[0] + 1 >= RG[c][0] and t[1] <= RG[c][1]][:NT]]; n = len(tiles)
lowA = set(g.choice(n, int(0.08 * n), replace=False)); lowB = set(g.choice(n, int(0.08 * n), replace=False))
def mols(low, k):
    L = []
    for t, (c, a) in enumerate(tiles):
        for m in range(MOL):
            st = a + 1 + int(g.integers(0, 2000 - 12)); r = g.random(k)   # CpG indices start at 1 (Q0.2)
            s = "".join(("C" if x < 0.04 else "T") if t in low else ("T" if x < EPS else "C") for x in r)
            L.append((c, st, s))
    return L
def read(tag, M):
    d = tempfile.mkdtemp(); f = os.path.join(d, tag + ".pat.gz"); M.sort(key=lambda x: (AUT.index(x[0]), x[1]))
    with gzip.open(f, "wt") as fh:
        for c, st, s in M: fh.write(f"{c}\t{st}\t{s}\t1\n")
    b = os.path.join(d, "b.json"); p = subprocess.run([sys.executable, RS, "--pat", f, "--specimen", "plasma", "--id", tag, "--out", os.path.join(d, "r.html"), "--bundle", b], capture_output=True, text=True)
    q = (json.load(open(b)).get("iam_a") or {}) if os.path.exists(b) else {}
    if not q.get("count"): sys.exit(f"{tag}: run_sample.py exit {p.returncode}: {str(q)[:300]} {p.stdout[-500:]} {p.stderr[-500:]}")
    c = q["count"]; f_ = lambda v, n: "none" if v is None else f"{v:.{n}f}"
    return f"{tag:22} loose share {f_(c.get('loose_share'), 4)} | held error {f_(c.get('held_err'), 5)} (opportunities {c.get('held_opp')}) | IAM-A {q.get('A')} {q.get('state') or q.get('refusal') or ''}"
def mix(A, B, wa):
    k = int(round(wa * len(A))); ia = g.choice(len(A), k, replace=False); ib = g.choice(len(B), len(A) - k, replace=False)
    return [A[i] for i in ia] + [B[i] for i in ib]
L = [f"DEV-ONE-DRAW-01 synthetic test cfDNA 1: {n} tiles on 22 autosomes, {MOL} molecules per tile, per-call loss {EPS}, seed 20261024"]
for k in (12, 8, 6, 5):
    A, B = mols(lowA, k), mols(lowB, k)
    for tag, M in ((f"cell A, {k} calls", list(A)), (f"cell B, {k} calls", list(B)), (f"mix 70/30, {k} calls", mix(A, B, 0.7)), (f"mix 50/50, {k} calls", mix(A, B, 0.5))):
        L.append(read(tag.replace(" ", "_").replace(",", "").replace("/", "-"), M)); print(L[-1], flush=True)
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
