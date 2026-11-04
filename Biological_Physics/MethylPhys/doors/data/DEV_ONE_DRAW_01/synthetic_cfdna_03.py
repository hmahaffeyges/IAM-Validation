"""DEV-ONE-DRAW-01 synthetic test cfDNA 3 (written and committed before it runs; plan 40.11): the qualifying rule's length bias removed
(stage_q2_selfref.held_err_length, recorded by Stage Q2 as held_err_length, used by no reading yet). The instrument's job: a sample's held
error does not depend on how long its molecules are. Constructed molecules as in synthetic test cfDNA 1 (the same tiles and form), one cell
held marked with a per-call loss of EPS, cut to 6, 8, 12 and 20 calls; each file through run_sample.py --pat --specimen plasma.
Known answer: with no qualifying rule a molecule shows isolated losses at EPS (1 - EPS)^2. Bars: 1. the length-corrected held error within
5 % of that at every length; 2. its spread across the four lengths under half the raw held error's spread.
Usage: python3 synthetic_cfdna_03.py [OUT.txt]"""
import sys, os, gzip, json, subprocess, tempfile, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_q2_selfref as Q2
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "synthetic_cfdna_03_summary.txt")
g = np.random.default_rng(20261103); T = Q2.hg19_tiles(); AUT = [f"chr{i}" for i in range(1, 23)]; NT, MOL, EPS = 40, 70, 0.035
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
    return q["count"]
TRUTH = EPS * (1 - EPS) ** 2   # the held error a molecule shows with no qualifying rule: an isolated loss, e (1 - e)^2
L = [f"DEV-ONE-DRAW-01 synthetic test cfDNA 3: {n} tiles on 22 autosomes, {MOL} molecules per tile, per-call loss {EPS}, seed 20261103; truth {TRUTH:.5f}"]
raw, cor = [], []
for k in (6, 8, 12, 20):
    c = read(f"cell_A_{k}_calls", mols(lowA, k)); raw.append(c["held_err"]); cor.append(c.get("held_err_length"))
    L.append(f"{k} calls: held error {c['held_err']:.5f} | length-corrected {c.get('held_err_length') if c.get('held_err_length') is None else round(c['held_err_length'], 5)} | opportunities {c['held_opp']}"); print(L[-1], flush=True)
b1 = all(x is not None and abs(x / TRUTH - 1) <= 0.05 for x in cor); b2 = all(x is not None for x in cor) and (max(cor) - min(cor)) < 0.5 * (max(raw) - min(raw))
L += [f"bar 1 every length within 5 % of the truth: {'met' if b1 else 'not met'}", f"bar 2 the corrected spread across lengths under half the raw spread ({max(cor) - min(cor):.5f} against {max(raw) - min(raw):.5f}): {'met' if b2 else 'not met'}" if all(x is not None for x in cor) else "bar 2: not met (a length was not corrected)",
      f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
