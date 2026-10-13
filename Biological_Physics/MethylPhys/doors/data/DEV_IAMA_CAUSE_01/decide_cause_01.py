"""DEV-IAMA-CAUSE-01 decision (rule fixed 2026-10-12, before any part is read). Input: a table with run, part (0, 40, 80), eps (Stage Q copy
error of that part, stage_q_iam_a.pat_site_table: (err_A+err_B)/(opp_A+opp_B)), role (low / normal). Rule:
  A = binary entropy H(eps). Reference at each part = mean A of the two normal runs at that part. r = A / reference.
  A low run is LIBRARY if r < 0.97 at all three parts; FILE-PART if r < 0.97 at part 0 and 0.97 <= r <= 1.03 at parts 40 and 80; otherwise MIXED.
  Overall: all three FILE-PART -> "file part": reading spread across the run becomes the pipeline's rule and the commissioning is re-sealed on it,
  disclosed. All three LIBRARY -> "library": across laboratories stays NOT MET for EMSeq. Anything else -> "mixed", reported as it is.
  Check (reported): each run's part 0 reproduces its Stage B copy error (stageB_libraries.csv) within 2 % relative.
Usage: python3 decide_cause_01.py PARTS.csv [STAGEB.csv]   |   python3 decide_cause_01.py --planted"""
# NO CHAIN: this script computes no physics. The copy error of every part is produced by the CHAIN
# (read_parts_01.py calls stage_q_iam_a.pat_site_table, which is where the reading is defined); this file only takes the
# binary entropy of those readings, divides by the control mean at the same part, and applies the thresholds sealed on
# 2026-10-12. Nothing here re-derives a reading, so there is nothing for the chain to do.

def whole(path):
    """Refuse a truncated or doubly written .pat.gz before it is read (2026-10-13, LESSONS C12): read the whole member and
    return its line count, so a part written by two runs at once cannot reach the decision."""
    import gzip
    with gzip.open(path, "rt") as g:
        n = sum(1 for _ in g)
    if n < 1000: raise SystemExit(f"decide_cause_01: {path} has only {n} lines; rerun that part")
    return n
import sys, os, json, numpy as np, pandas as pd
H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e))
def decide(T):
    T = T.copy(); T["A"] = H(T.eps); ref = T[T.role == "normal"].groupby("part").A.mean(); T["r"] = T.A / T.part.map(ref)
    v = {}
    for run, g in T[T.role == "low"].groupby("run"):
        r = g.set_index("part").r
        if all(r[p] < 0.97 for p in (0, 40, 80)): v[run] = "LIBRARY"
        elif r[0] < 0.97 and all(0.97 <= r[p] <= 1.03 for p in (40, 80)): v[run] = "FILE-PART"
        else: v[run] = "MIXED"
    s = set(v.values()); overall = "file part" if s == {"FILE-PART"} else "library" if s == {"LIBRARY"} else "mixed"
    return dict(per_run=v, overall=overall, r={f"{a}p{b}": round(c, 4) for a, b, c in T[["run", "part", "r"]].itertuples(index=False)})
if __name__ == "__main__":
    if sys.argv[1] == "--planted":
        def tab(low, norm=0.056):
            R = [dict(run=f"N{i}", part=p, eps=norm, role="normal") for i in (1, 2) for p in (0, 40, 80)]
            R += [dict(run=f"L{i}", part=p, eps=low[i][j], role="low") for i in range(3) for j, p in enumerate((0, 40, 80))]
            return pd.DataFrame(R)
        cases = [("low everywhere", tab([[.046] * 3] * 3), "library"), ("low at start only", tab([[.046, .056, .0557]] * 3), "file part"),
                 ("one run differs", tab([[.046] * 3, [.046, .056, .056], [.046] * 3]), "mixed"), ("low then high", tab([[.046, .056, .066]] * 3), "mixed")]
        out = []; ok = True
        for name, T, want in cases:
            got = decide(T)["overall"]; ok &= got == want; out.append(f"{'PASS' if got == want else 'FAIL'} {name}: {got} (want {want})")
        out.append("PLANTED TEST " + ("PASSED" if ok else "FAILED")); HERE = os.path.dirname(os.path.abspath(__file__))
        open(os.path.join(HERE, "decide_cause_01_planted_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out)); sys.exit(0 if ok else 1)
    T = pd.read_csv(sys.argv[1]); O = decide(T)
    if len(sys.argv) > 2:
        B = pd.read_csv(sys.argv[2]).set_index("run").eps
        O["p0_vs_stageB"] = {r: round(float(e / B[r] - 1), 4) for r, e in T[T.part == 0].set_index("run").eps.items()}
        O["p0_check"] = "PASSED" if all(abs(x) <= 0.02 for x in O["p0_vs_stageB"].values()) else "FAILED"
    print(json.dumps(O, indent=1))
