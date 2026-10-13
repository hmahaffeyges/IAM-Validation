"""DEV-IAMA-CAUSE-01: read the 15 parts through the chain's Stage Q and write the table the decision rule takes.

The chain unchanged: stage_q_iam_a.pat_site_table on each .pat.gz, copy error = (err_A + err_B) / (opp_A + opp_B), exactly as the
decision rule's docstring states. Each part file is read whole first (decide_cause_01.whole) so a truncated or doubly written part
cannot reach the decision (LESSONS C12). Roles come from cause_runs.csv, which was committed before any part was read.

Usage: python3 read_parts_01.py PART_DIR [OUTDIR]        (PART_DIR holds <run>p<0|40|80>.pat.gz)
Writes cause_parts_01.csv: run, part, eps, role, lines.
"""
import os, re, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
sys.path.insert(0, HERE)
from stage_q_iam_a import pat_site_table
from decide_cause_01 import whole

PART_DIR = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
R = pd.read_csv(os.path.join(HERE, "cause_runs.csv")).set_index("run").role
rows = []
for f in sorted(os.listdir(PART_DIR)):
    m = re.match(r"(SRR\d+)p(\d+)\.pat\.gz$", f)
    if not m: continue
    run, part = m.group(1), int(m.group(2))
    p = os.path.join(PART_DIR, f)
    n = whole(p)                                        # refuses a truncated part before it is read
    T = pat_site_table(p)
    eps = float((T.err_A.sum() + T.err_B.sum()) / (T.opp_A.sum() + T.opp_B.sum()))
    rows.append(dict(run=run, part=part, eps=eps, role=R.get(run, "unknown"), lines=n, sites=int(len(T))))
    print(f"{run} p{part}: eps {eps:.6f} | {n:,} lines | {len(T):,} sites", flush=True)
D = pd.DataFrame(rows).sort_values(["run", "part"])
assert (D.role != "unknown").all(), "a part has no role in cause_runs.csv"
assert len(D) == 15, f"expected 15 parts, found {len(D)}"
D.to_csv(os.path.join(OUT, "cause_parts_01.csv"), index=False)
print("wrote cause_parts_01.csv", len(D), "parts")
