#!/usr/bin/env python3
"""Synthetic for run_sample.py --dev-count-sites with a comma list (2026-11-16). The islands synthetic's planted file (islands 0.004, elsewhere
0.015) is read twice through the one command: --dev-count-sites islands, and --dev-count-sites islands,epitoc2. Checks: (1) the islands count in
the two-set run equals the islands-only run exactly; (2) count_sites_all holds both sets; (3) count_sites is the first set. Exit 1 on failure."""
import os, sys, json, subprocess
H = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
src = open(os.path.join(H, "islands_synthetic.py")).read(); exec(src[:src.index("L, bad = [")])
pat = os.path.join(W, "multi.pat.gz"); make_pat(pat, 0.004, 0.015)
def run(sets, tag):
    b = os.path.join(W, f"multi_{tag}.json")
    p = subprocess.run([sys.executable, "run_sample.py", "--pat", pat, "--specimen", "cultured cells", "--seq-cell", "syn", "--id", "syn",
                        "--dev-count-only", "--dev-allow-partial-genome", "--dev-count-genome", "hg19", "--dev-count-sites", sets,
                        "--out", os.path.join(W, f"r_{tag}.html"), "--bundle", b], cwd=IF, capture_output=True, text=True)
    if p.returncode != 0: print("::error::run failed " + sets + " " + p.stdout[-400:] + p.stderr[-400:]); sys.exit(1)
    return json.load(open(b))["iam_a_count_only"]
one = run("islands", "one"); two = run("islands,epitoc2", "two")
k = [x for x in one["count_sites"] if x not in ("cell", "sites")]
ok1 = all(one["count_sites"][x] == two["count_sites_all"]["islands"][x] for x in k)
ok2 = set(two["count_sites_all"]) == {"islands", "epitoc2"}
ok3 = two["count_sites"]["cell"] == "islands"
msg = f"multi-site synthetic: islands equal {ok1}; both sets kept {ok2}; first set in count_sites {ok3}; islands gain_err {one['count_sites'].get('gain_err')}; epitoc2 gain_opp {two['count_sites_all']['epitoc2'].get('gain_opp')}"
os.makedirs("xout", exist_ok=True); open("xout/multi_sites_synthetic_summary.txt", "w").write(msg + "\n"); print("::notice::" + msg)
sys.exit(0 if ok1 and ok2 and ok3 else 1)
