"""Copy error of a whole .pat(.gz) by Stage Q's own rule (stage_q_iam_a.pat_site_table): errors, opportunities, eps. Usage: eps_of.py CHAIN OUT.json PAT..."""
import sys, json, os
sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
out = {}
for p in sys.argv[3:]:
    T = Q.pat_site_table(p); e = float(T.err_A.sum() + T.err_B.sum()); o = float(T.opp_A.sum() + T.opp_B.sum())
    out[os.path.basename(p)] = dict(errors=e, opportunities=o, eps=e / o); print(os.path.basename(p), out[os.path.basename(p)], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
