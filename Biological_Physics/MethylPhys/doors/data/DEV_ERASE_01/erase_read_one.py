"""DEV-ERASE-01: Stage Q's counting on one sample's .pat (committed before any read). Writes <gsm>_stageq.json."""
import sys, os, json
sys.path.insert(0, os.getcwd())
from stage_q_iam_a import pat_site_table
pat, gsm = sys.argv[1], sys.argv[2]; T = pat_site_table(pat)
o, e = int(T.opp_A.sum() + T.opp_B.sum()), int(T.err_A.sum() + T.err_B.sum())
R = dict(gsm=gsm, opp=o, err=e, eps=(e / o if o else None), eps_A=float(T.err_A.sum() / T.opp_A.sum()) if T.opp_A.sum() else None,
         eps_B=float(T.err_B.sum() / T.opp_B.sum()) if T.opp_B.sum() else None, n_lines=T.attrs["n_lines"], n_qualifying_lines=T.attrs["n_qualifying_lines"],
         n_molecules=T.attrs["n_molecules"], qualifying_share=T.attrs["n_qualifying_lines"] / max(1, T.attrs["n_lines"]))
json.dump(R, open(f"{gsm}_stageq.json", "w"), indent=1); print(json.dumps(R))
