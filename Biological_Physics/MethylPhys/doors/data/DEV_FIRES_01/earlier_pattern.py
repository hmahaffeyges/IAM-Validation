"""DEV-FIRES-01: the earlier instrument's GSE51032 results (VAL-047 phase 12, VAL047_phase12_gse51032_combined_results.json in the author's
evidence folder; not in this repository, because it carries a retired class-era floor) printed rounded as DEV_FIRES_01.md quotes them, into
earlier_pattern_summary.txt beside this file.  Usage: python3 earlier_pattern.py PATH_TO_THAT_JSON"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); import sys; J = json.load(open(sys.argv[1]))
L = ["Earlier instrument (VAL-047 phase 12) on GSE51032, 6-site immune panel (xu_6_immune): cases against controls, by years before diagnosis"]
for c, v in J["cancers"].items():
    for w, x in v["windows"].items():
        r = x.get("xu_6_immune")
        L.append(f"{c:<10} {w:<10} " + (f"n {x['n_cases']:3} | effect size {r['median_d']:.2f} | permutation p {r['perm_p']:.4f}" if r else f"{x.get('status')} (n {x.get('n_cases')})"))
open(os.path.join(H, "earlier_pattern_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
