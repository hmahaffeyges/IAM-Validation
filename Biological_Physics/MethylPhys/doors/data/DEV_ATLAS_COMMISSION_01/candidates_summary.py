"""DEV-ATLAS-COMMISSION-01: summary of the 2026-10-12 search for independent purified healthy cells (candidates_search_2026-10-12.json,
metadata only; the search's declared limits are its 'deviations'). Writes candidates_summary_output.txt. Usage: python3 candidates_summary.py"""
import os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); J = json.load(open(os.path.join(HERE, "candidates_search_2026-10-12.json")))
so = J["structured_output"]; A = pd.DataFrame(so["array_candidates"]); S = pd.DataFrame(so["sequencing_candidates"]); M = pd.DataFrame(so["mixtures"])
ind = lambda T: T["independent"].astype(str).str.lower().str.startswith(("true", "yes"))
out = [f"array candidates {len(A)} (independent {int(ind(A).sum())}); sequencing {len(S)} (independent {int(ind(S).sum())}); mixtures/truth sets {len(M)} (independent {int(ind(M).sum())})",
       f"cell types with no independent candidate: {len(so['cell_types_without_candidate'])}"]
for acc in ("GSE224807", "GSE127824", "GSE123995", "GSE146376", "GSE107226"):
    r = pd.concat([A, S, M])[lambda T: T.accession == acc]
    if len(r): x = r.iloc[0]; out.append(f"{acc}: {x.get('platform')} | {str(x.get('sample_and_donor_counts', x.get('notes', '')))[:180]}")
out.append("declared limits: " + " | ".join(d[:120] for d in J["deviations"]))
open(os.path.join(HERE, "candidates_summary_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
