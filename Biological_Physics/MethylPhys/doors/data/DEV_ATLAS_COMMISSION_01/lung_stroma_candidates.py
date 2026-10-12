"""DEV-ATLAS-COMMISSION-01: purified healthy lung stromal cells for the atlas, metadata only (no IDAT opened). Re-queries GEO for the two
candidates found 2026-10-12 and counts the healthy, untreated arrays. Writes lung_stroma_candidates_output.txt. Usage: python3 lung_stroma_candidates.py"""
import os, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
def gsm(acc):
    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": acc, "targ": "gsm", "form": "text", "view": "brief"}, timeout=180).text.replace("\r", "")
    R = []
    for b in t.split("^SAMPLE = ")[1:]:
        get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        ch = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
        R.append(dict(gsm=b.split("\n")[0].strip(), platform=get("!Sample_platform_id")[0], idat=sum(".idat" in u.lower() for u in get("!Sample_supplementary_file")), **ch))
    return pd.DataFrame(R)
out = []
A = gsm("GSE146376"); h = A[(A.asthma == "No") & (A.treatment == "Vehicle")]
out.append(f"GSE146376 airway smooth muscle (primary culture): {len(A)} arrays, {A.donor.nunique()} donors; untreated (Vehicle) non-asthma arrays {len(h)} from {h.donor.nunique()} donors; platforms {sorted(A.platform.unique())}; arrays with IDATs {int((A.idat > 0).sum())}")
F = gsm("GSE107226"); n = F[F["disease state"] == "normal"]
out.append(f"GSE107226 lung fibroblasts (culture): {len(F)} arrays; normal lung {len(n)}; platforms {sorted(F.platform.unique())}; arrays with IDATs {int((F.idat > 0).sum())}")
open(os.path.join(HERE, "lung_stroma_candidates_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
