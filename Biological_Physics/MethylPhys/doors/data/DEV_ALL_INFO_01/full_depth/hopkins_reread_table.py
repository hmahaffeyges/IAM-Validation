"""DEV-ALL-INFO-01: the study re-read with its cell type given (sorted naive B cells, IAM-Atlas v2.1's naive B-cell column), each value copied
from the one command's re-read bundle (nothing computed here), beside the same sample's earlier full-depth reading's held-error chances (the
same molecules must give the same count). Usage: python3 hopkins_reread_table.py LAND_DIR OUT.csv"""
import sys, os, json, glob, csv
H = os.path.dirname(os.path.abspath(__file__)); S = {}
for r in csv.DictReader(open(os.path.join(H, "hopkins_reread_samples.csv"))): S.setdefault(r["gsm"], r)
E = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "hopkins_full_depth.csv")))} if os.path.exists(os.path.join(H, "hopkins_full_depth.csv")) else {}
F = ["gsm", "title", "group", "iam_a", "iam_a_state", "met_a_sorted", "coupling_a", "coupling_state", "islands_A", "domains_A", "met_a_whole",
     "held_chances", "held_chances_before", "same_molecules", "reference"]
rows = []
for g, r in S.items():
    f = sorted(glob.glob(os.path.join(sys.argv[1], f"hr_{g}", "**", g + "_bundle.json"), recursive=True))
    if not f: continue
    B = json.load(open(f[-1])); q = B.get("iam_a") or {}; k = B.get("coupling_a") or {}; t = (B.get("territories") or {}).get("sets") or {}
    c = (q.get("count") or {}).get("held_opp")
    old = sorted(x for x in glob.glob(os.path.join(sys.argv[1], "split_*", "**", g + "_bundle.json"), recursive=True) if "/pass1" not in x)
    b = (((json.load(open(old[-1])).get("iam_a") or {}).get("count") or {}).get("held_opp")) if old else None   # the earlier full-depth reading
    rows.append(dict(gsm=g, title=r["title"], group=r["genotype"], iam_a=q.get("A"), iam_a_state=q.get("state"),
                     met_a_sorted=((B.get("development") or {}).get("metaa_atlas") or {}).get("A"), coupling_a=k.get("A"), coupling_state=k.get("state"),
                     islands_A=(t.get("islands") or {}).get("A"), domains_A=(t.get("domains") or {}).get("A"), met_a_whole=(B.get("met_a_whole") or {}).get("A"),
                     held_chances=c, held_chances_before=b, same_molecules=None if (c is None or not b) else str(int(float(c)) == int(float(b))),
                     reference=B.get("whole_sample_reference")))
with open(sys.argv[2], "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=F); w.writeheader(); w.writerows(rows)
print("re-read", len(rows), "of", len(S))
