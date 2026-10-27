"""DEV-ALL-INFO-01: the study's samples at full depth in one table, each value copied from the one command's newest full-depth bundle for that
sample (nothing computed here; A-score gauge only). Usage: python3 hopkins_table.py LAND_DIR OUT.csv"""
import sys, os, json, glob, csv
H = os.path.dirname(os.path.abspath(__file__)); LAND, OUT = sys.argv[1:3]
S = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "..", "all_samples.csv"), newline="", encoding="utf-8-sig"))}
rows = []
for g, s in S.items():
    f = sorted([x for x in glob.glob(os.path.join(LAND, "split_*", "**", f"{g}_bundle.json"), recursive=True) if "/pass1" not in x], key=os.path.getmtime)
    if not f: continue
    B = json.load(open(f[-1])); q = B.get("iam_a") or {}; m = B.get("met_a_whole") or {}; T = (B.get("territories") or {}).get("sets") or {}
    cs = sorted((B.get("cell_shares") or {}).get("cells") or [], key=lambda c: -(c.get("share") or 0))
    rows.append(dict(gsm=g, sample=s["title"].replace("WGBS ", "").strip(), group=s["genotype"].strip(), specimen=(s.get("specimen") or "").strip(),
                     iam_a=q.get("A"), iam_a_state=q.get("state"), held_error=q.get("held_err"), met_a_whole=m.get("A"), met_a_ceiling=m.get("A_sat"),
                     islands_A=(T.get("islands") or {}).get("A"), domains_A=(T.get("domains") or {}).get("A"),
                     top_cell=cs[0]["cell"] if cs else None, top_share=round(cs[0]["share"], 3) if cs else None))
with open(OUT, "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r["group"], r["gsm"])))
print(len(rows), "samples of", len(S))
