"""The B and T sequencing commissioning test, third reading (every read pair of every run, after the genome-order finding), on the full working chain against the IAM Sequencing Atlas (written and committed
before any of these readings exists). Same 12 healthy sorted cells from two laboratories neither atlas holds (GSE134664: 2 naive and 2 memory B
cells; GSE98837: 8 CD8 T cells), each laboratory's molecules re-merged from the first reading's saved chunks (bt_seq2_chunks.csv; GSM3963004's
empty chunk 12 aligned again) and read once by the one command with the flags in bt_seq2_samples.csv (the healthy passes' flags, against every
donor of the sequencing atlas, no reference half: none of these is an atlas input). The instruments' job: a healthy cell reads Normal.
 bar 1 Met-A in the sample's own cell: at least 9 of 12 within 0.95 to 1.05 (the gauge's Normal; the first reading's band was never built)
 bar 2 coupling-A: at least 9 of 12 Normal
 bar 3 IAM-A: at least 9 of 12 Normal
Recorded, not scored: whole-sample Met-A, the islands, the domains, the C-Score, the Deconvolver's own-cell share, the library floor.
Usage: python3 bt_seq2_score.py BUNDLE_DIR OUT.csv OUT.txt"""
import sys, os, json, glob, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "bt_seq3_samples.csv")).drop_duplicates("gsm"); rows = []
for r in S.itertuples():
    f = [x for x in glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_bundle.json"), recursive=True)]
    o = dict(gsm=r.gsm, series=r.gse, cell=r.cell)
    if f:
        B = json.load(open(f[0])); D = B.get("development") or {}; T = (B.get("territories") or {}).get("sets") or {}
        sh = {x["cell"]: x["share"] for x in ((B.get("cell_shares") or {}).get("cells") or [])}
        o.update({"Met-A own cell": (D.get("metaa_atlas") or {}).get("A"), "Met-A reason": (D.get("metaa_atlas") or {}).get("reason"),
                  "coupling-A": (B.get("coupling_a") or {}).get("A"), "coupling state": (B.get("coupling_a") or {}).get("state"),
                  "IAM-A": (B.get("iam_a") or {}).get("A"), "IAM-A state": (B.get("iam_a") or {}).get("state"),
                  "Met-A whole sample": (B.get("met_a_whole") or {}).get("A"), "islands": (T.get("islands") or {}).get("A"), "domains": (T.get("domains") or {}).get("A"),
                  "C-Score": ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"), "own-cell share": sh.get(r.cell),
                  "floor": ((D.get("library_floor") or {}).get("floor"))})
    else: o["not read"] = "no bundle"
    rows.append(o)
d = pd.DataFrame(rows); d.to_csv(sys.argv[2], index=False)
m = pd.to_numeric(d.get("Met-A own cell"), errors="coerce"); n1 = int(m.between(0.95, 1.05).sum())
n2 = int((d.get("coupling state") == "Normal").sum()); n3 = int((d.get("IAM-A state") == "Normal").sum()); N = len(d)
L = [f"B and T sequencing test, second reading: {int(m.notna().sum())} of {N} with Met-A read"]
for _, r in d.iterrows(): L.append("  " + " | ".join(f"{k} {r[k]}" for k in d.columns if k not in ("series",) and pd.notna(r[k])))
L += [f"bar 1 Met-A own cell within 0.95 to 1.05: {n1} of {N}: {'met' if n1 >= 9 else 'not met'}",
      f"bar 2 coupling-A Normal: {n2} of {N}: {'met' if n2 >= 9 else 'not met'}",
      f"bar 3 IAM-A Normal: {n3} of {N}: {'met' if n3 >= 9 else 'not met'}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
