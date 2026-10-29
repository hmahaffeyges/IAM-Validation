"""DEV-METAA-ATLAS-CURVE-01, sequencing commissioning test for B and T cells (written and committed before any of these samples is read).
12 healthy sorted cells sequenced by whole-genome bisulfite in two laboratories IAM-Atlas v2.1 has never held (GSE134664: 2 naive and 2
memory B cells; GSE98837: 8 CD8 T cells, naive, effector memory, central memory and stem-cell memory), each read by the one command from its
first 140 million read pairs (bt_seq_samples.csv). Scored, each instrument's job being that a healthy cell reads Normal on its gauge:
 bar 1 Met-A against the atlas on sequencing: at least 9 of the 12 (75 %) inside Met-A's sequencing band (metaa_seq_band_v1.json, built
       from the atlas's 153 healthy sequencing inputs before these are read)
 bar 2 coupling-A: at least 9 of 12 inside its Normal band (coupling_reference_v1.json)
 bar 3 IAM-A: at least 9 of 12 inside band v1 (GSE98837's laboratory was IAM-A's CD8 test laboratory; its CD8 cells are not new to IAM-A)
Read and reported, not scored: the territories, the Deconvolver's largest share, the C-Score, depth.
Usage: python3 bt_seq_score.py BUNDLE_DIR OUT.txt"""
import sys, os, json, glob, csv
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, *[".."] * 4, "chain", "Runtime Matrices")
S = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "bt_seq_samples.csv")))}
_bp = os.path.join(CH, "Atlas_V21", "metaa_seq_band_v1.json")
if not os.path.exists(_bp):   # the band is built from the atlas's sequencing inputs before this test is scored; until then nothing is scored
    open(sys.argv[2], "w").write("Met-A's sequencing band is not built yet; the test is scored once it is\n"); print("band not built yet"); raise SystemExit(0)
MB = json.load(open(_bp))
L = [f"Met-A sequencing band {MB['lo']:.4f} to {MB['hi']:.4f} (n {MB['n']})", "gsm | cell | Met-A | coupling-A | IAM-A | islands, domains | top share | held-error chances"]
n1 = n2 = n3 = read = 0
for g, r in S.items():
    f = [x for x in glob.glob(os.path.join(sys.argv[1], "**", g + "_bundle.json"), recursive=True) if "/pass1" not in x]
    if not f: L.append(f"{g} | {r['cell']} | not landed"); continue
    B = json.load(open(f[0])); read += 1
    m = ((B.get("development") or {}).get("metaa_atlas") or {}).get("A"); k = B.get("coupling_a") or {}; q = B.get("iam_a") or {}
    t = (B.get("territories") or {}).get("sets") or {}; cs = (B.get("cell_shares") or {}).get("cells") or []
    top = max(cs, key=lambda c: c["share"]) if cs else {}
    n1 += m is not None and MB["lo"] <= m <= MB["hi"]; n2 += k.get("state") == "Normal"; n3 += q.get("state") == "Normal"
    L.append(f"{g} | {r['cell']} | {m} | {k.get('A')} {k.get('state')} | {q.get('A')} {q.get('state')} | "
             f"{(t.get('islands') or {}).get('A')}, {(t.get('domains') or {}).get('A')} | {top.get('cell')} {top.get('share')} | {(q.get('count') or {}).get('held_opp')}")
L += [f"read {read} of {len(S)}",
      f"bar 1 Met-A on sequencing: {n1} of {len(S)} inside: {'met' if n1 >= 9 else 'not met'}",
      f"bar 2 coupling-A: {n2} of {len(S)} Normal: {'met' if n2 >= 9 else 'not met'}",
      f"bar 3 IAM-A: {n3} of {len(S)} Normal: {'met' if n3 >= 9 else 'not met'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:]))
