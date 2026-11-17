#!/usr/bin/env python3
"""A8P step 2 (register A8, packing lead; exploration, 2026-11-16). Runs ON A GITHUB MACHINE. For each Roadmap epigenome named in the names
table (written before this ran; passed as a JSON argument {chain_cell: [EIDs]}), reads its 15-state ChromHMM segmentation (open) and reports
the share of the genome (bp) in each state, and the two packing shares fixed in the plan before reading: P1 = 9_Het + 15_Quies;
P2 = P1 + 13_ReprPC + 14_ReprPCWk. Only the shares come back as notice lines."""
import sys, json, gzip, io, urllib.request, collections
B = "https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/"
N = json.loads(sys.argv[1]); out = {}
for cell, eids in N.items():
    tot = collections.Counter()
    for e in eids:
        raw = urllib.request.urlopen(B + f"{e}_15_coreMarks_mnemonics.bed.gz", timeout=300).read()
        for l in gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines():
            p = l.split("\t")
            if len(p) >= 4 and p[0] not in ("chrY", "chrM"): tot[p[3]] += int(p[2]) - int(p[1])
    g = sum(tot.values()); sh = {k: round(v / g, 4) for k, v in sorted(tot.items())}
    P1 = sh.get("9_Het", 0) + sh.get("15_Quies", 0); P2 = P1 + sh.get("13_ReprPC", 0) + sh.get("14_ReprPCWk", 0)
    out[cell] = dict(eids=eids, P1=round(P1, 4), P2=round(P2, 4), shares=sh)
    print("::notice::A8P SHARE " + json.dumps(dict(cell=cell, eids=eids, P1=out[cell]["P1"], P2=out[cell]["P2"])))
import os; os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8p_shares.json", "w"), indent=1)
