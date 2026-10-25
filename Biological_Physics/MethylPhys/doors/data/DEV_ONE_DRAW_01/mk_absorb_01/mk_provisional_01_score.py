"""DEV-ONE-DRAW-01 provisional reference test 1, scoring (written after mk_provisional_01.py's own scoring stopped on an error: the chain's
sample-id hashing had replaced the cell name 'megakaryocytes' inside the pure pool's bundle, whose id was the same word; disclosed). The same
three bars as mk_provisional_01.py, read from its saved bundles; the provisional column is found by its ' (provisional)' suffix.
Usage: python3 mk_provisional_01_score.py OUTDIR"""
import sys, os, json
o = sys.argv[1]; SI = ("lung interstitial macrophages", "erythrocyte progenitors", "MEP (bone marrow)", "eosinophils"); R = {}
for t in ("megakaryocytes", "plasma_plus_mk_00pct", "plasma_plus_mk_10pct", "plasma_plus_mk_25pct", "plasma_plus_mk_40pct"):
    P = {x["cell"]: x["share"] for x in (json.load(open(os.path.join(o, t + ".json"))).get("cell_shares_provisional") or {}).get("cells") or []}
    k = next(c for c in P if c.endswith(" (provisional)")); R[t] = dict(mk=P[k], stand=sum(P.get(x, 0) for x in SI), top=max(P, key=P.get) == k)
p, b = R["megakaryocytes"], R["plasma_plus_mk_00pct"]
L = ["DEV-ONE-DRAW-01 provisional reference test 1 (column from GSE87197 donor 1; read on donor 2)",
     f"pure donor 2 pool: megakaryocytes (provisional) {p['mk']:.4f} (largest: {p['top']}) | stand-ins summed {p['stand']:.4f}",
     f"bar 1 (largest share and at least 0.50): {'met' if p['top'] and p['mk'] >= 0.50 else 'not met'}"]
ok, prev = True, b["mk"]; L.append(f"  0 %: megakaryocytes {b['mk']:.4f} | stand-ins {b['stand']:.4f}")
for f in (10, 25, 40):
    r = R[f"plasma_plus_mk_{f:02d}pct"]; e = (1 - f / 100) * b["mk"] + f / 100 * p["mk"]; ok &= r["mk"] > prev; prev = r["mk"]
    L.append(f"  {f} %: megakaryocytes {r['mk']:.4f} | simple mixing {e:.4f} | stand-ins {r['stand']:.4f}")
L.append(f"bar 2 (rises at each step): {'met' if ok else 'not met'}")
d = R["plasma_plus_mk_40pct"]["stand"] - b["stand"]; L.append(f"bar 3 (stand-ins rise at most 0.02, 0 to 40 %): rise {d:.4f}: {'met' if d <= 0.02 else 'not met'}")
open(os.path.join(o, "mk_provisional_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
