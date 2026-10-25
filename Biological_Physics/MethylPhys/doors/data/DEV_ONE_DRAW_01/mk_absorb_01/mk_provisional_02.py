"""DEV-ONE-DRAW-01 provisional reference test 2 (written and committed 2026-10-24 before column version 2 is read on donor 2; plan 40.18).
Column version 2 (stage_provisional.column_v2: level from the shrunk proportion, counting spread through the inverse transfer by its interval
ends, sites with at least 5 reads), built from GSE87197 donor 1 only. DEVELOPMENT on donor 2: donor 2 was read in test 1 and is spent, so
a pass here is a development result; the confirmation is donor 3 (unread). The readings are test 1's saved files (donor 2's pure pool and
the spent healthy plasma with 0, 10, 25 and 40 % of donor 2 planted), read again through the one command with --dev-provisional
megakaryocytes@2. Bars: test 1's three (pure pool largest and at least 0.50; rises at each step; stand-ins rise at most 0.02, 0 to 40 %).
Usage: python3 mk_provisional_02.py DIR_WITH_TEST1_FILES OUTDIR"""
import sys, os, json, subprocess
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py")
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
F = {"pool": "mk_d2_pool.lbeta", "plasma_mk00": "plasma_mk00.lbeta", "plasma_mk10": "plasma_mk10.lbeta", "plasma_mk25": "plasma_mk25.lbeta", "plasma_mk40": "plasma_mk40.lbeta"}
SI = ("lung interstitial macrophages", "erythrocyte progenitors", "MEP (bone marrow)", "eosinophils"); R = {}
for tag, f in F.items():
    b = os.path.join(out, tag + ".json")
    subprocess.run([sys.executable, RS, "--dev-seq-beta", os.path.join(src, f), "--dev-seq-values", "1", "--specimen", "plasma", "--id", "v2_" + tag,
                    "--dev-provisional", "megakaryocytes@2", "--out", os.path.join(out, tag + ".html"), "--bundle", b], capture_output=True, text=True)
    Q = json.load(open(b)).get("cell_shares_provisional") or {}; P = {x["cell"]: x["share"] for x in Q.get("cells") or []}
    k = "megakaryocytes (provisional)"; R[tag] = dict(mk=P.get(k), stand=sum(P.get(x, 0) for x in SI), top=bool(P) and max(P, key=P.get) == k, markers=Q.get("markers"))
    print(tag, R[tag], flush=True)
p, b = R["pool"], R["plasma_mk00"]
L = ["DEV-ONE-DRAW-01 provisional reference test 2 (column version 2 from donor 1; read on donor 2, spent: development)",
     f"markers {p['markers']} | pure donor 2 pool: megakaryocytes {p['mk']:.4f} (largest: {p['top']}) | stand-ins {p['stand']:.4f}",
     f"bar 1 (largest share and at least 0.50): {'met' if p['top'] and p['mk'] >= 0.50 else 'not met'}", f"  0 %: megakaryocytes {b['mk']:.4f} | stand-ins {b['stand']:.4f}"]
ok, prev = True, b["mk"]
for f in (10, 25, 40):
    r = R[f"plasma_mk{f:02d}"]; e = (1 - f / 100) * b["mk"] + f / 100 * p["mk"]; ok &= r["mk"] > prev; prev = r["mk"]
    L.append(f"  {f} %: megakaryocytes {r['mk']:.4f} | simple mixing {e:.4f} | stand-ins {r['stand']:.4f}")
L.append(f"bar 2 (rises at each step): {'met' if ok else 'not met'}")
d = R["plasma_mk40"]["stand"] - b["stand"]; L.append(f"bar 3 (stand-ins rise at most 0.02, 0 to 40 %): rise {d:.4f}: {'met' if d <= 0.02 else 'not met'}")
open(os.path.join(out, "mk_provisional_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
