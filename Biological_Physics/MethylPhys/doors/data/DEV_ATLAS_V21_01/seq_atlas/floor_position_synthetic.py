"""Synthetic for the library floor by molecule position (run_sample.py --build seq-floor-position; written and committed 2026-11-09 before it
runs; development). Known answer: constructed .pat files of molecules over the floor sites (chain/Runtime Matrices/Seq_Atlas/
floor_sites_v1.parquet), 6 to 14 consecutive CpGs each, a third of them pairs with a gap ('.') inside, every opposite call planted: at
interior CpGs with rate F_INT, at the first and last CpG read with rate F_END. Libraries: (a) F_INT 0.005, F_END 0.005 (no end effect);
(b) F_INT 0.005, F_END 0.05 (ten times at the ends); (c) F_INT 0.03, F_END 0.06 (a plasma-level floor, twice at the ends). 400,000
molecules each, seeded. Each read by the one command's build route. Bars: 1. (a) reads end over interior within 0.8 to 1.25; 2. (b) and (c)
read it within 10 % of the planted 10 and 2; 3. every distance from 1 CpG on reads the interior rate within 15 % in all three; 4 (added 2026-11-09 with the site-matched measure, before
its run). the site-matched end over inner reads (a) 0.8 to 1.25 and (b), (c) within 15 % of the planted 10 and 2 (inner calls include distances 1 to 4).
Usage: floor_position_synthetic.py OUTDIR"""
import os, sys, json, gzip, subprocess
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain")); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
F = pd.read_parquet(os.path.join(CH, "Runtime Matrices", "Seq_Atlas", "floor_sites_v1.parquet")); side = dict(zip(F.cpg.astype(np.int64), F.side)); fc = F.cpg.astype(np.int64).to_numpy()
LIBS = {"a": (0.005, 0.005), "b": (0.005, 0.05), "c": (0.03, 0.06)}; res = {}
for name, (fi, fe) in LIBS.items():
    g = np.random.default_rng({"a": 20261109, "b": 20261110, "c": 20261111}[name]); f = os.path.join(out, f"syn_{name}.pat.gz")
    with gzip.open(f, "wt") as o:
        for _ in range(400000):
            L = int(g.integers(6, 15)); c0 = int(fc[g.integers(len(fc))]) - int(g.integers(0, L)); s = []
            gap = (g.random() < 1 / 3) and L >= 8; ga = int(g.integers(3, L - 3)) if gap else -1
            for j in range(L):
                if gap and j in (ga, ga + 1): s.append("."); continue
                sd = side.get(c0 + j); end = j == 0 or j == L - 1
                if sd is None: s.append("C" if g.random() < 0.5 else "T"); continue
                opp = g.random() < (fe if end else fi)
                s.append(("C" if opp else "T") if sd == "unmarked" else ("T" if opp else "C"))
            o.write(f"chr1\t{c0}\t{''.join(s)}\t1\n")
    bj = os.path.join(out, f"syn_{name}.json")
    r = subprocess.run([sys.executable, RS, "--build", "seq-floor-position", "--build-input", f, "--bundle", bj], capture_output=True, text=True)
    res[name] = json.load(open(bj)) if r.returncode == 0 else {"rc": r.returncode, "err": r.stdout[-300:] + r.stderr[-300:]}; os.remove(f)
L_ = []; ok = []
for name, (fi, fe) in LIBS.items():
    R = res[name]; eo = R.get("end_over_interior"); bd = R.get("by_distance_from_end") or {}; ir = (bd.get("interior") or {}).get("rate")
    mids = [(k, (bd.get(k) or {}).get("rate")) for k in ("1", "2", "3", "4")]
    L_.append(f"library {name} (planted interior {fi}, ends {fe}): end over interior {eo}; interior {ir}; distances 1-4 {mids}")
    if name == "a": ok.append(eo is not None and 0.8 <= eo <= 1.25)
    else: ok.append(eo is not None and abs(eo / (fe / fi) - 1) <= 0.10)
    ok.append(ir is not None and all(v is not None and abs(v / ir - 1) <= 0.15 for _, v in mids))
m1 = ok[0]; m2 = ok[2] and ok[4]; m3 = ok[1] and ok[3] and ok[5]
sm_ = {nm: ((res[nm].get("site_matched") or {}).get("both") or {}).get("end_over_inner") for nm in LIBS}
m4 = sm_["a"] is not None and 0.8 <= sm_["a"] <= 1.25 and all(sm_[nm] is not None and abs(sm_[nm] / (LIBS[nm][1] / LIBS[nm][0]) - 1) <= 0.15 for nm in ("b", "c"))
L_.append(f"site-matched end over inner: {sm_}")
L_ += [f"bar 1 (no end effect reads none): {'met' if m1 else 'not met'}", f"bar 2 (planted end effects recovered within 10 %): {'met' if m2 else 'not met'}",
       f"bar 3 (inner distances read the interior rate): {'met' if m3 else 'not met'}", f"bar 4 (site-matched): {'met' if m4 else 'not met'}", "MET" if (m1 and m2 and m3 and m4) else "NOT MET"]
open(os.path.join(out, "floor_position_synthetic.txt"), "w").write("\n".join(L_) + "\n"); print("\n".join(L_))
