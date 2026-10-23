"""DEV-EACH-CELL-01, exploration 7 (written and committed 2026-10-22 before it runs; exploration, not sealed; to-do 39.9). Can a loss across a
cell's WHOLE pattern be put on the right cell? Exploration 4 found the identity sites could not (other cells moved 92 to 121 % as much),
because most held sites are held by the cell's relatives too. Here the same reading is also made on sites held by each cell and differing by at
least 0.1 or 0.2 from every other cell of the six-cell panel (run_sample.py --dev-metaa-specific 0.1,0.2). Every step is the one command, from
the raw IDATs of GSE235717 GSM7508429 (spent): healthy, then a 5 % loss at every site each of four cells holds (--dev-plant CELL:0.05:all).
Reported per plant and site set: the planted cell's move in its own error over sites, the largest move in another cell as a share of it, and
the sites used. Nothing scored. Usage: python3 specific_07.py OUTDIR"""
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True); G = "GSM7508429"; LOSS = 0.05; PLANT = ["monocytes", "cd4 t cells", "b cells", "nk cells"]
base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{G[:7]}nnn/{G}/suppl/"; files = {}
for k in range(5):
    try:
        for n in sorted(set(re.findall(rf"{G}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(base, timeout=120).read().decode()))):
            f = os.path.join(out, n); urllib.request.urlretrieve(base + n, f); files["grn" if "_Grn" in n else "red"] = f
        break
    except Exception as e: print("download retry", type(e).__name__, flush=True); time.sleep(60 * (k + 1))
assert set(files) == {"grn", "red"}, files
def one(tag, extra):
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--grn", files["grn"], "--red", files["red"], "--specimen", "whole blood", "--id", G, "--out",
                        os.path.join(out, tag + ".html"), "--bundle", b] + extra, capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    B = json.load(open(b)); return B, {x["cell"]: x for x in (B.get("met_a_each_cell") or {}).get("cells", [])}, {x["cell"]: x["share"] for x in B["cell_shares"]["cells"]}
def sets(B):
    o = {"identity": {x["cell"]: x for x in (B.get("met_a_each_cell") or {}).get("cells", [])}}
    for d, R in (B.get("met_a_each_cell_specific") or {}).items(): o[f"specific {d}"] = {x["cell"]: x for x in R.get("cells", [])}
    return o
X = ["--dev-metaa-specific", "0.1,0.2"]
K0 = sets(one("healthy", X)[0]); L = ["DEV-EACH-CELL-01 exploration 7: a 5 % loss at every site a cell holds, read on identity sites and on sites specific to each cell; all by run_sample.py on GSM7508429",
                                     "sites used: " + " | ".join(f"{k}: " + ", ".join(f"{c} {v.get('n_sites') or v.get('n_identity_sites')}" for c, v in R.items()) for k, R in K0.items())]
for c in PLANT:
    K1 = sets(one(f"all_{c.replace(' ', '_')}", X + ["--dev-plant", f"{c}:0.05:all"])[0])
    for k in K0:
        a0, a1 = (K0[k].get(c) or {}).get("A"), (K1[k].get(c) or {}).get("A")
        if a0 is None or a1 is None: L.append(f"plant {c:<12} | {k:<13}: not read ({(K1[k].get(c) or {}).get('reason')})"); continue
        se = K0[k][c].get("se_over_sites") or float("nan"); mv = a1 - a0
        oth = max(((K1[k][q]["A"] - K0[k][q]["A"]) / mv if mv else 0.0, q) for q in K0[k] if q != c and K0[k][q].get("A") is not None and (K1[k].get(q) or {}).get("A") is not None)
        L.append(f"plant {c:<12} | {k:<13}: moved {mv:+.5f} = {mv / se:+.1f} of its own error | largest other move {oth[1]} {100 * oth[0]:.0f}% of it"); print(L[-1], flush=True)
for f in files.values(): os.remove(f)
open(os.path.join(out, "specific_07_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
