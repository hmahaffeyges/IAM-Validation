"""DEV-EACH-CELL-01, exploration 5 (written and committed 2026-10-22 before it runs; exploration, not sealed; to-do 38.1). The globe's change
layer for each cell: does a CLUSTERED loss in one cell light that cell's layer? Every step is the one command, run_sample.py, from the raw IDATs of
GSE235717 GSM7508429 (spent): healthy, then for each of four cells a loss planted by the chain in 10 runs of 30 neighboring identity sites
(--dev-plant CELL:LOSS:runs), at three sizes (5, 20, 50 % of the way to 0.5, scaled by the cell's share). Read from each bundle's globe record
(cell_blocks_changed): the planted cell's changed blocks against healthy, and the largest increase in another cell. Nothing scored; the instrument
tells how large a clustered loss in a cell of a given share must be to show. Usage: python3 globe_cell_05.py OUTDIR"""
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); G = "GSM7508429"; LOSS = 0.05; PLANT = ["monocytes", "cd4 t cells", "b cells", "nk cells"]
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
def blocks(B): return {c: v["changed"] for c, v in ((B.get("globe") or {}).get("cell_blocks_changed") or {}).items()}
B0 = one("healthy", [])[0]; K0 = blocks(B0); S0 = {x["cell"]: x["share"] for x in B0["cell_shares"]["cells"]}; NB = {c: v["blocks"] for c, v in B0["globe"]["cell_blocks_changed"].items()}
L = ["DEV-EACH-CELL-01 exploration 5: clustered losses (10 runs of 30 identity sites) planted by run_sample.py; each cell's globe layer, changed blocks",
     "healthy: " + ", ".join(f"{c} {K0[c]} of {NB[c]}" for c in K0)]
for c in PLANT:
    for loss in (0.05, 0.2, 0.5):
        K1 = blocks(one(f"runs_{c.replace(' ', '_')}_{int(loss * 100)}", ["--dev-plant", f"{c}:{loss}:runs"])[0])
        oth = max(((K1.get(q, 0) - K0[q]), q) for q in K0 if q != c)
        L.append(f"planted {c:<12} (share {S0[c]}) loss {loss:.0%}: {c} {K0[c]} -> {K1.get(c)} of {NB[c]} blocks | largest rise elsewhere {oth[1]} {oth[0]:+d}")
for f in files.values(): os.remove(f)
open(os.path.join(out, "globe_cell_05_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
