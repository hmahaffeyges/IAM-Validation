"""DEV-EACH-CELL-01, exploration 4 (written and committed 2026-10-22 before it runs; exploration, not sealed). Author: "Met-A should be able to
read 800K sites so long as they correspond to the cells in the atlas." Two ways of reading Met-A for each cell, both by the one command
(run_sample.py --dev-metaa-allsites records both in one bundle): (I) the cell's up to 6,000 identity sites (read_in_mix); (W) every atlas site,
each weighted by the shift a loss of that cell would make, share x |0.5 - mu| (read_in_mix_allsites). Two constructed plants, both by the one
command: a 5 % loss at the cell's identity sites only (--dev-plant CELL:0.05) and at every atlas site the cell has a value (CELL:0.05:all; a real
loss of hold reaches every held site). For each reading, plant and planted cell: the move in the reading's own standard error over sites (how far
it stands out) and the largest move of another cell as a share of the planted cell's (spillover). Nothing is scored; the instruments tell which
reading separates a loss more clearly. Raw IDATs of GSE235717 GSM7508429 (spent). Usage: python3 metaa_allsites_04.py OUTDIR"""
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
def rec(B, k): return {x["cell"]: x for x in (B.get(k) or {}).get("cells", []) if x.get("A") is not None}
B0 = one("healthy", ["--dev-metaa-allsites"])[0]; R0 = {"I": rec(B0, "met_a_each_cell"), "W": rec(B0, "met_a_each_cell_allsites")}
L = ["DEV-EACH-CELL-01 exploration 4: Met-A for each cell, identity sites (I) against every atlas site weighted by its shift (W); GSM7508429 by run_sample.py"]
for v in ("I", "W"):
    L.append(f"healthy {v}: " + ", ".join(f"{c} A {x['A']} se {x['se_over_sites']} sites {x.get('effective_sites', x.get('n_sites'))}" for c, x in R0[v].items()))
for scope in ("identity", "all"):
    for c in PLANT:
        B1 = one(f"planted_{scope}_" + c.replace(" ", "_"), ["--dev-metaa-allsites", "--dev-plant", f"{c}:{LOSS}" + (":all" if scope == "all" else "")])[0]
        for v, k in (("I", "met_a_each_cell"), ("W", "met_a_each_cell_allsites")):
            R1 = rec(B1, k); mv = {q: R1[q]["A"] - R0[v][q]["A"] for q in R0[v] if q in R1}
            if c not in mv: L.append(f"plant {scope} {c} | {v}: not read"); continue
            z = mv[c] / R0[v][c]["se_over_sites"]; sp = max(abs(m) for q, m in mv.items() if q != c) / abs(mv[c]) if mv[c] else float("nan")
            L.append(f"plant {scope:<8} {c:<12} | {v}: moved {mv[c]:+.5f} = {z:+.1f} of its own error | largest other move {sp:.0%} of it")
for f in files.values(): os.remove(f)
open(os.path.join(out, "metaa_allsites_04_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
