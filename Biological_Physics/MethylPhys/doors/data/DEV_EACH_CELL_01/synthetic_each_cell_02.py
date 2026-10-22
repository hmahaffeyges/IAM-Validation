"""DEV-EACH-CELL-01, synthetic test 2 (2026-10-22; exploration, not sealed). Test 1 planted the loss in this script; author: "it MUST use the chain
or it didnt happen". Here every step is the one command, chain/MethylPhys_Interface/run_sample.py, from the raw IDAT pair: once as it is, and
once per planted cell with --dev-plant CELL:0.05 (the chain plants the constructed loss itself after Stage 1, scaled by the cell's share).
This file only downloads the public IDATs (GSE235717 GSM7508429, spent) and reads the bundles.
The instrument's job: each cell's Met-A reading moves for the cell whose pattern was lost, by the shift it predicts at that share.
Usage: python3 synthetic_each_cell_02.py OUTDIR"""
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
B0, E0, S0 = one("healthy", [])
L = ["DEV-EACH-CELL-01 synthetic test 2: every reading and every plant by run_sample.py from the raw IDATs (GSM7508429); a 5 % loss of one cell's pattern at its share",
     "healthy, untared: " + ", ".join(f"{c} A {E0[c]['A']} (share {S0[c]})" for c in E0 if E0[c].get("A") is not None)]
for c in PLANT:
    B1, E1, S1 = one("planted_" + c.replace(" ", "_"), ["--dev-plant", f"{c}:{LOSS}"]); p = B1.get("planted") or {}
    mv = {k: round(E1[k]["A"] - E0[k]["A"], 5) for k in E0 if E0[k].get("A") is not None and E1.get(k, {}).get("A") is not None}
    pred = round(LOSS / 0.01 * E0[c]["shift_per_1pct_loss"], 5); o = max((abs(v), k) for k, v in mv.items() if k != c)
    L.append(f"planted {c} (chain record: share {p.get('share')}, {p.get('sites_moved')} sites): moved {mv[c]:+.5f}, predicted {pred:+.5f} ({mv[c] / pred:.0%}) | "
             f"largest other move {o[1]} {o[0]:.5f} ({o[0] / abs(mv[c]):.0%} of the planted cell's) | share {S0[c]} -> {S1[c]}")
    L.append("   every cell: " + ", ".join(f"{k} {v:+.5f}" for k, v in mv.items()))
for f in files.values(): os.remove(f)
open(os.path.join(out, "synthetic_each_cell_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
