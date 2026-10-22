"""DEV-EACH-CELL-01, exploration 3 (2026-10-22; not sealed). Author: "whole blood and plasma both get filtered and whatever it finds it scores,
we dont assume anything until the deconvolver tells us what is really there." Every step is the one command, chain/MethylPhys_Interface/
run_sample.py, from the raw IDAT pair of the healthy example blood (GSE235717 GSM7508429, spent), with --cell-panel atlas (all of
IAM-Atlas v2.1's cell types; development), once as it is and once with --dev-plant monocytes:0.05. This file only downloads the IDATs and reads
the bundles. Reported: every cell the Deconvolver finds above a share of 0.001 with its 95 % interval; the six blood cells' shares on the full
panel against the six-cell panel; each found cell's Met-A (untared) and its predicted shift for a 1 % loss; the planted cell's move.
Usage: python3 full_panel_03.py OUTDIR"""
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); G = "GSM7508429"
base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{G[:7]}nnn/{G}/suppl/"; files = {}
for k in range(5):
    try:
        for n in sorted(set(re.findall(rf"{G}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(base, timeout=120).read().decode()))):
            f = os.path.join(out, n); urllib.request.urlretrieve(base + n, f); files["grn" if "_Grn" in n else "red"] = f
        break
    except Exception as e: print("download retry", type(e).__name__, flush=True); time.sleep(60 * (k + 1))
assert set(files) == {"grn", "red"}, files
def one(tag, extra):
    b = os.path.join(out, tag + "_bundle.json"); t0 = time.time()
    r = subprocess.run([sys.executable, RS, "--grn", files["grn"], "--red", files["red"], "--specimen", "whole blood", "--id", G, "--out",
                        os.path.join(out, tag + ".html"), "--bundle", b] + extra, capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    print(tag, f"{time.time() - t0:.0f} s", flush=True); return json.load(open(b))
six = one("six", []); full = one("full", ["--cell-panel", "atlas"]); pl = one("full_planted", ["--cell-panel", "atlas", "--dev-plant", "monocytes:0.05"])
S6 = {x["cell"]: x["share"] for x in six["cell_shares"]["cells"]}; F = full["cell_shares"]; SF = {x["cell"]: x for x in F["cells"]}
E = {x["cell"]: x for x in (full.get("met_a_each_cell") or {}).get("cells", [])}; EP = {x["cell"]: x for x in (pl.get("met_a_each_cell") or {}).get("cells", [])}
L = [f"DEV-EACH-CELL-01 exploration 3: GSM7508429 by run_sample.py --cell-panel atlas ({len(SF)} cell types, {F.get('markers')} marker sites, "
     f"{F.get('markers_measured')} measured; raw sum {F.get('raw_sum')})", "cells found above a share of 0.001 (share, 95 % interval, Met-A untared, shift per 1 % loss):"]
for c, x in sorted(SF.items(), key=lambda kv: -kv[1]["share"]):
    if x["share"] < 0.001: continue
    e = E.get(c, {}); L.append(f"  {c}: {x['share']} [{x['interval_95'][0]}, {x['interval_95'][1]}] | Met-A {e.get('A')} | shift {e.get('shift_per_1pct_loss')}"
                               + (f" | {e.get('reason')}" if e.get("A") is None else ""))
L.append(f"cells below 0.001: {sum(1 for x in SF.values() if x['share'] < 0.001)}")
L.append("six-cell panel shares: " + ", ".join(f"{c} {v}" for c, v in S6.items()))
m0, m1 = (E.get("monocytes") or {}).get("A"), (EP.get("monocytes") or {}).get("A")
if m0 is not None and m1 is not None:
    oth = max((abs(EP[c]["A"] - E[c]["A"]), c) for c in E if c != "monocytes" and E[c].get("A") is not None and EP.get(c, {}).get("A") is not None)
    L.append(f"planted 5 % monocyte loss (chain record {pl.get('planted')}): monocytes moved {m1 - m0:+.5f}, predicted {5 * E['monocytes']['shift_per_1pct_loss']:+.5f}; "
             f"largest other move {oth[1]} {oth[0]:.5f}")
else: L.append("planted: monocytes not read on the full panel: " + str((E.get("monocytes") or {}).get("reason")))
for f in files.values(): os.remove(f)
open(os.path.join(out, "full_panel_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
