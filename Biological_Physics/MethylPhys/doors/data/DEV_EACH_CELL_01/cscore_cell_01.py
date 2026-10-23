"""DEV-EACH-CELL-01, synthetic C-Score test 1 (written and committed 2026-10-23 before it runs; plan 41.2, 41.3; not sealed). The C-Score for EVERY
cell the IAM-Deconvolver finds (flagged below 1 %, stage_metaa_atlas.CS_FLAG_SHARE), and for the whole genome (stage_metaa_atlas.cscore_in_mix),
read by the one command, run_sample.py --cell-panel atlas (all 79 atlas cells), from one array's raw IDATs. Spent arrays only: GSE235717 GSM7508429
(healthy whole blood) and three of GSE122126's known mixtures of tissue DNA in blood (GSM3455853 hepatocytes 10 % and lung 3.5 %; GSM3455856 colon
10 % and neurons 5 %; GSM3455858 lung 10 % and neurons 3.5 %), one array per run. The array is read clean; then, for every cell the IAM-Deconvolver found (any share above 0), the
chain plants a loss in that cell (--dev-plant CELL:LOSS:SCOPE, scaled by its share): clustered, 50 % in 10 runs of 30 neighboring identity sites
('runs'); and spread, 20 % at every site the cell holds ('all'). Expected from the statistic, written before it runs: a clustered loss raises the
planted cell's C above its clean C, and more than a spread loss does; other cells' C move less. Nothing scored; the instrument shows how a loss
reads, cell by cell.  Usage: python3 cscore_cell_01.py GSM OUTDIR   (writes OUTDIR/cscore_cell_01_GSM_summary.txt)"""
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
G, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); assert G in ("GSM7508429", "GSM3455853", "GSM3455856", "GSM3455858"), G
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
    r = subprocess.run([sys.executable, RS, "--grn", files["grn"], "--red", files["red"], "--specimen", "whole blood", "--id", G, "--cell-panel", "atlas",
                        "--out", os.path.join(out, tag + ".html"), "--bundle", b] + extra, capture_output=True, text=True)
    if r.returncode: print(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-400:]} {r.stderr[-400:]}", flush=True); return None
    c = json.load(open(b)).get("cscore_each_cell") or {}
    return {x["cell"]: x.get("C") for x in c.get("cells", [])}, {x["cell"]: x.get("share") for x in c.get("cells", [])}, (c.get("whole_genome") or {}).get("C"), c.get("reason")
C0, S0, W0, why = one("clean", [])
cells = [c for c, v in C0.items() if v is not None]
L = [f"DEV-EACH-CELL-01 synthetic C-Score test 1, {G}: every cell found planted by run_sample.py --cell-panel atlas; C untared",
     f"clean: whole genome {W0} | " + ", ".join(f"{c} ({S0[c]}) {C0[c]}" for c in cells) + (f" | {why}" if why else "")]
for c in cells:
    for p in ("0.5:runs", "0.2:all"):
        r = one(f"{c.replace(' ', '_')}_{p.replace(':', '_')}", ["--dev-plant", f"{c}:{p}"])
        if r is None: L.append(f"planted {c:<22} {p:<8}: run failed"); continue
        C1, _, W1, _ = r; oth = max((((C1.get(q) or 0) - (C0.get(q) or 0)), q) for q in cells if q != c) if len(cells) > 1 else (0.0, "-")
        L.append(f"planted {c:<22} {p:<8}: {c} C {C0[c]} -> {C1.get(c)} | whole genome {W0} -> {W1} | largest rise elsewhere {oth[1]} {oth[0]:+.3f}")
        print(L[-1], flush=True)
for f in files.values(): os.remove(f)
open(os.path.join(out, f"cscore_cell_01_{G}_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
