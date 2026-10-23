"""DEV-EACH-CELL-01, synthetic C-Score test 1 (written and committed 2026-10-23 before it runs; plan 41.2, 41.3; not sealed). The C-Score for each
cell and for the whole genome (stage_metaa_atlas.cscore_in_mix), read by the one command, run_sample.py, from the raw IDATs of GSE235717 GSM7508429
(healthy whole blood, spent): healthy; then losses planted by the chain (--dev-plant) in one cell at a time, clustered (10 runs of 30 neighboring
identity sites, 'runs') and spread over every identity site of the cell ('all'). Expected from the statistic, written before it runs: a clustered
loss raises the planted cell's C well above its healthy C, and more than a spread loss does; other cells' C move less than the planted cell's.
Nothing scored; the instrument shows how a clustered loss reads.  Usage: python3 cscore_cell_01.py OUTDIR"""
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
def cs(B):
    c = B.get("cscore_each_cell") or {}
    return {x["cell"]: x.get("C") for x in c.get("cells", [])}, (c.get("whole_genome") or {}).get("C"), c.get("reason")
B0, _, S0 = one("healthy", []); C0, W0, why = cs(B0)
L = ["DEV-EACH-CELL-01 synthetic C-Score test 1: losses planted by run_sample.py, C-Score for each cell and the whole genome (untared)",
     "healthy: whole genome " + str(W0) + " | " + ", ".join(f"{c} (share {S0.get(c)}) {v}" for c, v in C0.items() if v is not None) + (f" | reason {why}" if why else "")]
for c, plants in (("cd4 t cells", ("0.2:runs", "0.5:runs", "0.05:all", "0.2:all")), ("neutrophils", ("0.2:runs", "0.05:all")), ("nk cells", ("0.5:runs",)), ("monocytes", ("0.5:runs",))):
    for p in plants:
        C1, W1, _ = cs(one(f"cs_{c.replace(' ', '_')}_{p.replace(':', '_')}", ["--dev-plant", f"{c}:{p}"])[0])
        oth = max(((C1.get(q) or 0) - (C0.get(q) or 0), q) for q in C0 if q != c and C0.get(q) is not None)
        L.append(f"planted {c:<12} {p:<9}: {c} C {C0.get(c)} -> {C1.get(c)} | whole genome {W0} -> {W1} | largest rise elsewhere {oth[1]} {oth[0]:+.3f}")
for f in files.values(): os.remove(f)
open(os.path.join(out, "cscore_cell_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
