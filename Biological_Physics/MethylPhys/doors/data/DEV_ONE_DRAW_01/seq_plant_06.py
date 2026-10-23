"""DEV-ONE-DRAW-01, exploration 6 (written and committed 2026-10-22 before it runs; exploration, not sealed; to-do 40.11). What a loss still
moves Met-A on sequencing at each depth. Every step is the chain: run_sample.py calibrates the spent example blood (GSE235717 GSM7508429) and
saves its values; run_sample.py --build constructed-draw --dev-plant CELL:LOSS plants a loss of one cell's pattern in those values BEFORE the
reads are counted, then counts them at the depth given; run_sample.py --dev-seq-beta reads the buffy-coat half. Draws: healthy; CD4 T cells
5 % and 20 %; NK cells 20 %. Reported: each planted cell's Met-A healthy and planted, the move, and the move in the reading's own error over
sites. Nothing scored: the instrument tells what a loss moves at each depth. Usage: python3 seq_plant_06.py OUTDIR DEPTH"""
# 2026-10-22: OUTDIR is made absolute, because the chain's builds run from their own folder
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = os.path.abspath(sys.argv[1]); DEPTH = int(sys.argv[2]); os.makedirs(out, exist_ok=True); G = "GSM7508429"
CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface"))
base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{G[:7]}nnn/{G}/suppl/"; files = {}
for k in range(5):
    try:
        for n in sorted(set(re.findall(rf"{G}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(base, timeout=120).read().decode()))):
            f = os.path.join(out, n); urllib.request.urlretrieve(base + n, f); files["grn" if "_Grn" in n else "red"] = f
        break
    except Exception as e: print("download retry", type(e).__name__, flush=True); time.sleep(60 * (k + 1))
assert set(files) == {"grn", "red"}, files
bv = os.path.join(out, "blood_values.csv")
r = subprocess.run([sys.executable, RS, "--grn", files["grn"], "--red", files["red"], "--specimen", "whole blood", "--id", G, "--out", os.path.join(out, "blood.html"),
                    "--bundle", os.path.join(out, "blood.json"), "--save-betas", bv], capture_output=True, text=True)
if r.returncode or not os.path.exists(bv): sys.exit(f"calibration: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
def read(tag, plant):
    d = os.path.join(out, tag); cmd = [sys.executable, RS, "--build", "constructed-draw", "--build-input", bv, "--build-mix", "", "--build-depth", f"{DEPTH},{DEPTH}", "--build-out", d]
    r = subprocess.run(cmd + (["--dev-plant", plant] if plant else []), capture_output=True, text=True, cwd=CH)
    if r.returncode: sys.exit(f"{tag}: build exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}")
    b = os.path.join(d, "b.json"); r = subprocess.run([sys.executable, RS, "--dev-seq-beta", os.path.join(d, "buffy.beta.gz"), "--specimen", "whole blood", "--id", tag,
                                                      "--out", os.path.join(d, "r.html"), "--bundle", b], capture_output=True, text=True, cwd=CH)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}")
    for f in ("buffy.beta.gz", "plasma.beta.gz"): os.remove(os.path.join(d, f))
    return {x["cell"]: x for x in (json.load(open(b)).get("met_a_each_cell") or {}).get("cells", [])}
E0 = read("healthy", None); L = [f"DEV-ONE-DRAW-01 exploration 6: losses planted before counting, read at {DEPTH} reads per site; all by the chain (constructed from {G}, spent)",
                                 "healthy: " + ", ".join(f"{c} {x.get('A')} (se {x.get('se_over_sites')})" for c, x in E0.items())]
for c, loss in (("cd4 t cells", 0.05), ("cd4 t cells", 0.2), ("nk cells", 0.2)):
    E1 = read(f"plant_{c.replace(' ', '_')}_{int(loss * 100)}", f"{c}:{loss}"); a0, a1 = E0[c].get("A"), E1[c].get("A")
    if a0 is None or a1 is None: L.append(f"{c} {loss:.0%}: not read ({E1[c].get('reason')})"); continue
    se = E0[c].get("se_over_sites") or float("nan"); oth = max(((E1[q]["A"] - E0[q]["A"]) / (a1 - a0) if a1 != a0 else 0, q) for q in E0 if q != c and E1.get(q, {}).get("A") is not None)
    L.append(f"{c} {loss:.0%}: {a0} -> {a1}, moved {a1 - a0:+.5f} = {(a1 - a0) / se:+.1f} of its own error | largest other move {oth[1]} {100 * oth[0]:.0f}% of it"); print(L[-1], flush=True)
for f in files.values(): os.remove(f)
open(os.path.join(out, f"seq_plant_06_depth{DEPTH}_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
