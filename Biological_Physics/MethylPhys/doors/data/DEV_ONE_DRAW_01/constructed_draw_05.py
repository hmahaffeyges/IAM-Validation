"""DEV-ONE-DRAW-01, exploration 5 (written and committed 2026-10-22 before it runs; exploration, not sealed). The one-draw path on CONSTRUCTED
draws with known answers (the open matched set, GSE186888, stays unread for the test that counts; author: "If you can save them do it").
Every step is the chain: run_sample.py calibrates the spent example blood (GSE235717 GSM7508429) from its raw IDATs and saves its values;
run_sample.py --build constructed-draw makes each draw (buffy coat: the blood; plasma: the blood plus hepatocyte, vascular endothelium and colon
epithelium, each at share S, read as k of n reads per site); run_draw.py reads both halves on IAM-Atlas v2.1's full panel. S = 0 (the control),
0.5, 1, 2, 5 %; buffy coat at 30 reads per site, plasma at the depth given. Reported per draw: each added tissue's share found against the
truth, and the largest share found for any tissue that was NOT added (the noise floor at that depth). Nothing scored: the instrument tells the
smallest share it reads at each depth. Usage: python3 constructed_draw_05.py OUTDIR PLASMA_DEPTH"""
# 2026-10-22: OUTDIR is made absolute, because the chain's builds run from their own folder
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = os.path.abspath(sys.argv[1]); DEPTH = int(sys.argv[2]); os.makedirs(out, exist_ok=True); G = "GSM7508429"
CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface")); RD = os.path.join(CH, "run_draw.py")
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
ADD = ["hepatocyte", "vascular endothelium", "colon epithelium"]; L = [f"DEV-ONE-DRAW-01 exploration 5: constructed draws from {G} (spent), plasma at {DEPTH} reads per site, buffy coat at 30; all by the chain"]
for s in (0.0, 0.005, 0.01, 0.02, 0.05):
    d = os.path.join(out, f"draw_{int(s * 1000):03d}"); mix = ",".join(f"{t}:{s}" for t in ADD)
    r = subprocess.run([sys.executable, RS, "--build", "constructed-draw", "--build-input", bv, "--build-mix", mix, "--build-depth", f"30,{DEPTH}", "--build-out", d], capture_output=True, text=True, cwd=CH)
    if r.returncode: L.append(f"share {s:.1%}: build exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}"); continue
    o = os.path.join(d, "read"); r = subprocess.run([sys.executable, RD, "--id", f"CONSTRUCTED_{int(s * 1000):03d}", "--out", o, "--buffy-seqbeta", os.path.join(d, "buffy.beta.gz"),
                                                   "--plasma-seqbeta", os.path.join(d, "plasma.beta.gz")], capture_output=True, text=True, cwd=CH)
    if r.returncode: L.append(f"share {s:.1%}: run_draw exit {r.returncode}: {(r.stdout + r.stderr)[-400:]}"); continue
    U = json.load(open(os.path.join(o, "draw.json")))["plasma_not_explained_by_own_blood_cells"]
    found = {t: (U.get(t) or {}).get("plasma_share", 0.0) for t in ADD}; other = max(((v.get("plasma_share", 0.0), c) for c, v in U.items() if c not in ADD), default=(0.0, "none"))
    L.append(f"share {s:.1%}: found " + ", ".join(f"{t} {v:.4f}" for t, v in found.items()) + f" | largest not added: {other[1]} {other[0]:.4f}"); print(L[-1], flush=True)
    for f in ("buffy.beta.gz", "plasma.beta.gz"): os.remove(os.path.join(d, f))
for f in files.values(): os.remove(f)
open(os.path.join(out, f"constructed_draw_05_depth{DEPTH}_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
