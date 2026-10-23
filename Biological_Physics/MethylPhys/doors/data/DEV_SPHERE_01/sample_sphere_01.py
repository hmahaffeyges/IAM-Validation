"""DEV-SPHERE-01, sample test 1 (written and committed 2026-10-23 before it runs; plan 35.2, 35.4; development, synthetic, nothing scored).
The information sphere's sample layer (stage_info_sphere.sample_reading) read by the one command, run_sample.py --dev-info-sphere, beside the
address globe, on GSE235717 GSM7508429 (healthy whole blood, spent): healthy; then losses planted by the chain (--dev-plant) in one cell at a time,
spread over every identity site of the cell ('all': a program failing at sites on every chromosome) and clustered in 10 runs of 30 neighboring
sites ('runs': a loss along stretches of DNA), at three sizes. Written before it runs, what each use predicts:
  use 1, distance from its own healthy self: the Fisher-Rao distance rises with the size of the loss, healthy lowest;
  use 2, drift: across the sizes, read as a person drifting, the sphere's territory for the planted cell moves before or with the address globe;
  use 3, a program failing together: an 'all' loss lights the planted cell's family territory and few others; the address globe lights
         comparatively little; a 'runs' loss is the reverse (the address globe's blocks light; the territories need not).
  and the healthy blood: few territories lit (its expected value is 0 lit beyond a 5 % family-wise line, if the rule holds).
Uses 4 (where cell-free DNA came from) and 5 (a lab offset) need a pair reading and constructed arrays the one command does not read yet.
Usage: python3 sample_sphere_01.py OUTDIR [CELL,CELL]   (each machine reads the healthy blood again, so every summary stands alone)"""
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
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--grn", files["grn"], "--red", files["red"], "--specimen", "whole blood", "--id", G, "--out",
                        os.path.join(out, tag + ".html"), "--bundle", b, "--dev-info-sphere"] + extra, capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    B = json.load(open(b)); return B, {x["cell"]: x for x in (B.get("met_a_each_cell") or {}).get("cells", [])}, {x["cell"]: x["share"] for x in B["cell_shares"]["cells"]}
def read(B):
    S = B.get("info_sphere") or {}; Gl = B.get("globe") or {}
    lit = [f"{g['family']} {'meth' if g['methylated_there'] else 'unmeth'} {g['z']:+.1f}" for g in S.get("territories", []) if g.get("changed")]
    return dict(fr=S.get("fisher_rao_rms_rad"), lit=S.get("territories_changed"), of=S.get("territories_read"), which=lit, reason=S.get("reason"),
                genome=Gl.get("blocks_changed"), nblocks=Gl.get("blocks"), metaa=(Gl.get("metaa_blocks_changed") or {}).get("changed"))
def line(tag, r): return (f"{tag:<34} | sphere: {r['lit']} of {r['of']} territories lit, distance {r['fr']} rad | address globe: {r['genome']} of {r['nblocks']} blocks; "
                          f"Met-A sites {r['metaa']} blocks" + (f" | lit: {'; '.join(r['which'])}" if r["which"] else "") + (f" | reason {r['reason']}" if r["reason"] else ""))
B0 = one("healthy", [])[0]; R0 = read(B0); L = ["DEV-SPHERE-01 sample test 1: the sample on the information sphere beside the address globe (losses planted by run_sample.py)", line("healthy", R0)]
json.dump(B0.get("info_sphere"), open(os.path.join(out, "healthy_info_sphere.json"), "w"), indent=1)
CELLS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cd4 t cells", "neutrophils", "monocytes", "b cells"]   # one machine per cell when given
for c in CELLS:
    for kind in ("all", "runs"):
        for loss in (0.05, 0.2, 0.5):
            tag = f"{c.replace(' ', '_')}_{kind}_{int(loss * 100)}"; B1 = one(tag, ["--dev-plant", f"{c}:{loss}:{kind}"])[0]; L.append(line(f"{c} {kind} {loss:.0%}", read(B1)))
            json.dump(B1.get("info_sphere"), open(os.path.join(out, tag + "_info_sphere.json"), "w"), indent=1); print(L[-1], flush=True)
for f in files.values(): os.remove(f)
open(os.path.join(out, "sample_sphere_01_" + "_".join(c.replace(" ", "-") for c in CELLS) + "_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
