"""DEV-ONE-DRAW-01, exploration 1 (2026-10-22; not sealed). To-do step 40: how few molecules each instrument needs, told by the instruments.
Every step is the one command, chain/MethylPhys_Interface/run_sample.py. A deep healthy CD8 T-cell WGBS sample (GSM5652282, spent: it built
IAM-A's line) is fetched by the chain (--geo-pat), then read at full depth and at seeded thinned depths (--dev-thin), each time with IAM-A and
with sequencing values at the atlas's sites (--dev-seq-values 1) for Met-A against IAM-Atlas v2.1's CD8 T cells and the cell shares. This file
only runs the one command and reads the bundles.
Reported at each depth: molecules kept; IAM-A (A, state); sites with a value and median calls per site; Met-A against the atlas (A untared, sites
read, or why not); the CD8 T share and the largest other share. Usage: python3 seq_depth_01.py OUTDIR"""
import os, sys, json, glob, subprocess, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); G = "GSM5652282"; GD = os.path.join(out, "geo"); FR = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
def one(tag, src):
    b = os.path.join(out, tag + "_bundle.json"); t0 = time.time()
    r = subprocess.run([sys.executable, RS] + src + ["--specimen", "isolated cd8 t cells", "--seq-cell", "cd8 t cells", "--id", G, "--dev-seq-values", "1",
                        "--out", os.path.join(out, tag + ".html"), "--bundle", b], capture_output=True, text=True)
    print(tag, "exit", r.returncode, f"{time.time() - t0:.0f} s", flush=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:], flush=True); return None
    return json.load(open(b))
L = [f"DEV-ONE-DRAW-01 exploration 1: {G} (healthy CD8 T cells, WGBS, spent) read by run_sample.py at seeded thinned depths"]
pat = None
for f in FR:
    if f == 1.0: B = one("depth_1", ["--geo-pat", G, "--geo-dir", GD]); pat = (glob.glob(os.path.join(GD, "*.pat.gz")) or [None])[0]
    else: B = one(f"depth_{f:g}", ["--pat", pat, "--dev-thin", str(f)]) if pat else None
    if B is None: L.append(f"fraction {f:g}: the one command failed (log above)"); continue
    q = B.get("iam_a") or {}; v = B.get("sequencing_values") or {}; m = (B.get("development") or {}).get("metaa_atlas") or {}
    sh = sorted(((x["share"], x["cell"]) for x in (B.get("cell_shares") or {}).get("cells", [])), reverse=True)
    L.append(f"fraction {f:g}: molecules {(B.get('thinned') or {}).get('molecules_kept', 'all')} | IAM-A {q.get('A')} {str(q.get('state', q.get('refusal', '')))[:60]} | "
             f"sites with a value {v.get('sites_with_value')}, median calls {v.get('calls_per_site_median')} | Met-A vs atlas CD8 {m.get('A')} "
             f"({m.get('n_sites', '')} sites{'; ' + str(m.get('reason')) if m.get('A') is None else ''}) | shares {', '.join(f'{c} {s}' for s, c in sh[:3])}")
for f in glob.glob(os.path.join(GD, "*")): os.remove(f)
open(os.path.join(out, "seq_depth_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
