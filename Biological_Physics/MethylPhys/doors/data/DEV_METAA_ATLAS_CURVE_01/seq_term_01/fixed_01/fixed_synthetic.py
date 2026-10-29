"""DEV-METAA-ATLAS-CURVE-01, Met-A on sequencing at a fixed depth: synthetic (written and committed before any real reading at a fixed depth).
One planted healthy b-cell donor: its level at every atlas site is the atlas's b-cell column in sequencing units moved by a donor term (normal,
sd 0.02), kept within 0.01 to 0.99 (seed 20261029). The same donor is read by the one command (run_sample.py --dev-seq-beta, specimen sorted b
cells) at three depths, reads per site Poisson with mean 12, 20 and 40; and once with 5 % of its pattern lost (every level moved 5 % toward 0.5)
at mean 20. The instrument's job: the same donor reads the same whatever its depth. Bars, at k = 6 reads per identity site: (1) the three
depths read within 0.02 of each other; (2) the healthy donor reads within 0.05 of 1; (3) the 5 % loss reads above the healthy donor at the same
depth. Also reported, not scored: the same at every k of FIXED_K and the reading without a fixed depth. Usage: python3 fixed_synthetic.py OUT.txt WORKDIR"""
import sys, os, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, *[".."] * 5, "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_values as SV
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); WD = sys.argv[2]; os.makedirs(WD, exist_ok=True)
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
rng = np.random.default_rng(20261029); base = SA.seq_units(SA._col("mu", "b cells").dropna()); base = base[base.index.isin(K.index)]
p = (base + rng.normal(0, 0.02, len(base))).clip(0.01, 0.99)
def one_command(level, depth, name):
    n = np.maximum(rng.poisson(depth, len(level)), 0); k = rng.binomial(n, level.to_numpy())
    i = K.reindex(level.index).astype("int64").to_numpy() - 1; a = np.zeros((SV.HG19_CPGS, 2), np.uint16); a[i, 0] = k; a[i, 1] = n
    f = f"{WD}/{name}.lbeta"; a.tofile(f); b = f"{WD}/{name}.json"
    subprocess.run([sys.executable, RS, "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", name,
                    "--out", f"{WD}/{name}.html", "--bundle", b], capture_output=True, text=True, cwd=os.path.dirname(RS)); os.remove(f)
    return ((json.load(open(b)).get("development") or {}).get("metaa_atlas") or {}) if os.path.exists(b) else {}
R = {f"depth {d}": one_command(p, d, f"healthy_{d}") for d in (12, 20, 40)}
R["5 % loss, depth 20"] = one_command(p + 0.05 * (0.5 - p), 20, "loss_20")
fk = lambda r, k: ((r.get("fixed_depth") or {}).get(str(k)) or {}).get("A")
L = ["reading | without a fixed depth | " + " | ".join(f"k {k}" for k in SA.FIXED_K)]
for nm, r in R.items(): L.append(f"{nm} | {r.get('A')} | " + " | ".join(str(fk(r, k)) for k in SA.FIXED_K))
h = [fk(R[f"depth {d}"], 6) for d in (12, 20, 40)]; ls = fk(R["5 % loss, depth 20"], 6)
ok = [None not in h and max(h) - min(h) <= 0.02, None not in h and all(abs(x - 1) <= 0.05 for x in h), None not in (ls, h[1]) and ls > h[1]]
L.append(f"bars at k 6: depths within 0.02 {ok[0]} | healthy within 0.05 of 1 {ok[1]} | loss above healthy {ok[2]} -> {'met' if all(ok) else 'not met'}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
