"""DEV-METAA-ATLAS-CURVE-01, Met-A on sequencing against the atlas's own sequencing columns: synthetic (written before any real input is read).
For two cells (b cells, cd8 t cells), four planted healthy donors: each donor's level at every atlas site is the cell's array column in
sequencing units moved by a donor term (normal, sd 0.02) and kept at least 0.01, so an unmarked site still carries an occasional marked read as
real sequencing does; 20 reads a site (seed 20261029). Donors 1 to 3 are written as values files and built into sequencing columns by
build_seq_columns.py; every reading is made by the one command (run_sample.py --dev-seq-beta, the sorted cell as specimen) with the columns
in a scratch folder (IAM_SEQ_COLUMNS). Bars, in both cells: (1) donor 4, not an input, reads within 0.05 of 1; (2) donor 1, an input, read
under its own name (its contribution left out) reads within 0.05 of 1; (3) donor 4 with 5 % of its pattern lost (every level moved 5 % toward
0.5) reads above donor 4. Usage: python3 seq_cols_synthetic.py OUT.txt WORKDIR"""
import sys, os, json, subprocess, csv, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, *[".."] * 4, "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_values as SV
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); WD = sys.argv[2]; os.makedirs(WD + "/vals", exist_ok=True)
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
rng = np.random.default_rng(20261029); NR = 20; D = {}; rows = []
for cell in ("b cells", "cd8 t cells"):
    base = SA.seq_units(SA._col("mu", cell).dropna()); base = base[base.index.isin(K.index)]
    for d in range(1, 5):
        p = (base + rng.normal(0, 0.02, len(base))).clip(0.01, 0.99)
        D[(cell, d)] = p; k = rng.binomial(NR, p.to_numpy())
        if d <= 3:
            g = f"SYN_{cell.replace(' ', '_')}_{d}"; rows.append(dict(gsm=g, cell=cell))
            pd.DataFrame({"beta": (k / NR).astype("float32"), "calls": np.full(len(k), NR, "int32")}, index=p.index).to_parquet(f"{WD}/vals/{g}_values.parquet")
        else: D[(cell, "k4")] = k
with open(WD + "/syn_inputs.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=["gsm", "cell"]); w.writeheader(); w.writerows(rows)
subprocess.run([sys.executable, os.path.join(H, "build_seq_columns.py"), WD + "/vals", WD + "/cols", WD + "/syn_inputs.csv"], check=True, capture_output=True)
env = dict(os.environ, IAM_SEQ_COLUMNS=WD + "/cols")
def one_command(cell, beta, name):
    i = K.reindex(beta.index).dropna().astype("int64"); a = np.zeros((SV.HG19_CPGS, 2), np.uint16)
    a[i.to_numpy() - 1, 0] = np.round(beta[i.index].to_numpy() * NR).astype(np.uint16); a[i.to_numpy() - 1, 1] = NR
    f = f"{WD}/{name}.lbeta"; a.tofile(f); b = f"{WD}/{name}.json"
    subprocess.run([sys.executable, RS, "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + cell, "--id", name,
                    "--out", f"{WD}/{name}.html", "--bundle", b], capture_output=True, text=True, env=env, cwd=os.path.dirname(RS)); os.remove(f)
    return ((json.load(open(b)).get("development") or {}).get("metaa_atlas") or {}).get("A") if os.path.exists(b) else None
L = []; ok = []
for cell in ("b cells", "cd8 t cells"):
    c = cell.replace(" ", "_"); p4 = D[(cell, 4)]
    a4 = one_command(cell, pd.Series(D[(cell, "k4")] / NR, index=p4.index), f"d4_{c}")
    v1 = pd.read_parquet(f"{WD}/vals/SYN_{c}_1_values.parquet")["beta"].astype("float64"); a1 = one_command(cell, v1, f"SYN_{c}_1")
    pl = p4 + 0.05 * (0.5 - p4); al = one_command(cell, pd.Series(rng.binomial(NR, pl.to_numpy()) / NR, index=pl.index), f"loss_{c}")
    ok += [a4 is not None and abs(a4 - 1) <= 0.05, a1 is not None and abs(a1 - 1) <= 0.05, None not in (al, a4) and al > a4]
    L.append(f"{cell}: donor 4 (not an input) {a4} | donor 1 (an input, own contribution left out) {a1} | donor 4 with a 5 % loss {al}")
L.append(f"synthetic (all bars, two cells): {'met' if all(ok) else 'not met'} {ok}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
