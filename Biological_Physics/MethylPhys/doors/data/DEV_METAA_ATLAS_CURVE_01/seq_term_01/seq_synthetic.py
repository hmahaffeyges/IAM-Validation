"""DEV-METAA-ATLAS-CURVE-01, Met-A on sequencing with the atlas's source term: synthetic (written before any sequencing sample is read with it).
For three atlas columns, a planted healthy sequencing sample: at each identity site, n reads (n from 4 to 30, seed 20261028) drawn from the
column's level in sequencing units (sequencing_term_v1); read by stage_metaa_atlas.read with its counts. Bars: (1) the healthy plant reads
within 0.03 of 1 in every column; (2) the same plant read against the column on the array scale (the previous rule) reads below 0.9, the
gap the 17 sequencing inputs showed; (3) a plant that lost 5 % of its pattern (every level moved 5 % toward 0.5) reads above its healthy
plant in every column.
Run 2 (amended after run 1, disclosed): run 1 read one plant per column and missed bar 1 in b cells (1.0339); over 30 plants the column
reads mean 1.0004, sd 0.0209, so a single plant carries counting scatter of about 0.02. Bar 1 now tests bias, as intended: the mean of 30
plants within 0.01 of 1 in every column; bars 2 and 3 unchanged (read on the first plant). Run 3 (added at the push gate, rule 23; bars 1 to 3 unchanged): bar 4, the first plant of each column written as a sequencing values file
over the whole hg19 numbering (sites_sequencing_key) and read by the one command (run_sample.py --dev-seq-beta, the sorted cell as
specimen) reads what the stage reads for it within 0.001, and within 0.06 of 1 (three times one plant's counting scatter).
Usage: python3 seq_synthetic.py OUT.txt [WORKDIR]"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "chain"))
import stage_metaa_atlas as SA, stage_seq_values as SV, subprocess, json, tempfile
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "chain"); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
WD = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(); os.makedirs(WD, exist_ok=True)
def one_command(cell, x, n):
    """write the plant as k of n reads over the whole hg19 numbering and read it with the one command"""
    i = K.reindex(x.index).dropna().astype("int64"); a = np.zeros((SV.HG19_CPGS, 2), np.uint16)
    a[i.to_numpy() - 1, 0] = np.round(x[i.index].to_numpy() * n[i.index].to_numpy()).astype(np.uint16); a[i.to_numpy() - 1, 1] = n[i.index].to_numpy().astype(np.uint16)
    f = os.path.join(WD, "plant.lbeta"); a.tofile(f); b = os.path.join(WD, "plant.json")
    subprocess.run([sys.executable, RS, "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + cell, "--id", "plant",
                    "--out", os.path.join(WD, "plant.html"), "--bundle", b], capture_output=True, text=True); os.remove(f)
    return ((json.load(open(b)).get("development") or {}).get("metaa_atlas") or {}).get("A") if os.path.exists(b) else None
rng = np.random.default_rng(20261028); ok = []
L = [f"sequencing term (sequencing_term_v1): a {SA.SEQ_TERM['a']:.4f}, b {SA.SEQ_TERM['b']:.4f}"]
for cell in ("b cells", "naive cd4 t cells", "cd8 t cells"):
    S = SA.identity_sites(cell); mu = SA._col("mu", cell).reindex(S).dropna()
    ps = (mu * SA.SEQ_TERM["b"] + SA.SEQ_TERM["a"]).clip(0, 1); n = pd.Series(rng.integers(4, 31, len(ps)), index=ps.index)
    def plant(p): return pd.Series(rng.binomial(n.to_numpy(), p.to_numpy()) / n.to_numpy(), index=p.index)
    x1 = plant(ps); H30 = [SA.read(x1, cell, "sorted " + cell, counts=n)["A"]] + [SA.read(plant(ps), cell, "sorted " + cell, counts=n)["A"] for _ in range(29)]; h = H30[0]   # run 2: 30 plants
    oc = one_command(cell, x1, n)   # run 3: the first plant through the one command
    t0 = dict(SA.SEQ_TERM); SA.SEQ_TERM.update(a=0.0, b=1.0); old = SA.read(plant(ps), cell, "sorted " + cell, counts=n)["A"]; SA.SEQ_TERM.update(t0)
    lost = SA.read(plant(ps + 0.05 * (0.5 - ps)), cell, "sorted " + cell, counts=n)["A"]
    ok += [abs(float(np.mean(H30)) - 1) <= 0.01, old < 0.9, lost > h, oc is not None and abs(oc - h) <= 0.001 and abs(oc - 1) <= 0.06]
    L.append(f"{cell}: healthy, mean of 30 {np.mean(H30):.4f} (sd {np.std(H30):.4f}) | array-scale column (previous rule) {old:.4f} | 5 % loss {lost:.4f} | one command, first plant {oc} (stage {h:.4f}) | sites {len(ps)}")
L.append(f"synthetic (all bars, three columns): {'met' if all(ok) else 'not met'} {ok}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
