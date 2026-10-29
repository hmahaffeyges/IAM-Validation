"""DEV-TERRITORIES-01, the territories on sequencing in sequencing units: synthetic (written before any reading on the amended chain).
For two atlas cells, a planted healthy sequencing sample: at every atlas site with a sequencing address (sites_sequencing_key), 20 reads drawn
from the cell's column in sequencing units (seq_units; seed 20261028), written as a values file over the whole hg19 numbering and read by the
one command (run_sample.py --dev-seq-beta, the sorted cell as specimen). Bars: (1) islands and domains each within 0.05 of 1 in both cells;
(2) the same plant with 30 % of its island sites gaining a mark (the chain's own plant_territory rule, size 0.5) reads islands above the
healthy plant. Usage: python3 terr_seq_synthetic.py OUT.txt [WORKDIR]"""
import sys, os, json, subprocess, tempfile, numpy as np, pandas as pd
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_values as SV
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); WD = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(); os.makedirs(WD, exist_ok=True)
K = pd.read_parquet(os.path.join(CH, "..", "atlas", "v2.1", "sites_sequencing_key.parquet")).drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
rng = np.random.default_rng(20261028); L = []; ok = []
def read(cell, x, tag):
    i = K.reindex(x.index).dropna().astype("int64"); a = np.zeros((SV.HG19_CPGS, 2), np.uint16)
    a[i.to_numpy() - 1, 0] = np.round(x[i.index].to_numpy() * 20).astype(np.uint16); a[i.to_numpy() - 1, 1] = 20
    f = os.path.join(WD, tag + ".lbeta"); a.tofile(f); b = os.path.join(WD, tag + ".json")
    subprocess.run([sys.executable, RS, "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + cell, "--id", tag,
                    "--out", os.path.join(WD, tag + ".html"), "--bundle", b], capture_output=True, text=True); os.remove(f)
    t = (json.load(open(b)).get("territories") or {}).get("sets") or {} if os.path.exists(b) else {}
    return (t.get("islands") or {}).get("A"), (t.get("domains") or {}).get("A")
for cell in ("naive b cells", "cd8 t cells"):
    mu = SA._col("mu", cell).dropna(); ps = SA.seq_units(mu)
    x = pd.Series(rng.binomial(20, ps.to_numpy()) / 20, index=ps.index)
    hi, hd = read(cell, x, "healthy")
    g, n = SA.plant_territory(x, "islands", 0.3, 0.5); gi, _ = read(cell, g, "gain")
    ok += [hi is not None and hd is not None and abs(hi - 1) <= 0.05 and abs(hd - 1) <= 0.05, gi is not None and hi is not None and gi > hi]
    L.append(f"{cell}: healthy islands {hi}, domains {hd} | island gain on {n} sites: islands {gi}")
L.append(f"synthetic (both bars, two cells): {'met' if all(ok) else 'not met'} {ok}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
