"""The transfer function of the chain on sequencing (checklist D9): how much of a planted change at the CpG islands the chain returns.
Written and committed before it runs. The instrument's job: a planted change at the islands is read back at its full size.
Known answer: base, the spent healthy B-cell input GSM5652316 (its .beta). At every island site (stage_metaa_atlas.territory_sets), the
base's own level there (k / n) raised by delta, and its reads drawn again at that level, Binomial(n, level + delta), n the base's own reads;
every other site unchanged. delta 0, 0.01, 0.02, 0.05 and 0.1, three draws each (seed 20261116 + 100 x draw + 1000 x delta). Each read by the
one command with healthy pass 5's settings for this input (healthy_fullchain5_samples.csv) and --dev-terr-sites. Recorded per read: the
islands' level change (the sample's level minus the expected, the chain's own number), the islands reading and the counted islands reading,
whole-sample Met-A. Scored by transfer_score.py: bar 1, the slope of the level change on delta (least squares over all reads) within 0.95
to 1.05; bar 2, the median counted islands reading rises at every step of delta.
Usage: python3 transfer_synthetic.py BASE.beta WORKDIR ROWS.csv DELTA [DELTA ...]"""
import os, sys, json, subprocess, numpy as np, pandas as pd
BASE, W, ROWS = sys.argv[1:4]; DELTAS = [float(x) for x in sys.argv[4:]]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, pyarrow.parquet as pq
HG19 = 28217448; P = pd.Index(SA.territory_sets()["islands"])
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"].reindex(P).dropna().astype(np.int64)
cpg = np.unique(K.to_numpy())
S = pd.read_csv(os.path.join(CH, "..", "doors", "data", "DEV_METAA_ATLAS_CURVE_01", "seq_atlas_01", "healthy_fullchain5_samples.csv")); S["gsm"] = S.gsm.str.strip()
r0 = S[S.gsm == "GSM5652316"].iloc[0]; FL = str(r0.rsflags).split()
b = np.fromfile(BASE, np.uint8).reshape(-1, 2).astype(np.uint16); assert len(b) == HG19, len(b)
k0 = b[cpg - 1, 0].astype(np.int64); n0 = b[cpg - 1, 1].astype(np.int64); has = n0 > 0; IF = os.path.join(CH, "MethylPhys_Interface"); rows = []
for d in DELTAS:
    for t in range(3):
        rng = np.random.default_rng(20261116 + 100 * t + int(round(1000 * d))); a = b.copy()
        p = np.clip(np.where(has, k0 / np.maximum(n0, 1), 0) + d, 0, 1); a[cpg - 1, 0] = np.where(has, rng.binomial(n0, p), 0)
        f = os.path.join(W, f"d{d}_t{t}.lbeta"); a.tofile(f); J = os.path.join(W, f"d{d}_t{t}.json")
        r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", str(r0.get("specimen") or "sorted b cells"), "--id", f"d{d}t{t}",
                            *FL, "--dev-terr-sites", "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
        B = json.load(open(J)) if os.path.exists(J) else {}; I_ = ((B.get("territories") or {}).get("sets") or {}).get("islands", {})
        rows.append(dict(delta=d, draw=t, level_change=I_.get("level_change"), islands=I_.get("A"), islands_plugin=I_.get("A_plugin"), metaa_whole=(B.get("met_a_whole") or {}).get("A"),
                         sites=(I_.get("counted") or {}).get("sites"), rc=r.returncode))
        if I_.get("level_change") is None: print("no reading", d, t, r.stdout[-300:], r.stderr[-600:], flush=True)
        os.remove(f); print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(ROWS, index=False)
