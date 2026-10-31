"""The sky on sequencing, against the sample's own mix of the IAM Sequencing Atlas (DV.sky_in_mix): synthetic, written and committed before it
runs. The instrument's job: a healthy sample shows no structure beyond the block-shuffle null; a real regional change does.

Known answer: the 6 constructed healthy samples of the in-mix synthetic (same cells, mixes, depths and seed 20261101), and one more (the
first sample's mix at 20 reads) with every site of chromosome 8 moved 0.15 toward full methylation before the reads are drawn, each read by
the one command with the full chain (run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --dev-seq-ref auto --full-chain --specimen
unknown). Bar (operational, set before the run): at most 1 of the 6 healthy samples shows structure (look-elsewhere p < 0.05: the null's
own rate, 0.05 of 6, is 0.3). Run 3 (amended before it ran, after runs 1 and 2 and the diagnostic, disclosed): the whole-chromosome plant is
dropped (a within-chromosome block shuffle absorbs it by construction); what the sky can see is measured, not scored: the first sample's mix
at 20 reads with the first 3,000 and the first 10,000 sites of chromosome 8 in genome order moved 0.15 toward full methylation. Also recorded, not scored: what the full chain says for NILC,
the template split and B cells on sequencing.
Usage: python3 sky_seq_synthetic.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV, stage_metaa_atlas as SA, dev_stages as DV
ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids).map(K); have = cpg.notna().to_numpy(); cpg = cpg[have].astype(np.int64).to_numpy()
chrom = DV._mapping().chr.reindex(pd.Index(ids)[have]).astype(str).to_numpy()
def col(c):
    for k in (1, 2, 3):
        p = os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")
        if c in pq.read_schema(p).names:
            u = pq.read_table(p, columns=[c]).column(0).to_numpy(); v = u / 1e4; v[u == 65535] = np.nan; return v[have]
cells = [c for k in (1, 2, 3) for c in pq.read_schema(os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")).names]
rng = np.random.default_rng(20261101); pick = list(rng.choice(cells, 12, replace=False))
mixes = [({pick[0]: 1.0}, 20, False), ({pick[1]: 1.0}, 8, False), ({pick[2]: 0.7, pick[3]: 0.3}, 20, False), ({pick[4]: 0.7, pick[5]: 0.3}, 8, False),
         ({pick[6]: 0.5, pick[7]: 0.3, pick[8]: 0.2}, 20, False), ({pick[9]: 0.5, pick[10]: 0.3, pick[11]: 0.2}, 8, False), ({pick[0]: 1.0}, 20, 3000), ({pick[0]: 1.0}, 20, 10000)]
L = ["The sky on sequencing (the sample's own mix of the IAM Sequencing Atlas), synthetic run 3: 6 healthy, 2 planted (3,000 and 10,000 sites of chromosome 8, +0.15)",
     "sample | reads | planted | lee p | structure | reference | NILC | template split | B cells"]; hs, pl = [], None
for j, (mx, N, plant) in enumerate(mixes):
    V = [col(c) for c in mx]; m = np.all([~np.isnan(v) for v in V], 0); p = sum(f * np.nan_to_num(v) for f, v in zip(mx.values(), V))
    if plant:   # run 3: the first `plant` sites of chromosome 8 in genome order
        o8 = np.flatnonzero(np.char.lstrip(chrom.astype(str), "chr") == "8"); o8 = o8[np.argsort(cpg[o8])][:plant]; p = p.copy(); p[o8] = p[o8] + 0.15 * (1 - p[o8])
    a = np.zeros((SV.HG19_CPGS, 2), np.uint16); ii = cpg[m]; kk = rng.binomial(N, np.clip(p[m], 0, 1))
    a[ii - 1, 0] = kk; a[ii - 1, 1] = N; f = os.path.join(W, f"k{j}.lbeta"); a.tofile(f); b = os.path.join(W, f"k{j}_bundle.json")
    subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--full-chain", "--specimen", "unknown",
                    "--id", f"k{j}", "--out", os.path.join(W, f"k{j}.html"), "--bundle", b], cwd=os.path.join(C, "MethylPhys_Interface"), capture_output=True, text=True)
    D = json.load(open(b)).get("development") or {}; s = D.get("sky") or {}
    why = lambda k: (D.get(k) or {}).get("status", "absent") + ((": " + str((D.get(k) or {}).get("reason"))[:80]) if (D.get(k) or {}).get("reason") else "")
    L.append(f"{j} | {N} | " + ", ".join(f"{c} {v:.1f}" for c, v in mx.items()) + (f" + {plant} chr8 sites 0.15" if plant else "") +
             f" | {s.get('lee_p')} | {s.get('structure_beyond_null')} | {s.get('reference') or s.get('reason')} | {why('nilc')} | {why('atlas_e')} | {why('percell_b')}")
    (hs.append(s.get("structure_beyond_null")) if not plant else None); os.remove(f)
b1 = len(hs) == 6 and all(x is not None for x in hs) and sum(bool(x) for x in hs) <= 1
L += [f"bar healthy with structure: {sum(bool(x) for x in hs if x is not None)} of {len(hs)} (at most 1): {'met' if b1 else 'not met'}",
      "what the sky sees (measured, not scored): the planted rows above", f"synthetic: {'met' if b1 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
