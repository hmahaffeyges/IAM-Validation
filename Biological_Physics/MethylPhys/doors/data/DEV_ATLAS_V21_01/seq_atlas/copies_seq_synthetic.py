"""The cell-count floor on sequencing (checklist D7; stage_metaa_atlas.read_in_mix, copies_per_site): synthetic, written and committed
before it runs, read by the one command. The instrument's job: each cell's reading says how many copies of that cell stand behind each
site, and flags the cells with fewer than one.
Known answer (seed 20261120): 4 constructed sequenced samples, each two cells of the IAM Sequencing Atlas (b cells and monocytes, shares
0.9 and 0.1), every site of IAM-Atlas v2.1 given exactly n reads (n 4, 8, 20, 60) with methylated reads drawn from the mixed level; copies
of a cell per site are its share times n (monocytes 0.4, 0.8, 2, 6; b cells 3.6, 7.2, 18, 54), as found by the Deconvolver. Written as .lbeta
and read by run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --dev-seq-ref auto --full-chain --specimen unknown. Bars: every reported
copies_per_site within 10 % of the Deconvolver's share times n; the floor flag on exactly the cells with copies below 1.
Usage: python3 copies_seq_synthetic.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV, stage_metaa_atlas as SA
ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids).map(K); have = cpg.notna().to_numpy(); cpg = cpg[have].astype(np.int64).to_numpy()
def col(c):
    for k in (1, 2, 3):
        p = os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")
        if c in pq.read_schema(p).names:
            u = pq.read_table(p, columns=[c]).column(0).to_numpy(); v = u / 1e4; v[u == 65535] = np.nan; return v[have]
cells = [c for k in (1, 2, 3) for c in pq.read_schema(os.path.join(Q.ARRAY_COLS, f"mu_part{k}.parquet")).names]
rng = np.random.default_rng(20261120); IF = os.path.join(C, "MethylPhys_Interface"); HG19 = 28217448
mu_b, mu_m = col("b cells"), col("monocytes"); okm = np.isfinite(mu_b) & np.isfinite(mu_m); res = []
for n in (4, 8, 20, 60):
    lv = 0.9 * mu_b + 0.1 * mu_m; a = np.zeros((HG19, 2), np.uint8); c_ = cpg[okm]
    a[c_ - 1, 1] = n; a[c_ - 1, 0] = rng.binomial(n, lv[okm]); f = os.path.join(W, f"mix{n}.lbeta"); a.astype(np.uint16).tofile(f); J = os.path.join(W, f"mix{n}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--full-chain", "--specimen", "unknown",
                        "--id", f"mix{n}", "--out", os.path.join(W, f"mix{n}.html"), "--bundle", J], cwd=IF, capture_output=True, text=True); os.remove(f)
    B = json.load(open(J)) if os.path.exists(J) else {}
    for x in ((B.get("met_a_each_cell") or {}).get("cells") or []):
        if x.get("copies_per_site") is not None: res.append(dict(n=n, cell=x["cell"], share=x["share"], copies=x["copies_per_site"], expected=x["share"] * n, flag="below_cell_count_floor" in x))
    print(n, [(x["cell"], x["share"], x["copies"], x["flag"]) for x in res if x["n"] == n], "| rc", r.returncode, flush=True)
    if r.returncode: print(r.stderr[-800:], flush=True)
D = pd.DataFrame(res); ok = len(D) > 0
b1 = ok and bool((abs(D.copies - D.expected) <= 0.1 * D.expected).all()); b2 = ok and bool((D.flag == (D.copies < 1)).all())
mono = D[D.cell == "monocytes"] if ok else D
L = ["The cell-count floor, synthetic (seed 20261120), read by run_sample.py: " + "; ".join(f"n {r.n} {r.cell} share {r.share} copies {r.copies} flag {r.flag}" for r in D.itertuples()),
     f"bar 1 copies within 10 % of share times reads: {'met' if b1 else 'not met'}", f"bar 2 flag exactly below one copy: {'met' if b2 else 'not met'}",
     f"monocyte cells read at each depth: {len(mono)} of 4", f"synthetic: {'met' if b1 and b2 and len(mono) == 4 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
