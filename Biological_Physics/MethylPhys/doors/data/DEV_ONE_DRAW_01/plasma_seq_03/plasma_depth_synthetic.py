"""Plasma at plasma depth: synthetic, written and committed before it runs (development). The instrument's job: a healthy plasma, read as a
mix of the atlas's cells in its own shares, reads Normal.
Why: the 30 tuning plasmas read the domains 0.8666 and whole-sample Met-A 0.9205 against the 57-cell IAM Sequencing Atlas, at a median of 1
read per site; the healthy passes read both near 1 at about 26. This separates the instrument at plasma depth from the plasmas themselves.
Known answer: a spent plasma's molecules (its .pat) give its read count n at each atlas site (stage_seq_values.values_from_pat) and its
plasma_seq_02 bundle gives its cell shares F (cells at 0.01 or more, renormalized). Six constructed plasmas (seed 20261103): at each site,
k ~ Binomial(n, sum over cells of F x the cell's pooled level in the committed array columns); every read from a cell of the mix, and the
truth exactly the atlas, so a healthy plasma by construction (no person spread, no laboratory floor). Written as uint16 .lbeta at the atlas
sites' hg19 CpGs (other CpGs empty) and read by the one command with plasma_seq_02's settings (--dev-seq-beta, --specimen plasma).
Bars (two): the six constructed plasmas' median domains reading within 0.02 of 1; their median whole-sample Met-A within 0.02 of 1.
Recorded: the islands, Met-A at the identity sites of the cells found, and the shares the Deconvolver returns beside F.
Usage: python3 plasma_depth_synthetic.py PLASMA.pat.gz PLASMA_bundle.json WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
PAT, BUN, W, OUT = sys.argv[1:5]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as Q, stage_seq_values as SV
SA.SEQ_ATLAS = Q.ARRAY_COLS; SA.SEQ_LEVEL = "pooled"; rng = np.random.default_rng(20261103); HG19 = 28217448
_, n, _s = SV.values_from_pat(PAT, 1)
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
n = n[n.index.isin(K.index)]; cpg = K.reindex(n.index).astype(np.int64).to_numpy()
F = {x["cell"]: x["share"] for x in (json.load(open(BUN)).get("cell_shares") or {}).get("cells") or [] if x["share"] >= 0.01}; t_ = sum(F.values()); F = {k: v / t_ for k, v in F.items()}
p = sum(f * SA._col("mu", c).reindex(n.index).to_numpy(dtype="float64") for c, f in F.items())
ok = np.isfinite(p); p = np.clip(np.where(ok, p, 0), 0, 1); nn = np.where(ok, n.to_numpy(), 0).astype(np.int64)
IF = os.path.join(CH, "MethylPhys_Interface"); res = []
for t in range(6):
    a = np.zeros((HG19, 2), np.uint16); a[cpg - 1, 1] = nn; a[cpg - 1, 0] = rng.binomial(nn, p); f = os.path.join(W, f"p{t}.lbeta"); a.tofile(f); J = os.path.join(W, f"p{t}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "plasma", "--id", f"p{t}", "--full-chain", "--dev-seq-ref", "auto",
                        "--dev-whole-sites", "--dev-seq-level", "donors", "--dev-terr-counted", "--out", os.path.join(W, "p.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    B = json.load(open(J)) if os.path.exists(J) else {}; s = (B.get("territories") or {}).get("sets") or {}; w = B.get("met_a_whole") or {}
    sh = {x["cell"]: x["share"] for x in ((B.get("cell_shares") or {}).get("cells") or [])}
    res.append(dict(t=t, domains=(s.get("domains") or {}).get("A"), whole=w.get("A"), islands=(s.get("islands") or {}).get("A"),
                    identity=((w.get("by_place") or {}).get("identity sites of the cells found") or {}).get("A"), share_gap=round(sum(abs(sh.get(c, 0) - f) for c, f in F.items()), 4)))
    if res[-1]["domains"] is None: print(t, r.stdout[-400:], r.stderr[-400:])
    os.remove(f); print(res[-1], flush=True)
D = pd.DataFrame(res); D.to_csv(os.path.join(W, "readings.csv"), index=False); md, mw = D.domains.median(), D.whole.median()
b1 = D.domains.notna().all() and abs(md - 1) <= 0.02; b2 = D.whole.notna().all() and abs(mw - 1) <= 0.02
L = [f"Plasma at plasma depth: synthetic, seed 20261103, depths and shares from {os.path.basename(PAT).split('.')[0]}; {int((nn > 0).sum())} sites with reads, median {int(np.median(nn[nn > 0]))} per site",
     "shares planted: " + ", ".join(f"{c} {f:.3f}" for c, f in sorted(F.items(), key=lambda x: -x[1])),
     "domains: " + ", ".join(f"{v:.4f}" for v in D.domains.dropna()), "whole-sample Met-A: " + ", ".join(f"{v:.4f}" for v in D.whole.dropna()),
     "islands (recorded): " + ", ".join(f"{v:.4f}" for v in D.islands.dropna()), "Met-A at the identity sites of the cells found (recorded): " + ", ".join(f"{v:.4f}" for v in D.identity.dropna()),
     "share gap, sum over planted cells of |returned - planted| (recorded): " + ", ".join(f"{v:.4f}" for v in D.share_gap.dropna()),
     f"bar 1 domains median {md:.4f} (within 0.02 of 1): {'met' if b1 else 'not met'}", f"bar 2 whole-sample Met-A median {mw:.4f} (within 0.02 of 1): {'met' if b2 else 'not met'}",
     f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
