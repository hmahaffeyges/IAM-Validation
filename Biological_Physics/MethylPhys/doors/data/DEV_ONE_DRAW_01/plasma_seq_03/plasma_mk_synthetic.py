"""A plasma holding a cell the IAM Sequencing Atlas lacks (megakaryocytes): synthetic, written and committed before it runs (development).
The question: does DNA from a cell absent from the ruler move the domains and whole-sample Met-A the way the 30 real plasmas read
(domains 0.8666, whole-sample Met-A 0.9205), when the same construction without it reads Normal (the depth sweep, 20 reads)?
Truth: the plasma_depth_synthetic shares (parsed from the committed plasma_depth_synthetic.txt), each cell at its pooled level in the
committed 57-cell array columns, and a megakaryocyte fraction m at the provisional megakaryocyte column's pooled level (meth / reads over
donors 1 and 2, the bucket's seq_atlas/pieces_mk). A level counted from a few reads sits at 0 or 1 and would look ordered, so the
construction uses only atlas sites where the megakaryocyte column has at least T reads: T = 10, or the first of 8 and 6 that leaves the
Deconvolver at least 500 of its markers (stated). p = (1 - m) x the mix + m x megakaryocyte, m in 0, 0.1, 0.25; Poisson(20) reads at each
used site; three constructed plasmas at each m (seed 20261104); read by the one command with plasma_seq_02's settings.
Bar (one): at m = 0 (the control) the median domains reading and median whole-sample Met-A within 0.02 of 1. Recorded: every m, and the
Deconvolver's five largest shares at m = 0.25 (where the megakaryocyte DNA lands).
Usage: python3 plasma_mk_synthetic.py MK_PIECES_DIR WORKDIR OUT.txt"""
import os, sys, re, glob, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
MKD, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as Q
SA.SEQ_ATLAS = Q.ARRAY_COLS; SA.SEQ_LEVEL = "pooled"; rng = np.random.default_rng(20261104); HG19 = 28217448
sl = [l for l in open(os.path.join(HERE, "plasma_depth_synthetic.txt")) if l.startswith("shares planted:")][0]
F = {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"([^,:]+?) ([0-9.]+)(?:,|$)", sl.split(":", 1)[1].strip())}; t_ = sum(F.values()); F = {k: v / t_ for k, v in F.items()}
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").drop_duplicates("loyfer_cpg_index").set_index("loyfer_cpg_index")["probe"]
mm, rr = [], []
for f in sorted(glob.glob(os.path.join(MKD, "piece_*.parquet"))):
    T = pq.read_table(f, columns=["cpg", "megakaryocytes|meth", "megakaryocytes|reads"]).to_pandas(); T = T[T.cpg.isin(K.index)]
    r = T["megakaryocytes|reads"].to_numpy().astype("float64"); m = T["megakaryocytes|meth"].to_numpy().astype("float64"); bad = (r == 65535) | (m == 65535)
    mm.append(pd.Series(np.where(bad, np.nan, m), index=K.reindex(T.cpg).to_numpy())); rr.append(pd.Series(np.where(bad, 0, r), index=K.reindex(T.cpg).to_numpy()))
MM, RR = pd.concat(mm), pd.concat(rr); MM = MM[~MM.index.duplicated()]; RR = RR[~RR.index.duplicated()]
mk_markers = pd.read_parquet(Q.DECONV_REF).probe if "probe" in pd.read_parquet(Q.DECONV_REF).columns else pd.read_parquet(Q.DECONV_REF).index
for T_ in (10, 8, 6):
    use = RR.index[RR >= T_]
    if pd.Index(mk_markers).isin(use).sum() >= 500: break
use = pd.Index(use); pmk = (MM[use] / RR[use]).to_numpy()
pmix = sum(f * SA._col("mu", c).reindex(use).to_numpy(dtype="float64") for c, f in F.items()); ok = np.isfinite(pmix) & np.isfinite(pmk)
use, pmk, pmix = use[ok], pmk[ok], pmix[ok]; cpg = pd.Series(K.index, index=K.to_numpy()).reindex(use).astype(np.int64).to_numpy()
dom = pd.Index(SA.territory_sets()["domains"]); nmk = int(pd.Index(mk_markers).isin(use).sum())
IF = os.path.join(CH, "MethylPhys_Interface"); res = []; top = {}
for m_ in (0.0, 0.1, 0.25):
    p = np.clip((1 - m_) * pmix + m_ * pmk, 0, 1)
    for j in range(3):
        n = rng.poisson(20, len(use)).astype(np.int64); a = np.zeros((HG19, 2), np.uint16); a[cpg - 1, 1] = n; a[cpg - 1, 0] = rng.binomial(n, p)
        f = os.path.join(W, f"k{j}.lbeta"); a.tofile(f); J = os.path.join(W, f"k{int(m_ * 100)}_{j}.json")
        r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "plasma", "--id", f"k{j}", "--full-chain", "--dev-seq-ref", "auto",
                            "--dev-whole-sites", "--dev-seq-level", "donors", "--dev-terr-counted", "--out", os.path.join(W, "k.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
        B = json.load(open(J)) if os.path.exists(J) else {}; s = (B.get("territories") or {}).get("sets") or {}; w = B.get("met_a_whole") or {}
        sh = sorted(((x["cell"], x["share"]) for x in ((B.get("cell_shares") or {}).get("cells") or [])), key=lambda x: -x[1])
        res.append(dict(m=m_, j=j, domains=(s.get("domains") or {}).get("A"), whole=w.get("A"), islands=(s.get("islands") or {}).get("A"),
                        identity=((w.get("by_place") or {}).get("identity sites of the cells found") or {}).get("A")))
        if m_ == 0.25 and j == 0: top = sh[:5]
        if res[-1]["whole"] is None: print(m_, j, str((B.get("cell_shares") or {}).get("reason"))[:200], r.stdout[-300:], r.stderr[-300:])
        os.remove(f); print(res[-1], flush=True)
D = pd.DataFrame(res); D.to_csv(os.path.join(W, "readings.csv"), index=False); G = D.groupby("m").median(numeric_only=True)
b1 = 0.0 in G.index and abs(G.loc[0.0, "domains"] - 1) <= 0.02 and abs(G.loc[0.0, "whole"] - 1) <= 0.02
L = [f"A plasma holding megakaryocyte DNA the atlas lacks: synthetic, seed 20261104; {len(use)} atlas sites with at least {T_} megakaryocyte reads "
     f"({int(dom.isin(use).sum())} domain sites, {nmk} Deconvolver markers); Poisson(20) reads", "shares (from plasma_depth_synthetic.txt): " + ", ".join(f"{c} {f:.3f}" for c, f in sorted(F.items(), key=lambda x: -x[1]))]
L += [f"m {m_}: domains {r.domains:.4f}, whole-sample Met-A {r.whole:.4f}, islands {r.islands:.4f}, identity sites {r.identity:.4f} (medians of 3)" for m_, r in G.iterrows()]
L += ["Deconvolver's five largest shares at m 0.25 (recorded): " + ", ".join(f"{c} {v:.3f}" for c, v in top),
      f"bar 1 at m 0, domains and whole-sample Met-A within 0.02 of 1: {'met' if b1 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
