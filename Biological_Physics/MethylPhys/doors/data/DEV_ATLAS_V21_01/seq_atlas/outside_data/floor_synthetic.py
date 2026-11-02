"""The common library floor (stage_metaa_atlas FLOOR_EQUAL, run_sample.py --dev-floor-equal): synthetic, written and committed before it
runs, read by the one command. The instrument's job: a healthy sample from another laboratory, whose library floor (the share of methylated
calls at sites unmethylated in every cell) differs from the atlas laboratory's, reads Normal.
Known answer (seed 20261124): base, the spent healthy B-cell input GSM5652316's .beta (its reads at every CpG; the Deconvolver reads it as it
is). Truth at the floor sites (floor_sites_v1, unmarked in every cell) 0 in every cell; at B cells' Met-A identity sites the cell's level in
the Met-A reference, at the island sites each cell's pooled level (clipped to 0.0001..0.3), all clipped to 0.0001..0.9999 elsewhere; each
person's own level around it (logit-normal, sd 0.3, stated). Every library reads each read methylated with probability f + (1 - f) p,
p the person's truth, f the library's floor. Reference: per cell two donors in each sealed half at Poisson(30) reads and floor 0.006 (the
atlas laboratory's median, measured: outside_data/library_floors.txt), written as constructed donor, island and floors files in the
committed formats. Nine new B-cell people at the base's own reads, three laboratories of three: floors 0.0015 (BLUEPRINT's median), 0.006,
0.012. Each read twice by the one command (--full-chain --dev-seq-ref auto --dev-seq-ref-half B --dev-seq-level donors --dev-terr-counted
and the constructed files), with and without --dev-floor-equal.
Bars, with the common floor: 1. each laboratory's median sorted-cell Met-A within 0.03 of 1; 2. each laboratory's median islands reading
within 0.03 of 1. Recorded: the readings without it (the offset the floor makes) and each library's measured floor.
Usage: python3 floor_synthetic.py BASE.beta WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.special import expit, logit
BASE, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as Q, pyarrow.parquet as pq
SA.SEQ_ATLAS = Q.ARRAY_COLS; SA.SEQ_LEVEL = "pooled"; rng = np.random.default_rng(20261124); HG19 = 28217448; FR = 0.006
KEY = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
FS = pd.read_parquet(Q.FLOOR_SITES); fset = set(FS.probe)
cells = sorted(pd.read_csv(os.path.join(CH, "..", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv")).cell.str.strip().unique())
person = lambda lv: np.where(lv <= 0, 0.0, expit(logit(np.clip(lv, 1e-4, 1 - 1e-4)) + rng.normal(0, 0.3, len(lv))))
obs = lambda n, p, f: rng.binomial(n, f + (1 - f) * p)
# islands: constructed per-half counts for every cell
P = pd.Index(SA.territory_sets()["islands"]); Ki = KEY.reindex(P).dropna().astype(np.int64); ip = Ki.index
ilev = {c: np.where(ip.isin(fset), 0.0, np.clip(SA._col("mu", c).reindex(ip).fillna(0.001).to_numpy(), 1e-4, 0.3)) for c in cells}
rows, frows = [], []
for c in cells:
    d = {}
    for h in ("A", "B"):
        m_ = np.zeros(len(ip), np.int64); n_ = np.zeros(len(ip), np.int64)
        for t in range(2): n1 = rng.poisson(30, len(ip)); m_ += obs(n1, person(ilev[c]), FR); n_ += n1; frows.append(dict(gsm=f"{c}|{h}{t}", cell=c, half=h, floor=FR, calls=0))
        d["m" + h] = m_.astype(np.uint16); d["n" + h] = n_.astype(np.uint16)
    rows.append(pd.DataFrame(dict(cell=c, probe=ip, **d)))
TF = os.path.join(W, "terr_counts.parquet"); pd.concat(rows, ignore_index=True).to_parquet(TF, index=False)
# B cells' identity sites: constructed per-donor counts
R = pd.read_parquet(Q.METAA_REF); R = R[R.cell == "b cells"]
lv = np.where(R.level.to_numpy() == 65535, np.nan, R.level.to_numpy() / 1e4); R = R.assign(lv=np.where(R.probe.isin(fset), 0.0, np.nan_to_num(lv, nan=0.5))).drop_duplicates("probe")
drows = []
for h in ("A", "B"):
    for t in range(2):
        g = f"b cells|{h}{t}"; n1 = rng.poisson(30, len(R)); drows.append(pd.DataFrame(dict(cell="b cells", probe=R.probe.to_numpy(), cpg=R.cpg.to_numpy(), gsm=g, half=h, m=obs(n1, person(R.lv.to_numpy()), FR), n=n1)))
DF = os.path.join(W, "donor_counts.parquet"); pd.concat(drows, ignore_index=True).to_parquet(DF, index=False)
FF = os.path.join(W, "floors.parquet"); pd.DataFrame(frows).to_parquet(FF, index=False)
# nine people: truth at island, identity and floor sites, the base's reads everywhere
b = np.fromfile(BASE, np.uint8).reshape(-1, 2).astype(np.uint16); assert len(b) == HG19, len(b)
Kf = KEY.reindex(FS.probe).dropna().astype(np.int64)
IF = os.path.join(CH, "MethylPhys_Interface"); res = []
for lab, f in (("floor 0.0015", 0.0015), ("floor 0.006", 0.006), ("floor 0.012", 0.012)):
    for t in range(3):
        a = b.copy()
        for cp, p_ in ((Ki.to_numpy(), person(ilev["b cells"])), (R.cpg.to_numpy(), person(R.lv.to_numpy())), (Kf.to_numpy(), np.zeros(len(Kf)))):
            n = a[cp - 1, 1].astype(np.int64); a[cp - 1, 0] = obs(n, p_, f)
        fn = os.path.join(W, "s.lbeta"); a.tofile(fn)
        for eq in (False, True):
            J = os.path.join(W, "s.json")
            cmd = [sys.executable, "run_sample.py", "--dev-seq-beta", fn, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", f"s{t}", "--full-chain", "--dev-seq-ref", "auto",
                   "--dev-seq-ref-half", "B", "--dev-seq-level", "donors", "--dev-terr-counted", "--dev-terr-counts", TF, "--dev-seq-donors", DF, "--dev-input-floors", FF, "--out", os.path.join(W, "s.html"), "--bundle", J]
            r = subprocess.run(cmd + (["--dev-floor-equal"] if eq else []), cwd=IF, capture_output=True, text=True)
            B_ = json.load(open(J)) if os.path.exists(J) else {}; D_ = B_.get("development") or {}
            isl = (((B_.get("territories") or {}).get("sets") or {}).get("islands") or {}).get("counted") or {}
            res.append(dict(lab=lab, person=t, floor_equal=eq, metaa=(D_.get("metaa_atlas") or {}).get("A"), islands=isl.get("A"),
                            measured_floor=((D_.get("library_floor") or {}).get("unmarked") or {}).get("rate"), floor_record=json.dumps((D_.get("metaa_atlas") or {}).get("floor"))))
            if res[-1]["metaa"] is None: print(r.stdout[-400:], r.stderr[-600:])
            if os.path.exists(J): os.remove(J)
            print(res[-1], flush=True)
        os.remove(fn)
D = pd.DataFrame(res); D.to_csv(os.path.join(W, "floor_readings.csv"), index=False)
L = ["The common library floor: synthetic, read by run_sample.py (seed 20261124, base GSM5652316, reference floor 0.006, nine people in three laboratories)"]
ok = True
for (lab, eq), g in D.groupby(["lab", "floor_equal"]):
    m1, m2 = pd.to_numeric(g.metaa, errors="coerce").median(), pd.to_numeric(g.islands, errors="coerce").median()
    L.append(f"{lab}, {'with' if eq else 'without'} the common floor: sorted-cell Met-A median {m1:.4f}, islands median {m2:.4f}; measured floors {', '.join(f'{x:.5f}' for x in g.measured_floor.dropna())}")
    if eq: ok &= bool(abs(m1 - 1) <= 0.03) and bool(abs(m2 - 1) <= 0.03) and len(g) == 3
w = D[D.floor_equal]; b1 = all(abs(pd.to_numeric(g.metaa).median() - 1) <= 0.03 for _, g in w.groupby("lab")); b2 = all(abs(pd.to_numeric(g.islands).median() - 1) <= 0.03 for _, g in w.groupby("lab"))
L += [f"bar 1 each laboratory's sorted-cell Met-A median within 0.03 of 1 (common floor): {'met' if b1 else 'not met'}",
      f"bar 2 each laboratory's islands median within 0.03 of 1 (common floor): {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 and ok else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
