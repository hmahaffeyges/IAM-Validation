"""VERSION 2 (written and committed 2026-11-07 before it runs; version 1 not met: it counted the islands' own low methylation, and its
constructed people carried an island spread the pooled reference does not, so the atlas laboratory's own case read 1.235 uncorrected). Here
the chain counts the floor only at the library-floor sites inside the islands (every healthy cell unmarked); every constructed person and
donor half carries the atlas's own island levels (no person spread at the islands); the floor sites inside islands take the island share of
the false-call rate, as the other island sites do. Seed 20261108; outputs floor_class_v2_*. Bars unchanged.

The floor counted at the islands read (stage_metaa_atlas FLOOR_CLASS, run_sample.py --dev-floor-class): synthetic, written and committed
before it runs, read by the one command; built on floor_synthetic.py (11-02), whose reference passed its whole floor to the islands where the
real atlas laboratory passes 0.291 of it (floor_by_site_class, 11-03). The instrument's job: a healthy sample from another laboratory reads
Normal at the islands. Known answer (seed 20261107): as floor_synthetic.py, except that each library's false calls reach the islands at its
own share of its floor: the reference (atlas laboratory, floor 0.006 at the floor sites) at share 0.29, and five outside laboratories of
three people each at floor and share (0.0015, 1.0), (0.005, 1.05), (0.006, 1.0), (0.009, 1.05), (0.016, 0.75), the measured span of the 89
outside libraries. Each person read three times by the one command: no floor, the common floor (--dev-floor-equal), the class floor
(--dev-floor-class). Bars, with the class floor: 1. each laboratory's median islands reading within 0.03 of 1; 2. the reference's own
laboratory (floor 0.006, share 0.29, as the atlas inputs) within 0.03 of 1 too, so the class floor does not overcorrect where the common floor
did. Recorded: the readings without it and with the common floor, and each floor the chain measured at the islands.
Range of the data: real: library floors 0.0016 to 0.0158 (crosslab_chain_table.csv), share of the floor at the islands 0.29 in the atlas
laboratory and 0.75 to 1.05 outside (crosslab_floor_by_class.txt); synthetic: floors 0.0015 to 0.016, shares 0.29 and 0.75 to 1.05.
Usage: python3 floor_class_synthetic.py BASE.beta WORKDIR OUT.txt [PART]   (PART 0 to 17: one laboratory setting and person, its rows only,
       written to WORKDIR/floor_class_part<PART>.csv; 2026-11-07, so 54 runs go on 18 machines, each saved as it finishes)
       python3 floor_class_synthetic.py combine FOLDER_OF_PARTS OUT.txt   (the 18 parts scored by the same bars)"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.special import expit, logit
import glob as _glob
COMBINE = sys.argv[1] == "combine"
BASE, W, OUT = sys.argv[1:4]; PART = int(sys.argv[4]) if len(sys.argv) > 4 else None
if not COMBINE: os.makedirs(W, exist_ok=True)
def score(D):
    L = ["The floor counted at the islands read: synthetic version 2, read by run_sample.py (seed 20261108, base GSM5652316; reference floor 0.006 passing 0.29 of it to the islands; three people per laboratory)"]
    for (lab, eq), g in D.groupby(["lab", "floor_equal"], sort=False):
        L.append(f"{lab}, {eq}: islands median {pd.to_numeric(g.islands, errors='coerce').median():.4f}; sorted-cell Met-A median {pd.to_numeric(g.metaa, errors='coerce').median():.4f} (the cell the Deconvolver found most of, label_check: {', '.join(sorted(set(map(str, g.chain_cell))))}); islands floor {str(g.islands_floor.iloc[0])[:160]}")
    w = D[D.floor_equal == "--dev-floor-class"]; med = {lab: pd.to_numeric(g.islands, errors="coerce").median() for lab, g in w.groupby("lab")}
    b1 = len(med) == 6 and all(abs(v - 1) <= 0.03 for v in med.values()) and all(len(g) == 3 for _, g in w.groupby("lab"))
    b2 = abs(med.get("atlas laboratory 0.006 x 0.29", 9) - 1) <= 0.03
    L += [f"bar 1 (class floor: each laboratory's median islands within 0.03 of 1): {'met' if b1 else 'not met'}",
          f"bar 2 (class floor: the atlas laboratory's own case within 0.03 of 1, no overcorrection): {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 else 'not met'}"]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
if COMBINE:
    D = pd.concat([pd.read_csv(x) for x in sorted(_glob.glob(os.path.join(W, "**", "floor_class_v2_part*.csv"), recursive=True))], ignore_index=True)
    D.to_csv(os.path.join(os.path.dirname(os.path.abspath(OUT)), "floor_class_v2_readings.csv"), index=False); score(D); sys.exit(0)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as Q, pyarrow.parquet as pq
SA.SEQ_ATLAS = Q.ARRAY_COLS; SA.SEQ_LEVEL = "pooled"; rng = np.random.default_rng(20261108); HG19 = 28217448; FR = 0.006
KEY = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
FS = pd.read_parquet(Q.FLOOR_SITES); fset = set(FS.probe)
cells = sorted(pd.read_csv(os.path.join(CH, "..", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv")).cell.str.strip().unique())
person = lambda lv: np.where(lv <= 0, 0.0, expit(logit(np.clip(lv, 1e-4, 1 - 1e-4)) + rng.normal(0, 0.3, len(lv))))
obs = lambda n, p, f: rng.binomial(n, f + (1 - f) * p)
SH_REF = 0.29   # the atlas laboratory's share of its floor at the islands (measured)
# islands: constructed per-half counts for every cell
P = pd.Index(SA.territory_sets()["islands"]); Ki = KEY.reindex(P).dropna().astype(np.int64); ip = Ki.index
ilev = {c: np.where(ip.isin(fset), 0.0, np.clip(SA._col("mu", c).reindex(ip).fillna(0.001).to_numpy(), 1e-4, 0.3)) for c in cells}
rows, frows = [], []
for c in cells:
    d = {}
    for h in ("A", "B"):
        m_ = np.zeros(len(ip), np.int64); n_ = np.zeros(len(ip), np.int64)
        for t in range(2): n1 = rng.poisson(30, len(ip)); m_ += obs(n1, ilev[c], FR * SH_REF); n_ += n1; frows.append(dict(gsm=f"{c}|{h}{t}", cell=c, half=h, floor=FR, calls=0))
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
LABS = (("atlas laboratory 0.006 x 0.29", 0.006, 0.29), ("0.0015 x 1.0", 0.0015, 1.0), ("0.005 x 1.05", 0.005, 1.05), ("0.006 x 1.0", 0.006, 1.0), ("0.009 x 1.05", 0.009, 1.05), ("0.016 x 0.75", 0.016, 0.75))
for lab, f, sh in LABS:
    for t in range(3):
        _k = [l_ for l_, _, _ in LABS].index(lab) * 3 + t
        if PART is not None and _k != PART: continue
        if PART is not None: rng = np.random.default_rng(20261108 + 1000 + _k)   # 2026-11-07: one part per machine, its own seed
        a = b.copy()
        for cp, p_, ff in ((Ki.to_numpy(), ilev["b cells"], f * sh), (R.cpg.to_numpy(), person(R.lv.to_numpy()), f), (Kf.to_numpy(), np.zeros(len(Kf)), np.where(np.isin(Kf.to_numpy(), Ki.to_numpy()), f * sh, f))):
            n = a[cp - 1, 1].astype(np.int64); a[cp - 1, 0] = obs(n, p_, ff)
        fn = os.path.join(W, "s.lbeta"); a.tofile(fn)
        for eq in (None, "--dev-floor-equal", "--dev-floor-class"):
            J = os.path.join(W, "s.json")
            cmd = [sys.executable, "run_sample.py", "--dev-seq-beta", fn, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", f"s{t}", "--full-chain", "--dev-seq-ref", "auto",
                   "--dev-seq-ref-half", "B", "--dev-seq-level", "donors", "--dev-terr-counted", "--dev-terr-counts", TF, "--dev-seq-donors", DF, "--dev-input-floors", FF, "--out", os.path.join(W, "s.html"), "--bundle", J]
            r = subprocess.run(cmd + ([eq] if eq else []), cwd=IF, capture_output=True, text=True)
            B_ = json.load(open(J)) if os.path.exists(J) else {}; D_ = B_.get("development") or {}
            isl = (((B_.get("territories") or {}).get("sets") or {}).get("islands") or {}).get("counted") or {}
            res.append(dict(lab=lab, person=t, floor_equal=eq or "none", islands_floor=json.dumps(isl.get("floor")), chain_cell=(B_.get("label_check") or {}).get("chain_cell"), metaa=(D_.get("metaa_atlas") or {}).get("A"), islands=isl.get("A"),
                            measured_floor=((D_.get("library_floor") or {}).get("unmarked") or {}).get("rate"), floor_record=json.dumps((D_.get("metaa_atlas") or {}).get("floor"))))
            if res[-1]["metaa"] is None: print(r.stdout[-400:], r.stderr[-600:])
            if os.path.exists(J): os.remove(J)
            print(res[-1], flush=True)
        os.remove(fn)
if PART is not None: pd.DataFrame(res).to_csv(os.path.join(W, f"floor_class_v2_part{PART}.csv"), index=False); sys.exit(0)
D = pd.DataFrame(res); D.to_csv(os.path.join(W, "floor_class_readings.csv"), index=False)
score(D)
