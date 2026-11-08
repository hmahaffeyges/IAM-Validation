"""VERSION 3 of the class-floor synthetic (written and committed 2026-11-07 before it runs; development). Version 2 was met, but the rehearsal
on the 89 spent outside samples was not: every laboratory read below Normal by a shared 0.06 to 0.15 (DEV_ATLAS_V21_01.md). Version 2 built
both its reference island counts and its people from the atlas's average levels, so they agreed by construction, and it planted 0.29 of the
floor at the floor sites inside the islands, where the real outside libraries carry their whole floor (xl_class_readings.csv: 0.0013, 0.0049,
0.0091 and 0.0120 there against library floors 0.0016, 0.0051, 0.0092 and 0.0158). Here nothing is constructed but the planted false calls:
the reference is the real IAM Sequencing Atlas, one sealed donor half (the chain's own committed island and donor counts); each person is a
real atlas donor of the other half, its own reads at every CpG (GEO GSE186458 .beta), read by the one command with --dev-seq-ref-half set to
the half it is not in. Three people: B cells GSM5652316 and GSM5652318 (half A, read against B), GSM5652317 (half B, read against A).
Planting, from the measured values: a laboratory of floor f and island share s gets at every library-floor site (inside the islands too)
methylated calls drawn at rate f (every methylated call there is false), and at the other island sites extra methylated calls on the
unmethylated reads at rate max(0, f*s - 0.006*0.29) (the atlas laboratory's own island false calls are already in the donor's reads). The
atlas laboratory's own case plants nothing.
The instrument's job: a healthy sample from another laboratory reads Normal at the islands.
Reference: IAM Sequencing Atlas only (the donor half the person is not in); no tare.
Bars, with the class floor (--dev-floor-class): 1. each laboratory's median islands reading within 0.03 of 1; 2. the atlas laboratory's own
case within 0.03 of 1. Recorded: the readings with no floor and with the common floor, and each floor the chain measured.
Range of the data: real outside libraries: floors 0.0016 to 0.0158 (crosslab_chain_table.csv), island shares 0.75 to 1.05 and 0.29 in the
atlas laboratory (crosslab_floor_by_class.txt), the floor at the floor sites inside islands equal to the library floor (xl_class_readings.csv);
synthetic: floors 0.0015 to 0.016, shares 0.75 to 1.05, real atlas donors and the real atlas reference.
Usage: python3 floor_class_synthetic_v3.py BASE.beta HALF_TO_READ_AGAINST PERSON LAB_INDEX WORKDIR OUT_PART.csv
       python3 floor_class_synthetic_v3.py combine FOLDER_OF_PARTS OUT.txt"""
import os, sys, json, subprocess, glob, numpy as np, pandas as pd
def score(D, out):
    L = ["The floor counted at the islands read: synthetic version 3 (real atlas donors against the other sealed half; seed 20261109)"]
    for (lab, eq), g in D.groupby(["lab", "floor_equal"], sort=False):
        L.append(f"  {lab:32} {eq:20} islands median {pd.to_numeric(g.islands, errors='coerce').median():.4f} | each {[round(x, 4) if pd.notna(x) else None for x in pd.to_numeric(g.islands, errors='coerce')]}")
    C = D[D.floor_equal == "--dev-floor-class"]; m = C.groupby("lab").islands.median()
    b1 = bool(len(m) == 6 and ((m - 1).abs() <= 0.03).all()); b2 = bool(abs(m.get("atlas laboratory 0.006 x 0.29", np.nan) - 1) <= 0.03)
    L += [f"bar 1 (class floor: each laboratory's median islands within 0.03 of 1): {'met' if b1 else 'not met'}",
          f"bar 2 (class floor: the atlas laboratory's own case within 0.03 of 1): {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 else 'not met'}"]
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
if sys.argv[1] == "combine":
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(sys.argv[2], "floor_class_v3_part*.csv")))], ignore_index=True)
    D.to_csv(os.path.join(os.path.dirname(os.path.abspath(sys.argv[3])), "floor_class_v3_readings.csv"), index=False); score(D, sys.argv[3]); sys.exit(0)
BASE, HALF, PERSON, LK, W, OUTP = sys.argv[1:7]; LK = int(LK); os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as Q, pyarrow.parquet as pq
HG19 = 28217448
KEY = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
FS = pd.read_parquet(Q.FLOOR_SITES); FS = FS[FS.side == "unmarked"] if "side" in FS else FS
Kf = KEY.reindex(FS.probe).dropna().astype(np.int64).to_numpy()
Ki = KEY.reindex(pd.Index(SA.territory_sets()["islands"])).dropna().astype(np.int64).to_numpy(); Ki = np.setdiff1d(Ki, Kf)
b = np.fromfile(BASE, np.uint8).reshape(-1, 2).astype(np.int64); assert len(b) == HG19, len(b)
LABS = (("atlas laboratory 0.006 x 0.29", None, None), ("0.0015 x 1.0", 0.0015, 1.0), ("0.005 x 1.05", 0.005, 1.05), ("0.006 x 1.0", 0.006, 1.0), ("0.009 x 1.05", 0.009, 1.05), ("0.016 x 0.75", 0.016, 0.75))
IF = os.path.join(CH, "MethylPhys_Interface"); res = []
for k, (lab, f, s) in enumerate(LABS):
    if k != LK: continue
    rng = np.random.default_rng(20261109 + 100 * k + int(PERSON[-2:]))
    a = b.copy()
    if f is not None:
        n = a[Kf - 1, 1]; a[Kf - 1, 0] = rng.binomial(n, f)
        n = a[Ki - 1, 1]; m = a[Ki - 1, 0]; a[Ki - 1, 0] = m + rng.binomial(np.maximum(n - m, 0), max(0.0, f * s - 0.006 * 0.29))
    assert a.max() < 256 and (a[:, 0] <= a[:, 1]).all()   # the .beta format: two 0-255 counts a CpG, methylated never above total
    fn = os.path.join(W, "s.lbeta"); a.astype(np.uint8).tofile(fn)
    for eq in (None, "--dev-floor-equal", "--dev-floor-class"):
        J = os.path.join(W, "s.json")
        cmd = [sys.executable, "run_sample.py", "--dev-seq-beta", fn, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", f"{PERSON}", "--full-chain", "--dev-seq-ref", "auto",
               "--dev-seq-ref-half", HALF, "--dev-seq-level", "donors", "--dev-terr-counted", "--out", os.path.join(W, "s.html"), "--bundle", J]
        r = subprocess.run(cmd + ([eq] if eq else []), cwd=IF, capture_output=True, text=True)
        B_ = json.load(open(J)) if os.path.exists(J) else {}
        isl = (((B_.get("territories") or {}).get("sets") or {}).get("islands") or {}).get("counted") or {}
        res.append(dict(lab=lab, person=PERSON, half=HALF, floor_equal=eq or "none", islands=isl.get("A"), islands_floor=json.dumps(isl.get("floor")),
                        chain_cell=(B_.get("label_check") or {}).get("chain_cell"), cells=json.dumps(isl.get("cells"))[:400], rc=r.returncode))
        if res[-1]["islands"] is None: print(r.stdout[-400:], r.stderr[-600:], flush=True)
        if os.path.exists(J): os.remove(J)
        print(res[-1], flush=True)
        pd.DataFrame(res).to_csv(OUTP, index=False)   # saved after every reading
    os.remove(fn)
