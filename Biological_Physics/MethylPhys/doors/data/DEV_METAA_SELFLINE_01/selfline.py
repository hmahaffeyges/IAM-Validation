"""DEV-METAA-SELFLINE-01 (COM-METAA-039): can a healthy array's own loose share take out its laboratory's Met-A offset? Development.
Written 2026-10-15 before any loose share of these arrays is computed. KISS.
WHY. Met-A on isolated neutrophils is read against a reference from the same laboratory (the tare), because untared readings from other labs
sit high: committed readings give series medians 1.00 (GSE110554, GSE167998, the reference lab), 1.065 (GSE118144), 1.075 (GSE122244),
1.19 (GSE247193), 1.07 (GSE247195). On sequencing, a cell's held error follows its own loose share (PHY-045). If a laboratory's offset also
moves the array's own loose share, a line through the two removes the offset with no same-lab reference.
DATA. Healthy purified neutrophil arrays only, EPIC v1, Met-A A as already committed (untared): DEV_SELFTARE_02/selftare02_floor.csv
(GSE110554, the six reference arrays), DEV_INTAKE_02/intake02_purified_neutrophils_tared.csv (GSE167998; GSE118144 healthy), and
DEV_SELFTARE_02/selftare02_otherlab.csv (GSE122244, GSE247193, GSE247195). Betas: DEV-BASE-CHAIN-01's Stage 1 betas (S3).
LOOSE SHARE, the arrays' own rule (DEV-ARCH-01 arch_box.py): open-sea probes (more than 4 kb from any UCSC hg19 CpG island, cpgIslandExt),
binned by 100 kb, bins with >= 5 probes; L = share of bins whose median beta is below 0.70.
TEST, leave one series out (the two reference-lab series leave together): fit A = a + b L by least squares on the other series' arrays;
the held-out series' corrected offset is median(A - (a + b L)) and its raw offset is median(A) - 1 (the gauge's healthy center).
BARS (development; showcase, not a gate): median |corrected offset| <= half the median |raw offset| over the series, AND the corrected offset
is smaller than the raw one in at least 4 of the 6 series, AND no series' within-series SD grows by more than 25 percent. Otherwise NOT MET,
which says the offset is not carried by the loose share and the same-lab tare stays.
Usage: python selfline.py synthetic | score <arrays.csv with series, gsm, A, L>"""
import sys, os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); REF = {"GSE110554", "GSE167998"}
def score(T):
    units = [sorted(REF & set(T.series))] + [[s] for s in sorted(set(T.series) - REF)]; R = []
    for u in units:
        tr, te = T[~T.series.isin(u)], T[T.series.isin(u)]
        b, a = np.polyfit(tr.L, tr.A, 1); res = te.A - (a + b * te.L)
        for s in u:
            x = te[te.series == s]; rr = res[te.series == s]
            R.append(dict(series=s, n=len(x), raw=float(x.A.median() - 1), corrected=float(rr.median()), sd_raw=float(x.A.std()), sd_corr=float(rr.std()), slope=float(b)))
    R = pd.DataFrame(R)
    b1 = R.corrected.abs().median() <= 0.5 * R.raw.abs().median(); b2 = int((R.corrected.abs() < R.raw.abs()).sum()) >= min(4, len(R))
    b3 = bool((R.sd_corr <= 1.25 * R.sd_raw).all())
    return R, dict(half=bool(b1), majority=bool(b2), spread=b3, verdict="MET" if (b1 and b2 and b3) else "NOT MET")
if sys.argv[1] == "synthetic":
    rng = np.random.default_rng(39); L_ = []
    def make(carried):
        rows = []
        for s, d in zip(["GSE110554", "GSE167998", "S3", "S4", "S5", "S6"], [0, 0, 0.065, 0.075, 0.19, 0.07]):
            for _ in range(8):
                l = 0.30 + (0.8 * d if carried else 0.0) + rng.normal(0, 0.004); A = 1 + d + rng.normal(0, 0.02)
                rows.append(dict(series=s, A=A, L=l + (0 if carried else rng.normal(0, 0.02))))
        return pd.DataFrame(rows)
    for lab, carried, want in (("offset carried by the loose share", True, "MET"), ("offset NOT carried", False, "NOT MET")):
        R, V = score(make(carried)); L_.append(f"{lab}: median |raw| {R.raw.abs().median():.3f} -> |corrected| {R.corrected.abs().median():.3f}; {V['verdict']} (want {want})")
        L_.append(("ok" if V["verdict"] == want else "WRONG"))
    ok = L_.count("ok") == 2; L_ = [x for x in L_ if x not in ("ok", "WRONG")]
    L_.append(f"SYNTHETIC: {'MET' if ok else 'NOT MET'} (the method removes an offset the loose share carries, and refuses one it does not)")
    open(os.path.join(H, "selfline_synthetic.txt"), "w").write("\n".join(L_) + "\n"); print("\n".join(L_))
else:
    T = pd.read_csv(sys.argv[2]); n0 = len(T); T = T.dropna(subset=["A", "L"])   # arrays the chain withheld (no Met-A) carry no reading
    print(f"arrays with a reading: {len(T)} of {n0}"); R, V = score(T); R.to_csv(os.path.join(H, "selfline_series.csv"), index=False)
    json.dump(V, open(os.path.join(H, "selfline_verdict.json"), "w"), indent=1); print(R.round(4).to_string()); print(V)
