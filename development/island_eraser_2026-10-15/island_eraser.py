"""Whiteboard part 7 (2026-10-15): island gain error with the eraser removed, read once after a synthetic check.
Counter: PROC-CHANNEL-01's rules unchanged (reads with >= 6 covered CpGs; <= 20 % methylated: isolated C between T's per CpG = island
gain; >= 80 %: isolated T between C's per CpG = loss). Molecules: DEV-ERASE-01 primary units U1-U5, our S3 results/DEV_ERASE_01/reads/.
Usage: python island_eraser.py synthetic | real <workdir>"""
import os, sys, gzip, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
def count(lines):
    iso_C = nC_u = iso_T = nC_m = 0
    for line in lines:
        p = line.split("\t")
        if len(p) < 4: continue
        s = [c for c in p[2] if c != "."]; cnt = int(p[3])
        if len(s) < 6: continue
        f = s.count("C") / len(s)
        if f <= 0.2: nC_u += cnt * len(s); iso_C += cnt * sum(1 for i in range(1, len(s) - 1) if s[i] == "C" and s[i-1] == "T" and s[i+1] == "T")
        elif f >= 0.8: nC_m += cnt * len(s); iso_T += cnt * sum(1 for i in range(1, len(s) - 1) if s[i] == "T" and s[i-1] == "C" and s[i+1] == "C")
    return dict(gain=iso_C / max(nC_u, 1), loss=iso_T / max(nC_m, 1), n_u=nC_u, n_m=nC_m)
def planted(g, l, N=200_000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for k in range(N):
        meth = k % 2 == 0; fl = rng.random(10) < (l if meth else g)
        out.append("chr1\t%d\t%s\t1" % (1000 + k, "".join(("T" if f else "C") if meth else ("C" if f else "T") for f in fl)))
    return out
if sys.argv[1] == "synthetic":
    L = []
    for g, l in ((0.0124, 0.0212), (0.0212, 0.0212), (0.0062, 0.04)):
        r = count(planted(g, l)); L.append(f"planted gain {g} loss {l}: counted gain {r['gain']:.5f} (expected {g*(1-g)**2*0.8:.5f}), loss {r['loss']:.5f} (expected {l*(1-l)**2*0.8:.5f})")
    a = count(planted(0.0124, 0.0212, seed=1)); b = count(planted(0.0124 * 1.7, 0.0212, seed=2)); L.append(f"planted ratio 1.70 -> counted {b['gain']/a['gain']:.3f}")
    open(os.path.join(H, "island_eraser_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
else:
    import boto3; W = sys.argv[2]; s3 = boto3.client("s3"); B = "methylphys-data-945451304272-us-west-2-an"
    S = pd.read_csv(os.path.join(H, "..", "..", "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ERASE_01", "erase_samples.csv")).drop_duplicates("gsm")
    rows = []
    for _, r in S[S.unit.isin(["U1", "U2", "U3", "U4", "U5"])].iterrows():
        f = os.path.join(W, r.gsm + ".pat.gz")
        if not os.path.exists(f): s3.download_file(B, f"results/DEV_ERASE_01/reads/{r.gsm}.pat.gz", f)
        with gzip.open(f, "rt") as fh: rows.append(dict(unit=r.unit, gse=r.gse, gsm=r.gsm, genotype=r.genotype, **count(fh)))
    pd.DataFrame(rows).to_csv(os.path.join(H, "island_counts.csv"), index=False)
