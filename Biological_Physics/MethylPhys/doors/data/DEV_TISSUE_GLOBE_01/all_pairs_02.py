"""DEV-TISSUE-GLOBE-01, run 2 (written and committed 2026-10-21 before it runs; development, exploratory, nothing sealed or scored).

The question (author: "run quite a few of any kind of cancer we have enough samples of to determine the behavior"): on the chain's globe in
pair mode (chain/MethylPhys_Interface/report_globe_v4.py build_pair, the fixed block rule), does a tumor against the same person's own adjacent
normal light more than the differences between two people, by which measure, and in which tissue?

Every sealed pair of the three tissue tests, all already scored and spent: colon (DEV-TISSUE-COLON-01, GSE131013 from GEO, 88), breast
(DEV-TISSUE-BREAST-01, TCGA-BRCA from the NCI GDC, 53), liver (DEV-TISSUE-LIVER-01, TCGA-LIHC from the NCI GDC, 50). Arrays calibrated by
chain Stage 1. Three kinds of comparison, each with the same rule:
  tumor    a person's tumor against their own adjacent normal
  control  one person's adjacent normal against another person's adjacent normal of the same organ (each normal against the next in its
           shard, so there are as many controls as pairs)
  healthy  colon only: healthy donors' mucosa (GSE131013's 48 people without cancer), one against the next
Every comparison is counted twice: all sites, and the X and Y left out (their sites from the pinned 450K manifest Stage 1 uses), so a
difference of sex is not counted as a change. Reported for each: changed blocks (and above / below), and the share of the globe's
measured pixels the change lights. Reported, not scored: per tissue the median and range of each kind, and the share of tumors above
the largest control.

Usage: all_pairs_02.py shard K N OUT      (rows for shard K of N -> OUT/rows_K.csv)
       all_pairs_02.py combine ROWSDIR OUT (-> OUT/all_pairs_02_summary.txt, all_pairs_02_rows.csv)"""
import os, sys, re, json, gzip, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H_, "..", "..", "..")); CH = os.path.join(MP, "chain"); DD = os.path.join(MP, "doors", "data")
def items():
    L = []
    for p in json.load(open(os.path.join(DD, "DEV_TISSUE_COLON_01", "sealed_colon_tissue.json")))["pairs"]:
        L.append(dict(tissue="colon", id=p["pid"], tumor=("geo", p["tumour_gsm"]), normal=("geo", p["normal_gsm"])))
    for t, d, f in (("breast", "DEV_TISSUE_BREAST_01", "tcga_brca_450k_files.csv"), ("liver", "DEV_TISSUE_LIVER_01", "tcga_lihc_450k_files.csv")):
        T = pd.read_csv(os.path.join(DD, d, f)); S = json.load(open(os.path.join(DD, d, f"sealed_{t}_tissue.json")))["pairs"]
        for p in S:
            nf = {("Grn" if "_Grn" in r.file_name else "Red"): dict(file_id=r.file_id, file_name=r.file_name) for r in T[T["sample"] == p["normal_sample"]].itertuples()}
            L.append(dict(tissue=t, id=p["case"], tumor=("gdc", {k: dict(file_id=v["file_id"], file_name=v["file_name"]) for k, v in p["tumour_files"].items()}), normal=("gdc", nf)))
    C = pd.read_csv(os.path.join(DD, "DEV_TISSUE_COLON_01", "colon_samples.csv"))
    for r in C[C.tissue == "Mucosa"].itertuples(): L.append(dict(tissue="colon_healthy", id=str(r.pid), tumor=None, normal=("geo", r.gsm)))
    return L
def shard(k, n, out):
    sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
    import report_globe_v4 as G
    from stage_1_idat_calibration import calibrate_idat_to_beta
    W = os.path.join(out, f"idat_{k}"); os.makedirs(W, exist_ok=True)
    def geo(g):
        u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"; got = {}
        for nm in sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))):
            p = os.path.join(W, nm); urllib.request.urlretrieve(u + nm, p); got["Grn" if "_Grn" in nm else "Red"] = p
        return got
    def gdc(files):
        got = {}
        for ch, f in files.items():
            p = os.path.join(W, f["file_name"]); urllib.request.urlretrieve(f"https://api.gdc.cancer.gov/data/{f['file_id']}", p); got[ch] = p
        return got
    def beta(src):
        f = geo(src[1]) if src[0] == "geo" else gdc(src[1]); o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False)
        b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
        for q in f.values(): os.remove(q)
        return b
    mine = [x for i, x in enumerate(items()) if i % n == k]; XY = None; rows = []; normals = {}
    def add(kind, tissue, a, b_, s, r):
        nonlocal XY
        if XY is None:
            m = os.path.join(os.path.expanduser("~"), ".methylprep_manifest_files", "HumanMethylation450k_15017482_v3.csv.gz")
            with gzip.open(m, "rt") as fh:
                lines = fh.read().split("\n")
            i0 = next(i for i, l in enumerate(lines) if l.startswith("IlmnID")); M = pd.read_csv(pd.io.common.StringIO("\n".join(lines[i0:])), usecols=["IlmnID", "CHR"], low_memory=False)
            XY = set(M.loc[M.CHR.astype(str).isin(["X", "Y"]), "IlmnID"]); print("X and Y sites", len(XY), flush=True)
        for drop, lab in ((None, "all"), (XY, "noXY")):
            g = G.build_pair(s, r, "normal", drop=drop)
            rows.append(dict(kind=kind, tissue=tissue, a=a, b=b_, sites=lab, blocks=g["blocks"], changed=g["blocks_changed"], up=g["blocks_up"], down=g["blocks_down"], lit=g["pixels_lit_share"]))
    for x in mine:
        try:
            nb = beta(x["normal"]); normals.setdefault(x["tissue"], []).append((x["id"], nb))
            if x["tumor"]: add("tumor", x["tissue"], x["id"], x["id"], beta(x["tumor"]), nb)
            print(x["tissue"], x["id"], flush=True)
        except Exception as e: rows.append(dict(kind="error", tissue=x["tissue"], a=x["id"], b=f"{type(e).__name__}: {e}"[:200])); print("ERR", x["id"], e, flush=True)
    for t, L in normals.items():
        if len(L) < 2: continue
        for i in range(len(L)):
            (a, s), (b_, r) = L[i], L[(i + 1) % len(L)]
            if len(L) == 2 and i == 1: break
            add("healthy" if t == "colon_healthy" else "control", t.replace("_healthy", ""), a, b_, s, r)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"rows_{k}.csv"), index=False)
def combine(d, out):
    import glob
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "**", "rows_*.csv"), recursive=True))]); R.to_csv(os.path.join(out, "all_pairs_02_rows.csv"), index=False)
    E = R[R.kind == "error"]; R = R[R.kind != "error"]; L = [f"comparisons {len(R) // 2} (each counted twice), errors {len(E)}"]
    for t in ("colon", "breast", "liver"):
        for lab in ("all", "noXY"):
            S = R[(R.tissue == t) & (R.sites == lab)]
            for m in ("changed", "lit"):
                q = lambda k: S.loc[S.kind == k, m]
                parts = [f"{k} n {len(q(k))} median {q(k).median():.4g} range {q(k).min():.4g}-{q(k).max():.4g}" for k in ("tumor", "control", "healthy") if len(q(k))]
                above = f"tumors above the largest control {(q('tumor') > q('control').max()).mean():.2f}" if len(q("control")) and len(q("tumor")) else ""
                L.append(f"{t} {lab} {m}: " + " | ".join(parts) + (" | " + above if above else ""))
            T_ = S[S.kind == "tumor"]
            if len(T_): L.append(f"{t} {lab} tumors: blocks above the normal {int(T_.up.sum())}, below {int(T_.down.sum())}")
    open(os.path.join(out, "all_pairs_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
if __name__ == "__main__":
    if sys.argv[1] == "shard": shard(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else: combine(sys.argv[2], sys.argv[3])
