"""DEV-TISSUE-GLOBE-01, run 3 (written and committed 2026-10-21 before it runs; development, exploratory, nothing sealed or scored).

Run 2 found colon's person-to-person controls lighting nearly as much as its tumors (median 542 changed blocks against 917, X and Y left out),
while breast and liver controls lit about a third of their tumors. Each colon tumor sits on the same slide as its own normal; run 2's colon
controls compared people mostly on different slides. The question: matched on the slide, do colon's controls fall?
The same comparisons as run 2 (chain Stage 1, report_globe_v4.build_pair, the fixed block rule, X and Y left out and kept), now only within
a slide: on each of GSE131013's 22 slides, each person's adjacent normal against the next person's on the same slide, and each healthy
donor's mucosa against the next on the same slide. The tumors are run 2's (each already on its own normal's slide).
Usage: colon_slide_03.py shard K N OUT  |  colon_slide_03.py combine ROWSDIR RUN2_ROWS OUT"""
import os, sys, re, gzip, json, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H_, "..", "..", "..")); CH = os.path.join(MP, "chain"); DD = os.path.join(MP, "doors", "data")
def slides():
    C = pd.read_csv(os.path.join(DD, "DEV_TISSUE_COLON_01", "colon_samples.csv")); C = C[C.tissue.isin(["Normal", "Mucosa"])]
    P = {p["normal_gsm"] for p in json.load(open(os.path.join(DD, "DEV_TISSUE_COLON_01", "sealed_colon_tissue.json")))["pairs"]}
    C = C[(C.tissue == "Mucosa") | C.gsm.isin(P)]
    return [(s, g.sort_values("gsm")) for s, g in sorted(C.groupby("id_sentrix"), key=lambda x: str(x[0]))]
def shard(k, n, out):
    sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
    import report_globe_v4 as G
    from stage_1_idat_calibration import calibrate_idat_to_beta
    W = os.path.join(out, f"idat_{k}"); os.makedirs(W, exist_ok=True)
    def beta(g):
        u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"; got = {}
        for nm in sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))):
            p = os.path.join(W, nm); urllib.request.urlretrieve(u + nm, p); got["Grn" if "_Grn" in nm else "Red"] = p
        o = calibrate_idat_to_beta(got["Grn"], got["Red"], verbose=False); b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
        for q in got.values(): os.remove(q)
        return b
    m = os.path.join(os.path.expanduser("~"), ".methylprep_manifest_files", "HumanMethylation450k_15017482_v3.csv.gz"); XY = None; rows = []
    for i, (s, g) in enumerate(slides()):
        if i % n != k: continue
        for kind, tis in (("control", "Normal"), ("healthy", "Mucosa")):
            L = [(r.gsm, None) for r in g[g.tissue == tis].itertuples()]
            if len(L) < 2: continue
            B = {}
            for gsm, _ in L:
                try: B[gsm] = beta(gsm)
                except Exception as e: rows.append(dict(kind="error", tissue="colon", a=gsm, b=f"{type(e).__name__}: {e}"[:200]))
            ks = list(B)
            if XY is None and ks:
                t = gzip.open(m, "rt").read().split("\n"); i0 = next(i for i, l in enumerate(t) if l.startswith("IlmnID"))
                M = pd.read_csv(pd.io.common.StringIO("\n".join(t[i0:])), usecols=["IlmnID", "CHR"], low_memory=False); XY = set(M.loc[M.CHR.astype(str).isin(["X", "Y"]), "IlmnID"])
            for j in range(len(ks) if len(ks) > 2 else len(ks) - 1):
                a, b_ = ks[j], ks[(j + 1) % len(ks)]
                for drop, lab in ((None, "all"), (XY, "noXY")):
                    r = G.build_pair(B[a], B[b_], "normal", drop=drop)
                    rows.append(dict(kind=kind, tissue="colon", slide=str(s), a=a, b=b_, sites=lab, blocks=r["blocks"], changed=r["blocks_changed"], up=r["blocks_up"], down=r["blocks_down"], lit=r["pixels_lit_share"]))
            print(s, kind, len(ks), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"rows_{k}.csv"), index=False)
def combine(d, run2, out):
    import glob
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "**", "rows_*.csv"), recursive=True))]); R.to_csv(os.path.join(out, "colon_slide_03_rows.csv"), index=False)
    T = pd.read_csv(run2); T = T[(T.tissue == "colon") & (T.kind == "tumor")]; E = R[R.kind == "error"]; R = R[R.kind != "error"]
    L = [f"same-slide comparisons {len(R) // 2} (each counted twice), errors {len(E)}"]
    for lab in ("all", "noXY"):
        for mm in ("changed", "lit"):
            q = lambda k, lab=lab, mm=mm: (lambda F: F.loc[F.sites == lab, mm])(T if k == "tumor" else R[R.kind == k])
            L.append(f"colon {lab} {mm}: " + " | ".join(f"{k} n {len(q(k))} median {q(k).median():.4g} range {q(k).min():.4g}-{q(k).max():.4g}" for k in ("tumor", "control", "healthy") if len(q(k)))
                     + f" | tumors above the largest same-slide control {(q('tumor') > q('control').max()).mean():.2f}")
    open(os.path.join(out, "colon_slide_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
if __name__ == "__main__":
    if sys.argv[1] == "shard": shard(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else: combine(sys.argv[2], sys.argv[3], sys.argv[4])
