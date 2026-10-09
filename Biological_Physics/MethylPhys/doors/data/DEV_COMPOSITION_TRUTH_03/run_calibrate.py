"""Download + Stage 1 calibrate the GSE224807 paired arrays (DEV-COMPOSITION-TRUTH-03). One parquet per array; resumable."""
import sys, os, warnings, urllib.request, pandas as pd, traceback
W = sys.argv[1]; CH = os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/chain"); sys.path.insert(0, CH)
OUT = os.path.join(W, "pair224807"); os.makedirs(os.path.join(OUT, "idat"), exist_ok=True); os.makedirs(os.path.join(OUT, "betas"), exist_ok=True)
def one(row):
    warnings.filterwarnings("ignore")
    gsm = row["gsm"]; dst = os.path.join(OUT, "betas", f"{gsm}.parquet")
    if os.path.exists(dst): return gsm, "done"
    try:
        p = {}
        for k in ("grn", "red"):
            f = os.path.join(OUT, "idat", row[k].rsplit("/", 1)[-1]); p[k] = f
            if not os.path.exists(f): urllib.request.urlretrieve(row[k], f + ".part"); os.replace(f + ".part", f)
        import stage_1_idat_calibration as S1
        o = S1.calibrate_idat_to_beta(p["grn"], p["red"], verbose=False); b = o[0] if isinstance(o, tuple) else o
        b.rename("beta").to_frame().to_parquet(dst)
        for f in p.values(): os.remove(f)
        return gsm, f"ok {len(b)}"
    except Exception as e:
        return gsm, "ERR " + repr(e)[:200]
if __name__ == "__main__":
    m = pd.read_csv(os.path.join(OUT, "manifest.csv")).sort_values(["gpl", "gsm"], ascending=[False, True]).reset_index(drop=True)
    k, n = int(sys.argv[2]), int(sys.argv[3])          # shard k of n (separate processes; the sandbox blocks process pools)
    for i, row in m.iloc[k::n].iterrows():
        g, st = one(row); print(i, g, st, flush=True)
    print("SHARD_DONE", k, flush=True)
