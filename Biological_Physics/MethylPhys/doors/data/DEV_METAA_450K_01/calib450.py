"""450K neutrophil commissioning, step 1: Stage 1 calibration of every array in the five series; betas to S3 as one parquet per series.
Run on the box from the chain directory. Usage: calib450.py WORKDIR NWORKERS"""
import os, sys, glob, tarfile, gzip, shutil, warnings, subprocess
from concurrent.futures import ProcessPoolExecutor
import pandas as pd, boto3
warnings.filterwarnings("ignore")
W = sys.argv[1]; NW = int(sys.argv[2]); CH = os.path.expanduser("~/IAM-Validation/Biological_Physics/MethylPhys/chain"); sys.path.insert(0, CH); os.chdir(CH)
B = "methylphys-data-945451304272-us-west-2-an"; s3 = boto3.client("s3", region_name="us-west-2")
SERIES = ["GSE88824", "GSE124565", "GSE65097", "GSE35069", "GSE318669"]
def cal(args):
    g, r = args
    import stage_1_idat_calibration as S1
    try:
        out = S1.calibrate_idat_to_beta(g, r, verbose=False); b = out[0] if isinstance(out, tuple) else out
        return os.path.basename(g).replace("_Grn.idat", ""), b.astype("float32")
    except Exception as e:
        return os.path.basename(g).replace("_Grn.idat", ""), str(e)[:200]
for ser in SERIES:
    D = os.path.join(W, ser); os.makedirs(D, exist_ok=True); key = f"results/K450_COMMISSION/betas_{ser}.parquet"
    try: s3.head_object(Bucket=B, Key=key); print(ser, "already in S3", flush=True); continue
    except Exception: pass
    stem = ser[:-3] + "nnn"
    subprocess.run(f"curl -sfL -o {D}/RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/{stem}/{ser}/suppl/{ser}_RAW.tar && tar xf {D}/RAW.tar -C {D} && rm {D}/RAW.tar", shell=True, check=True)
    for f in glob.glob(D + "/*.idat.gz"):
        with gzip.open(f, "rb") as a, open(f[:-3], "wb") as b: shutil.copyfileobj(a, b)
        os.remove(f)
    pairs = [(g, g.replace("_Grn.idat", "_Red.idat")) for g in sorted(glob.glob(D + "/*_Grn.idat")) if os.path.exists(g.replace("_Grn.idat", "_Red.idat"))]
    cols, errs = {}, {}
    with ProcessPoolExecutor(NW) as ex:
        for k, v in ex.map(cal, pairs):
            (errs if isinstance(v, str) else cols)[k] = v
    M = pd.DataFrame(cols); M.index = M.index.astype(str); p = os.path.join(W, f"betas_{ser}.parquet"); M.to_parquet(p)
    s3.upload_file(p, B, key); shutil.rmtree(D)
    print(ser, "arrays", M.shape[1], "probes", M.shape[0], "errors", len(errs), list(errs.items())[:2], flush=True)
print("CALIB_DONE")
