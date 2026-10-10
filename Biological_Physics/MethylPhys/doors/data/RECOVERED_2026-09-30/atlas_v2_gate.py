# Atlas v2 acceptance gates on the WHOLE atlas (all 700 blocks from S3), 2026-09-29.
# Each box's end-of-run summary only saw its own blocks, so the gate is run here over everything.
#   V1  completeness: 700 blocks x 3 files present, SHA-256 of every file into ATLAS_V2_OUTPUT_MANIFEST.json
#   V2  distinctness (flatness lesson): every cell pair, mean |mu_a - mu_b| over shared MEASURED loci (n_obs>0, >=200 loci);
#       FAIL if any pair < 0.005. Computed on every 7th block (100 blocks), as pre-written in run_all.sh.
#   V3  coverage: per cell, fraction of loci measured; unmeasured must be NaN (no filled values)
#   V4  convergence: R-hat < 1.01 and ESS > 400 per (cell, locus) on measured pairs; worst block listed
import os, json, hashlib, itertools, numpy as np, pandas as pd, concurrent.futures as cf
B = "methylphys-data-945451304272-us-west-2-an"; OUT = "atlas_v2_blocks"; os.makedirs(OUT, exist_ok=True)
# Two steps, as actually run on 2026-09-29: fetch_and_manifest() (needs boto3 + AWS credentials; ran in the python env) downloads
# the 700 block parquets and writes ATLAS_V2_OUTPUT_MANIFEST.json; run() (needs pandas + pyarrow; ran in the methylprep env)
# computes V2-V4 from the local files and wrote atlas_v2_gate.json and atlas_v2_distinctness.csv.
s3 = None
def _s3():
    global s3
    if s3 is None:
        import boto3; s3 = boto3.client("s3", region_name="us-west-2")
    return s3

def fetch(b):
    got = {}
    for suf in (".parquet", "_draws.npz", "_prior.parquet"):
        f = f"block_{b:05d}{suf}"; p = f"{OUT}/{f}"
        if suf != ".parquet":            # draws and priors are checksummed from S3 without keeping a local copy
            obj = _s3().get_object(Bucket=B, Key=f"atlas_v2/blocks_v2/{f}"); data = obj["Body"].read()
            got[f] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()); continue
        if not os.path.exists(p): _s3().download_file(B, f"atlas_v2/blocks_v2/{f}", p)
        got[f] = dict(bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
    return got

def fetch_and_manifest():
    have = {o["Key"].split("/")[-1] for page in _s3().get_paginator("list_objects_v2").paginate(Bucket=B, Prefix="atlas_v2/blocks_v2/") for o in page.get("Contents", [])}
    missing = [f"block_{b:05d}{s}" for b in range(700) for s in (".parquet", "_draws.npz", "_prior.parquet") if f"block_{b:05d}{s}" not in have]
    R = {"V1_missing_files": missing[:20], "V1_n_missing": len(missing)}
    if missing: return R
    man = {}
    with cf.ThreadPoolExecutor(16) as ex:
        for d in ex.map(fetch, range(700)): man.update(d)
    json.dump(man, open("ATLAS_V2_OUTPUT_MANIFEST.json", "w"), indent=0); R["V1_files"] = len(man)
    return R

def run():
    man=json.load(open("ATLAS_V2_OUTPUT_MANIFEST.json")); R={"V1_files":len(man),"V1_blocks_local":len([f for f in os.listdir(OUT) if f.endswith(".parquet") and "_prior" not in f])}
    # V3 + V4 over all blocks, streamed
    cov_n = {}; cov_m = {}; filled = 0; rh_bad = 0; es_bad = 0; npairs = 0; worst = []
    for b in range(700):
        D = pd.read_parquet(f"{OUT}/block_{b:05d}.parquet", columns=["cell", "cpg", "mu", "n_obs", "rhat", "ess"])
        m = D.n_obs > 0
        filled += int(D.loc[~m, "mu"].notna().sum())
        g = D.groupby("cell").n_obs
        for c, s in g: cov_n[c] = cov_n.get(c, 0) + len(s); cov_m[c] = cov_m.get(c, 0) + int((s > 0).sum())
        M = D[m]; npairs += len(M); rb = int((M.rhat >= 1.01).sum()); eb = int((M.ess <= 400).sum()); rh_bad += rb; es_bad += eb
        worst.append((b, float(np.nanpercentile(M.rhat, 99)), float(np.nanpercentile(M.ess, 1)), rb))
    worst.sort(key=lambda x: -x[1])
    R["V3_filled_values_where_unmeasured"] = filled
    R["V3_measured_fraction_by_cell"] = {c: round(cov_m[c] / cov_n[c], 4) for c in sorted(cov_n)}
    R["V4"] = dict(measured_pairs=npairs, rhat_ge_1_01=rh_bad, ess_le_400=es_bad, frac_rhat_bad=rh_bad / npairs, frac_ess_bad=es_bad / npairs,
                   worst_blocks_by_rhat_p99=[dict(block=b, rhat_p99=round(r, 4), ess_p01=round(e, 1), n_rhat_bad=n) for b, r, e, n in worst[:8]])
    # V2 distinctness on every 7th block
    F = [f"{OUT}/block_{b:05d}.parquet" for b in range(0, 700, 7)]
    D = pd.concat([pd.read_parquet(f, columns=["cell", "cpg", "mu", "n_obs"]) for f in F]); D = D[D.n_obs > 0]
    M = D.pivot(index="cpg", columns="cell", values="mu"); res = []
    for a, b in itertools.combinations(M.columns, 2):
        k = M[a].notna() & M[b].notna()
        if k.sum() >= 200: res.append((a, b, float((M[a][k] - M[b][k]).abs().mean()), float(np.corrcoef(M[a][k], M[b][k])[0, 1]), int(k.sum())))
    T = pd.DataFrame(res, columns=["a", "b", "mean_abs_diff", "r", "n_loci"]).sort_values("mean_abs_diff"); T.to_csv("atlas_v2_distinctness.csv", index=False)
    R["V2"] = dict(pairs=len(T), loci=int(M.shape[0]), cells=int(M.shape[1]), near_identical_lt_0_005=int((T.mean_abs_diff < 0.005).sum()),
                   median=round(float(T.mean_abs_diff.median()), 4), closest=T.head(8).round(4).to_dict("records"), PASS=bool((T.mean_abs_diff >= 0.005).all()))
    json.dump(R, open("atlas_v2_gate.json", "w"), indent=1)
    return R
