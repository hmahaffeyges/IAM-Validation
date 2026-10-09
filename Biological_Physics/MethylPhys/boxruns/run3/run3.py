"""Box Run 3 (boxruns/run3/JOBS.md): commissioned Met-A on GSE315366 (CH) and GSE315367 (leukaemia serial), EPIC v1 peripheral blood."""
import os, sys, glob, json, gzip, re, tarfile, warnings, subprocess
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, boto3
warnings.filterwarnings("ignore")
W = os.path.expanduser("~/run3"); CH = os.path.expanduser("~/IAM-Validation/Biological_Physics/MethylPhys/chain"); sys.path.insert(0, CH); os.chdir(CH)
B = "methylphys-data-945451304272-us-west-2-an"; s3 = boto3.client("s3", region_name="us-west-2")


def meta(mfile):
    L = [l.rstrip("\n").split("\t") for l in gzip.open(mfile, "rt", errors="replace") if l.startswith(("!Sample_geo_accession", "!Sample_characteristics_ch1", "!Sample_source_name_ch1", "!Sample_title"))]
    gs = [x.strip('"') for x in next(r for r in L if r[0] == "!Sample_geo_accession")[1:]]; d = pd.DataFrame({"gsm": gs})
    for r in L:
        if r[0] == "!Sample_source_name_ch1": d["src"] = [x.strip('"') for x in r[1:]]
        if r[0] == "!Sample_title": d["title"] = [x.strip('"') for x in r[1:]]
        if r[0] == "!Sample_characteristics_ch1":
            v = [x.strip('"') for x in r[1:]]; k = v[0].split(": ")[0]
            d[k] = [x.split(": ", 1)[1] if ": " in x else x for x in v]
    return d


def read_one(a):
    gsm, grn, red = a
    import stage_1_idat_calibration as S1, conductor_v3 as C
    try:
        b, _ = S1.calibrate_idat_to_beta(grn, red, verbose=False)
        o = C.run_neutrophil(b, specimen="whole blood", sample_id=gsm); m = o.get("met_a") or {}
        return dict(gsm=gsm, slide=os.path.basename(grn).split("_")[1], A=m.get("A"), f_neu=m.get("fraction"), N=m.get("noise_index"),
                    refusal=o.get("refusal_code") or m.get("refusal"), C=(o.get("met_a_cscore") or {}).get("C"), status="ok")
    except Exception as e:
        return dict(gsm=gsm, status=f"error: {type(e).__name__}: {str(e)[:150]}")


def tare(r, healthy):
    import conductor_v3 as C
    hh = r[healthy & r.A.notna()]; out = []
    for _, x in r.iterrows():
        if pd.isna(x.A): out.append((None, None, 0)); continue
        res = None
        for scope in ("slide", "series"):
            ref = hh[(hh.gsm != x.gsm) & ((hh.slide == x.slide) if scope == "slide" else True)]
            if len(ref) >= 3: res = (float(x.A / ref.A.median()), scope, len(ref)); break
        out.append(res or (None, None, 0))
    r["A_rel"], r["ref_scope"], r["n_refs"] = zip(*out)
    r["state"] = r.A_rel.map(lambda a: None if pd.isna(a) else ("Normal" if 0.95 <= a <= 1.05 else ("above Normal" if a > 1.05 else "below Normal")))
    return r


if __name__ == "__main__":
    os.makedirs(W, exist_ok=True); summ = {}
    for ser in ("GSE315366", "GSE315367"):
        D = os.path.join(W, ser); os.makedirs(D + "/idat", exist_ok=True); pre = f"downloads/A_blood_immune/{ser}/"
        for o in s3.list_objects_v2(Bucket=B, Prefix=pre)["Contents"]:
            k = o["Key"]; f = os.path.join(D, k.split("/")[-1])
            if k.endswith(("RAW.tar", "series_matrix.txt.gz")) and not os.path.exists(f): s3.download_file(B, k, f)
        md = meta(glob.glob(D + "/*series_matrix.txt.gz")[0])
        with tarfile.open(glob.glob(D + "/*RAW.tar")[0]) as t: t.extractall(D + "/idat")
        for p in glob.glob(D + "/idat/*.gz"): subprocess.run(["gunzip", "-f", p])
        tissue = md["tissue"].str.lower()
        md = md[tissue.str.contains("peripheral blood")]
        jobs = [(g, gr[0], rd[0]) for g in md.gsm for gr, rd in [(glob.glob(f"{D}/idat/{g}_*_Grn.idat"), glob.glob(f"{D}/idat/{g}_*_Red.idat"))] if gr and rd]
        print(ser, "peripheral bloods", len(md), "with IDATs", len(jobs), flush=True)
        with ProcessPoolExecutor(28) as ex: rows = list(ex.map(read_one, jobs))
        r = pd.DataFrame(rows).merge(md, on="gsm", how="left")
        if ser == "GSE315366":
            r = tare(r, r["clonal hematopoiesis"].eq("FALSE"))
        else:
            r["A_rel"] = None; r["state"] = None
        r.to_csv(os.path.join(W, f"{ser}_rows.csv"), index=False)
        s3.upload_file(os.path.join(W, f"{ser}_rows.csv"), B, f"results/BOXRUN3/{ser}_rows.csv")
        summ[ser] = dict(n=len(r), errors=int((r.status != "ok").sum()), refused=int(r.refusal.notna().sum()) if "refusal" in r else None)
        print(ser, summ[ser], flush=True)
    json.dump(summ, open(os.path.join(W, "summary.json"), "w")); s3.upload_file(os.path.join(W, "summary.json"), B, "results/BOXRUN3/summary.json")
    print("DONE")
