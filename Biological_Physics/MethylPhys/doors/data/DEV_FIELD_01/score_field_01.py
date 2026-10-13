"""DEV-FIELD-01 scorer (written 2026-10-13 before any GSE287331 array is downloaded or read).

seal WORKDIR : METADATA ONLY (gse287331_samples.csv from field_meta_01.py; the EPIC v1 manifest; the sealed breast site list). Fixes
  - sites: the 6,000 composition-blind breast sites of DEV-TISSUE-BREAST-01 (sealed_breast_tissue.json) that exist on EPIC v1;
  - F1/F2 pairs: every patient with contralateral (CUB), other-quadrant (OQ) and tumor (TU) arrays all on ONE slide;
  - F3 pairs: every other patient with all three;
  - the critical counts, from the one-sided sign test at p < 0.05.
  Writes sealed_field_01.json with this file's sha256.
score BETADIR : refuses unless sealed_field_01.json is committed unchanged and this file's sha256 matches the seal.
Reading: M = mean binary entropy at the sealed sites present in both arrays compared (the sealed DEV-TISSUE-BREAST-01 reading).
  F1  R = M(OQ) / M(CUB) per F1 patient; MET if R > 1 in >= crit_F1 patients.
  F2  R = M(TU) / M(OQ) per F1 patient; MET if R > 1 in >= crit_F1 patients (positive control).
  F3  per F3 patient, each array's M divided by the median M of all CUB and OQ arrays on its own slide, then R = OQ / CUB;
      MET if R > 1 in >= crit_F3 patients.
  F4  descriptive, not decided: median M of every tissue, and the spread of slide medians over the healthy-donor (HDB) slides.
  F5  descriptive, not decided (amendment 2026-10-13, before any reading): each woman's other breast against the healthy-donor slide medians.
  Verdict labels (amended before any reading): F1 not met with F2 met reads "no difference between a woman's two breasts", NOT a break -
  a field over both breasts would give exactly that.
Planted test: planted_field_01.py (P1 null 3.6 %, P2 a 1 % loss found in 100 % of draws); slide positions: field_positions_01.py.
"""
import gzip, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); ME = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
SEAL = os.path.join(HERE, "sealed_field_01.json")
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
crit = lambda n: int(next(k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) < 0.05))
def M(b, sites):
    v = b.reindex(sites).values; v = v[np.isfinite(v)]; return float(H(v).mean()), int(len(v))
mode = sys.argv[1]
if mode == "seal":
    W = sys.argv[2]
    T = pd.read_csv(os.path.join(HERE, "gse287331_samples.csv"), dtype=str)
    S0 = json.load(open(os.path.join(HERE, "..", "DEV_TISSUE_BREAST_01", "sealed_breast_tissue.json")))
    man = os.path.join(W, "mphome", ".methylprep_manifest_files", "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz")
    epic = set(pd.read_csv(man, usecols=["IlmnID"], dtype=str).IlmnID)
    sites = [s for s in S0["sites_methylated"] + S0["sites_unmethylated"] if s in epic]
    f1, f3 = [], []
    for pid, g in T[T.tissue.isin(["CUB", "OQ", "TU"])].groupby("patient"):
        if not {"CUB", "OQ", "TU"} <= set(g.tissue): continue
        x = g.drop_duplicates("tissue").set_index("tissue")
        rec = dict(patient=pid, **{t: x.loc[t, "gsm"] for t in ("CUB", "OQ", "TU")}, **{t + "_slide": x.loc[t, "slide"] for t in ("CUB", "OQ", "TU")})
        (f1 if g.slide.nunique() == 1 else f3).append(rec)
    J = dict(test="DEV-FIELD-01", sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scorer_sha256=ME, series="GSE287331",
             sites=sites, n_sites=len(sites), pairs_F1=f1, pairs_F3=f3, crit_F1=crit(len(f1)), crit_F3=crit(len(f3)),
             tare_F3="median M of all CUB and OQ arrays on the array's own slide", amended="2026-10-13, after calibration began and before any reading: verdict label for F1-not-met changed from BROKEN to not-ruled-out; F5 descriptive added; bars unchanged (author)",
             rule=("F1 MET if M(OQ)/M(CUB) > 1 in >= crit_F1 of the F1 patients; F2 MET if M(TU)/M(OQ) > 1 in >= crit_F1 of them; F3 MET if the "
                   "slide-tared OQ/CUB ratio > 1 in >= crit_F3 of the F3 patients; F4 and F5 descriptive only. F1 not met with F2 met means no "
                   "difference between a woman's two breasts; it does NOT rule out a field covering both breasts or the whole body."))
    json.dump(J, open(SEAL, "w"), indent=1); print(f"sealed: {len(sites)} sites, F1 {len(f1)} patients (crit {J['crit_F1']}), F3 {len(f3)} (crit {J['crit_F3']})")
elif mode == "score":
    BD = sys.argv[2]; J = json.load(open(SEAL))
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", SEAL], cwd=HERE).returncode == 0
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", SEAL], cwd=HERE, capture_output=True).returncode == 0
    if not (clean and tracked and J["scorer_sha256"] == ME): sys.exit("REFUSED: seal not committed unchanged, or this scorer differs from the sealed one")
    sites = J["sites"]; cache = {}
    def m(g):
        if g not in cache:
            b = pd.read_parquet(os.path.join(BD, g + ".parquet")); cache[g] = (b["beta"] if "beta" in b.columns else b.iloc[:, 0])
        return M(cache[g], sites)[0]
    r1 = [m(p["OQ"]) / m(p["CUB"]) for p in J["pairs_F1"]]; r2 = [m(p["TU"]) / m(p["OQ"]) for p in J["pairs_F1"]]
    T = pd.read_csv(os.path.join(HERE, "gse287331_samples.csv"), dtype=str); T = T[[os.path.isfile(os.path.join(BD, g + ".parquet")) for g in T.gsm]]
    T["M"] = [m(g) for g in T.gsm]
    tare = T[T.tissue.isin(["CUB", "OQ"])].groupby("slide").M.median()
    r3 = [(T.set_index("gsm").M[p["OQ"]] / tare[p["OQ_slide"]]) / (T.set_index("gsm").M[p["CUB"]] / tare[p["CUB_slide"]]) for p in J["pairs_F3"]]
    k1, k2, k3 = sum(x > 1 for x in r1), sum(x > 1 for x in r2), sum(x > 1 for x in r3)
    hdb = T[T.tissue == "HDB"].groupby("slide").M.median()
    O = dict(F1=dict(k=k1, n=len(r1), crit=J["crit_F1"], verdict="MET" if k1 >= J["crit_F1"] else "NOT MET", R_median=float(np.median(r1))),
             F2=dict(k=k2, n=len(r2), crit=J["crit_F1"], verdict="MET" if k2 >= J["crit_F1"] else "NOT MET", R_median=float(np.median(r2))),
             F3=dict(k=k3, n=len(r3), crit=J["crit_F3"], verdict="MET" if k3 >= J["crit_F3"] else "NOT MET", R_median=float(np.median(r3))),
             F4=dict(median_M={t: float(g.M.median()) for t, g in T.groupby("tissue")}, n={t: int(len(g)) for t, g in T.groupby("tissue")},
                     hdb_slide_medians=dict(n=int(len(hdb)), min=float(hdb.min()), max=float(hdb.max()))),
             arrays_read=int(len(T)), scored_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # AMENDED 2026-10-13 before any reading (author: both breasts are one system and may both read high). F1 compares a woman's two
    # breasts, so a field reaching BOTH breasts makes F1 flat; F1 not met is therefore NOT a break of the picture. F5 places each
    # woman's other breast (CUB) against the spread of healthy-donor slide medians - descriptive, slide-confounded, never decided.
    cub = T[T.tissue == "CUB"].M
    O["F5_descriptive"] = dict(cub_median=float(cub.median()), hdb_slide_median_min=float(hdb.min()), hdb_slide_median_max=float(hdb.max()),
                               cub_arrays_above_every_hdb_slide_median=int((cub > hdb.max()).sum()), cub_arrays=int(len(cub)),
                               note="slide-confounded: healthy donors sit on slides of their own; a hint only")
    if O["F2"]["verdict"] != "MET": O["picture"] = "UNDECIDED (positive control not met)"
    elif O["F1"]["verdict"] == "MET": O["picture"] = "a field inside the breast with the cancer, beyond her other breast"
    else: O["picture"] = "no difference between a woman's two breasts; a field covering both breasts or the whole body is NOT ruled out (see F5)"
    pd.DataFrame(dict(patient=[p["patient"] for p in J["pairs_F1"]], R_F1=r1, R_F2=r2)).to_csv(os.path.join(HERE, "scored_field_01_F1F2.csv"), index=False)
    pd.DataFrame(dict(patient=[p["patient"] for p in J["pairs_F3"]], R_F3=r3)).to_csv(os.path.join(HERE, "scored_field_01_F3.csv"), index=False)
    T.to_csv(os.path.join(HERE, "scored_field_01_arrays.csv"), index=False)
    json.dump(O, open(os.path.join(HERE, "scored_field_01.json"), "w"), indent=1, default=int); print(json.dumps(O, indent=1, default=int))
