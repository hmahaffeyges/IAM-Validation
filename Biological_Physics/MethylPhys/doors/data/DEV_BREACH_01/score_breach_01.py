"""DEV-BREACH-01 scorer (written 2026-10-13 before any GSE281307 array is downloaded or read).

seal : METADATA ONLY (gse281307_samples.csv from breach_meta_01.py; the sealed breast site list). Fixes the arrays (DCIS lesions untreated
  by radiation AND hormones, classified Progressor or Non-progressor), the 6,000 composition-blind sites of DEV-TISSUE-BREAST-01, and the
  slide split for B2 (slides shuffled with seed 20261013, first half A). Writes sealed_breach_01.json with this file's sha256.
score BETADIR : refuses unless sealed_breach_01.json is committed unchanged and this file's sha256 matches.
Reading: M = mean binary entropy at the sealed sites; r = M / median M of the sealed arrays on the same slide.
  B1 MET if progressors' r > non-progressors' r (one-sided Mann-Whitney, p < 0.05).
  B2 MET if the line t that maximizes (share of progressors with r >= t) - (share of non-progressors with r >= t) on slide half A puts more
     progressors than non-progressors at or above it on half B (one-sided Fisher, p < 0.05). The line is reported only if B2 is MET.
  The breach is NOT VISIBLE at this stage on this reading if B1 is not met.
Planted test: planted_breach_01.py (null 5.0 %; a 1 % loss meets B1 in 100 %, B2 in 96 %).
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); ME = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
SEAL = os.path.join(HERE, "sealed_breach_01.json")
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
mode = sys.argv[1]
if mode == "seal":
    T = pd.read_csv(os.path.join(HERE, "gse281307_samples.csv"), dtype=str)
    U = T[(T.radiationtreatment == "Untreated") & (T.hormonetreatment == "Untreated") & T.classification.isin(["Progressor", "Non-progressor"])]
    S0 = json.load(open(os.path.join(HERE, "..", "DEV_TISSUE_BREAST_01", "sealed_breast_tissue.json")))
    sl = sorted(U.slide.unique()); rg = np.random.default_rng(20261013); sh = list(rg.permutation(sl)); A = sorted(sh[:len(sh) // 2])
    J = dict(test="DEV-BREACH-01", sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scorer_sha256=ME, series="GSE281307",
             sites=S0["sites_methylated"] + S0["sites_unmethylated"],
             arrays=[dict(gsm=r.gsm, slide=r.slide, outcome=r.classification) for r in U.itertuples()],
             n=U.classification.value_counts().to_dict(), slides_A=A, slides_B=sorted(sh[len(sh) // 2:]),
             rule="B1 one-sided Mann-Whitney p < 0.05 on slide-tared r; B2 Youden line on half A, one-sided Fisher p < 0.05 on half B; line reported only if B2 MET.")
    json.dump(J, open(SEAL, "w"), indent=1); print("sealed", J["n"], "slides A", len(A), "B", len(J["slides_B"]))
elif mode == "score":
    BD = sys.argv[2]; J = json.load(open(SEAL))
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", SEAL], cwd=HERE).returncode == 0
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", SEAL], cwd=HERE, capture_output=True).returncode == 0
    if not (clean and tracked and J["scorer_sha256"] == ME): sys.exit("REFUSED: seal not committed unchanged, or this scorer differs from the sealed one")
    A = pd.DataFrame(J["arrays"])
    def M(g):
        b = pd.read_parquet(os.path.join(BD, g + ".parquet")); b = (b["beta"] if "beta" in b.columns else b.iloc[:, 0]).reindex(J["sites"]).values
        b = b[np.isfinite(b)]; return float(H(b).mean())
    A["M"] = [M(g) for g in A.gsm]; A["r"] = A.M / A.groupby("slide").M.transform("median"); A["prog"] = A.outcome == "Progressor"
    p1 = float(mannwhitneyu(A.r[A.prog], A.r[~A.prog], alternative="greater").pvalue)
    a, b = A[A.slide.isin(J["slides_A"])], A[A.slide.isin(J["slides_B"])]
    th = float(max(np.unique(a.r), key=lambda t: (a.r[a.prog] >= t).mean() - (a.r[~a.prog] >= t).mean()))
    tab = [[int((b.r[b.prog] >= th).sum()), int((b.r[b.prog] < th).sum())], [int((b.r[~b.prog] >= th).sum()), int((b.r[~b.prog] < th).sum())]]
    p2 = float(fisher_exact(tab, alternative="greater")[1])
    O = dict(B1=dict(p=p1, median_r_progressor=float(A.r[A.prog].median()), median_r_non=float(A.r[~A.prog].median()), verdict="MET" if p1 < 0.05 else "NOT MET"),
             B2=dict(p=p2, table_B=tab, verdict="MET" if p2 < 0.05 else "NOT MET"), line=(th if p2 < 0.05 else None),
             breach="NOT VISIBLE at DCIS on this reading" if p1 >= 0.05 else "visible", n=J["n"], scored_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    A.to_csv(os.path.join(HERE, "scored_breach_01_arrays.csv"), index=False)
    json.dump(O, open(os.path.join(HERE, "scored_breach_01.json"), "w"), indent=1); print(json.dumps(O, indent=1))
