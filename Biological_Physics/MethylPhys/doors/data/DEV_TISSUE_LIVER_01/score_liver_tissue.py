"""DEV-TISSUE-LIVER-01 scorer (written 2026-10-12 before any tumor array is fetched or read). Derived from DEV-TISSUE-BREAST-01 and -LUNG-01.
seal POSTDIR BETADIR: normal arrays + tumor METADATA only. Fixes (i) the composition-blind sites (liver_synthetic.py v2 rule) as a list;
(ii) the per-pair bar = 95th percentile of R = Met-A(normal j) / Met-A(normal k) over same-plate pairs of two different patients' normal livers;
(iii) the pairs: every patient whose 'Primary Tumor' 450K aliquot shares a plate with the 'Solid Tissue Normal' (tcga_lihc_450k_files.csv; the
first such tumor aliquot by barcode), with both file_ids; (iv) the group rule chosen by test 5 (COUNT): FINGERPRINT if >= CRIT of the pairs read R
above the bar, CRIT = smallest count with binomial p < 0.01 at 0.05 per pair; else NO FINGERPRINT. Writes sealed_liver_tissue.json (the seal
fetch_tcga_lihc.py and calib_tcga_lihc.py check) with this file's sha256.
score POSTDIR BETADIR TUMORBETADIR: refuses unless sealed_liver_tissue.json is committed unchanged and this file's sha256 matches the seal.
Synthetic tests: liver_synthetic.py (v2; v1 kept as liver_synthetic_v1_FAILED_*), liver_direction_07.py (test 7, replaces the independence bar)."""
import os, sys, json, hashlib, subprocess, time, numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); SEAL = os.path.join(HERE, "sealed_liver_tissue.json"); ME = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
STEP, POST, BD = sys.argv[1], sys.argv[2], sys.argv[3]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
def beta(path): b = pd.read_parquet(path).beta; b.index = b.index.astype(str); return b
if STEP == "seal":
    assert not os.path.exists(SEAL), "a seal already exists; it is never overwritten"
    src = open(os.path.join(HERE, "liver_synthetic.py")).read(); sys.argv = [sys.argv[0], POST, BD]; exec(src[:src.index('out.append(f"2-6 cases')])
    F = pd.read_csv(os.path.join(HERE, "tcga_lihc_450k_files.csv")); F["ch"] = F.file_name.str.extract(r"_(Grn|Red)\.idat")[0]
    N = F[F.sample_type == "Solid Tissue Normal"]; Tm = F[F.sample_type == "Primary Tumor"]; P = []
    for case, n in N.groupby("case"):
        t = Tm[(Tm.case == case) & (Tm.plate == n.plate.iloc[0])].sort_values("aliquot")
        if len(t) == 0 or n["sample"].iloc[0] not in Bt: continue
        ta = t.aliquot.iloc[0]; tt = t[t.aliquot == ta]
        P.append(dict(case=case, plate=n.plate.iloc[0], normal_sample=n["sample"].iloc[0], tumour_sample=tt["sample"].iloc[0], tumour_aliquot=ta,
                      tumour_files={r.ch: dict(file_id=r.file_id, file_name=r.file_name, md5sum=r.md5sum) for _, r in tt.iterrows()}))
    crit_ = int(binom.ppf(0.99, len(P), 0.05)) + 1
    S_ = dict(test="DEV-TISSUE-LIVER-01", sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scorer_sha256=ME, sites_methylated=list(hi), sites_unmethylated=list(lo),
              bar_pair=bar, crit=crit_, n_pairs=len(P), pairs=P,
              rule=f"FINGERPRINT if >= {crit_} of {len(P)} pairs read Met-A(tumor)/Met-A(own normal liver, same plate) above {bar:.4f} at the sealed blind sites; else NO FINGERPRINT")
    json.dump(S_, open(SEAL, "w"), indent=1); print(S_["rule"], "| sites", len(hi), "+", len(lo))
elif STEP == "score":
    TB = sys.argv[4]; S_ = json.load(open(SEAL)); assert S_["scorer_sha256"] == ME, "scorer changed since the seal"
    assert subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", os.path.basename(SEAL)], capture_output=True).returncode == 0 and \
           subprocess.run(["git", "-C", HERE, "diff", "--quiet", "HEAD", "--", os.path.basename(SEAL)]).returncode == 0, "seal not committed unchanged"
    S = S_["sites_methylated"] + S_["sites_unmethylated"]; rows = []
    for p in S_["pairs"]:
        n = beta(os.path.join(BD, p["normal_sample"] + ".parquet")).reindex(S); t = beta(os.path.join(TB, p["tumour_sample"] + ".parquet")).reindex(S); ok = n.notna() & t.notna()
        rows.append(dict(case=p["case"], plate=p["plate"], sites=int(ok.sum()), R=float(H(t[ok].values).mean() / H(n[ok].values).mean())))
    R = pd.DataFrame(rows); k = int((R.R > S_["bar_pair"]).sum())
    out = dict(test=S_["test"], pairs=len(R), above_bar=k, crit=S_["crit"], bar=S_["bar_pair"], R_median=float(R.R.median()), VERDICT="FINGERPRINT" if k >= S_["crit"] else "NO FINGERPRINT", rows=rows)
    json.dump(out, open(os.path.join(HERE, "scored_liver_tissue.json"), "w"), indent=1); print({a: b for a, b in out.items() if a != "rows"})
