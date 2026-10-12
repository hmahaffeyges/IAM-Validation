"""DEV-TISSUE-LUNG-04 scorer (written 2026-10-12 before any tumor array is fetched or read). Derived from DEV-TISSUE-LIVER-01 score_liver_tissue.py.
seal POSTDIR BETADIR STROMA_BETADIR (v3, 2026-10-12): normal arrays + tumor METADATA only. Fixes (i) the composition-blind sites (luad_synthetic.py rule) as a list;
(ii) the per-pair bar = 95th percentile of R = Met-A(normal j) / Met-A(normal k) over same-plate pairs of two different patients' normal lungs;
(iii) the pairs: every patient whose 'Primary Tumor' 450K aliquot shares a plate with the 'Solid Tissue Normal' (tcga_luad_450k_files.csv; the
first such tumor aliquot by barcode), with both file_ids; (iv) the group rule chosen by test 5 of luad_synthetic.py (read from luad_synthetic_summary.txt): COUNT = FINGERPRINT if >= CRIT pairs read R above
the bar (binomial p < 0.01 at 0.05 per pair); EVERY-PAIR = FINGERPRINT if the one-sided exact Wilcoxon signed-rank test of log R > 0 over all
pairs gives p < 0.01; else NO FINGERPRINT. Writes sealed_luad_tissue.json (the seal
fetch_tcga_luad.py and calib_tcga_luad.py check) with this file's sha256.
score POSTDIR BETADIR TUMORBETADIR: refuses unless sealed_luad_tissue.json is committed unchanged and this file's sha256 matches the seal.
Synthetic tests: luad_synthetic.py (outputs luad_synthetic_output.csv, luad_direction_output.csv, luad_synthetic_summary.txt)."""
import os, sys, json, hashlib, subprocess, time, numpy as np, pandas as pd
from scipy.stats import binom, wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); SEAL = os.path.join(HERE, "sealed_luad_tissue.json"); ME = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
STEP, POST, BD = sys.argv[1], sys.argv[2], sys.argv[3]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
def beta(path): b = pd.read_parquet(path).beta; b.index = b.index.astype(str); return b
if STEP == "seal":
    assert not os.path.exists(SEAL), "a seal already exists; it is never overwritten"
    src = open(os.path.join(HERE, "luad_synthetic.py")).read(); sys.argv = [sys.argv[0], POST, BD, sys.argv[4]]; exec(src[:src.index('out.append("2, 4, 5 cases:")')])
    RULE = [l.split("CHOSEN RULE: ")[1].strip() for l in open(os.path.join(HERE, "luad_synthetic_summary.txt")) if "CHOSEN RULE: " in l][0]; assert RULE in ("COUNT", "EVERY-PAIR"), RULE
    F = pd.read_csv(os.path.join(HERE, "tcga_luad_450k_files.csv")); F["ch"] = F.file_name.str.extract(r"_(Grn|Red)\.idat")[0]
    N = F[F.sample_type == "Solid Tissue Normal"]; Tm = F[F.sample_type == "Primary Tumor"]; P = []
    for case, n in N.groupby("case"):
        t = Tm[(Tm.case == case) & (Tm.plate == n.plate.iloc[0])].sort_values("aliquot")
        if len(t) == 0 or n["sample"].iloc[0] not in Bt: continue
        ta = t.aliquot.iloc[0]; tt = t[t.aliquot == ta]
        P.append(dict(case=case, plate=n.plate.iloc[0], normal_sample=n["sample"].iloc[0], tumour_sample=tt["sample"].iloc[0], tumour_aliquot=ta,
                      tumour_files={r.ch: dict(file_id=r.file_id, file_name=r.file_name, md5sum=r.md5sum) for _, r in tt.iterrows()}))
    crit_ = int(binom.ppf(0.99, len(P), 0.05)) + 1
    S_ = dict(test="DEV-TISSUE-LUNG-04", group_rule=RULE, sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scorer_sha256=ME, sites_methylated=list(hi), sites_unmethylated=list(lo),
              bar_pair=bar, crit=crit_, n_pairs=len(P), pairs=P,
              rule=(f"FINGERPRINT if >= {crit_} of {len(P)} pairs read Met-A(tumor)/Met-A(own normal lung, same plate) above {bar:.4f}" if RULE == "COUNT" else
                    f"FINGERPRINT if the one-sided exact Wilcoxon signed-rank test of log[Met-A(tumor)/Met-A(own normal lung, same plate)] > 0 over the {len(P)} pairs gives p < 0.01") + " at the sealed blind sites; else NO FINGERPRINT")
    json.dump(S_, open(SEAL, "w"), indent=1); print(S_["rule"], "| sites", len(hi), "+", len(lo))
elif STEP == "score":
    TB = sys.argv[4]; S_ = json.load(open(SEAL)); assert S_["scorer_sha256"] == ME, "scorer changed since the seal"
    assert subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", os.path.basename(SEAL)], capture_output=True).returncode == 0 and \
           subprocess.run(["git", "-C", HERE, "diff", "--quiet", "HEAD", "--", os.path.basename(SEAL)]).returncode == 0, "seal not committed unchanged"
    S = S_["sites_methylated"] + S_["sites_unmethylated"]; rows = []
    for p in S_["pairs"]:
        n = beta(os.path.join(BD, p["normal_sample"] + ".parquet")).reindex(S); t = beta(os.path.join(TB, p["tumour_sample"] + ".parquet")).reindex(S); ok = n.notna() & t.notna()
        rows.append(dict(case=p["case"], plate=p["plate"], sites=int(ok.sum()), R=float(H(t[ok].values).mean() / H(n[ok].values).mean())))
    R = pd.DataFrame(rows); k = int((R.R > S_["bar_pair"]).sum()); pw = float(wilcoxon(np.log(R.R.values), alternative="greater", method="exact").pvalue)
    hit = (k >= S_["crit"]) if S_["group_rule"] == "COUNT" else (pw < 0.01)
    out = dict(test=S_["test"], group_rule=S_["group_rule"], pairs=len(R), above_bar=k, crit=S_["crit"], bar=S_["bar_pair"], signed_rank_p=pw, R_median=float(R.R.median()), VERDICT="FINGERPRINT" if hit else "NO FINGERPRINT", rows=rows)
    json.dump(out, open(os.path.join(HERE, "scored_luad_tissue.json"), "w"), indent=1); print({a: b for a, b in out.items() if a != "rows"})
