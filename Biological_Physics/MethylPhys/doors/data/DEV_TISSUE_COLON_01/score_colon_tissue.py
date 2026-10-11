"""DEV-TISSUE-COLON-01 scorer (written 2026-10-11 before any tumour array is fetched or read).
seal POSTDIR BETADIR: non-tumour arrays + tumour METADATA only. Fixes (i) the composition-blind sites (colon_synthetic.py rule) as a list;
(ii) the per-pair bar = 95th percentile of R = Met-A(normal j) / Met-A(normal k) over stand-in pairs of two different adjacent normals on one slide;
(iii) the pairs: every patient with tumour and adjacent normal on the same slide, both arrays QC 'OK' in the submitters' Sentrix sheet
(colon_samples.csv; degraded or unrated arrays out); (iv) the group rule: FINGERPRINT if >= CRIT of the pairs read R above the bar, CRIT = smallest
count with binomial p < 0.01 at 0.05 per pair (colon_synthetic.py v2); else NO FINGERPRINT. Writes sealed_colon_tissue.json with this file's sha256.
score POSTDIR BETADIR TUMOURBETADIR: refuses unless sealed_colon_tissue.json is committed unchanged and this file's sha256 matches the seal.
R per pair = Met-A(tumour) / Met-A(own adjacent normal) at the sealed sites; verdict by the sealed rule. Writes scored_colon_tissue.json.
Synthetic tests: colon_synthetic.py (outputs colon_synthetic_output.csv, colon_synthetic_summary.txt; v1 failed, kept as colon_synthetic_v1_FAILED_*)."""
import os, sys, json, hashlib, subprocess, time, itertools, numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); SEAL = os.path.join(HERE, "sealed_colon_tissue.json"); ME = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
STEP, POST, BD = sys.argv[1], sys.argv[2], sys.argv[3]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
def beta(path): b = pd.read_parquet(path).beta; b.index = b.index.astype(str); return b
if STEP == "seal":
    assert not os.path.exists(SEAL), "a seal already exists; it is never overwritten"
    src = open(os.path.join(HERE, "colon_synthetic.py")).read(); sys.argv = [sys.argv[0], POST, BD]; exec(src[:src.index("out = [f\"arrays:")])
    bar = float(np.quantile([base[j] / base[k] for j, k in pairs], .95))
    A = pd.read_csv(os.path.join(HERE, "colon_samples.csv")); P = []
    for pid, g in A[A.tissue.isin(["Normal", "Tumor"])].groupby("pid"):
        n, t = g[g.tissue == "Normal"], g[g.tissue == "Tumor"]
        if len(n) != 1 or len(t) != 1: continue
        n, t = n.iloc[0], t.iloc[0]
        if n.quality_control_status != "OK" or t.quality_control_status != "OK" or n.id_sentrix != t.id_sentrix or n.gsm not in Bt: continue
        P.append(dict(pid=pid, slide=str(n.id_sentrix), plate=n.plate, normal_gsm=n.gsm, tumour_gsm=t.gsm, stage=t.stage))
    crit = int(binom.ppf(0.99, len(P), 0.05)) + 1
    S_ = dict(test="DEV-TISSUE-COLON-01", sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scorer_sha256=ME, sites_methylated=list(hi), sites_unmethylated=list(lo),
              bar_pair=bar, crit=crit, n_pairs=len(P), pairs=P,
              rule=f"FINGERPRINT if >= {crit} of {len(P)} pairs read Met-A(tumour)/Met-A(own adjacent normal, same slide) above {bar:.4f} at the sealed blind sites; else NO FINGERPRINT")
    json.dump(S_, open(SEAL, "w"), indent=1); print(S_["rule"], "| sites", len(hi), "+", len(lo))
elif STEP == "score":
    TB = sys.argv[4]; S_ = json.load(open(SEAL)); assert S_["scorer_sha256"] == ME, "scorer changed since the seal"
    assert subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", os.path.basename(SEAL)], capture_output=True).returncode == 0 and \
           subprocess.run(["git", "-C", HERE, "diff", "--quiet", "HEAD", "--", os.path.basename(SEAL)]).returncode == 0, "seal not committed unchanged"
    S = S_["sites_methylated"] + S_["sites_unmethylated"]; rows = []
    for p in S_["pairs"]:
        n = beta(os.path.join(BD, p["normal_gsm"] + ".parquet")).reindex(S); t = beta(os.path.join(TB, p["tumour_gsm"] + ".parquet")).reindex(S); ok = n.notna() & t.notna()
        rows.append(dict(pid=p["pid"], slide=p["slide"], plate=p["plate"], stage=p["stage"], sites=int(ok.sum()), R=float(H(t[ok].values).mean() / H(n[ok].values).mean())))
    R = pd.DataFrame(rows); k = int((R.R > S_["bar_pair"]).sum()) if len(R) else 0
    out = dict(test=S_["test"], pairs=len(R), above_bar=k, crit=S_["crit"], bar=S_["bar_pair"], R_median=float(R.R.median()) if len(R) else None,
               VERDICT="FINGERPRINT" if k >= S_["crit"] else "NO FINGERPRINT", rows=rows)
    json.dump(out, open(os.path.join(HERE, "scored_colon_tissue.json"), "w"), indent=1); print({a: b for a, b in out.items() if a != "rows"})
