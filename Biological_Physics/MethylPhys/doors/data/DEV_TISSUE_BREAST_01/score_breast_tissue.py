"""DEV-TISSUE-BREAST-01 scorer (written 2026-10-11 before any tumour array is downloaded or read).
seal POSTDIR BETADIR TEMPLATE: normal arrays + tumour METADATA only. Fixes (i) the composition-blind sites (blind_sites_test.py rule) as a list,
(ii) the per-pair bar = 95th percentile of R = Met-A(normal j) / Met-A(normal k) over same-plate normal pairs (power_pairs.py NULL), (iii) the group
rule: FINGERPRINT if >= CRIT of the pairs read R above the bar, CRIT = smallest count with binomial p < 0.05 at 0.05 per pair; NO FINGERPRINT if
fewer than CRIT; (iv) the pairs: every patient whose 'Primary Tumor' 450K aliquot shares a plate with its 'Solid Tissue Normal' (GDC aliquot
barcodes; the first such tumour aliquot by barcode), with both file_ids. Writes sealed_breast_tissue.json (with this file's sha256).
score POSTDIR BETADIR TUMOURBETADIR: refuses unless sealed_breast_tissue.json is committed unchanged and this file's sha256 matches the seal.
R per pair = Met-A(tumour) / Met-A(own normal) at the sealed sites; verdict by the sealed rule. Writes scored_breast_tissue.json.
Synthetic tests: development/sims/breast_tissue_01.py, _02.py; null_normals.py, plate_tare_check.py, plant_real_normals.py, blind_sites_test.py,
blind_vs_composition.py, power_pairs.py (this folder)."""
import os, sys, json, hashlib, subprocess, time, requests, numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); SEAL = os.path.join(HERE, "sealed_breast_tissue.json"); ME = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
STEP = sys.argv[1]; POST, BD = sys.argv[2], sys.argv[3]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
def beta(path): b = pd.read_parquet(path).beta; b.index = b.index.astype(str); return b
if STEP == "seal":
    assert not os.path.exists(SEAL), "a seal already exists; it is never overwritten"
    TPL = sys.argv[4]; sys.argv = [sys.argv[0], POST, BD]
    src = open(os.path.join(HERE, "blind_sites_test.py")).read(); exec(src[:src.index("tared = np.array")])
    import itertools
    pr = [(j, k) for j, k in itertools.permutations(keep, 2) if plates[j] == plates[k]]
    bar = float(np.quantile([base[j] / base[k] for j, k in pr], .95))
    F = pd.read_csv(os.path.join(HERE, "tcga_brca_450k_files.csv")); al = {}
    for k_ in range(0, len(F), 500):
        for i in range(6):
            try:
                r = requests.post("https://api.gdc.cancer.gov/files", json={"filters": {"op": "in", "content": {"field": "file_id", "value": F.file_id.tolist()[k_:k_ + 500]}},
                      "fields": "file_id,cases.samples.portions.analytes.aliquots.submitter_id", "size": 1000, "format": "JSON"}, timeout=120); r.raise_for_status(); break
            except Exception: time.sleep(5 * (i + 1))
        for h in r.json()["data"]["hits"]: al[h["file_id"]] = h["cases"][0]["samples"][0]["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"]
    F["aliquot"] = F.file_id.map(al); F["plate"] = F.aliquot.str.split("-").str[5]; F["ch"] = F.file_name.str.extract(r"_(Grn|Red)\.idat")[0]
    N = F[F.sample_type == "Solid Tissue Normal"]; Tm = F[F.sample_type == "Primary Tumor"]; P = []
    for case, n in N.groupby("case"):
        t = Tm[(Tm.case == case) & (Tm.plate == n.plate.iloc[0])].sort_values("aliquot")
        if len(t) == 0 or n["sample"].iloc[0] not in keep: continue
        ta = t.aliquot.iloc[0]; tt = t[t.aliquot == ta]
        P.append(dict(case=case, plate=n.plate.iloc[0], normal_sample=n["sample"].iloc[0], tumour_sample=tt["sample"].iloc[0], tumour_aliquot=ta,
                      tumour_files={r.ch: dict(file_id=r.file_id, file_name=r.file_name, md5sum=r.md5sum) for _, r in tt.iterrows()}))
    crit = int(binom.ppf(0.95, len(P), 0.05)) + 1
    S_ = dict(test="DEV-TISSUE-BREAST-01", sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), scorer_sha256=ME, sites_methylated=list(hi), sites_unmethylated=list(lo),
              bar_pair=bar, crit=crit, n_pairs=len(P), pairs=P, template_used_in_synthetic_tests=os.path.basename(TPL),
              rule=f"FINGERPRINT if >= {crit} of {len(P)} pairs read Met-A(tumour)/Met-A(own normal) above {bar:.4f} at the sealed blind sites; else NO FINGERPRINT")
    json.dump(S_, open(SEAL, "w"), indent=1); print(S_["rule"], "| pairs", len(P), "| sites", len(hi), "+", len(lo))
elif STEP == "score":
    TB = sys.argv[4]; S_ = json.load(open(SEAL)); assert S_["scorer_sha256"] == ME, "scorer changed since the seal"
    rel = os.path.relpath(SEAL, subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip())
    assert subprocess.run(["git", "-C", HERE, "diff", "--quiet", "HEAD", "--", rel]).returncode == 0 and subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0, "seal not committed unchanged"
    S = S_["sites_methylated"] + S_["sites_unmethylated"]; rows = []
    for p in S_["pairs"]:
        n = beta(os.path.join(BD, p["normal_sample"] + ".parquet")).reindex(S); t = beta(os.path.join(TB, p["tumour_sample"] + ".parquet")).reindex(S); ok = n.notna() & t.notna()
        rows.append(dict(case=p["case"], plate=p["plate"], sites=int(ok.sum()), R=float(H(t[ok].values).mean() / H(n[ok].values).mean())))
    R = pd.DataFrame(rows); k = int((R.R > S_["bar_pair"]).sum())
    out = dict(test=S_["test"], pairs=len(R), above_bar=k, crit=S_["crit"], bar=S_["bar_pair"], R_median=float(R.R.median()), VERDICT="FINGERPRINT" if k >= S_["crit"] else "NO FINGERPRINT", rows=rows)
    json.dump(out, open(os.path.join(HERE, "scored_breast_tissue.json"), "w"), indent=1); print({a: b for a, b in out.items() if a != "rows"})
