"""
AIBL Stage 2-5 driver — runs the full SOP v1.2 chain on 726 samples.

Output: per-sample CSV with:
  - gsm, sentrix, arm, gender
  - 8 class fractions (Walther deconvolution)
  - Walther status + diagnostics (n_class_markers_matched, class_residual_mae, ...)
  - 8 class A-scores
  - 115 cell-type A-scores

Estimated time: ~12-15 minutes (Walther ~1s/sample × 726).
"""
import sys, time, json, csv
sys.path.insert(0, "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime/Walther_iam_deconvolver")
sys.path.insert(0, "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime/A_Scoring_Module")
import walther_iam_deconvolver as wid
import iamatlas_a_scoring as ascore

RUNTIME = "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime"

print(f"[{time.strftime('%H:%M:%S')}] Loading Walther + atlas...")
sys.stdout.flush()
deconv = wid.WaltherIAMDeconvolver(
    matrix_path="/home/claude/ad_work/IAMAtlas.csv",
    celltype_class_map=f"{RUNTIME}/IAMAtlas_REBUILD/IAMAtlasREBUILD_celltype_to_class.json",
    verbose=False,
)

print(f"[{time.strftime('%H:%M:%S')}] Loading A-scoring artifact (v0_2 markers + H_min)...")
sys.stdout.flush()
artifact_meta, celltype_markers, celltype_to_class, h_min_by_class = ascore.load_artifact(
    f"{RUNTIME}/Celltype_Marker/iamatlas_celltype_markers_v0_2.json"
)
print(f"  H_min frozen: {h_min_by_class}")
print(f"  115 cell types, 8 classes")

# Build per-class markers from the cell-type markers (one-vs-rest pattern)
# For class A-scoring we want each class to have its own marker list
class_markers = {cls: [] for cls in h_min_by_class.keys()}
for ct, markers in celltype_markers.items():
    cls = celltype_to_class.get(ct)
    if cls in class_markers:
        class_markers[cls].extend(markers)
# Dedup per class
for cls in class_markers:
    class_markers[cls] = list(dict.fromkeys(class_markers[cls]))
print(f"  Per-class marker counts: { {k: len(v) for k, v in class_markers.items()} }")
sys.stdout.flush()

print(f"\n[{time.strftime('%H:%M:%S')}] Loading AIBL β matrix (transposed CpG→sample)...")
sys.stdout.flush()
with open("/home/claude/ad_work/GSE153712_betas_union.csv") as f:
    r = csv.reader(f)
    header = next(r)
    samples = header[1:]
    sample_betas = {s: {} for s in samples}
    for row in r:
        cpg = row[0]
        for i, v in enumerate(row[1:]):
            if v:
                try: sample_betas[samples[i]][cpg] = float(v)
                except ValueError: pass

print(f"  {len(samples)} samples loaded; sample 0 has {len(sample_betas[samples[0]])} β values")

# Load clinical metadata
clin = json.load(open("/home/claude/ad_work/GSE153712_clinical_metadata.json"))
clin_map = {s["sentrix"]: s for s in clin}
# Fix MCI label
for s in clin_map.values():
    if s["disease_status"] == "Mild Cognitive Impairment":
        s["arm"] = "mci"

print(f"\n[{time.strftime('%H:%M:%S')}] Running Stages 2 + 4 on all {len(samples)} samples...")
sys.stdout.flush()

CLASSES = ['stem_pluri', 'stem_adult', 'progenitor', 'cycling', 'secretory', 'immune', 'terminal', 'stromal']
out_path = "/home/claude/ad_work/GSE153712_AIBL_full_results.csv"

# Header: metadata + 8 class fractions + 8 class A-scores + 115 cell-type A-scores + diagnostics
celltypes_ordered = sorted(celltype_markers.keys())
columns = (["sentrix", "gsm", "arm", "gender", "walther_status",
            "n_class_markers_matched", "class_residual_mae"]
           + [f"frac_{c}" for c in CLASSES]
           + [f"Ascore_{c}" for c in CLASSES]
           + [f"Acelltype_{ct}" for ct in celltypes_ordered])

t_start = time.time()
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(columns)
    
    for i, sample in enumerate(samples):
        if i and i % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(samples) - i)
            print(f"  [{time.strftime('%H:%M:%S')}] sample {i}/{len(samples)} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
            sys.stdout.flush()
        
        betas = sample_betas[sample]
        meta = clin_map.get(sample, {})
        
        # Stage 2: Walther deconvolution → class fractions
        result = deconv.deconvolve(betas, refine_celltypes=False)
        class_fracs = result.class_fractions
        diag = result.diagnostics
        
        # Stage 4 class A-scores
        cls_A = ascore.score_per_class(betas, class_markers, h_min_by_class)
        # Stage 4 per-cell-type A-scores
        ct_A = ascore.score_per_celltype(betas, celltype_markers, celltype_to_class, h_min_by_class)
        
        row = [
            sample, meta.get("gsm", ""), meta.get("arm", "unknown"), meta.get("gender", ""),
            result.status,
            diag.get("n_class_markers_matched", ""), f"{diag.get('class_residual_mae', 0):.4f}",
        ]
        row += [f"{class_fracs.get(c, 0):.4f}" for c in CLASSES]
        row += [f"{cls_A.get(c, {}).get('A', 'NA')}" if isinstance(cls_A.get(c, {}).get('A'), (int, float)) else "NA" for c in CLASSES]
        row += [f"{ct_A.get(ct, {}).get('A', 'NA')}" if isinstance(ct_A.get(ct, {}).get('A'), (int, float)) else "NA" for ct in celltypes_ordered]
        w.writerow(row)

elapsed = time.time() - t_start
print(f"\n[{time.strftime('%H:%M:%S')}] DONE — {len(samples)} samples in {elapsed:.0f}s ({elapsed/len(samples):.2f}s/sample)")
print(f"  Output: {out_path}")

# Quick summary
import csv
with open(out_path) as f:
    rows = list(csv.DictReader(f))
print(f"  Rows written: {len(rows)}")
arms = {}
for r in rows:
    arms[r["arm"]] = arms.get(r["arm"], 0) + 1
print(f"  Arms: {arms}")

# Quick AD vs HC d on Mahalanobis-like 8-class A-score Σ
print(f"\n=== Quick check: AD vs HC on summed 8-class A-score ===")
import numpy as np
def class_sum(r):
    vals = []
    for c in CLASSES:
        v = r[f"Ascore_{c}"]
        try: vals.append(float(v))
        except (ValueError, TypeError): pass
    return sum(vals) if vals else None

ad_sums = [class_sum(r) for r in rows if r["arm"] == "ad"]
ad_sums = [x for x in ad_sums if x is not None]
hc_sums = [class_sum(r) for r in rows if r["arm"] == "hc"]
hc_sums = [x for x in hc_sums if x is not None]

if ad_sums and hc_sums:
    mu_a, mu_h = np.mean(ad_sums), np.mean(hc_sums)
    sd_p = np.sqrt(((np.std(ad_sums, ddof=1)**2) * (len(ad_sums)-1) +
                   (np.std(hc_sums, ddof=1)**2) * (len(hc_sums)-1)) /
                  (len(ad_sums) + len(hc_sums) - 2))
    d = (mu_a - mu_h) / sd_p
    print(f"  AD={len(ad_sums)}, HC={len(hc_sums)}")
    print(f"  Σ A-score: AD mean={mu_a:.3f}, HC mean={mu_h:.3f}")
    print(f"  Cohen's d = {d:+.3f}")
