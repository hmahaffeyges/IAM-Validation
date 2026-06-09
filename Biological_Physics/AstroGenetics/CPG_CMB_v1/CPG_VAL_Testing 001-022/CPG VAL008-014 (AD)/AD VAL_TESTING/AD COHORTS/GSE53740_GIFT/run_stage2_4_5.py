"""
Stage 2+4+5 driver for AddNeuroMed + GSE53740.
Loads Walther + atlas once, processes both cohorts sequentially.
"""
import sys, time, json, csv
sys.path.insert(0, "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime/Walther_iam_deconvolver")
sys.path.insert(0, "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime/A_Scoring_Module")
sys.path.insert(0, "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime/Mahalanobis_healthy_reference")
import walther_iam_deconvolver as wid
import iamatlas_a_scoring as ascore
from iamatlas_mahalanobis_scoring import MahalanobisHealthyHull

RUNTIME = "/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime"

print(f"[{time.strftime('%H:%M:%S')}] Loading Walther...", flush=True)
deconv = wid.WaltherIAMDeconvolver(
    matrix_path="/home/claude/ad_work/IAMAtlas.csv",
    celltype_class_map=f"{RUNTIME}/IAMAtlas_REBUILD/IAMAtlasREBUILD_celltype_to_class.json",
    verbose=False,
)
print(f"[{time.strftime('%H:%M:%S')}] Loading A-scoring artifact + Mahalanobis hull...", flush=True)
artifact_meta, celltype_markers, celltype_to_class, h_min_by_class = ascore.load_artifact(
    f"{RUNTIME}/Celltype_Marker/iamatlas_celltype_markers_v0_2.json"
)
hull = MahalanobisHealthyHull(f"{RUNTIME}/Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v0_1.json")

# Per-class markers from cell-type markers
class_markers = {cls: [] for cls in h_min_by_class.keys()}
for ct, markers in celltype_markers.items():
    cls = celltype_to_class.get(ct)
    if cls in class_markers:
        class_markers[cls].extend(markers)
for cls in class_markers:
    class_markers[cls] = list(dict.fromkeys(class_markers[cls]))

CLASSES = ['stem_pluri', 'stem_adult', 'progenitor', 'cycling', 'secretory', 'immune', 'terminal', 'stromal']
celltypes_ordered = sorted(celltype_markers.keys())


def run_cohort(cohort_name, beta_csv, meta_json, out_results_csv, out_mahalanobis_csv, gsm_col="gsm"):
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Cohort: {cohort_name}")
    print(f"{'='*60}", flush=True)
    
    # Load betas: CpGs as rows → sample dict
    with open(beta_csv) as f:
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
    print(f"  {len(samples)} samples, sample 0 has {len(sample_betas[samples[0]])} β values", flush=True)
    
    # Load clinical metadata
    clin = json.load(open(meta_json))
    clin_map = {c[gsm_col]: c for c in clin}
    
    # Run Stage 2 + 4 on every sample
    columns = (["gsm", "arm", "gender", "walther_status",
                "n_class_markers_matched", "class_residual_mae"]
               + [f"frac_{c}" for c in CLASSES]
               + [f"Ascore_{c}" for c in CLASSES]
               + [f"Acelltype_{ct}" for ct in celltypes_ordered])
    
    # Add cohort-specific extra columns
    extra_cols = []
    if cohort_name == "AddNeuroMed":
        extra_cols = ["mci_subclass", "age"]
    elif cohort_name == "GSE53740_GIFT":
        extra_cols = ["tauopathy_class", "age", "diagnosis_raw", "batch"]
    columns += extra_cols
    
    t0 = time.time()
    rows_out = []
    with open(out_results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for i, sample in enumerate(samples):
            if i and i % 50 == 0:
                el = time.time() - t0
                print(f"  [{time.strftime('%H:%M:%S')}] sample {i}/{len(samples)} | {el:.0f}s | ETA {el/i*(len(samples)-i):.0f}s", flush=True)
            
            betas = sample_betas[sample]
            meta = clin_map.get(sample, {})
            result = deconv.deconvolve(betas, refine_celltypes=False)
            cls_A = ascore.score_per_class(betas, class_markers, h_min_by_class)
            ct_A = ascore.score_per_celltype(betas, celltype_markers, celltype_to_class, h_min_by_class)
            
            row = [
                sample, meta.get("arm", "unknown"), meta.get("gender", ""),
                result.status,
                result.diagnostics.get("n_class_markers_matched", ""),
                f"{result.diagnostics.get('class_residual_mae', 0):.4f}",
            ]
            row += [f"{result.class_fractions.get(c, 0):.4f}" for c in CLASSES]
            row += [f"{cls_A.get(c, {}).get('A', 'NA')}" if isinstance(cls_A.get(c, {}).get('A'), (int, float)) else "NA" for c in CLASSES]
            row += [f"{ct_A.get(ct, {}).get('A', 'NA')}" if isinstance(ct_A.get(ct, {}).get('A'), (int, float)) else "NA" for ct in celltypes_ordered]
            row += [str(meta.get(k, "")) for k in extra_cols]
            w.writerow(row)
            rows_out.append({"sample": sample, "arm": meta.get("arm"), "gender": meta.get("gender"), "ct_A": ct_A})
    
    el = time.time() - t0
    print(f"  [{time.strftime('%H:%M:%S')}] DONE — {len(samples)} samples in {el:.0f}s ({el/len(samples):.2f}s/sample)", flush=True)
    print(f"  Output: {out_results_csv}")
    
    # Mahalanobis hyper-volume
    print(f"\n  [{time.strftime('%H:%M:%S')}] Computing Mahalanobis on 115-cell A-score vectors...", flush=True)
    mahal_rows = []
    for r in rows_out:
        ct_A_vec = {ct: d["A"] for ct, d in r["ct_A"].items() if isinstance(d.get("A"), (int, float))}
        if len(ct_A_vec) < hull.n_features * 0.8:
            continue
        m = hull.score(ct_A_vec)
        mahal_rows.append({
            "gsm": r["sample"],
            "arm": r["arm"],
            "gender": r["gender"],
            "mahalanobis_d": m.get("mahalanobis_distance"),
            "status": m.get("status", ""),
            "n_features_matched": m.get("n_features_matched", 0),
        })
    
    with open(out_mahalanobis_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gsm", "arm", "gender", "mahalanobis_d", "status", "n_features_matched"])
        w.writeheader()
        w.writerows(mahal_rows)
    print(f"  Output: {out_mahalanobis_csv}")


if __name__ == "__main__":
    cohort = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if cohort in ("all", "AddNeuroMed"):
        run_cohort(
            "AddNeuroMed",
            "/home/claude/ad_work/GSE144858_AddNeuroMed/GSE144858_betas_union.csv",
            "/home/claude/ad_work/GSE144858_AddNeuroMed/GSE144858_clinical_metadata.json",
            "/home/claude/ad_work/GSE144858_AddNeuroMed/GSE144858_full_results.csv",
            "/home/claude/ad_work/GSE144858_AddNeuroMed/GSE144858_mahalanobis.csv",
        )
    
    if cohort in ("all", "GSE53740_GIFT"):
        run_cohort(
            "GSE53740_GIFT",
            "/home/claude/ad_work/GSE53740_GIFT/GSE53740_betas_union.csv",
            "/home/claude/ad_work/GSE53740_GIFT/GSE53740_clinical_metadata.json",
            "/home/claude/ad_work/GSE53740_GIFT/GSE53740_full_results.csv",
            "/home/claude/ad_work/GSE53740_GIFT/GSE53740_mahalanobis.csv",
        )
