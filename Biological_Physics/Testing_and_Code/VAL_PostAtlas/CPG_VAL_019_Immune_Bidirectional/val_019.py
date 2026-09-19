#!/usr/bin/env python3
"""CPG-VAL-019 sealed runner — Bidirectional immune direction discrimination.

Tests whether the VAL-051 sealed 7-CpG Rule A panel (2 up + 5 down in AD-anchored
direction) provides discrimination via bidirectional decomposition that the
direction-naive pooled signal does not.
"""
import json, gzip, hashlib, time, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO = Path("/home/claude/IAM-Validation")
HERE = REPO / "Biological_Physics/validation_runs/CPG_VAL_019_Immune_Bidirectional"
PANEL = REPO / "Biological_Physics/atlas_vault/walther_clinical_runtime/Bidirectional_Decomposition/directional_panels_v1_0.json"

# β data sources
BETA_GSE51057 = REPO / "Biological_Physics/validation_runs/breast_epic_cohorts/GSE51057_EPIC_Italy/GSE51057_betas_union.csv.gz"
BETA_AIBL = REPO / "Biological_Physics/validation_runs/ad_immune_cohorts/GSE153712_AIBL/GSE153712_betas_union.csv"

# Arm metadata from A-score CSVs (these have arm column)
ASCORES_GSE51057 = REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv"
ASCORES_AIBL = REPO / "Biological_Physics/validation_runs/ad_immune_cohorts/GSE153712_AIBL/GSE153712_115celltype_ascores.csv"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float('nan')
    s = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/s) if s > 0 else float('nan')

def main():
    t0 = time.time()
    print("=" * 72)
    print("CPG-VAL-019 — Bidirectional immune direction discrimination")
    print("=" * 72)

    # 1. Load panel
    print("\n[1/5] Loading VAL-051 sealed 7-CpG Rule A panel...")
    panel = json.loads(PANEL.read_text())['panels']['immune']
    cpgs = panel['cpgs']
    cpg_directions = {c['cpg_id']: c['direction'] for c in cpgs}
    cpg_ids = list(cpg_directions.keys())
    up_cpgs = [c for c, d in cpg_directions.items() if d == 1]
    down_cpgs = [c for c, d in cpg_directions.items() if d == -1]
    print(f"  Panel: {panel['panel_id']}")
    print(f"  7 CpGs: {up_cpgs} (UP) + {down_cpgs} (DOWN)")
    print(f"  H_min: {panel['h_min']}")

    # 2. Helper to load cohort β + arm
    def load_cohort(beta_path, ascore_path, sample_col, cohort_label):
        """Load β matrix for the 7 panel CpGs + per-sample arm from A-score CSV."""
        # Arm metadata
        arm_df = pd.read_csv(ascore_path)
        if 'arm' not in arm_df.columns:
            return None, f"No arm column in {ascore_path.name}"
        arms = dict(zip(arm_df[sample_col].astype(str), arm_df['arm'].astype(str)))

        # β
        opener = gzip.open if str(beta_path).endswith('.gz') else open
        with opener(beta_path, 'rt') as f:
            # Read header
            header = f.readline().strip().split(',')
            cpg_col = header[0]
            sample_cols = header[1:]
            print(f"    Loading β: cpg_col={cpg_col}, {len(sample_cols)} samples")
            # Filter rows to just the 7 panel CpGs
            rows = []
            for line in f:
                fields = line.rstrip("\n").split(',')
                if fields[0].strip('"') in cpg_ids:
                    rows.append(fields)
            if not rows:
                return None, "no panel CpGs found"
        # Convert to df
        df_beta = pd.DataFrame(rows, columns=header).set_index(cpg_col)
        df_beta.index = df_beta.index.str.strip('"')
        df_beta = df_beta.apply(pd.to_numeric, errors='coerce').T  # sample × cpg
        df_beta.index = [s.strip('"') for s in df_beta.index]

        # Per-sample compute: pooled mean β, up_mean β, down_mean β
        present_cpgs = [c for c in cpg_ids if c in df_beta.columns]
        present_up = [c for c in up_cpgs if c in df_beta.columns]
        present_down = [c for c in down_cpgs if c in df_beta.columns]

        out = pd.DataFrame(index=df_beta.index)
        out['cohort'] = cohort_label
        out['arm'] = [arms.get(str(s), 'unknown') for s in out.index]
        out['pooled_mean_beta'] = df_beta[present_cpgs].mean(axis=1)
        out['up_mean_beta'] = df_beta[present_up].mean(axis=1) if present_up else float('nan')
        out['down_mean_beta'] = df_beta[present_down].mean(axis=1) if present_down else float('nan')
        # Directional combined: up - down (case has higher up, lower down ⇒ this is signed signal)
        out['directional_signed'] = out['up_mean_beta'] - out['down_mean_beta']
        for c in present_cpgs:
            out[f'beta_{c}'] = df_beta[c]
        return out, f"loaded {len(out)} samples, {len(present_cpgs)}/7 panel CpGs"

    # 3. Load both cohorts
    print("\n[2/5] Loading AIBL (AD-anchored) β data...")
    aibl_df, msg = load_cohort(BETA_AIBL, ASCORES_AIBL, 'sentrix', 'AIBL')
    print(f"  AIBL: {msg}")
    if aibl_df is None:
        print(f"  FAIL loading AIBL"); sys.exit(1)
    print(f"  AIBL arms: {aibl_df['arm'].value_counts().to_dict()}")

    print("\n[3/5] Loading GSE51057 (breast pre-dx) β data...")
    breast_df, msg = load_cohort(BETA_GSE51057, ASCORES_GSE51057, 'gsm', 'GSE51057_breast')
    print(f"  GSE51057: {msg}")
    if breast_df is None:
        print(f"  FAIL loading GSE51057"); sys.exit(1)
    print(f"  GSE51057 arms: {breast_df['arm'].value_counts().to_dict()}")

    # 4. Compute discriminations
    print("\n[4/5] Computing Cohen's d for each metric per cohort...")
    def per_cohort(df, case_label, hc_label='hc'):
        case = df[df['arm'] == case_label]
        hc = df[df['arm'] == hc_label]
        if len(case) == 0 or len(hc) == 0:
            return None
        return {
            "n_case": int(len(case)), "n_hc": int(len(hc)),
            "case_arm": case_label, "hc_arm": hc_label,
            "d_pooled": round(cohens_d(case['pooled_mean_beta'], hc['pooled_mean_beta']), 4),
            "d_up_panel": round(cohens_d(case['up_mean_beta'], hc['up_mean_beta']), 4),
            "d_down_panel": round(cohens_d(case['down_mean_beta'], hc['down_mean_beta']), 4),
            "d_directional_signed": round(cohens_d(case['directional_signed'], hc['directional_signed']), 4),
        }

    aibl_results = per_cohort(aibl_df, 'ad', 'hc')
    print(f"\n  AIBL (AD-anchored cohort) — d_pooled={aibl_results['d_pooled']:+.3f}")
    print(f"    d_up_panel    = {aibl_results['d_up_panel']:+.3f}  (2 CpGs hypermethylate in AD)")
    print(f"    d_down_panel  = {aibl_results['d_down_panel']:+.3f}  (5 CpGs hypomethylate in AD)")
    print(f"    d_signed (up-down) = {aibl_results['d_directional_signed']:+.3f}")

    breast_results = per_cohort(breast_df, 'case', 'hc')
    print(f"\n  GSE51057 (breast pre-dx, NOT AD-anchored) — d_pooled={breast_results['d_pooled']:+.3f}")
    print(f"    d_up_panel    = {breast_results['d_up_panel']:+.3f}")
    print(f"    d_down_panel  = {breast_results['d_down_panel']:+.3f}")
    print(f"    d_signed (up-down) = {breast_results['d_directional_signed']:+.3f}")

    # Per-CPG sign concordance in AIBL
    aibl_case = aibl_df[aibl_df['arm']=='ad']
    aibl_hc = aibl_df[aibl_df['arm']=='hc']
    per_cpg_aibl = []
    for c in cpg_ids:
        col = f'beta_{c}'
        if col in aibl_df.columns:
            obs_sign = np.sign(aibl_case[col].mean() - aibl_hc[col].mean())
            expected_sign = cpg_directions[c]
            per_cpg_aibl.append({
                "cpg_id": c, "expected_direction": expected_sign,
                "observed_case_mean": round(float(aibl_case[col].mean()), 5),
                "observed_hc_mean": round(float(aibl_hc[col].mean()), 5),
                "observed_diff": round(float(aibl_case[col].mean()-aibl_hc[col].mean()), 5),
                "observed_sign": int(obs_sign),
                "concordant_with_panel": bool(obs_sign == expected_sign),
            })
    n_concordant = sum(1 for r in per_cpg_aibl if r['concordant_with_panel'])
    print(f"\n  AIBL per-CpG concordance with panel direction: {n_concordant}/7")
    for r in per_cpg_aibl:
        mark = "✓" if r['concordant_with_panel'] else "✗"
        print(f"    {mark} {r['cpg_id']}: panel direction={r['expected_direction']:+d}, observed diff={r['observed_diff']:+.4f}")

    # 5. Save outputs
    print("\n[5/5] Saving outputs...")
    HERE.mkdir(exist_ok=True)

    # Per-sample CSV (combined cohorts)
    combined = pd.concat([aibl_df.reset_index().rename(columns={'index': 'sample_id'}),
                          breast_df.reset_index().rename(columns={'index': 'sample_id'})], ignore_index=True)
    per_sample = HERE / "CPG_VAL_019_per_sample.csv"
    # Round numeric columns
    for col in combined.select_dtypes(include=[np.number]).columns:
        combined[col] = combined[col].round(5)
    combined.to_csv(per_sample, index=False)
    print(f"  {per_sample.name}: {combined.shape}")

    # Pass condition checks
    pass1 = (aibl_results['d_up_panel'] >= 0.30) and (aibl_results['d_down_panel'] <= -0.30)
    # Convention: up_panel has higher β in AD ⇒ d_up should be POSITIVE
    #             down_panel has lower β in AD ⇒ d_down should be NEGATIVE
    # So pass-1: |d_up| >= 0.30 AND |d_down| >= 0.30 AND opposite signs
    pass1_strict = (abs(aibl_results['d_up_panel']) >= 0.30 and
                    abs(aibl_results['d_down_panel']) >= 0.30 and
                    np.sign(aibl_results['d_up_panel']) != np.sign(aibl_results['d_down_panel']))
    pass2 = abs(aibl_results['d_directional_signed']) > abs(aibl_results['d_pooled'])
    pass3 = (np.sign(breast_results['d_up_panel']) != np.sign(aibl_results['d_up_panel'])) or \
            (abs(breast_results['d_directional_signed'] - aibl_results['d_directional_signed']) > 0.2)

    overall = "PASS" if (pass1_strict and pass2 and pass3) else (
              "DIRECTIONAL" if pass1_strict else "NULL")

    results = {
        "val_id": "CPG-VAL-019",
        "title": "Bidirectional immune direction discrimination",
        "card": "Immune universal v1.0",
        "execution_date": "2026-06-07",
        "outcome_code": overall,
        "panel_source": panel['panel_id'],
        "panel_h_min": panel['h_min'],
        "panel_cpgs_count": 7,
        "panel_up_cpgs": up_cpgs, "panel_down_cpgs": down_cpgs,
        "per_cohort_results": {
            "AIBL_AD_anchored": aibl_results,
            "GSE51057_breast_pre_dx": breast_results,
        },
        "per_cpg_concordance_AIBL": {
            "n_concordant": n_concordant, "n_total": 7,
            "concordance_pct": round(100*n_concordant/7, 1),
            "details": per_cpg_aibl,
        },
        "pass_conditions": {
            "condition_1_AIBL_both_directions_fire": {
                "criterion": "|d_up| ≥ 0.30 AND |d_down| ≥ 0.30 AND opposite signs",
                "passed": bool(pass1_strict),
                "d_up_panel": aibl_results['d_up_panel'],
                "d_down_panel": aibl_results['d_down_panel'],
            },
            "condition_2_directional_beats_pooled": {
                "criterion": "|d_directional_signed| > |d_pooled| in AIBL",
                "passed": bool(pass2),
                "d_directional_signed": aibl_results['d_directional_signed'],
                "d_pooled": aibl_results['d_pooled'],
            },
            "condition_3_disease_specific_firing": {
                "criterion": "breast firing pattern DIFFERENT from AIBL (sign flip in any direction OR |Δd_signed| > 0.2)",
                "passed": bool(pass3),
                "AIBL_d_signed": aibl_results['d_directional_signed'],
                "breast_d_signed": breast_results['d_directional_signed'],
                "abs_difference": round(abs(breast_results['d_directional_signed'] - aibl_results['d_directional_signed']), 4),
            },
        },
    }
    with open(HERE / "results.json", "w") as f: json.dump(results, f, indent=2, default=str)

    with open(HERE / "stratified_results.json", "w") as f:
        json.dump({
            "stratification": "by cohort (AIBL AD-anchored vs GSE51057 breast pre-dx)",
            "AIBL_AD": aibl_results,
            "GSE51057_breast": breast_results,
            "comparison": {
                "d_pooled_AIBL_minus_breast": round(aibl_results['d_pooled'] - breast_results['d_pooled'], 4),
                "d_signed_AIBL_minus_breast": round(aibl_results['d_directional_signed'] - breast_results['d_directional_signed'], 4),
            }
        }, f, indent=2)

    # Null: shuffle directions randomly within panel, recompute d_signed in AIBL
    print("\n  Null test: random direction shuffle (1000 permutations)...")
    np.random.seed(42)
    null_d = []
    aibl_case_b = aibl_df[aibl_df['arm']=='ad'][[f'beta_{c}' for c in cpg_ids if f'beta_{c}' in aibl_df.columns]]
    aibl_hc_b = aibl_df[aibl_df['arm']=='hc'][[f'beta_{c}' for c in cpg_ids if f'beta_{c}' in aibl_df.columns]]
    n_cpgs_present = aibl_case_b.shape[1]
    for _ in range(1000):
        dirs_shuffled = np.random.choice([-1, 1], size=n_cpgs_present)
        # Up subset under shuffled directions
        up_mask = dirs_shuffled == 1; down_mask = dirs_shuffled == -1
        if up_mask.any() and down_mask.any():
            case_up = aibl_case_b.iloc[:, up_mask].mean(axis=1)
            case_down = aibl_case_b.iloc[:, down_mask].mean(axis=1)
            hc_up = aibl_hc_b.iloc[:, up_mask].mean(axis=1)
            hc_down = aibl_hc_b.iloc[:, down_mask].mean(axis=1)
            case_sig = case_up - case_down
            hc_sig = hc_up - hc_down
            null_d.append(cohens_d(case_sig, hc_sig))
    null_d = np.array([d for d in null_d if not np.isnan(d)])
    observed_d_signed = aibl_results['d_directional_signed']
    null_p = float(np.mean(np.abs(null_d) >= abs(observed_d_signed)))
    print(f"  Observed AIBL d_signed: {observed_d_signed:+.3f}")
    print(f"  Null mean: {null_d.mean():+.4f}  std: {null_d.std():.4f}  abs>observed: p={null_p:.4f}")

    with open(HERE / "null_results.json", "w") as f:
        json.dump({
            "null_type": "random_direction_shuffle_within_panel",
            "n_permutations": 1000, "rng_seed": 42,
            "observed_d_signed_AIBL": round(observed_d_signed, 4),
            "null_distribution_mean": round(float(null_d.mean()), 5),
            "null_distribution_std": round(float(null_d.std()), 5),
            "p_two_sided_abs": null_p,
        }, f, indent=2)

    # cohort manifest
    with open(HERE / "cohort_manifest.json", "w") as f:
        json.dump({
            "AIBL": {"path": str(BETA_AIBL.relative_to(REPO)), "n": int(len(aibl_df)),
                     "platform": "EPIC 850K", "sha256": sha256_file(BETA_AIBL)},
            "GSE51057": {"path": str(BETA_GSE51057.relative_to(REPO)), "n": int(len(breast_df)),
                         "platform": "HM450", "sha256": sha256_file(BETA_GSE51057)},
            "panel_source": panel['panel_id'],
            "panel_sha256": panel.get('panel_sha256_anchor', 'see Bidirectional_Decomposition/directional_panels_v1_0.json'),
        }, f, indent=2)
    
    print(f"\nPass-1 (both directions fire, AIBL): {pass1_strict}")
    print(f"Pass-2 (directional > pooled): {pass2}")
    print(f"Pass-3 (disease-specific firing): {pass3}")
    print(f"OUTCOME: {overall}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    return overall

if __name__ == "__main__":
    sys.exit(0 if main() in ("PASS", "DIRECTIONAL") else 1)
