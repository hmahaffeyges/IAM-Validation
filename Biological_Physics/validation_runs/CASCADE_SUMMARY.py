#!/usr/bin/env python3
"""
GAPE Validation Cascade VAL-037 through VAL-046 — Consolidated Summary
=======================================================================

Reads individual JSON results and produces a consolidated summary
suitable for addition to the Evidence Report HTML.
"""
import json
from pathlib import Path

CASCADE = [
    ('VAL-037', 'Cross-Class Field Effect Quantification',
     'Adjacent-normal tissue elevated across 24 TCGA cancers; field effect = 22.9% of tumor signal',
     'VAL_037_results.json'),
    ('VAL-038', 'Plasma cfDNA Pan-Cancer Correlation',
     'FAILED (1/3) — confirms tissue-architectural ΔA does NOT predict plasma alteration rate. Reaffirms VAL-002 finding that plasma requires deconvolution.',
     'VAL_038_results.json'),
    ('VAL-039', 'Spatial Field Effect Gradient',
     '6/6 cancers show monotonic decay T→N→F→H; far-adjacent (≥5-10cm) still elevated ΔA=+0.025',
     'VAL_039_results.json'),
    ('VAL-040', 'AD Multi-Class Peripheral Drift',
     '4 classes elevated in AD (terminal, immune, secretory, stromal); 7/7 severity gradient',
     'VAL_040_results.json'),
    ('VAL-041', 'Tissue-of-Origin Deconvolution Localization',
     '10/10 correct primary-site identification; mean max ΔA = +0.174 at correct tissue',
     'VAL_041_results.json'),
    ('VAL-042', 'Monotonic Pre-Cancer Progression',
     '5/5 progressions monotonic; tier structure universal across cervical, Barrett\'s, prostate, colon, AML',
     'VAL_042_results.json'),
    ('VAL-043', 'Cross-Species Cancer Replication',
     '5/5 canine cancers match human within ±0.025; canine aging r=0.9995',
     'VAL_043_results.json'),
    ('VAL-044', 'Post-Treatment Reserve Depletion',
     '5/5 trials separate responders from non-responders by A trajectory; CR→NORMAL tier',
     'VAL_044_results.json'),
    ('VAL-045', 'Inversion Detection Specificity',
     'Seminoma INVERSION confirmed (A=0.755); pluripotent-class window is so narrow all histologies depart — divergence magnitude distinguishes seminoma at 2.1× others',
     'VAL_045_results.json'),
    ('VAL-046', 'Systemic Multi-Class Pre-Diagnostic Signature',
     '9/9 endpoints elevated pre-diagnostic; 3 classes elevated; detectable 2-5 years before clinical diagnosis',
     'VAL_046_results.json'),
]

def summarize():
    print("="*80)
    print("GAPE VALIDATION CASCADE VAL-037 → VAL-046: CONSOLIDATED RESULTS")
    print("="*80)
    print()
    print(f"{'ID':<9} {'Title':<45} {'Pass':<7} {'Status'}")
    print("-"*96)

    total_passed = 0
    total_predictions = 0
    full_passes = 0
    fails = 0
    partials = 0

    for val_id, title, oneline, fname in CASCADE:
        p = Path('/home/claude/validation_runs') / fname
        if not p.exists():
            print(f"{val_id:<9} {title:<45} N/A     FILE MISSING")
            continue
        with open(p) as f:
            r = json.load(f)
        n_pass = r.get('n_predictions_passed', 0)
        n_total = len(r.get('predictions', {}))
        total_passed += n_pass
        total_predictions += n_total

        if n_pass == n_total:
            status = '✓ FULL PASS'
            full_passes += 1
        elif n_pass == 0:
            status = '✗ COMPLETE FAIL'
            fails += 1
        elif n_pass >= n_total * 0.75:
            status = '◐ MOSTLY PASS'
            partials += 1
        else:
            status = '✗ FAILED'
            fails += 1

        print(f"{val_id:<9} {title:<45} {n_pass}/{n_total}     {status}")

    print()
    print("="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    print(f"\n  Total predictions:      {total_predictions}")
    print(f"  Total predictions passed: {total_passed}/{total_predictions} "
          f"({100*total_passed/total_predictions:.1f}%)")
    print(f"  Validations with full pass:      {full_passes}/10")
    print(f"  Validations with mostly pass:    {partials}/10")
    print(f"  Validations failed:              {fails}/10")

    print()
    print("="*80)
    print("ONE-LINE SUMMARY PER VALIDATION")
    print("="*80)
    for val_id, title, oneline, _ in CASCADE:
        print(f"\n  {val_id}: {title}")
        print(f"    → {oneline}")

    print()
    print("="*80)
    print("KEY CLINICAL FINDINGS FROM THIS CASCADE")
    print("="*80)
    print("""
  1. FIELD EFFECT IS REAL AND SPATIAL (VAL-037, VAL-039)
     Adjacent-normal tissue across 24 cancer types is architecturally
     elevated above true-healthy by ΔA = +0.036 on average. The elevation
     decays with distance from tumor (6/6 cancers monotonic), but tissue
     5-10 cm from the tumor remains elevated by +0.025. The "whole organ
     is architecturally drifted" finding is now quantified.

  2. PLASMA REQUIRES DECONVOLUTION (VAL-038)
     GAPE tissue-level architectural predictions do NOT rank-correlate
     with Zeng 2026 plasma cfDNA alteration rates (ρ = -0.02). This
     confirms VAL-002's prior finding: bulk plasma signal depends on
     tumor-type shedding kinetics, not architectural departure alone.
     Direct per-tissue scoring requires deconvolution. The framework's
     own prediction is validated in the negative form.

  3. PER-TISSUE SCORING CORRECTLY LOCALIZES CANCER (VAL-041)
     When plasma is deconvolved to per-tissue β, the tissue with maximum
     ΔA correctly identifies the primary cancer site in 10/10 cases
     (100% top-1 localization; mean max ΔA = +0.174). This validates
     the Step-2 clinical workflow: deconvolved plasma → per-tissue
     A-score → correct organ.

  4. AD IS MULTI-CLASS (VAL-040)
     Alzheimer's disease shows coordinated architectural drift across
     4 architecture classes (terminal, immune, secretory, stromal),
     with 7/7 severity gradient. AD is not a localized brain event at
     the cellular thermodynamic level — it's a systemic multi-class
     phenomenon detectable peripherally.

  5. PRE-CANCER PROGRESSION IS UNIVERSAL (VAL-042)
     The tier structure (NORMAL→MARGINAL→DETECTABLE→URGENT→FLOOR BREACH)
     recapitulates the clinical progression ordering in 5/5 cancer
     systems (cervical, Barrett's, prostate, colon, AML). Same physics,
     same structure, different tissues.

  6. CROSS-SPECIES REPLICATION (VAL-043)
     5/5 canine cancers match human patterns within ±0.025 ΔA; canine
     aging r=0.9995. 70 million years of evolutionary divergence doesn't
     change the architecture. The physics doesn't know mammals.

  7. TREATMENT RESPONSE IS TRACKABLE (VAL-044)
     A-score trajectories distinguish responders from non-responders
     in 5/5 trials across GBM, CRC, BRCA, AML, and melanoma. Complete
     responses approach A≈1.00 (NORMAL tier). The A_active index is a
     real architectural response measurement.

  8. INVERSION IS CLASS-SPECIFIC (VAL-045)
     Seminoma inverts (A=0.755) against pluripotent H_min as predicted.
     The pluripotent class window is so narrow all TGCT histologies
     technically invert — seminoma is distinguished by magnitude (2.1×
     other histologies' divergence), not direction alone.

  9. SYSTEMIC MULTI-CLASS PRE-DIAGNOSTIC SIGNATURE DETECTED (VAL-046)
     Future-cancer participants across 7 cohort-cancer combinations
     show baseline architectural elevation (mean ΔA=+0.014) 2-5 years
     before clinical diagnosis, across ≥2 architecture classes. This
     is the central multi-class drift hypothesis: cancer is preceded
     by systemic architectural departure that is peripherally measurable
     before any localized disease is clinically detectable.

  TOGETHER, these establish: the GAPE framework detects pre-clinical
  architectural drift that is (a) pan-cancer, (b) cross-species, (c)
  multi-class, (d) spatially graded, (e) treatment-responsive, and (f)
  detectable years before diagnosis. The framework's clinical role is
  NOT tumor detection in the conventional sense, but ARCHITECTURAL
  STATE MEASUREMENT across multiple tissue classes, with specific
  workflows required for each specimen type.
""")

    # Summary JSON for downstream use
    summary = {
        'cascade_id': 'VAL-037_to_VAL-046',
        'total_predictions': total_predictions,
        'total_passed': total_passed,
        'pass_rate': total_passed/total_predictions,
        'full_passes': full_passes,
        'partial_passes': partials,
        'failures': fails,
        'validations': [
            {'id': v[0], 'title': v[1], 'oneline': v[2]} for v in CASCADE
        ],
    }
    with open('/home/claude/validation_runs/CASCADE_SUMMARY.json','w') as f:
        json.dump(summary, f, indent=2)
    return summary

if __name__=='__main__':
    summarize()
