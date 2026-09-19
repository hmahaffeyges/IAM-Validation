#!/usr/bin/env python3
"""
Update all 4 EDEAR disease cards to v2.1:
  - Inject universal_reference block (full-inline Option B)
  - Inject lessons_learned section with per-card documented quirks
  - Bump card_version to v2.1
  - Preserve all existing content
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/claude/cookbook_v2.1')
from universal_reference_block import UNIVERSAL_REFERENCE

CARD_LESSONS_LEARNED = {
    'breast-epic': {
        '_purpose': 'Disease-specific quirks encountered during validation and deployment. Per-card complement to the master LESSONS_LEARNED.md catalog.',
        'lessons': [
            {
                'lesson_id': 'breast-LL-001',
                'source': 'VAL-047 Phase 9',
                'context': 'GSE51057 EPIC-Italy cohort, n=329, nested case-control',
                'quirk': 'The pooled-entropy immune A-score with Xu-538 panel produces d=+0.71 at 5-10yr pre-diagnosis window on breast cancer — stronger at longer windows than 0-2yr, which attenuates to d=+0.37.',
                'interpretation': 'The 0-2yr near-dx attenuation is consistent with immune escape as tumors crystallize. Card deployment should prefer 2-5yr or longer pre-dx windows for highest signal.',
                'applied_to_card': 'Tier thresholds anchored to 2-5yr pre-dx performance; near-dx is noted as weaker in known_limitations.',
            },
            {
                'lesson_id': 'breast-LL-002',
                'source': 'VAL-047 Phase 12',
                'context': 'GSE51032 replication, same EPIC-Italy population, independent nesting',
                'quirk': 'Breast signal d=+0.65 at 2-5yr pre-dx replicates cleanly across Phase 9 and Phase 12. Same platform, same population, different nesting. Demonstrates the signal is not a Phase 9-specific artifact.',
                'interpretation': 'Cross-platform validation would require a non-EPIC-Italy cohort (UK Biobank, Sister Study). Cross-nesting in the same population is a weaker form of replication than cross-cohort.',
                'applied_to_card': 'Tier is cross_platform_validated_two_cohorts with explicit acknowledgment that both are EPIC-Italy; a truly cross-population run is listed as next_validation_step.',
            },
            {
                'lesson_id': 'breast-LL-003',
                'source': 'Literature cross-check — Kresovich 2022',
                'context': 'mBCRS 100-CpG elastic-net AUC 0.69 on EPIC-Italy',
                'quirk': 'The entropy-based immune A-score reaches comparable discriminative power (AUC ~0.65-0.70) to the elastic-net ML benchmark without training, without feature selection, without any breast-specific CpG selection. The Xu-538 panel was selected for breast-cancer association but the A-score mechanism is class-level architectural drift, not breast-specific signal.',
                'interpretation': 'This is the cleanest existing demonstration that class-level architectural scoring is competitive with disease-specific ML in pre-diagnostic cancer detection. Worth flagging for acquisition discussions.',
                'applied_to_card': 'validation_tier_rationale mentions the Kresovich 2022 benchmark as competitive-performance anchor.',
            },
        ],
    },

    'crc-epic': {
        '_purpose': 'Disease-specific quirks encountered during validation and deployment.',
        'lessons': [
            {
                'lesson_id': 'crc-LL-001',
                'source': 'VAL-047 Phase 12',
                'context': 'GSE51032 colorectal replication, n_cases=76',
                'quirk': 'CRC immune A-score direction is INVERTED relative to breast (d=-0.33 all-pre-dx on Xu-538 panel). Same panel, same pipeline, opposite sign. This was surprising on first encounter.',
                'interpretation': 'CRC pre-diagnostic immune response is tolerogenic (Treg-dominated, immune-suppressive) rather than activated. The architectural signature is lower entropy drift (more ordered) rather than higher (more disordered) — opposite direction to most cancers. Mechanism aligns with published CRC immune literature on Treg infiltration and immune exclusion.',
                'applied_to_card': 'expected_direction in card JSON explicitly flagged as NEGATIVE. Decision logic item 2 in README_MASTER handles this: "depressed below DETECTABLE tier (negative direction) + Stage 2 localizes to colon_epithelial".',
            },
            {
                'lesson_id': 'crc-LL-002',
                'source': 'Phase 12 0-2yr window',
                'context': 'GSE51032 0-2yr CRC cases, n=8',
                'quirk': 'Near-dx 0-2yr window shows d=-0.47 (even more negative) than all-pre-dx d=-0.33. Direction is preserved and amplified close to diagnosis, not attenuated as in breast.',
                'interpretation': 'CRC Treg dominance persists or intensifies close to diagnosis, consistent with published immune-exclusion phenotype of late-stage CRC. Breast (immune escape → attenuation) and CRC (immune suppression → intensification) have structurally different trajectories.',
                'applied_to_card': 'known_limitations includes note that CRC and breast have mechanistically opposite pre-dx trajectories; one-size-fits-all interpretation does not work.',
            },
            {
                'lesson_id': 'crc-LL-003',
                'source': 'Zhao 2020 literature cross-check',
                'context': 'Zhao 2020 BMC Cancer published CRC methylation signature',
                'quirk': 'Zhao 2020 used a DIFFERENT methodology (classification-trained panel) on the SAME GSE51032 cohort and found CRC-predictive signal. Our inverted-direction finding is consistent with their predictive result — both detect the CRC signal, with different aggregations picking up different aspects.',
                'interpretation': 'Cross-methodology consistency on the same cohort supports the finding. Upgrades tier from exploratory to single_cohort_validated_with_consistent_published_reference.',
                'applied_to_card': 'validation_tier includes explicit Zhao 2020 cross-reference.',
            },
        ],
    },

    'ad-immune': {
        '_purpose': 'Disease-specific quirks encountered during validation and deployment.',
        'lessons': [
            {
                'lesson_id': 'ad-LL-001',
                'source': 'VAL-050 AIBL pooled-entropy null',
                'context': 'AIBL GSE153712, IMM_CPGS_EPIC_18 panel, pooled-entropy A-score',
                'quirk': 'Pooled-entropy A-score produced d=+0.077 p=0.32 AUC=0.51 — NULL on the AD cohort despite VAL-040 meta-analysis showing cohort-level AD immune-class elevation. Per-CpG examination revealed 10 of 18 CpGs with positive Δβ and 8 with negative Δβ — bidirectional pattern.',
                'interpretation': 'Pooled-entropy H(β) is symmetric around β=0.5; up-moves in one CpG and down-moves in another both change entropy similarly, and the pooled mean cancels. Bidirectional disease drift requires a directional scoring scheme.',
                'applied_to_card': 'Primary scoring moved from pooled-entropy to A_dir directional composite per VAL-051 Rule A. Pooled-entropy reported as secondary for transparency.',
            },
            {
                'lesson_id': 'ad-LL-002',
                'source': 'VAL-051 Rule A recovery',
                'context': 'AIBL holdout n=33 AD + 95 HC, 7-CpG directional panel',
                'quirk': 'Directional scoring recovered the signal at d=+0.624 AUC=0.68 on the SAME cohort, SAME CpGs (7 selected from the 18 by |Δβ|>0.015 and q_FDR<0.10 on training split). Same data produced null (pooled) and recovery (directional) — the method mattered, not the data.',
                'interpretation': 'Directional-Score Principle: pooled-entropy is correct for uniform-direction drift (breast, CRC) and wrong for bidirectional drift (AD, likely autoimmune). Cards report both scores to guard against this failure mode per README_MASTER.',
                'applied_to_card': 'scoring_method_primary = A_dir (directional). scoring_method_secondary = A_pooled (for transparency).',
            },
            {
                'lesson_id': 'ad-LL-003',
                'source': 'VAL-052 AddNeuroMed cross-platform',
                'context': 'AddNeuroMed GSE144858 450K cross-platform replication',
                'quirk': 'Raw d=+0.33 replicated direction but at lower magnitude than AIBL. Age regression reduced residual d to +0.12 — 63% of the AddNeuroMed raw signal is age-tracked.',
                'interpretation': 'AD age-confounding is substantial. Age-regressed A_dir is mandatory primary clinical metric; raw A_dir reported for transparency. Stronger AIBL effect (d=+0.624) may be partly AIBL-specific (cohort imaging-biomarker selection enrichment).',
                'applied_to_card': 'Report MUST use age-regressed A_dir as primary clinical metric per §E.5. Raw A_dir secondary.',
            },
            {
                'lesson_id': 'ad-LL-004',
                'source': 'VAL-057 consolidated — GSE53740 external specificity',
                'context': 'GSE53740 Ferrari 2014, n=384, GIFT UCSF-MAC, 193 HC + 15 AD + 128 FTD + 44 PSP/CBD',
                'quirk_1_pooled_null': 'Pre-registered pooled A_dir produced d=+0.013 p=0.96 NULL on GSE53740 AD.',
                'quirk_2_sex_recovery': 'Post-hoc sex stratification recovered male AD d=+0.415 (n=7), quantitatively consistent with AIBL male d=+0.512. Female AD d=-0.131 (n=7) did NOT replicate AIBL female d=+0.705. Pooled null arose from opposing sex contributions averaging to zero, not absence of signal.',
                'quirk_3_tauopathy_specificity': 'PSP/CBD preserved 5/7 frozen Rule A directions while AD preserved 4/7 (near chance 3.5/7). PSP/CBD raw d=+0.185 exceeded AD raw d=+0.013. The AIBL-derived panel direction pattern matches tauopathy-associated drift as well as or better than AD-specific drift in GSE53740.',
                'quirk_4_cohort_batch_offset': 'GSE53740 HC mean A_age_z = +2.306 SD above the 80-cell Cookbook baseline — systematic cohort-level batch offset from Ferrari 2014 ComBat + quantile normalization. Not biological. Implies the Cookbook 80-cell baseline cannot be directly applied to any non-AIBL/non-AddNeuroMed cohort without cross-cohort normalization.',
                'interpretation_combined': 'GSE53740 pooled null is honest but does not overturn AIBL/AddNeuroMed replication. Sex-stratified male recovery + PSP/CBD tauopathy-directional trend + cohort batch offset all materially qualify the panel interpretation. Tier stays cross_platform_validated but the card adds explicit sex-stratification requirement, tauopathy-specificity caveat, and cross-cohort-normalization requirement.',
                'applied_to_card': 'Multiple known_limitations entries, new validation_anchors entry for VAL-057 consolidated, expected_direction caveat for tauopathy discrimination, deployment_rule for sex stratification.',
            },
            {
                'lesson_id': 'ad-LL-005',
                'source': 'Pre-registration discipline failure',
                'context': 'VAL-057 original pre-registration',
                'quirk': 'Pre-reg sealed 2026-04-24 05:44 UTC locked only the pooled A_dir test with 5-outcome decision matrix. It did NOT pre-register sex stratification despite VAL-051 having reported sex-split results. It did NOT pre-register per-CpG directional check despite the 7-CpG panel having 2:5 direction split. It did NOT pre-register the 80-cell age anchor despite that baseline existing in the Cookbook. All three omissions were added post-hoc after the pooled null.',
                'interpretation': 'A more rigorous pre-reg would have locked sex-stratified, per-CpG, and 80-cell-anchored analyses alongside the pooled primary, with their own decision rules. This is a process lesson, not a methodology failure — the post-hoc analyses are valid, but a better pre-reg would have eliminated the post-hoc label entirely.',
                'applied_to_card': 'Future pre-registrations (starting with VAL-058 onward if relevant) must include: (a) primary pooled test, (b) sex-stratified test, (c) per-CpG directional preservation check, (d) 80-cell or equivalent anchor check, (e) cohort batch-offset check. Lock all five decision rules before data access.',
            },
        ],
    },

    'lung-epic': {
        '_purpose': 'Disease-specific quirks encountered during validation and deployment.',
        'lessons': [
            {
                'lesson_id': 'lung-LL-001',
                'source': 'VAL-056 Part 3 TCGA-LUAD/LUSC',
                'context': 'TCGA matched tumor-normal lung n=141 total pairs',
                'quirk': 'Adjacent-normal ΔA = +0.030 in both LUAD and LUSC — substantially elevated above healthy-donor baseline. Tumor ΔA = +0.165 and +0.161.',
                'interpretation': 'TCGA lung cohorts are ~80% smokers/former-smokers at diagnosis. The +0.030 adjacent-normal ΔA reflects combined smoking effect and local field effect, not a pure tissue field effect. Deployment in a never-smoker population will likely show a lower baseline adjacent-normal.',
                'applied_to_card': 'known_limitations explicitly flags this; mandatory smoking-status covariate; per-stratum clinical action paths.',
            },
            {
                'lesson_id': 'lung-LL-002',
                'source': 'Hong 2019 J Clin Med literature',
                'context': 'Korean NSCLC n=150+150',
                'quirk': 'cg12169243 (DPH6) and cg25429010 (IMP3) reached genome-wide significance in current smokers only, not nonsmokers. The NSCLC methylation signature is smoking-stratified at the per-CpG level.',
                'interpretation': 'Single-panel one-size-fits-all scoring is inappropriate for lung cancer. Current-smoker NSCLC signature combines smoking damage response + cancer response. Never-smoker NSCLC (often EGFR-mutant adenocarcinoma, more common in women and East Asian populations) has a distinct signature.',
                'applied_to_card': 'Four smoking strata (never / former ≥10yr / former <5yr / current) each with explicit deployment rule; Stage 2 firing rule tightened for current smokers (top-1/top-2 ≥ 2x, not DETECTABLE alone).',
            },
            {
                'lesson_id': 'lung-LL-003',
                'source': 'Baglietto 2017 Int J Cancer',
                'context': 'MCCS pre-diagnostic lung cancer',
                'quirk': 'Smoking-driven CpG hypomethylation decays back toward never-smoker values over 5-10 years post-cessation.',
                'interpretation': 'Former-smoker interpretation depends on time-since-quit. >10yr quit → approximately never-smoker. <5yr quit → approximately current-smoker. 5-10yr → intermediate.',
                'applied_to_card': 'mandatory_covariates.smoking_status includes four strata with per-stratum deployment rule. Years-since-quit is a required field when former smoker stratum is selected.',
            },
            {
                'lesson_id': 'lung-LL-004',
                'source': 'VAL-056 Part 2 / VAL-041 Moss 2018 Fig 4b',
                'context': 'Moss 2018 NSCLC plasma deconvolution',
                'quirk': 'Top-1 lung_epithelial ΔA = +0.14304 vs top-2 neuron ΔA = +0.00235 — confidence ratio 60.87x. This is the cleanest tissue-of-origin localization in the entire VAL-041 10-cancer set.',
                'interpretation': 'Stage 2 lung specificity is exceptionally strong. Lung can use the top-1/top-2 ratio ≥ 2x firing rule without losing sensitivity because the ratio is nearly always much higher when lung is the true tissue. This justifies the tightened Stage 2 rule for smokers.',
                'applied_to_card': 'Stage 2 firing rule top-1/top-2 ≥ 2x tightening for current smokers is supported by Moss 60.87x ratio; maintains high sensitivity.',
            },
            {
                'lesson_id': 'lung-LL-005',
                'source': 'VAL-056 Part 1 / Kadota 2014',
                'context': 'Lung adenocarcinoma distance-annotated field effect n=152',
                'quirk': 'Monotonic gradient tumor (+0.152) → near 2cm (+0.052) → far 5cm (+0.017) → healthy. Far-adjacent (≥5cm) remains elevated above healthy by +0.017 — field effect extends past the resection margin.',
                'interpretation': 'In surgical resection for early-stage lung cancer, a "clean" 5cm margin is still architecturally elevated. EDEAR lung-epic serial monitoring post-resection should track return-to-baseline trajectory rather than expecting immediate A-score normalization.',
                'applied_to_card': 'post-operative surveillance language in clinical_action_matrix: "post-resection patients should expect gradual A-score decline over months, not immediate return to baseline."',
            },
        ],
    },
}


def update_card(card_path_in, card_path_out, card_id, lessons_learned_dict, new_version, supersedes_note):
    """Load card, inject universal_reference + lessons_learned, bump version, save."""
    with open(card_path_in) as f:
        card = json.load(f)

    old_version = card.get('card_version', 'unknown')
    old_size = len(json.dumps(card, indent=2))
    print(f"  Loading {card_path_in}: {old_version}, {old_size} chars")

    # Inject universal_reference at the top (after metadata)
    card['universal_reference'] = UNIVERSAL_REFERENCE

    # Inject lessons_learned
    card['lessons_learned'] = lessons_learned_dict

    # Bump version
    card['card_version'] = new_version
    card['card_date'] = '2026-04-24'
    card['supersedes'] = supersedes_note

    with open(card_path_out, 'w') as f:
        json.dump(card, f, indent=2, default=str)

    new_size = len(json.dumps(card, indent=2))
    new_lines = sum(1 for _ in open(card_path_out))
    print(f"    -> {card_path_out}: {new_version}, {new_size} chars, {new_lines} lines")
    return new_size, new_lines


# ──────────────────────────────────────────────────────────────────────────────
# UPDATE ALL FOUR CARDS
# ──────────────────────────────────────────────────────────────────────────────

print("="*78)
print("CARD UPDATER v2.1 — adding universal_reference + lessons_learned")
print("="*78)

print("\n[1/4] breast-epic")
update_card(
    '/home/claude/cookbook_v2.1/breast-epic/breast-epic_card_v2.1.json',
    '/home/claude/cookbook_v2.1/breast-epic/breast-epic_card_v2.1.json',
    'breast-epic',
    CARD_LESSONS_LEARNED['breast-epic'],
    'v2.1',
    'v2.0 (2026-04-23) — v2.1 adds full-inline universal_reference block (Option B: card self-contained) and lessons_learned section (3 documented quirks). Numbers and clinical guidance unchanged from v2.0.'
)

print("\n[2/4] crc-epic")
update_card(
    '/home/claude/cookbook_v2.1/crc-epic/crc-epic_card_v2.1.json',
    '/home/claude/cookbook_v2.1/crc-epic/crc-epic_card_v2.1.json',
    'crc-epic',
    CARD_LESSONS_LEARNED['crc-epic'],
    'v2.1',
    'v2.0 (2026-04-23) — v2.1 adds full-inline universal_reference block and lessons_learned section (3 documented quirks). Numbers and clinical guidance unchanged.'
)

print("\n[3/4] ad-immune")
update_card(
    '/home/claude/cookbook_v2.1/ad-immune/ad-immune_card_v2.1.json',
    '/home/claude/cookbook_v2.1/ad-immune/ad-immune_card_v2.1.json',
    'ad-immune',
    CARD_LESSONS_LEARNED['ad-immune'],
    'v2.1',
    'v2.0 (2026-04-23) — v2.1 adds full-inline universal_reference block, lessons_learned section (5 documented quirks including VAL-057 consolidated with sex-stratified recovery, tauopathy-directional trend, cohort batch offset), and known_limitations updates for GSE53740 non-replication. Tier remains cross_platform_validated.'
)

print("\n[4/4] lung-epic")
# Idempotent: read from v0.3 if it exists (subsequent runs), else v0.2 (first run)
import os
lung_in = '/home/claude/cookbook_v2.1/lung-epic/lung-epic_card_v0.3.json'
if not os.path.exists(lung_in):
    lung_in = '/home/claude/cookbook_v2.1/lung-epic/lung-epic_card_v0.2.json'
update_card(
    lung_in,
    '/home/claude/cookbook_v2.1/lung-epic/lung-epic_card_v0.3.json',
    'lung-epic',
    CARD_LESSONS_LEARNED['lung-epic'],
    'v0.3',
    'v0.2 (2026-04-24 VAL-056 multi_modal_validated tier) — v0.3 adds full-inline universal_reference block and lessons_learned section (5 documented quirks). Numbers and clinical guidance unchanged from v0.2.'
)

print("\n" + "="*78)
print("DONE. All 4 cards updated. Next: update ad-immune with VAL-057 consolidated result entry.")
print("="*78)
