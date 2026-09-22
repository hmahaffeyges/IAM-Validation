"""
Canonical universal_reference block injected into every EDEAR disease card.
Full-inline (Option B): each card self-contained. A new analyst loading any
single card + GAPE_WEB_v13.py can run the full pipeline end-to-end without
cross-referencing any other file.
"""
import json

UNIVERSAL_REFERENCE = {
    "_purpose": (
        "Self-contained reference block embedded in every EDEAR disease card. "
        "Full-inline Option B: do not rely on README_MASTER or any external file "
        "for pipeline invariants. A new analyst loading only this card + "
        "GAPE_WEB_v13.py must be able to reproduce the full Stage 1 + Stage 2 + "
        "Stage 3 pipeline without reading anything else."
    ),
    "schema_version": "universal_reference_v1.0",
    "last_updated": "2026-04-24",

    "universal_stage_1_pipeline": {
        "invariant_rule": (
            "Every patient's first test is identical regardless of suspected disease: "
            "Stage 1 is always immune-class on buffy-coat DNA using the Xu-538 panel "
            "with H_min(immune) = 0.838889. This applies to every EDEAR card. What "
            "varies per card: expected direction, Stage 2 tissue target, tier "
            "thresholds, mandatory covariates, clinical action paths."
        ),
        "specimen": "buffy coat DNA (peripheral blood leukocytes, ~70% immune cells)",
        "platforms_supported": ["Illumina 450K", "Illumina EPIC 850K"],
        "panel_id": "Xu2020_breast_cancer_replicated_full",
        "panel_sha256": "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6",
        "panel_n_cpgs": 538,
        "panel_source_paper": "Xu Z, Sandler DP, Taylor JA. JNCI 2020;112(1):87-94",
        "panel_source_doi": "10.1093/jnci/djz065",
        "panel_source_doi_url": "https://doi.org/10.1093/jnci/djz065",
        "panel_coverage_epic": "approximately 500/538 (93%)",
        "panel_coverage_450k": "538/538 (100%)",
        "h_min_immune": 0.838889,
        "h_min_source": "G-003b MCMC posterior mean, R-hat < 1.001 (frozen)",
        "scoring_method_primary": "pooled entropy A-score = mean(H(β)/H_min) across panel",
        "scoring_method_secondary_for_bidirectional_diseases": (
            "directional A_dir = mean(direction_i × z_i) when disease drives "
            "bidirectional per-CpG drift. Applicable to AD via 7-CpG Rule A panel. "
            "Other cards use pooled entropy as primary."
        ),
    },

    "universal_h_min_table": {
        "_note": (
            "Class H_min constants from GAPE_WEB_v13.py lines 87-96. Frozen at G-003b "
            "MCMC posterior means, R-hat < 1.001. Used at Stage 2 to compute per-tissue "
            "A-scores from deconvolved β values."
        ),
        "cycling": 0.856055,
        "secretory": 0.843264,
        "immune": 0.838889,
        "terminal": 0.772837,
        "stromal": 0.862950,
        "stem_adult": 0.873718,
        "progenitor": 0.852216,
        "stem_pluri": 0.982166,
    },

    "universal_stage_2_moss_deconvolution": {
        "method": "Moss 2018 NNLS deconvolution of the Stage 1 IDAT into 18 per-tissue β values",
        "scipy_call": "scipy.optimize.nnls on Moss 2018 reference matrix",
        "production_module_status": (
            "G-DECONV-001 OPEN-DEFERRED. VAL-041 proved the workflow at published-β "
            "level (10/10 top-1 correct localization across 10 cancer types). "
            "Production per-IDAT deployment requires: (1) 30 MB Moss 2018 reference "
            "matrix locked into the repo; (2) Salas 2018 QC harness implemented; "
            "(3) assay version tag (L1 Illumina EPIC + Moss markers, L2 custom "
            "capture panel, L3 full 5-substrate MESA+DELFI)."
        ),
        "tissue_classes": {
            "colon_epithelial": "cycling",
            "lung_epithelial": "cycling",
            "gastric_epithelial": "cycling",
            "bladder_epithelial": "cycling",
            "cervical_epithelial": "cycling",
            "kidney_epithelial": "cycling",
            "hepatocyte": "secretory",
            "pancreatic_exocrine": "secretory",
            "breast_ductal": "secretory",
            "prostate_epithelial": "secretory",
            "neuron": "terminal",
            "oligodendrocyte": "terminal",
            "vascular_endothelial": "stromal",
            "fibroblast": "stromal",
            "neutrophil": "immune",
            "lymphocyte": "immune",
            "monocyte": "immune",
            "hsc": "stem_adult",
        },
        "healthy_reference_beta_by_tissue": {
            "_source": "Moss 2018 Table S1",
            "colon_epithelial": 0.741,
            "lung_epithelial": 0.738,
            "gastric_epithelial": 0.739,
            "bladder_epithelial": 0.737,
            "cervical_epithelial": 0.740,
            "kidney_epithelial": 0.739,
            "hepatocyte": 0.742,
            "pancreatic_exocrine": 0.738,
            "breast_ductal": 0.744,
            "prostate_epithelial": 0.743,
            "neuron": 0.779,
            "oligodendrocyte": 0.775,
            "vascular_endothelial": 0.731,
            "fibroblast": 0.728,
            "neutrophil": 0.762,
            "lymphocyte": 0.751,
            "monocyte": 0.758,
            "hsc": 0.734,
        },
    },

    "universal_stage_3_epidish_subcomposition": {
        "method": "Teschendorff 2017 EpiDISH RPC mode, Salas 2018 reference",
        "cell_types_resolved": ["CD4+ T", "CD8+ T", "NK", "B", "monocyte", "neutrophil"],
        "when_applied": (
            "Runs when Stage 1 flags AND Stage 2 returns no solid-organ localization. "
            "Distinguishes chronic inflammation (neutrophil shift), hematologic immune "
            "drift (lymphocyte composition shift), autoimmune patterns, AD-type patterns "
            "(brain tissue not in buffy coat)."
        ),
        "salas_qc_bounds": {
            "neutrophil_fraction": [0.45, 0.75],
            "lymphocyte_fraction": [0.20, 0.40],
            "monocyte_fraction": [0.03, 0.12],
            "cd4_fraction": [0.10, 0.30],
            "cd8_fraction": [0.05, 0.25],
            "nk_fraction": [0.03, 0.15],
            "b_fraction": [0.03, 0.15],
        },
        "qc_gate_rule": "IDATs outside Salas bounds get QC flag; Stage 2 output not released",
    },

    "universal_80_cell_age_baseline_immune_class": {
        "_purpose": (
            "Age-decade mean and SD of pooled-entropy immune-class A-score in healthy "
            "reference cohorts. Used to compute age-matched percentile for Stage 1 tier "
            "calls. Applies to every card's NORMAL / MARGINAL / DETECTABLE tier logic."
        ),
        "_sources": [
            "Hannum 2013 Mol Cell (GSE40279 blood methylation aging)",
            "Horvath 2013 Genome Biol (multi-tissue age clock)",
            "Roadmap Epigenomics Consortium 2015 Nature (127 reference epigenomes)",
            "Moss 2018 Nat Commun (cfDNA tissue-of-origin atlas)",
            "Lister 2013 Science (frontal cortex neuron reference)",
            "Alisch 2012 Genome Research (pediatric age-associated methylation)",
        ],
        "_critical_caveat_cross_cohort": (
            "This baseline is derived from the source cohorts using standard preprocessing "
            "(primarily minfi/sesame normalization). Cohorts with different preprocessing "
            "pipelines (e.g. Ferrari 2014 GSE53740 with ComBat + quantile normalization) "
            "can show cohort-level batch offsets of +2 SD or more relative to this "
            "baseline. Example: VAL-057 found GSE53740 HC sit at A_age_z = +2.306 vs "
            "this baseline. Any card deployment on a non-AIBL/non-AddNeuroMed cohort "
            "requires either (a) re-anchoring to within-cohort HC, or (b) a normalization "
            "bridge to the 80-cell scale. Do not assume universal applicability of these "
            "constants across preprocessing pipelines."
        ),
        "age_decades": {
            "00-09": {"A_mean": 0.9402, "A_sd": 0.0291},
            "10-19": {"A_mean": 0.9468, "A_sd": 0.0305},
            "20-29": {"A_mean": 0.9531, "A_sd": 0.0318},
            "30-39": {"A_mean": 0.9590, "A_sd": 0.0334},
            "40-49": {"A_mean": 0.9618, "A_sd": 0.0356},
            "50-59": {"A_mean": 0.9638, "A_sd": 0.0368},
            "60-69": {"A_mean": 0.9652, "A_sd": 0.0380},
            "70-79": {"A_mean": 0.9671, "A_sd": 0.0394},
            "80-89": {"A_mean": 0.9688, "A_sd": 0.0403},
            "90-99": {"A_mean": 0.9710, "A_sd": 0.0415},
        },
    },

    "universal_tier_thresholds": {
        "_source": "80-cell healthy baseline reference; same across breast-epic, crc-epic, ad-immune, lung-epic",
        "_note": (
            "Per-card tier thresholds MAY deviate from these universals if disease-"
            "specific literature justifies. When they deviate, the card MUST cite the "
            "deviation explicitly. Silent deviation is forbidden."
        ),
        "NORMAL": {"A_threshold": "< 1.01", "action": "No action; serial-sample per screening cadence"},
        "MARGINAL": {"A_threshold": "≥ 1.01", "action": "Note; serial-sample in 6 months"},
        "DETECTABLE": {"A_threshold": "≥ 1.05, age-percentile ≥ p90", "action": "Run Stage 2 localization"},
        "URGENT": {"A_threshold": "≥ 1.07, age-percentile ≥ p90", "action": "Run Stage 2; expedited workup per disease card"},
        "FLOOR_BREACH": {"A_threshold": "≥ 1.10", "action": "Run Stage 2; urgent clinical workup regardless of localization"},
    },

    "universal_sex_stratification_rule": {
        "rule": (
            "Every card's deployment MUST stratify reports by sex when sex-differential "
            "signal has been documented in validation. At v2.1 this applies to ad-immune "
            "(VAL-051: female d=+0.71 vs male d=+0.51 AIBL; VAL-057 male recovery, female "
            "non-replication in GSE53740). For other cards, sex stratification is "
            "recommended in deployment even where no sex-differential has been documented, "
            "since absence of evidence is not evidence of absence."
        ),
        "minimum_reporting": "Report must include patient sex. Card scoring MAY differ by sex if documented in validation.",
    },

    "universal_language_discipline": {
        "allowed": [
            "consistent with", "tested against", "data are consistent with",
            "architectural signal detected", "elevated above age-matched baseline",
            "predictions within the framework", "flag that changes downstream workup",
        ],
        "forbidden": [
            "confirms", "validates", "proves", "diagnoses", "first derivation",
            "154 years no one has", "resolves", "definitive", "cure", "treat",
        ],
        "rationale": (
            "Every report must present EDEAR output as a flag that changes downstream "
            "workup, never as a diagnosis. Final diagnosis requires clinical assessment "
            "plus standard-of-care workup (imaging, tissue biopsy, amyloid PET/CSF for "
            "AD, etc.) per the relevant clinical guidelines."
        ),
    },

    "universal_cohort_batch_offset_warning": {
        "_critical": True,
        "_discovered_in": "VAL-057 (2026-04-24)",
        "description": (
            "VAL-057 established that cohort-level preprocessing artifacts can shift "
            "pooled-entropy A-score by +2 SD or more relative to the 80-cell Cookbook "
            "baseline. This is a systematic offset, not a biological finding. Applies "
            "to any new cohort before its baseline can be used for cross-cohort "
            "comparison."
        ),
        "example": (
            "GSE53740 (Ferrari 2014, 450K + ComBat + quantile normalization): "
            "HC mean A_age_z = +2.306 vs 80-cell immune baseline."
        ),
        "deployment_rule": (
            "(1) On AIBL or AddNeuroMed-equivalent preprocessing: 80-cell baseline "
            "usable directly. (2) On any other cohort: re-anchor to within-cohort HC "
            "for tier thresholds OR run a normalization bridge to the 80-cell scale "
            "before applying Cookbook tier thresholds."
        ),
    },

    "universal_no_fabrication_rule": {
        "rule": (
            "Every numeric value, CpG ID, cohort size, validation result, or citation "
            "in this card traces to a primary source or a SHA-locked validation run. "
            "Nothing is invented or estimated. If a value is a projection or estimate, "
            "it is explicitly labeled as such in its field."
        ),
    },

    "gape_web_version_reference": {
        "canonical_file": "GAPE_WEB_v13.py",
        "h_min_constants_line_range": "87-96",
        "port": 8080,
        "frozen_at": "2026-04-23",
    },
}


def get_universal_reference_json_str(indent=2):
    return json.dumps(UNIVERSAL_REFERENCE, indent=indent, ensure_ascii=False)


if __name__ == '__main__':
    print(get_universal_reference_json_str())
