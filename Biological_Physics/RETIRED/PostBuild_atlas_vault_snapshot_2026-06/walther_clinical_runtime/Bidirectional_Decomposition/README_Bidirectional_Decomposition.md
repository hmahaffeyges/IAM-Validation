# Bidirectional_Decomposition — Stage 4.5 module

**Module:** `bidirectional_decomposition.py`
**Stage:** 4.5 — Bidirectional decomposition BEFORE pooled A-score (catches the VAL-050/051 cancellation pattern at patient runtime)
**Reference VAL:** [val_051_ad_directional](../../validation_runs/val_051_ad_directional/) — the sealed validation that motivated this stage

## What this module does

Implements the four-step bidirectional discipline that the pre-build cookbook documented:

1. **Direction multiplication** — each CpG's z-score (vs frozen training-set HC mean/SD) is multiplied by its frozen sign (+1 for "up in disease", −1 for "down in disease")
2. **Aggregation across all panel CpGs** — direction-multiplied z-scores are averaged into a single signed composite
3. **Partial-coverage gating** — if fewer than 70% of panel CpGs are present in the patient's β data, the directional composite is not reported
4. **CHK gate** — the composite is paired with the pooled-entropy A-score from Stage 4 to detect the bidirectional-cancellation signature

## Why this matters

At Stage 4, the pooled-entropy A-score (`A = H(β_mean) / H_min`) is **direction-agnostic** because Shannon entropy is symmetric around β=0.5. When a disease produces a bidirectional pattern — some CpGs going UP, others going DOWN — the pooled β_mean barely moves and the pooled A-score reads NULL.

**VAL-050** hit this exactly: on the 18-CpG IMM panel applied to AIBL AD vs HC, pooled-entropy A returned `d = +0.077` (effectively null). **VAL-051** then recovered the same signal at `d = +0.624` using a directional weighted composite z-score on a 7-CpG sub-panel. **The directional decomposition is what made the AD-instance immune pattern visible.**

The VAL discipline catches this because every VAL has a PREREG specifying direction. Patient runtime has no PREREG per patient — the engine MUST autonomously decompose and decide. That's what Stage 4.5 does.

## The sealed formula

This module mirrors `val051_analyze.py:112-121` exactly:

```python
def a_dir_score(patient_beta, panel_cpgs_with_stats):
    contribs = []
    for cpg, r in panel_cpgs_with_stats.items():
        b = patient_beta.get(cpg)
        if b is None or not (0 < b < 1):
            continue
        z = (b - r['mean_hc_train']) / r['sd_hc_train'] if r['sd_hc_train'] > 0 else 0
        contribs.append(r['direction'] * z)
    if len(contribs) < max(3, int(0.7 * len(panel_cpgs_with_stats))):
        return None
    return sum(contribs) / len(contribs)
```

- **Positive composite** → patient methylation matches the disease direction
- **Negative composite** → patient methylation matches the HC direction (anti-disease)
- **Near-zero composite** → no directional signal

## Bidirectional flag rule

`FLAG_BIDIRECTIONAL` fires when:
- Pooled-entropy A-score is **near baseline** (within ±0.05 of 1.0)
  AND
- |directional composite z-score| > 0.40

Translation: pooled is mute but directional is loud. That's the cancellation signature.

When flagged, Stage 7 tier reporting uses the directional composite (with sign + magnitude) rather than the pooled A-score. The customer's report says "mixed-direction immune pattern detected" instead of "immune A-score normal."

## v1.0 panels — what's sealed, what's pending

| Class | Sealed panel? | Source VAL | n_CpGs |
|---|---|---|---|
| immune | ✅ YES | VAL-051 Rule A | 7 (2 pos + 5 neg) |
| stem_pluri | ❌ NO | future | — |
| stem_adult | ❌ NO | future | — |
| stromal | ❌ NO | future | — |
| progenitor | ❌ NO | future | — |
| cycling | ❌ NO | future | — |
| secretory | ❌ NO | future | — |
| terminal | ❌ NO | future | — |

**Honest disclosure:** v1.0 has bidirectional decomposition for the immune class only. The 7 other classes return `NO_PANEL` at Stage 4.5 — Stage 4 pooled-entropy A-score is the only A-score reported for those classes until future sealed VALs populate per-class directional panels.

Future expansion roadmap:
- CPG-VAL-019 (in v1.0 VAL set) — cancer-positive vs AD-negative direction discrimination → broadens the immune-class panel beyond AD-direction-only
- Per-card directional panels for breast, AD, kidney, cardio (and beyond) as the CPG-VAL series produces sealed evidence

## Engine integration

Called from `walther_clinical.py` at Stage 4.5 (after Stage 4 pooled A-scoring, before Stage 5 Mahalanobis and Stage 7 tier breakpoints):

```python
from bidirectional_decomposition import (
    load_directional_panels,
    compute_per_class_bidirectional_decomposition,
    save_bidirectional_report,
)

panels = load_directional_panels(
    "Bidirectional_Decomposition/directional_panels_v1_0.json"
)
report = compute_per_class_bidirectional_decomposition(
    patient_beta=patient_beta_cleaned,    # Stage 3 output (foreground-cleaned)
    panels=panels,
    patient_id=patient_metadata["patient_id"],
)

if report.any_bidirectional_flagged:
    # Customer-facing tier at Stage 7 uses the directional composite
    # for the flagged classes (typically immune class only in v1.0)
    flagged = report.flagged_classes  # e.g. ["immune"]
    ...

save_bidirectional_report(report, out_dir=f"reports/{patient_id}/stage_4_5/")
```

## Panel JSON schema

`directional_panels_v1_0.json` schema:

```json
{
  "version": "v1.0",
  "panels": {
    "immune": {
      "panel_id": "VAL-051 Rule A 7-CpG AD-direction-anchored",
      "panel_source_val": "VAL-051",
      "panel_sha256_anchor": "52061285...",
      "h_min": 0.838889,
      "pooled_panel_cpgs": ["cg00431549", "cg01127300", ...],   // 18-CpG parent panel
      "cpgs": [
        {
          "cpg_id": "cg16867657",
          "direction": 1,
          "mean_hc_train": 0.7309,
          "sd_hc_train": 0.0474,
          "q_fdr": 8.37e-05,
          "delta_beta": 0.0246
        },
        ...
      ]
    },
    "stem_pluri": null,
    ...
  }
}
```

CpG-level fields:
- **cpg_id** — Illumina probe ID (cgNNNNNNNN)
- **direction** — +1 (up in disease) or −1 (down in disease); frozen at VAL training time
- **mean_hc_train** — training-set HC mean β (the z-score reference center)
- **sd_hc_train** — training-set HC SD β (the z-score reference scale)
- **q_fdr** — FDR q-value at panel selection (audit only)
- **delta_beta** — training-set Δβ = mean(disease) − mean(HC) (audit only)

## Validation

- **Syntax** — `python -c "import ast; ast.parse(open('bidirectional_decomposition.py').read())"` passes
- **Smoke test** — `python bidirectional_decomposition.py --panel-json directional_panels_v1_0.json --smoke-test` loads panels for 8 classes (1 sealed, 7 pending), runs synthetic patient (β=0.5 uniform), confirms immune panel produces composite = −2.098 (anti-AD direction) as expected (β=0.5 is below the HC training mean for most panel CpGs)
- **Sealed-formula match** — the `score_directional_composite` function reproduces `val051_analyze.a_dir_score` line-for-line. Verifiable by importing this module + re-running VAL-051 holdout against `aibl_imm_betas.json` and confirming `A_dir_A` matches the sealed `VAL_051_RESULTS.json` per-sample numbers.

## File integrity

| File | SHA-256 anchor |
|---|---|
| `val051_panel_ruleA.json` (sealed) | `52061285fc97bfff871ba7b62f625b14d953bccf25ee24e35f328e15b9827998` (per `VAL_051_SEAL.txt`) |
| `val051_analyze.py` (sealed formula source) | `6e2c820ffd483e1a1939c74e0f71b2da653bed6ee588ed2148df1189bb64c41d` (per `VAL_051_SEAL.txt`) |

The panel data in `directional_panels_v1_0.json` is a direct transcription of `val051_panel_ruleA.json` plus the 18-CpG pooled-comparator panel from `val051_panel_ruleB.json`. No new training, no new selection — this is the SEALED panel made operational at runtime.
