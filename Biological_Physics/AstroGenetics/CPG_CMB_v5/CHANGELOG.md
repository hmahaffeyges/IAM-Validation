# CPG pipeline update — disease-wall matcher + RUN-everything sweep

## 1. Per-cell directional matcher  (walther_clinical.py, 1521 -> 1554 lines)
Replaced the absolute-magnitude cosine (whose |dep|>=0.15 floor gated out subtle
pre-dx directional signal) with a weighted directional matcher over each disease's
SIGNAL cells (|Cohen d| >= 0.20). 'cosine' now carries directional concordance in
[-1,+1] for downstream compatibility; the report is untouched.
- Hardened specificity gate: the neutrophil-to-lymphocyte / progenitor-expansion axis
  (myeloid-up, progenitor-up, lymphoid-down) is now NON_SPECIFIC_GENERIC. A tissue/origin
  cell or a lineage break makes a match SPECIFIC.
- STRONG needs dc>=0.70, coverage>=0.40, >=3 moved cells, mean |dep|>=0.15.
- Verified: injected myeloma -> flags multiple_myeloma (specific, strong); injected AML ->
  correctly NON_SPECIFIC (blood pattern alone cannot name AML vs CML vs reactive); the three
  subtle bundles produce no false concern.

## 2. RUN-everything residual sweep  (stage_5_second_chain.py, 284 -> 349 lines)
The second chain no longer fires only on the per-cell top match. It now ALWAYS sweeps every
available residual map (breast, AD, immune cross-disease universal alarm) for a whole-blood-
compatible patient, independent of the per-cell rank. This is the fix for the breast pre-dx
miss: that case's signal is distributional (secretory homogenization), so the per-cell matcher
correctly leaves it quiet and the matched filter must screen it on its own.
- A DETECTION is a POSITIVE-rho fire (consistent direction). A negative-rho fire is an
  anti-correlation, NOT a detection. Null check across EPIC-Italy healthy controls confirmed
  healthy blood anti-correlates with AD/immune maps (rho ~ -0.1 to -0.17) and sits at zero on
  the breast map, while breast cases fire breast positive (+0.058, +0.093).
- The confirmation verdict now flags only the top SPECIFIC, concern-worthy per-cell match;
  non-specific generic-axis matches are handled by the report's Mode 1 line, never escalated.

## 3. Report  (cpg_report_builder.py)
- New 'C - RUN-everything residual sweep' table in the Confirmation section (per-map rho, CI,
  CpGs, detection). None-safe for the no-per-cell-flag case.
- Updated the non-specific Mode 1 line to the generic-axis wording (was myeloid/lymphoid).

## 4. Doctor-facing mahalanobis cleanup (2 surgical strings)
- breast-epic_card_v3_1.json honest_limitations: "Universal Mahalanobis may capture..." ->
  "The universal architectural screen (now the residual matched-filter sweep) may capture...".
  Honest point preserved; only the removed-mechanism name corrected.
- flowchart_v4.html component list: mahalanobis -> matched_filter.
- LEFT INTACT: the matrix CSV and breast-card validation records that cite Mahalanobis d-values
  (e.g. d=+1.876/+2.097). Those are accurate validation HISTORY, not stale mechanism, and were
  not altered. A holistic card v3.2 reconciliation (matched filter throughout) is a separate
  discussion, not a unilateral edit.

## Validation summary
- breast pre-dx (GSM1235926): per-cell matcher quiet (correct); residual sweep fires breast
  (+0.058, CI [+0.001,+0.114]); AD + immune null. CAUGHT.
- healthy control (GSM1235534): second chain gate closed, no false flag.
- positive control (injected MM): flags multiple_myeloma, confirmed.

## 5. Systemic stress / inflammatory wellness signal  (walther_clinical.py + report)
New detect_systemic_stress_pattern(patient_departure): a wellness-level read (NONE / MILD /
NOTABLE) of the neutrophil-to-lymphocyte axis (myeloid + progenitor up, lymphoid down). It is
NEVER a disease call. Fires only on a coherent, real-magnitude pattern (n>=4 axis cells, mean
|dep|>=0.10, coherence>=0.60); flat noise and incoherent departures stay quiet.
- Wired into run_pipeline -> bundle["systemic_stress"].
- Report executive summary now carries "Wellness signal (Mode 3)": an actionable, non-alarming
  callout (lifestyle, weight, diet, trajectory monitoring, family-history vigilance) when the
  pattern is present. This turns the formerly dead-end non-specific generic pattern into the
  "act on it early" signal -- the whole point of pre-diagnostic detection.
- Calibration: flat-healthy and incoherent -> NONE; breast pre-dx / HCC / CRC -> NOTABLE.
  Large healthy-cohort calibration is honest future work.

## 6. Patient straw man  (deliverables/IAM_Patient_StrawMan.html)
The companion to the crown jewel: the patient's own per-cell architecture on the SAME eight-class
grid, so the two prints can be laid side by side.
- Top row = the patient's bloodwork: A-departure per scorable cell, GREEN = healthy band
  (A 0.93-1.07), red = elevated, blue = suppressed, intensity by magnitude. A bright outline
  marks a CONFIDENT departure (95% CI clears the band); a hatch marks mild / CI-uncertain.
- Beneath it, every disease the patient flagged is pulled straight from the crown jewel (here the
  full breast trajectory) for direct shape comparison. The wellness stress banner sits on top.
- Builders included under builders/ so the wall regenerates for any patient bundle.

## 6b. Patient straw man tier correction
Patient cells are now coloured by the FIVE gauge tiers (cpg_gauge.py / tier_breakpoints v1.3),
not a flat green band: SUPPRESSED <0.95, NORMAL 0.95-1.04 (green/healthy), ELEVATED 1.04-1.07
(amber), SIGNIFICANTLY_ELEVATED 1.07-1.10 (orange, past the Warburg line), BREACH >=1.10 (red).
Each cell shows its A-score; a white outline marks a confident departure (95% CI clears NORMAL),
a hatch marks CI-uncertain. This matches the report gauge exactly.
NOTE: the tier file puts the NORMAL->ELEVATED edge at 1.04 (used here), not 1.05.
