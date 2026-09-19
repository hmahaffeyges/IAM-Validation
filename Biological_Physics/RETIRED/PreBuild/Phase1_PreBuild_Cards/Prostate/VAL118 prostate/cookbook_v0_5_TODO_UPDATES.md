# CROSS_CARD_CALIBRATION_TODO_v0_5 — proposed v0.6 surgical additions

This is a delta-only memo for Heath. It enumerates what v0.6 should add to the existing v0.5 TODO based on prostate-epic v0.3 sprint findings. Heath signs off; Walther applies the v0.6 edits surgically (preserves v0.5 verbatim).

## Proposed v0.6 additions

### Guardrail #14 — Cell-of-origin atlas tile preregs use magnitude-based |d| with direction labels

Adds DISC-PROSTATE-002 / CHK-2.7 / prostate-LL-007 as a top-level guardrail. Cookbook-wide rule.

### Pre-flight checklist additions

- **Substrate-floor pre-lock check (CHK-2.8)** — before pre-locking CHK-3.1B coverage threshold, identify substrate floor for the calibration cohort. Default 95% is wrong for TCGA HM450K sesame Level 3 (substrate floor ~80% per VAL-117 + cardio precedent). EPIC 850K and minfi-preprocessFunnorm have different floors.
- **Direction-ambiguity pre-flight** — for any cell-of-origin atlas tile, ask: is the direction biologically uniform or possibly bidirectional? If bidirectional, require magnitude-based threshold + direction labels per CHK-2.7.

### DISC-PROSTATE-NNN section

Adds three new findings to the inventory of cross-card discoveries:
- DISC-PROSTATE-001 (gene-promoter atlas family fitness extends DISC-CARDIO-004)
- DISC-PROSTATE-002 (magnitude-based threshold rule formalized as CHK-2.7)
- DISC-PROSTATE-003 (LE tile reads tumor strongly NEGATIVE = luminal dedifferentiation)

### Reference example: prostate-epic card — DONE 2026-04-30

Adds prostate-epic v0.3 as the second worked example after cardio-epic. Phase 0 / A.1-A.4 / B / C / D / E / F all walked through with VAL-117 + VAL-118 specifics.

### Master template clarification

Phase A.3 (bridge engineering) — add explicit example: ProstateRef Entrez→array CpG bridge using EpiSCORE probeInfo450k.lv via R extraction (rpy2 not required if R is run separately and probeInfo CSV is staged). Bridge engineering reusable infrastructure; same template that produced HeartRef + BreastRef bridges.

### Wave 4 cards — prostate-epic now MOVED from Wave 4 to DONE

Updates the execution order table. Remaining Wave 4: kidney-epic, glioma-epic. Kidney sprint should benefit from ProstateRef-and-cardio precedent on the gene-promoter atlas family fitness question.

## Total proposed v0.6 additions

~50-80 lines of surgical inserts to v0.5; preserves all v0.5 content verbatim per Heath's standing rule. Walther does NOT apply these edits without Heath's explicit sign-off.

