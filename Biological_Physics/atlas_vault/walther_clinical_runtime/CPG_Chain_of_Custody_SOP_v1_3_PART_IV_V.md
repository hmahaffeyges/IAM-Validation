# CPG Chain-of-Custody SOP — Part IV + Part V (Failure modes + Reference) (v1.2 — walkthrough aligned)

> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them. Where a step's logic currently lives inside `GAPE_WEB_v13.py` (the
> production engine — see `SYSTEM_INVENTORY.md`), the SOP now says so honestly. Where
> a step's runtime artifact or output file location was invented, the SOP now reads
> "TBD per orchestrator design" — that orchestrator (working name `web.commercial.py`)
> is a separate conversation Heath and Walther will have. All real paths in this v1.1
> are documented in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be
> treated as not-yet-existing until verified against the repo.

---


**Continues from Part III (§80-§91). This document contains §92-§96 (failure modes & decision trees) and §97-§102 (reference tables, glossary, H_min canonical, plate cross-reference, change log).**

---

# Part IV — Failure modes & decision trees

Part II established every step. Part III established the audit machinery. Part IV does what neither could in isolation: it tells the operator what to DO when something goes wrong. The discipline: **every failure mode has a declared response.** No silent degradation. No "I'll figure it out at the report."

The structure of Part IV is failure-mode-first rather than step-first because real operations don't fail nicely along step boundaries. A sample with a sex mismatch (§18) might also have low call rate (§17) and a borderline cross-method gate (§33) — three flags from three different chain links, requiring a single coherent response. Part IV organizes the response logic around the failure pattern, not the step.

---

## §92. Failure mode catalog by stage

This is the master table of every failure mode declared across §11-§91, organized by stage, with the canonical response for each.

### Stage 0 (Intake) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Missing IDAT pair | §11 | HARD | Reject; require new sample submission |
| Zero-byte or truncated IDAT | §11 | HARD | Reject; transmission failure |
| Missing manifest entry | §12 | HARD | Reject; require complete metadata |
| Schema-mismatched manifest | §12 | HARD | Reject; require manifest correction |
| Cleartext PII in manifest | §12 | HARD | Reject; security violation |
| IDAT integrity hash mismatch | §13 | HARD | Quarantine; investigate file transmission/storage |
| Bisulfite Conversion I/II control fail | §14 | HARD | Reject sample; wet-lab BS conversion failed |
| Specificity I/II control fail | §14 | HARD | Reject sample; wet-lab specificity failed |
| Non-polymorphic control deviation | §14 | SOFT | Flag for review; do not auto-reject |
| Negative control elevated | §14 | SOFT | Flag for contamination review |
| Detection p-value >5% probes failed | §15 | HARD | Reject sample; DNA quality issue |
| Detection p-value 1-5% probes failed | §15 | SOFT | Proceed with `detection_qc=marginal` flag |
| Bead count >5% probes failed | §16 | HARD | Combine with §15; if both fail, reject |
| Sample-level call rate <95% | §17 | HARD | Reject |
| Sample-level call rate 95-98% | §17 | SOFT | Proceed with `call_rate=marginal` flag |
| Sex check MISMATCH | §18 | HARD | Quarantine; chain-of-custody investigation |
| Sex check AMBIGUOUS | §18 | SOFT | Flag for review; possibly atypical karyotype |

### Stage 1 (Calibration & β) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Dye-bias correction factor extreme | §20 | SOFT | Flag for review |
| Normalization method failure (funnorm) | §21 | SOFT | Fall back to noob |
| Normalization failure (both funnorm + noob) | §21 | HARD | Quarantine; sample is uncalibratable |
| ComBat with confounded design | §22 | HARD | Halt; reject cohort design |
| Insufficient samples per batch for ComBat | §22 | SOFT | Skip ComBat with warning; downstream confidence penalty |
| Bisulfite efficiency < 0.95 | §23 | HARD | Retroactive quarantine of sample |
| Bisulfite efficiency 0.95-0.98 | §23 | SOFT | Flag as marginal |
| β values outside [0,1] | §24 | HARD | Quarantine; arithmetic error upstream |
| β distribution unimodal | §25 | HARD | Quarantine; sample distribution anomalous |
| Bimodality coefficient 0.45-0.55 | §25 | SOFT | Flag as marginal |

### Stage 2 (Deconvolution) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Atlas SHA mismatch | §28 | HARD | Halt; atlas reference corrupted |
| Atlas file missing | §28 | HARD | Halt; atlas must be present |
| Marker pool empty for a class | §29 | SOFT | Flag class as INSUFFICIENT_COVERAGE |
| Marker SHA mismatch | §29 | HARD | Halt; marker artifact corrupted |
| NNLS does not converge | §30 | HARD | Quarantine patient |
| Residual MAE > 0.10 | §30 | SOFT | Flag as POOR_DECONVOLUTION; require §33 gate verification |
| Per-class confidence <0.5 broadly | §31 | SOFT | Flag for substrate review |
| NILC does not converge | §32 | SOFT | Cross-method gate becomes single-method; reduced confidence |
| Cross-method gate FLAG | §33 | SOFT | Proceed with cross-method-uncertainty annotation |
| Cross-method gate FAIL | §33 | HARD | Quarantine; manual review required |
| Cohort-wide FLAG rate >5% | §33 | HARD | Halt cohort; investigate systematic |

### Stage 3 (Foreground subtraction) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Patient age outside age-axis calibration range | §35 | SOFT | Apply with `age_extrapolation` flag |
| Age layer SHA mismatch | §35 | HARD | Halt; age layer corrupted |
| Sex/batch/ancestry/smoking foreground unimplemented | §36-§39 | DOCUMENTED GAP | Pass-through with module-status flag |

### Stage 4 (A-score) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Class has <20 markers in QC-passed pool | §41 | SOFT | Flag class A-score with INSUFFICIENT_MARKERS |
| Cell type has <20 markers | §44 | SOFT | Flag cell-type A-score |
| Cell type has <60% coverage | §44 | SOFT | Status MARGINAL_COVERAGE |
| Cell type has <20% coverage | §44 | HARD | Status NO_MARKER_OVERLAP; A-score not returned |
| H_min file SHA mismatch | §43 | HARD | Halt; canonical reference corrupted |
| A-score < 0 | §43 | HARD | Halt; calculation bug |
| Disease panel CpGs <60% available | §45 | SOFT | Flag panel A-score with INSUFFICIENT_MARKERS |

### Stage 5 (Mahalanobis) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| >15 cell types imputed | §47 | HARD | Flag patient INSUFFICIENT_DATA; do not return Mahalanobis |
| 6-15 imputed | §47 | SOFT | Proceed with PARTIAL_DATA flag |
| HC reference SHA mismatch | §48 | HARD | Halt |
| MAHALANOBIS_NUMERIC_FAIL | §49 | SOFT | Flag; report omits the headline |
| Cohort-wide imputation >10% patients with >5 imputations | §47 | HARD | Flag cohort substrate-atlas mismatch |

### Stage 6 (Cellular age) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Age reference SHA mismatch | §53 | HARD | Halt |
| Per-class A out of baseline range (saturation) | §54 | DATA SIGNAL | Report with SAT_HIGH or SAT_LOW status; not a bug |
| Patient chronological age outside reference (4-95) | §57 | SOFT | Flag with AGE_OUT_OF_RANGE; report boundary percentile |
| Baseline matrix non-monotonic | §54 | HARD | Halt; atlas build issue |

### Stage 7 (Tier breakpoints) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Substrate missing for cfDNA branch | §61 | SOFT | Skip cfDNA step |
| Global FLOOR_BREACH (≥4 classes) | §62 | HARD | Hold report; manual review at Stage 9 |
| Multi-class breach (2-3 classes) | §62 | SOFT | Flag for review |

### Stage 8 (Card matching) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Disease matrix SHA mismatch | §65 | HARD | Halt |
| Card rule syntax error | §67 | HARD | Caught at deployment, not runtime |
| Phase verdict ambiguity | §67 | SOFT | Report highest-confidence phase with ambiguous flag |
| Residual map CpG coverage <80% | §66 | SOFT | Mark residual channel INSUFFICIENT; fall back to per-class matching |
| Patient metadata missing for covariate | §68 | SOFT | Flag UNADJUSTED; report explicit "covariate-uninformative" disclaimer |
| Covariate out of declared range | §68 | SOFT | Clamp to range; flag CLIPPED_COVARIATE |

### Stage 9 (Report assembly) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Mapping table version mismatch | §70 | HARD | Halt |
| Engine tier outside mapping range | §70 | HARD | Report fails legal-boundary at §76; does not ship |
| Literature anchor 404 / missing | §71 | HARD AT DEPLOY | Caught at deployment |
| Literature anchor effect-summary conflict | §71 | SOFT | Use conservative wrapper |
| No anchor for fired card | §71 | SOFT | Card reports with "limited published context" disclaimer |
| Cancer prior table SHA mismatch | §72 | HARD | Halt |
| Patient sex/age missing for prior | §72 | SOFT | Fall back to overall-population prior |
| Family history multiplier missing for disease | §73 | SOFT | Default 1.0; flag |
| Card sex-condition ambiguous | §74 | HARD AT DEPLOY | Caught at deployment |
| Renderer slot unfilled | §75 | HARD | Report INCOMPLETE; does not advance |
| Legal-boundary check finds CANNOT_SAY violation | §76 | HARD | HALT_FOR_REVIEW; report does not ship |

### Stage 10 (Delivery) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Rendering pipeline failure | §77 | SOFT | Retain cleared Markdown; manual repackaging |
| Channel authentication failure | §78 | SOFT | Bounded retry; manual handoff after exhaustion |
| Channel throughput limit | §78 | SOFT | Queue with rate-limit backoff |
| Audit capture failure | §79 | HARD | Hold report; cannot ship without audit |

### L9 (Audit machinery) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Framework dispatch failure (null crashes) | §80 | HARD | Return FAIL with framework_error flag |
| Declared null not runnable (missing metadata) | §80-§88 | SOFT | INSUFFICIENT; framework aggregates to RESTATE |
| Insufficient permutation budget | §81 | SOFT | Report empirical percentile with explicit budget note |
| Cohort too small to split for N4 | §84 | SOFT | INSUFFICIENT_SAMPLE; framework reports limitation |
| N6 recovery sign opposite injection | §86 | HARD | Halt; full audit required |
| N7 recovery bias outside tolerance | §87 | HARD | RESTATE or RETRACT |
| PREREG modified after data observation | §90 | HARD | Auto-RETRACT |

**Severity codes:**
- **HARD**: chain halts at this step; sample/cohort/VAL cannot advance without intervention.
- **SOFT**: chain advances with explicit flag propagated downstream; reduces confidence but does not block.
- **DATA SIGNAL**: not a failure; the "anomalous" state is itself a measurement (e.g., saturation in Stage 6).
- **DOCUMENTED GAP**: the step is unimplemented in current production; passed through with a status flag pending Phase B/C/D/E delivery.

---

## §93. Cross-stage decision tree

When multiple flags fire on a single patient, the canonical response is the **most-conservative** of the declared per-flag responses. Tabular decision tree:

| Combination | Combined response |
|---|---|
| Any HARD anywhere → | Halt; do not advance |
| ≥1 SOFT in Stage 0-3 + ≥1 SOFT in Stage 5-7 → | Proceed; bundle all flags into report `data_quality` block |
| ≥2 SOFT flags in same stage → | Flag stage as `multi_flag`; escalate confidence penalty |
| Sex mismatch + low call rate + cross-method FLAG → | This pattern indicates sample swap + DNA quality + atlas mismatch all together. Treat as HARD even though individually one is SOFT. Quarantine. |
| L9 RESTATE + Stage 8 card fires → | Card reports with restated VAL caveat in literature anchor block (§71) |
| L9 RETRACT + Stage 8 card fires on retracted VAL → | Card does NOT fire (retracted VAL is not referenced anywhere in production) |
| Cellular age SAT_HIGH on ≥6 of 8 classes → | DATA SIGNAL, not failure. Report all 8 ages with their saturation flags. The pattern itself is the measurement. |

The decision tree is **not** "use the worst flag." It is "compose the per-flag responses without losing any information." Every flag survives in the audit trail (§79) even when it does not block delivery.

**Inputs.** All flags from all stages.
**Atlas reference.** None.
**Files invoked.** `<cross-stage flag composition logic — currently inline in `GAPE_WEB_v13.py`>`.
**The math.** Boolean composition declared in this section.
**CMB equivalent.** Planck's per-release quality assessment composes flags from per-bolometer, per-scan-ring, per-frequency, per-component-separation-method, and per-likelihood-channel diagnostics. The composition rule is non-trivial: a single "warning" in one diagnostic combined with a "warning" in another may exceed the significance of either alone. Same here.
**How the methylome differs.** Per-patient combination rather than per-release.
**How it's the same.** Multi-source flag composition without information loss.
**Outputs.** A per-patient `combined_response ∈ {PROCEED, FLAGGED, QUARANTINE, REJECT}` field plus the audit trail of all contributing flags.
**Decision points.** The combined response itself is the decision.
**Failure modes.** Cross-stage logic bugs (a combination not anticipated by the decision tree). Detected by manual sampling of edge cases; updates to the decision tree itself.
**Canonical cross-references.** §11-§91 of this SOP.
**CPG Plate references.** None.
**Chain-link assignment.** Above all (cross-cutting).

---

## §94. Cross-method disagreement protocol

Specifically for the Walther vs NILC cross-method gate at §33 — the most common SOFT-to-HARD escalation point in production. Detailed protocol:

**Step 94.1 — Classify the disagreement.**

| Walther verdict | NILC verdict | Classification |
|---|---|---|
| OK | OK | Cross-method PASS |
| OK | FAIL (low conf, status NOT OK) | SINGLE_METHOD; proceed with reduced confidence |
| FAIL (low conf) | OK | Same: SINGLE_METHOD |
| Both OK with substrate L1 disagreement >0.25 median | Both OK | Substrate-disagreement FLAG; proceed with annotation |
| Inference direction disagreement on disease-relevant class | (either) | INFERENCE FLAG; pause patient |
| Both FAIL | Both FAIL | Quarantine; sample uninterpretable in 8-class basis |

**Step 94.2 — Apply the response.**

- **Cross-method PASS** → §34 Stage 2 output proceeds normally.
- **SINGLE_METHOD** → §34 proceeds; manifest flag `cross_method=single`; downstream report carries the flag.
- **Substrate-disagreement FLAG** → §34 proceeds; cross-method-uncertainty annotation on report; the Walther fractions are still the production answer but the report explicitly notes "substrate-level cross-method disagreement at <X median L1>; biological inference layer was concordant."
- **INFERENCE FLAG** → STOP at §33. Manual review required before Stage 3 advancement. The review either resolves (sample/cohort issue identified and corrected) or escalates to substrate-mismatch quarantine.
- **Quarantine** → patient exits the chain at §33. Sample's substrate is not interpretable in the current IAMAtlas's 8-class basis. Note for future atlas expansion (Roadmap §10.4 multi-substrate work).

**Step 94.3 — Document.**

Every cross-method disagreement, regardless of severity, gets a row in `Biological_Physics/chain_of_custody/L4_component_separation/nilc_walther_crosscheck_v2.json`. The log is the running audit trail of inter-method behavior across all production patients. It is the empirical foundation for any future Phase B5 work on disagreement-resolution (e.g., adding SMICA-style and SEVEM-style methylome variants if the cross-method discipline ever needs more than two voices).

**CMB equivalent.** Planck's **Commander vs NILC vs SMICA vs SEVEM comparison logs** drive Phase-Z calibration adjustments. When the four methods disagree at the per-pixel level by more than expected, Planck doesn't pick a "right" method — Planck publishes all four and notes the disagreement as a propagated systematic. CPG follows the same discipline at the patient-report level.

**How the methylome differs.** Two methods (currently) vs four. The disagreement-handling protocol is the same.

**How it's the same.** Cross-method discipline is the single strongest claim a measurement can make. Documentation of disagreement is the discipline; resolution-attempts come later if at all.

**Outputs.** The cross-method-log artifact + per-patient disagreement classification.

**Decision points.** Above five-row table.

**Failure modes.**
- **Method registry mismatch** (e.g., NILC has been upgraded to v3 but Walther still v2.x). Caught at engine deployment; both methods must be at registered versions.
- **Cross-method log corruption.** SHA-hashed at every append.

**Canonical cross-references.** §32 (NILC), §33 (gate), Phase_B2_1_FINDING.md.

**CPG Plate references.** **Plate 4 Panel B** (chr16/chr17 cold patches visible to BOTH deconvolvers) is the visualization of the cross-method-consensus discipline: the systematic is real because both methods see it.

**Chain-link assignment.** L4 + L9.

---

## §95. Out-of-calibration handling (saturation, low coverage, missing metadata)

**Out-of-calibration means: the patient sits in a region of measurement space where the IAMAtlas reference does not provide reliable interpretation.** Three canonical patterns:

**95.1 Saturation (Stage 6 cellular age).** Patient's per-class A-score is above the highest decade's baseline A_mean or below the lowest. Status SAT_HIGH or SAT_LOW. **This is not a failure — it is a measurement.** Report all 8 per-class cellular ages with their saturation flags. Customer report at Stage 9 communicates: "Your terminal-class cellular reading is outside the range of the IAMAtlas calibration cohort. This is documented; the calibration cohort had limited representation in your reading's region. Consider discussing with your physician."

**95.2 Low coverage (Stages 2, 4, 5).** Patient's per-class or per-cell-type marker pool has fewer than 60% (cell type) or fewer than 20 markers (class) in the QC-passed pool. Status INSUFFICIENT_MARKERS or NO_MARKER_OVERLAP. **Selective failure.** The specific class/cell-type's measurement is not returned; other classes/cell-types proceed. Report at Stage 9 notes the missing measurements with explicit "could not measure [X] reliably for this sample" language.

**95.3 Missing metadata (Stages 0, 1, 3, 9).** Required metadata (age, sex, plate position, ancestry, smoking status) is missing or unparseable. Per-step canonical response: declare the dependency (which steps required this metadata), set the affected steps to PASS_THROUGH or fallback (e.g., overall-population prior in §72), flag at every downstream consumer. Report at Stage 9 communicates the limitation honestly.

**Common across all three:** the framework **never** silently imputes. Every out-of-calibration handling is recorded in the audit trail; every customer report explicitly states the limitation.

**Inputs.** Flag accumulator from §11-§79.
**Atlas reference.** Indirect (out-of-calibration is defined relative to the atlas's calibrated range).
**Files invoked.** Each affected stage's failure-mode handler.
**The math.** Per-stage declared in §11-§91.
**CMB equivalent.** Planck's **galactic mask** (out-of-calibration sky region — the foregrounds are too strong to extract cosmological signal). Planck does not silently impute the masked region — Planck reports cosmological parameters as derived from the unmasked sky with the masked region's size explicitly documented. CPG's saturation, low-coverage, missing-metadata handling is the methylome implementation of the same discipline.
**How the methylome differs.** Per-patient out-of-calibration (different patients have different out-of-range regions). Galactic mask is fixed per-sky.
**How it's the same.** The mask is declared, not hidden.
**Outputs.** Per-patient out-of-calibration flag list propagated through to §79 audit.
**Decision points.** Above three patterns each have their own decision logic, declared in this section.
**Failure modes.** Mis-classification of out-of-calibration as failure (resulting in unnecessary quarantine) or as success (resulting in spurious confident reading). The audit trail allows retrospective detection of either error.
**Canonical cross-references.** §47 (Mahalanobis imputation), §54 (cellular age saturation), §63 (engine-to-customer collapse).
**CPG Plate references.** **Plate 4 Panel E** (MCMC Coverage Map) — the stromal galactic mask visualization is the per-CpG out-of-calibration map. The dark region IS the framework's declared limitation, made empirically visible.
**Chain-link assignment.** Cross-cutting (every stage that touches the atlas).

---

## §96. Chain re-run protocol

Sometimes the chain must be re-run on the same sample or cohort — typically because an upstream version updated (new IAMAtlas, new marker pool, new H_min after recalibration, new card deployed) and the existing results need to be regenerated under the updated configuration.

**Re-run conditions:**

| Trigger | Re-run scope |
|---|---|
| IAMAtlas version update | Full chain Stage 2-10 on all affected cohorts |
| Marker pool update (e.g., v0_1 → v0_3) | Stage 2-10 |
| H_min recalibration | Stage 4-10 |
| Disease signature matrix update (v1.4 → v1.5) | Stage 8-10 |
| Card deployment (new card or version) | Stage 8-10 for that card |
| Literature anchor table update | Stage 9-10 |
| Cannot-say list update | Stage 9-10 |
| L9 framework update | Re-run L9 audit on affected sealed VALs |
| Synthetic patient generator update | Re-run N6, N7 nulls on affected VALs |

**Re-run protocol:**

1. **Declare the trigger.** What changed; which artifacts have new SHA-256 hashes; which downstream stages depend on the changed artifact.
2. **Identify the scope.** Which cohorts/patients/VALs are affected. The audit trail (§79) makes this query trivial — any sample whose audit log references the now-superseded artifact SHA is in scope.
3. **Lock the new artifacts.** SHA-256 hash the new artifact; pin in the appropriate canonical (IAMAtlasREBUILD_provenance.json, etc.).
4. **Re-run from the affected stage.** Use the existing per-sample data from upstream stages (no need to re-do Stage 0 intake or Stage 1 calibration if those didn't change).
5. **Re-compute the audit trail.** Each affected patient gets a new audit record with the new SHA references. The old audit record is preserved at `audit_logs/superseded/`.
6. **For sealed VALs:** if the VAL's nulls depended on the changed artifact, the framework re-runs the affected nulls and re-issues a sealing verdict. **A VAL that was SEALED under the old artifact must be re-sealed under the new artifact** — sealing is not transitive across artifact versions.
7. **Customer notification.** When a customer report is regenerated under updated configuration, the customer receives an updated report with explicit "Updated under [new artifact version], [date]" annotation. Old reports are archived but not deleted.

**Inputs.** Trigger + affected scope + new artifact.
**Atlas reference.** New atlas version, if the trigger was atlas-related.
**Files invoked.** `<chain re-run orchestration — TBD when `web.commercial.py` is built>` orchestrates.
**The math.** Same as the original chain.
**CMB equivalent.** Planck **Release re-issue cycles**. When Planck recalibrated a frequency channel between Release 2 and Release 3, every dependent analysis was re-run; the published cosmological parameters changed slightly; the discipline was that the Release 3 cosmological-parameter values were tagged "Release 3 / 2018 / Planck-2018-Calibration-v3" so that any future paper citing them could trace the provenance.
**How the methylome differs.** Per-patient re-runs vs per-release re-runs. Same provenance discipline.
**How it's the same.** Updates do not silently overwrite; every version is traceable.
**Outputs.** Updated per-patient reports + updated audit trails + (when applicable) re-issued VAL sealing verdicts.
**Decision points.** Trigger-condition determines scope; scope determines which stages re-run.
**Failure modes.**
- **Partial re-run** (some affected patients not re-processed). Detected by the audit-trail-vs-current-artifact-SHA query; framework refuses to deliver a report under a stale audit until re-run completes.
- **Re-sealing conflict** (VAL that was SEALED under old artifact RESTATEs under new artifact). Documented honestly in the new OUTCOME; old SEALED status is preserved with the supersession note.

**Canonical cross-references.** Recipe §13 (versioning discipline). Roadmap §10.5 (Phase G/H/I/J post-engine card work, which is built on re-run capability).

**CPG Plate references.** None directly. (Plates are typically re-generated when the underlying VAL data changes; re-run is the operational mechanism behind plate updates.)

**Chain-link assignment.** Cross-cutting (any chain link can trigger; any chain link can be affected).

---

# Part V — Reference

The final part. Tables, glossaries, indices. Designed for at-a-glance lookup during operations rather than narrative reading.

---

## §97. File-to-stage mapping table

Authoritative inventory: `Biological_Physics/atlas_vault/SYSTEM_INVENTORY.md`. This SOP section mirrors that inventory and adds the SOP-section cross-reference column.

§97 is organized in three blocks: **(A) Real artifacts that exist as files in the repo today; (B) Logic currently embedded inside `GAPE_WEB_v13.py` — the production engine — and not yet isolated as standalone modules; (C) Pending — does not exist yet, awaits the `web.commercial.py` orchestrator design discussion.**

If an operator is looking for a path that this SOP names anywhere from §11 to §96, this is the table that says whether it is a real file (A), embedded logic (B), or pending (C).

### §97.A — Real files / modules per `SYSTEM_INVENTORY.md`

| Artifact | Path | Used at SOP § | Role |
|---|---|---|---|
| **IAMAtlas REBUILD** | `IAMAtlasREBUILD.csv` (proprietary; not in repo) | §28 + all atlas-consulting steps | The calibrated instrument |
| Atlas provenance | `IAMAtlasREBUILD_provenance.json` | §28, §43, §99 | H_min source of truth (frozen 2026-04-06) |
| **Walther deconvolver** | `Biological_Physics/atlas_vault/deconvolver/walther_iam_deconvolver.py` | §30, §31 | Production NNLS deconvolution |
| Walther README | `Biological_Physics/atlas_vault/deconvolver/README.md` | §30, §31 | Deconvolver-specific docs |
| **NILC v2 deconvolver** | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_deconvolver.py` | §32 | Cross-method GLS deconvolution |
| NILC v1 fractions | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_fractions_all.csv` | (audit) | Historical v1 output |
| NILC v2 fractions | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_fractions_v2_departure.csv` | (audit) | Current v2 output |
| NILC v1 crosscheck | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_walther_crosscheck.json` | (audit) | Historical v1 gate report |
| NILC v2 crosscheck | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_walther_crosscheck_v2.json` | §33, §94 | Current cross-method gate report |
| Phase B2 finding | `Biological_Physics/chain_of_custody/L4_component_separation/Phase_B2_FINDING.md` | (historical) | Initial NILC+Walther cross-check |
| Phase B2.1 finding | `Biological_Physics/chain_of_custody/L4_component_separation/Phase_B2_1_FINDING.md` | §33 | Current cross-method gate documentation |
| **Age-axis foreground** | `Biological_Physics/atlas_vault/components/age_axis_foreground.py` | §35 | Per-CpG age regression / β subtraction (Phase B3) |
| Age layer matrix | `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv` | §35 | Per-CpG (α, γ, R², n) — 8,199 CpGs, 100% convergence |
| Age layer diagnostics | `Biological_Physics/atlas_vault/components/age_layer_diagnostics.json` | §35 | Per-CpG fit diagnostics |
| **A-score scoring** | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_a_scoring.py` | §41–§45 | `score_per_class()` + `score_per_celltype()` |
| Marker artifact | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` | §29, §44, §65 | Per-cell-type one-vs-rest top-100, 115 cell types |
| Marker SHA anchor | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.sha256` | §29 | SHA: `46ea5be1db377f2b8773a02418a7f481a191630e0fa833d3294eab1fd19c47bd` |
| **Mahalanobis scoring** | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_mahalanobis_scoring.py` | §47–§51 | `MahalanobisHealthyHull` class — distance + top-10 axis decomp |
| Mahalanobis reference | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/mahalanobis_healthy_reference_v0_1.json` | §48 | HC centroid + covariance (n_hc=601, shrinkage=0.0088) |
| Mahalanobis validation | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/mahalanobis_per_patient_breast_predx_validation.csv` | (audit) | Per-patient breast pre-dx distances (n=648) |
| **Cellular age scoring** | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iam_cellular_age_scoring.py` | §52–§58 | `IAMCellularAge` class — canonical Recipe §6.3 inversion |
| Age reference matrix (JSON) | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.json` | §53 | 80-cell baseline: 8 classes × 10 decades |
| Age reference matrix (CSV) | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.csv` | §53 | Same data, flat CSV |
| Age reference matrix (PY) | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.py` | §53 | Same data, with interpolation helpers |
| Cellular age validation | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/cellular_ages_v4_epic_italy_validation.csv` | (audit) | 1,174 EPIC-Italy v4 output |
| Tier breakpoints | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/tier_breakpoints.json` | §59, §60, §63 | A-score thresholds + customer-label collapse |
| cfDNA weights | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/cfdna_weight.json` | §61 | Healthy-blood per-class expected (Snyder/Moss) |
| Literature anchors | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/literature_anchors.json` | §71 | Per-class published anchors |
| Cancer prior | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/cancer_prior.json` | §72 | US lifetime incidence per class |
| Family history mult. | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/family_history_multiplier.json` | §73 | First-degree-relative RR per class |
| Disease signature matrix v1.5 | embedded inside `GAPE_WEB_v13.py` at present writing | §65 | 77×131 card-level lookup; standalone binary export pending at `pipeline_runtime_matrices/disease_signature_matrix/` |
| Card residual maps | placeholder dir `pipeline_runtime_matrices/card_residual_maps/` | §66 | Currently empty; populated per card as VALs lock thresholds |
| **L9 null runner** | `Biological_Physics/chain_of_custody/L9_null_suite/cpg_null_runner.py` | §80–§88, §91 | Unified 8-null framework (N1–N8) |
| **Synthetic patients** | `Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py` | §86, §87, §89 | FFP10/NPIPE analog — signal-injection harness |
| Per-VAL null outputs | `Biological_Physics/chain_of_custody/L9_null_suite/test_runs/CPG_VAL_00X/` | (audit) | Phase A: 5 sealed, 2 RESTATE (VAL-004 gain/loss, VAL-006 chr6 LEC) |
| **Production engine** | `GAPE_WEB_v13.py` | §11–§79 (where logic is embedded) | The actual runtime — see §97.B |
| System inventory | `Biological_Physics/atlas_vault/SYSTEM_INVENTORY.md` | (this table) | Source of truth for paths |

### §97.B — Logic currently embedded inside `GAPE_WEB_v13.py`

The following operations described in §11–§79 are performed by the production engine `GAPE_WEB_v13.py` directly, not as separate module files. v1 of this SOP fabricated module paths for each (e.g. `cpg_engine/qc/sex_check.py`, `cpg_engine/calibration/dye_bias.py`); those were wrong. The real situation: one engine, one file, no per-stage module isolation yet.

| SOP § | Operation | Notes |
|---|---|---|
| §11–§19 | Stage 0 — IDAT intake, control probes, detection-p, bead count, call rate, sex check, decision gate | Stage 0 QC runs inside Walther per `SYSTEM_INVENTORY.md` §1; the wider intake/manifest layer is engine-internal. |
| §20–§27 | Stage 1 — dye bias, probe-type normalization, ComBat, BS conversion, β computation, β sanity | Wraps `minfi`/`methylprep` upstream; β values come into the deconvolver pre-computed. |
| §28, §29 | Atlas loading + marker-pool extraction | Engine loads `IAMAtlasREBUILD.csv` and `iamatlas_celltype_markers_v0_2.json` at startup. |
| §34 | Stage 2 output consolidation | Engine bookkeeping; no separate packager file. |
| §40 | Stage 3 output consolidation | Same — engine bookkeeping. |
| §46 | Stage 4 output consolidation | Same. The scoring module (`iamatlas_a_scoring.py`) is real and standalone; the consolidation is engine-internal. |
| §65, §67–§68 | Stage 8 card matching — disease-signature lookup, multi-class rule eval, within-card covariate adjustment | All inside the engine; card registry and disease-signature matrix v1.5 are embedded constants. |
| §66 | Residual-overlap (breast-epic) | Logic inside engine; the residual-map CSV will live at `pipeline_runtime_matrices/card_residual_maps/breast-epic/` once VAL-003 locks. |
| §70–§76 | Stage 9 report assembly — language collapse, literature lookup, prior lookup, family history, sex adjustment, renderer, legal-boundary gate | All inside the engine. Lookup JSONs (anchors, prior, multiplier) are real files; the orchestration is engine-internal. |
| §77–§79 | Stage 10 delivery — packaging, routing, audit capture | All inside the engine today; expected to move into the future `web.commercial.py` orchestrator. |
| §93, §96 | Cross-stage flag composition; chain re-run orchestration | Inline within `GAPE_WEB_v13.py` today; cleanly modularized when orchestrator is built. |

### §97.C — Pending: does not exist yet

| Item | What it would be | Blocked by |
|---|---|---|
| `web.commercial.py` orchestrator | Top-level driver that calls the modules in §97.A in sequence, replacing the engine-internal orchestration in §97.B. Working name — final naming TBD. | Open design discussion between Heath and Walther |
| Standalone disease-signature-matrix v1.5 CSV | Binary export of the matrix currently embedded inside `GAPE_WEB_v13.py`. Placeholder dir already created at `pipeline_runtime_matrices/disease_signature_matrix/`. | Pending decision on whether to externalize before orchestrator design |
| Per-card residual maps for cards other than breast-epic | Per-card CSVs at `pipeline_runtime_matrices/card_residual_maps/<card>/`. | Per-card VALs locking thresholds |
| Sex / batch / ancestry / smoking foreground modules | Companions to `age_axis_foreground.py`; would live in `Biological_Physics/atlas_vault/components/`. | Phase B4 per Roadmap §10.2.2 |
| Probe response function (L3) | Per-probe transfer function as a separate module; documented gap in L3 grading. | Atlas-wide probe characterization work |
| Manifest schema (standalone JSON Schema) | The intake schema as a versioned file rather than engine-embedded validation. | Orchestrator design |
| CANNOT_SAY list (standalone JSON) | Legal-boundary regex/keyword list as a versioned file rather than engine-embedded checks. | Orchestrator design + legal review |

### §97.D — How to use this table

If the SOP names a path you cannot find in the repo: search §97.A first. If not there, it is in §97.B (logic inside the engine, no separate file) or §97.C (does not exist yet). Anything NOT appearing in any of §97.A / B / C should be treated as a SOP error and flagged for v1.2 cleanup.

The "Used at SOP §" column tells the operator which SOP sections describe each artifact's role.

---

## §98. Glossary — CMB ↔ methylome term map

The two-way translation between cosmology vocabulary and methylome vocabulary. Use this when reading a CMB paper to understand its CPG equivalent, or when writing CPG documentation to find the right CMB analog.

| CMB term | Methylome term | Definition (CPG context) |
|---|---|---|
| **Bolometer / detector** | Methylation probe | Single measurement element on the array |
| **Bolometer timestream / TOD** | IDAT file pair | Raw detector intensities pre-calibration |
| **Frequency channel** | Probe type (Type I / Type II) | Detector class with distinct response characteristics |
| **Multi-frequency observation** | Multi-substrate (HM450K / EPIC) | Same biology measured by different detector chemistries |
| **Calibration source** | Bisulfite conversion control / Specificity control | On-chip references with known expected response |
| **L1 data (raw timestream)** | IDAT raw intensities | Pre-calibration |
| **L2 data (calibrated)** | β values | Post-calibration physical measurement per coordinate |
| **L3 data (sky map)** | Per-CpG β matrix | Calibrated map ready for component separation |
| **Component separation** | Deconvolution | Decompose multi-component signal into per-component fractions |
| **Commander algorithm** | Walther IAM Deconvolver | Primary Bayesian/optimization-based component separation |
| **NILC algorithm** | NILC v2 deconvolver | Cross-method GLS in departure-from-consensus space |
| **SMICA / SEVEM** | (future methylome variants) | Additional cross-method voices |
| **Foreground** | Cell composition / age / sex / batch / ancestry / smoking | Known contaminating signals that overlap with the target |
| **Galactic dust** | Age-axis foreground | Strongest single foreground; per-CpG / per-frequency template |
| **Galactic mask** | Stromal MCMC coverage gap | Region where the calibrated reference is insufficient |
| **Beam profile / response function** | Probe response function (PRF) | Per-detector signal-vs-truth transfer function |
| **Beam deconvolution** | PRF correction (provisional) | Inverting the per-detector transfer function |
| **Power spectrum C_ℓ** | Per-class entropy H | Information-theoretic decomposition by component basis |
| **Dimensionless cosmological parameter** | A-score | Calibrated ratio against physical reference (H_min) |
| **Critical density ρ_crit** | Architectural floor H_min | Reference scale for dimensionless ratio |
| **Mahalanobis distance / parameter tension** | Mahalanobis hyper-volume distance | Multi-D departure under appropriate covariance |
| **Joint posterior banana** | 2D A-score posterior shape | Curved 2D contour encoding parameter degeneracy (TODO 2.3) |
| **Look-elsewhere effect (LEE)** | N8 correction | Multiple-testing correction for searched feature space |
| **FFP10 / NPIPE simulations** | Synthetic patient generator | End-to-end pipeline test on known-truth synthetic data |
| **Signal injection / recovery test** | N6 / N7 | Inject known signal, verify chain recovers it |
| **Jackknife null** | N1 (label permutation) | Shuffle case/HC labels, recompute, verify signal disappears under null |
| **Stratified jackknife** | N2 (age strata) / N3 (sex strata) | Permutation within potential-confound strata |
| **Half-mission-difference null** | N4 (cohort split replication) | Verify signal replicates across cohort sub-samples |
| **Scan-direction null** | N5 (plate position null) | Test for systematics tied to measurement-apparatus property |
| **Acoustic peak (in C_ℓ)** | TODO 2.1 — methylation correlation length peaks | Future analysis (Phase C) |
| **Bispectrum** | TODO 2.2 — methylation bispectrum | Three-point correlation (Phase C) |
| **CMB lensing** | TODO 4.3 — field-effect reconstruction | Lensing-style methylome distortion (Phase D+) |
| **Mollweide projection** | Same | Equal-area sky projection used in all four plates |
| **HEALPix pixelization** | Same | Equal-area hierarchical pixelization (NSIDE=128 for methylome, 512 for CMB plates) |
| **Sealed result** | Sealed CPG-VAL | Result that passed declared null suite |
| **Restated result** | Restated CPG-VAL | Result with one null failure; preserved with limitation |
| **Retracted result** | Retracted CPG-VAL | Result withdrawn after multi-null failure |
| **Chain of custody** | Chain of custody | Link-by-link audit trail; identical concept |
| **Confirm (not validate)** | Confirm (not validate) | Framework predicted, data consistent (not pass/fail testing) |

---

## §99. The eight Mahaffey Numbers — H_min values frozen 2026-04-06

**These are the only constants in the framework.** Every A-score in the system divides by one of these eight numbers. They are derived from the IAMAtlas REBUILD MCMC posteriors and frozen on 2026-04-06.

| Class | H_min value | Convergence |
|---|---|---|
| terminal | 0.7728 | R̂ < 1.001 |
| immune | 0.838889 | R̂ < 1.001 |
| secretory | 0.843264 | R̂ < 1.001 |
| cycling | 0.856055 | R̂ < 1.001 |
| progenitor | 0.852216 | R̂ < 1.001 |
| stromal | 0.86295 | R̂ < 1.001 (with 4.93% MCMC coverage — the methylome's galactic mask) |
| stem_adult | 0.873718 | R̂ < 1.001 |
| stem_pluri | 0.982166 | R̂ < 1.001 |

**Single source of truth:** `IAMAtlasREBUILD_provenance.json`, key `h_min_values_frozen_2026_04_06`.

**Plain-language interpretation (from The Cellular Margin):**
- terminal class (H_min=0.7728) is most-committed (lowest entropy floor); a fully differentiated neuron carries the tightest specification.
- stem_pluri (H_min=0.9822) is least-committed (highest entropy floor); a pluripotent stem cell deliberately carries a looser pattern.
- Healthy human cells run at A ≈ 21× the underlying ATP/RT thermal floor; H_min normalizes the entropy reading per class.

**Discipline rules:**
- These values do not change between runs unless the atlas itself is rebuilt.
- A rebuild triggers the §96 re-run protocol on every sealed VAL.
- Any code that hardcodes these values WITHOUT reading from the canonical JSON is a chain-of-custody violation; the engine refuses to deploy.

---

## §100. Canonical cross-reference index

Maps every SOP section to its primary upstream sources. **Use this to find where a step is also discussed in the Recipe, Roadmap, or VAL inventory.**

| SOP § | Step | Recipe § | Roadmap § | Other |
|---|---|---|---|---|
| §11-§19 | Stage 0 intake | §3.1 | §10.1.1 L1 | — |
| §20-§27 | Stage 1 calibration | §3.2-§3.3 | §10.1.1 L2-L3 | — |
| §28 | Atlas load | §4 | — | Part I §4 |
| §29 | Marker pool | §4, §6.1 | — | — |
| §30-§31 | Walther | §6.2 | §10.1.1 L4 | Capability Translator §1, §2 |
| §32-§33 | NILC + cross-method gate | §6.2 | §10.2.2 Phase B2 | Phase_B2_1_FINDING.md |
| §35 | Age foreground | §5 | §10.2.2 Phase B3 | Phase_B3_FINDING.md |
| §41-§43 | Per-class A-score | §6.3 | §4 | Part I §7 |
| §44 | Per-cell-type A-score | §6.3 | §3.15 | Capability Translator §6 |
| §45 | Disease panel A-score | §6.5 | §7 | Capability Translator §7 |
| §47-§51 | Mahalanobis | §6.4 | §3.13 | VAL-002 |
| §52-§58 | Cellular age | §6.3 | TODO 2.4 / E1 | The Cellular Margin |
| §59-§64 | Tier breakpoints | §7 | — | Capability Translator §3, §4 |
| §65-§69 | Card matching | §8 | §10.2.6 Phase F | disease_signature_matrix_README.md |
| §70-§76 | Report assembly | §9 | — | Part I §10 |
| §77-§79 | Delivery | §10 | — | — |
| §80-§91 | L9 audit | §11 | §10.2.1 Phase A | §12 of Roadmap |

**Where the Recipe is the operational source-of-truth** (specific algorithms, exact formulas, runtime artifacts), this SOP cross-references it. Where the **Roadmap is the strategic source-of-truth** (Phase ordering, grading, planned upgrades), this SOP cross-references it. Where neither suffices, the SOP **declares its own**.

The Recipe is the proprietary vault (NOT in repo). The Roadmap is the canonical scoping doc (also Heath-only). The SOP is **operator-facing**; it teaches HOW the chain runs, not WHAT the proprietary derivation is.

---

## §101. CPG Plates 1-4 cross-reference

The four plates are the visual anchor of the framework. Each plate illustrates specific steps of the chain.

**Plate 1 — The Cosmic Microwave Methylome (`CPG_Plate_01_Cosmic_Microwave_Methylome.png`).**
- Sections illustrated: §6 (CMB ↔ methylome principle), §28 (Atlas load), §31 (per-class confidence — the stromal panel's MCMC coverage gap), §42 (Shannon entropy on per-class β).
- Visual content: 8-panel Mollweide projection of IAMAtlas REBUILD posterior β across architectural classes, in genomic order. Each panel shows one class's per-CpG posterior mean β across 481,966 CpGs.
- Key feature: stromal panel's "galactic mask" (4.93% MCMC coverage gap) is the methylome's declared known-unknown.

**Plate 2 — The Breast Pre-Diagnostic Anisotropy (`CPG_Plate_02_Breast_Anisotropy.png`).**
- Sections illustrated: §45 (disease panel A-score), §66 (per-card residual map application), §88 (N8 look-elsewhere on the chr6 zoom panel).
- Visual content: Full-sky scatter of 1,392 concordant breast-pre-dx CpGs (signed Cohen's d, sized by |d|, colored cyan-blue hypomethylated vs orange-yellow hypermethylated, 5.4:1 hypomethylation dominance). Lower panel: chr6 zoom showing MHC region enrichment (HLA classes I+II+III) carrying disproportionate concordant signal.
- Key feature: visualizes both the cross-genome anisotropy field-effect signature AND the chr6 MHC concentration that VAL-006 originally claimed (and that N8 honestly restated).

**Plate 3 — The Cosmic Microwave Methylome (Grandaddy plate — `CPG_Plate_03_Grandaddy_CMM_vs_CMB.png`).**
- Sections illustrated: §6 (CMB ↔ methylome principle), §25 (β distribution shape — the methylome's bimodality vs CMB's Gaussianity), §27 (the β matrix that becomes the methylome map), §87 (N7 end-to-end simulation — the right panel is the CMB equivalent of CPG's synthetic patient generation).
- Visual content: Side-by-side methylome vs CMB at matched projection (Mollweide), matched colormap, matched pixelization conventions. Top panels: full-sky overview. Bottom panels: zoom on small-scale anisotropy texture.
- Key feature: makes Heath's "math doesn't care if it's a CMB or a methylome" visually irrefutable while also making the biological difference (bimodal vs Gaussian) visible at the texture level.

**Plate 4 — Patterns Discovered (`CPG_Plate_04_Patterns_Discovered.png`).**
- Sections illustrated:
  - Panel A (Class-Difference Map): §29 (marker pool selection), §43 (per-class A-score).
  - Panel B (chr16+chr17 Cold-Patch Zones): §29 (marker concentration), §66 (residual map systematic in chr16/17), §94 (cross-method visible).
  - Panel C (Concordant Signal Density): §45 (panel A-score density visualization).
  - Panel D (Differentiation Gradient): §35 (age-axis foreground subtraction — what the age component looks like after isolation).
  - Panel E (MCMC Coverage Map): §31 (per-class confidence, the stromal mask), §95 (out-of-calibration handling).
  - Panel F (Breast Pre-Diagnostic Anisotropy): §66 (residual map at the cohort level), §88 (LEE).
- Key feature: each panel demonstrates a different chain-of-custody concept made visually concrete.

---

## §102. Change log

| Date | Version | Change | Authority |
|---|---|---|---|
| 2026-05-31 | v1 | Initial release. Parts I-V complete: §1-§102. Foundations + step-by-step + audit machinery + failure modes + reference. | Heath W. Mahaffey |

**Change discipline.** Every future modification to this SOP creates a new version entry. The previous version is archived (not overwritten). Changes that modify CANNOT-SAY language, H_min values, atlas references, or VAL sealing protocols require Heath's explicit authority. Changes that update file paths, add new failure modes catalog entries, or expand the glossary can be made by Walther under standing instruction. The change log is the chain-of-custody for the SOP itself.

---

**End of Part V (§97-§102).**

**End of CPG Chain-of-Custody SOP v1.** 102 sections. Five parts. Quantum-level granular operator manual from IDAT-on-server (§11) to audit-trail-closed (§79), with L9 chain-integrity scaffolding (§80-§91), failure-mode response catalog (§92-§96), and complete reference apparatus (§97-§102).

*The math doesn't care if it's a CMB or a methylome. — Heath W. Mahaffey, 2026.*

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*
