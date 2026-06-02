# CPG Chain-of-Custody SOP — Part II-A (Stages 0 through 4) (v1.2 — walkthrough aligned)

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


*Continuation of CPG_Chain_of_Custody_SOP_v1_PART_I.md. Read Part I first.*

---

# Part II — The step-by-step chain of custody

---

## Stage 0 — Sample intake (L1)

---

### §11. Step 0.1 — IDAT file arrival on server

**What this step does.** An Illumina methylation array sample arrives at the EDEAR server as a pair of IDAT files — one for the Cy3 (green) channel, one for the Cy5 (red) channel. The pair is keyed by Sentrix Barcode + Sentrix Position (e.g., `200123456789_R01C01_Grn.idat` + `200123456789_R01C01_Red.idat`). This step receives the upload, verifies the pair is complete, and stamps the arrival into the intake log.

**Inputs.**
- Two `.idat` files per sample (Grn + Red)
- A sample manifest entry (CSV or JSON) declaring: Sentrix ID, array type (HM450K or EPIC v1 or EPIC v2), patient identifier, intake date, declared substrate (whole blood / cfDNA / tissue), declared sex, declared chronological age

**Atlas reference.** Not consulted at this step. The atlas enters at Stage 2.

**Files invoked.**
- Intake handler: `<intake handler inside `GAPE_WEB_v13.py`>`
- Manifest schema: `<manifest schema — TBD per orchestrator design>`
- Intake log: `<intake log — TBD>`

**The math.** None at this step. Pure I/O + metadata.

**CMB equivalent.** This is the **raw bolometer timestream arrival** stage of a CMB experiment. Planck's pipeline receives 50,000+ bolometer scan files per day; each is logged, paired with its scan ring metadata, and routed to the calibration pipeline. The methylome's IDAT files are equivalent — raw fluorescence intensities awaiting calibration.

**How the methylome differs in implementation.** IDAT files are static (one pair per sample) whereas CMB bolometer streams are continuous (time-ordered samples across a scan ring). The methylome's "scan" was already completed by the array hybridization; we receive the final intensities, not a time series.

**How it's the same in principle.** Both are raw detector intensities awaiting calibration. Both need integrity verification before downstream processing. Both produce the same kind of failure mode if the upload is incomplete: the entire downstream chain refuses to operate until the input is whole.

**Outputs.**
- A canonical filename pair in `<intake staging directory — TBD per orchestrator design>`
- An intake log row: `(intake_timestamp, sentrix_id, array_type, substrate, declared_sex, declared_age, status=STAGED)`

**Decision points.**
- IF both Grn and Red files present AND manifest entry complete → proceed to §12 (Step 0.2 Sample manifest creation)
- IF Grn or Red file missing → quarantine; notify operator; do not advance
- IF manifest fields missing → quarantine with `INCOMPLETE_MANIFEST` flag

**Failure modes.**
- *Truncated IDAT upload.* Detected by file-size sanity check (IDAT files for HM450K are ~8 MB each; EPIC ~14 MB each; anything below 1 MB is suspicious).
- *Misnamed Sentrix ID.* Detected when manifest references a barcode not present in the file pair.
- *Wrong array type declared.* The IDAT header encodes the array type; mismatch with manifest triggers a flag.
- *Duplicate intake.* Same Sentrix ID arriving twice within 24 hours triggers a soft-warn (might be re-run; might be operator error).

**Canonical cross-references.**
- Recipe §1.1 (Sample Intake)
- Roadmap §10.1.1 L1 row
- Capability Translator §3 (Status codes)

**CPG Plate references.** Not applicable at this step.

**Chain-link assignment.** L1.

---

### §12. Step 0.2 — Sample manifest creation

**What this step does.** Cross-references the IDAT file pair with the intake manifest to construct the canonical per-sample record. Every downstream step reads from this record. Once written, the record is immutable for that sample run — re-intakes get new run IDs.

**Inputs.**
- Staged IDAT file pair from §11
- Sample manifest entry from intake CSV/JSON

**Atlas reference.** Not consulted.

**Files invoked.**
- Manifest builder: `<inside `GAPE_WEB_v13.py`>`
- Schema validator: `<manifest schema — TBD per orchestrator design>`
- Per-sample record output: `<intake staging directory — TBD per orchestrator design>per_sample.json`

**The math.** None. Structural validation.

**CMB equivalent.** This is the **scan-ring metadata association** step. Planck's pipeline attaches each bolometer file to its corresponding pointing, gain calibration, and noise characterization records. Without this association, no downstream analysis can proceed — the data is meaningless until it's keyed to its physical context.

**How the methylome differs in implementation.** The methylome's "scan ring metadata" is patient demographics (age, sex, substrate type) rather than telescope pointing data. The keying is simpler — one IDAT pair to one patient record.

**How it's the same in principle.** Both attach raw detector output to the physical context required for downstream calibration. Without the manifest association, every downstream calibration step would be operating blind.

**Outputs.** <per-sample run record — exact structure TBD per orchestrator design> with structure:
```json
{
  "sample_run_id": "<UUID>",
  "sentrix_id": "200123456789_R01C01",
  "array_type": "EPIC_v1",
  "substrate": "whole_blood",
  "patient_id": "<hashed>",
  "declared_sex": "F",
  "declared_chronological_age": 54.3,
  "intake_timestamp": "2026-05-31T14:22:00Z",
  "status": "MANIFEST_COMPLETE"
}
```

**Decision points.**
- IF manifest validates against schema → proceed to §13 (Step 0.3 IDAT integrity hash check)
- IF schema validation fails → quarantine with `MANIFEST_INVALID` flag

**Failure modes.**
- *Schema drift.* If the intake CSV uses an old column name, the validator rejects. Operator must update the intake CSV to the current schema.
- *Encoded vs cleartext PII.* Patient identifiers must be hashed before reaching the engine. Cleartext PII triggers an immediate quarantine and notification.

**Canonical cross-references.**
- Recipe §1.2 (Manifest validation)
- Roadmap §10.1.1 L1 row

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §13. Step 0.3 — IDAT integrity hash check

**What this step does.** Computes SHA-256 hashes of both IDAT files in the pair. Compares against any prior intake of the same Sentrix ID (re-run detection) and registers the new hashes in the integrity log. This is the formal L1 integrity protocol the Roadmap §10.1.1 grades B+ today.

**Inputs.** The staged IDAT file pair from §11.

**Atlas reference.** Not consulted.

**Files invoked.**
- Hash module: `<inside `GAPE_WEB_v13.py`>`
- Integrity log: `<integrity log — TBD>`

**The math.** `sha256(grn_bytes)`, `sha256(red_bytes)`. Compare to prior runs.

**CMB equivalent.** This is **bolometer scan integrity verification.** Planck logs the checksum of every scan file at receipt to detect transmission corruption and re-runs. Identical checksums of the same scan ring indicate re-transmission; differing checksums of the same Sentrix ID indicate either a re-run with a fresh array (legitimate) or data corruption (problem).

**How the methylome differs in implementation.** Methylation IDAT files are smaller than CMB scan rings and don't compress further at runtime, so a fixed SHA-256 over the byte stream is sufficient. CMB pipelines often add stream-level checksums for chunked transmission; not needed here.

**How it's the same in principle.** Both verify the bytes that arrived match the bytes that were intended. Both treat hash mismatches as red flags that block the chain.

**Outputs.**
- Integrity log row: `(sentrix_id, intake_timestamp, grn_sha256, red_sha256, prior_hash_match)`
- A `STATUS=INTEGRITY_OK` stamp on the per-sample record

**Decision points.**
- IF both hashes new (no prior record) → proceed to §14 (Step 0.4 Control probe validation)
- IF hashes identical to prior intake → flag `RE_TRANSMISSION_DETECTED`; ask operator whether to overwrite or quarantine
- IF Sentrix ID matches prior intake but hashes differ → flag `LIKELY_RE_RUN_FRESH_ARRAY`; proceed with new run ID

**Failure modes.**
- *Hash collision with prior contaminated sample.* Extremely unlikely with SHA-256 but documented in the integrity log for traceability.
- *Bit-flip during storage.* If the staged file changes hash between intake and Stage 0.4 invocation, the integrity check at Stage 0.4 detects it.

**Canonical cross-references.** Roadmap §10.1.1 L1 row ("Declare formally: IDAT integrity check protocol... Document.")

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §14. Step 0.4 — Control probe validation

**What this step does.** Illumina arrays include ~600 control probes whose intended responses are known (positive controls, negative controls, hybridization controls, bisulfite conversion controls, extension controls, target removal controls, etc.). This step reads the control probe intensities from the IDAT pair and verifies each control class is within expected bounds. Out-of-bounds controls indicate a chemistry problem with the array — the patient sample is technically unreliable regardless of the biological signal.

**Inputs.** Decoded IDAT files (parsed via `illumina-idat-reader` Python package or equivalent).

**Atlas reference.** Not consulted — the control probe spec is Illumina's, not IAMAtlas's.

**Files invoked.**
- Control probe extractor: `<inside `GAPE_WEB_v13.py`>`
- Illumina control probe manifest: `<Illumina control probe manifest from `methylprep`>`
- QC log: `<control probe QC log — TBD>`

**The math.** Per-control-class summary statistics:
- Median intensity vs expected range (Illumina docs)
- Bisulfite conversion efficiency = median(BS-Conv-I) / (median(BS-Conv-I) + median(BS-Conv-II))
- Hybridization signal = median(Hyb-High) − median(Hyb-Low)
- Extension efficiency = ratio of methylated-extension to unmethylated-extension responses

Each control class has a manufacturer-specified pass range. Failure of any class flags the sample.

**CMB equivalent.** This is **detector calibration verification before science use.** Planck's pipeline validates each bolometer's gain, noise temperature, and linearity against on-board calibration sources before that bolometer's data is used in the cosmological analysis. A bolometer with anomalous calibration is masked out. Equivalent here: a sample with anomalous control probes is quarantined before its β values enter the chain.

**How the methylome differs in implementation.** Control probes are spatially co-located with science probes on the same array (vs CMB bolometers which calibrate against time-domain reference signals). One per-array calibration per sample, not a time-series calibration.

**How it's the same in principle.** Both verify the measurement apparatus is functioning correctly before using its readings. Both treat apparatus failure as a hard block on downstream analysis — bad calibration cannot be repaired by clever processing.

**Outputs.**
- A control probe QC report: `<control probe QC artifact — TBD>`
- A `CTRL_QC=PASS` or `CTRL_QC=FAIL_<class>` flag on the per-sample record

**Decision points.**
- IF all control classes pass → proceed to §15 (Step 0.5 Detection p-value QC)
- IF bisulfite conversion < 95% → quarantine with `BS_CONVERSION_LOW` (downstream β values are unreliable)
- IF hybridization controls fail → quarantine with `HYB_FAIL` (array chemistry was compromised)
- IF extension controls fail → quarantine with `EXT_FAIL`

**Failure modes.**
- *Operator pipetting error during array prep.* Manifests as low hybridization controls.
- *Insufficient bisulfite conversion.* Manifests as elevated β bias toward unmethylated state across the array. The control probe check catches this before β values lie about the biology.
- *Sample dilution / RNA contamination.* Detected by low total fluorescence across the array.

**Canonical cross-references.** Roadmap §10.1.1 L1 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §15. Step 0.5 — Detection p-value QC per probe

**What this step does.** For each of the ~485,000 (HM450K) or ~865,000 (EPIC) probes on the array, computes a detection p-value: the probability that the observed intensity is distinguishable from the array's background (negative-control distribution). Probes with detection p > 0.01 are flagged as "undetected" — their β value is unreliable.

**Inputs.** Decoded IDAT files + Illumina negative-control probe set.

**Atlas reference.** Not consulted — this is array-internal QC.

**Files invoked.**
- Detection p-value calculator: `<inside `GAPE_WEB_v13.py`>`
- Per-probe detection log (one row per probe, one column per sample, sparse): `<detection-p log — TBD>`

**The math.** For each probe with intensity I_probe and background distribution from negative controls (mean μ_bg, sd σ_bg):

> **detection_p = 1 − Φ((I_probe − μ_bg) / σ_bg)**

where Φ is the standard normal CDF. A probe with detection_p > 0.01 is flagged.

The Illumina default uses a slightly different formulation (the `minfi` R package's `detectionP` function), and CPG mirrors that exact computation for cross-platform compatibility.

**CMB equivalent.** This is **pixel-level signal-to-noise gating.** Planck masks pixels where the integration time is insufficient or the noise estimate exceeds a per-pixel threshold. Same operation: a measurement that hasn't risen above the noise floor cannot contribute to the final map.

**How the methylome differs in implementation.** The methylome's "noise" is the negative-control distribution embedded within the array. CMB noise comes from the bolometer's thermal characterization. Different physical noise sources; same statistical treatment.

**How it's the same in principle.** A measurement that doesn't cross the SNR floor produces a NULL, not a value. The downstream chain treats NULLs explicitly — never imputes silently.

**Outputs.**
- A per-probe detection p-value array (length 485,512 for HM450K, 865,919 for EPIC)
- A summary statistic: `pct_probes_detected_p_le_01` — the fraction of probes that passed
- A `DETECTION_QC=PASS` or `=FAIL_LOW_DETECTION` flag

**Decision points.**
- IF >99% of probes have detection p ≤ 0.01 → proceed to §16
- IF 95–99% detected → flag `DETECTION_BORDERLINE`, allow with warning
- IF <95% detected → quarantine with `DETECTION_QC_FAIL` (sample is broadly under-hybridized)

**Failure modes.**
- *Whole-array under-hybridization.* Manifests as bulk failure (e.g., 80% probes detected).
- *Localized array damage.* Specific physical region of the chip showing detection failure. Detected via spatial clustering of failed probes (not currently implemented; flagged in Phase B4 as `B4.2`).
- *Probe-class systematic failure.* Type I vs Type II probes with differential failure rates. Handled by Step 1.2 (probe-type normalization).

**Canonical cross-references.** Recipe §1.5 (Detection QC). Roadmap §10.1.1 L1 row.

**CPG Plate references.** Plate 1 stromal panel — the 4.9% MCMC coverage gap visible as the "galactic mask" is partly a consequence of detection failures in stromal-marker CpGs across the cohort that built the atlas.

**Chain-link assignment.** L1.

---

### §16. Step 0.6 — Bead count QC

**What this step does.** Each methylation probe is represented on the array by multiple beads (Illumina bead chemistry replicates each probe ~15-20 times per sample for redundancy). The bead count per probe is reported in the IDAT file. Probes with very low bead count have unreliable β values regardless of detection p-value.

**Inputs.** Decoded IDAT files.

**Atlas reference.** Not consulted.

**Files invoked.**
- Bead count reader: `<inside `GAPE_WEB_v13.py`>`

**The math.** Per probe: `bead_count_probe` extracted from IDAT. Threshold: `bead_count_probe >= 3` (Illumina default; can be tightened to ≥5 for high-stakes applications).

**CMB equivalent.** This is **integration-time validation per pixel.** Planck's pipeline checks that each sky pixel was visited enough times during the survey to give a reliable temperature average. Pixels with fewer than N hits are masked.

**How the methylome differs in implementation.** Bead replication is built into the array's physical design (Illumina manufactures redundant beads per probe) rather than driven by scan strategy.

**How it's the same in principle.** Both verify that the measurement at each spatial coordinate had enough independent samples to be reliable.

**Outputs.**
- Per-probe bead count vector
- Summary statistic: `pct_probes_bead_count_ge_3`
- A `BEAD_QC=PASS/FAIL` flag

**Decision points.**
- IF >99.5% of probes have bead_count ≥ 3 → proceed to §17
- IF <99.5% → flag with warning; consider tightening downstream confidence thresholds

**Failure modes.**
- *Manufacturing defects in array.* Rare but possible; detected by clusters of low-bead-count probes.
- *Damage during sample prep.* Detected by global bead count reduction.

**Canonical cross-references.** Recipe §1.5.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §17. Step 0.7 — Sample-level call rate

**What this step does.** Aggregates §15 (detection p-value) and §16 (bead count) into a single sample-level call rate: the fraction of probes that passed BOTH detection and bead count thresholds. This is the headline quality metric for the array as a whole.

**Inputs.** Outputs of §15 and §16.

**Atlas reference.** Not consulted.

**Files invoked.**
- Call rate calculator: `<inside `GAPE_WEB_v13.py`>`

**The math.** 
> **call_rate = N_probes_passing_both / N_probes_total**

Threshold: call_rate ≥ 0.98 to proceed.

**CMB equivalent.** This is **overall scan quality for a single observation epoch.** CMB pipelines compute the fraction of bolometers + the fraction of scans that pass all upstream QC; samples below a threshold are flagged for review or excluded.

**How the methylome differs in implementation.** The "sample" is a single array; the call rate is per-array. CMB call rates are typically per-scan-ring or per-bolometer-night.

**How it's the same in principle.** Single rolled-up metric that summarizes upstream QC. Operator-friendly threshold.

**Outputs.**
- Call rate value (float in [0,1])
- A `CALL_RATE=<value>` field on the per-sample record

**Decision points.**
- IF call_rate ≥ 0.98 → proceed to §18
- IF 0.95 ≤ call_rate < 0.98 → flag `CALL_RATE_BORDERLINE`, proceed with downstream confidence penalty
- IF call_rate < 0.95 → quarantine with `CALL_RATE_FAIL`

**Failure modes.** Same as §15 and §16 (rolled up).

**Canonical cross-references.** Recipe §1.5.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---


**Step 0.7b — HM450 ≥80% coverage gate + platform tagging.** When the patient was assayed on a 450K platform, verify that ≥80% of the HM450 reference CpGs the downstream chain depends on (deconvolver markers, A-score markers, Mahalanobis features, breast-epic / future card panels) returned valid β values. Halt with HARD failure below 80%. Per VAL-091 ad-LL-006, every downstream output is tagged with the platform identifier (`450K` or `EPIC`) so platform-stratified thresholds can be applied at Stage 7 / Stage 8 where 450K coverage gaps materially affect call rates. The platform tag flows into the audit trail at Stage 10.

### §18. Step 0.8 — Sex check vs metadata

**What this step does.** Computes the predicted sex of the sample from chrY-specific and chrX-specific probe intensities, then compares it to the declared sex in the manifest. A mismatch indicates either a sample-swap (catastrophic — patient identity is wrong) or a true biological discordance (rare — XXY, XX male, etc.).

**Inputs.** Decoded IDAT files + manifest's declared sex.

**Atlas reference.** Not consulted directly. (IAMAtlas does NOT carry sex-stratified posteriors at v0.1; sex foreground is handled separately at Stage 3.)

**Files invoked.**
- Sex predictor: `<inside `GAPE_WEB_v13.py`>`
- chrY/chrX probe coordinates: Illumina manifest files

**The math.** Standard `minfi::getSex()` algorithm:
- Mean log2-intensity of chrY probes: `log2_Y`
- Mean log2-intensity of chrX probes: `log2_X`
- Decision boundary in 2D (log2_X, log2_Y) space → predicted_sex ∈ {M, F}

**CMB equivalent.** This is **pointing-direction sanity check.** Planck verifies that each scan ring's pointing direction (RA, Dec) matches its declared target by cross-referencing with the spacecraft attitude solution. A mismatch indicates a metadata error that must be resolved before the data is used.

**How the methylome differs in implementation.** The biological "pointing direction" is sex; the verification is a 2D intensity check.

**How it's the same in principle.** Both verify the metadata describing the measurement matches what the measurement itself says. Mismatches are NEVER silently corrected — they're flagged for operator resolution.

**Outputs.**
- Predicted sex
- A `SEX_CHECK=PASS` (predicted == declared) or `SEX_CHECK=MISMATCH` flag

**Decision points.**
- IF predicted == declared → proceed to §19
- IF mismatch → quarantine with `SEX_MISMATCH`; require operator investigation before proceeding

**Failure modes.**
- *Sample swap at the lab.* Most common cause of sex mismatch. Investigation involves checking the lab's chain of custody for the IDAT pair.
- *Genuine biological sex chromosome variant.* Rare. Resolved by operator after consulting patient record.
- *Tumor contamination with loss-of-chrY.* Manifests as M-declared-but-F-predicted in male patients with chromosomal instability. Investigated case-by-case.

**Canonical cross-references.** Recipe §1.5.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §19. Step 0.9 — Stage 0 decision gate

**What this step does.** Consolidates all Stage 0 outputs into a single proceed/quarantine/reject decision. This is the formal end of L1 — every QC check has been run, and the sample either advances to Stage 1 (β computation) or it does not.

**Inputs.** Outputs of §11-§18.

**Atlas reference.** Not consulted.

**Files invoked.**
- Decision gate: `<inside `GAPE_WEB_v13.py`>`
- Stage 0 outcome log: `<Stage 0 verdict log — TBD>`

**The math.** Boolean conjunction:
```
PROCEED = (
    INTEGRITY_OK AND
    CTRL_QC == PASS AND
    DETECTION_QC in (PASS, BORDERLINE) AND
    BEAD_QC == PASS AND
    CALL_RATE >= 0.98 AND
    SEX_CHECK == PASS
)
```

Any single FAIL routes to quarantine. Any BORDERLINE proceeds with downstream confidence penalty applied.

**CMB equivalent.** This is the **pre-science decision gate.** A CMB scan ring either passes all upstream QC and enters the cosmological analysis, or it gets routed to the masked-data archive for forensic review.

**How the methylome differs in implementation.** One gate per array (one per patient sample). CMB gates can be per-bolometer-per-scan-ring (much finer-grained).

**How it's the same in principle.** Hard boolean decision based on conjunction of all upstream checks. No silent imputation. No "we'll fix it downstream."

**Outputs.**
- Final Stage 0 status: `STAGE_0=PROCEED` / `=QUARANTINE` / `=REJECT`
- A complete Stage 0 record in `<Stage 0 verdict log — TBD>`
- For PROCEED samples: per-sample record + cleaned IDAT pair handoff to Stage 1

**Decision points.**
- IF PROCEED → handoff to §20 (Step 1.1 Dye-bias correction)
- IF QUARANTINE → notify operator; do not advance
- IF REJECT → archive; document reason; do not retry without operator override

**Failure modes.**
- Any upstream Stage 0 failure cascades here.

**Canonical cross-references.** Recipe §1.5 (Decision gate). Roadmap §10.1.1 L1 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1 (closes Stage 0).

---

## Stage 1 — Calibration & β computation (L2 + L3)

---

### §20. Step 1.1 — Dye-bias correction

**What this step does.** Illumina's methylation arrays use two dye colors (Cy3 green, Cy5 red) to distinguish methylated and unmethylated DNA. The two dyes have slightly different intensity characteristics that vary across arrays, batches, and even within a single chip. Dye-bias correction normalizes these differences so that downstream β values reflect actual methylation, not dye chemistry artifacts.

**Inputs.** The cleaned IDAT file pair from §19.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Dye-bias correction module: `<inside `GAPE_WEB_v13.py` (uses `minfi`/`methylprep` upstream)>`
- Implementation: typically wraps `minfi::preprocessFunnorm()` or `minfi::preprocessNoob()` from the standard methylation analysis stack

**The math.** The standard `noob` (normal-exponential out-of-band) correction:
1. Fit a normal-exponential model to the negative-control probe intensities per channel
2. Subtract the modeled background from each probe's intensity
3. Equalize the red and green channel intensity distributions to a common reference

For Type II probes (where both methylated and unmethylated states are measured in the same channel), additional correction equalizes the two states' baseline intensities.

**CMB equivalent.** This is **bandpass correction.** Each CMB frequency channel has its own bandpass profile that affects how the underlying signal is measured. Bandpass mismatch correction is a standard step in Planck's calibration pipeline. Same operation here: two channels (red, green) with different response profiles, corrected to a common reference.

**How the methylome differs in implementation.** Two channels (vs nine in Planck). Static, not time-varying. The correction parameters are learned per-sample from in-array control probes rather than per-mission from external calibration sources.

**How it's the same in principle.** Both correct for known instrumental response differences between channels before combining their signals into a single physical quantity.

**Outputs.**
- Dye-bias-corrected intensities per probe per channel
- A `DYE_BIAS_PARAMS` record documenting the correction parameters applied

**Decision points.**
- IF correction converges (negative-exponential fit succeeds) → proceed to §21
- IF correction fails (typically because of pathological control probe distributions) → quarantine with `DYE_BIAS_FAIL`

**Failure modes.**
- *Saturated controls.* Negative-control probes that themselves saturated the detector cannot anchor the background model. Detected during fit.
- *Bimodal control distribution.* Indicates contamination or chip damage. Cannot be cleanly modeled.

**Canonical cross-references.** Recipe §2.1 (Calibration). Roadmap §10.1.1 L2 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §21. Step 1.2 — Probe-type normalization

**What this step does.** Illumina's HM450K and EPIC arrays have two probe chemistries: Type I (two beads per CpG, one for methylated, one for unmethylated) and Type II (one bead per CpG, distinguishing methylation by dye color). The two chemistries produce subtly different β distributions for the same biological methylation level. Probe-type normalization brings them to a common distribution.

**Inputs.** Dye-bias-corrected intensities from §20.

**Atlas reference.** Not consulted.

**Files invoked.**
- Normalization module: `<inside `GAPE_WEB_v13.py` (uses `minfi`/`methylprep` upstream)>`
- Implementation options (configurable):
  - **funnorm** (functional normalization — Fortin 2014; standard for cross-cohort comparisons)
  - **noob** (Triche 2013; lighter touch, preserves more biological variance)
  - **SWAN** (Subset-quantile Within-Array Normalization; legacy)
  - **BMIQ** (Beta MIxture Quantile dilation; corrects Type II β distribution to match Type I)

**The math.** Each method differs:
- **funnorm**: regresses out the first few principal components of control probe intensities (which capture technical variation) from each probe's β value
- **noob**: applies dye-bias correction as a standalone normalization
- **SWAN**: rank-normalizes Type II probes to match Type I within each sample
- **BMIQ**: fits Beta-mixture distributions to Type I and Type II separately, transforms Type II to match Type I distribution shape

**Configuration default for CPG production: funnorm** (chosen because the engine operates across heterogeneous cohorts and funnorm has the strongest cross-cohort consistency in the literature). Other methods are available for cards that require a specific normalization.

**CMB equivalent.** This is **inter-channel calibration to a reference frequency.** Planck's nine frequency channels are calibrated against a common reference (typically the CMB dipole or a known astronomical source) so that the underlying physical signal is comparable across channels. Same operation here: two probe types calibrated to a common β distribution.

**How the methylome differs in implementation.** Methods are statistical (β-distribution matching) rather than physical-source calibration. The reference is a within-array population of probes, not an external calibration source.

**How it's the same in principle.** Both adjust the measurement so that the same underlying physical quantity produces the same number regardless of which detector chemistry measured it.

**Outputs.**
- Normalized β-equivalent intensities per probe (still pre-β-computation in some methods, post-β in others)
- A `NORMALIZATION_METHOD` record on the per-sample data structure

**Decision points.**
- IF normalization converges → proceed to §22
- IF normalization fails (e.g., funnorm's PCA cannot find stable components because the array is broadly damaged) → fall back to noob; if noob also fails, quarantine

**Failure modes.**
- *Insufficient control probe variation for funnorm.* Detected during PCA fit.
- *Bimodal sample population causing BMIQ misfit.* Manifest as broken Beta-mixture convergence.

**Canonical cross-references.** Recipe §2.1. Roadmap §10.1.1 L2 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §22. Step 1.3 — Batch correction (ComBat)

**What this step does.** When a card processes patients from multiple batches (different intake dates, different lab runs, different plate-positions on the same chip), inter-batch variation can dwarf biological signal. ComBat is the standard empirical-Bayes batch correction that removes additive and multiplicative batch effects while preserving the biological signal of interest. This step is conditional — it only runs when multi-batch data is being processed together.

**Inputs.** Normalized β-equivalent intensities from §21 across multiple samples in the current run.

**Atlas reference.** Not consulted.

**Files invoked.**
- Batch correction module: `<inside `GAPE_WEB_v13.py` (wraps `sva::ComBat`/`combat-py`)>`
- Implementation: typically wraps `sva::ComBat()` from the standard methylation analysis stack

**The math.** ComBat fits an empirical-Bayes hierarchical model:
> **β_corrected[i, sample] = (β_raw[i, sample] − γ_batch − α_i) / δ_batch + α_i**

where γ_batch is the additive batch shift, δ_batch is the multiplicative batch scaling, and α_i is the per-probe biological intercept. The hyperparameters are estimated empirically across all samples in the batch.

**CMB equivalent.** This is **mission-to-mission systematic correction.** When Planck combines data across observation seasons, gain drifts and calibration drifts between seasons must be removed before the data can be co-added. The empirical-Bayes structure is similar — characterize the drift parameters from the data itself, then correct.

**How the methylome differs in implementation.** Multiple "batches" can co-exist on one array run (different intake dates processed together). CMB seasons are temporally sequential and self-evident.

**How it's the same in principle.** Both estimate systematic drift between observation epochs and correct for it before downstream physics analysis. Both treat per-batch drift as a nuisance parameter to be marginalized out.

**Outputs.**
- ComBat-corrected β-equivalent intensities
- A `COMBAT_PARAMS` record documenting the per-batch γ and δ corrections applied
- A diagnostic plot of pre/post-correction batch separation (saved to QC artifact directory)

**Decision points.**
- IF single-batch run → SKIP this step; proceed directly to §23
- IF multi-batch and ComBat converges → proceed to §23
- IF multi-batch and ComBat fails (typically: too few samples per batch) → flag `BATCH_CORRECTION_INSUFFICIENT_N` and proceed without correction, with downstream confidence penalty

**Failure modes.**
- *Confounding between batch and biology.* If all cases are in batch A and all controls are in batch B, ComBat cannot distinguish batch effect from biological effect. Detected at intake and rejected with `CONFOUNDED_DESIGN`.
- *Insufficient samples per batch.* ComBat needs ≥3 samples per batch for stable estimation. Below this, falls back to no correction with warning.

**Canonical cross-references.** Recipe §2.1. Roadmap §10.1.1 L2 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §23. Step 1.4 — Bisulfite conversion efficiency check

**What this step does.** Methylation array chemistry requires bisulfite-treating the DNA before hybridization. Bisulfite converts unmethylated cytosines to uracil while leaving methylated cytosines intact. If the conversion is incomplete, β values are biased toward false hypermethylation. This step computes the conversion efficiency from internal control probes and either confirms it's adequate or flags the sample.

**Inputs.** Bisulfite conversion control probe intensities (from the §14 control probe extraction).

**Atlas reference.** Not consulted.

**Files invoked.**
- BS efficiency calculator: `<inside `GAPE_WEB_v13.py`>`

**The math.** Bisulfite conversion controls come in two classes (BS-Conv-I and BS-Conv-II). The efficiency is:
> **bs_efficiency = median_intensity(BS-Conv-I_converted) / (median_intensity(BS-Conv-I_converted) + median_intensity(BS-Conv-I_unconverted))**

A fully-converted sample reaches efficiency ≈ 1.0. A failed conversion drops below 0.95 — at which point β values are no longer reliable indicators of methylation state.

**CMB equivalent.** This is **detector gain stability verification.** Planck checks that each bolometer's gain has not drifted outside its calibration window. If gain has drifted, downstream temperature measurements are biased. Same idea: if chemistry has drifted (incomplete BS conversion), downstream methylation measurements are biased.

**How the methylome differs in implementation.** Chemistry drift (BS conversion failure) is irreversible — the sample is contaminated, not just miscalibrated. CMB gain drift can sometimes be corrected post-hoc; BS conversion failure cannot.

**How it's the same in principle.** Both treat the measurement apparatus's working state as a precondition for trusting its readings. Below threshold, the data does not enter the chain.

**Outputs.**
- `bs_efficiency` numeric value
- A `BS_CONVERSION` flag (PASS/BORDERLINE/FAIL)

**Decision points.**
- IF bs_efficiency ≥ 0.98 → proceed to §24
- IF 0.95 ≤ bs_efficiency < 0.98 → flag BORDERLINE, proceed with downstream confidence penalty
- IF bs_efficiency < 0.95 → quarantine with `BS_CONVERSION_FAIL` (sample chemistry failed, β values are unreliable)

**Failure modes.**
- *Incomplete sodium bisulfite reaction during sample prep.* Most common cause. Sample-prep batch should be re-investigated.
- *DNA degradation pre-conversion.* Detected by combination of low BS efficiency + low overall intensity.

**Canonical cross-references.** Recipe §2.1. Roadmap §10.1.1 L2 row ("Add bisulfite-conversion-efficiency nuisance parameter").

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §24. Step 1.5 — β-value computation

**What this step does.** This is the formal map-making step — converting calibrated fluorescence intensities into the methylation level β at each CpG. β = M / (M + U + 100), where M is the methylated-state intensity, U is the unmethylated-state intensity, and 100 is the standard stabilization offset (prevents division by zero at very low intensities; cosmetic at high intensities). β is the canonical per-CpG measurement that every downstream step consumes.

**Inputs.** Calibrated intensities from §20-§22 (dye-bias corrected, probe-type normalized, optionally batch-corrected).

**Atlas reference.** Not consulted — β computation is intrinsic to the array, not the atlas.

**Files invoked.**
- β calculator: `<inside `GAPE_WEB_v13.py`>`

**The math.**
> **β = M / (M + U + 100)**

Per CpG per sample. Output is a number in [0, 1] (with theoretical extremes never reached because of the +100 stabilization).

For Type II probes, M and U are extracted from the green and red channels of the same bead. For Type I probes, M and U come from separate beads (one each).

**CMB equivalent.** This is **map-making.** The conversion from calibrated bolometer intensities (in detector units) to a physical sky temperature map (in μK) is performed by CMB pipelines using the Madam/MADmap solver. The CMB equivalent equation is more complex (involves the pointing matrix, the inverse-noise covariance, and a destriping operator), but the principle is identical: raw detector quantities → calibrated physical measurement per pixel.

The β formula is the methylome's exact analog of:
> **T_pixel = (Σ_i w_i × I_i) / Σ_i w_i**

— a weighted combination of detector readings producing a physical measurement per spatial coordinate.

**How the methylome differs in implementation.** The methylome's "pixel" is a CpG; its formula is simpler (two intensities + offset, no pointing matrix). CMB map-making involves time-ordered data; methylation β is computed once per sample.

**How it's the same in principle.** Both produce a per-coordinate physical quantity from underlying detector intensities. Both are the input that all downstream physics analyses operate on. The map-making step is the boundary between "raw instrument data" and "physical measurement."

**Outputs.** 
- A per-sample β matrix: N_probes × 1 (typically 485,512 for HM450K or 865,919 for EPIC, restricted to autosomes for most cards)
- A `BETA_COMPUTED=TRUE` stamp on the per-sample record

**Decision points.**
- IF all β values are in [0, 1] → proceed to §25
- IF any β values fall outside [0, 1] → arithmetic error upstream; quarantine for forensic review

**Failure modes.**
- *Numerical artifacts from extreme intensities.* Detected as β values exactly equal to 0 or 1 across many probes simultaneously.
- *Missing M or U value for a probe.* Per-probe NULL handling (the probe is dropped, not imputed).

**Canonical cross-references.** Recipe §2.2 (β computation). Roadmap §10.1.1 L3 row.

**CPG Plate references.** Plates 1, 3 (the β values themselves are what get rendered as the methylome sky maps).

**Chain-link assignment.** L3 (map-making).

---

### §25. Step 1.6 — β-value sanity checks

**What this step does.** Before β values feed into the deconvolver, run final sanity checks on the per-sample β matrix: distribution shape, modality, range coverage. This catches catastrophic failures of upstream calibration that didn't trip per-probe flags but produced an entire β matrix that doesn't look biologically plausible.

**Inputs.** β matrix from §24.

**Atlas reference.** Not consulted at this step — comparing β distribution to a healthy reference happens at Stage 4 (A-score), not here.

**Files invoked.**
- β QC module: `<inside `GAPE_WEB_v13.py`>`

**The math.** Three checks:
1. **Range coverage:** the β distribution should span [0.01, 0.99]. If 99% of values cluster in a narrow band (e.g., all between 0.4 and 0.6), the sample is degenerate.
2. **Bimodality:** methylation is fundamentally bimodal — most CpGs are either methylated (β ≈ 0.85) or unmethylated (β ≈ 0.15), with relatively few in between. Hartigan's dip test on the β distribution should reject unimodality (dip statistic D > 0.01).
3. **Cohort consistency:** the sample's median β should be within ±0.05 of the cohort's median β (when multi-sample run).

**CMB equivalent.** This is **final sanity check on the pixel-temperature distribution before science analysis.** Planck checks that the CMB map's temperature distribution has the expected variance (~80 μK) and is approximately Gaussian. Aberrant distributions get flagged before parameter inference. Same idea here: a β distribution that doesn't look biologically plausible is flagged before A-score computation.

**How the methylome differs in implementation.** The methylome's distribution is expected to be **bimodal** rather than Gaussian. Detecting bimodality is the test; detecting Gaussianity would be the test for CMB.

**How it's the same in principle.** Both check that the calibrated map distribution matches the expected statistical character before downstream inference. Failures here indicate upstream calibration didn't work even though all individual flags passed.

**Outputs.**
- A β-sanity record: `(range_coverage_pct, hartigan_dip_stat, dip_p, median_consistency_z)`
- A `BETA_SANITY=PASS/WARN/FAIL` flag

**Decision points.**
- IF all three checks pass → proceed to §26
- IF one check warns → flag with warning, proceed with reduced downstream confidence
- IF two or more checks fail → quarantine with `BETA_SANITY_FAIL`

**Failure modes.**
- *Bulk hypermethylation artifact.* Manifests as β distribution clustered near 1.0. Usually indicates failed BS conversion (should have been caught at §23, but this is the redundancy check).
- *Loss of bimodality.* Manifests as β distribution centered at 0.5. Indicates broad upstream signal degradation.
- *Sample-cohort outlier.* Single sample's median β far from cohort median. May indicate sample swap or genuine biological outlier — operator decides.

**Canonical cross-references.** Recipe §2.2. Roadmap §10.1.1 L3 row.

**CPG Plate references.** Plate 1 (the architectural-class β distributions are themselves a reference for what "biologically plausible" looks like).

**Chain-link assignment.** L3.

---

### §26. Step 1.7 — Probe response function (provisional)

**What this step does.** Marks where the L3 chain link needs further work per the Roadmap §10.1.1 grading. Each probe has a slightly different response function (how raw fluorescence scales to true methylation fraction). The standard β formula assumes uniform response across probes; in reality, probe-specific calibration would refine the β estimate. This is currently NOT implemented in production — the SOP declares it as a known L3 gap to be filled.

**Inputs.** Per-probe β values from §24.

**Atlas reference.** Not consulted.

**Files invoked.** *None in current production.* Future: `<future module — not built; documented gap per Roadmap §10.1.1 L3>`.

**The math.** Conceptually: β_true = f_probe(β_observed) where f_probe is a learned per-probe calibration curve. Currently f_probe is the identity function for all probes (no correction applied).

**CMB equivalent.** This is **per-detector transfer function correction.** Planck applies per-bolometer transfer functions to account for individual detector nonlinearities. CPG's equivalent is currently the identity transfer — every probe assumed identical.

**How the methylome differs in implementation.** Currently NOT implemented at production grade. CMB pipelines have per-bolometer transfer functions; CPG has uniform per-probe response.

**How it's the same in principle.** Acknowledging the gap is itself part of the L3 audit. The current grade of L3 = B reflects this missing component.

**Outputs.** Identity-corrected β values (i.e., the §24 β values, unchanged).

**Decision points.** Proceed to §27 in all cases (no current gating).

**Failure modes.** Not applicable in current implementation. When implemented, would catch probe-specific calibration drift.

**Canonical cross-references.** Roadmap §10.1.1 L3 row ("Declare as L3 explicitly; add probe-response-function nuisance treatment.") This step's existence in the SOP is the formal declaration.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L3 (provisional).

---

### §27. Step 1.8 — Stage 1 output: per-CpG β matrix

**What this step does.** Packages the calibrated β matrix and its provenance into the canonical Stage 1 output: a per-sample β file ready for Stage 2 deconvolution.

**Inputs.** β matrix from §24 + provenance records from §20-§26.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Output packager: `<inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** This is the **L3 → L4 handoff:** the calibrated all-sky temperature map (a HEALPix array) being handed off to the foreground-separation pipeline. In CPG terms: the cleaned β matrix being handed off to deconvolution.

**How the methylome differs in implementation.** The methylome's "all-sky map" is a per-CpG vector, not a HEALPix pixelization. (Plate 3 demonstrates that the per-CpG β values CAN be rendered as a HEALPix map — that's the foundational data structure for L5 work in Phase C — but the runtime chain operates on the per-CpG vector directly.)

**How it's the same in principle.** Same handoff: calibration is done; component-separation is next.

**Outputs.** A canonical per-sample β file containing:
- The β matrix (one number per CpG, NaN where probe failed QC)
- Per-probe QC flags (passing/failing detection, bead count)
- Calibration provenance (which normalization method, which batch correction applied)
- The per-sample QC summary from §19

Location: `<β matrix output — emitted internally by `GAPE_WEB_v13.py`>` (plus `.json` provenance).

**Decision points.** Handoff to §28 (Step 2.1 IAMAtlas REBUILD load).

**Failure modes.** Output packaging is mechanical; failures here indicate file-system issues, not chain issues.

**Canonical cross-references.** Recipe §2.2 (end of calibration). Roadmap §10.1.1 L3 row.

**CPG Plate references.** Plate 3 (the rendered β maps — illustrate what the Stage 1 output LOOKS like when projected to HEALPix).

**Chain-link assignment.** L3 (closes).

---

## Stage 2 — Deconvolution (L4 component separation, primary)

---

### §28. Step 2.1 — IAMAtlas REBUILD load

**What this step does.** Loads the proprietary IAMAtlas REBUILD reference matrix into memory. This is the first moment in the chain where the calibrated instrument enters. Every subsequent step that consults IAMAtlas reads from this loaded reference (or from a runtime-matrix derivative of it).

**Inputs.** The proprietary IAMAtlasREBUILD.csv file (605 MB) — proprietary IP, not in the public repository.

**Atlas reference.** **The atlas IS being loaded at this step.** This is the "atlas enters the chain" moment.

**Files invoked.**
- Atlas loader: `<atlas-loading logic inside `GAPE_WEB_v13.py`>`
- Source data: `IAMAtlasREBUILD.csv` (proprietary)
- Provenance: `IAMAtlasREBUILD_provenance.json` (H_min values frozen 2026-04-06)

**The math.** None. Memory-resident table indexed by CpG ID, with per-class posterior columns (mean, sd, ci_lo, ci_hi).

**CMB equivalent.** This is **loading the published Planck data products** (e.g., the Planck 2018 likelihood, the foreground templates, the dust polarization map). Once loaded, these data products serve as the reference against which the science measurement is compared. Same here: IAMAtlas is the reference against which patient β values are scored.

**How the methylome differs in implementation.** IAMAtlas is proprietary; Planck data products are public. CPG ships derivative artifacts (markers, age baseline, Mahalanobis reference) to the repo; the source atlas stays in the vault.

**How it's the same in principle.** Both load a calibrated, version-pinned reference data product into memory at the start of every analysis run. Both treat the loaded reference as immutable for that run.

**Outputs.** An in-memory IAMAtlas object with:
- Per-CpG × per-class posterior moments
- H_min lookup by class (read from `IAMAtlasREBUILD_provenance.json`)
- A SHA-256 fingerprint of the loaded data (logged for run audit)

**Decision points.**
- IF atlas loads cleanly → proceed to §29
- IF atlas file SHA-256 mismatches the expected fingerprint → ABORT (someone modified the atlas; chain integrity compromised)

**Failure modes.**
- *Atlas file corruption.* Detected by SHA-256 mismatch.
- *Atlas file version drift.* If the loaded atlas's provenance doesn't match the engine's expected version, refuse to run until reconciled.

**Canonical cross-references.** Recipe §2.3 (Atlas load). Roadmap §10.1.1 L4 row (atlas as the calibrated reference for L4).

**CPG Plate references.** Plate 1 (all four plates are renderings of IAMAtlas REBUILD).

**Chain-link assignment.** L4 (atlas is the calibrated reference for component separation).

---

### §29. Step 2.2 — Per-class marker pool extraction

**What this step does.** From the loaded atlas, extract the per-class marker CpG lists — the subset of CpGs that are maximally informative for each architectural class. These are pre-computed (delivered as the runtime artifact `iamatlas_celltype_markers_v0_2.json`); this step loads them and pairs them with the patient's β values at the corresponding CpGs.

**Inputs.** The loaded IAMAtlas object from §28 + the per-sample β matrix from §27.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the marker CpG lists at `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` (115 cell types × 100 markers, derived from IAMAtlas REBUILD's per-cell-type posteriors by the one-vs-rest top-N algorithm).

**Files invoked.**
- Marker loader: `<marker-pool extraction inside `GAPE_WEB_v13.py`>`
- Marker artifact: `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json`
- SHA-256 anchor: `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.sha256`

**The math.** Selection criterion from the artifact's metadata:
> `|target_celltype_mean − mean(other_114_celltype_means)|, top N=100 by score`

Computed at atlas build time, frozen in the artifact. At runtime, this step is pure lookup.

**CMB equivalent.** This is **multipole binning / aperture-mask selection.** Planck's foreground separation operates on selected angular scales and frequency channels chosen for their informativeness about the foreground component being separated. Same operation here: from the full 481K-CpG atlas, select the ~11,500 CpGs (115 × 100) most informative for cell-type discrimination.

**How the methylome differs in implementation.** Selection is per-cell-type by one-vs-rest contrast in posterior mean; CMB selection is per-multipole-band by foreground sensitivity. Different selection criteria, same purpose: focus the analysis on the maximally informative subset.

**How it's the same in principle.** Both reduce the dimensionality of the inverse problem (decomposition into classes/components) by restricting to the most informative measurements.

**Outputs.** A patient-specific marker pool table:
- 115 cell types
- For each cell type: its top-100 marker CpGs
- For each marker CpG: the patient's β value at that CpG (NaN if probe failed QC)
- For each marker CpG: the IAMAtlas posterior mean and sd per class

**Decision points.**
- IF ≥80% of marker CpGs have valid β values for the patient → proceed to §30
- IF 50-80% → flag `MARKER_COVERAGE_BORDERLINE`, proceed with downstream confidence penalty
- IF <50% → quarantine with `INSUFFICIENT_MARKER_COVERAGE`

**Failure modes.**
- *Probe-level QC failures concentrating in marker CpGs.* Causes marker coverage to drop disproportionately. Detected here.
- *Stale marker artifact.* If the marker JSON's SHA-256 doesn't match the recorded anchor, refuse to run.

**Canonical cross-references.** Recipe §3.1 (Marker selection). Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plate 4 Panel B (chr16+chr17 cold-patch zones — the marker pool concentrates heavily on these chromosomes, which has implications for §35 foreground subtraction).

**Chain-link assignment.** L4.

---

### §30. Step 2.3 — Walther IAM Deconvolver (Path 1, NNLS)

**What this step does.** Runs the production deconvolver — Walther — on the patient's β values at the marker pool. Walther solves a constrained non-negative least-squares (NNLS) problem to estimate the per-class cellular fractions that best explain the observed β values, given the IAMAtlas reference matrix. This is the production answer that feeds Stages 3, 4, 8.

**Inputs.** Marker pool table from §29.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the per-class β posterior means at marker CpGs serve as the reference matrix X in the NNLS system. Loaded from `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` extended with the atlas posterior moments at those CpGs.

**Files invoked.**
- Walther deconvolver: `Biological_Physics/atlas_vault/deconvolver/walther_iam_deconvolver.py`
- Class-CpG reference matrix: built at runtime from the atlas posterior at marker CpGs

**The math.** For each patient, Walther solves:
> **min ‖X · f − β_observed‖₂²    subject to f ≥ 0**

where:
- **X** is the (m × 8) reference matrix: m marker CpGs × 8 architectural class posterior means
- **β_observed** is the patient's m × 1 β vector at those CpGs
- **f** is the (8 × 1) per-class fraction vector to be estimated

NNLS is solved via the Lawson-Hanson active-set algorithm (`scipy.optimize.nnls`). The non-negativity constraint reflects the physical fact that cellular fractions cannot be negative.

After NNLS returns **f_raw**, Walther normalizes: **f_normalized = f_raw / sum(f_raw)** (the fractions sum to 1 by construction, since the per-class atlas posteriors span the cellular composition space).

Walther also runs streaming-mode: if the input contains >10K CpGs, it processes in chunks of 1K markers, accumulating partial sums and re-solving every 10 chunks. This is the methylome's analog of streaming map-making — handles arbitrarily large CpG counts without exceeding memory.

**CMB equivalent.** This is **Commander Bayesian component separation.** Planck's Commander algorithm solves the analogous inverse problem: given multi-frequency CMB observations, decompose them into cosmological + foreground components using a Gibbs sampler over the joint posterior. Walther's NNLS is a constrained-optimization analog of Commander's Bayesian Gibbs sampling — same inverse problem, different solver. NNLS gives a point estimate; Commander gives a posterior distribution. CPG's Phase E will add a proper posterior to Walther's output (currently just a point estimate + status flags).

**How the methylome differs in implementation.** Walther produces a point estimate per patient (faster, simpler) while Commander produces a posterior (slower, richer). The choice reflects clinical pipeline constraints — patient reports need fast, deterministic outputs.

**How it's the same in principle.** Both decompose a multi-channel observation into class/component contributions using a known reference matrix and a physical constraint (non-negativity / positivity of densities).

**Outputs.** Per patient:
- 8 architectural-class fractions: stem_pluri, stem_adult, progenitor, cycling, secretory, immune, terminal, stromal (each in [0,1], summing to 1)
- 115 per-cell-type fractions (Walther's Tier 2 output — finer-grained decomposition within each class)
- Per-class residual MAE: how well the reference matrix explains the patient's β values at marker CpGs

**Decision points.**
- IF NNLS converges and class residual MAE < 0.05 → proceed to §31 (status code assignment)
- IF NNLS converges but residual MAE 0.05-0.10 → flag `MODERATE_RESIDUAL`, proceed
- IF NNLS doesn't converge OR residual MAE > 0.10 → flag `POOR_DECONVOLUTION`, do not advance until §33 cross-method gate verifies

**Failure modes.**
- *Singular reference matrix.* Detected by NNLS condition number check.
- *Borderline patient mass between two classes.* NNLS pushes mass to the single best class (vs GLS which splits) — this is the expected NNLS inductive bias and is documented in Phase B2.1 finding.

**Canonical cross-references.** Recipe §3.2 (Walther deconvolver). Roadmap §10.1.1 L4 row. Capability Translator §1, §2, §5, §6.

**CPG Plate references.** Plate 1 (the 8-class atlas Walther deconvolves against).

**Chain-link assignment.** L4 (primary component separation).

---

### §31. Step 2.4 — Walther per-class confidence + status codes

**What this step does.** Walther emits per-class confidence indicators alongside each fraction estimate. Confidence is bounded [0, 1] and is NOT a calibrated probability — it's a coverage × fit-quality composite. Status codes flag specific failure modes (insufficient markers, no overlap, etc.) so downstream stages can route appropriately.

**Inputs.** NNLS solution from §30 + the per-class marker coverage metrics.

**Atlas reference.** Not consulted further at this step — uses outputs from §30.

**Files invoked.** Same module as §30 (`walther_iam_deconvolver.py`).

**The math.** Per class:
> **confidence_class = coverage_class × max(0, 1 − dispersion_class / 0.20)**

where:
- **coverage_class** = (markers_matched_for_class / markers_expected_for_class) — bounded [0,1]
- **dispersion_class** = stdev of per-CpG β within the class's marker pool (a high stdev indicates poor agreement between CpGs supposedly diagnostic of the same class)
- The 0.20 ceiling converts dispersion to a [0,1] penalty

Confidence is conservative — designed to fail loudly when something is wrong rather than report a falsely high number.

**Status codes:**
- `OK` — normal operation
- `INSUFFICIENT_MARKERS` — fewer than 20 markers matched for the class
- `NO_MARKER_OVERLAP` — zero markers matched
- `HIGH_DISPERSION` — dispersion exceeds 0.30 (markers disagree internally)
- `RESIDUAL_HIGH` — class-level residual exceeds the cohort-typical threshold

**CMB equivalent.** This is the **per-component confidence diagnostic in Commander.** Commander outputs not just the component amplitudes but also their posterior widths, residual amplitudes per pixel, and goodness-of-fit per region. CPG's confidence + status codes are the operational simplification of that diagnostic suite — calibrated [0,1] confidence for clinician communication + explicit failure flags for chain routing.

**How the methylome differs in implementation.** Single scalar confidence per class + discrete status code, vs Commander's per-pixel residual maps.

**How it's the same in principle.** Both attach explicit reliability indicators to every decomposition output. Both refuse to silently report a low-quality decomposition as if it were high-quality.

**Outputs.** Per class:
- `confidence_class` (float in [0,1])
- `status_class` (one of the codes above)

**Decision points.**
- IF all 8 classes have status=OK → proceed to §32
- IF any class has INSUFFICIENT_MARKERS or NO_MARKER_OVERLAP → flag that specific class, proceed with others
- IF >2 classes have non-OK status → quarantine with `BROAD_DECONVOLUTION_FAILURE`

**Failure modes.** Cascade from §30; specific failure modes are captured by the status codes themselves.

**Canonical cross-references.** Recipe §3.2. Capability Translator §2, §3.

**CPG Plate references.** Plate 1 stromal panel (the 4.9% MCMC coverage gap manifests as low stromal confidence for many patients — the methylome's known unknown).

**Chain-link assignment.** L4.

---


**Step 2.4b — Credible intervals from MCMC posteriors.** The atlas is constructed via MCMC over per-CpG β posteriors per architecture class (frozen 2026-04-06, R-hat < 1.001). Each atlas cell `(CpG, class)` has both a posterior mean (μ) AND a posterior SD (σ). The Walther deconvolver consumes both: μ to solve the NNLS system, σ to propagate uncertainty into the returned fractions. The Stage 2 output therefore carries per-class fraction point estimates AND credible intervals — a 0.05 fraction with CI [0.02, 0.08] is reported differently from a 0.05 fraction with CI [0.04, 0.06] in the doctor report's Quality section (§75). This is the only place in the V1 chain where atlas L8 MCMC information enters the per-patient flow; downstream stages operate on the point estimates with the CIs carried as a parallel uncertainty channel.

### §32. Step 2.5 — NILC v2 deconvolver (Path 2, departure-from-consensus GLS)

**What this step does.** Runs the cross-method check — NILC v2 — on the same patient β values, same marker pool, same atlas reference. NILC uses a fundamentally different inversion algorithm (departure-from-consensus generalized least squares with simplex projection) to produce an independent per-class fraction estimate. This is the methylome's Commander/NILC/SMICA discipline.

**Inputs.** Same as §30: marker pool table from §29.

**Atlas reference.** **IAMAtlas consulted.** Same reference matrix as Walther — but used differently (GLS in departure space rather than NNLS in raw space).

**Files invoked.**
- NILC deconvolver: `Biological_Physics/chain_of_custody/L4_component_separation/nilc_deconvolver.py`

**The math.** NILC v2 (Phase B2.1 algorithm):
1. **Build the consensus signal** — at each marker CpG, compute the mean of the 8 class posterior means (the "consensus β" at that CpG).
2. **Compute departures** — subtract the consensus from the reference matrix and from the patient β. Both are now expressed as departures from consensus.
3. **GLS solve** — solve the unconstrained generalized least squares system on the departure space, with per-CpG inverse-variance weighting (uses the atlas posterior SD as the variance).
4. **Simplex project** — the raw GLS solution can have negative or unbounded values; project it onto the simplex (non-negative, summing to 1) to get the final fraction estimate.

The mathematical form:
> **f_GLS = (X_dep^T · W · X_dep)⁻¹ · X_dep^T · W · β_dep**

where X_dep is the departure-form reference matrix, β_dep is the departure-form patient β, and W is the diagonal inverse-variance weight matrix. After solving, f_GLS is simplex-projected.

**CMB equivalent.** This is **Planck NILC** — literally the same algorithm name. Planck's Needlet Internal Linear Combination decomposes the CMB into component signals by constrained linear combinations of multi-frequency channels, with the constraint that the combination preserves the desired signal while minimizing residual variance from contaminants. CPG's NILC v2 is the methylome implementation of this exact discipline. The "needlet" structure of CMB NILC (multipole-localized wavelets) is replaced by the marker-pool structure of CPG NILC (class-localized CpG groups).

**How the methylome differs in implementation.** Methylome operates in the discrete-class space (8 classes) rather than the continuous angular-scale space (multipoles). The departure-from-consensus reformulation is the methylome-specific innovation that handles the non-orthogonal columns of the class reference matrix.

**How it's the same in principle.** Both produce an independent per-component estimate via GLS-with-constraints, intended to cross-check a primary deconvolver (Commander or Walther) at the inference layer.

**Outputs.** Per patient:
- 8 architectural-class fractions (independent of Walther's)
- Per-class residual after simplex projection

**Decision points.** Proceed to §33 (cross-method gate). NILC failure does not block the chain at this point — the gate decides.

**Failure modes.**
- *Ill-conditioned departure matrix.* Detected by GLS condition number; falls back to a ridge-regularized variant.
- *Aggressive simplex projection (when GLS produces many negative values).* The simplex projection can flatten the distribution; documented as a known NILC characteristic.

**Canonical cross-references.** Phase B2.1 finding doc at `Biological_Physics/chain_of_custody/L4_component_separation/Phase_B2_1_FINDING.md`. Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plate 1 (the same 8 classes that NILC decomposes against).

**Chain-link assignment.** L4 (cross-method component separation).

---

### §33. Step 2.6 — Cross-method gate check

**What this step does.** Compares Walther's and NILC's per-class fraction estimates to verify they agree at the biological-inference layer (sign and rough magnitude of effects). Substrate-level disagreement (Walther vs NILC fractions differ by ~12 percentage points in median L1 on EPIC-Italy) is EXPECTED — that's the inductive-bias difference between NNLS and GLS. Inference-level disagreement (the two methods give opposite signs for the disease-relevant class) is a problem.

**Inputs.** Walther fractions from §30 + NILC fractions from §32.

**Atlas reference.** Not consulted at this step — operates on the two deconvolvers' outputs.

**Files invoked.**
- Gate logic: built into `nilc_deconvolver.py` as the cross-method comparison routine.

**The math.** Two-level comparison:
- **Substrate level (informational, not gating):** per-class L1 distance |f_Walther − f_NILC|; cohort median expected ≈ 0.10-0.25. Reported but does not gate.
- **Inference level (gating):** for the disease-relevant class (varies per card), compare the SIGN of (case − HC) effect under each deconvolver. If both methods produce the same sign on the disease-relevant class, the gate PASSES. If they disagree on sign, the gate FIRES.

In production for a single patient (no case/HC labels available), the gate uses a per-class agreement threshold: both methods must place the patient on the same SIDE of the cohort median for the dominant class. If the patient is in the top quartile for immune A-score by Walther but the bottom quartile by NILC, the gate fires.

**CMB equivalent.** This is **Planck's cross-method cosmology consistency check.** Planck publishes its primary cosmological parameter estimates from Commander but cross-validates against NILC, SMICA, and SEVEM. Disagreement at the per-pixel temperature level is expected and quantified as systematic uncertainty. Disagreement at the parameter level (e.g., different methods producing different H₀) is a real problem that gates publication.

**How the methylome differs in implementation.** Two methods (Walther + NILC) rather than four. Single per-patient inference (vs cosmological-parameter inference). Per-class agreement rather than per-parameter agreement.

**How it's the same in principle.** Both treat method-method agreement at the inference layer as a hard gate. Both document substrate-level disagreement as a systematic to propagate forward.

**Outputs.** Per patient:
- A cross-method gate verdict: PASS / FLAG / FAIL
- Per-class substrate disagreement (the L1 distances) for the audit trail
- For PASS: proceed with Walther's fractions as the production answer
- For FLAG: proceed but with cross-method-uncertainty annotation on the report
- For FAIL: do not advance; route for manual review

**Decision points.**
- IF gate PASS → proceed to §34 (Stage 2 output packaging)
- IF gate FLAG → proceed but stamp report with cross-method advisory
- IF gate FAIL → quarantine with `CROSS_METHOD_DISAGREEMENT`, require operator review

**Failure modes.**
- *Genuine biological anomaly.* The patient has cellular architecture that lies in a region where Walther's NNLS-bias and NILC's GLS-bias produce different inferences. Documented; the patient's case may itself be informative.
- *Upstream QC failure that wasn't caught.* The gate firing can be a downstream detection of an upstream calibration issue.
- *Reference matrix instability for this patient's substrate.* Rare; documented case by case.

**Canonical cross-references.** Phase B2.1 finding doc. Recipe §3.3 (Cross-method gate, added v3).

**CPG Plate references.** Plate 1 (cross-method comparison validates the per-class atlas posteriors are stable).

**Chain-link assignment.** L4 (cross-method component-separation discipline).

---

### §34. Step 2.7 — Stage 2 output

**What this step does.** Packages the Stage 2 outputs into the canonical structure consumed by Stage 3 and beyond.

**Inputs.** Walther fractions + NILC fractions + cross-method gate verdict.

**Atlas reference.** Not consulted.

**Files invoked.**
- Stage 2 packager: `<Stage 2 packaging inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** The **component-separated CMB map handoff** to the foreground-subtraction stage (galactic dust, synchrotron removed; cosmological CMB ready for further processing).

**How the methylome differs in implementation.** Per-patient single-sample output rather than full-sky pixelized map.

**How it's the same in principle.** Both produce the cleaned cosmological/biological signal for downstream physics analysis.

**Outputs.**
- Production fractions (Walther's, per Stage 2.3): 8 classes + 115 cell types
- Per-class confidence + status codes (per Stage 2.4)
- Cross-method gate verdict (per Stage 2.6)
- NILC fractions stored for audit trail (per Stage 2.5)

Location: `<Stage 2 output — emitted internally by `GAPE_WEB_v13.py`>`.

**Decision points.** Handoff to §35 (Step 3.1 Age-axis foreground subtraction) for the now-deconvolved data.

**Failure modes.** Output packaging is mechanical.

**Canonical cross-references.** Recipe §3 (end of deconvolution). Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plates 1, 4 (the deconvolved per-class signals are what get visualized in the architectural-class panels).

**Chain-link assignment.** L4 (primary closes; secondary begins at §35).

---

## Stage 3 — Foreground subtraction (L4 component separation, secondary)

---

### §35. Step 3.1 — Age-axis foreground subtraction

**What this step does.** Removes the per-CpG age-drift component from the patient's β values before downstream A-score computation. Methylation drifts predictably with age at many CpGs (the basis of all methylation clocks); this drift is a foreground in the same sense that galactic dust is a foreground for CMB cosmology. Subtracting it cleans the disease-signal component.

**Inputs.** The Stage 2-deconvolved β data + the patient's declared chronological age.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the per-CpG age layer at `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv` (8,199 CpGs × {α intercept, γ slope per year, R², n_samples}), built from EPIC-Italy HC by the Phase B3 age-axis foreground module.

**Files invoked.**
- Foreground module: `Biological_Physics/atlas_vault/components/age_axis_foreground.py`
- Age layer: `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv`

**The math.** Per CpG i:
> **β_corrected[i] = β_observed[i] − γ_i × (age − age_train_mean)**

where γ_i is the per-CpG age slope (from the age layer) and `age_train_mean` is the mean training age (also stored in the age layer). This subtracts the age-drift component while preserving everything else.

Per-CpG R² values are stored alongside; the engine could optionally weight subtraction by R² (only apply correction to CpGs where the age fit is reliable), but the default is unweighted subtraction across all CpGs in the age layer.

**CMB equivalent.** This is **galactic foreground dust subtraction.** Planck removes the galactic dust contribution from each pixel using a multi-frequency template (since dust has a known frequency dependence). Once subtracted, the remaining signal is closer to the pure cosmological CMB. Same operation here: subtract a known foreground component (age drift) to expose the disease signal underneath.

**How the methylome differs in implementation.** The "frequency dependence" of methylome foreground is the per-CpG slope vs age (a linear model). Galactic dust uses a power-law frequency spectrum.

**How it's the same in principle.** Both subtract a known foreground component using a per-coordinate template, leaving the cleaned signal of interest.

**Outputs.** β_corrected matrix (same shape as the input, with age-drift subtracted).

**Decision points.**
- IF patient's age is within the calibration range of the age layer → apply correction, proceed to §36
- IF patient's age is outside the calibration range (very young or very old) → apply with extrapolation warning, proceed with reduced confidence on downstream age-related readouts

**Failure modes.**
- *Patient declared age is wrong.* Manifests as anomalous post-correction β distribution. Not directly detectable here; surfaces at §41 (β_mean computation) if it's severe.
- *Per-CpG age slope is wrong because the calibration cohort doesn't generalize to this patient's population.* This is the known limitation of the EPIC-Italy-only calibration; Phase B4 will add population-specific age layers.

**Canonical cross-references.** Recipe §4.1 (Age foreground). Roadmap §10.2.2 Phase B3 (the age-axis module). Phase B3 finding doc at `Biological_Physics/atlas_vault/age_clock/Phase_B3_FINDING.md`.

**CPG Plate references.** Plate 4 Panel D (Differentiation Gradient — illustrates how age drift maps onto the methylome sphere).

**Chain-link assignment.** L4 (secondary component separation).

---

### §36. Step 3.2 — Sex-axis foreground subtraction (when present)

**What this step does.** Removes the per-CpG sex-specific methylation component. Methylation patterns differ systematically between males and females at many autosomal CpGs (not just chrX/chrY). For multi-sex cohorts, subtracting this foreground prevents sex from contaminating the disease signal.

**Inputs.** β_corrected from §35 + patient's declared sex (from manifest, validated at §18).

**Atlas reference.** **Currently not implemented at production grade** — the Roadmap Phase B4 (sex foreground module) is pending. The SOP declares this step as future work.

**Files invoked.** *Future:* `[Phase B4 deliverable per Roadmap §10.2.2 — module not yet built]`. Currently: SKIP if no module is loaded.

**The math.** *When implemented:* Per CpG i:
> **β_corrected[i] = β_observed[i] − δ_i × sex_indicator**

where δ_i is the per-CpG female-vs-male shift (learned from same-cohort HC by sex) and sex_indicator ∈ {−1, +1} maps {male, female} respectively. Centered so the cohort mean is zero across sexes.

**CMB equivalent.** This is **secondary foreground subtraction** — like removing synchrotron emission after dust. Each foreground has its own template and subtraction operator.

**How the methylome differs in implementation.** Currently unimplemented. The SOP declares this as Phase B4 work.

**How it's the same in principle.** When implemented, it follows the same pattern as age-axis subtraction.

**Outputs.** *When implemented:* β_corrected matrix with sex component removed.

**Decision points.** Currently: PASS THROUGH (no correction applied). Future: apply correction, proceed to §37.

**Failure modes.** *When implemented:* same pattern as age foreground.

**Canonical cross-references.** Roadmap §10.2.2 B4 (foreground modules pending).

**CPG Plate references.** Not yet applicable.

**Chain-link assignment.** L4 (secondary).

---

### §37. Step 3.3 — Batch/plate foreground subtraction (when present)

**What this step does.** Removes per-CpG systematic shifts due to plate position, intake batch, or processing date. ComBat (§22) handles this at the calibration level when multi-batch data is being processed together; this step would handle residual batch effects at the post-deconvolution level.

**Inputs.** β_corrected from §36 + batch metadata from the per-sample record.

**Atlas reference.** Not directly — batch shifts are sample-cohort-specific, not atlas-anchored.

**Files invoked.** *Future:* `[Phase B4 deliverable per Roadmap §10.2.2 — module not yet built]`. Currently: SKIP.

**The math.** *When implemented:* per-CpG residual batch shift learned from the current cohort's HC samples; subtracted before A-score computation.

**CMB equivalent.** This is **mission-mission residual systematic correction** — handling the residual after ComBat-equivalent global batch correction.

**How the methylome differs in implementation.** Currently unimplemented at this stage (ComBat handles it at calibration). Future Phase B4 work would add a downstream verification layer.

**How it's the same in principle.** Layered systematic-correction discipline — multiple correction stages, each catching residuals from upstream.

**Outputs.** *When implemented:* β_corrected with residual batch shifts removed.

**Decision points.** Currently: PASS THROUGH.

**Failure modes.** *When implemented:* documented per the §35 pattern.

**Canonical cross-references.** Roadmap §10.2.2 B4.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L4 (secondary).

---

### §38. Step 3.4 — Ancestry-axis foreground subtraction (when present)

**What this step does.** Removes per-CpG ancestry-specific methylation differences. Genetic ancestry correlates with methylation at thousands of CpGs (some via mQTL effects, some via developmental differences); separating ancestry signal from disease signal requires this foreground subtraction for diverse cohorts.

**Inputs.** β_corrected from §37 + patient's inferred or declared ancestry.

**Atlas reference.** **Currently not implemented at production grade** — the Roadmap Phase B4 covers this. Requires ancestry-informative-marker (AIM) panels and per-CpG ancestry slope coefficients.

**Files invoked.** *Future:* `[Phase B4 deliverable per Roadmap §10.2.2 — module not yet built]`. Currently: SKIP.

**The math.** *When implemented:* methylome PCA on AIM CpGs to infer ancestry coordinates, then per-CpG subtraction along the inferred ancestry direction.

**CMB equivalent.** This is **CMB secondary foreground subtraction (e.g., extragalactic point sources, Sunyaev-Zel'dovich clusters)** — a known foreground that varies in pattern but not in physical origin.

**How the methylome differs in implementation.** Currently unimplemented. Phase B4 deliverable.

**How it's the same in principle.** Same pattern as age and sex foregrounds when implemented.

**Outputs.** *When implemented:* β_corrected with ancestry component removed.

**Decision points.** Currently: PASS THROUGH.

**Failure modes.** *When implemented:* misinference of ancestry from AIM panel; detected via inconsistency with self-declared ancestry.

**Canonical cross-references.** Roadmap §10.2.2 B4.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L4 (secondary).

---

### §39. Step 3.5 — Smoking-axis foreground subtraction (when card requires it)

**What this step does.** Removes per-CpG smoking-status-specific methylation differences. Several disease cards (notably lung-epic) have smoking status as a known covariate that confounds the disease signal. Smoking foreground subtraction is card-conditional — it runs only when the card declares smoking as a covariate.

**Inputs.** β_corrected from §38 + patient's declared smoking status (from intake metadata, if collected).

**Atlas reference.** Not directly — smoking-CpG associations are well-characterized in the literature (notably AHRR cg05575921). The runtime artifact would be a per-CpG smoking shift table.

**Files invoked.** *Future:* `[Phase B4 deliverable per Roadmap §10.2.2 — module not yet built]`. Currently: handled per-card, not at this stage.

**The math.** *When implemented:* per-CpG smoking shift δ_smoking[i] subtracted for current smokers, with intermediate values for former smokers based on years since cessation.

**CMB equivalent.** This is **point-source masking + subtraction** — handling a known per-source contamination component.

**How the methylome differs in implementation.** Currently handled per-card (e.g., lung-epic adjusts thresholds based on smoking status) rather than as a global Stage 3 foreground.

**How it's the same in principle.** Card-specific covariate handling vs global foreground subtraction is a design choice; the principle of removing a known confounder before disease-signal extraction is the same.

**Outputs.** *When implemented:* β_corrected with smoking component removed.

**Decision points.** Currently: PASS THROUGH (cards handle smoking inline at §68).

**Failure modes.** *When implemented:* patient under-reports smoking, leading to under-correction.

**Canonical cross-references.** Recipe §4 (Smoking handling). Roadmap §10.2.2 B4.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L4 (secondary).

---

### §40. Step 3.6 — Stage 3 output: cleaned β matrix

**What this step does.** Packages the foreground-subtracted β matrix into the canonical Stage 3 output: a β matrix with age (and, when modules are implemented, sex/batch/ancestry/smoking) components removed, ready for Stage 4 A-score computation.

**Inputs.** β_corrected from §35-§39.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Stage 3 packager: `<Stage 3 packaging inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** The **foreground-cleaned CMB map** — what Planck publishes as the SMICA/Commander/NILC/SEVEM cleaned products, ready for power spectrum estimation.

**How the methylome differs in implementation.** Per-patient cleaned β matrix rather than full-sky cleaned map.

**How it's the same in principle.** Both represent the cleaned signal of interest, foregrounds-removed.

**Outputs.** A canonical Stage 3 output file containing:
- The cleaned β matrix
- The list of foregrounds subtracted (age, sex [if implemented], batch [if implemented], ancestry [if implemented], smoking [if card requires])
- The per-CpG slopes used for each subtraction (for audit trail)

Location: `<Stage 3 output — emitted internally by `GAPE_WEB_v13.py`>`.

**Decision points.** Handoff to §41 (Step 4.1 Per-class β_mean computation).

**Failure modes.** Output packaging is mechanical.

**Canonical cross-references.** Recipe §4 (end of foreground). Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plate 4 Panel A (Class-Difference Map shows what's visible AFTER foreground subtraction — the differentiation gradient between architectural classes).

**Chain-link assignment.** L4 (closes).

---

## Stage 4 — A-score computation (entropy scoring)

---

### §41. Step 4.1 — Per-class β_mean computation

**What this step does.** For each of the 8 architectural classes, computes the mean β across the class's marker CpGs (~2,978 markers per class for the 8-class scoring). This is the canonical per-class β statistic that feeds the entropy formula.

**Inputs.** Cleaned β matrix from §40 + per-class marker CpG lists from §29.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the per-class marker CpG lists at `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json`. (Markers are AGGREGATED to class — each class's marker pool is the union of its constituent cell types' marker pools.)

**Files invoked.**
- A-scoring module: `pipeline_runtime_matrices/iamatlas_a_scoring.py` (specifically: `score_per_class()`)

**The math.** For each class c:
> **β_mean_c = mean(β_corrected[i]) for i in class_markers[c]**

This is the CANONICAL β_mean per Recipe §6.3 — H is computed on the MEAN β across class markers, NOT the mean of per-CpG entropies. The choice matters: Jensen's inequality means mean(H(β_i)) ≤ H(mean(β_i)), so the two formulas give different numbers. Recipe canonical is H of the mean.

**CMB equivalent.** This is the **bandpower computation** at the per-class level — collapsing many individual measurements (per-CpG β values) into a single class-level statistic (β_mean) the way Planck collapses many per-pixel temperature measurements into a single multipole-bin power.

**How the methylome differs in implementation.** Aggregation is by class membership (not by angular scale).

**How it's the same in principle.** Both reduce many measurements to a single summary statistic that captures the per-component physical quantity of interest.

**Outputs.** A per-class β_mean vector: 8 values (one per class), each in [0, 1].

**Decision points.**
- IF all 8 classes have ≥30 marker CpGs with valid β → proceed to §42
- IF any class has <30 valid markers → flag class with `INSUFFICIENT_CPGS`, set its A-score to NULL, proceed with remaining classes

**Failure modes.**
- *All CpGs failing in one class.* Detected here. Causes that class's A-score to be NULL.
- *Atlas marker list staleness.* If the marker file's SHA-256 doesn't match the expected anchor, refuse to compute (caught upstream at §29).

**Canonical cross-references.** Recipe §6.1 (β_mean), §6.3 (canonical formula). Capability Translator §4.

**CPG Plate references.** Plate 1 (the per-class atlas panels themselves are the visualization of the per-class β reference structure that markers sample).

**Chain-link assignment.** Between L3 (β values) and L4 (decomposition).

---

### §42. Step 4.2 — Shannon entropy H(β_mean) calculation

**What this step does.** For each class's β_mean, computes the binary Shannon entropy.

**Inputs.** Per-class β_mean from §41.

**Atlas reference.** Not consulted at this step.

**Files invoked.** Same module as §41 (`iamatlas_a_scoring.py`).

**The math.** For each class c:
> **H_c = −β_mean_c · log₂(β_mean_c) − (1 − β_mean_c) · log₂(1 − β_mean_c)**

(with the convention 0 · log(0) = 0). H is bounded [0, 1] — maximum at β = 0.5, zero at β ∈ {0, 1}.

**CMB equivalent.** This is the **per-component variance computation** — extracting the information-theoretic content of the class-level signal. CMB cosmology uses the C_ℓ power spectrum (variance per multipole) as its primary information-theoretic quantity; CPG uses entropy per class as its analog. Both quantify "how much information is encoded in this component."

**How the methylome differs in implementation.** Entropy of a Bernoulli probability rather than variance of a Gaussian — appropriate because methylation β is a probability (fraction methylated), not a continuous-valued temperature.

**How it's the same in principle.** Both compute an information-theoretic quantity that captures the per-component signal content.

**Outputs.** A per-class entropy vector: 8 values, each in [0, 1].

**Decision points.** Proceed to §43.

**Failure modes.**
- *β_mean = 0 or β_mean = 1 (boundary cases).* H = 0 by convention. Detected and propagated; downstream A-score will be 0 for that class, which is itself a measurement (extreme commitment to one methylation state).

**Canonical cross-references.** Recipe §6.2 (Shannon entropy). Roadmap §7 (Bidirectional calibration solved by entropy space).

**CPG Plate references.** Plate 3 (the bimodal vs Gaussian texture comparison — the methylome's non-Gaussianity is what makes the entropy framework powerful).

**Chain-link assignment.** Between L3 and L4.

---

### §43. Step 4.3 — Per-class A-score: A = H(β_mean) / H_min(class)

**What this step does.** Divides each class's entropy by the class's H_min (the Mahaffey Number) to produce the canonical A-score: a dimensionless ratio measuring departure from the architectural floor.

**Inputs.** Per-class entropy from §42 + H_min values for each class.

**Atlas reference.** **H_min values consulted** from `IAMAtlasREBUILD_provenance.json` (one source of truth, frozen 2026-04-06).

**Files invoked.** Same module as §41 (`iamatlas_a_scoring.py`).

**The math.** For each class c:
> **A_c = H_c / H_min(c)**

The Mahaffey Numbers (frozen 2026-04-06):
- terminal: H_min = 0.7728
- immune: H_min = 0.838889
- secretory: H_min = 0.843264
- cycling: H_min = 0.856055
- progenitor: H_min = 0.852216
- stromal: H_min = 0.86295
- stem_adult: H_min = 0.873718
- stem_pluri: H_min = 0.982166

**A_c < 1** means below the floor (suppressed below baseline order).
**A_c = 1** means at the floor (architectural minimum).
**A_c > 1** means above the floor (departure from baseline — disorder above the minimum, which is what disease drives).

**CMB equivalent.** This is **normalizing the per-component variance against a known physical scale** — like Planck computing Ω_class = ρ_class / ρ_critical, where ρ_critical is the critical density (a fixed physical constant). The dimensionless ratio carries more information than the raw variance because it's calibrated against a meaningful reference.

**How the methylome differs in implementation.** The reference (H_min) is the architectural floor of each class; ρ_critical is the cosmological scale for a flat universe.

**How it's the same in principle.** Both produce dimensionless physical ratios — direct, interpretable, calibrated against a known anchor — rather than raw variance.

**Outputs.** A per-class A-score vector: 8 values. Each value is a dimensionless ratio.

**Decision points.** Proceed to §44.

**Failure modes.**
- *H_min file out of date.* If `IAMAtlasREBUILD_provenance.json` SHA doesn't match the engine's expected hash, refuse to run.
- *Class entropy NULL (from §42).* A-score is NULL for that class; propagates downstream.

**Canonical cross-references.** Recipe §6.3 (canonical A-score formula). Capability Translator §4 (A-score plain language). Roadmap §4 (What physics brings).

**CPG Plate references.** All four plates (the A-score derivation underlies the visualization of departure from the architectural floor across the sphere).

**Chain-link assignment.** Between L3 and L4.

---

### §44. Step 4.4 — Per-cell-type A-score (115 cell types)

**What this step does.** Same operation as §41-§43 but at finer granularity — 115 cell types rather than 8 architectural classes. The cell type's H_min is looked up via its class membership (each cell type inherits its class's H_min). This produces the per-cell-type A-score surface that the Mahalanobis distance (§47-§50) operates on.

**Inputs.** Cleaned β matrix from §40 + per-cell-type marker CpG lists from `iamatlas_celltype_markers_v0_2.json` + the `celltype_to_class` mapping from the same artifact.

**Atlas reference.** **IAMAtlas consulted.** Per-cell-type markers + class assignment from `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json`. H_min values still read from `IAMAtlasREBUILD_provenance.json`.

**Files invoked.** `iamatlas_a_scoring.py` (specifically: `score_per_celltype()`).

**The math.** For each cell type ct (115 total):
1. Compute β_mean_ct = mean(β over the cell type's ~100 marker CpGs)
2. Compute H_ct = Shannon entropy of β_mean_ct
3. Look up class assignment: c = celltype_to_class[ct]
4. Compute A_ct = H_ct / H_min(c)

The class assignment is necessary because H_min is per-class (the architectural floor lives at the class level, not the cell-type level — many cell types share the same architectural identity).

**CMB equivalent.** This is **per-narrow-band power spectrum estimation** — extracting the signal at finer angular scales than the headline per-multipole-bin estimates. CMB cosmologists often compute both broad ℓ-binned and narrow ℓ-binned power spectra for different inference targets.

**How the methylome differs in implementation.** 115 cell types vs many possible ℓ-binnings in CMB.

**How it's the same in principle.** Both produce a finer-grained decomposition of the signal alongside the headline per-class output.

**Outputs.** A per-cell-type A-score vector: 115 values + per-cell-type confidence + status codes (same structure as the per-class output).

**Decision points.** Proceed to §45.

**Failure modes.**
- *Cell types with insufficient markers in patient β.* Common for rare cell types where many markers failed QC. Status code `INSUFFICIENT_MARKERS` set; A-score NULL; downstream Mahalanobis imputes via median (with the imputation count tracked).

**Canonical cross-references.** Recipe §6.4 (Per-cell-type A-score). Capability Translator §6. Roadmap §10.2.1 TODO 1.1 (per-cell-type one-vs-rest).

**CPG Plate references.** Plate 2 (per-CpG residual at marker level — the cell-type-specific markers are what carry the disease-residual signal).

**Chain-link assignment.** Between L3 and L4.

---

### §45. Step 4.5 — Disease panel A-score (when card has a curated panel)

**What this step does.** When a specific disease card includes a curated CpG panel (e.g., the Xu-538 immune panel used for breast pre-dx), compute the A-score on that panel using the same H(β)/H_min formula but restricted to the panel's CpGs and using the class H_min that the panel anchors to.

**Inputs.** Cleaned β matrix from §40 + the card's panel CpG list + the class H_min the panel anchors to.

**Atlas reference.** Indirectly — the H_min anchor comes from IAMAtlas. The panel itself is card-specific (defined by the card author per the card spec).

**Files invoked.**
- Same module: `iamatlas_a_scoring.py`
- Card-specific panel artifact: e.g., `<card-specific panel CpG list — currently embedded in card config inside `GAPE_WEB_v13.py`>`

**The math.** Same as §43, but:
- β_mean_panel = mean(β_corrected[i] for i in panel_cpgs)
- A_panel = H(β_mean_panel) / H_min(class_anchor)

For the Xu-538 panel anchored to immune class: A_panel = H(β_mean over 538 CpGs) / 0.838889.

**CMB equivalent.** This is **targeted-scale analysis** — running a power spectrum on a pre-selected multipole window known to be informative for a specific cosmological question (e.g., the recombination acoustic peak window for sound-horizon measurement).

**How the methylome differs in implementation.** Panels are typically card-specific and disease-specific (e.g., Xu-538 is breast-pre-dx-specific). CMB targeted windows are typically question-specific (e.g., the ℓ ≈ 220 window for sound horizon).

**How it's the same in principle.** Both compute the same core physical quantity (power / entropy) restricted to a pre-selected informative subset of the data.

**Outputs.** A per-card disease panel A-score: 1 value per card per patient.

**Decision points.**
- IF card has a curated panel → compute panel A-score, proceed to §46
- IF card uses only class/cell-type A-scores → SKIP, proceed to §46

**Failure modes.**
- *Panel CpGs not present in patient's IDAT.* Detected here; falls back to class-level A-score if panel coverage is too low.

**Canonical cross-references.** Recipe §6.5 (Panel A-scores). Capability Translator §7. Roadmap §10.2.5 E1 (per-card likelihoods include panel A-scores).

**CPG Plate references.** Plate 2 (the breast pre-dx panel signal visualized on the methylome sphere).

**Chain-link assignment.** Between L3 and L4 (panel-level).

---

### §46. Step 4.6 — Stage 4 output

**What this step does.** Packages all A-scores (per-class, per-cell-type, per-card panel) into the canonical Stage 4 output.

**Inputs.** A-scores from §43-§45.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Stage 4 packager: `<Stage 4 packaging inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** The **full multi-scale power spectrum package** — per-multipole, per-band, per-target-window — handed off to inference.

**How the methylome differs in implementation.** Per-patient package rather than per-mission spectrum.

**How it's the same in principle.** Both package the complete set of physical measurements ready for inference.

**Outputs.** A canonical Stage 4 output file containing:
- 8 per-class A-scores + confidence + status
- 115 per-cell-type A-scores + confidence + status
- Per-card panel A-scores (one per applicable card)
- The H_min values used (audit trail)
- The atlas SHA-256 fingerprint (audit trail)

Location: `<Stage 4 output — emitted internally by `GAPE_WEB_v13.py`>`.

**Decision points.** Handoff to §47 (Step 5.1 Patient 115-cell-type A-score vector assembly).

**Failure modes.** Output packaging is mechanical.

**Canonical cross-references.** Recipe §6 (end of A-score computation). Capability Translator §4-§7.

**CPG Plate references.** Plate 4 Panel F (the per-CpG breast-anisotropy signal at the panel level).

**Chain-link assignment.** Between L3 and L4.

---

*End of Stages 0-4. Continued in Part II Stages 5-10 + Parts III-V.*
