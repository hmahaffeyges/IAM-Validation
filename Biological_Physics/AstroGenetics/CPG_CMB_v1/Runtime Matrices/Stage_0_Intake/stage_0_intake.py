"""Stage 0 — Sample intake (L1).

Built one step at a time against CPG_Chain_of_Custody_SOP_v1_3.md.
Implemented so far:
  Step 0.1 (SOP §11) — IDAT file arrival on server.
  Step 0.2 (SOP §12) — Sample manifest creation (patient_manifest.json + covariates).
  Step 0.3 (SOP §13) — IDAT integrity hash check (SHA-256 + re-transmission/re-run detection).
  Step 0.4 (SOP §14) — Control probe validation (BS conversion / hybridization / extension gates).

Design choices flagged for review (SOP marked these "TBD per orchestrator design"):
  - Handler lives here in the Walther runtime, NOT in GAPE_WEB_v13.py (frozen demo).
  - All paths (staging dir, intake log) are passed in by the caller, not hardcoded.
  - Manifest is JSON with the seven SOP-required fields.
  - Array-type verification reads only nSNPsRead from the IDAT header; full IDAT
    decoding is Stage 1's responsibility and is not duplicated here.

No math in this stage. Pure I/O + metadata + decision gate.
"""
from __future__ import annotations

import json
import hashlib
import os
import struct
import uuid
from datetime import datetime, timezone, timedelta

# --- SOP §11 spec constants -------------------------------------------------

REQUIRED_MANIFEST_FIELDS = (
    "sentrix_id",
    "array_type",
    "patient_id",
    "intake_date",
    "substrate",
    "declared_sex",
    "declared_chronological_age",
)

VALID_ARRAY_TYPES = ("HM450K", "EPIC_v1", "EPIC_v2")

# IDAT file-size sanity floors/expectations (SOP §11 failure modes).
MIN_PLAUSIBLE_IDAT_BYTES = 1_000_000          # < 1 MB is suspicious (truncated upload)
EXPECTED_IDAT_BYTES = {                         # rough per-channel sizes, for soft-warn
    "HM450K": 8_000_000,
    "EPIC_v1": 14_000_000,
    "EPIC_v2": 14_000_000,
}

# nSNPsRead -> platform inference. HM450 ~622k, EPIC v1 ~1.05M, EPIC v2 ~1.1M.
def _infer_platform(nsnps: int) -> str | None:
    if nsnps is None:
        return None
    if nsnps < 800_000:
        return "HM450K"
    if nsnps < 1_080_000:
        return "EPIC_v1"
    return "EPIC_v2"


# --- minimal IDAT header reader (nSNPsRead only) ----------------------------

def read_idat_nsnps(path: str):
    """Read only the magic + nSNPsRead (field code 1000) from an IDAT file.
    Returns (nsnps:int|None, status:str). Does NOT decode intensities (Stage 1)."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"IDAT":
                return None, "NOT_IDAT"
            struct.unpack("<q", f.read(8))[0]            # version (unused here)
            nfields = struct.unpack("<i", f.read(4))[0]
            offsets = {}
            for _ in range(nfields):
                code = struct.unpack("<H", f.read(2))[0]
                off = struct.unpack("<q", f.read(8))[0]
                offsets[code] = off
            if 1000 not in offsets:
                return None, "NO_NSNPS_FIELD"
            f.seek(offsets[1000])
            nsnps = struct.unpack("<i", f.read(4))[0]
            return nsnps, "OK"
    except FileNotFoundError:
        return None, "FILE_NOT_FOUND"
    except Exception as e:  # malformed header
        return None, f"IDAT_READ_ERROR:{type(e).__name__}"


# --- Step 0.1 ---------------------------------------------------------------

def step_0_1_idat_arrival(manifest_entry: dict,
                          grn_path: str,
                          red_path: str,
                          intake_log_path: str | None = None) -> dict:
    """SOP §11. Verify the Grn+Red IDAT pair + manifest, stamp arrival, gate.

    Returns a result dict with status, flags, advance(bool), and the canonical
    per-sample fields. `advance=True` means proceed to Step 0.2; False means quarantine.
    """
    flags: list[str] = []
    result = {
        "step": "0.1",
        "sample_run_id": str(uuid.uuid4()),
        "intake_timestamp": datetime.now(timezone.utc).isoformat(),
        "sentrix_id": manifest_entry.get("sentrix_id"),
        "array_type_declared": manifest_entry.get("array_type"),
        "array_type_detected": None,
        "substrate": manifest_entry.get("substrate"),
        "declared_sex": manifest_entry.get("declared_sex"),
        "declared_chronological_age": manifest_entry.get("declared_chronological_age"),
        "status": None,
        "flags": flags,
        "advance": False,
    }

    # (1) Manifest completeness -> INCOMPLETE_MANIFEST quarantine.
    missing = [k for k in REQUIRED_MANIFEST_FIELDS
               if manifest_entry.get(k) in (None, "")]
    if missing:
        flags.append("INCOMPLETE_MANIFEST:" + ",".join(missing))
        result["status"] = "QUARANTINE_INCOMPLETE_MANIFEST"
        _log_intake(intake_log_path, result)
        return result

    if manifest_entry["array_type"] not in VALID_ARRAY_TYPES:
        flags.append(f"UNKNOWN_ARRAY_TYPE:{manifest_entry['array_type']}")
        result["status"] = "QUARANTINE_INCOMPLETE_MANIFEST"
        _log_intake(intake_log_path, result)
        return result

    # (2) Pair completeness -> MISSING_CHANNEL quarantine (do not advance).
    have_grn, have_red = os.path.isfile(grn_path), os.path.isfile(red_path)
    if not (have_grn and have_red):
        missing_ch = []
        if not have_grn:
            missing_ch.append("Grn")
        if not have_red:
            missing_ch.append("Red")
        flags.append("MISSING_CHANNEL:" + ",".join(missing_ch))
        result["status"] = "QUARANTINE_MISSING_CHANNEL"
        _log_intake(intake_log_path, result)
        return result

    # (3) File-size sanity -> TRUNCATED_UPLOAD (hard) / size soft-warn.
    for ch, p in (("Grn", grn_path), ("Red", red_path)):
        sz = os.path.getsize(p)
        if sz < MIN_PLAUSIBLE_IDAT_BYTES:
            flags.append(f"TRUNCATED_UPLOAD:{ch}={sz}B")
    if any(f.startswith("TRUNCATED_UPLOAD") for f in flags):
        result["status"] = "QUARANTINE_TRUNCATED_UPLOAD"
        _log_intake(intake_log_path, result)
        return result

    # (4) IDAT-encoded array type vs declared -> ARRAY_TYPE_MISMATCH flag.
    nsnps, idat_status = read_idat_nsnps(grn_path)
    detected = _infer_platform(nsnps)
    result["array_type_detected"] = detected
    if idat_status != "OK":
        flags.append(f"IDAT_HEADER_UNREADABLE:{idat_status}")
    elif detected is not None and detected != manifest_entry["array_type"]:
        # EPIC_v1/EPIC_v2 are near in probe count; only hard-flag HM450<->EPIC family swaps.
        fam = lambda t: "450" if t == "HM450K" else "EPIC"
        if fam(detected) != fam(manifest_entry["array_type"]):
            flags.append(f"ARRAY_TYPE_MISMATCH:declared={manifest_entry['array_type']},detected={detected}")
            result["status"] = "QUARANTINE_ARRAY_TYPE_MISMATCH"
            _log_intake(intake_log_path, result)
            return result
        flags.append(f"ARRAY_SUBTYPE_NOTE:declared={manifest_entry['array_type']},detected={detected}")

    # (5) Duplicate intake within 24h -> soft-warn (still advances).
    if _is_duplicate_within_24h(intake_log_path, manifest_entry["sentrix_id"]):
        flags.append("DUPLICATE_INTAKE_24H_SOFTWARN")

    # All gates passed.
    result["status"] = "STAGED"
    result["advance"] = True
    _log_intake(intake_log_path, result)
    return result


# --- intake log helpers -----------------------------------------------------

def _log_intake(intake_log_path: str | None, result: dict) -> None:
    if not intake_log_path:
        return
    os.makedirs(os.path.dirname(intake_log_path), exist_ok=True)
    row = {k: result[k] for k in (
        "intake_timestamp", "sample_run_id", "sentrix_id", "array_type_declared",
        "array_type_detected", "substrate", "declared_sex",
        "declared_chronological_age", "status", "flags")}
    with open(intake_log_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _is_duplicate_within_24h(intake_log_path: str | None, sentrix_id: str) -> bool:
    if not intake_log_path or not os.path.isfile(intake_log_path):
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        with open(intake_log_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("sentrix_id") == sentrix_id:
                    ts = row.get("intake_timestamp", "")
                    try:
                        if datetime.fromisoformat(ts) >= cutoff:
                            return True
                    except ValueError:
                        continue
    except OSError:
        return False
    return False


# ============================================================================
# Step 0.2 (SOP §12) — Sample manifest creation
# Builds the canonical immutable patient_manifest.json: the §12 core record plus
# a covariate block parsed from the patient intake questionnaire v1.0.
#
# Design choices flagged for review:
#   - The covariate block lives INSIDE patient_manifest.json (the questionnaire says
#     covariates ride with the sample; §12's minimal structure didn't show them).
#   - The digital questionnaire submits answers keyed by the engine-use field names
#     the questionnaire itself defines (e.g. "known_autoimmune_condition").
#   - One immutable file per sample_run_id; re-intakes get a new run id (per §12).
# ============================================================================

# Canonical covariate field -> consuming stage (from the questionnaire's own tags).
COVARIATE_FIELDS = {
    "sex_at_birth": "Stage 7 stratification",
    "smoking_status": "Stage 7 bin + Stage 3 foreground",
    "smoking_quit_timing": "Stage 7 bin refinement",
    "smoking_years_total": "Stage 7 bin refinement",
    "recent_illness_within_3_months": "Stage 9 report context",
    "recent_vaccination_within_3_months": "Stage 9 report context",
    "current_pregnancy_with_trimester": "Stage 7 CONTEXT_PREGNANCY",
    "given_birth_within_6_months": "Stage 7 context",
    "menopause_status": "Stage 7 context",
    "hrt_status": "Stage 7 CONTEXT_HRT_BASELINE",
    "trt_status": "Stage 7 context",
    "current_glp1_or_weight_loss_medication": "Stage 7 CONTEXT_WEIGHT_LOSS_INTERVENTION",
    "bariatric_surgery_within_18_months": "Stage 7 context",
    "known_autoimmune_condition": "Stage 7 TRAJECTORY_WATCH",
    "known_chronic_inflammatory_disease": "Stage 7 TRAJECTORY_WATCH",
    "current_immunosuppression": "Stage 7 EXPECTED_SUPPRESSION",
    "current_cancer_in_treatment": "Stage 7 TREATMENT_RESPONSE",
    "prior_cancer_history": "Stage 7 remission-baseline context",
    "prior_chemotherapy_history": "Stage 7 context + Stage 8 progenitor card",
    "prior_radiation_history": "Stage 7 context",
    "transplant_status": "Stage 7 EXPECTED_SUPPRESSION",
    "hiv_status": "Stage 7 TRAJECTORY_WATCH (shifted thresholds)",
    "current_medications_systemic": "Stage 9 report context",
}

# Covariate (when answered and not a No/null) -> Stage 7 interpretation mode.
_CONTEXT_MODE_TRIGGERS = [
    ("current_cancer_in_treatment", "TREATMENT_RESPONSE"),
    ("current_immunosuppression", "EXPECTED_SUPPRESSION"),
    ("transplant_status", "EXPECTED_SUPPRESSION"),
    ("known_autoimmune_condition", "TRAJECTORY_WATCH"),
    ("known_chronic_inflammatory_disease", "TRAJECTORY_WATCH"),
    ("hiv_status", "TRAJECTORY_WATCH"),
    ("current_pregnancy_with_trimester", "CONTEXT_PREGNANCY"),
    ("hrt_status", "CONTEXT_HRT_BASELINE"),
    ("current_glp1_or_weight_loss_medication", "CONTEXT_WEIGHT_LOSS_INTERVENTION"),
]

_NULL_FOR_STORAGE = {"", "prefer_not_to_say", "prefer not to say"}
_NEGATIVE = {"no", "not_applicable", "not applicable", "n/a", "na", "none"}


def _is_null_answer(v) -> bool:
    """For storage: blank or 'prefer not to say' -> None. 'No' is preserved as a real answer."""
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULL_FOR_STORAGE)


def _answer_present(v) -> bool:
    """For context-mode triggers: an affirmative answer (not null, PNTS, or a No/NA)."""
    if v is None:
        return False
    return str(v).strip().lower() not in (_NULL_FOR_STORAGE | _NEGATIVE)


def _age_from_dob(dob, on_date) -> float | None:
    """Decimal-year chronological age from DOB and intake date (ISO yyyy-mm-dd)."""
    try:
        d0 = datetime.fromisoformat(dob)
        d1 = datetime.fromisoformat(on_date)
        return round((d1 - d0).days / 365.25, 1)
    except (ValueError, TypeError):
        return None


def parse_questionnaire(answers: dict, intake_date: str) -> dict:
    """Map a digital questionnaire submission (keyed by the engine-use field names
    the questionnaire defines) to the canonical covariate record. Missing or
    'prefer not to say' answers become None (engine handles them conservatively).
    Derives chronological_age, smoking_bin, and the Stage 7 context-mode list."""
    cov = {f: (None if _is_null_answer(answers.get(f)) else answers.get(f))
           for f in COVARIATE_FIELDS}
    cov["chronological_age"] = _age_from_dob(answers.get("date_of_birth"), intake_date)
    ss = (answers.get("smoking_status") or "").lower()
    cov["smoking_bin"] = ("current" if "current" in ss else
                          "former" if "former" in ss else
                          "never" if "never" in ss else None)
    modes = []
    for field, mode in _CONTEXT_MODE_TRIGGERS:
        if _answer_present(answers.get(field)) and mode not in modes:
            modes.append(mode)
    cov["stage7_context_modes"] = modes
    return cov


def _looks_like_cleartext_pii(patient_id) -> bool:
    """SOP §12: patient_id must be hashed before the engine. Cleartext (spaces,
    email, or too-short/non-token) -> quarantine."""
    if not isinstance(patient_id, str) or not patient_id:
        return True
    if " " in patient_id or "@" in patient_id:
        return True
    token = patient_id.replace("_", "").replace("-", "")
    return len(token) < 16 or not token.isalnum()


def _write_manifest(manifest_dir, record) -> None:
    """Write the immutable per-sample patient_manifest.json (one per sample_run_id)."""
    if not manifest_dir:
        return
    os.makedirs(manifest_dir, exist_ok=True)
    path = os.path.join(manifest_dir, f"patient_manifest_{record.get('sample_run_id')}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)


def step_0_2_manifest_creation(step_0_1_result: dict,
                               questionnaire_answers: dict | None,
                               manifest_dir: str | None = None,
                               intake_date: str | None = None) -> dict:
    """SOP §12. Build the canonical immutable patient_manifest.json from the staged
    Step 0.1 record + parsed questionnaire. Validate, PII-check, write, gate to 0.3."""
    flags = list(step_0_1_result.get("flags", []))
    intake_date = intake_date or (step_0_1_result.get("intake_timestamp", "") or "")[:10] or None
    qa = questionnaire_answers or {}
    record = {
        "sample_run_id": step_0_1_result.get("sample_run_id"),
        "sentrix_id": step_0_1_result.get("sentrix_id"),
        "array_type": step_0_1_result.get("array_type_declared"),
        "substrate": step_0_1_result.get("substrate"),
        "patient_id": qa.get("patient_id") or step_0_1_result.get("patient_id"),
        "declared_sex": step_0_1_result.get("declared_sex"),
        "declared_chronological_age": step_0_1_result.get("declared_chronological_age"),
        "intake_timestamp": step_0_1_result.get("intake_timestamp"),
        "covariates": parse_questionnaire(qa, intake_date) if qa else {},
        "flags": flags,
        "status": None,
        "advance": False,
    }
    if _looks_like_cleartext_pii(record["patient_id"]):
        flags.append("CLEARTEXT_PII:patient_id_not_hashed")
        record["status"] = "QUARANTINE_MANIFEST_INVALID"
        _write_manifest(manifest_dir, record)
        return record
    core = ("sample_run_id", "sentrix_id", "array_type", "substrate", "patient_id",
            "declared_sex", "declared_chronological_age", "intake_timestamp")
    missing = [k for k in core if record.get(k) in (None, "")]
    if missing:
        flags.append("MANIFEST_INVALID:" + ",".join(missing))
        record["status"] = "QUARANTINE_MANIFEST_INVALID"
        _write_manifest(manifest_dir, record)
        return record
    record["status"] = "MANIFEST_COMPLETE"
    record["advance"] = True
    _write_manifest(manifest_dir, record)
    return record


# ============================================================================
# Step 0.3 (SOP §13) — IDAT integrity hash check
# SHA-256 both IDATs; compare against prior intake of the same Sentrix ID;
# register in the integrity log; stamp INTEGRITY_OK and gate to 0.4.
# ============================================================================

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prior_hashes_for_sentrix(integrity_log_path, sentrix_id) -> list:
    """(grn_sha, red_sha) from prior integrity-log rows for this Sentrix ID."""
    out = []
    if not integrity_log_path or not os.path.isfile(integrity_log_path):
        return out
    with open(integrity_log_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("sentrix_id") == sentrix_id:
                out.append((row.get("grn_sha256"), row.get("red_sha256")))
    return out


def _log_integrity(integrity_log_path, row) -> None:
    if not integrity_log_path:
        return
    os.makedirs(os.path.dirname(integrity_log_path), exist_ok=True)
    with open(integrity_log_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def step_0_3_integrity_hash(record: dict, grn_path: str, red_path: str,
                            integrity_log_path: str | None = None) -> dict:
    """SOP §13. SHA-256 both IDATs, compare to prior intakes of the same Sentrix ID,
    register in the integrity log, stamp integrity_status, gate to 0.4.

    Decision (per §13):
      - no prior record           -> INTEGRITY_OK, advance
      - identical hashes to prior  -> RE_TRANSMISSION_DETECTED, HOLD for operator
      - same Sentrix, diff hashes  -> LIKELY_RE_RUN_FRESH_ARRAY, advance (new run id)
    """
    flags = record.setdefault("flags", [])
    grn_sha = _sha256_file(grn_path)
    red_sha = _sha256_file(red_path)
    record["grn_sha256"] = grn_sha
    record["red_sha256"] = red_sha
    sentrix = record.get("sentrix_id")

    prior = _prior_hashes_for_sentrix(integrity_log_path, sentrix)
    prior_match = (grn_sha, red_sha) in prior

    if prior_match:
        flags.append("RE_TRANSMISSION_DETECTED")
        record["integrity_status"] = "RE_TRANSMISSION_DETECTED"
        record["advance"] = False
    else:
        if prior:
            flags.append("LIKELY_RE_RUN_FRESH_ARRAY")
        record["integrity_status"] = "INTEGRITY_OK"
        record["advance"] = True

    _log_integrity(integrity_log_path, {
        "sentrix_id": sentrix,
        "intake_timestamp": record.get("intake_timestamp"),
        "grn_sha256": grn_sha, "red_sha256": red_sha,
        "prior_hash_match": prior_match,
    })
    return record


# ============================================================================
# Step 0.4 (SOP §14) — Control probe validation
# Validation LOGIC + gates built here and fully testable. Control-probe INTENSITY
# EXTRACTION from the IDAT pair is deferred to the shared IDAT decoder (Stage 1,
# methylprep / illumina-idat-reader) rather than duplicated. When no control
# summary and no decoder are available, the step marks CTRL_QC DEFERRED and
# advances (honest gap marker) instead of fabricating a pass.
#
# Thresholds: BS conversion 0.95 is the SOP §14 explicit value. The hybridization
# and extension thresholds are provisional and must be confirmed against the
# Illumina control-probe specification.
# ============================================================================

BS_CONVERSION_MIN = 0.95               # SOP §14 explicit
HYB_HIGH_LOW_MIN_RATIO = 2.0           # provisional — confirm vs Illumina control spec
EXTENSION_BALANCE_RANGE = (0.2, 5.0)   # provisional meth/unmeth ratio sanity band


def extract_control_probes(grn_path, red_path, array_type):
    """Extract Illumina control-probe intensities grouped by control class from the
    IDAT pair. DEFERRED: backed by the shared IDAT decoder (Stage 1, methylprep).
    Raises NotImplementedError until that decoder exists so callers never get fake data."""
    raise NotImplementedError(
        "Control-probe extraction requires the shared IDAT decoder (Stage 1, methylprep). "
        "Supply a precomputed control_summary to validate_control_probes() in the meantime."
    )


def validate_control_probes(control_summary: dict) -> dict:
    """SOP §14 control-probe math + gates. Input: per-control-class intensity medians
    extracted from the IDAT pair. Returns the CTRL_QC verdict."""
    flags, metrics = [], {}

    bs1 = control_summary.get("bisulfite_conversion_I_median")
    bs2 = control_summary.get("bisulfite_conversion_II_median")
    if bs1 is not None and bs2 is not None and (bs1 + bs2) > 0:
        bs_eff = bs1 / (bs1 + bs2)
        metrics["bisulfite_conversion_efficiency"] = round(bs_eff, 4)
        if bs_eff < BS_CONVERSION_MIN:
            flags.append("BS_CONVERSION_LOW")

    hh = control_summary.get("hyb_high_median")
    hl = control_summary.get("hyb_low_median")
    if hh is not None and hl is not None:
        metrics["hyb_signal"] = round(hh - hl, 2)
        if not (hl > 0 and hh / hl >= HYB_HIGH_LOW_MIN_RATIO):
            flags.append("HYB_FAIL")

    em = control_summary.get("extension_meth_median")
    eu = control_summary.get("extension_unmeth_median")
    if em is not None and eu is not None and eu > 0:
        ratio = em / eu
        metrics["extension_ratio"] = round(ratio, 3)
        lo, hi = EXTENSION_BALANCE_RANGE
        if not (lo <= ratio <= hi):
            flags.append("EXT_FAIL")

    status = "PASS" if not flags else "FAIL_" + ",".join(flags)
    return {"ctrl_qc": status, "metrics": metrics, "flags": flags, "advance": not flags}


def step_0_4_control_probe_validation(record, grn_path, red_path,
                                      control_summary=None) -> dict:
    """SOP §14. Validate control probes, stamp CTRL_QC, gate to 0.5. If no control
    summary is supplied, attempt extraction (deferred to the Stage 1 decoder); when that
    is unavailable, mark CTRL_QC DEFERRED and advance rather than fabricate a pass."""
    flags = record.setdefault("flags", [])
    if control_summary is None:
        try:
            control_summary = extract_control_probes(grn_path, red_path, record.get("array_type"))
        except NotImplementedError:
            record["ctrl_qc"] = "DEFERRED_PENDING_STAGE1_DECODER"
            flags.append("CTRL_QC_DEFERRED:no_control_intensities")
            record["advance"] = True
            return record
    verdict = validate_control_probes(control_summary)
    record["ctrl_qc"] = verdict["ctrl_qc"]
    record["ctrl_qc_metrics"] = verdict["metrics"]
    flags.extend(verdict["flags"])
    record["advance"] = verdict["advance"]
    return record


if __name__ == "__main__":
    # Self-test: manifest-validation branches (no real IDATs needed).
    good = {
        "sentrix_id": "200123456789_R01C01", "array_type": "EPIC_v1",
        "patient_id": "hashed_abc", "intake_date": "2026-06-10",
        "substrate": "whole_blood", "declared_sex": "F",
        "declared_chronological_age": 54.3,
    }
    r = step_0_1_idat_arrival(good, "nope_Grn.idat", "nope_Red.idat")
    assert r["status"] == "QUARANTINE_MISSING_CHANNEL", r["status"]

    bad = dict(good); bad.pop("substrate")
    r2 = step_0_1_idat_arrival(bad, "x_Grn.idat", "x_Red.idat")
    assert r2["status"] == "QUARANTINE_INCOMPLETE_MANIFEST", r2["status"]
    assert any("substrate" in f for f in r2["flags"])

    print("Step 0.1 self-test: PASS (manifest + missing-channel gates)")

    # Step 0.2: build the canonical patient_manifest from a staged 0.1 record + questionnaire.
    staged = {
        "sample_run_id": "test-run-001", "sentrix_id": "200123456789_R01C01",
        "array_type_declared": "EPIC_v1", "substrate": "whole_blood",
        "declared_sex": "F", "declared_chronological_age": 54.3,
        "intake_timestamp": "2026-06-10T14:22:00+00:00", "flags": [], "patient_id": None,
    }
    answers = {
        "patient_id": "a1b2c3d4e5f6a7b8c9d0",  # hashed-looking token
        "date_of_birth": "1971-03-01", "sex_at_birth": "Female",
        "smoking_status": "Former smoker", "known_autoimmune_condition": "Hashimoto's thyroiditis",
        "current_pregnancy_with_trimester": "No", "hiv_status": "Prefer not to say",
    }
    rec = step_0_2_manifest_creation(staged, answers)
    assert rec["status"] == "MANIFEST_COMPLETE", rec["status"]
    assert rec["covariates"]["smoking_bin"] == "former"
    assert abs(rec["covariates"]["chronological_age"] - 55.3) < 0.2
    assert "TRAJECTORY_WATCH" in rec["covariates"]["stage7_context_modes"]      # autoimmune
    assert "CONTEXT_PREGNANCY" not in rec["covariates"]["stage7_context_modes"]  # answered No
    assert rec["covariates"]["hiv_status"] is None                              # PNTS -> null

    cleartext = step_0_2_manifest_creation(staged, {"patient_id": "Jane Doe", "date_of_birth": "1971-03-01"})
    assert cleartext["status"] == "QUARANTINE_MANIFEST_INVALID"
    assert any("CLEARTEXT_PII" in f for f in cleartext["flags"])

    print("Step 0.2 self-test: PASS (manifest build, covariate derivation, PII gate)")

    # Step 0.3: integrity hashing + re-transmission / re-run detection.
    import tempfile
    _d = tempfile.mkdtemp()
    _g = os.path.join(_d, "g.idat"); _r = os.path.join(_d, "r.idat")
    open(_g, "wb").write(b"GREEN-BYTES"); open(_r, "wb").write(b"RED-BYTES")
    _log = os.path.join(_d, "integrity.log")
    base = {"sentrix_id": "200123456789_R01C01", "intake_timestamp": "2026-06-10T14:22:00+00:00", "flags": []}
    r3a = step_0_3_integrity_hash(dict(base), _g, _r, _log)
    assert r3a["integrity_status"] == "INTEGRITY_OK" and r3a["advance"], r3a
    assert len(r3a["grn_sha256"]) == 64
    # identical bytes re-arrive -> re-transmission, HOLD.
    r3b = step_0_3_integrity_hash(dict(base), _g, _r, _log)
    assert r3b["integrity_status"] == "RE_TRANSMISSION_DETECTED" and not r3b["advance"], r3b
    # same Sentrix, different bytes -> fresh-array re-run, advance.
    open(_g, "wb").write(b"GREEN-BYTES-RERUN")
    r3c = step_0_3_integrity_hash(dict(base), _g, _r, _log)
    assert r3c["integrity_status"] == "INTEGRITY_OK" and r3c["advance"]
    assert any("LIKELY_RE_RUN_FRESH_ARRAY" in f for f in r3c["flags"]), r3c

    print("Step 0.3 self-test: PASS (sha256 + re-transmission + fresh-array re-run gates)")

    # Step 0.4: control-probe validation math + gates (synthetic control summaries).
    passing = {"bisulfite_conversion_I_median": 9000, "bisulfite_conversion_II_median": 300,
               "hyb_high_median": 20000, "hyb_low_median": 4000,
               "extension_meth_median": 8000, "extension_unmeth_median": 7500}
    v_pass = validate_control_probes(passing)
    assert v_pass["ctrl_qc"] == "PASS" and v_pass["advance"], v_pass
    assert v_pass["metrics"]["bisulfite_conversion_efficiency"] >= 0.95

    low_bs = dict(passing, bisulfite_conversion_I_median=850, bisulfite_conversion_II_median=300)
    v_bs = validate_control_probes(low_bs)
    assert "BS_CONVERSION_LOW" in v_bs["flags"] and not v_bs["advance"], v_bs

    bad_hyb = dict(passing, hyb_high_median=4200, hyb_low_median=4000)
    assert "HYB_FAIL" in validate_control_probes(bad_hyb)["flags"]

    # deferred path (no summary, no decoder) -> advances with CTRL_QC_DEFERRED, no fake pass.
    r4 = step_0_4_control_probe_validation({"array_type": "EPIC_v1", "flags": []}, "g", "r")
    assert r4["ctrl_qc"] == "DEFERRED_PENDING_STAGE1_DECODER" and r4["advance"]

    print("Step 0.4 self-test: PASS (BS/HYB/EXT gates + deferred-extraction marker)")
    print("  NOTE: IDAT size/array-type/duplicate + control-probe extraction need real IDATs (Stage 1 decoder).")
