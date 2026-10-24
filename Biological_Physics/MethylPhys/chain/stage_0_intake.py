"""Stage 0 - sample intake for an IDAT pair (chain v4). Every step and gate is explained in the Chain Decoder, manual/OM_v4_decoder.md,
section 2. The steps:
  Step 0.1 (SOP §11) — IDAT file arrival on server.
  Step 0.2 (SOP §12) — Sample manifest creation (patient_manifest.json + covariates).
  Step 0.3 (SOP §13) — IDAT integrity hash check (SHA-256 + re-transmission/re-run detection).
  Step 0.4 (SOP §14) — Control probe validation (BS conversion / hybridization / extension gates).
  Step 0.5 (SOP §15) — Detection p-value QC per probe (detected-fraction gate).
  Step 0.6 (SOP §16) — Bead count QC (>= 3 beads/probe, warn-only gate).
  Step 0.7 (SOP §17) — Sample-level call rate (detection AND bead; 0.98 / 0.93 lines from intake_thresholds_v1.json).
  Step 0.7b           — HM450 >= 80% reference-CpG coverage gate + platform tag (VAL-091).
  Step 0.8 (SOP §18) — Sex check vs metadata (the minfi getSex rule; mismatch -> quarantine).
  Step 0.9 (SOP §19) — Stage 0 decision gate (PROCEED / PROCEED_WITH_PENALTY / QUARANTINE).

Design choices flagged for review (SOP marked these "TBD per orchestrator design"):
  - Handler lives here in the chain runtime, NOT in MPHYS_WEB_v13.py (frozen demo).
  - All paths (staging dir, intake log) are passed in by the caller, not hardcoded.
  - Manifest is JSON with the seven SOP-required fields.
  - Array-type verification reads only nSNPsRead from the IDAT header; full IDAT
    decoding is Stage 1's responsibility and is not duplicated here.

The intensities come from the QC hand-off (stage_0_1_qc_handoff.py), the same decoder Stage 1 uses; the arithmetic of each gate is here.
"""
from __future__ import annotations

import json
import gzip
import hashlib
import os
import struct
import uuid
from datetime import datetime, timezone, timedelta

# 2026-09-27 PROC-INTAKE-01: thresholds are runtime constants (Runtime Matrices/Intake/intake_thresholds_v1.json), not
# numbers in code. The QUARANTINE line is the author's 0.93 - see the file's _meta for the 48-array measurement behind it.
def _intake_thresholds():
    import json as _j, os as _o
    here=_o.path.dirname(_o.path.abspath(__file__))
    for root in (here, _o.path.join(here,"Runtime Matrices","Intake")):
        for dp,_,fs in _o.walk(root):
            if "RETIRED" in dp: continue
            if "intake_thresholds_v2.json" in fs: return _j.load(open(_o.path.join(dp,"intake_thresholds_v2.json")))   # v2 2026-10-21 (PROC-STAGE0-04); v1 kept as the record
    raise FileNotFoundError("intake_thresholds_v2.json (Runtime Matrices/Intake) - the intake gate has no thresholds without it")
_TH=_intake_thresholds()

# --- SOP §11 spec constants -------------------------------------------------

REQUIRED_MANIFEST_FIELDS = (
    "sentrix_id",
    "array_type",
    "patient_id",
    "intake_date",
    "substrate",
)
# Optional (author decision F, 2026-10-04, DEV-INTAKE-02): recorded when given, never required. No v3 stage reads age; sex is checked
# against the array only when it is declared.
OPTIONAL_MANIFEST_FIELDS = ("declared_sex", "declared_chronological_age")

# Specimens intake accepts (author decision L, 2026-10-04, DEV-INTAKE-02): the specimens chain v3 has a reference for. Every other
# specimen is refused at intake with a report and no reading, until it has its own reference.
ACCEPTED_SPECIMENS = {
    "whole blood": "whole blood", "blood": "whole blood", "peripheral blood": "whole blood", "venous blood": "whole blood",
    "isolated neutrophils": "isolated neutrophils", "sorted neutrophils": "isolated neutrophils",
    "purified neutrophils": "isolated neutrophils", "neutrophils": "isolated neutrophils",
    "constructed dna mixture": "constructed DNA mixture",   # test material: DNA of purified blood cells mixed in known amounts; read as whole blood
    "buffy coat": "buffy coat",   # 2026-10-18: the white-cell layer of whole blood; read on the whole-blood path, labeled as its own specimen (not commissioned)
}
REFUSED_SPECIMEN_REASONS = (
    (("pbmc", "mononuclear"), "PBMC (peripheral blood mononuclear cells) has no reference in the chain: the neutrophil reading needs whole blood or isolated neutrophils"),
    (("sorted", "purified", "isolated", "gmdsc", "mdsc", "cd4", "cd8", "t cells", "b cells", "nk cells", "monocytes", "basophils", "eosinophils"),
     "a sorted cell fraction other than neutrophils has no reference in the chain yet; each cell type is added only after the three new-cell tests (SOP section 2b)"),
    (("cell line", "cultured", "culture"), "a cell line has no reference in the chain"),
    (("unspecified", "unknown", "not given", ""), "the specimen is not stated: intake reads only a stated specimen it has a reference for"),
)


def normalise_specimen(specimen):
    return " ".join(str(specimen or "").strip().lower().replace("_", " ").split())


ATLAS_CELL_WORDS = ("isolated ", "sorted ", "purified ")
_atlas_cells = []
def atlas_cells():
    """The cell names of IAM-Atlas v2.1 (its best-value columns), read once."""
    if not _atlas_cells:
        import pyarrow.parquet as pq
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlas", "v2.1")
        for part in (1, 2, 3): _atlas_cells.extend(n for n in pq.read_schema(os.path.join(d, f"mu_part{part}.parquet")).names if n != "cpg")
    return _atlas_cells

def atlas_cell_of(specimen):
    """2026-10-22 (author: read and flag): an isolated, sorted or purified cell of any IAM-Atlas v2.1 type other than neutrophils is accepted
    and read by Met-A against the atlas (development, not commissioned). Returns that atlas cell's name, else None."""
    s = normalise_specimen(specimen)
    for w in ATLAS_CELL_WORDS:
        c = {x.lower(): x for x in atlas_cells()}.get(s[len(w):])
        if s.startswith(w) and c and c != "neutrophils": return c
    return None

PLASMA_WORDS = ("plasma", "cfdna", "cell-free dna", "cell free dna", "plasma cfdna", "plasma cell-free dna", "serum cfdna")
def is_plasma(specimen):
    """True for plasma cell-free DNA (2026-10-22, to-do steps 39-40): accepted at intake and read on IAM-Atlas v2.1's full panel; development."""
    return normalise_specimen(specimen) in PLASMA_WORDS

TISSUE_WORDS = ("tissue", "solid tissue", "tumor", "tumor tissue", "biopsy", "unknown", "unknown specimen", "not stated")
def is_tissue(specimen):
    """2026-10-23 (author: read it and flag it; blind runs: 'we dont even know what the chips are from'): a solid tissue, a tumor, or a specimen
    not named. Read on IAM-Atlas v2.1's full panel, flagged, as plasma is: the IAM-Deconvolver says what the sample is made of, and Met-A reads
    each cell it finds against the atlas in the sample's own shares. The commissioned readers read blood and are not applied."""
    s = normalise_specimen(specimen)
    # 2026-10-24 (author: read and flag): bone marrow, marrow aspirate and blood-and-marrow mixes are read on this route, the full atlas panel
    # with the marrow progenitors kept in the mix; it was refused at intake before ("no reference"), against the read-and-flag rule.
    if "marrow" in s: return True
    return s in TISSUE_WORDS or s.endswith(" tissue") or s.endswith(" tumor") or s.endswith(" biopsy")

def specimen_refusal(specimen):
    """None when intake accepts the specimen (ACCEPTED_SPECIMENS, or an isolated IAM-Atlas v2.1 cell); otherwise the plain refusal text naming the specimen (decision L)."""
    s = normalise_specimen(specimen)
    if s in ACCEPTED_SPECIMENS or atlas_cell_of(specimen) or is_plasma(specimen) or is_tissue(specimen): return None
    why = next((r for keys, r in REFUSED_SPECIMEN_REASONS if any(k and k in s for k in keys) or (s == "" and "" in keys)), None)
    why = why or f"'{specimen}' is not a blood specimen the chain has a reference for (tissue, placenta and other specimens each need their own reference)"
    return (f"specimen '{specimen}' refused at intake: {why}. Accepted: whole blood, buffy coat, isolated / sorted / purified neutrophils, constructed DNA mixtures of blood cells (test material), and "
            f"isolated / sorted / purified cells of any IAM-Atlas v2.1 type (read by Met-A against the atlas, not commissioned). "
            f"Nothing is read; this specimen can be read once it has its own reference.")

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
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rb") as f:
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
        "patient_id": manifest_entry.get("patient_id"),
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
# 2026-10-20 (Chain Decoder finding F7): the stage names below (Stage 7 stratification and context modes, Stage 8, Stage 9) are the
# questionnaire's tags for the class-floor engine retired 2026-10-03. No chain v4 stage reads these answers: they are recorded with the run
# (patient_manifest.json) and change no reading. Only the field names are used, as the set of answers to keep.
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

# Covariate (when answered and not a No/null) -> Stage 7 interpretation mode (recorded only: the retired engine's Stage 7 is not in chain v4).
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
            "intake_timestamp")   # declared sex and age are optional (decision F, 2026-10-04): recorded when given
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
# Thresholds (intake_thresholds_v2.json, calibrated 2026-10-21, PROC-STAGE0-04, on 337 healthy whole-blood arrays of three
# laboratories): bisulfite conversion at least 0.80 (the author's choice: the 2.5th percentile with each laboratory counted
# equally); hybridization spread at least 2.0 and extension balance 0.2 to 5.0 (no healthy array outside either).
# ============================================================================

BS_CONVERSION_MIN = float(_TH["bisulfite_conversion"]["min"])   # intake_thresholds_v2.json: 0.80, calibrated 2026-10-21 on 337 healthy arrays of 3 laboratories
BS_THRESHOLD_CALIBRATED = str(_TH["bisulfite_conversion"].get("status", "")).startswith("CALIBRATED")   # True since 2026-10-21 (PROC-STAGE0-04)
HYB_HIGH_LOW_MIN_RATIO = float(_TH["hybridization"]["min_spread"])   # 2.0, calibrated 2026-10-21: the spread between the high and low hybridization controls
EXTENSION_BALANCE_RANGE = tuple(_TH["extension"]["range"])   # (0.2, 5.0), calibrated 2026-10-21: green (C, G) over red (A, T); the record keys say meth/unmeth, which they are not


def extract_control_probes(grn_path, red_path, array_type):
    """Extract Illumina control-probe intensities grouped by control class from the IDAT pair.

    Backed by the same decoder Stage 1 uses (methylprep), reading the control addresses from the array's own
    manifest: stage_0_1_qc_handoff.decode_qc_inputs. Per-sample only - the patient's own control probes."""
    from stage_0_1_qc_handoff import decode_qc_inputs
    return decode_qc_inputs(grn_path, red_path, array_type)["control_summary"]


def validate_control_probes(control_summary: dict) -> dict:
    """SOP §14 control-probe math + gates. Input: per-control-class intensity medians
    extracted from the IDAT pair. Returns the CTRL_QC verdict."""
    flags, metrics = [], {}

    bs1 = control_summary.get("bisulfite_conversion_I_median")
    bs2 = control_summary.get("bisulfite_conversion_II_median")
    if bs1 is not None and bs2 is not None and (bs1 + bs2) > 0:
        bs_eff = bs1 / (bs1 + bs2)
        metrics["bisulfite_conversion_efficiency"] = round(bs_eff, 4)
        if control_summary.get("bisulfite_pairs"): metrics["bisulfite_pairs"] = control_summary["bisulfite_pairs"]   # each pair's channel (2026-10-20, F8)
        if bs_eff < BS_CONVERSION_MIN:
            flags.append("BS_CONVERSION_LOW")

    hh = control_summary.get("hyb_high_median")
    hl = control_summary.get("hyb_low_median")
    if hh is not None and hl is not None:
        metrics["hyb_signal"] = round(hh - hl, 2)
        metrics["hyb_high_median"] = round(float(hh), 1); metrics["hyb_low_median"] = round(float(hl), 1)
        metrics["hyb_high_low_ratio"] = round(float(hh) / float(hl), 3) if hl > 0 else None
        # 2026-10-20 (PROC-STAGE0-04 exploration, 24 arrays of GSE184269): on two whole slides High and Low trade values exactly (21,304 <-> 20,040;
        # 6,514 <-> 6,845) while Medium is unchanged (12,822 vs 12,834). A failed hybridization squeezes all three together; a swap keeps the
        # spread. The gate tests the spread (larger over smaller, with Medium between them); a swapped order is recorded, not a stop.
        hm = control_summary.get("hyb_medium_median")
        lo_, hi_ = min(float(hh), float(hl)), max(float(hh), float(hl))
        spread = hi_ / lo_ if lo_ > 0 else None
        metrics["hyb_spread"] = round(spread, 3) if spread else None
        if hm is not None: metrics["hyb_medium_median"] = round(float(hm), 1)
        between = hm is None or lo_ <= float(hm) <= hi_
        if not (spread and spread >= HYB_HIGH_LOW_MIN_RATIO and between):
            flags.append("HYB_FAIL")
        elif hh < hl:
            metrics["hyb_order"] = "swapped: the high control reads where the low one should and the reverse, with the medium control between them and the spread intact (a label or reagent swap, not a failed hybridization)"

    em = control_summary.get("extension_meth_median")
    eu = control_summary.get("extension_unmeth_median")
    if em is not None and eu is not None and eu > 0:
        ratio = em / eu
        metrics["extension_ratio"] = round(ratio, 3)
        metrics["extension_meth_median"] = round(float(em), 1); metrics["extension_unmeth_median"] = round(float(eu), 1)
        lo, hi = EXTENSION_BALANCE_RANGE
        if not (lo <= ratio <= hi):
            flags.append("EXT_FAIL")

    # BS_CONVERSION_MIN has never been measured against healthy arrays on this construction (PROC-STAGE0-04
    # will set it from the 732-array distribution). Until it is, a sample whose ONLY failing gate is the
    # bisulfite threshold is reported as uncalibrated rather than failed: the number is printed, and the
    # decision gate treats it as deferred. No sample is passed that a calibrated gate would fail.
    if flags == ["BS_CONVERSION_LOW"] and not BS_THRESHOLD_CALIBRATED:
        return {"ctrl_qc": "PROVISIONAL_BS_THRESHOLD_UNCALIBRATED", "metrics": metrics,
                "flags": ["BS_THRESHOLD_UNCALIBRATED"], "advance": True}
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
        except (NotImplementedError, ImportError):   # no decoder installed: deferred, never a fabricated pass
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


# ============================================================================
# Step 0.5 (SOP §15) — Detection p-value QC per probe
# Per-probe detection_p = 1 - Φ((I - μ_bg)/σ_bg) vs the negative-control background;
# probes with detection_p > 0.01 are undetected. Gate on the detected fraction.
# Math + summary + gates built and tested here; per-probe intensity + negative-control
# extraction comes from the QC hand-off (stage_0_1_qc_handoff.decode_qc_inputs), the same decoder Stage 1 uses.
#
# NOTE: this mirrors the SOP's stated formula. The exact minfi detectionP (M+U total
# vs median/MAD negative-control background) is the production target and is flagged
# to be reconciled to minfi for cross-platform exactness.
# ============================================================================

DETECTION_P_THRESHOLD = 0.01
DETECTION_PASS_FRACTION = float(_TH["detection"]["pass_fraction"])
DETECTION_BORDERLINE_FRACTION = float(_TH["detection"]["borderline_fraction"])


def _normal_sf(z):
    """Survival function 1 - Φ(z), vectorized. scipy if available, else erfc fallback."""
    try:
        from scipy.stats import norm
        return norm.sf(z)
    except Exception:
        import numpy as _np
        import math as _math
        return 0.5 * _np.vectorize(_math.erfc)(_np.asarray(z, dtype=float) / _math.sqrt(2.0))


def compute_detection_p(probe_intensities, mu_bg, sigma_bg):
    """detection_p = 1 - Φ((I - μ_bg)/σ_bg) per probe (SOP §15 stated formula)."""
    import numpy as np
    I = np.asarray(probe_intensities, dtype=float)
    sigma_bg = sigma_bg if sigma_bg > 0 else 1e-9
    return _normal_sf((I - mu_bg) / sigma_bg)


def validate_detection_p(detection_p_array) -> dict:
    """Gate on the detected fraction (intake_thresholds_v1.json detection): > pass_fraction PASS, >= borderline_fraction BORDERLINE, else FAIL."""
    import numpy as np
    dp = np.asarray(detection_p_array, dtype=float)
    n = dp.size
    pct = float(np.mean(dp <= DETECTION_P_THRESHOLD)) if n else 0.0
    if pct > DETECTION_PASS_FRACTION:
        status, advance = "PASS", True
    elif pct >= DETECTION_BORDERLINE_FRACTION:
        status, advance = "DETECTION_BORDERLINE", True
    else:
        status, advance = "FAIL_LOW_DETECTION", False
    return {"detection_qc": status,
            "pct_probes_detected_p_le_01": round(pct, 5),
            "n_probes": int(n), "advance": advance}


def step_0_5_detection_pvalue_qc(record, probe_intensities=None,
                                 neg_control_stats=None) -> dict:
    """SOP §15. Compute per-probe detection p, summarize, gate to 0.6. If intensities
    or negative-control stats are absent (extraction deferred to the Stage 1 decoder),
    mark DETECTION_QC deferred and advance rather than fabricate a pass."""
    flags = record.setdefault("flags", [])
    if probe_intensities is None or neg_control_stats is None:
        record["detection_qc"] = "DEFERRED_PENDING_STAGE1_DECODER"
        flags.append("DETECTION_QC_DEFERRED:no_probe_intensities")
        record["advance"] = False   # 2026-09-27: deferred never advances
        return record
    dp = compute_detection_p(probe_intensities,
                             neg_control_stats["mu_bg"], neg_control_stats["sigma_bg"])
    v = validate_detection_p(dp)
    record["detection_qc"] = v["detection_qc"]
    record["pct_probes_detected_p_le_01"] = v["pct_probes_detected_p_le_01"]
    if v["detection_qc"] == "DETECTION_BORDERLINE":
        flags.append("DETECTION_BORDERLINE")
    elif v["detection_qc"] == "FAIL_LOW_DETECTION":
        flags.append("DETECTION_QC_FAIL")
    record["advance"] = v["advance"]
    return record


# ============================================================================
# Step 0.6 (SOP §16) — Bead count QC. Per-probe bead count >= 3 (Illumina default).
# Summary fraction; warn (not hard-fail) below 99.5% per §16. Extraction deferred.
# ============================================================================
BEAD_COUNT_MIN = 3
BEAD_PASS_FRACTION = float(_TH["bead"]["pass_fraction"])   # intake_thresholds_v1.json bead.pass_fraction


def validate_bead_count(bead_counts, min_beads=BEAD_COUNT_MIN) -> dict:
    import numpy as np
    bc = np.asarray(bead_counts)
    n = bc.size
    pct = float(np.mean(bc >= min_beads)) if n else 0.0
    status = "PASS" if pct >= BEAD_PASS_FRACTION else "WARN_LOW_BEAD_COUNT"
    return {"bead_qc": status, "pct_probes_bead_count_ge_3": round(pct, 5),
            "n_probes": int(n), "advance": True}  # §16: warn only, no hard fail


def step_0_6_bead_count_qc(record, bead_counts=None) -> dict:
    flags = record.setdefault("flags", [])
    if bead_counts is None:
        record["bead_qc"] = "DEFERRED_PENDING_STAGE1_DECODER"
        flags.append("BEAD_QC_DEFERRED:no_bead_counts")
        record["advance"] = True
        return record
    v = validate_bead_count(bead_counts)
    record["bead_qc"] = v["bead_qc"]
    record["pct_probes_bead_count_ge_3"] = v["pct_probes_bead_count_ge_3"]
    if v["bead_qc"] != "PASS":
        flags.append("LOW_BEAD_COUNT")
    record["advance"] = True
    return record


# ============================================================================
# Step 0.7 (SOP §17) — Sample-level call rate = fraction passing BOTH detection
# and bead thresholds. >=0.98 proceed; 0.93-0.98 borderline (penalty); <0.93 QUARANTINE (author, 2026-09-27).
# ============================================================================
CALL_RATE_PASS = float(_TH["call_rate"]["proceed_at_or_above"])
CALL_RATE_BORDERLINE = float(_TH["call_rate"]["quarantine_below"])


def validate_call_rate(n_passing_both: int, n_total: int) -> dict:
    cr = (n_passing_both / n_total) if n_total else 0.0
    if cr >= CALL_RATE_PASS:
        status, advance = "PASS", True
    elif cr >= CALL_RATE_BORDERLINE:
        status, advance = "CALL_RATE_BORDERLINE", True
    else:
        status, advance = "CALL_RATE_FAIL", False
    return {"call_rate": round(cr, 5), "call_rate_status": status, "advance": advance}


def step_0_7_call_rate(record, detection_pass_mask=None, bead_pass_mask=None) -> dict:
    flags = record.setdefault("flags", [])
    if detection_pass_mask is None or bead_pass_mask is None:
        record["call_rate"] = None
        record["call_rate_status"] = "DEFERRED_PENDING_STAGE1_DECODER"
        flags.append("CALL_RATE_DEFERRED:no_probe_masks")
        # 2026-09-27: a deferred check is never a pass (FINDING_GSE125105_LOW_SIGNAL.md - Munich's 0.875 call rate walked
        # through on this line for a week). An IDAT input that reaches here without masks is refused.
        record["advance"] = False
        return record
    import numpy as np
    dpm = np.asarray(detection_pass_mask, dtype=bool)
    bpm = np.asarray(bead_pass_mask, dtype=bool)
    v = validate_call_rate(int(np.sum(dpm & bpm)), int(dpm.size))
    record["call_rate"] = v["call_rate"]
    record["call_rate_status"] = v["call_rate_status"]
    if v["call_rate_status"] == "CALL_RATE_BORDERLINE":
        flags.append("CALL_RATE_BORDERLINE")
    elif v["call_rate_status"] == "CALL_RATE_FAIL":
        flags.append("CALL_RATE_FAIL")
    record["advance"] = v["advance"]
    return record


# ============================================================================
# Step 0.7b — HM450 >=80% reference-CpG coverage gate + platform tagging (VAL-091).
# Platform tag always stamped; on 450K, HARD-fail below 80% reference coverage.
# ============================================================================
HM450_COVERAGE_MIN = 0.80


def step_0_7b_platform_coverage(record, reference_cpg_coverage=None) -> dict:
    flags = record.setdefault("flags", [])
    array_type = record.get("array_type") or ""
    platform = "450K" if "450" in array_type else "EPIC"
    record["platform_tag"] = platform
    if platform != "450K":
        record["hm450_coverage_gate"] = "NA_EPIC"
        if reference_cpg_coverage is not None:   # 2026-10-20 (Decoder finding F9): recorded, not gated; Stage M withholds Met-A below 90 % of its sites
            record["epic_identity_site_coverage"] = round(reference_cpg_coverage, 4)
        record["advance"] = True
        return record
    if reference_cpg_coverage is None:
        record["hm450_coverage_gate"] = "DEFERRED_PENDING_STAGE1_DECODER"
        flags.append("HM450_COVERAGE_DEFERRED:no_beta_coverage")
        record["advance"] = True
        return record
    record["hm450_reference_coverage"] = round(reference_cpg_coverage, 4)
    if reference_cpg_coverage < HM450_COVERAGE_MIN:
        flags.append("HM450_COVERAGE_FAIL")
        record["hm450_coverage_gate"] = "FAIL"
        record["advance"] = False
    else:
        record["hm450_coverage_gate"] = "PASS"
        record["advance"] = True
    return record


# ============================================================================
# Step 0.8 (SOP §18) — Sex check vs metadata. minfi getSex: predicted F if
# (log2_Y_median - log2_X_median) < -2 else M. Mismatch -> quarantine (never silently fixed).
# ============================================================================
SEX_GETSEX_CUTOFF = -2.0


def predict_sex(log2_x_median, log2_y_median, cutoff=SEX_GETSEX_CUTOFF) -> str:
    return "F" if (log2_y_median - log2_x_median) < cutoff else "M"


def validate_sex(log2_x_median, log2_y_median, declared_sex) -> dict:
    pred = predict_sex(log2_x_median, log2_y_median)
    declared = (declared_sex or "").strip().upper()[:1]
    if declared not in ("F", "M"):   # sex not declared (optional since 2026-10-04, decision F): the array's sex is recorded, nothing is compared
        return {"predicted_sex": pred, "sex_check": "NOT_DECLARED", "advance": True}
    match = pred == declared
    return {"predicted_sex": pred, "sex_check": "PASS" if match else "MISMATCH", "advance": match}


def step_0_8_sex_check(record, sex_intensities=None) -> dict:
    flags = record.setdefault("flags", [])
    if sex_intensities is None:
        record["sex_check"] = "DEFERRED_PENDING_STAGE1_DECODER"
        flags.append("SEX_CHECK_DEFERRED:no_chrXY_intensities")
        record["advance"] = True
        return record
    v = validate_sex(sex_intensities["log2_x_median"], sex_intensities["log2_y_median"],
                     record.get("declared_sex"))
    record["predicted_sex"] = v["predicted_sex"]
    record["sex_check"] = v["sex_check"]
    if v["sex_check"] == "MISMATCH":
        flags.append("SEX_MISMATCH")
    record["advance"] = v["advance"]
    return record


# ============================================================================
# Step 0.9 (SOP §19) — Stage 0 decision gate. Consolidate all QC into
# PROCEED / PROCEED_WITH_PENALTY / QUARANTINE. Hard FAIL -> quarantine; borderline/warn
# -> proceed with penalty; deferred QC (pending Stage 1 decoder) noted, non-blocking.
# ============================================================================

# ============================================================================
# Every decision Stage 0 makes, explained (author rule, 2026-10-20: a gate that blocks must say exactly what flagged it). For each check
# that stopped, penalized or deferred the array: what was measured on THIS array, the limit, where the limit comes from and whether it is
# provisional, what it means, and what the user can do. release_check_v4.py (V9) fails if any Stage 0 stop has no entry here.
# ============================================================================
_TH_SRC = "Runtime Matrices/Intake/intake_thresholds_v2.json (detection, bead and call rate: PROC-INTAKE-01, 48 arrays; control probes: PROC-STAGE0-04, 337 healthy arrays of 3 laboratories)"
def _fmt(x, nd=3):
    return "not measured" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else str(x))
def explain_stage0(record, hard_fail, borderline, deferred):
    out = []
    def add(check, outcome, measured, limit, source, provisional, means, todo):
        out.append(dict(check=check, outcome=outcome, measured=measured, limit=limit, limit_source=source, provisional=bool(provisional), means=means, what_to_do=todo))
    st = str(record.get("status") or ""); fl = record.get("flags") or []
    if st.startswith("QUARANTINE"):
        add("intake files and manifest (Steps 0.1-0.2)", "stopped", st + ("; flags: " + ", ".join(fl) if fl else ""), "a complete manifest, both IDAT channels, each IDAT at least 1 MB, the declared array type matching the IDAT header",
            "SOP sections 11-12 (fixed rules, not measured limits)", False, "the files or the manifest are incomplete or do not match; nothing about the biology was looked at",
            "check the flag named above: re-upload the missing or truncated IDAT, or correct the manifest field")
    if record.get("integrity_status") and record.get("integrity_status") != "INTEGRITY_OK":
        add("file integrity (Step 0.3)", "stopped", str(record.get("integrity_status")), "the same Sentrix ID with identical bytes not already taken in under this intake log (a re-transmission); a re-run with new bytes is read", "SOP section 13 (fixed rule)", False,
            "this exact pair of files was submitted before", "confirm the sample identifier; a true re-run of the same array is not a new specimen")
    m = record.get("ctrl_qc_metrics") or {}; cq = str(record.get("ctrl_qc") or "")
    if str(m.get("hyb_order") or "").startswith("swapped"):
        add("hybridization control order (Step 0.4)", "recorded, not used to stop",
            f"high-concentration control {_fmt(m.get('hyb_high_median'),0)}, medium {_fmt(m.get('hyb_medium_median'),0)}, low {_fmt(m.get('hyb_low_median'),0)}",
            "high above medium above low", "the Illumina hybridization control design (fixed rule)", False,
            "high and low have traded places while the medium control sits between them and the spread is intact: a label or reagent swap on this array, not a failed hybridization (seen on whole slides of GSE184269)",
            "nothing; if a whole slide shows it, note it with the laboratory")
    if cq.startswith(("FAIL", "PROVISIONAL")):
        hr = m.get("hyb_high_low_ratio"); er = m.get("extension_ratio"); bs = m.get("bisulfite_conversion_efficiency")
        if "HYB_FAIL" in cq:
            inv = hr is not None and hr < 1
            add("hybridization control probes (Step 0.4)", "stopped",
                f"high-concentration control {_fmt(m.get('hyb_high_median'),0)}, medium {_fmt(m.get('hyb_medium_median'),0)}, low-concentration control {_fmt(m.get('hyb_low_median'),0)}, spread {_fmt(m.get('hyb_spread'))}",
                f"the larger of high and low at least {HYB_HIGH_LOW_MIN_RATIO:g} times the smaller, with medium between them", "Runtime Matrices/Intake/intake_thresholds_v2.json (PROC-STAGE0-04: 337 healthy arrays of 3 laboratories; 0 of them stopped)", False,
                ("the high control reads BELOW the low one, the reverse of what the chemistry gives; on whole slides this is more often a control-address or labeling difference in the files than a failed hybridization, and the array's methylation values may still be sound"
                 if inv else "the hybridization controls are weaker than the array chemistry should give; the array may have hybridized poorly"),
                "healthy arrays read a spread of 2.69 or more (337 arrays, 3 laboratories); rerun the array, and if a whole slide is stopped this way report it; the calibrated values can still be read with run_sample.py --betas, marked as having bypassed this check")
        if "EXT_FAIL" in cq:
            lo, hi = EXTENSION_BALANCE_RANGE
            add("extension control probes (Step 0.4)", "stopped", f"green (C and G) {_fmt(m.get('extension_meth_median'),0)}, red (A and T) {_fmt(m.get('extension_unmeth_median'),0)}, ratio {_fmt(er)}",
                f"ratio between {lo:g} and {hi:g}", "Runtime Matrices/Intake/intake_thresholds_v2.json (PROC-STAGE0-04: 337 healthy arrays of 3 laboratories; 0 of them stopped)", False, "the single-base extension step is unbalanced between the two colors",
                "healthy arrays read 0.45 to 1.22 (337 arrays, 3 laboratories); rerun the array, and report it with its slide")
        if "BS_CONVERSION_LOW" in cq or cq.startswith("PROVISIONAL"):
            add("bisulfite conversion control probes (Step 0.4)", "printed, not used to stop" if cq.startswith("PROVISIONAL") or not BS_THRESHOLD_CALIBRATED else "stopped",
                f"conversion efficiency {_fmt(bs)}", f"at least {BS_CONVERSION_MIN:g}", _TH_SRC + ("" if BS_THRESHOLD_CALIBRATED else "; status PROVISIONAL"), not BS_THRESHOLD_CALIBRATED,
                "the bisulfite step converted too little of this array's DNA: unconverted DNA reads as methylated, so every value is pushed up; healthy arrays read a median of 0.93 to 0.97 by laboratory, and 1 of 337 healthy arrays read below 0.80",
                "convert and hybridize the sample again; the calibrated values can still be read with run_sample.py --betas, marked as having bypassed this check")
    dq = str(record.get("detection_qc") or "")
    if dq == "FAIL_LOW_DETECTION" or dq == "DETECTION_BORDERLINE" or (dq.startswith("DEFERRED") and "detection" in deferred):
        add("probe detection (Step 0.5)", {"FAIL_LOW_DETECTION": "stopped", "DETECTION_BORDERLINE": "read with a penalty flag"}.get(dq, "stopped (could not be measured)"),
            f"{_fmt(record.get('pct_probes_detected_p_le_01'))} of probes detected above background" if not dq.startswith("DEFERRED") else "detection could not be measured on this file",
            f"at least {DETECTION_PASS_FRACTION:g} to pass; below {DETECTION_BORDERLINE_FRACTION:g} stops", _TH_SRC, False,
            "too many probes read at background level, so the methylation values are unreliable", "re-scan or re-hybridize the array; a low-signal laboratory batch shows up as many arrays stopped together")
    crs = str(record.get("call_rate_status") or "")
    if crs in ("CALL_RATE_FAIL", "CALL_RATE_BORDERLINE") or (crs.startswith("DEFERRED") and "call_rate" in deferred):
        add("call rate (Step 0.7)", {"CALL_RATE_FAIL": "stopped", "CALL_RATE_BORDERLINE": "read with a penalty flag"}.get(crs, "stopped (could not be measured)"),
            f"call rate {_fmt(record.get('call_rate'))}", f"at least {CALL_RATE_PASS:g} to pass; below {CALL_RATE_BORDERLINE:g} stops", _TH_SRC + "; the 0.93 line was set by the author on 2026-09-27", False,
            "too few sites give a usable value", "re-scan or re-hybridize the array")
    if str(record.get("bead_qc") or "").startswith("WARN"):
        add("bead counts (Step 0.6)", "read with a penalty flag", f"{_fmt(record.get('pct_probes_bead_count_ge_3'))} of probes with at least {BEAD_COUNT_MIN} beads", f"at least {BEAD_PASS_FRACTION:g}", _TH_SRC, False,
            "some probes rest on few beads, so their values are noisier", "nothing required; the flag is printed on the report")
    if str(record.get("hm450_coverage_gate") or "") == "FAIL":
        add("450K reference coverage (Step 0.7b)", "stopped", f"coverage {_fmt(record.get('hm450_reference_coverage'))}", f"at least {HM450_COVERAGE_MIN:g} of the reference sites measured", "stage_0_intake.HM450_COVERAGE_MIN (fixed rule)", False,
            "the array lacks too many of the sites the references use", "the chain reads EPIC v1 arrays")
    if str(record.get("sex_check") or "") == "MISMATCH":
        add("sex check (Step 0.8)", "stopped", f"declared {record.get('declared_sex')}, the array's X and Y signals say {record.get('predicted_sex')}", f"declared and measured sex agree (getSex cutoff {SEX_GETSEX_CUTOFF:g})",
            "SOP section 18 (fixed rule)", False, "the array may belong to another person (a sample swap) or the manifest is wrong", "check the sample identity and the declared sex in the manifest")
    return out


def step_0_9_decision_gate(record, verdict_log_path=None) -> dict:
    flags = record.setdefault("flags", [])
    hard_fail, borderline, deferred = [], [], []

    # Step 0.1 / 0.2 quarantines are hard failures. (Added 2026-09-19, PROC-STAGE0-01: before this
    # guard a QUARANTINE_INCOMPLETE_MANIFEST or QUARANTINE_MISSING_CHANNEL sample fell through to PROCEED.)
    if str(record.get("status") or "").startswith("QUARANTINE"):
        hard_fail.append("intake_" + record["status"].lower())

    if record.get("integrity_status") and record.get("integrity_status") != "INTEGRITY_OK":
        hard_fail.append("integrity")

    cq = record.get("ctrl_qc", "") or ""
    if cq.startswith("PROVISIONAL"):
        deferred.append("ctrl_qc_threshold_uncalibrated")
    elif cq.startswith("FAIL"):
        hard_fail.append("ctrl_qc")
    elif cq.startswith("DEFERRED"):
        deferred.append("ctrl_qc")

    dq = record.get("detection_qc", "") or ""
    if dq == "FAIL_LOW_DETECTION":
        hard_fail.append("detection")
    elif dq == "DETECTION_BORDERLINE":
        borderline.append("detection")
    elif dq.startswith("DEFERRED"):
        deferred.append("detection")

    bq = record.get("bead_qc", "") or ""
    if bq.startswith("WARN"):
        borderline.append("bead")
    elif bq.startswith("DEFERRED"):
        deferred.append("bead")

    crs = record.get("call_rate_status", "") or ""
    if crs == "CALL_RATE_FAIL":
        hard_fail.append("call_rate")
    elif crs == "CALL_RATE_BORDERLINE":
        borderline.append("call_rate")
    elif crs.startswith("DEFERRED"):
        deferred.append("call_rate")

    hg = record.get("hm450_coverage_gate", "") or ""
    if hg == "FAIL":
        hard_fail.append("hm450_coverage")
    elif hg.startswith("DEFERRED"):
        deferred.append("hm450_coverage")

    sc = record.get("sex_check", "") or ""
    if sc == "MISMATCH":
        hard_fail.append("sex")
    elif sc.startswith("DEFERRED"):
        deferred.append("sex")

    # 2026-09-27 PROC-INTAKE-01 B5: a DEFERRED detection or call-rate check is not a pass. The gate had recorded 'deferred'
    # and advanced for months, which is how a low-signal laboratory was read (FINDING_GSE125105_LOW_SIGNAL.md). Now a
    # deferred detection or call rate quarantines the specimen (QUARANTINE_INTAKE_DEFERRED). Bead count (not extracted by
    # Stage 1) remains recorded-deferred, as the SOP states for it. The bisulfite threshold is calibrated since 2026-10-21 (PROC-STAGE0-04).
    if "detection" in deferred or "call_rate" in deferred:
        hard_fail.append("intake_deferred:" + "+".join(x for x in ("detection", "call_rate") if x in deferred))
    if hard_fail:
        verdict, advance = "QUARANTINE", False
    elif borderline:
        verdict, advance = "PROCEED_WITH_PENALTY", True
    else:
        verdict, advance = "PROCEED", True

    record["stage0_verdict"] = verdict
    record["stage0_explained"] = explain_stage0(record, hard_fail, borderline, deferred)
    record["stage0_hard_fail"] = hard_fail
    record["stage0_borderline"] = borderline
    record["stage0_deferred_qc"] = deferred
    record["advance"] = advance
    if verdict_log_path:
        _log_integrity(verdict_log_path, {
            "sample_run_id": record.get("sample_run_id"), "sentrix_id": record.get("sentrix_id"),
            "stage0_verdict": verdict, "hard_fail": hard_fail,
            "borderline": borderline, "deferred": deferred,
        })
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
    # bisulfite alone below the line: PROVISIONAL while BS_THRESHOLD_CALIBRATED is False (recorded, not refused)
    assert (v_bs["ctrl_qc"] == "PROVISIONAL_BS_THRESHOLD_UNCALIBRATED" and v_bs["advance"]) if not BS_THRESHOLD_CALIBRATED \
        else ("BS_CONVERSION_LOW" in v_bs["flags"] and not v_bs["advance"]), v_bs

    bad_hyb = dict(passing, hyb_high_median=4200, hyb_low_median=4000)
    assert "HYB_FAIL" in validate_control_probes(bad_hyb)["flags"]
    swap = validate_control_probes(dict(passing, hyb_high_median=6845, hyb_low_median=20040, hyb_medium_median=12834))   # the GSE184269 slides
    assert "HYB_FAIL" not in swap["flags"] and swap["metrics"].get("hyb_order", "").startswith("swapped"), swap

    # deferred path (no summary, no decoder) -> advances with CTRL_QC_DEFERRED, no fake pass.
    r4 = step_0_4_control_probe_validation({"array_type": "EPIC_v1", "flags": []}, "g", "r")
    assert r4["ctrl_qc"] == "DEFERRED_PENDING_STAGE1_DECODER" and r4["advance"]

    print("Step 0.4 self-test: PASS (BS/HYB/EXT gates + deferred-extraction marker)")

    # Step 0.5: detection p-value math + detected-fraction gates (synthetic intensities).
    import numpy as _np
    neg = {"mu_bg": 200.0, "sigma_bg": 50.0}
    detected = [2000.0] * 1000                      # far above background -> p ~ 0
    v_all = validate_detection_p(compute_detection_p(detected, **neg))
    assert v_all["detection_qc"] == "PASS" and v_all["pct_probes_detected_p_le_01"] == 1.0, v_all

    borderline = [2000.0] * 970 + [180.0] * 30       # 97% detected
    v_bd = validate_detection_p(compute_detection_p(borderline, **neg))
    assert v_bd["detection_qc"] == "DETECTION_BORDERLINE" and v_bd["advance"], v_bd

    failing = [2000.0] * 900 + [180.0] * 100         # 90% detected
    v_fail = validate_detection_p(compute_detection_p(failing, **neg))
    assert v_fail["detection_qc"] == "FAIL_LOW_DETECTION" and not v_fail["advance"], v_fail

    r5 = step_0_5_detection_pvalue_qc({"flags": []})   # deferred path
    assert r5["detection_qc"] == "DEFERRED_PENDING_STAGE1_DECODER" and not r5["advance"]   # a deferred detection never advances (2026-09-27)

    print("Step 0.5 self-test: PASS (detection-p PASS/BORDERLINE/FAIL gates + deferred marker)")

    # Step 0.6: bead count.
    assert validate_bead_count([5] * 1000)["bead_qc"] == "PASS"
    assert validate_bead_count([5] * 990 + [1] * 10)["bead_qc"] == "WARN_LOW_BEAD_COUNT"

    # Step 0.7: call rate from detection + bead masks.
    dpm = [True] * 990 + [False] * 10
    bpm = [True] * 1000
    assert validate_call_rate(sum(1 for a, b in zip(dpm, bpm) if a and b), 1000)["call_rate_status"] == "PASS"
    assert validate_call_rate(960, 1000)["call_rate_status"] == "CALL_RATE_BORDERLINE"
    assert validate_call_rate(900, 1000)["call_rate_status"] == "CALL_RATE_FAIL"

    # Step 0.7b: platform tag + 450K coverage gate.
    assert step_0_7b_platform_coverage({"array_type": "EPIC_v1", "flags": []})["platform_tag"] == "EPIC"
    r7b = step_0_7b_platform_coverage({"array_type": "HM450K", "flags": []}, reference_cpg_coverage=0.72)
    assert r7b["hm450_coverage_gate"] == "FAIL" and not r7b["advance"]
    assert step_0_7b_platform_coverage({"array_type": "HM450K", "flags": []}, reference_cpg_coverage=0.91)["hm450_coverage_gate"] == "PASS"

    # Step 0.8: sex check (minfi getSex boundary).
    assert validate_sex(11.0, 12.5, "M")["sex_check"] == "PASS"        # Y-X = +1.5 -> M
    assert validate_sex(12.5, 8.0, "F")["sex_check"] == "PASS"         # Y-X = -4.5 -> F
    assert validate_sex(11.0, 12.5, "F")["sex_check"] == "MISMATCH"    # predicted M, declared F

    # Step 0.9: decision gate over a record.
    clean = {"integrity_status": "INTEGRITY_OK", "ctrl_qc": "PASS", "detection_qc": "PASS",
             "bead_qc": "PASS", "call_rate_status": "PASS", "hm450_coverage_gate": "NA_EPIC",
             "sex_check": "PASS", "flags": []}
    assert step_0_9_decision_gate(dict(clean))["stage0_verdict"] == "PROCEED"
    pen = dict(clean, detection_qc="DETECTION_BORDERLINE")
    assert step_0_9_decision_gate(pen)["stage0_verdict"] == "PROCEED_WITH_PENALTY"
    fail = dict(clean, sex_check="MISMATCH")
    g = step_0_9_decision_gate(fail)
    assert g["stage0_verdict"] == "QUARANTINE" and not g["advance"] and "sex" in g["stage0_hard_fail"]
    deferred = {"integrity_status": "INTEGRITY_OK", "ctrl_qc": "DEFERRED_PENDING_STAGE1_DECODER",
                "detection_qc": "DEFERRED_PENDING_STAGE1_DECODER", "flags": []}
    gd = step_0_9_decision_gate(deferred)
    # a deferred detection (or call rate) quarantines (PROC-INTAKE-01 B5, 2026-09-27); deferred controls alone are recorded
    assert gd["stage0_verdict"] == "QUARANTINE" and set(gd["stage0_deferred_qc"]) >= {"ctrl_qc", "detection"} \
        and "intake_deferred:detection" in gd["stage0_hard_fail"], gd

    print("Step 0.6-0.9 self-tests: PASS (bead / call-rate / platform-coverage / sex / decision gate)")
    print("  NOTE: these self-tests run the gates on constructed inputs; on a real array the intensities come from stage_0_1_qc_handoff.py,")
    print("        and a check that cannot be measured is marked deferred, never passed.")
