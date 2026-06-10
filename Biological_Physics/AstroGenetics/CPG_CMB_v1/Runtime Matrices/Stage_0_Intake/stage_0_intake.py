"""Stage 0 — Sample intake (L1).

Built one step at a time against CPG_Chain_of_Custody_SOP_v1_3.md.
Implemented so far:
  Step 0.1 (SOP §11) — IDAT file arrival on server.

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
    print("  NOTE: IDAT size/array-type/duplicate gates require real IDAT files to exercise.")
