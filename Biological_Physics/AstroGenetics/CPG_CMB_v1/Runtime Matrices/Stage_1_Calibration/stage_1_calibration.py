"""Stage 1 — Calibration & beta computation (L2 + L3).

Built against CPG_Chain_of_Custody_SOP_v1_3.md §20-§27.

ARCHITECTURE (decided 2026-06-10):
  Stage 1 wraps the standard methylation stack (methylprep) for IDAT decoding (1.1) +
  per-sample noob normalization, rather than reimplementing validated tools. The IDAT
  decoder here is the SHARED decoder that also feeds Stage 0's deferred QC (control probes,
  detection-p, bead counts, chrX/Y intensities). Normalization = noob (see NORMALIZATION_METHOD
  for the rationale). ComBat (1.3) is a SKIP for single-patient deployment (single batch);
  it exists in the SOP only for multi-sample runs.

  Standing principle: H_min is DERIVED (IAMAtlas REBUILD MCMC + first principles), so the
  A-score floor is an absolute constant. There is NO cohort calibration and NO cross-cohort
  scale-shift step anywhere in this stage. The normalizer only produces clean patient beta.

Implemented now (IAM-native, dependency-free, tested):
  Step 1.4 (§23) — bisulfite-conversion efficiency check (reuses Stage 0 §14 controls).
  Step 1.5 (§24) — beta = M / (M + U + 100), the map-making step.
  Step 1.6 (§25) — beta sanity checks (range coverage, bimodality, cohort consistency).
  Step 1.7 (§26) — probe response function (identity; documented L3 gap, pass-through).
  Step 1.8 (§27) — Stage 1 output packaging (beta matrix + QC flags + provenance).

Deferred to the shared decoder / external stack (NotImplementedError until wired):
  IDAT decode, Step 1.1 dye-bias (noob), Step 1.2 probe-type normalization (funnorm),
  Step 1.3 ComBat (multi-batch only).
"""
from __future__ import annotations

import json
import os

BETA_STABILIZATION_OFFSET = 100        # SOP §24: beta = M / (M + U + 100)
# DECISION (2026-06-10, Heath-delegated): noob, NOT the SOP's funnorm default.
# Chosen purely on clinical-product merits. NOT for any cohort-calibration reason: the
# framework DERIVES H_min from the IAMAtlas REBUILD MCMC posteriors + first principles, so
# the A-score floor is an absolute derived constant and there is NO cross-cohort scale-shift
# problem to solve here. The normalizer only has to produce clean, reproducible patient beta:
#   - Per-sample deterministic. noob normalizes each sample from its own in-array controls,
#     so the same IDAT always yields the same beta + A-score regardless of what runs
#     alongside it. funnorm learns technical PCs across a sample BATCH -> batch-dependent,
#     non-reproducible per sample. Unacceptable for a clinical product.
#   - Lighter touch preserves the bimodality the entropy A-score reads against the floor.
#   - Pure Python (methylprep), no R/Bioconductor dependency — simpler, containerizable.
NORMALIZATION_METHOD = "noob"
BS_PASS = 0.98
BS_BORDERLINE = 0.95


# --- shared IDAT decoder + external-stack steps (deferred) ------------------

def decode_idat_pair(grn_path, red_path, array_type):
    """Decode an IDAT pair to per-probe M/U intensities plus the QC byproducts Stage 0
    defers to this decoder (control-probe intensities, bead counts, per-probe detection
    inputs, chrX/Y intensities). DEFERRED: wraps methylprep/minfi. Raises until wired so
    no fabricated intensities ever flow downstream."""
    raise NotImplementedError(
        "IDAT decoding wraps the standard methylation stack (methylprep / minfi), pending "
        "the wrap-vs-reimplement and funnorm-vs-noob sign-off. Supply M/U arrays directly "
        "to compute_beta() / run_stage_1() in the meantime."
    )


def step_1_1_dye_bias_correction(mu_intensities, method="noob"):
    raise NotImplementedError("Step 1.1 dye-bias wraps methylprep/minfi (noob). Pending sign-off.")


def step_1_2_probe_type_normalization(intensities, method=NORMALIZATION_METHOD):
    raise NotImplementedError(f"Step 1.2 probe-type normalization wraps minfi ({method}). Pending sign-off.")


def step_1_3_batch_correction_combat(beta_matrix, batch_labels):
    raise NotImplementedError("Step 1.3 ComBat wraps sva/combat-py; multi-batch only. Pending sign-off.")


# --- Step 1.4 bisulfite-conversion efficiency (§23) -------------------------

def step_1_4_bs_conversion_check(bs_conv_converted, bs_conv_unconverted) -> dict:
    """SOP §23: bs_eff = converted / (converted + unconverted). >=0.98 PASS,
    0.95-0.98 BORDERLINE (penalty), <0.95 FAIL (quarantine)."""
    denom = bs_conv_converted + bs_conv_unconverted
    if denom <= 0:
        return {"bs_efficiency": None, "bs_conversion": "FAIL", "advance": False}
    eff = bs_conv_converted / denom
    if eff >= BS_PASS:
        status, advance = "PASS", True
    elif eff >= BS_BORDERLINE:
        status, advance = "BORDERLINE", True
    else:
        status, advance = "FAIL", False
    return {"bs_efficiency": round(eff, 4), "bs_conversion": status, "advance": advance}


# --- Step 1.5 beta computation (§24) — the map-making step ------------------

def compute_beta(M, U, offset=BETA_STABILIZATION_OFFSET):
    """SOP §24: beta = M / (M + U + offset) per CpG. NaN where M or U is missing
    (per-probe NULL handling, never imputed)."""
    import numpy as np
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = M / (M + U + offset)
    beta[np.isnan(M) | np.isnan(U)] = np.nan
    return beta


def step_1_5_beta_computation(record, M=None, U=None) -> dict:
    flags = record.setdefault("flags", [])
    if M is None or U is None:
        record["beta_computed"] = False
        flags.append("BETA_DEFERRED:no_MU_intensities")
        record["advance"] = True
        return record
    import numpy as np
    beta = compute_beta(M, U)
    finite = beta[np.isfinite(beta)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        flags.append("BETA_OUT_OF_RANGE")
        record["beta_computed"] = False
        record["advance"] = False
        return record
    record["_beta"] = beta  # internal in-memory handoff to 1.6/1.7/1.8 (never JSON-serialized)
    record["beta_computed"] = True
    record["n_cpgs"] = int(beta.size)
    record["n_cpgs_null"] = int(np.sum(~np.isfinite(beta)))
    record["advance"] = True
    return record


# --- Step 1.6 beta sanity checks (§25) --------------------------------------

def beta_sanity(beta, cohort_median=None) -> dict:
    """SOP §25: range coverage (methylation lives at the extremes), bimodality
    (Hartigan dip if diptest present, else Sarle bimodality-coefficient fallback),
    and cohort-median consistency (when multi-sample)."""
    import numpy as np
    b = np.asarray(beta, dtype=float)
    b = b[np.isfinite(b)]
    n = b.size
    extremes_frac = float(np.mean((b < 0.2) | (b > 0.8))) if n else 0.0

    dip_stat, bimodal, method = None, None, None
    try:
        import diptest as _dt
        dip_stat, _ = _dt.diptest(b)
        bimodal, method = dip_stat > 0.01, "hartigan_dip"
    except Exception:
        if n > 3:
            m, s = float(np.mean(b)), float(np.std(b))
            if s > 0:
                z = (b - m) / s
                skew = float(np.mean(z ** 3))
                kurt = float(np.mean(z ** 4))
                bc = (skew ** 2 + 1) / kurt if kurt > 0 else 0.0
                dip_stat, bimodal, method = round(bc, 4), bc > 0.555, "sarle_bc_fallback"

    med = float(np.median(b)) if n else None
    med_consistency = (abs(med - cohort_median)
                       if (cohort_median is not None and med is not None) else None)

    checks = {
        "range_ok": extremes_frac >= 0.5,
        "bimodal_ok": bool(bimodal) if bimodal is not None else True,
        "cohort_ok": (med_consistency is None or med_consistency <= 0.05),
    }
    n_fail = sum(1 for v in checks.values() if v is False)
    status, advance = ("PASS", True) if n_fail == 0 else ("WARN", True) if n_fail == 1 else ("FAIL", False)
    return {"beta_sanity": status, "extremes_frac": round(extremes_frac, 4),
            "bimodality_stat": dip_stat, "bimodality_method": method,
            "median_consistency": (round(med_consistency, 4) if med_consistency is not None else None),
            "checks": checks, "advance": advance}


def step_1_6_beta_sanity(record, cohort_median=None) -> dict:
    flags = record.setdefault("flags", [])
    beta = record.get("_beta")
    if beta is None:
        record["beta_sanity"] = "DEFERRED_NO_BETA"
        record["advance"] = True
        return record
    v = beta_sanity(beta, cohort_median)
    record["beta_sanity"] = v["beta_sanity"]
    record["beta_sanity_detail"] = {k: v[k] for k in
                                    ("extremes_frac", "bimodality_stat", "bimodality_method", "median_consistency")}
    if v["beta_sanity"] == "WARN":
        flags.append("BETA_SANITY_WARN")
    elif v["beta_sanity"] == "FAIL":
        flags.append("BETA_SANITY_FAIL")
    record["advance"] = v["advance"]
    return record


# --- Step 1.7 probe response (§26) — identity, documented L3 gap ------------

def step_1_7_probe_response(record) -> dict:
    """SOP §26: per-probe response calibration. PROVISIONAL — identity transfer in
    production (declared L3 gap). Pass-through; no gating."""
    record["probe_response"] = "identity_provisional_L3_gap"
    record["advance"] = True
    return record


# --- Step 1.8 output packaging (§27) ----------------------------------------

def step_1_8_package_output(record, beta_output_dir=None) -> dict:
    """SOP §27: package the calibrated beta matrix + QC flags + provenance for Stage 2."""
    import numpy as np
    beta = record.get("_beta")
    provenance = {
        "normalization_method": NORMALIZATION_METHOD,
        "beta_offset": BETA_STABILIZATION_OFFSET,
        "probe_response": record.get("probe_response"),
        "bs_conversion": record.get("bs_conversion"),
        "beta_sanity": record.get("beta_sanity"),
        "stage0_verdict": record.get("stage0_verdict"),
        "platform_tag": record.get("platform_tag"),
        "n_cpgs": record.get("n_cpgs"), "n_cpgs_null": record.get("n_cpgs_null"),
    }
    record["stage1_output"] = {"beta_available": beta is not None, "provenance": provenance}
    if beta_output_dir and beta is not None:
        os.makedirs(beta_output_dir, exist_ok=True)
        rid = record.get("sample_run_id", "sample")
        bpath = os.path.join(beta_output_dir, f"beta_{rid}.npy")
        np.save(bpath, beta)
        with open(os.path.join(beta_output_dir, f"beta_{rid}_provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2)
        record["stage1_output"]["beta_path"] = bpath
    record["advance"] = True
    return record


# --- Stage 1 orchestration --------------------------------------------------

def run_stage_1(record, M=None, U=None, cohort_median=None,
                bs_controls=None, beta_output_dir=None) -> dict:
    """Run Stage 1: 1.1-1.3 deferred (external stack); 1.4 BS check (if controls supplied);
    1.5 beta; 1.6 sanity; 1.7 identity; 1.8 package. M/U come from the shared IDAT decoder."""
    flags = record.setdefault("flags", [])
    if bs_controls:
        bs = step_1_4_bs_conversion_check(bs_controls["converted"], bs_controls["unconverted"])
        record["bs_efficiency"] = bs["bs_efficiency"]
        record["bs_conversion"] = bs["bs_conversion"]
        if bs["bs_conversion"] == "FAIL":
            flags.append("BS_CONVERSION_FAIL")
            record["advance"] = False
            return record
        if bs["bs_conversion"] == "BORDERLINE":
            flags.append("BS_CONVERSION_BORDERLINE")
    record = step_1_5_beta_computation(record, M, U)
    if not record.get("advance"):
        return record
    record = step_1_6_beta_sanity(record, cohort_median)
    if not record.get("advance"):
        return record
    record = step_1_7_probe_response(record)
    return step_1_8_package_output(record, beta_output_dir)


if __name__ == "__main__":
    import numpy as np

    # Step 1.4 BS conversion
    assert step_1_4_bs_conversion_check(9800, 100)["bs_conversion"] == "PASS"
    assert step_1_4_bs_conversion_check(965, 35)["bs_conversion"] == "BORDERLINE"
    assert step_1_4_bs_conversion_check(900, 100)["bs_conversion"] == "FAIL"

    # Step 1.5 beta = M/(M+U+100)
    b = compute_beta([8000, 200], [200, 8000])
    assert abs(b[0] - 8000 / 8300) < 1e-9 and abs(b[1] - 200 / 8300) < 1e-9

    # Step 1.6 sanity: bimodal -> PASS, degenerate-unimodal -> FAIL
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(0.9, 0.03, 500), rng.normal(0.1, 0.03, 500)]).clip(0, 1)
    assert beta_sanity(bimodal)["beta_sanity"] == "PASS"
    degenerate = rng.normal(0.5, 0.03, 1000).clip(0, 1)
    assert beta_sanity(degenerate)["beta_sanity"] == "FAIL"

    # Full Stage 1 on synthetic M/U (bimodal) -> packaged output
    M = np.array([8000.0] * 500 + [200.0] * 500)
    U = np.array([200.0] * 500 + [8000.0] * 500)
    rec = run_stage_1({"sample_run_id": "t1", "flags": [], "platform_tag": "EPIC"}, M=M, U=U)
    assert rec["beta_computed"] and rec["beta_sanity"] == "PASS" and rec["advance"]
    assert rec["stage1_output"]["provenance"]["beta_offset"] == 100

    # deferred path (no M/U) -> advances, beta not computed
    rd = run_stage_1({"sample_run_id": "t2", "flags": []})
    assert rd["beta_computed"] is False and rd["advance"]

    print("Stage 1 self-test: PASS (1.4 BS / 1.5 beta / 1.6 sanity / 1.7 identity / 1.8 package + deferred)")
    print("  NOTE: IDAT decode + 1.1 dye-bias (noob) + 1.2 normalization (noob) + 1.3 ComBat wrap methylprep (deferred).")
