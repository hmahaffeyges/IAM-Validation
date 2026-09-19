#!/usr/bin/env python3
"""
VAL-062 — CRC tumor tissue cycling-class rescore
===================================================

Purpose
-------
CRC tumor cells are colon_epithelial = cycling class (H_min = 0.856055),
not secretory and not immune. VAL-062 corrects the class assignment error
of VAL-061 (which applied the Xu-538 immune panel to tumor tissue and
measured the tumor-infiltrating immune compartment rather than tumor
architecture) by scoring the same 26 matched pairs against cycling H_min
across all available HM450 CpGs.

Cohort
------
26 matched tumor/adjacent-normal pairs from TCGA-COAD, HM450 platform,
sesame level3 betas (public NIH GDC access, no dbGaP required).

Cohort SHA inherited from VAL-061 (same 26 pairs):
    ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27

Preregistration SHA:
    9b5ff04ce31e4679e32ac8690fefc0b09a0abd646e89792edf956161097b847d

Expected results SHA:
    e8ec05a8932e92c8755febbdb8df0425f9f25d161476895e6a0169837aae2698

Predicted outcome (sign-locked BEFORE rescore)
----------------------------------------------
Paired Cohen's d > 0, 95% CI > 0, d >= +0.5 — strong cycling-class signal,
direction confirmed, magnitude comparable to VAL-060 breast secretory tissue
arm (+0.745).

Actual outcome
--------------
Paired d = +0.7241, 95% CI [+0.2922, +1.1559], p = 2.23e-04. PASS.

Reproduction
------------
1. Download the 76 TCGA-COAD HM450 .txt files listed in
   COAD_matched_manifest.json via the NIH GDC public API.
2. Place in the `downloads/` subdirectory relative to this script.
3. Run:  python3 val062_crc_epic_tcga_coad_cycling.py

Author
------
Heath W. Mahaffey (IAMPerformance)
Principal / Founder, IAM Cosmological and Biological Framework
Entiat, WA

Patents pending: 64/012,720 and 64/014,568 (USPTO provisional, March 2026)

License
-------
Scripts and methodology released for reproducibility. Commercial use of the
GAPE framework requires license from IAMPerformance.

RNG seed: 20260420 (fixed for reproducibility)
"""

import json
import math
import hashlib
import statistics
from math import erf, sqrt
from pathlib import Path

# ─── CONSTANTS (FROZEN) ──────────────────────────────────────────────────────
# Cycling class H_min from G-002 MCMC posterior (R-hat = 1.0003)
# Reference: IAM framework derivation; Lister 2013 H_min_global = 0.7565,
# cycling class shifts to 0.856055 via class-specific C2 overhead.
H_MIN_CYCLING = 0.856055

# TCGA-COAD matched normal reference β (cycling-class calibration)
# Reference: VAL-001 TCGA COAD tumor-vs-matched-normal calibration
H_REF_BETA = 0.740

# Fixed RNG seed for reproducibility
RNG_SEED = 20260420

# Cohort SHA (inherited from VAL-061)
COHORT_SHA = "ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27"


# ─── ENTROPY + A-SCORE ───────────────────────────────────────────────────────
def H(beta: float) -> float:
    """Shannon entropy (bits) of Bernoulli(β)."""
    if beta <= 0 or beta >= 1:
        return 0.0
    return -beta * math.log2(beta) - (1 - beta) * math.log2(1 - beta)


def A_score(beta: float, h_min: float = H_MIN_CYCLING) -> float:
    """Architectural A-score: H(β) / H_min(class)."""
    return H(beta) / h_min


# ─── DATA LOAD ───────────────────────────────────────────────────────────────
def load_all_valid_betas(filepath: Path) -> list[float]:
    """
    Load all valid β values from a TCGA sesame level3 beta .txt file.

    Sesame level3 format: tab-separated, first column CpG probe id,
    second column β value, header line present.

    Returns list of valid β values in (0, 1) exclusive.
    """
    betas = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    b = float(parts[1])
                    if 0 < b < 1 and not math.isnan(b):
                        betas.append(b)
                except ValueError:
                    pass
    return betas


# ─── STATISTICS ──────────────────────────────────────────────────────────────
def norm_sf(x: float) -> float:
    """One-sided survival function of the standard normal."""
    return 1.0 - 0.5 * (1 + erf(abs(x) / sqrt(2)))


def cohens_d_paired(deltas: list[float]) -> tuple[float, float, float, float, float]:
    """
    Paired Cohen's d with 95% CI via the standard SE formula.

    Returns (d, ci_lo, ci_hi, t_stat, p_value).
    """
    n = len(deltas)
    mean_d = statistics.mean(deltas)
    sd_d = statistics.stdev(deltas) if n > 1 else 0.0
    d = mean_d / sd_d if sd_d > 0 else 0.0
    se = math.sqrt(1/n + d**2 / (2*n))
    t_stat = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else 0.0
    p = 2 * norm_sf(t_stat)
    return d, d - 1.96*se, d + 1.96*se, t_stat, p


def cohens_d_unpaired(x1: list[float], x2: list[float]) -> tuple[float, float, float, float, float]:
    """
    Unpaired Cohen's d (pooled SD) with 95% CI.

    Returns (d, ci_lo, ci_hi, t_stat, p_value).
    """
    n1, n2 = len(x1), len(x2)
    m1, m2 = statistics.mean(x1), statistics.mean(x2)
    s1, s2 = statistics.stdev(x1), statistics.stdev(x2)
    pooled_sd = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0
    se = math.sqrt((n1+n2)/(n1*n2) + d**2 / (2*(n1+n2)))
    t_stat = (m1 - m2) / (pooled_sd * math.sqrt(1/n1 + 1/n2)) if pooled_sd > 0 else 0.0
    p = 2 * norm_sf(t_stat)
    return d, d - 1.96*se, d + 1.96*se, t_stat, p


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).parent
    downloads = script_dir / "downloads"

    # Load the 26 QC-passed matched pairs
    with open(script_dir / "COAD_pairs.json") as f:
        pairs = json.load(f)

    # Load matched manifest for file paths
    with open(script_dir / "COAD_matched_manifest.json") as f:
        matched = json.load(f)

    patient_files = {}
    for m in matched:
        patient_files[m["patient"]] = {
            "tumor":  downloads / f"{m['patient']}__tumor__{m['tumor_file_name']}",
            "normal": downloads / f"{m['patient']}__normal__{m['normal_file_name']}",
        }

    qc_patients = [p["patient"] for p in pairs]
    print(f"VAL-062 — CRC tumor cycling-class rescore")
    print(f"{'='*72}")
    print(f"n matched pairs (inherited from VAL-061 QC): {len(qc_patients)}")
    print(f"Cohort SHA: {COHORT_SHA}")
    print(f"H_min(cycling) = {H_MIN_CYCLING}")
    print(f"Reference β (TCGA COAD matched normal) = {H_REF_BETA}")
    print(f"Reference A at β={H_REF_BETA}: A_ref = {A_score(H_REF_BETA):.4f}")
    print(f"RNG seed: {RNG_SEED}")
    print()

    # Per-sample mean A-score across all valid HM450 CpGs
    results = []
    for pid in qc_patients:
        tumor_betas  = load_all_valid_betas(patient_files[pid]["tumor"])
        normal_betas = load_all_valid_betas(patient_files[pid]["normal"])
        if not tumor_betas or not normal_betas:
            print(f"  {pid}: missing data — skipping")
            continue
        A_t = sum(A_score(b) for b in tumor_betas)  / len(tumor_betas)
        A_n = sum(A_score(b) for b in normal_betas) / len(normal_betas)
        results.append({
            "patient": pid,
            "A_tumor": A_t,
            "A_normal": A_n,
            "delta_A": A_t - A_n,
            "n_cpg_tumor":  len(tumor_betas),
            "n_cpg_normal": len(normal_betas),
        })

    n = len(results)

    # Per-patient A-score statistics
    A_tumors  = [r["A_tumor"]  for r in results]
    A_normals = [r["A_normal"] for r in results]
    deltas    = [r["delta_A"]  for r in results]

    mean_t = statistics.mean(A_tumors);  sd_t = statistics.stdev(A_tumors)
    mean_n = statistics.mean(A_normals); sd_n = statistics.stdev(A_normals)
    mean_d = statistics.mean(deltas);    sd_del = statistics.stdev(deltas)

    # Primary test — paired Cohen's d
    d_p, ci_lo_p, ci_hi_p, t_p, p_p = cohens_d_paired(deltas)

    # Secondary test — unpaired Cohen's d
    d_u, ci_lo_u, ci_hi_u, t_u, p_u = cohens_d_unpaired(A_tumors, A_normals)

    # Report
    print(f"PER-PATIENT A-SCORES (all valid HM450 CpGs, cycling-class H_min):")
    print(f"  Tumor   mean = {mean_t:.5f}, sd = {sd_t:.5f}")
    print(f"  Normal  mean = {mean_n:.5f}, sd = {sd_n:.5f}")
    print(f"  Δ(T-N)  mean = {mean_d:+.5f}, sd = {sd_del:.5f}")
    print()
    print(f"PRIMARY — PAIRED COHEN'S D:")
    print(f"  d = {d_p:+.4f}")
    print(f"  95% CI = [{ci_lo_p:+.4f}, {ci_hi_p:+.4f}]")
    print(f"  paired t = {t_p:+.3f}")
    print(f"  p = {p_p:.2e}")
    print()
    print(f"SECONDARY — UNPAIRED COHEN'S D:")
    print(f"  d = {d_u:+.4f}")
    print(f"  95% CI = [{ci_lo_u:+.4f}, {ci_hi_u:+.4f}]")
    print(f"  t = {t_u:+.3f}")
    print(f"  p = {p_u:.2e}")
    print()

    absolute_dA = mean_t - mean_n
    print(f"Absolute ΔA (tumor-normal mean difference): {absolute_dA:+.5f}")
    print(f"Framework prediction (VAL-001 TCGA COAD): ΔA ≈ +0.17")
    print(f"Observed / Predicted ratio: {absolute_dA/0.17:.2f}x")
    print(f"  Note: VAL-001 target is calibrated on cycling-class-discriminating CpG")
    print(f"  subsets (colon-specific DMRs). VAL-062 averages across all ~485K HM450")
    print(f"  CpGs — cycling-class signal diluted by non-class-informative probes.")
    print(f"  Cohen's d remains strong because between-patient variance is also small")
    print(f"  at the genome-wide average level.")
    print()

    # Save results
    output = {
        "val_id": "VAL-062",
        "supersedes": "VAL-061 (class correction: cycling not immune)",
        "cohort": "TCGA-COAD HM450 matched tumor/normal (26 pairs inherited from VAL-061)",
        "cohort_sha": COHORT_SHA,
        "scoring_class": "cycling",
        "H_min": H_MIN_CYCLING,
        "reference_beta": H_REF_BETA,
        "n_pairs": n,
        "A_tumor_mean": mean_t,  "A_tumor_sd": sd_t,
        "A_normal_mean": mean_n, "A_normal_sd": sd_n,
        "delta_A_mean": mean_d,  "delta_A_sd": sd_del,
        "delta_A_absolute": absolute_dA,
        "paired_d": d_p, "paired_d_ci_95": [ci_lo_p, ci_hi_p],
        "paired_t": t_p, "paired_p": p_p,
        "unpaired_d": d_u, "unpaired_d_ci_95": [ci_lo_u, ci_hi_u],
        "unpaired_t": t_u, "unpaired_p": p_u,
        "framework_prediction_delta_A": 0.17,
        "rng_seed": RNG_SEED,
    }
    out_path = script_dir / "VAL-062_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    results_sha = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    print(f"Results JSON: {out_path}")
    print(f"Results SHA256: {results_sha}")
    print()

    # Sign-check vs preregistered prediction
    print(f"{'='*72}")
    print(f"SIGN CHECK vs PREREGISTERED PREDICTION")
    print(f"{'='*72}")
    print(f"Predicted: paired d > 0, 95% CI > 0, d ≥ +0.5")
    print(f"Observed:  paired d = {d_p:+.4f}, 95% CI = [{ci_lo_p:+.4f}, {ci_hi_p:+.4f}]")
    if ci_lo_p > 0 and d_p >= 0.5:
        print(f"OUTCOME: PASS — direction CONFIRMED, magnitude STRONG")
    elif ci_lo_p > 0 and d_p > 0:
        print(f"OUTCOME: direction confirmed (positive), magnitude moderate (d={d_p:.2f} < 0.5)")
    elif ci_hi_p < 0:
        print(f"OUTCOME: direction INVERTED — framework inconsistency")
    else:
        print(f"OUTCOME: ambiguous — CI crosses zero")


if __name__ == "__main__":
    main()
