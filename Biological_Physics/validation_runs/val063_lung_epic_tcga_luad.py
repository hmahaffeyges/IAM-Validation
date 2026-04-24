#!/usr/bin/env python3
"""
VAL-063 — Lung-EPIC tissue arm on TCGA-LUAD HM450 matched tumor/normal
=======================================================================

Purpose
-------
First tissue-arm validation of the lung-epic card. Lung adenocarcinoma cells
are lung_epithelial = cycling class (H_min = 0.856055, reference β = 0.738
from TCGA-LUAD matched normal). VAL-063 scores TCGA-LUAD matched tumor /
adjacent-normal pairs against cycling H_min across all valid HM450 CpGs.

Same methodology as VAL-062 (crc-epic tissue arm), extended to lung.

Cohort
------
29 matched tumor/adjacent-normal pairs from TCGA-LUAD, HM450 platform,
sesame level3 betas (public NIH GDC access, no dbGaP required).

Manifest SHA:   6e87cc32b84f278d1b77ad766a050f2a378aa3a8e3da78e7232b2511514d278c
Cohort SHA:     53718abc88680e0793b0455ac51fbf8e6a128f615c508f0de60dc8d8cfd4d6e9
Prereg SHA:     f56ebe0ab015d856c86573e502fde132743a95fcb1d3667074a5001993f4108e
Expected results SHA: 809025760e30b42f040c41f8e95b94ad771bf0bb58b631a74207d2340409a9ba

Predicted outcome (sign-locked BEFORE analysis)
-----------------------------------------------
Paired Cohen's d > 0, 95% CI > 0, d >= +0.5 — strong cycling-class signal,
magnitude comparable to VAL-062 CRC cycling tissue arm (+0.724).

Actual outcome
--------------
Paired d = +1.0202, 95% CI [+0.5714, +1.4690], p = 3.93e-08. PASS (strong).
Largest cycling-class tissue effect to date across all Cookbook cards.

Reproduction
------------
1. Download the 58 TCGA-LUAD HM450 .txt files listed in
   LUAD_matched_manifest.json via the NIH GDC public API:
       https://api.gdc.cancer.gov/data/{file_id}
2. Place in the `downloads/` subdirectory relative to this script.
3. Run:  python3 val063_lung_epic_tcga_luad.py

Author
------
Heath W. Mahaffey (IAMPerformance), Entiat, WA
Patents pending: 64/012,720 and 64/014,568 (USPTO provisional, March 2026)

License: scripts and methodology released for reproducibility.
Commercial use of the GAPE framework requires license from IAMPerformance.

RNG seed: 20260420
"""

import json
import math
import hashlib
import statistics
from math import erf, sqrt
from pathlib import Path

# ─── CONSTANTS (FROZEN) ──────────────────────────────────────────────────────
# Cycling class H_min from G-002 MCMC posterior (R-hat = 1.0003)
H_MIN_CYCLING = 0.856055

# TCGA-LUAD matched normal reference β (cycling-class calibration)
# Reference: VAL-041 / VAL-056 lung_epithelial healthy reference
H_REF_BETA = 0.738

# QC threshold — minimum valid CpGs per sample (HM450 has ~485K; 400K = ~82%)
MIN_VALID_CPGS = 400_000

# Fixed RNG seed for reproducibility
RNG_SEED = 20260420


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
def load_all_valid_betas(filepath: Path) -> list[float] | None:
    """
    Load all valid β values from a TCGA sesame level3 beta .txt file.

    Sesame level3 format: tab-separated, first column CpG probe id,
    second column β value, header line present.

    Returns list of valid β values in (0, 1) exclusive, or None if file
    cannot be read.
    """
    if not filepath.exists():
        return None
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


def cohens_d_paired(deltas):
    n = len(deltas)
    mean_d = statistics.mean(deltas)
    sd_d = statistics.stdev(deltas) if n > 1 else 0.0
    d = mean_d / sd_d if sd_d > 0 else 0.0
    se = math.sqrt(1/n + d**2 / (2*n))
    t_stat = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else 0.0
    p = 2 * norm_sf(t_stat)
    return d, d - 1.96*se, d + 1.96*se, t_stat, p


def cohens_d_unpaired(x1, x2):
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

    with open(script_dir / "LUAD_matched_manifest.json") as f:
        manifest = json.load(f)

    print(f"VAL-063 — Lung-EPIC tissue arm, TCGA-LUAD, cycling-class scoring")
    print(f"{'='*72}")
    print(f"H_min(cycling)       = {H_MIN_CYCLING}")
    print(f"Reference β (LUAD N) = {H_REF_BETA}")
    print(f"RNG seed             = {RNG_SEED}")
    print()

    # Load + QC each matched pair
    results = []
    skipped = []
    for m in manifest:
        pid = m["patient"]
        tpath = downloads / f"{pid}__tumor__{m['tumor_file_name']}"
        npath = downloads / f"{pid}__normal__{m['normal_file_name']}"
        tbetas = load_all_valid_betas(tpath)
        nbetas = load_all_valid_betas(npath)
        if tbetas is None or nbetas is None:
            skipped.append((pid, "missing file"))
            continue
        if len(tbetas) < MIN_VALID_CPGS or len(nbetas) < MIN_VALID_CPGS:
            skipped.append((pid, f"coverage: t={len(tbetas)} n={len(nbetas)}"))
            continue
        A_t = sum(A_score(b) for b in tbetas) / len(tbetas)
        A_n = sum(A_score(b) for b in nbetas) / len(nbetas)
        results.append({
            "patient": pid,
            "A_tumor": A_t,
            "A_normal": A_n,
            "delta_A": A_t - A_n,
            "n_cpg_tumor": len(tbetas),
            "n_cpg_normal": len(nbetas),
        })

    n = len(results)
    print(f"QC-passed pairs: {n}   (skipped: {len(skipped)})")

    qc_patients = sorted([r["patient"] for r in results])
    cohort_sha = hashlib.sha256(json.dumps(qc_patients).encode()).hexdigest()
    print(f"Cohort SHA:   {cohort_sha}")
    print()

    A_tumors  = [r["A_tumor"]  for r in results]
    A_normals = [r["A_normal"] for r in results]
    deltas    = [r["delta_A"]  for r in results]

    m_t = statistics.mean(A_tumors); sd_t = statistics.stdev(A_tumors)
    m_n = statistics.mean(A_normals); sd_n = statistics.stdev(A_normals)
    m_d = statistics.mean(deltas);    sd_d = statistics.stdev(deltas)

    d_p, ci_lo_p, ci_hi_p, t_p, p_p = cohens_d_paired(deltas)
    d_u, ci_lo_u, ci_hi_u, t_u, p_u = cohens_d_unpaired(A_tumors, A_normals)

    print(f"PER-PATIENT A-SCORES (all valid HM450 CpGs, cycling-class H_min):")
    print(f"  Tumor  mean = {m_t:.5f}, sd = {sd_t:.5f}")
    print(f"  Normal mean = {m_n:.5f}, sd = {sd_n:.5f}")
    print(f"  Δ(T-N) mean = {m_d:+.5f}, sd = {sd_d:.5f}")
    print()
    print(f"PRIMARY — PAIRED COHEN'S D:")
    print(f"  d = {d_p:+.4f}")
    print(f"  95% CI = [{ci_lo_p:+.4f}, {ci_hi_p:+.4f}]")
    print(f"  paired t = {t_p:+.3f}, p = {p_p:.2e}")
    print()
    print(f"SECONDARY — UNPAIRED COHEN'S D:")
    print(f"  d = {d_u:+.4f}")
    print(f"  95% CI = [{ci_lo_u:+.4f}, {ci_hi_u:+.4f}]")
    print(f"  t = {t_u:+.3f}, p = {p_u:.2e}")
    print()

    absolute_dA = m_t - m_n
    print(f"Absolute ΔA (genome-wide mean): {absolute_dA:+.5f}")
    print(f"Framework expectation (cycling-class-informative CpGs): ~+0.14")
    print(f"Genome-wide mean dilutes class signal (same caveat as VAL-062);")
    print(f"Cohen's d remains strong because between-patient variance is also small.")
    print()

    output = {
        "val_id": "VAL-063",
        "card": "lung-epic",
        "cohort": "TCGA-LUAD HM450 matched tumor/normal",
        "cohort_sha": cohort_sha,
        "scoring_class": "cycling",
        "H_min": H_MIN_CYCLING,
        "reference_beta": H_REF_BETA,
        "n_pairs": n,
        "A_tumor_mean": m_t, "A_tumor_sd": sd_t,
        "A_normal_mean": m_n, "A_normal_sd": sd_n,
        "delta_A_mean": m_d, "delta_A_sd": sd_d,
        "delta_A_absolute": absolute_dA,
        "paired_d": d_p, "paired_d_ci_95": [ci_lo_p, ci_hi_p],
        "paired_t": t_p, "paired_p": p_p,
        "unpaired_d": d_u, "unpaired_d_ci_95": [ci_lo_u, ci_hi_u],
        "unpaired_t": t_u, "unpaired_p": p_u,
        "rng_seed": RNG_SEED,
    }
    out_path = script_dir / "VAL-063_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    results_sha = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    print(f"Results JSON: {out_path}")
    print(f"Results SHA256: {results_sha}")
    print()

    print(f"{'='*72}")
    print(f"SIGN CHECK vs PREREGISTERED PREDICTION")
    print(f"{'='*72}")
    print(f"Predicted: paired d > 0, 95% CI > 0, d ≥ +0.5")
    print(f"Observed:  paired d = {d_p:+.4f}, CI = [{ci_lo_p:+.4f}, {ci_hi_p:+.4f}]")
    if ci_lo_p > 0 and d_p >= 0.5:
        print(f"OUTCOME: PASS — direction CONFIRMED, magnitude STRONG")
    elif ci_lo_p > 0 and d_p > 0:
        print(f"OUTCOME: weak PASS — direction confirmed, magnitude moderate (d={d_p:.2f})")
    elif ci_hi_p < 0:
        print(f"OUTCOME: INVERTED — framework inconsistency")
    else:
        print(f"OUTCOME: ambiguous — CI crosses zero")


if __name__ == "__main__":
    main()
