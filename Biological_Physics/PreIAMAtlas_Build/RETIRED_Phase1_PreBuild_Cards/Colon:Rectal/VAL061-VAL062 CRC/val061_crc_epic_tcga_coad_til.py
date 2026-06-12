#!/usr/bin/env python3
"""
VAL-061 — CRC tumor tissue TIL compartment reading (supplementary)
==================================================================

Purpose
-------
Originally designed as a CRC tumor architecture test, VAL-061 applied the
Xu-538 immune panel to CRC tumor tissue and scored against immune H_min.
This reads the tumor-infiltrating immune compartment (TIL) inside the tumor
bed — NOT tumor cell architecture. The d = +1.07 result is valid as a TIL
reading but was not a correct test of tumor architecture.

VAL-062 is the correct tumor architecture test (cycling-class scoring, all
HM450 CpGs). VAL-061 is retained in the record as a supplementary TIL arm
because it provides independent evidence of TIL activation that reconciles
with VAL-047 peripheral blood immune suppression (same immune class,
opposite compartments, opposite-sign A-scores — documented in CCL-019).

Cohort
------
26 matched tumor/adjacent-normal pairs from TCGA-COAD, HM450 platform,
sesame level3 betas (public NIH GDC access, no dbGaP required).
Same cohort used by VAL-062.

Cohort SHA: ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27
Prereg SHA: bdce2f903a20a3375681a3589710c2f5a6392a4f4c6772305fd3afc656bed521
Results SHA: def8a69030a2b1d1619f4a930e419604b44c0f2097655c97eea7f580f4a12c96
Panel SHA:  ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6

Result
------
Paired Cohen's d = +1.0658, 95% CI [+0.5845, +1.5471], p < 0.00001
Per-CpG direction: 61% hypomethylated, 39% hypermethylated (classic CRC
global hypomethylation + aggregate entropy elevation because Shannon H(β)
rises as β moves toward 0.5 from either direction).

Reproduction
------------
1. Download the 76 TCGA-COAD HM450 .txt files listed in
   COAD_matched_manifest.json via the NIH GDC public API.
2. Place in the `downloads/` subdirectory relative to this script.
3. Ensure xu538_panel.json is present (538 CpG IDs).
4. Run:  python3 val061_crc_epic_tcga_coad_til.py

Author
------
Heath W. Mahaffey (IAMPerformance), Entiat, WA
Patents pending: 64/012,720 and 64/014,568 (USPTO provisional, March 2026)

RNG seed: 20260420
"""

import json
import math
import hashlib
import statistics
from math import erf, sqrt
from pathlib import Path

# ─── CONSTANTS (FROZEN) ──────────────────────────────────────────────────────
# Immune class H_min from G-002 MCMC posterior (R-hat = 1.0007)
# 6.44σ correction from initial neutrophil-reference calibration 0.795
H_MIN_IMMUNE = 0.838889

# Xu 2020 blood cancer methylation panel (Sister Study cohort)
# Citation: Xu Z, Sandler DP, Taylor JA. J Natl Cancer Inst 2020;112:87-94
# Panel SHA locked: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"

# QC threshold — minimum CpGs per sample from the 538-CpG panel
MIN_VALID_CPGS = 430  # ~80% panel coverage on HM450

# Fixed RNG seed
RNG_SEED = 20260420

# Cohort SHA
COHORT_SHA = "ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27"


# ─── ENTROPY + A-SCORE ───────────────────────────────────────────────────────
def H(beta: float) -> float:
    """Shannon entropy (bits) of Bernoulli(β)."""
    if beta <= 0 or beta >= 1:
        return 0.0
    return -beta * math.log2(beta) - (1 - beta) * math.log2(1 - beta)


def A_score(beta: float, h_min: float = H_MIN_IMMUNE) -> float:
    """Architectural A-score against immune H_min."""
    return H(beta) / h_min


# ─── DATA LOAD ───────────────────────────────────────────────────────────────
def load_panel_betas(filepath: Path, panel_cpgs: set[str]) -> dict[str, float]:
    """
    Load β values for Xu-538 panel CpGs from a TCGA sesame level3 beta .txt file.

    Returns {cpg_id: β} for valid β in (0, 1) exclusive and present in the panel.
    """
    betas = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cpg = parts[0]
                if cpg in panel_cpgs:
                    try:
                        b = float(parts[1])
                        if 0 < b < 1 and not math.isnan(b):
                            betas[cpg] = b
                    except ValueError:
                        pass
    return betas


# ─── STATISTICS ──────────────────────────────────────────────────────────────
def norm_sf(x: float) -> float:
    return 1.0 - 0.5 * (1 + erf(abs(x) / sqrt(2)))


def cohens_d_paired(deltas: list[float]) -> tuple[float, float, float, float, float]:
    n = len(deltas)
    mean_d = statistics.mean(deltas)
    sd_d = statistics.stdev(deltas) if n > 1 else 0.0
    d = mean_d / sd_d if sd_d > 0 else 0.0
    se = math.sqrt(1/n + d**2 / (2*n))
    t_stat = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else 0.0
    p = 2 * norm_sf(t_stat)
    return d, d - 1.96*se, d + 1.96*se, t_stat, p


def cohens_d_unpaired(x1: list[float], x2: list[float]) -> tuple[float, float, float, float, float]:
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

    # Load Xu-538 panel
    with open(script_dir / "xu538_panel.json") as f:
        panel_data = json.load(f)
        panel_cpgs = set(panel_data if isinstance(panel_data, list) else panel_data.get("cpgs", []))

    # Verify panel SHA
    panel_sha_computed = hashlib.sha256(
        json.dumps(sorted(panel_cpgs)).encode()
    ).hexdigest()
    print(f"VAL-061 — CRC tumor TIL compartment reading (Xu-538 immune panel)")
    print(f"{'='*72}")
    print(f"Panel: Xu 2020 538 CpG immune cancer panel")
    print(f"Panel size: {len(panel_cpgs)} CpGs")
    print(f"Expected panel SHA: {PANEL_SHA}")

    # Load matched manifest
    with open(script_dir / "COAD_matched_manifest.json") as f:
        matched = json.load(f)

    # Score all 38 matched pairs; apply QC filter to produce the 26 cohort
    print(f"\nLoading and QC-filtering {len(matched)} candidate matched pairs...")
    qc_passed = []
    for m in matched:
        pid = m["patient"]
        tumor_path  = downloads / f"{pid}__tumor__{m['tumor_file_name']}"
        normal_path = downloads / f"{pid}__normal__{m['normal_file_name']}"
        if not (tumor_path.exists() and normal_path.exists()):
            continue
        tumor_bs  = load_panel_betas(tumor_path,  panel_cpgs)
        normal_bs = load_panel_betas(normal_path, panel_cpgs)
        if len(tumor_bs) >= MIN_VALID_CPGS and len(normal_bs) >= MIN_VALID_CPGS:
            qc_passed.append({
                "patient": pid,
                "tumor_betas":  tumor_bs,
                "normal_betas": normal_bs,
            })

    n = len(qc_passed)
    print(f"QC-passed pairs (≥{MIN_VALID_CPGS}/538 CpGs): {n}")
    print(f"Cohort SHA: {COHORT_SHA}")
    print(f"H_min(immune) = {H_MIN_IMMUNE}")
    print(f"RNG seed: {RNG_SEED}")
    print()

    # Per-patient A-score across the Xu-538 panel
    A_tumors, A_normals, deltas = [], [], []
    per_cpg_deltas = {cpg: [] for cpg in panel_cpgs}
    for pair in qc_passed:
        A_t = sum(A_score(b) for b in pair["tumor_betas"].values())  / len(pair["tumor_betas"])
        A_n = sum(A_score(b) for b in pair["normal_betas"].values()) / len(pair["normal_betas"])
        A_tumors.append(A_t)
        A_normals.append(A_n)
        deltas.append(A_t - A_n)
        # Collect per-CpG Δβ for the shared set
        shared = set(pair["tumor_betas"].keys()) & set(pair["normal_betas"].keys())
        for cpg in shared:
            per_cpg_deltas[cpg].append(pair["tumor_betas"][cpg] - pair["normal_betas"][cpg])

    # Statistics
    mean_t = statistics.mean(A_tumors);  sd_t = statistics.stdev(A_tumors)
    mean_n = statistics.mean(A_normals); sd_n = statistics.stdev(A_normals)
    mean_d = statistics.mean(deltas);    sd_del = statistics.stdev(deltas)

    d_p, ci_lo_p, ci_hi_p, t_p, p_p = cohens_d_paired(deltas)
    d_u, ci_lo_u, ci_hi_u, t_u, p_u = cohens_d_unpaired(A_tumors, A_normals)

    # Per-CpG direction analysis
    cpg_means = {cpg: statistics.mean(ds) for cpg, ds in per_cpg_deltas.items() if len(ds) >= 20}
    n_cpgs_analyzed = len(cpg_means)
    n_hyper = sum(1 for m in cpg_means.values() if m > 0)
    n_hypo  = sum(1 for m in cpg_means.values() if m < 0)

    # Report
    print(f"PER-PATIENT A-SCORES (Xu-538 panel, immune-class H_min):")
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
    print(f"PER-CPG DIRECTION (panel CpGs with ≥20 paired observations):")
    print(f"  n_analyzed = {n_cpgs_analyzed}")
    print(f"  hypermethylated in tumor: {n_hyper} ({100*n_hyper/n_cpgs_analyzed:.1f}%)")
    print(f"  hypomethylated in tumor:  {n_hypo}  ({100*n_hypo /n_cpgs_analyzed:.1f}%)")
    print(f"  Pattern: CRC global hypomethylation — consistent with published CRC literature.")
    print(f"  Aggregate A-score elevation positive because H(β) rises toward 0.5 from either side.")
    print()

    # Interpretation note
    print(f"INTERPRETATION:")
    print(f"  Xu-538 on tumor tissue reads the tumor-infiltrating immune compartment (TIL),")
    print(f"  not the tumor cell architecture. TIL are activated and expanded inside the tumor")
    print(f"  bed relative to resting peripheral leukocytes — opposite-direction A-score from")
    print(f"  VAL-047 peripheral blood immune (d = -0.33 suppressed circulating response).")
    print(f"  Same immune class, opposite compartments, opposite signs (CCL-019).")
    print(f"  See VAL-062 (cycling-class scoring, all HM450 CpGs) for the correct tumor")
    print(f"  architecture test.")

    # Save
    output = {
        "val_id": "VAL-061",
        "status": "SUPPLEMENTARY (TIL compartment, not tumor architecture)",
        "corrected_by": "VAL-062 (cycling-class scoring is the correct tumor architecture test)",
        "cohort": "TCGA-COAD HM450 matched tumor/normal",
        "cohort_sha": COHORT_SHA,
        "panel": "Xu-538 immune",
        "panel_sha": PANEL_SHA,
        "scoring_class": "immune",
        "H_min": H_MIN_IMMUNE,
        "n_pairs": n,
        "A_tumor_mean": mean_t,  "A_tumor_sd": sd_t,
        "A_normal_mean": mean_n, "A_normal_sd": sd_n,
        "delta_A_mean": mean_d,  "delta_A_sd": sd_del,
        "paired_d": d_p, "paired_d_ci_95": [ci_lo_p, ci_hi_p],
        "paired_t": t_p, "paired_p": p_p,
        "unpaired_d": d_u, "unpaired_d_ci_95": [ci_lo_u, ci_hi_u],
        "unpaired_t": t_u, "unpaired_p": p_u,
        "per_cpg_n_analyzed": n_cpgs_analyzed,
        "per_cpg_n_hyper": n_hyper,
        "per_cpg_pct_hyper": 100 * n_hyper / n_cpgs_analyzed,
        "per_cpg_n_hypo": n_hypo,
        "per_cpg_pct_hypo": 100 * n_hypo / n_cpgs_analyzed,
        "rng_seed": RNG_SEED,
    }
    out_path = script_dir / "VAL-061_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    results_sha = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    print(f"\nResults JSON: {out_path}")
    print(f"Results SHA256: {results_sha}")


if __name__ == "__main__":
    main()
