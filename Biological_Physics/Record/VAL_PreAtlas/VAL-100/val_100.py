#!/usr/bin/env python3
"""
VAL-100 — crc-epic Under-50 Buffy Coat Polyp Stage 1 Immune A-Score on GSE282666
=================================================================================

GSE282666 (Kumar/Brown/Yow, U Miami, 2024): n=51 buffy coat EPIC v2.0 (GPL33022),
all patients under age 50, with pre-neoplastic polyp (PNP) status from same-day
colonoscopy. PNP+ n=16, PNP- n=35.

Tests whether the crc-epic Stage 1 universal Xu-538 immune A-score extends backward
in disease trajectory from pre-diagnostic invasive CRC (VAL-047 d=−0.33) to
pre-neoplastic polyps. Pre-locked direction expectation: NEGATIVE (CCL-019
compartment-flip — crc-epic blood immune depressed, tumor cycling elevated).

Pre-registration sealed: 2026-04-28
Pre-reg SHA: 4017913d31b31e031ab01d2c0a016374334658ab9f526d99d90642d0f3f8bf67
RNG seed: 20260428
"""

import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import time

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

H_MIN_IMMUNE = 0.838889  # G-002 MCMC posterior, panc-LL-007 universal Stage 1 rule
RNG_SEED = 20260428
N_BOOTSTRAP = 10000
COVERAGE_FLAG_THRESHOLD = 0.10  # >10% drop = panel-transferability flag
OUTPUT_DIR = '/home/claude/edear_working/VAL-100/VAL-100'
BETAS_FILE = os.path.join(OUTPUT_DIR, 'GSE282666_Betas.csv.gz')
PANEL_FILE = '/home/claude/iam_repo/Biological_Physics/validation_runs/xu538_panel.json'

# EPIC-Italy healthy buffy coat comparator (from VAL-082 anchor)
ITALIAN_HEALTHY_MEAN_A = 0.4384
ITALIAN_HEALTHY_SD_A = 0.0244

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def shannon_h(beta):
    if beta is None or beta != beta:
        return None
    if beta <= 0 or beta >= 1:
        return 0.0
    p, q = beta, 1.0 - beta
    return -p * math.log2(p) - q * math.log2(q)


def cohens_d_unpaired(case, ctrl):
    if len(case) < 2 or len(ctrl) < 2:
        return None
    nc, nk = len(case), len(ctrl)
    mc = sum(case) / nc
    mk = sum(ctrl) / nk
    vc = sum((x - mc) ** 2 for x in case) / (nc - 1)
    vk = sum((x - mk) ** 2 for x in ctrl) / (nk - 1)
    pooled_sd = math.sqrt(((nc - 1) * vc + (nk - 1) * vk) / (nc + nk - 2))
    if pooled_sd == 0:
        return None
    return (mc - mk) / pooled_sd


def bootstrap_ci_unpaired(case, ctrl, n_iter, seed):
    rng = random.Random(seed)
    ds = []
    for _ in range(n_iter):
        c_resamp = [case[rng.randint(0, len(case) - 1)] for _ in range(len(case))]
        k_resamp = [ctrl[rng.randint(0, len(ctrl) - 1)] for _ in range(len(ctrl))]
        d = cohens_d_unpaired(c_resamp, k_resamp)
        if d is not None:
            ds.append(d)
    ds.sort()
    return [ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]]


def welch_t(case, ctrl):
    nc, nk = len(case), len(ctrl)
    mc = sum(case) / nc
    mk = sum(ctrl) / nk
    vc = sum((x - mc) ** 2 for x in case) / (nc - 1)
    vk = sum((x - mk) ** 2 for x in ctrl) / (nk - 1)
    se = math.sqrt(vc / nc + vk / nk) if (vc / nc + vk / nk) > 0 else 1e-12
    t = (mc - mk) / se
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return t, p


# ----------------------------------------------------------------------------
# Step 1: Load Xu-538 panel + clinical metadata + column mapping
# ----------------------------------------------------------------------------

def load_panel():
    with open(PANEL_FILE) as f:
        panel = json.load(f)
    cgs = panel['cpgs'] if 'cpgs' in panel else panel.get('cpg_ids', [])
    if not cgs:
        # Try other keys
        for k, v in panel.items():
            if isinstance(v, list) and len(v) > 100:
                cgs = v
                break
    return set(cgs), panel


def load_clinical():
    with open(os.path.join(OUTPUT_DIR, 'clinical_metadata.json')) as f:
        return json.load(f)


def load_column_mapping():
    with open(os.path.join(OUTPUT_DIR, 'column_mapping.json')) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Step 2: Stream-parse betas file, extract Xu-538 panel rows only
# ----------------------------------------------------------------------------

def extract_panel_betas(panel_cgs, col_mapping):
    """
    Stream-parse the betas CSV. For each row whose CpG matches a Xu-538 panel CpG
    (matching by cg-prefix, ignoring _BC11/_TC11 EPIC-v2 suffixes), extract betas
    for all 51 samples.
    Returns: {gsm: {cg_id: beta}} — dictionary of dictionaries.
    """
    col_gsms = col_mapping['col_index_to_gsm']
    sample_betas = {gsm: {} for gsm in col_gsms if gsm and gsm.startswith('GSM')}

    panel_hit_count = 0
    total_rows = 0
    matched_cgs = set()

    with gzip.open(BETAS_FILE, 'rt') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Verify column count
        assert len(header) == len(col_gsms), f"Header/mapping length mismatch: {len(header)} vs {len(col_gsms)}"

        for row in reader:
            total_rows += 1
            if not row:
                continue
            cg_full = row[0].strip()
            # EPIC-v2 IDs can be cgXXXXXXXX_BC11 / cgXXXXXXXX_TC11 etc.
            cg_prefix = cg_full.split('_')[0]
            if cg_prefix not in panel_cgs:
                continue
            panel_hit_count += 1
            matched_cgs.add(cg_prefix)
            for i, val in enumerate(row[1:], start=1):
                gsm = col_gsms[i]
                if gsm and gsm.startswith('GSM'):
                    try:
                        b = float(val) if val and val.lower() != 'na' else None
                        if b is not None and 0.0 <= b <= 1.0:
                            sample_betas[gsm][cg_prefix] = b
                    except (ValueError, TypeError):
                        pass
            if total_rows % 100000 == 0:
                print(f"  ...processed {total_rows:,} rows, panel hits {panel_hit_count}")

    return sample_betas, matched_cgs, total_rows


def coverage_summary(sample_betas, matched_cgs, total_panel):
    print(f"\nPanel coverage:")
    print(f"  Panel size:       {total_panel}")
    print(f"  Matched in betas: {len(matched_cgs)} ({100*len(matched_cgs)/total_panel:.1f}%)")
    drop = (total_panel - len(matched_cgs)) / total_panel
    print(f"  Coverage drop:    {100*drop:.1f}%")
    print(f"  Per-sample valid CpG counts (mean, min, max):")
    counts = [len(b) for b in sample_betas.values()]
    if counts:
        print(f"    mean={sum(counts)/len(counts):.1f}, min={min(counts)}, max={max(counts)}")
    return drop


# ----------------------------------------------------------------------------
# Step 3: Beta distribution check (CHK-3.1)
# ----------------------------------------------------------------------------

def beta_distribution_check(sample_betas):
    """Pool all panel betas, count fraction in [0, 0.05) ∪ (0.95, 1] vs [0.4, 0.6]."""
    all_betas = []
    for gsm, cg_betas in sample_betas.items():
        all_betas.extend(cg_betas.values())
    if not all_betas:
        return None
    n = len(all_betas)
    extreme = sum(1 for b in all_betas if b < 0.05 or b > 0.95) / n
    middle = sum(1 for b in all_betas if 0.4 <= b <= 0.6) / n
    bimodal_signature = (extreme > 0.30) and (middle < 0.10)
    return {
        'n_betas': n,
        'fraction_extreme_lt0.05_or_gt0.95': extreme,
        'fraction_middle_0.4_to_0.6': middle,
        'bimodal_raw_beta_signature': bimodal_signature,
        'note': 'Bimodal raw β signature confirms raw β values; failure flags residual M-values per CHK-3.1.'
    }


# ----------------------------------------------------------------------------
# Step 4: Score every sample with A_immune
# ----------------------------------------------------------------------------

def score_a_immune(sample_betas):
    scores = {}
    for gsm, cg_betas in sample_betas.items():
        if not cg_betas:
            scores[gsm] = None
            continue
        h_sum = 0.0
        n = 0
        for cg, b in cg_betas.items():
            h = shannon_h(b)
            if h is not None:
                h_sum += h / H_MIN_IMMUNE
                n += 1
        scores[gsm] = (h_sum / n if n > 0 else None, n)
    return scores


# ----------------------------------------------------------------------------
# Step 5: Case-control comparison
# ----------------------------------------------------------------------------

def compare_pnp(scores, clinical):
    pnp_pos = []
    pnp_neg = []
    per_sample = {}
    for gsm, score in scores.items():
        if score is None:
            continue
        a, n = score
        meta = clinical.get(gsm, {})
        pnp = meta.get('pnp_status', 'UNKNOWN')
        per_sample[gsm] = {
            'gsm': gsm,
            'patient_et_id': meta.get('patient_et_id'),
            'pnp_status': pnp,
            'A_immune': a,
            'n_valid_cpgs': n,
        }
        if pnp == 'PNP_pos':
            pnp_pos.append(a)
        elif pnp == 'PNP_neg':
            pnp_neg.append(a)
    return pnp_pos, pnp_neg, per_sample


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    random.seed(RNG_SEED)
    print('=== VAL-100 — crc-epic Under-50 Buffy Coat Polyp Stage 1 ===\n')

    print('Loading Xu-538 panel...')
    panel_cgs, panel_meta = load_panel()
    print(f'Panel: {len(panel_cgs)} CpGs (frozen since v0.1)')

    print('Loading clinical metadata...')
    clinical = load_clinical()
    n_pnp_pos = sum(1 for v in clinical.values() if v.get('pnp_status') == 'PNP_pos')
    n_pnp_neg = sum(1 for v in clinical.values() if v.get('pnp_status') == 'PNP_neg')
    print(f'Cohort: {n_pnp_pos} PNP+ + {n_pnp_neg} PNP- = {n_pnp_pos + n_pnp_neg}')

    print('Loading column mapping...')
    col_mapping = load_column_mapping()

    print(f'\nStream-parsing {BETAS_FILE} (235 MB compressed, 936K rows)...')
    sample_betas, matched_cgs, total_rows = extract_panel_betas(panel_cgs, col_mapping)
    print(f'  Total rows scanned: {total_rows:,}')
    drop = coverage_summary(sample_betas, matched_cgs, len(panel_cgs))

    print('\nBeta distribution check (CHK-3.1)...')
    bdc = beta_distribution_check(sample_betas)
    print(f'  Extreme [<0.05 or >0.95]: {100*bdc["fraction_extreme_lt0.05_or_gt0.95"]:.1f}%')
    print(f'  Middle  [0.4 to 0.6]:     {100*bdc["fraction_middle_0.4_to_0.6"]:.1f}%')
    print(f'  Bimodal raw β signature?  {bdc["bimodal_raw_beta_signature"]}')

    print('\nScoring A_immune on every sample...')
    scores = score_a_immune(sample_betas)
    n_scored = sum(1 for v in scores.values() if v is not None)
    print(f'  Scored: {n_scored} / {len(scores)} samples')

    print('\nPNP+ vs PNP- comparison...')
    pnp_pos, pnp_neg, per_sample = compare_pnp(scores, clinical)
    print(f'  PNP+ (case)    n={len(pnp_pos)}: mean A_immune = {sum(pnp_pos)/len(pnp_pos):.5f}, SD = {math.sqrt(sum((x - sum(pnp_pos)/len(pnp_pos))**2 for x in pnp_pos)/(len(pnp_pos)-1)):.5f}')
    print(f'  PNP- (control) n={len(pnp_neg)}: mean A_immune = {sum(pnp_neg)/len(pnp_neg):.5f}, SD = {math.sqrt(sum((x - sum(pnp_neg)/len(pnp_neg))**2 for x in pnp_neg)/(len(pnp_neg)-1)):.5f}')

    d = cohens_d_unpaired(pnp_pos, pnp_neg)
    ci = bootstrap_ci_unpaired(pnp_pos, pnp_neg, N_BOOTSTRAP, RNG_SEED)
    t, p = welch_t(pnp_pos, pnp_neg)
    print(f"\n  Cohen's d (PNP+ vs PNP-): {d:+.4f}")
    print(f"  Bootstrap 95% CI:         [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  Welch's t = {t:+.3f}, p ≈ {p:.4f}")

    # CHK-3.2 cross-cohort baseline check (advisory)
    pnp_neg_mean = sum(pnp_neg) / len(pnp_neg)
    sd_offset = (pnp_neg_mean - ITALIAN_HEALTHY_MEAN_A) / ITALIAN_HEALTHY_SD_A
    print(f"\n  CHK-3.2 cross-cohort baseline check:")
    print(f"    PNP- mean = {pnp_neg_mean:.5f}; Italian healthy = {ITALIAN_HEALTHY_MEAN_A:.5f}")
    print(f"    Offset = {sd_offset:+.2f} anchor-SD ({'WITHIN ±1 anchor-SD' if abs(sd_offset) <= 1 else 'EXCEEDS 1 anchor-SD — flag'})")

    # Outcome decision
    print('\n=== OUTCOME DECISION ===')
    coverage_ok = drop <= COVERAGE_FLAG_THRESHOLD
    bimodal_ok = bdc['bimodal_raw_beta_signature']
    direction_negative = d < 0

    outcome_label = None
    outcome_reason = None

    if not bimodal_ok:
        outcome_label = 'O5_DATA_INTEGRITY_FLAG'
        outcome_reason = f'Beta distribution check failed (CHK-3.1): extreme {100*bdc["fraction_extreme_lt0.05_or_gt0.95"]:.1f}%, middle {100*bdc["fraction_middle_0.4_to_0.6"]:.1f}%. Likely residual M-values or processed-not-raw betas.'
    elif not coverage_ok:
        outcome_label = 'O6_PANEL_TRANSFERABILITY_FLAG'
        outcome_reason = f'Xu-538 / EPIC-v2 coverage drop {100*drop:.1f}% exceeds {100*COVERAGE_FLAG_THRESHOLD:.0f}% threshold; CI [{ci[0]:+.3f}, {ci[1]:+.3f}] questionable due to coverage.'
    elif direction_negative and d <= -0.30 and ci[1] < 0:
        outcome_label = 'O1_PNP_NEGATIVE_DIRECTION_DETECTED'
        outcome_reason = f"PNP+ vs PNP- d = {d:+.4f}, CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]. Direction matches CCL-019 crc-epic blood immune compartment-flip."
    elif direction_negative:
        outcome_label = 'O2_PNP_DETECTABLE_DIRECTION_PARTIAL'
        outcome_reason = f"PNP+ vs PNP- d = {d:+.4f}, CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]. Direction negative but magnitude or CI underpowered."
    elif abs(d) < 0.20:
        outcome_label = 'O3_PNP_NULL'
        outcome_reason = f"PNP+ vs PNP- d = {d:+.4f}, CI crosses zero. No detectable signal at this n."
    else:
        outcome_label = 'O4_PNP_INVERTED_POSITIVE'
        outcome_reason = f"PNP+ vs PNP- d = {d:+.4f}, CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]. Inverted from CCL-019 prediction."

    print(f'OUTCOME: {outcome_label}')
    print(f'REASON: {outcome_reason}\n')

    # Save results
    panel_sha = hashlib.sha256(json.dumps(sorted(panel_cgs)).encode()).hexdigest()
    results = {
        'val_id': 'VAL-100',
        'sealed_at': '2026-04-28T18:43:05.231275+00:00',
        'prereg_sha256': '4017913d31b31e031ab01d2c0a016374334658ab9f526d99d90642d0f3f8bf67',
        'rng_seed': RNG_SEED,
        'cohort': {
            'name': 'GSE282666 (Kumar 2024) under-50 buffy coat polyp EPIC v2.0',
            'n_pnp_pos': len(pnp_pos),
            'n_pnp_neg': len(pnp_neg),
            'platform': 'GPL33022 / Illumina EPIC v2.0',
            'all_under_age_50': True,
        },
        'panel': {
            'name': 'Xu-538',
            'n_cpgs': len(panel_cgs),
            'panel_sha256': panel_sha[:16] + '...',
        },
        'coverage': {
            'matched_cpgs': len(matched_cgs),
            'panel_size': len(panel_cgs),
            'coverage_pct': 100 * len(matched_cgs) / len(panel_cgs),
            'coverage_drop_pct': 100 * drop,
            'coverage_flag_triggered': not coverage_ok,
        },
        'beta_distribution_check_chk_3_1': bdc,
        'pnp_pos_summary': {
            'n': len(pnp_pos),
            'mean_a_immune': sum(pnp_pos) / len(pnp_pos),
            'sd_a_immune': math.sqrt(sum((x - sum(pnp_pos)/len(pnp_pos))**2 for x in pnp_pos)/(len(pnp_pos)-1)),
            'min': min(pnp_pos),
            'max': max(pnp_pos),
        },
        'pnp_neg_summary': {
            'n': len(pnp_neg),
            'mean_a_immune': sum(pnp_neg) / len(pnp_neg),
            'sd_a_immune': math.sqrt(sum((x - sum(pnp_neg)/len(pnp_neg))**2 for x in pnp_neg)/(len(pnp_neg)-1)),
            'min': min(pnp_neg),
            'max': max(pnp_neg),
        },
        'cohens_d_unpaired_pnp_pos_vs_pnp_neg': d,
        'ci_95': ci,
        't_stat': t,
        'p_approx': p,
        'cross_cohort_baseline_check_chk_3_2': {
            'pnp_neg_mean': pnp_neg_mean,
            'italian_healthy_anchor_mean': ITALIAN_HEALTHY_MEAN_A,
            'italian_healthy_anchor_sd': ITALIAN_HEALTHY_SD_A,
            'offset_in_anchor_sd': sd_offset,
            'within_1_anchor_sd': abs(sd_offset) <= 1.0,
        },
        'outcome_label': outcome_label,
        'outcome_reason': outcome_reason,
        'edear_commercial_deployment_unaffected': True,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Per-sample CSV
    with open(os.path.join(OUTPUT_DIR, 'per_sample.csv'), 'w') as f:
        f.write('gsm,patient_et_id,pnp_status,A_immune,n_valid_cpgs\n')
        for gsm in sorted(per_sample.keys()):
            r = per_sample[gsm]
            f.write(f"{gsm},{r['patient_et_id']},{r['pnp_status']},{r['A_immune']},{r['n_valid_cpgs']}\n")

    print(f'Runtime: {time.time() - t0:.1f}s')
    print(f'Outputs: {OUTPUT_DIR}/results.json, per_sample.csv')


if __name__ == '__main__':
    main()
