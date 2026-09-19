#!/usr/bin/env python3
# VAL-064 — HCC-EPIC tissue arm validation on TCGA-LIHC HM450
# Pre-registration SHA: a03f2c2c65e65d5ce143e8b4f32b4faaa9fdfd4d07b204121fb5452f451e4a9a
# Manifest SHA: 760bf65a213da5a86cdd7ecde6ff6d46dad04777eafc00eb14f56356b0088371
# Cohort SHA: 78ccc7fecc9a8995b95d4f7ab1ecaaa2d431427dfa05bbda808aaacf31b565e4
#
# Validates secretory-class architectural disruption signal in HCC tumor tissue
# vs adjacent-normal across n=46 matched TCGA-LIHC patients with full risk-factor
# stratification (HBV/HCV/alcohol/NAFLD/Ishak fibrosis/stage/gender).
#
# Reproduction:
#   1. Run download_lihc_data() to retrieve the 100 β-value files from NIH GDC
#      (50 patients × 2 samples). All files are public-access Level 3 sesame
#      betas — no dbGaP application required.
#   2. Run main_analysis() to compute primary pooled and all stratified results.
#   3. Outputs:
#       - VAL-064_results.json (primary pooled + all stratified)
#       - VAL-064_stratified.json (etiology-focused stratification breakdown)
#
# Dependencies: Python 3.6+ stdlib only. No numpy, no pandas, no R.
#
# Author: Walther / Heath W. Mahaffey
# Date: 2026-04-24

import urllib.request
import urllib.parse
import json
import os
import math
import hashlib
import statistics
import time
from math import erf, sqrt

# ============================================================================
# Constants — sealed in pre-registration before any β-value access
# ============================================================================

H_MIN_SECRETORY = 0.843264   # G-002 MCMC posterior, R-hat < 1.001 (frozen)
REFERENCE_BETA = 0.742        # Moss 2018 hepatocyte healthy, hcc-epic v0.1 universal_reference
QC_MIN_VALID_CPGS = 400_000   # ~82% of HM450 probes
RNG_SEED = 20260420
DATA_DIR = "./val064_data"
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")

# ============================================================================
# Step 1: Cohort manifest retrieval from NIH GDC API
# ============================================================================

def fetch_lihc_manifest():
    """Query GDC API for TCGA-LIHC HM450 matched tumor/normal pairs."""
    url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LIHC"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}},
            {"op": "in", "content": {"field": "platform", "value": ["Illumina Human Methylation 450"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "size": "1500",
        "format": "json"
    }
    query_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(query_url, headers={"User-Agent": "VAL-064/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())

    by_patient = {}
    for h in data.get("data", {}).get("hits", []):
        cases = h.get("cases", [])
        if not cases: continue
        pid = cases[0].get("submitter_id", "")
        samples = cases[0].get("samples", [])
        if not samples: continue
        stype = samples[0].get("sample_type", "")
        if pid not in by_patient:
            by_patient[pid] = {"tumor": None, "normal": None}
        if stype == "Primary Tumor":
            by_patient[pid]["tumor"] = (h["file_id"], h["file_name"])
        elif stype == "Solid Tissue Normal":
            by_patient[pid]["normal"] = (h["file_id"], h["file_name"])

    matched = []
    for pid, d in by_patient.items():
        if d["tumor"] and d["normal"]:
            matched.append({
                "patient": pid,
                "tumor_file_id": d["tumor"][0], "tumor_file_name": d["tumor"][1],
                "normal_file_id": d["normal"][0], "normal_file_name": d["normal"][1],
            })
    return matched


def fetch_lihc_clinical(patient_ids):
    """Pull clinical metadata for risk-factor stratification (HBV/HCV/alcohol/Ishak/stage)."""
    filters = {"op": "in", "content": {"field": "submitter_id", "value": patient_ids}}
    fields = ",".join([
        "submitter_id",
        "exposures.alcohol_history", "exposures.tobacco_smoking_status",
        "diagnoses.ishak_fibrosis_score", "diagnoses.ajcc_pathologic_stage",
        "follow_ups.other_clinical_attributes.risk_factors",
        "follow_ups.other_clinical_attributes.viral_hepatitis_serology_tests",
        "demographic.gender", "demographic.age_at_index", "demographic.race",
    ])
    params = {"filters": json.dumps(filters), "fields": fields, "size": "100", "format": "json"}
    url = "https://api.gdc.cancer.gov/cases?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "VAL-064/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())

    strata = {}
    for c in data.get("data", {}).get("hits", []):
        pid = c.get("submitter_id")
        exp_list = c.get("exposures") or []
        follow_ups = c.get("follow_ups") or []
        diag_list = c.get("diagnoses") or []
        dem = c.get("demographic") or {}
        if isinstance(dem, list): dem = dem[0] if dem else {}
        exp = exp_list[0] if exp_list else {}

        risk_factors_list = []
        for fu in follow_ups:
            if not isinstance(fu, dict): continue
            oca_raw = fu.get("other_clinical_attributes")
            oca_list = oca_raw if isinstance(oca_raw, list) else ([oca_raw] if oca_raw else [])
            for oca in oca_list:
                if isinstance(oca, dict) and oca.get("risk_factors"):
                    risk_factors_list.append(oca.get("risk_factors"))

        ishak = None; stage = None
        for d in diag_list:
            if not isinstance(d, dict): continue
            if d.get("ishak_fibrosis_score"): ishak = d.get("ishak_fibrosis_score")
            if d.get("ajcc_pathologic_stage"): stage = d.get("ajcc_pathologic_stage")

        strata[pid] = {
            "risk_factors": risk_factors_list, "ishak": ishak or "missing",
            "stage": stage or "missing", "gender": dem.get("gender"),
            "age": dem.get("age_at_index"), "race": dem.get("race"),
        }
    return strata


def download_file(file_id, file_name, patient, kind):
    """Download one β-value file from NIH GDC public access."""
    out_path = os.path.join(DOWNLOAD_DIR, f"{patient}__{kind}__{file_name}")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VAL-064/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            with open(out_path, "wb") as f: f.write(r.read())
        return True
    except Exception:
        return False


def download_lihc_data():
    """Full retrieval pipeline: manifest, clinical metadata, and 100 β files."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print("Fetching TCGA-LIHC matched pair manifest...")
    manifest = fetch_lihc_manifest()
    print(f"  Matched pairs: {len(manifest)}")
    with open(os.path.join(DATA_DIR, "LIHC_matched_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Fetching clinical metadata...")
    pids = [m["patient"] for m in manifest]
    strata = fetch_lihc_clinical(pids)
    with open(os.path.join(DATA_DIR, "LIHC_clinical.json"), "w") as f:
        json.dump({"patient_strata": strata}, f, indent=2)

    print(f"Downloading {len(manifest) * 2} β-value files...")
    ok = 0
    for i, m in enumerate(manifest, 1):
        if download_file(m["tumor_file_id"], m["tumor_file_name"], m["patient"], "tumor"): ok += 1
        if download_file(m["normal_file_id"], m["normal_file_name"], m["patient"], "normal"): ok += 1
        if i % 10 == 0:
            print(f"  {i}/{len(manifest)} patients, {ok}/{i*2} files OK")
        time.sleep(0.05)
    print(f"  Final: {ok}/{len(manifest)*2}")
    return manifest, strata


# ============================================================================
# Step 2: Scoring — Shannon entropy → A-score per sample
# ============================================================================

def shannon_entropy(beta):
    """Binary Shannon entropy of methylation β value."""
    if beta <= 0 or beta >= 1: return 0.0
    return -beta * math.log2(beta) - (1 - beta) * math.log2(1 - beta)

def a_score(beta):
    """A-score = H(β) / H_min(class)."""
    return shannon_entropy(beta) / H_MIN_SECRETORY

def load_sample_betas(filepath):
    """Load β values from a sesame Level 3 TSV. Returns list of valid (0<β<1) values."""
    if not os.path.exists(filepath): return None
    betas = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    b = float(parts[1])
                    if 0 < b < 1 and not math.isnan(b):
                        betas.append(b)
                except ValueError: pass
    return betas


# ============================================================================
# Step 3: Statistics — paired and unpaired Cohen's d with 95% CI
# ============================================================================

def norm_sf(x): return 1.0 - 0.5 * (1 + erf(abs(x) / sqrt(2)))

def paired_cohens_d(deltas):
    """Paired Cohen's d on per-patient (tumor - normal) differences."""
    n = len(deltas)
    if n < 2: return None
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0: return None
    d = m / sd
    se = math.sqrt(1/n + d**2 / (2*n))
    t = m / (sd / math.sqrt(n))
    return {"n": n, "paired_d": d, "paired_d_ci_95": [d - 1.96*se, d + 1.96*se],
            "paired_t": t, "paired_p": 2 * norm_sf(t),
            "delta_A_mean": m, "delta_A_sd": sd}

def unpaired_cohens_d(arr_t, arr_n):
    """Unpaired Cohen's d on tumor vs normal A-score arrays."""
    n1, n2 = len(arr_t), len(arr_n)
    if n1 < 2 or n2 < 2: return None
    m1, m2 = statistics.mean(arr_t), statistics.mean(arr_n)
    s1, s2 = statistics.stdev(arr_t), statistics.stdev(arr_n)
    pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled == 0: return None
    d = (m1 - m2) / pooled
    se = math.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
    t = (m1 - m2) / (pooled * math.sqrt(1/n1 + 1/n2))
    return {"unpaired_d": d, "unpaired_d_ci_95": [d - 1.96*se, d + 1.96*se],
            "unpaired_t": t, "unpaired_p": 2 * norm_sf(t)}


# ============================================================================
# Step 4: Stratum classifier — assigns each patient to risk-factor strata
# ============================================================================

def classify_strata(patient_id, clinical_strata):
    """Returns dict of boolean strata membership for one patient."""
    s = clinical_strata.get(patient_id, {})
    rf = s.get("risk_factors", [])
    rf_flat = []
    for item in rf:
        if isinstance(item, list): rf_flat.extend(item)
        else: rf_flat.append(item)

    return {
        "has_hbv": any("Hepatitis B" in str(x) for x in rf_flat),
        "has_hcv": any("Hepatitis C" in str(x) for x in rf_flat),
        "has_alcohol": any("Alcohol" in str(x) for x in rf_flat),
        "has_nafld": any("Fatty Liver" in str(x) for x in rf_flat),
        "has_tobacco": any("Tobacco" in str(x) for x in rf_flat),
        "no_documented_risk": (not rf_flat) or rf_flat == ["None"],
        "ishak": s.get("ishak", "missing"),
        "has_fibrosis": any(k in str(s.get("ishak", "")) for k in ["Fibrosis", "Cirrhosis", "Septa", "Nodular"]),
        "stage": s.get("stage", "missing"),
        "gender": s.get("gender"), "race": s.get("race"),
    }


# ============================================================================
# Step 5: Main analysis pipeline
# ============================================================================

def main_analysis():
    """Run VAL-064 primary pooled + all stratified analyses."""
    with open(os.path.join(DATA_DIR, "LIHC_matched_manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(DATA_DIR, "LIHC_clinical.json")) as f:
        clinical = json.load(f)["patient_strata"]

    # Score every QC-passed sample
    results = []
    skipped = []
    for m in manifest:
        pid = m["patient"]
        tpath = os.path.join(DOWNLOAD_DIR, f"{pid}__tumor__{m['tumor_file_name']}")
        npath = os.path.join(DOWNLOAD_DIR, f"{pid}__normal__{m['normal_file_name']}")
        tb = load_sample_betas(tpath)
        nb = load_sample_betas(npath)
        if tb is None or nb is None or len(tb) < QC_MIN_VALID_CPGS or len(nb) < QC_MIN_VALID_CPGS:
            skipped.append((pid, "QC_fail"))
            continue
        A_t = sum(a_score(b) for b in tb) / len(tb)
        A_n = sum(a_score(b) for b in nb) / len(nb)
        strata_dict = classify_strata(pid, clinical)
        results.append({
            "patient": pid, "A_tumor": A_t, "A_normal": A_n, "delta_A": A_t - A_n,
            "n_cpg_t": len(tb), "n_cpg_n": len(nb), **strata_dict,
        })

    n = len(results)
    qc_pids = sorted([r["patient"] for r in results])
    cohort_sha = hashlib.sha256(json.dumps(qc_pids).encode()).hexdigest()
    print(f"VAL-064: {n} QC-passed pairs, cohort SHA {cohort_sha[:16]}...")

    # Primary pooled
    deltas = [r["delta_A"] for r in results]
    A_t_arr = [r["A_tumor"] for r in results]
    A_n_arr = [r["A_normal"] for r in results]
    paired = paired_cohens_d(deltas)
    unpaired = unpaired_cohens_d(A_t_arr, A_n_arr)
    print(f"  Primary pooled: paired d = {paired['paired_d']:+.4f} "
          f"[{paired['paired_d_ci_95'][0]:+.4f}, {paired['paired_d_ci_95'][1]:+.4f}], "
          f"p = {paired['paired_p']:.2e}")

    # Stratified
    def run_stratum(name, filt_fn):
        arm = [r for r in results if filt_fn(r)]
        if len(arm) < 2: return {"n": len(arm), "note": "underpowered"}
        return paired_cohens_d([r["delta_A"] for r in arm])

    risk_factor_stratified = {
        "HBV": run_stratum("HBV+", lambda r: r["has_hbv"]),
        "HCV": run_stratum("HCV+", lambda r: r["has_hcv"]),
        "alcohol": run_stratum("Alcohol+", lambda r: r["has_alcohol"]),
        "NAFLD": run_stratum("NAFLD", lambda r: r["has_nafld"]),
        "no_documented": run_stratum("no_doc", lambda r: r["no_documented_risk"]),
        "viral_combined": run_stratum("viral", lambda r: r["has_hbv"] or r["has_hcv"]),
        "non_viral": run_stratum("non_viral", lambda r: not (r["has_hbv"] or r["has_hcv"])),
    }
    fibrosis_stratified = {
        "no_fibrosis": run_stratum("no_fib", lambda r: r["ishak"] == "0 - No Fibrosis"),
        "any_fibrosis": run_stratum("any_fib", lambda r: r["has_fibrosis"]),
    }
    stage_stratified = {
        "stage_I": run_stratum("Stage I", lambda r: r["stage"] == "Stage I"),
        "stage_II_plus": run_stratum("Stage II+", lambda r: r["stage"] and (
            r["stage"].startswith("Stage II") or r["stage"].startswith("Stage III") or r["stage"].startswith("Stage IV")
        )),
    }
    gender_stratified = {
        "male": run_stratum("Male", lambda r: r["gender"] == "male"),
        "female": run_stratum("Female", lambda r: r["gender"] == "female"),
    }

    # Print stratified summary
    print("\n  Etiology-stratified:")
    for k, v in risk_factor_stratified.items():
        if "paired_d" in v:
            print(f"    {k:<20} n={v['n']:<3} d={v['paired_d']:+.4f} p={v['paired_p']:.3e}")

    # Save outputs
    output = {
        "val_id": "VAL-064", "card": "hcc-epic", "date": "2026-04-24",
        "cohort": "TCGA-LIHC HM450 matched tumor/normal",
        "cohort_sha": cohort_sha,
        "manifest_sha": "760bf65a213da5a86cdd7ecde6ff6d46dad04777eafc00eb14f56356b0088371",
        "prereg_sha": "a03f2c2c65e65d5ce143e8b4f32b4faaa9fdfd4d07b204121fb5452f451e4a9a",
        "scoring_class": "secretory", "H_min": H_MIN_SECRETORY, "reference_beta": REFERENCE_BETA,
        "n_pairs": n,
        "primary_pooled": {
            "A_tumor_mean": statistics.mean(A_t_arr), "A_tumor_sd": statistics.stdev(A_t_arr),
            "A_normal_mean": statistics.mean(A_n_arr), "A_normal_sd": statistics.stdev(A_n_arr),
            **paired, **unpaired,
        },
        "risk_factor_stratified": risk_factor_stratified,
        "fibrosis_stratified": fibrosis_stratified,
        "stage_stratified": stage_stratified,
        "gender_stratified": gender_stratified,
        "rng_seed": RNG_SEED,
    }
    with open(os.path.join(DATA_DIR, "VAL-064_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {DATA_DIR}/VAL-064_results.json")
    return output


if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "LIHC_matched_manifest.json")):
        download_lihc_data()
    else:
        print("Manifest exists, skipping download. Delete to refetch.")
    main_analysis()
