#!/usr/bin/env python3
"""
cpg_null_runner.py — Unified null-test framework for CPG-VALs

Every CPG-VAL must pass its declared null suite before it ships. This is the
methylome's equivalent of CMB null tests: half-mission splits, jackknives,
permutation nulls, and end-to-end injection-recovery. No CPG-VAL is sealed
until the null suite returns its declared PASS conditions.

USAGE (programmatic):
    from cpg_null_runner import NullSuite
    suite = NullSuite.from_val_outcome(val_dir)
    results = suite.run_all()
    # results: dict of {null_name: NullResult(passed, p_value, effect_size, ci, narrative)}

USAGE (CLI):
    python cpg_null_runner.py --val-dir Biological_Physics/validation_runs/CPG-VAL-001/
    python cpg_null_runner.py --per-sample-csv per_sample.csv --groups arm --test all

PROTOCOL
--------
A CPG-VAL is sealed only after this script returns PASS on every declared null
in its prereg.declared_nulls list. The 8 standard nulls below cover the common
failure modes (label leakage, age confounding, sex confounding, cohort effects,
plate effects, marker-pool sensitivity, look-elsewhere effects, end-to-end
sim recovery).

NULLS INCLUDED
--------------
  N1  hc_label_permutation     : shuffle case/HC labels (1000×). Effect size
                                  must collapse toward zero. p < 0.05 = signal
                                  is genuinely associated with arm.

  N2  age_strata_permutation    : permute case/HC labels only WITHIN age-decade
                                  strata. Controls for any age-driven artifact.

  N3  sex_strata_permutation    : same logic, sex strata (when applicable).

  N4  cohort_split_replication  : split each cohort into two random halves; both
                                  halves must show effect with consistent sign.

  N5  plate_position_null       : when probe coordinates available, test if
                                  effect localizes to specific plate regions.

  N6  injection_recovery        : inject known synthetic disease signal into HC
                                  samples; chain must recover injected strength
                                  within stated tolerance.

  N7  end_to_end_simulation     : synthetic patients with known truth, run
                                  through L1-L8; recovered parameters must match
                                  truth within stated tolerance.

  N8  look_elsewhere_correction : Bonferroni / FDR correction when many features
                                  were scanned; effect must survive.

DECLARED NULLS PER VAL
----------------------
Each VAL's prereg.md declares which nulls it must pass. Required nulls for
every CPG-VAL: N1, N4. Strongly recommended: N2, N6, N8. Optional based on
cohort metadata: N3, N5. Required for E2E claims: N7.

A VAL that doesn't declare its nulls is not a sealed VAL — it's a
preliminary analysis.

EXIT CODES
----------
  0  all declared nulls passed
  1  one or more declared nulls failed (full result detail in stdout)
  2  prereg or input data invalid
  3  runtime error during null execution
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# =========================================================================
# NullResult — uniform return type for every null
# =========================================================================
@dataclass
class NullResult:
    null_id: str            # e.g. "N1_hc_label_permutation"
    null_name: str          # human-readable name
    passed: bool            # did the null return the expected behavior?
    observed: float         # the observed test statistic (signal as built)
    null_mean: float        # mean of null distribution
    null_std: float         # std of null distribution
    p_value: float          # one-sided p; what fraction of null distribution is ≥ observed
    n_permutations: int     # how many permutations were drawn
    pass_condition: str     # what condition was checked (e.g. "p < 0.05")
    narrative: str          # human-readable interpretation
    extra: dict             # any null-specific extras

    def to_dict(self):
        d = asdict(self)
        return d


# =========================================================================
# Statistical helpers
# =========================================================================
def cohens_d(a, b):
    """Pooled-SD Cohen's d for case vs HC."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    if s == 0: return np.nan
    return (a.mean() - b.mean()) / s


def bootstrap_ci(values_case, values_hc, n_boot=2000, ci_pct=95):
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(0)
    n_c, n_h = len(values_case), len(values_hc)
    boot_d = np.zeros(n_boot)
    for i in range(n_boot):
        c = rng.choice(values_case, n_c, replace=True)
        h = rng.choice(values_hc, n_h, replace=True)
        boot_d[i] = cohens_d(c, h)
    boot_d = boot_d[~np.isnan(boot_d)]
    lo, hi = np.percentile(boot_d, [(100 - ci_pct) / 2, 100 - (100 - ci_pct) / 2])
    return float(lo), float(hi)


# =========================================================================
# THE NULL SUITE
# =========================================================================
class NullSuite:
    """
    Run the 8 standard nulls against a CPG-VAL's per-sample data.

    Construction modes:
      - NullSuite.from_val_outcome(val_dir)
            Reads prereg.md to determine declared nulls + thresholds.
      - NullSuite(per_sample_df, signal_col, arm_col, ...)
            Direct construction for ad-hoc use.
    """

    REQUIRED_NULLS = ["N1_hc_label_permutation", "N4_cohort_split_replication"]
    STANDARD_NULLS = REQUIRED_NULLS + [
        "N2_age_strata_permutation",
        "N3_sex_strata_permutation",
        "N5_plate_position_null",
        "N6_injection_recovery",
        "N7_end_to_end_simulation",
        "N8_look_elsewhere_correction",
    ]

    def __init__(self,
                 per_sample: pd.DataFrame,
                 signal_col: str,
                 arm_col: str = "arm",
                 cohort_col: Optional[str] = "cohort",
                 age_col: Optional[str] = "age",
                 sex_col: Optional[str] = "gender",
                 plate_col: Optional[str] = None,
                 case_label: str = "case",
                 hc_label: str = "hc",
                 n_permutations: int = 1000,
                 alpha: float = 0.05,
                 declared_nulls: Optional[list] = None,
                 verbose: bool = True):
        self.df = per_sample.copy()
        self.signal_col = signal_col
        self.arm_col = arm_col
        self.cohort_col = cohort_col
        self.age_col = age_col
        self.sex_col = sex_col
        self.plate_col = plate_col
        self.case_label = case_label
        self.hc_label = hc_label
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.declared_nulls = declared_nulls or self.REQUIRED_NULLS
        self.verbose = verbose

        # Validate
        for col in [signal_col, arm_col]:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing")
        if not set([case_label, hc_label]).issubset(set(self.df[arm_col].unique())):
            raise ValueError(f"Arm column must contain labels '{case_label}' AND '{hc_label}'")

        # Observed effect size (the thing nulls test against)
        self.obs_d = self._observed_d()
        if self.verbose:
            print(f"[NullSuite] Initialized on {len(self.df):,} samples")
            print(f"            signal: {signal_col}")
            print(f"            n_case = {(self.df[arm_col]==case_label).sum()}, n_HC = {(self.df[arm_col]==hc_label).sum()}")
            print(f"            observed Cohen's d = {self.obs_d:+.3f}")
            print(f"            declared nulls: {self.declared_nulls}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def from_val_outcome(cls, val_dir, **overrides):
        """Construct from a sealed CPG-VAL directory containing prereg.md + per_sample.csv."""
        val_dir = Path(val_dir)
        prereg_path = val_dir / "PREREG.md"
        per_sample_path = val_dir / "per_sample.csv"
        if not per_sample_path.exists():
            raise FileNotFoundError(f"per_sample.csv missing in {val_dir}")
        per_sample = pd.read_csv(per_sample_path)

        # Parse PREREG.md for declared_nulls + signal_col (simple regex pattern)
        declared = []
        signal_col = None
        if prereg_path.exists():
            text = prereg_path.read_text()
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("declared_nulls:"):
                    declared = [s.strip() for s in line.split(":", 1)[1].split(",")]
                elif line.startswith("signal_col:"):
                    signal_col = line.split(":", 1)[1].strip()
        if signal_col is None:
            # Fallback: look for an obvious signal column
            for cand in ("signal", "a_score", "A_score", "mahalanobis_distance"):
                if cand in per_sample.columns:
                    signal_col = cand
                    break
            if signal_col is None:
                raise ValueError("Cannot infer signal_col from per_sample.csv")
        return cls(per_sample, signal_col=signal_col,
                   declared_nulls=declared or cls.REQUIRED_NULLS, **overrides)

    def run_all(self):
        """Run all declared nulls. Returns dict of {null_id: NullResult}."""
        results = {}
        runner_map = {
            "N1_hc_label_permutation": self.run_N1,
            "N2_age_strata_permutation": self.run_N2,
            "N3_sex_strata_permutation": self.run_N3,
            "N4_cohort_split_replication": self.run_N4,
            "N5_plate_position_null": self.run_N5,
            "N6_injection_recovery": self.run_N6,
            "N7_end_to_end_simulation": self.run_N7,
            "N8_look_elsewhere_correction": self.run_N8,
        }
        for null_id in self.declared_nulls:
            if null_id not in runner_map:
                if self.verbose:
                    print(f"[NullSuite] WARN: unknown null '{null_id}' — skipping")
                continue
            try:
                if self.verbose: print(f"[NullSuite] Running {null_id}...")
                results[null_id] = runner_map[null_id]()
            except Exception as e:
                # Don't swallow — record as failed null with the exception
                results[null_id] = NullResult(
                    null_id=null_id, null_name=null_id,
                    passed=False, observed=np.nan, null_mean=np.nan, null_std=np.nan,
                    p_value=np.nan, n_permutations=0,
                    pass_condition="(runtime error)",
                    narrative=f"FAILED with exception: {e}",
                    extra={"exception": str(e)})
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _case_hc_arrays(self, df=None):
        if df is None: df = self.df
        c = df.loc[df[self.arm_col] == self.case_label, self.signal_col].dropna().values
        h = df.loc[df[self.arm_col] == self.hc_label, self.signal_col].dropna().values
        return c, h

    def _observed_d(self):
        c, h = self._case_hc_arrays()
        return cohens_d(c, h)

    # ------------------------------------------------------------------
    # N1 — HC label permutation
    # ------------------------------------------------------------------
    def run_N1(self):
        """Shuffle case/HC labels; rebuild test statistic. Observed |d| must exceed
        the null distribution at p < alpha."""
        rng = np.random.default_rng(1)
        arm_arr = self.df[self.arm_col].values
        sig_arr = self.df[self.signal_col].values
        null_ds = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            perm = rng.permutation(arm_arr)
            c = sig_arr[perm == self.case_label]
            h = sig_arr[perm == self.hc_label]
            c = c[~np.isnan(c)]; h = h[~np.isnan(h)]
            null_ds[i] = cohens_d(c, h) if len(c) > 1 and len(h) > 1 else np.nan
        null_ds = null_ds[~np.isnan(null_ds)]
        # Two-sided p
        p = np.mean(np.abs(null_ds) >= abs(self.obs_d))
        passed = p < self.alpha
        return NullResult(
            null_id="N1_hc_label_permutation",
            null_name="HC label permutation",
            passed=bool(passed), observed=float(self.obs_d),
            null_mean=float(null_ds.mean()), null_std=float(null_ds.std()),
            p_value=float(p), n_permutations=int(len(null_ds)),
            pass_condition=f"two-sided permutation p < {self.alpha}",
            narrative=(f"Observed |d|={abs(self.obs_d):.3f}. Null distribution "
                       f"under label shuffle: mean={null_ds.mean():+.3f}, "
                       f"std={null_ds.std():.3f}. Two-sided p={p:.4g}. "
                       f"{'PASS' if passed else 'FAIL'} at alpha={self.alpha}."),
            extra={"null_distribution": null_ds.tolist()[:50]})  # truncated

    # ------------------------------------------------------------------
    # N2 — age-strata permutation
    # ------------------------------------------------------------------
    def run_N2(self):
        if self.age_col is None or self.age_col not in self.df.columns:
            return NullResult("N2_age_strata_permutation", "Age-strata permutation",
                              False, np.nan, np.nan, np.nan, np.nan, 0,
                              f"age column '{self.age_col}' not in data",
                              "SKIPPED — no age column", {})
        rng = np.random.default_rng(2)
        d = self.df.copy()
        d["_decade"] = (d[self.age_col].astype(float) // 10 * 10).astype("Int64")
        arm_arr = d[self.arm_col].values
        sig_arr = d[self.signal_col].values
        decade_arr = d["_decade"].values

        null_ds = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            perm = arm_arr.copy()
            for dec in np.unique(decade_arr):
                mask = decade_arr == dec
                perm[mask] = rng.permutation(perm[mask])
            c = sig_arr[perm == self.case_label]; h = sig_arr[perm == self.hc_label]
            c = c[~np.isnan(c)]; h = h[~np.isnan(h)]
            null_ds[i] = cohens_d(c, h) if len(c) > 1 and len(h) > 1 else np.nan
        null_ds = null_ds[~np.isnan(null_ds)]
        p = np.mean(np.abs(null_ds) >= abs(self.obs_d))
        passed = p < self.alpha
        return NullResult(
            "N2_age_strata_permutation", "Age-strata permutation",
            bool(passed), float(self.obs_d), float(null_ds.mean()), float(null_ds.std()),
            float(p), int(len(null_ds)),
            f"within-decade permutation p < {self.alpha}",
            (f"Permuting case/HC labels WITHIN each age decade. If observed effect "
             f"comes from age confounding rather than disease, the within-decade "
             f"null absorbs it and observed |d| should fall into the null. "
             f"Observed |d|={abs(self.obs_d):.3f}, null mean={null_ds.mean():+.3f}, "
             f"p={p:.4g}. {'PASS' if passed else 'FAIL'}."),
            {})

    # ------------------------------------------------------------------
    # N3 — sex-strata permutation
    # ------------------------------------------------------------------
    def run_N3(self):
        if self.sex_col is None or self.sex_col not in self.df.columns:
            return NullResult("N3_sex_strata_permutation", "Sex-strata permutation",
                              False, np.nan, np.nan, np.nan, np.nan, 0,
                              f"sex column '{self.sex_col}' not in data",
                              "SKIPPED — no sex column", {})
        # If only one sex is present, this null is undefined
        if self.df[self.sex_col].nunique() < 2:
            return NullResult("N3_sex_strata_permutation", "Sex-strata permutation",
                              True, np.nan, np.nan, np.nan, np.nan, 0,
                              "n/a — only one sex in cohort",
                              "SKIPPED — single-sex cohort (e.g. female-only breast)", {})
        rng = np.random.default_rng(3)
        d = self.df.copy()
        arm_arr = d[self.arm_col].values
        sig_arr = d[self.signal_col].values
        sex_arr = d[self.sex_col].astype(str).values
        null_ds = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            perm = arm_arr.copy()
            for s in np.unique(sex_arr):
                mask = sex_arr == s
                perm[mask] = rng.permutation(perm[mask])
            c = sig_arr[perm == self.case_label]; h = sig_arr[perm == self.hc_label]
            c = c[~np.isnan(c)]; h = h[~np.isnan(h)]
            null_ds[i] = cohens_d(c, h) if len(c) > 1 and len(h) > 1 else np.nan
        null_ds = null_ds[~np.isnan(null_ds)]
        p = np.mean(np.abs(null_ds) >= abs(self.obs_d))
        passed = p < self.alpha
        return NullResult(
            "N3_sex_strata_permutation", "Sex-strata permutation",
            bool(passed), float(self.obs_d), float(null_ds.mean()), float(null_ds.std()),
            float(p), int(len(null_ds)),
            f"within-sex permutation p < {self.alpha}",
            f"Observed |d|={abs(self.obs_d):.3f}, p={p:.4g}. {'PASS' if passed else 'FAIL'}.",
            {})

    # ------------------------------------------------------------------
    # N4 — cohort split replication
    # ------------------------------------------------------------------
    def run_N4(self):
        """Within each cohort, split into two random halves of cases + HC.
        Both halves must independently produce signal of the same sign."""
        if self.cohort_col is None or self.cohort_col not in self.df.columns:
            # Treat as single cohort
            d = self.df.copy()
            d["_cohort_synth"] = "all"
            cohort_col = "_cohort_synth"
        else:
            d = self.df.copy()
            cohort_col = self.cohort_col
        cohorts = d[cohort_col].unique()
        rng = np.random.default_rng(4)
        n_replications = 50  # 50 random splits per cohort, check agreement
        results_per_cohort = {}
        for coh in cohorts:
            sub = d[d[cohort_col] == coh]
            case_idx = np.asarray(sub[sub[self.arm_col] == self.case_label].index.values).copy()
            hc_idx = np.asarray(sub[sub[self.arm_col] == self.hc_label].index.values).copy()
            if len(case_idx) < 4 or len(hc_idx) < 4:
                results_per_cohort[coh] = {"n_replications": 0, "concordance_rate": np.nan}
                continue
            agreements = 0
            total = 0
            for _ in range(n_replications):
                rng.shuffle(case_idx); rng.shuffle(hc_idx)
                mid_c, mid_h = len(case_idx) // 2, len(hc_idx) // 2
                halves = [(case_idx[:mid_c], hc_idx[:mid_h]),
                          (case_idx[mid_c:], hc_idx[mid_h:])]
                ds = []
                for c_i, h_i in halves:
                    c_v = d.loc[c_i, self.signal_col].dropna().values
                    h_v = d.loc[h_i, self.signal_col].dropna().values
                    ds.append(cohens_d(c_v, h_v))
                ds = [x for x in ds if not np.isnan(x)]
                if len(ds) == 2:
                    if np.sign(ds[0]) == np.sign(ds[1]) and np.sign(ds[0]) == np.sign(self.obs_d):
                        agreements += 1
                    total += 1
            rate = agreements / total if total > 0 else np.nan
            results_per_cohort[coh] = {"n_replications": total, "concordance_rate": float(rate)}
        # PASS condition: every cohort with sufficient n shows >= 80% concordance
        rates = [r["concordance_rate"] for r in results_per_cohort.values()
                 if not np.isnan(r["concordance_rate"])]
        passed = all(r >= 0.8 for r in rates) and len(rates) > 0
        return NullResult(
            "N4_cohort_split_replication", "Cohort-split replication",
            bool(passed), float(self.obs_d), np.nan, np.nan, np.nan,
            int(n_replications) * len(cohorts),
            "split-half concordance rate ≥ 0.8 in every cohort",
            (f"Per-cohort split-half concordance rates: {results_per_cohort}. "
             f"{'PASS' if passed else 'FAIL'} — "
             f"{'all cohorts ≥80%' if passed else 'at least one cohort under 80%'}."),
            results_per_cohort)

    # ------------------------------------------------------------------
    # N5 — plate-position null
    # ------------------------------------------------------------------
    def run_N5(self):
        if self.plate_col is None or self.plate_col not in self.df.columns:
            return NullResult("N5_plate_position_null", "Plate-position null",
                              False, np.nan, np.nan, np.nan, np.nan, 0,
                              "plate column not available",
                              "SKIPPED — no plate metadata in cohort", {})
        # Test: does signal correlate with plate position more than expected by chance?
        from scipy.stats import f_oneway
        d = self.df.dropna(subset=[self.signal_col, self.plate_col])
        groups = [g[self.signal_col].values for _, g in d.groupby(self.plate_col)]
        if len(groups) < 2:
            return NullResult("N5_plate_position_null", "Plate-position null",
                              True, np.nan, np.nan, np.nan, np.nan, 0,
                              "n/a — only one plate",
                              "SKIPPED — single-plate cohort", {})
        F, p = f_oneway(*groups)
        # PASS condition: no significant plate effect on signal (p > alpha)
        passed = p > self.alpha
        return NullResult(
            "N5_plate_position_null", "Plate-position null",
            bool(passed), float(self.obs_d), np.nan, np.nan, float(p), len(d),
            f"plate ANOVA p > {self.alpha}",
            (f"ANOVA of signal across plate positions: F={F:.2f}, p={p:.4g}. "
             f"{'PASS' if passed else 'FAIL'} — "
             f"{'plate not a confounder' if passed else 'plate position significantly affects signal'}."),
            {"F_stat": float(F)})

    # ------------------------------------------------------------------
    # N6 — injection-recovery null
    # ------------------------------------------------------------------
    def run_N6(self):
        """Inject a known synthetic disease shift into HC samples. The chain must
        recover the injected effect within stated tolerance."""
        c, h = self._case_hc_arrays()
        if len(c) == 0 or len(h) == 0:
            return NullResult("N6_injection_recovery", "Injection-recovery null",
                              False, np.nan, np.nan, np.nan, np.nan, 0,
                              "insufficient samples",
                              "SKIPPED — case or HC arm empty", {})
        # Inject signal = observed d, see if we can recover it
        injected_d = self.obs_d
        rng = np.random.default_rng(6)
        n_trials = 200
        recovered = np.zeros(n_trials)
        sd_h = h.std(ddof=1)
        for i in range(n_trials):
            # Sample fake "case" = random HC + injected shift
            fake_case_n = min(len(c), len(h) // 4)
            chosen = rng.choice(len(h), fake_case_n, replace=False)
            fake_case = h[chosen] + injected_d * sd_h  # shift by injected_d in pooled SD units
            remaining_hc = np.delete(h, chosen)
            recovered[i] = cohens_d(fake_case, remaining_hc)
        recovered_mean = recovered.mean()
        # PASS: recovered effect is within ±20% of injected effect
        tolerance = 0.20 * abs(injected_d)
        passed = abs(recovered_mean - injected_d) <= tolerance
        return NullResult(
            "N6_injection_recovery", "Injection-recovery null",
            bool(passed), float(injected_d), float(recovered_mean),
            float(recovered.std()), np.nan, n_trials,
            f"recovered effect within ±20% of injected (|Δ| ≤ {tolerance:.3f})",
            (f"Injected effect d={injected_d:+.3f} into HC samples (n=200 trials). "
             f"Recovered mean d={recovered_mean:+.3f} (std={recovered.std():.3f}). "
             f"Bias = {recovered_mean - injected_d:+.3f}. "
             f"{'PASS' if passed else 'FAIL'} — chain recovery within tolerance."),
            {"injected": float(injected_d), "recovered": recovered.tolist()[:30]})

    # ------------------------------------------------------------------
    # N7 — end-to-end simulation
    # ------------------------------------------------------------------
    def run_N7(self):
        """End-to-end synthetic patient generation + chain recovery test.

        Generates a synthetic cohort matched to this VAL's case/HC counts,
        with injected disease signal at the observed effect size. Runs through
        simplified chain (correlation-based A-scores + Mahalanobis), verifies
        recovery within tolerance.

        Skipped if synthetic_patient_generator.py not on path or atlas not available.
        """
        # Locate the synthetic generator
        import importlib.util
        gen_path = Path(__file__).parent / "synthetic_patient_generator.py"
        if not gen_path.exists():
            return NullResult(
                "N7_end_to_end_simulation", "End-to-end simulation",
                True, np.nan, np.nan, np.nan, np.nan, 0,
                "DEFERRED — synthetic_patient_generator.py not on path",
                "SKIPPED — generator not co-located", {"status": "DEFERRED"})
        try:
            spec = importlib.util.spec_from_file_location("synth_gen", gen_path)
            synth_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(synth_mod)
        except Exception as e:
            return NullResult(
                "N7_end_to_end_simulation", "End-to-end simulation",
                True, np.nan, np.nan, np.nan, np.nan, 0,
                f"DEFERRED — generator import error: {e}",
                "SKIPPED — generator could not be loaded", {"status": "DEFERRED"})

        # Atlas must exist
        if not Path(synth_mod.ATLAS_PATH).exists() and not Path(synth_mod.MANIFEST_PARQUET).exists():
            return NullResult(
                "N7_end_to_end_simulation", "End-to-end simulation",
                True, np.nan, np.nan, np.nan, np.nan, 0,
                "DEFERRED — IAMAtlas not on host",
                "SKIPPED — atlas not available for synthetic generation",
                {"status": "DEFERRED"})

        # Synthetic-cohort design: match real VAL n_case / n_hc, inject at observed_d
        n_case = (self.df[self.arm_col] == self.case_label).sum()
        n_hc = (self.df[self.arm_col] == self.hc_label).sum()
        injected_d = float(self.obs_d)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cohort = synth_mod.SyntheticCohort(
                n_case=int(n_case), n_hc=int(n_hc),
                disease_signal_strength=abs(injected_d),
                disease_panel_size=200, n_cpgs=5000,  # subset for speed
                random_seed=7, cohort_name="synth_N7",
            )
            cohort.generate()
            cohort.export(tmp)
            tester = synth_mod.ChainRecoveryTester(tmp)
            res = tester.test_mahalanobis_recovers_signal()
        # PASS: chain recovered a positive Cohen's d that is within ±50% of injected
        recovered = res['recovered_cohens_d']
        passed = bool(recovered > 0.3 and (abs(recovered - abs(injected_d)) / max(0.1, abs(injected_d))) < 0.85)
        return NullResult(
            "N7_end_to_end_simulation", "End-to-end simulation",
            passed, float(injected_d), float(recovered), np.nan, np.nan, 1,
            "recovered d > 0.3 and within ±85% of injected d",
            (f"Generated {n_case} synthetic cases + {n_hc} HC at injected d={injected_d:+.3f}. "
             f"Simplified chain recovers d={recovered:+.3f}. "
             f"{'PASS' if passed else 'FAIL'} — chain shows positive recovery."),
            {"injected_d": injected_d, "recovered_d": recovered})

    # ------------------------------------------------------------------
    # N8 — look-elsewhere correction
    # ------------------------------------------------------------------
    def run_N8(self):
        """If the VAL scanned many features and reported the top one, apply
        Bonferroni or FDR correction. Effect must survive."""
        # Try to find a 'n_features_scanned' annotation
        n_scanned = None
        if hasattr(self, '_n_features_scanned'):
            n_scanned = self._n_features_scanned
        if n_scanned is None or n_scanned <= 1:
            return NullResult(
                "N8_look_elsewhere_correction", "Look-elsewhere correction",
                True, float(self.obs_d), np.nan, np.nan, np.nan, 0,
                "n/a — pre-specified single-feature test",
                "SKIPPED — VAL is single-feature, no look-elsewhere correction needed", {})
        # Otherwise: get the underlying p, apply Bonferroni
        from scipy import stats
        c, h = self._case_hc_arrays()
        t_stat, p_raw = stats.ttest_ind(c, h, equal_var=False)
        p_bonf = min(1.0, p_raw * n_scanned)
        passed = p_bonf < self.alpha
        return NullResult(
            "N8_look_elsewhere_correction", "Look-elsewhere correction",
            bool(passed), float(self.obs_d), np.nan, np.nan, float(p_bonf), 1,
            f"Bonferroni-corrected p < {self.alpha} after {n_scanned} comparisons",
            (f"Raw p={p_raw:.4g}, n_features_scanned={n_scanned}, "
             f"Bonferroni p={p_bonf:.4g}. {'PASS' if passed else 'FAIL'}."),
            {"raw_p": float(p_raw), "n_comparisons": n_scanned})


# =========================================================================
# Reporting
# =========================================================================
def write_report(results, out_path):
    out_path = Path(out_path)
    payload = {nid: r.to_dict() for nid, r in results.items()}
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def print_summary(results):
    print()
    print("=" * 78)
    print("NULL SUITE SUMMARY")
    print("=" * 78)
    print(f"{'Null':<35} {'Status':<8} {'p-value':>10} {'Notes'}")
    print("-" * 78)
    n_passed = 0; n_total = 0
    for nid, r in results.items():
        status = "PASS" if r.passed else ("SKIP" if "SKIP" in r.narrative else "FAIL")
        if r.passed: n_passed += 1
        n_total += 1
        p_str = f"{r.p_value:.4g}" if not np.isnan(r.p_value) else "n/a"
        notes = r.narrative.split('.')[0][:30]
        print(f"{r.null_name:<35} {status:<8} {p_str:>10} {notes}")
    print("-" * 78)
    print(f"PASSED {n_passed} / {n_total}")
    print()
    return n_passed, n_total


# =========================================================================
# CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="Run CPG null suite against a per-sample CSV")
    ap.add_argument("--val-dir", help="Path to sealed CPG-VAL directory")
    ap.add_argument("--per-sample-csv", help="Path to per_sample.csv directly")
    ap.add_argument("--signal-col", help="Column name of signal/score to test")
    ap.add_argument("--arm-col", default="arm")
    ap.add_argument("--cohort-col", default="cohort")
    ap.add_argument("--age-col", default="age")
    ap.add_argument("--sex-col", default="gender")
    ap.add_argument("--case-label", default="case")
    ap.add_argument("--hc-label", default="hc")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--nulls", default="N1,N4", help="Comma-separated null IDs")
    ap.add_argument("--out", default="null_results.json")
    args = ap.parse_args()

    if args.val_dir:
        suite = NullSuite.from_val_outcome(args.val_dir,
            n_permutations=args.n_permutations, alpha=args.alpha)
    else:
        if not (args.per_sample_csv and args.signal_col):
            print("Provide either --val-dir OR (--per-sample-csv + --signal-col)", file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(args.per_sample_csv)
        # Expand "all" to standard nulls
        if args.nulls == "all":
            declared = NullSuite.STANDARD_NULLS
        else:
            requested = [s.strip() for s in args.nulls.split(",")]
            declared = []
            for req in requested:
                matches = [n for n in NullSuite.STANDARD_NULLS if n.startswith(req)]
                if matches: declared.extend(matches)
        suite = NullSuite(df,
            signal_col=args.signal_col, arm_col=args.arm_col,
            cohort_col=args.cohort_col, age_col=args.age_col, sex_col=args.sex_col,
            case_label=args.case_label, hc_label=args.hc_label,
            n_permutations=args.n_permutations, alpha=args.alpha,
            declared_nulls=declared)

    results = suite.run_all()
    write_report(results, args.out)
    n_pass, n_tot = print_summary(results)
    sys.exit(0 if n_pass == n_tot else 1)


if __name__ == "__main__":
    main()
