#!/usr/bin/env python3
"""
STEP 7 ORCHESTRATOR — Run all Step 7 validation checks in order.

After the production MCMC completes and merge_iamatlas_v0_1.py produces
IAMAtlas_v0_1.csv, run this script to validate the matrix end-to-end.

Sequence:
  Check A (exit gates)               — required, 30 sec
  Check B (predictive validation)    — required, 1 min
  Check C (AD cohort scoring)        — requires aibl/addn beta + manifest, 30 sec
  Check D (breast pre-dx scoring)    — requires VAL047 sample CSVs, 1 min
  Check E (GSE130748 trajectory)     — requires methylprep + IDAT extract, 10-30 min
  Check F (sex + age stratification) — depends on Check C/D output, 30 sec

Usage:
  cd ~/IAMPerformance
  python3 step_7_run_all.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


CHECKS = [
    ("A", "step_7_check_a_exit_gates.py", []),
    ("B", "step_7_check_b_predictive_validation.py", []),
    ("C", "step_7_check_c_ad_cohort_scoring.py", []),
    ("D", "step_7_check_d_breast_iam_scoring.py", []),
    # Check E is heavy — Heath runs separately when ready
    # ("E", "step_7_check_e_gse130748_trajectory.py", ["--idat_dir", "GSE130748_RAW"]),
    ("F", "step_7_check_f_sex_age_stratified.py", []),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Checks to skip, e.g. --skip C E")
    parser.add_argument("--include_e", action="store_true", help="Include Check E (GSE130748, slow)")
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    print("=" * 72)
    print("STEP 7 — Running validation suite")
    print("=" * 72)

    # Optionally include Check E
    checks_to_run = list(CHECKS)
    if args.include_e:
        checks_to_run.append(
            ("E", "step_7_check_e_gse130748_trajectory.py",
             ["--idat_dir", "GSE130748_RAW",
              "--series_matrix", "GSE130748_series_matrix.txt.gz"])
        )

    results = []
    for tag, script, extra_args in checks_to_run:
        if tag in args.skip:
            print(f"\n[SKIP] Check {tag}")
            continue
        script_path = here / script
        if not script_path.exists():
            print(f"\n[ERROR] Check {tag} script not found: {script_path}")
            results.append((tag, False, "missing"))
            continue
        print(f"\n{'#'*72}")
        print(f"# CHECK {tag}: {script}")
        print(f"{'#'*72}")
        cmd = ["python3", str(script_path)] + extra_args
        try:
            r = subprocess.run(cmd, check=False)
            ok = (r.returncode == 0)
            results.append((tag, ok, "complete" if ok else f"exit {r.returncode}"))
        except Exception as e:
            results.append((tag, False, str(e)))
            print(f"[CRASHED] {e}")

    print(f"\n{'='*72}\nSTEP 7 SUMMARY\n{'='*72}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"Passed: {n_pass}/{len(results)}")
    for tag, ok, reason in results:
        sym = "✓" if ok else "✗"
        print(f"  [{sym}] Check {tag}: {reason}")

    if n_fail == 0:
        print("\n✓ All checks passed. IAMAtlas v0.1 validation complete.")
        print("Next: review per-check report .md files, then proceed to Step 8 (atlas vault freeze).")
        sys.exit(0)
    else:
        print(f"\n✗ {n_fail} check(s) failed — review individual reports.")
        sys.exit(1)


if __name__ == "__main__":
    main()
