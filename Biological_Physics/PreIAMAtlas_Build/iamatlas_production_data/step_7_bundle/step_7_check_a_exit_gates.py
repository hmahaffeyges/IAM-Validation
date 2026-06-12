#!/usr/bin/env python3
"""
STEP 7 / CHECK A — Per-class exit-gate validator
==================================================

Reads each iamatlas_v0_1_<class>_result.json file produced by the production
MCMC and validates against the pilot benchmarks. This is the FIRST validation
run after the production MCMC completes — confirms the matrix is trustworthy
before downstream cohort tests.

Exit gates per class (from the gaming PC bundle README):
  - convergence.rhat_max     < 1.05  ✓ converged
  - convergence.ess_min      > 200   ✓ effective samples sufficient
  - convergence.n_diverging  = 0     ✓ no Hamiltonian divergences
  - predictive.pearson       > 0.90  ✓ matches pilot performance
  - predictive.mae           < 0.20  ✓ residual within tolerance
                                       (sparse classes — terminal, stromal —
                                        may run higher; documented separately)

Usage:
  cd ~/IAMPerformance
  python3 step_7_check_a_exit_gates.py

Output:
  step_7_exit_gate_report.md  — pass/fail per class, with diagnostics

Date: 2026-05-04
"""

import argparse
import json
from pathlib import Path

ALL_CLASSES = [
    "stem_pluri", "stem_adult", "progenitor", "stromal",
    "cycling", "secretory", "immune", "terminal",
]

# Exit gates
EXIT_GATES = {
    "convergence.rhat_max":     ("<",  1.05, "converged"),
    "convergence.ess_min":      (">",  200,  "effective samples sufficient"),
    "convergence.n_diverging":  ("==", 0,    "no Hamiltonian divergences"),
    "predictive.pearson":       (">",  0.90, "predictive accuracy"),
}
# Predictive MAE is class-stratified — sparse classes get higher tolerance
MAE_THRESHOLD_BY_CLASS = {
    "stem_pluri": 0.15,
    "stem_adult": 0.20,   # n=5 donors, expected wider posterior
    "progenitor": 0.15,
    "stromal":    0.20,   # only 97K rows, sparse
    "cycling":    0.10,
    "secretory":  0.10,
    "immune":     0.10,
    "terminal":   0.20,   # 30K rows, sparsest
}


def get_nested(d: dict, key: str):
    """Navigate dot-separated keys into a dict, e.g. 'convergence.rhat_max'."""
    parts = key.split(".")
    v = d
    for p in parts:
        if not isinstance(v, dict) or p not in v:
            return None
        v = v[p]
    return v


def evaluate_gate(value, op: str, threshold) -> bool:
    if value is None:
        return False
    try:
        if op == "<":  return value < threshold
        if op == ">":  return value > threshold
        if op == "==": return value == threshold
    except TypeError:
        return False
    return False


def check_class(result_path: Path) -> dict:
    """Check exit gates for one class result.json."""
    if not result_path.exists():
        return {"status": "MISSING", "path": str(result_path)}
    with open(result_path) as f:
        result = json.load(f)
    cls = result.get("class", result_path.stem.split("_")[2])
    gates_passed = []
    gates_failed = []
    for key, (op, threshold, label) in EXIT_GATES.items():
        value = get_nested(result, key)
        passed = evaluate_gate(value, op, threshold)
        gates_passed.append((key, value, op, threshold, label, passed))
        if not passed:
            gates_failed.append((key, value, op, threshold))
    # MAE check (class-stratified threshold)
    mae = get_nested(result, "predictive.mae")
    mae_threshold = MAE_THRESHOLD_BY_CLASS.get(cls, 0.10)
    mae_passed = mae is not None and mae < mae_threshold
    gates_passed.append(("predictive.mae", mae, "<", mae_threshold, "residual tolerance", mae_passed))
    if not mae_passed:
        gates_failed.append(("predictive.mae", mae, "<", mae_threshold))
    return {
        "class": cls,
        "status": "PASS" if not gates_failed else "FAIL",
        "n_cpg": result.get("n_cpg"),
        "n_obs": result.get("n_obs"),
        "elapsed_s": result.get("elapsed_s"),
        "gates_passed": gates_passed,
        "gates_failed": gates_failed,
        "raw_diagnostics": {
            "rhat_max": get_nested(result, "convergence.rhat_max"),
            "ess_min": get_nested(result, "convergence.ess_min"),
            "n_diverging": get_nested(result, "convergence.n_diverging"),
            "pearson": get_nested(result, "predictive.pearson"),
            "mae": mae,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Step 7 Check A: per-class exit-gate validator")
    parser.add_argument("--in_dir", default="iamatlas_v0_1_output",
                        help="Directory containing iamatlas_v0_1_<class>_result.json files")
    parser.add_argument("--report", default="step_7_exit_gate_report.md")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    print(f"Reading per-class results from: {in_dir}")
    print(f"Exit gates:")
    for k, (op, t, label) in EXIT_GATES.items():
        print(f"  {k}  {op}  {t}    ({label})")
    print(f"  predictive.mae < class-stratified threshold (sparse classes higher)")
    print()

    results = []
    for cls in ALL_CLASSES:
        result_path = in_dir / f"iamatlas_v0_1_{cls}_result.json"
        check = check_class(result_path)
        results.append(check)
        if check["status"] == "MISSING":
            print(f"  [MISSING] {cls}: result.json not found at {result_path}")
        else:
            symbol = "✓" if check["status"] == "PASS" else "✗"
            d = check["raw_diagnostics"]
            mae_str = f"{d['mae']:.4f}" if d['mae'] is not None else "N/A"
            print(f"  [{symbol}] {cls:<12}  rhat={d['rhat_max']:.3f}  ESS={d['ess_min']:.0f}  "
                  f"div={d['n_diverging']}  pearson={d['pearson']:.3f}  MAE={mae_str}")

    # Write report
    with open(args.report, "w") as f:
        f.write("# IAMAtlas v0.1 Exit-Gate Report — Step 7 / Check A\n\n")
        f.write(f"**Date:** 2026-05-04\n")
        f.write(f"**Source directory:** `{in_dir}`\n")
        f.write(f"**Classes evaluated:** {len(ALL_CLASSES)}\n\n")

        n_pass = sum(1 for r in results if r["status"] == "PASS")
        n_fail = sum(1 for r in results if r["status"] == "FAIL")
        n_missing = sum(1 for r in results if r["status"] == "MISSING")
        f.write(f"**Summary:** {n_pass} pass · {n_fail} fail · {n_missing} missing\n\n")

        f.write("## Per-class diagnostics\n\n")
        f.write("| Class | Status | R-hat max | ESS min | Divergent | Pearson | MAE | n CpG | n obs | Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            if r["status"] == "MISSING":
                f.write(f"| {r.get('class','?')} | MISSING | — | — | — | — | — | — | — | — |\n")
                continue
            d = r["raw_diagnostics"]
            mae = f"{d['mae']:.4f}" if d['mae'] is not None else "N/A"
            f.write(f"| {r['class']} | {r['status']} | {d['rhat_max']:.3f} | {d['ess_min']:.0f} | "
                    f"{d['n_diverging']} | {d['pearson']:.3f} | {mae} | {r['n_cpg']} | {r['n_obs']} | "
                    f"{r['elapsed_s']:.0f} |\n")

        if n_fail > 0:
            f.write("\n## Failed gates (per class)\n\n")
            for r in results:
                if r["status"] != "FAIL": continue
                f.write(f"### `{r['class']}`\n\n")
                for key, value, op, threshold in r["gates_failed"]:
                    f.write(f"- `{key}` = {value} (gate: `{op} {threshold}`) — **FAIL**\n")
                f.write("\n")

        f.write("\n## Decision\n\n")
        if n_fail == 0 and n_missing == 0:
            f.write("**✓ All 8 classes pass.** Production matrix is ready to merge into `IAMAtlas_v0_1.csv`.\n\n")
            f.write("Run the merge:\n```\npython3 merge_iamatlas_v0_1.py --in_dir iamatlas_v0_1_output --output IAMAtlas_v0_1.csv\n```\n\n")
            f.write("Then proceed to Step 7 / Check B (predictive validation against held-out cohorts).\n")
        elif n_missing > 0:
            f.write(f"**⏸ {n_missing} class(es) still pending.** Re-run after MCMC completes.\n")
        else:
            f.write(f"**✗ {n_fail} class(es) failed exit gates.** Investigate before merging:\n")
            f.write("- For R-hat or ESS failures: re-run that class with `--tune 2000 --draws 2000`\n")
            f.write("- For divergences: investigate prior specifications, may need `target_accept=0.95`\n")
            f.write("- For MAE failures: check class-specific input data quality\n")

    print(f"\nReport written: {args.report}")
    if n_fail == 0 and n_missing == 0:
        print("\n✓ All classes PASS — ready to merge.")
    elif n_missing > 0:
        print(f"\n⏸ {n_missing} class(es) still pending.")
    else:
        print(f"\n✗ {n_fail} class(es) failed — see report.")


if __name__ == "__main__":
    main()
