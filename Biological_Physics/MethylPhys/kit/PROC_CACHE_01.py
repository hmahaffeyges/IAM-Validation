#!/usr/bin/env python3
"""Invariance check for the deconvolver cache (2026-09-26).

The cache must not change ANY reported value. This recomputes specimens whose bundles were written BEFORE
the cache existed and requires every reported number to be identical - not close, identical. Fields that are
expected to differ between two runs of the same specimen (identifiers, timestamps, paths, durations) are
excluded by name and the exclusion list is printed, so it cannot quietly hide a real difference.
"""
import glob
import json
import os
import subprocess
import sys

VOLATILE = {"run_id", "when", "timestamp", "generated", "duration_s", "elapsed_s", "out", "path",
            "bundle_path", "report", "versions", "intake", "propagate"}

# Two flaws in the first version of this check, both mine, both found by running it:
#
#  1. It compared nan to nan and called them different, because nan != nan in Python. That produced 65
#     spurious "differences" on one specimen - every one of them a cell with no reading before AND after.
#  2. It demanded exact float equality. Measured: two runs of IDENTICAL code on the same specimen differ by
#     1.04e-14 relative, in diagnostic_cellular_age.per_class - float accumulation order is not stable
#     run to run. (handoff/determinism.json)
#
# So the criterion is split rather than loosened, because loosening a test until it passes is how a real
# change gets through. Every REPORTED READING must be EXACTLY equal - A, fraction, tier, placement, H_min,
# band, present, reportable. Derived quantities may differ only within the measured non-determinism floor.
READING = ("A", "A_abs", "A_mapped", "fraction", "tier", "placement", "H_min", "band", "present",
           "reportable", "foreign_fraction", "composition_verified", "n_loci", "gauge_surface")
DETERMINISM_FLOOR = 1e-12          # measured floor is 1.04e-14; this is two orders of headroom


def is_reading(key):
    return key.rsplit(".", 1)[-1] in READING or ".composition." in key


def equal(x, y):
    import math
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True                # no reading before, no reading after - not a difference
    return x == y


def flat(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in VOLATILE:
                continue
            yield from flat(v, f"{pre}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, o


def main():
    W = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    run = os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/chain/MethylPhys_Interface/run_sample.py")
    olds = sorted(glob.glob(os.path.join(W, "results/synth01/*_bundle.json")))[:6]
    print(f"excluded as volatile: {sorted(VOLATILE)}")
    print(f"specimens to recompute: {len(olds)}\n")
    worst = 0
    for b in olds:
        name = os.path.basename(b).replace("_bundle.json", "")
        csv = os.path.join(W, f"results/synth01/{name}.csv")
        if not os.path.exists(csv):
            print(f"  {name}: no input CSV - skipped")
            continue
        before = json.load(open(b))
        out = os.path.join(W, f"results/cache01/{name}.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([sys.executable, run, "--betas", csv, "--age", "55", "--sex", "F", "--lab", "SYNTH",
                        "--specimen", "whole blood", "--no-intake", "--out", out, "--id", name],
                       capture_output=True, text=True,
                       env={**os.environ, "HOME": os.path.join(W, "stage1/mp_home")})
        nb = out.replace(".html", "_bundle.json")
        if not os.path.exists(nb):
            print(f"  {name}: RECOMPUTE PRODUCED NO BUNDLE")
            worst += 1
            continue
        after = json.load(open(nb))
        a, c = dict(flat(before)), dict(flat(after))
        diff = [k for k in set(a) | set(c) if not equal(a.get(k), c.get(k))]
        readings_changed = [k for k in diff if is_reading(k)]
        mx, at = 0.0, None
        for k in diff:
            x, y = a.get(k), c.get(k)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                    and not isinstance(x, bool) and not isinstance(y, bool):
                d = abs(x - y) / max(abs(x), abs(y), 1e-12)
                if d > mx:
                    mx, at = d, k
        over = mx > DETERMINISM_FLOOR
        print(f"  {name:<28} {len(a):>5} values | readings changed: {len(readings_changed)} | "
              f"max rel diff {mx:.2e}{' OVER FLOOR' if over else ''}")
        for k in readings_changed[:4]:
            print(f"      READING CHANGED {k}: before {a.get(k)!r} -> after {c.get(k)!r}")
        if over and at:
            print(f"      largest difference at {at}: {a.get(at)!r} -> {c.get(at)!r}")
        worst += len(readings_changed) + (1 if over else 0)
    print(f"\nreadings changed, plus specimens over the non-determinism floor: {worst}")
    print("PASS - every reported reading is identical; derived values differ only within the\n         measured 1.04e-14 run-to-run floor" if worst == 0 else
          "FAIL - the cache altered a reported reading")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
