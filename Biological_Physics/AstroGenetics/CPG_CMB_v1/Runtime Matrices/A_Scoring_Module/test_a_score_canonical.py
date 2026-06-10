"""Canonical A-score fail-safe.

Guards the one invariant that matters: A = H(beta_mean) / H_min(class) is the
ENTROPY OF THE MEAN beta across the panel -- NOT the mean of the per-CpG
entropies. Those two agree only when every beta is identical (e.g. a uniform
beta=0.5 input), which is exactly why the old uniform self-test passed while the
production aggregation was backwards.

Run: python test_a_score_canonical.py   (exits non-zero on any failure)
Wire into startup / CI so a regression in the scoring math can never ship again.
"""
import importlib.util, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("asc", os.path.join(HERE, "iamatlas_a_scoring.py"))
asc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asc)

# Frozen floors (the eight Mahaffey Numbers), source of truth = IAMAtlasREBUILD_provenance.json
H_MIN = {"terminal": 0.7728, "secretory": 0.843264, "immune": 0.838889}


def _H(b):
    return 0.0 if b <= 0 or b >= 1 else -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def _score(betas, h_min):
    """Score a panel of beta values through the production _score_one."""
    cgs = [f"cg{i:08d}" for i in range(len(betas))]
    return asc._score_one(pd.Series(dict(zip(cgs, betas))), cgs, h_min)["A"]


def run():
    fails = []

    # 1. Canonical worked examples (single characteristic beta per cell, replicated
    #    across the panel so all CpGs agree). These verify the floor values + formula.
    for name, beta, cls, expected in [
        ("healthy neuron", 0.782, "terminal", 0.978),
        ("normal breast", 0.745, "secretory", 0.971),
        ("glioblastoma", 0.400, "terminal", 1.256),
    ]:
        a = _score([beta] * 30, H_MIN[cls])
        if abs(a - expected) > 0.01:
            fails.append(f"{name}: A={a:.3f}, canonical reference says {expected:.3f}")

    # 2. THE BIMODAL GUARD -- the test the old uniform self-test could not be.
    #    Half the panel at 0.05, half at 0.95 -> mean beta = 0.5 -> A must be
    #    H(0.5)/H_min = 1/H_min (the ceiling). If the module averages per-CpG
    #    entropies instead, it returns H(0.05)/H_min ~ 0.37 and this fails loudly.
    bimodal = [0.05] * 15 + [0.95] * 15
    a_bim = _score(bimodal, H_MIN["terminal"])
    ceiling = 1.0 / H_MIN["terminal"]  # 1.294
    if abs(a_bim - ceiling) > 0.02:
        fails.append(
            f"AGGREGATION REGRESSION: bimodal panel (mean beta=0.5) gave A={a_bim:.3f}; "
            f"must be H(mean)/H_min={ceiling:.3f}. A value near 0.37 means the module is "
            f"averaging per-CpG entropies instead of taking the entropy of the mean beta."
        )

    # 3. Ceiling invariant: A never exceeds 1/H_min.
    a_top = _score([0.5] * 30, H_MIN["immune"])
    if a_top > 1.0 / H_MIN["immune"] + 1e-9:
        fails.append(f"ceiling violated: A={a_top:.4f} > 1/H_min={1.0/H_MIN['immune']:.4f}")

    # 4. Healthy reference must land at the floor (~1.0), not suppressed (~0.5).
    a_healthy = _score([0.75] * 30, H_MIN["immune"])
    if not (0.95 <= a_healthy <= 1.04):
        fails.append(f"healthy beta=0.75 gave A={a_healthy:.3f}, outside the 0.95-1.04 normal band")

    if fails:
        print("A-SCORE FAIL-SAFE: FAILED")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("A-score fail-safe: PASS (4 checks incl. bimodal aggregation guard)")


if __name__ == "__main__":
    run()
