"""Canonical A-score fail-safe.

Guards the one invariant that matters: A = mean_i( H(beta_i) / H_min(class) ) is
the MEAN OF THE PER-CpG ENTROPIES across the panel -- NOT H(beta_mean), the
entropy of the mean beta. Those two agree only when every beta is identical
(e.g. a uniform beta=0.5 input), which is exactly why the old uniform self-test
passed while a backwards aggregation could still ship.

CORRECTED v1.4.0 (2026-06-30, SOP LESSON-ASCORE-02 / §105): a prior version of
this fail-safe asserted the H(beta_mean) form as canonical and would have BLOCKED
the validated module. The validated instrument -- the one that reproduces the
sealed GSE51032 anchor (115/115 A-scores, 460/460 Mahalanobis, Cohen's d +2.088)
-- computes the mean of per-CpG entropies. Any build that does not reproduce that
anchor is wrong by definition.

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


def check():
    """Run the canonical assertions; return a list of failure strings (empty = pass).
    Importable so the orchestrator can gate startup on it without sys.exit."""
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
    #    Half the panel at 0.05, half at 0.95. Each locus is locked (low entropy),
    #    so the CORRECT mean-of-per-CpG score is H(0.05)/H_min ~ 0.371. The BROKEN
    #    H(beta_mean) form would average beta to 0.5 first and return the ceiling
    #    1/H_min ~ 1.294 -- a healthy locked panel reading as max disorder (the
    #    false-breach regression). This guard fails loudly if the module reverts to
    #    entropy-of-the-mean.
    bimodal = [0.05] * 15 + [0.95] * 15
    a_bim = _score(bimodal, H_MIN["terminal"])
    correct_mean_of_H = _H(0.05) / H_MIN["terminal"]   # ~0.371
    broken_ceiling = 1.0 / H_MIN["terminal"]           # ~1.294
    if abs(a_bim - correct_mean_of_H) > 0.02:
        fails.append(
            f"AGGREGATION REGRESSION: bimodal panel (each locus locked) gave A={a_bim:.3f}; "
            f"correct mean-of-per-CpG = H(0.05)/H_min = {correct_mean_of_H:.3f}. A value near "
            f"the ceiling {broken_ceiling:.3f} means the module took the entropy of the mean "
            f"beta (H(beta_mean)) -- the regression. See SOP LESSON-ASCORE-02 (§105)."
        )

    # 3. Ceiling invariant: A never exceeds 1/H_min.
    a_top = _score([0.5] * 30, H_MIN["immune"])
    if a_top > 1.0 / H_MIN["immune"] + 1e-9:
        fails.append(f"ceiling violated: A={a_top:.4f} > 1/H_min={1.0/H_MIN['immune']:.4f}")

    # 4. Healthy reference must land at the floor (~1.0), not suppressed (~0.5).
    a_healthy = _score([0.75] * 30, H_MIN["immune"])
    if not (0.95 <= a_healthy <= 1.04):
        fails.append(f"healthy beta=0.75 gave A={a_healthy:.3f}, outside the 0.95-1.04 normal band")

    return fails


def run():
    fails = check()
    if fails:
        print("A-SCORE FAIL-SAFE: FAILED")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("A-score fail-safe: PASS (4 checks incl. bimodal aggregation guard)")


if __name__ == "__main__":
    run()
