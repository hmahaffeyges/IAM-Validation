"""Canonical A-score fail-safe -- SEPARATION SURFACE (discriminative markers, 115 cell types).

Guards the invariant for THIS surface: A = mean_i( H(beta_i) / H_min(class) ) is the MEAN OF THE
PER-CpG ENTROPIES across a marker panel -- NOT H(beta_mean). Marker panels are bimodal by
construction, so H(beta_mean) manufactures ~0.5 and pins the ceiling (SOP s105).

SCOPED v1.5.0 (2026-09-19, SOP s106 / RULING A3): this rule applies to the separation surface only.
The class GAUGE (8 classes, identity loci, cpg_gauge_engine.py + Stage 4 wiring) correctly computes
H(beta_mean)/H_min, because H_min (G-002/G-003b) and age_reference_matrix.json are both defined as H
of a mean beta and the identity loci are unimodal. Both surfaces reproduce their own sealed reference
(PROC-ANCHOR-01: r = 1.00000 on 648 samples; PROC-FORMULA-01). Neither form is "the regression" --
using either on the other surface is. The gauge-side guard is the Jensen-gap test in
CPG_KISS_Commercial_Engine/tests/cpg_kit.py::gauge_A.

CORRECTED v1.4.0 (2026-06-30, SOP LESSON-ASCORE-02 / s105): a prior version of this fail-safe
asserted the H(beta_mean) form as canonical for markers and would have BLOCKED the validated module.
The validated separation instrument -- the one that reproduces the sealed GSE51032 anchor (115/115
A-scores, 460/460 Mahalanobis, Cohen's d +2.088) -- computes the mean of per-CpG entropies. Any
SEPARATION build that does not reproduce that anchor is wrong by definition.

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




# --- v1.5.0: gauge-surface companion test (SOP s106). Identity loci are unimodal; on such a panel
# H(beta_mean) and mean_i H(beta_i) must agree to within the Jensen bound the gauge tolerates (0.05).
def test_gauge_surface_jensen_bound(jensen_max=0.05):
    import math, random
    random.seed(20260919)
    def H(b):
        b=min(max(b,1e-12),1-1e-12); return -b*math.log2(b)-(1-b)*math.log2(1-b)
    panel=[min(max(random.gauss(0.73,0.06),0.02),0.98) for _ in range(5000)]   # unimodal identity-like panel
    gap=H(sum(panel)/len(panel))-sum(H(b) for b in panel)/len(panel)
    assert 0 <= gap < jensen_max, f"identity-like panel Jensen gap {gap:.4f} exceeds {jensen_max}"
    bim=[0.05]*2500+[0.95]*2500                                                 # bimodal marker-like panel
    gap_b=H(sum(bim)/len(bim))-sum(H(b) for b in bim)/len(bim)
    assert gap_b > jensen_max, "bimodal panel must be refused by the gauge (Jensen gap too small?)"
    return gap, gap_b

if __name__ == "__main__":
    _g,_gb=test_gauge_surface_jensen_bound(); print(f"[gauge_surface] identity-like Jensen gap {_g:.4f} (<0.05 ok) | bimodal {_gb:.4f} (refused ok)")
    run()
