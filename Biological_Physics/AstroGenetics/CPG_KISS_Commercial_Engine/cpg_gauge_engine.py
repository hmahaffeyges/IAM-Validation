#!/usr/bin/env python3
"""
cpg_gauge_engine.py — the canonical A-score GAUGE (GAPE Issue 002 spec).

WHAT THIS IS
------------
The gauge that the doctor reads. One instrument, one job: take the ONE mean
beta of a cell (over its identity loci) and return where it sits on the
class-and-substrate ruler.

        A = H(beta_mean) / H_min(class, substrate)

H(beta) is the Shannon entropy of a SINGLE mean beta (Bernoulli(beta)), exactly
as the GAPE derivation suite defines it. This is entropy-of-the-mean, NOT
mean-of-per-CpG-entropies.

WHAT THIS IS NOT  (read this before you ever change the formula)
---------------------------------------------------------------
  * NOT the separation statistic. mean_i(H(beta_i)/H_min) over one-vs-rest
    discriminative markers is a DIFFERENT instrument (the Cohen's-d / Mahalanobis
    disease-matching surface, iamatlas_a_scoring.py). It answers "how separable
    is this patient from healthy," never "where does this cell sit on the ruler."
    Do not feed this gauge discriminative markers: they are bimodal (half locked
    high, half low), their mean beta collapses toward 0.5, and the gauge falsely
    pins the ceiling. That is the all-BREACH bug of 2026-06-11.
  * NOT forced to 1.0. A = 1.0 is the architectural COMMITMENT LINE (the H_min
    reference), not where healthy sits. Healthy reads in the age-matched NORMAL
    band ~0.95-1.00 and drifts up toward 1.0 across the lifespan.

THREE DIRECTIONS
----------------
  A < 0.95         INVERSION  — legitimate identity loss (seminoma 0.67,
                                senescence, aged HSC). A finding, never an error.
  0.95 <= A < 1.01 NORMAL     — the age-matched healthy band.
  1.01 <= A < 1.05 MARGINAL
  1.05 <= A < 1.07 DETECTABLE
  1.07 <= A < 1.10 URGENT
  A >= 1.10        FLOOR BREACH  (ceiling at 1/H_min)

INPUT SCALE
-----------
The gauge is pure: it assumes the incoming mean beta is already on the IAMAtlas
scale. Aligning a raw array sample to that scale (the ~0.05 per-class/substrate
offset demonstrated on whole-blood immune) is a Stage-1 normalization job, NOT
the gauge's. Keep it upstream so the gauge stays a clean physics reading.

SUBSTRATE
---------
Every (class, substrate) pair has its own floor. cfDNA is the frag/wps/nucl
substrates — score it on those floors, NEVER the methyl floor.

Source: Mahaffey 2026 GAPE Issue 002 (H_MIN_TABLE, substrate registry, tiers,
healthy baselines, three-component decomposition). Zero fitted parameters.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Tuple

# ─── Core: Shannon binary entropy of ONE mean beta ───────────────────────────
def H(beta: float) -> float:
    """Shannon binary entropy of a Bernoulli(beta) variable. Singular beta."""
    if beta <= 0.0 or beta >= 1.0:
        return 0.0
    return -beta * math.log2(beta) - (1.0 - beta) * math.log2(1.0 - beta)

# ─── The 40-cell H_min grid (G-002 methyl + G-003b four substrates) ──────────
SUB_ORDER = ['methyl', 'nucl', 'fuzz', 'wps', 'frag']
H_MIN_TABLE = {
    'cycling':    (0.856055, 0.980072, 0.819030, 0.627429, 0.687936),
    'secretory':  (0.843264, 0.982560, 0.847947, 0.634534, 0.697718),
    'immune':     (0.838889, 0.989930, 0.830377, 0.589644, 0.711534),
    'terminal':   (0.772837, 0.992027, 0.736973, 0.958909, 0.624938),
    'stromal':    (0.862950, 0.985667, 0.832386, 0.612686, 0.724691),
    'stem_pluri': (0.982166, 0.799818, 0.962920, 0.905004, 0.973583),
    'stem_adult': (0.873718, 0.960866, 0.980754, 0.988964, 0.841327),
    'progenitor': (0.852216, 0.972790, 0.961900, 0.988046, 0.808978),
}
# AUC weights (published single-substrate discrimination); cfDNA = frag/wps/nucl.
AUC_W = {'methyl': 0.8663, 'nucl': 0.852, 'fuzz': 0.779, 'wps': 0.761, 'frag': 0.940}

# ─── Age-matched healthy baselines (8 classes x 10 decades) ──────────────────
# Compiled from Hannum 2013, Horvath 2013, Roadmap 2015, Moss 2018, Lister 2013,
# Alisch 2012. Column order matches _BASELINE_CLASSES below.
_BASELINE_CLASSES = ['cycling', 'secretory', 'immune', 'terminal',
                     'stromal', 'stem_adult', 'progenitor', 'stem_pluri']
HEALTHY_BASELINE = {
    '0-9':   [0.9383, 0.9506, 0.9062, 0.9077, 0.9438, 0.9375, 0.9557, 0.8292],
    '10-19': [0.9458, 0.9583, 0.9212, 0.9210, 0.9510, 0.9428, 0.9611, 0.8308],
    '20-29': [0.9514, 0.9639, 0.9316, 0.9393, 0.9563, 0.9462, 0.9666, 0.8324],
    '30-39': [0.9568, 0.9695, 0.9397, 0.9520, 0.9615, 0.9497, 0.9701, 0.8340],
    '40-49': [0.9604, 0.9732, 0.9477, 0.9619, 0.9667, 0.9531, 0.9736, 0.8340],
    '50-59': [0.9640, 0.9768, 0.9556, 0.9692, 0.9734, 0.9564, 0.9789, 0.8356],
    '60-69': [0.9693, 0.9822, 0.9652, 0.9789, 0.9784, 0.9614, 0.9840, 0.8356],
    '70-79': [0.9762, 0.9892, 0.9764, 0.9930, 0.9849, 0.9664, 0.9907, 0.8356],
    '80-89': [0.9830, 0.9962, 0.9873, 1.0067, 0.9913, 0.9728, 0.9973, 0.8371],
    '90+':   [0.9912, 1.0046, 0.9996, 1.0244, 0.9991, 0.9791, 1.0038, 0.8371],
}

# ─── Tier structure — three directions ───────────────────────────────────────
INVERSION_LINE = 0.95
BREACH = 1.10
SATURATION_MARGIN = 0.005
_TIERS = [  # (label, lo_inclusive, hi_exclusive)
    ('INVERSION',  -math.inf,      INVERSION_LINE),
    ('NORMAL',     INVERSION_LINE, 1.01),
    ('MARGINAL',   1.01,           1.05),
    ('DETECTABLE', 1.05,           1.07),
    ('URGENT',     1.07,           1.10),
    ('BREACH',     1.10,           math.inf),
]

# ─── Input-scale offset hook (per class, substrate) ──────────────────────────
# Aligns a raw array mean-beta to the IAMAtlas scale. Default 0.0 — the gauge is
# pure and expects atlas-scale input. Any offset belongs to Stage-1 normalization
# and must be VALIDATED (frozen per class/substrate) before it is set non-zero.
# The whole-blood immune/methyl offset demonstrated ~0.054 but is NOT frozen
# (batch structure; awaits the production normalizer).
OFFSET: Dict[Tuple[str, str], float] = {}


def H_min_for(cls: str, sub: str = 'methyl') -> float:
    return H_MIN_TABLE[cls][SUB_ORDER.index(sub)]


def a_ceiling(cls: str, sub: str = 'methyl') -> float:
    """Physical maximum A on this class x substrate: 1/H_min (beta=0.5, H=1)."""
    return 1.0 / H_min_for(cls, sub)


def a_score(mean_beta: float, cls: str, sub: str = 'methyl') -> float:
    """THE GAUGE. A = H(one mean beta) / H_min(class, substrate).

    mean_beta is expected on the IAMAtlas scale (Stage-1 normalized).
    """
    shift = OFFSET.get((cls, sub), 0.0)
    return H(mean_beta - shift) / H_min_for(cls, sub)


def _decade(age: int) -> str:
    if age is None:
        return '40-49'  # neutral middle-adult default
    if age >= 90:
        return '90+'
    lo = (int(age) // 10) * 10
    return f'{lo}-{lo + 9}'


def healthy_baseline(cls: str, age: Optional[int]) -> float:
    """Age-matched expected healthy A for this class (the NORMAL-band center)."""
    return HEALTHY_BASELINE[_decade(age)][_BASELINE_CLASSES.index(cls)]


def tier(A: float) -> str:
    """Three-direction tier label from the absolute A-score."""
    for label, lo, hi in _TIERS:
        if lo <= A < hi:
            return label
    return 'BREACH'


def departure(A: float, cls: str, age: Optional[int]) -> float:
    """A minus the age-matched healthy baseline. >0 elevated, <0 suppressed."""
    return A - healthy_baseline(cls, age)


def is_saturated(A: float, cls: str, sub: str, margin: float = SATURATION_MARGIN) -> bool:
    """Runtime: within margin of the physical ceiling (this sample)."""
    return A >= a_ceiling(cls, sub) - margin


def is_structurally_saturated(cls: str, sub: str, threshold: float = BREACH) -> bool:
    """Class-level: ceiling itself sits below BREACH (sample-independent)."""
    return a_ceiling(cls, sub) < threshold


def a_combined(sub_means: Dict[str, float], cls: str) -> Optional[float]:
    """AUC-weighted gauge across all provided substrates. sub_means: {sub: mean_beta}."""
    ws = wa = 0.0
    for sub, mb in sub_means.items():
        if mb is None or not (0.01 < mb < 0.99):
            continue
        w = AUC_W[sub]
        ws += w
        wa += w * a_score(mb, cls, sub)
    return (wa / ws) if ws else None


def a_active(sub_means: Dict[str, float], cls: str) -> Optional[float]:
    """AUC-weighted gauge over NON-saturated substrates only (reserve/response signal)."""
    ws = wa = 0.0
    for sub, mb in sub_means.items():
        if mb is None or not (0.01 < mb < 0.99):
            continue
        Ai = a_score(mb, cls, sub)
        if is_saturated(Ai, cls, sub):
            continue
        w = AUC_W[sub]
        ws += w
        wa += w * Ai
    return (wa / ws) if ws else None


H_MIN_GLOBAL = H(0.782)  # 0.756499 — frontal-cortex-neuron Landauer anchor (Lister 2013)


def three_component(mean_beta: float, cls: str, sub: str = 'methyl') -> Tuple[float, float, float]:
    """(f_C1, f_C2, f_C3) fractions of entropy. C1 universal Landauer floor,
    C2 class overhead, C3 accessible clinical gap. Valid where H >= H_min(class)."""
    h = H(mean_beta)
    if h <= 0:
        return (0.0, 0.0, 0.0)
    hm = H_min_for(cls, sub)
    return (H_MIN_GLOBAL / h, (hm - H_MIN_GLOBAL) / h, max(0.0, h - hm) / h)


def read(mean_beta: float, cls: str, age: Optional[int] = None,
         sub: str = 'methyl') -> Dict:
    """Full gauge reading for one cell: the object the report renders."""
    A = a_score(mean_beta, cls, sub)
    return {
        'A': A,
        'tier': tier(A),
        'ceiling': a_ceiling(cls, sub),
        'healthy_baseline': healthy_baseline(cls, age),
        'departure_from_age_band': departure(A, cls, age),
        'structurally_saturated': is_structurally_saturated(cls, sub),
        'class': cls, 'substrate': sub, 'age_decade': _decade(age),
    }


# ─── Self-test: reproduce the GAPE Issue 002 published examples ──────────────
def _selftest() -> bool:
    ok = True
    def chk(name, cond, detail=''):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))

    print("=" * 66)
    print("cpg_gauge_engine — GAPE Issue 002 self-test")
    print("=" * 66)
    chk("H(0.5)=1, H(0)=H(1)=0",
        abs(H(0.5) - 1) < 1e-9 and H(0) == 0 and H(1) == 0)
    a = a_score(0.685, 'cycling', 'methyl')
    chk("gauge: beta=0.685 cycling -> A=1.050 (DETECTABLE)",
        abs(a - 1.0502) < 1e-3 and tier(a) == 'DETECTABLE', f"A={a:.4f}")
    a_h = a_score(0.720, 'cycling', 'methyl')
    chk("gauge: healthy cycling beta~0.72 -> A~1.0 NORMAL band",
        0.95 <= a_h < 1.01, f"A={a_h:.4f} tier={tier(a_h)}")
    a_sem = a_score(0.18, 'stem_pluri', 'methyl')
    chk("inversion: seminoma beta=0.18 stem_pluri -> A<0.75 INVERSION",
        a_sem < 0.75 and tier(a_sem) == 'INVERSION', f"A={a_sem:.4f}")
    chk("ceiling cycling methyl = 1/H_min = 1.168",
        abs(a_ceiling('cycling', 'methyl') - 1.1681) < 1e-3)
    chk("structural saturation: stem_pluri methyl ceiling below BREACH",
        is_structurally_saturated('stem_pluri', 'methyl'))
    chk("age-matched: immune healthy 30s < 60s (drifts up with age)",
        healthy_baseline('immune', 35) < healthy_baseline('immune', 65))
    chk("three-component sums to 1 at beta=0.64 cycling",
        abs(sum(three_component(0.64, 'cycling')) - 1.0) < 1e-9)
    fc = a_combined({'methyl': 0.740, 'nucl': 0.615, 'frag': 0.790}, 'cycling')
    chk("A_combined healthy cycling near NORMAL", fc is not None and 0.90 < fc < 1.05,
        f"A_combined={fc:.4f}")
    print("=" * 66)
    print(f"{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    print("=" * 66)
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() else 1)
