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
    reference), not where healthy sits. Healthy reads on the age-matched curve
    (age_reference_matrix): immune 0.906 @ age 4 rising to 1.000 @ age 95. The
    gauge places a patient against that age band, never a fixed line.

TWO AXES  (age matters — the root of the A-score methodology)
-------------------------------------------------------------
  PLACEMENT — vs age-matched healthy peers (age_reference_matrix p10..p90):
    A <  A_p10(class,age)     BELOW BAND
    A_p10 <= A <= A_p90       IN BAND
    A >  A_p90(class,age)     ABOVE BAND
  SEVERITY — elevation alarm ladder (Issue 002); the tier, not the placement:
    A >= 1.10                 FLOOR BREACH   (ceiling at 1/H_min)
    1.07 <= A < 1.10          URGENT
    1.05 <= A < 1.07          DETECTABLE
    1.01 <= A < 1.05          MARGINAL
    below 1.01                NORMAL         — fidelity maintained, including a
                                mild dip below the age cohort (not an alarm)
    far below the age cohort  INVERSION      — legitimate identity loss (seminoma
                                0.67, senescence, aged HSC). A finding, not error.

INPUT SCALE
-----------
The gauge is pure: it assumes the incoming mean beta is already on the IAMAtlas
scale. Stage-1 normalization owns any raw-array alignment. The per-class input
offset explored earlier is RETIRED — it was reinventing age_reference_matrix and
forcing healthy onto 1.0 (the age-95 value only). Healthy IS the age-matched
band, so no offset is applied in the gauge.

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

# ─── Age-matched band half-width (fallback only) ─────────────────────────────
# Decade-averaged A_sd per class from age_reference_matrix, used to synthesize a
# p10..p90 band ONLY when the canonical age_reference_matrix.json is unreachable.
# When it is reachable (default), its own A_p10/A_p90 percentiles are used verbatim.
HEALTHY_SD = {
    'cycling': 0.030, 'secretory': 0.028, 'immune': 0.034, 'terminal': 0.038,
    'stromal': 0.031, 'stem_adult': 0.026, 'progenitor': 0.024, 'stem_pluri': 0.005,
}
_Z90 = 1.2816  # p10/p90 = mean -/+ 1.2816*sd (normal approx; JSON percentiles win)

# ─── Absolute severity ladder (Issue 002; low end is age-relative) ───────────
BREACH = 1.10
SATURATION_MARGIN = 0.005

# ─── Canonical age_reference_matrix loader ───────────────────────────────────
# The gauge places a patient against the age-matched band. The matrix (8 classes
# x 10 age bins, A_mean/A_p10/A_p90/beta_mean) is loaded from the co-located
# runtime copy (sourced from atlas_vault). No input offset — healthy IS this band.
AGE_MATRIX_JSON: Optional[str] = None   # Stage-4 may override the path
_BANDS_CACHE: Optional[Dict[str, list]] = None
_MIDPOINTS = [4, 14, 24, 34, 44, 54, 64, 74, 84, 95]


def _load_bands() -> Dict[str, list]:
    """Return {class: [(age, A_mean, A_p10, A_p90), ...]} from the canonical JSON,
    or {} to signal the embedded A_mean fallback should be used."""
    global _BANDS_CACHE
    if _BANDS_CACHE is not None:
        return _BANDS_CACHE
    import json, os
    cands = ([AGE_MATRIX_JSON] if AGE_MATRIX_JSON else []) + [
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'Runtime Matrices', 'A_Scoring_Module',
                     'age_reference_matrix.json'),
    ]
    for p in cands:
        if p and os.path.exists(p):
            try:
                d = json.load(open(p))
                out = {c: [(r['age_midpoint'], r['A_mean'], r['A_p10'], r['A_p90'])
                           for r in d.get(c, [])] for c in _BASELINE_CLASSES}
                if all(out[c] for c in _BASELINE_CLASSES):
                    _BANDS_CACHE = out
                    return out
            except Exception:
                pass
    _BANDS_CACHE = {}
    return _BANDS_CACHE


def H_min_for(cls: str, sub: str = 'methyl') -> float:
    return H_MIN_TABLE[cls][SUB_ORDER.index(sub)]


def a_ceiling(cls: str, sub: str = 'methyl') -> float:
    """Physical maximum A on this class x substrate: 1/H_min (beta=0.5, H=1)."""
    return 1.0 / H_min_for(cls, sub)


def a_score(mean_beta: float, cls: str, sub: str = 'methyl') -> float:
    """THE GAUGE. A = H(one mean beta) / H_min(class, substrate).

    mean_beta is expected on the IAMAtlas scale (Stage-1 normalized). No offset:
    healthy is the age-matched band, read downstream, not a shift applied here.
    """
    return H(mean_beta) / H_min_for(cls, sub)


def _decade(age: int) -> str:
    if age is None:
        return '40-49'  # neutral middle-adult default
    if age >= 90:
        return '90+'
    lo = (int(age) // 10) * 10
    return f'{lo}-{lo + 9}'


def _interp(pts: List[Tuple[float, float]], age: Optional[float]) -> float:
    """Linear interpolation of (age, value) points at age, clamped to the ends."""
    if age is None:
        age = 44.0
    if age <= pts[0][0]:
        return pts[0][1]
    if age >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if age <= pts[i][0]:
            (x0, y0), (x1, y1) = pts[i - 1], pts[i]
            return y0 + (y1 - y0) * (age - x0) / (x1 - x0)
    return pts[-1][1]


def age_band(cls: str, age: Optional[int]) -> Tuple[float, float, float]:
    """Age-matched (A_p10, A_mean, A_p90) for this class.

    Uses the canonical age_reference_matrix percentiles when reachable; otherwise
    synthesizes p10/p90 from the embedded A_mean +/- 1.2816*A_sd.
    """
    bands = _load_bands()
    if bands.get(cls):
        pts = bands[cls]
        mean = _interp([(a, m) for a, m, lo, hi in pts], age)
        p10 = _interp([(a, lo) for a, m, lo, hi in pts], age)
        p90 = _interp([(a, hi) for a, m, lo, hi in pts], age)
        return (p10, mean, p90)
    mean = healthy_baseline(cls, age)
    sd = HEALTHY_SD[cls]
    return (mean - _Z90 * sd, mean, mean + _Z90 * sd)


def healthy_baseline(cls: str, age: Optional[int]) -> float:
    """Age-matched expected healthy A for this class (the NORMAL-band center)."""
    return HEALTHY_BASELINE[_decade(age)][_BASELINE_CLASSES.index(cls)]


def placement(A: float, cls: str, age: Optional[int]) -> str:
    """Where the patient sits vs age-matched healthy peers."""
    p10, mean, p90 = age_band(cls, age)
    if A < p10:
        return 'BELOW_BAND'
    if A > p90:
        return 'ABOVE_BAND'
    return 'IN_BAND'


def tier(A: float, cls: Optional[str] = None, age: Optional[int] = None) -> str:
    """Absolute elevation severity (Issue 002 ladder). Following GAPE_WEB_v13, the
    tier is the elevation alarm: healthy sits below the threshold and reads NORMAL,
    including when it is mildly below its age cohort (a batch/normalization dip is
    not an alarm). Only a genuine FAR-below-cohort reading — identity loss such as
    seminoma (0.67) or senescence — reads INVERSION, a real finding. The age-cohort
    placement (below/in/above band) is a separate context axis, not a tier."""
    if A >= 1.10:
        return 'BREACH'
    if A >= 1.07:
        return 'URGENT'
    if A >= 1.05:
        return 'DETECTABLE'
    if A >= 1.01:
        return 'MARGINAL'
    if cls is not None:
        p10, mean, p90 = age_band(cls, age)
        sd = max((mean - p10) / _Z90, 1e-6)
        if A <= p10 - 2.0 * sd:      # genuine identity loss, far below cohort
            return 'INVERSION'
    return 'NORMAL'


def departure(A: float, cls: str, age: Optional[int]) -> float:
    """A minus the age-matched healthy mean. >0 elevated, <0 suppressed."""
    return A - age_band(cls, age)[1]


def read(mean_beta: float, cls: str, age: Optional[int] = None,
         sub: str = 'methyl') -> Dict[str, object]:
    """Full doctor readout for one class/cell: the A-score, its age-matched band,
    placement vs peers, absolute severity tier, departure, and the ceiling."""
    A = a_score(mean_beta, cls, sub)
    p10, mean, p90 = age_band(cls, age)
    return {
        'A': A, 'class': cls, 'age': age, 'substrate': sub,
        'band': {'p10': p10, 'mean': mean, 'p90': p90},
        'placement': placement(A, cls, age),
        'tier': tier(A, cls, age),
        'departure': A - mean,
        'healthy_baseline': mean,
        'ceiling': a_ceiling(cls, sub),
        'structurally_saturated': is_structurally_saturated(cls, sub),
        'age_decade': _decade(age),
    }


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


def cellular_age(A: float, cls: str) -> Optional[float]:
    """Invert the age-matched A_mean(class, age) curve: the age at which a HEALTHY
    <class> cell reads this A. For immune and most classes A_mean rises with age
    (cells lose fidelity, beta drifts toward the H_min anchor), so a higher A reads
    as OLDER cells. Interpolates within the table and extrapolates past the ends
    using the end slope. Returns the cellular age in years, or None if unavailable.

    NOTE: accuracy depends on Stage-1 normalization — an un-normalized sample that
    reads a few hundredths off in beta will bias the inferred age; read alongside
    the placement/departure, not in isolation.
    """
    bands = _load_bands()
    if not bands.get(cls):
        return None
    pts = [(a, m) for a, m, lo, hi in bands[cls]]
    ages = [p[0] for p in pts]
    means = [p[1] for p in pts]
    if means[-1] == means[0]:
        return None
    ascending = means[-1] > means[0]
    # below the youngest reference -> extrapolate with the first slope
    if (ascending and A <= means[0]) or (not ascending and A >= means[0]):
        s = (means[1] - means[0]) / (ages[1] - ages[0])
        return ages[0] + (A - means[0]) / s if s else None
    # above the oldest reference -> extrapolate with the last slope
    if (ascending and A >= means[-1]) or (not ascending and A <= means[-1]):
        s = (means[-1] - means[-2]) / (ages[-1] - ages[-2])
        return ages[-1] + (A - means[-1]) / s if s else None
    for i in range(1, len(pts)):
        m0, m1 = means[i - 1], means[i]
        if (m0 <= A <= m1) or (m1 <= A <= m0):
            a0, a1 = ages[i - 1], ages[i]
            return a0 + (a1 - a0) * (A - m0) / (m1 - m0) if m1 != m0 else a0
    return None


def overall_cellular_age(class_ascores: Dict[str, object],
                         chronological_age: Optional[int] = None,
                         weights: Optional[Dict[str, float]] = None) -> Optional[Dict]:
    """Overall cellular age = mean per-class cellular age over assessable classes.

    class_ascores accepts {class: A_float} or {class: {'A': ...}} (a stage-4 rec).
    Returns {cellular_age, per_class, n, chronological_age, delta_years} where
    delta_years = cellular_age - chronological_age (>0 = cells read older than age).
    """
    per = {}
    for cls, v in class_ascores.items():
        a = v.get("A") if isinstance(v, dict) else v
        if a is None:
            continue
        try:
            a = float(a)
        except (TypeError, ValueError):
            continue
        if a != a:  # NaN
            continue
        ca = cellular_age(a, cls)
        if ca is not None:
            per[cls] = ca
    if not per:
        return None
    if weights:
        num = sum(per[c] * weights.get(c, 1.0) for c in per)
        den = sum(weights.get(c, 1.0) for c in per)
        overall = num / den if den else None
    else:
        overall = sum(per.values()) / len(per)
    out = {"cellular_age": overall, "per_class": per, "n": len(per),
           "chronological_age": chronological_age}
    if overall is not None and chronological_age is not None:
        out["delta_years"] = overall - float(chronological_age)
    return out


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
        abs(a - 1.0502) < 1e-3 and tier(a, 'cycling', 55) == 'DETECTABLE', f"A={a:.4f}")
    # healthy cycling at its age-matched beta reads IN its age band (not a fixed line)
    a_h = a_score(0.720, 'cycling')
    chk("gauge: healthy cycling beta~0.72 age55 -> IN_BAND NORMAL",
        placement(a_h, 'cycling', 55) == 'IN_BAND' and tier(a_h, 'cycling', 55) == 'NORMAL',
        f"A={a_h:.4f} band={tuple(round(x,3) for x in age_band('cycling',55))}")
    # young healthy immune sits at ~0.906 — BELOW a fixed 0.95 line, but IN its age band
    a_yi = a_score(0.780, 'immune')
    chk("age matters: young immune ~0.906 is IN_BAND (would false-invert on fixed 0.95)",
        placement(a_yi, 'immune', 4) == 'IN_BAND', f"A={a_yi:.4f} band p10={age_band('immune',4)[0]:.3f}")
    a_sem = a_score(0.18, 'stem_pluri', 'methyl')
    chk("inversion: seminoma beta=0.18 stem_pluri -> A<0.75 INVERSION",
        a_sem < 0.75 and tier(a_sem, 'stem_pluri', 40) == 'INVERSION', f"A={a_sem:.4f}")
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
