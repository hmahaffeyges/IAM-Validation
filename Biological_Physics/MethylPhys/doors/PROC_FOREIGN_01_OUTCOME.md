# PROC-FOREIGN-01 — outcome: COMMISSIONED. The immune tier is withheld when a specimen is not whole blood.

**Sealed 2026-09-25** against the bars in [`PROC_FOREIGN_01_PREREG.md`](PROC_FOREIGN_01_PREREG.md), fixed
before any mixture was scored. 318 healthy whole-blood arrays × three non-blood classes × five mixing
fractions = **5,088 scored mixtures**, all through the chain's own path.
Evidence: [`PROC_FOREIGN_01.json`](../kit/results/PROC_FOREIGN_01.json) ·
[`PROC_FOREIGN_01_mixtures.json`](../kit/results/PROC_FOREIGN_01_mixtures.json) (every mixture) ·
scripts [`PROC_FOREIGN_01.py`](../kit/PROC_FOREIGN_01.py) · [`PROC_FOREIGN_01_analyse.py`](../kit/PROC_FOREIGN_01_analyse.py).

**This is the first change adopted into the chain from the enhancement list.**

## The verdict

| bar | result | |
|---|---|---|
| **B1** foreign material corrupts the reading | median \|ΔA″\| exceeds 2σ at **f = 0.20** for secretory and terminal | MET |
| **B2** the tier actually flips | at that f, **96.9 %** (secretory) and **68.6 %** (terminal) of hosts change tier | MET |
| **B3** a detector the chain already computes | sensitivity **1.000** at both points | MET |
| **B4** it fires before the tier misreads | at every f where ≥ 10 % flip, the guard catches **95.9–100 %** of exactly those hosts | MET |
| **B5** silent on healthy specimens | fires on **15 of 318 = 0.0472** (bar ≤ 0.05) | MET |
| **B6** the reading does not move | immune A″ vs PROC-BAND-01: **max \|ΔA″\| = 0.000e+00** on 318 arrays | MET |

**Decision rule, applied:** all bars met, so the withholding rule is adopted.

## What was adopted, exactly

The detector is not new: `stage_4_6_patient_cmb` already defines the blood lineage as *immune, progenitor,
stem_adult*, and the composition step already reports every class fraction. The guard is their complement —
**the share of the specimen assigned outside the blood lineage** — with its threshold in a runtime matrix
([`composition_guard_v1.json`](../chain/Runtime%20Matrices/A_Scoring_Module/composition_guard_v1.json)),
not in code:

| | |
|---|---|
| threshold | **0.0207** — the largest value firing on at most 15 of the 318 healthy arrays |
| where it acts | `stage_b_identity`, at the same place the chain already withholds for a missing laboratory zero |
| what is withheld | the **tier word only**. A″, its placement and the composition are all still reported |
| what a reader is told | *"composition unverified: 18.5 % of this specimen is assigned outside the blood lineage, above the 2.1 % the gauge was commissioned for. The reading stands; no tier is printed."* |
| fallback | a specimen whose composition cannot be computed is **not** withheld by this guard, and the field says so |

**Verified end to end on a real array, not only in the analysis.** The same Uppsala donor, run through
[`run_sample.py`](../chain/MethylPhys_Interface/run_sample.py) twice:

| | A″ | tier | foreign fraction |
|---|---|---|---|
| unspiked | 1.0162 | **NORMAL** | −0.0001 |
| + 20 % secretory | 1.0750 | **withheld** | 0.185 |

A″ of 1.075 is above the band: without the guard that specimen prints **ELEVATED**. That is the misread this
procedure exists to prevent, and it is now prevented on a specimen that was never part of the calibration.

## The finding worth keeping: not all foreign material is equal

| foreign class | median \|ΔA″\| at f = 0.35 | tiers flipped | caught |
|---|---|---|---|
| **secretory** | 0.1129 (2.70 × 2σ) | 100 % | 100 % |
| **terminal** | 0.0758 (1.81 × 2σ) | 98.7 % | 100 % |
| **stromal** | 0.0201 (**0.48** × 2σ) | 16.7 % | 98.1 % |

**Stromal material barely perturbs the immune identity gauge** — a third of the specimen moves the reading by
less than one healthy standard deviation. Epithelial and terminal material is what corrupts it. That is a
useful asymmetry for whoever collects the specimens: connective-tissue contamination in a draw is far less
dangerous to the reading than epithelial or tumour material, and it is the epithelial case the guard catches
at 100 %.

## Two things stated rather than glossed

**A threshold-rule correction, made before the verdict was written.** The pre-registration fixed the *rule* —
*a threshold giving ≤ 0.05 false positives on the healthy arrays, set on those arrays alone*. The first
implementation used the 95th percentile, which **overshoots** that constraint: with 318 arrays the achievable
rates are k/318, and the 95th percentile fires on 16 (0.0503). The threshold satisfying the pre-registered
constraint is the one firing on 15 (0.0472). Both were computed from the healthy arrays only, with no
reference to spiked performance, and — this is what makes the correction safe rather than convenient — the
two give **identical** worst-case sensitivity, 0.959. Nothing was bought by the choice.

**The mixtures are optimistic foreign specimens.** An atlas class mean is smoother than real tissue. The
guard passing here does not establish that it catches real tumour material at the same rate, and the
real-tissue check is a named follow-up: the EPIC adenoma in the test package is withheld today for a
*different* reason (no laboratory zero for its laboratory), so it cannot yet test this guard.

## What adoption cost, measured on the commissioning arrays

Running the eleven commissioning arrays through the guard: **six keep their tier, five are withheld.** Four of
the five are the tissue and tumour specimens (foreign 0.65–0.80), which is the guard doing its job. The fifth
is **GSM2333950 — a blood array, 96.6 % immune, foreign 0.0282** — withheld because it sits just above the
threshold. That is not a defect: it is one instance of the ≤ 5 % healthy false-positive rate the
pre-registration fixed and B5 measured at 0.0472. It does mean the sealed gauge-switch conformance test,
which expected a tier on every healthy blood array, now accepts a withheld tier when
`composition_verified` is False — the change is recorded in that test file with this procedure named, and it
still demands a tier whenever the composition *is* verified.

### The one judgement left open, with its price

| threshold | healthy arrays withheld | worst detection of hosts whose tier would misread |
|---|---|---|
| **0.0207 (commissioned)** | 4.72 % | **0.959** |
| 0.0300 | 1.89 % | 0.811 |
| 0.0500 | 0 % | 0.358 |
| 0.0800 | 0 % | 0.038 |

A threshold of 0.03 would keep GSM2333950's tier and withhold from only one healthy array in fifty — but it
catches 81.1 % of the specimens that misread, **below the 0.90 that B4 fixed**. So 0.03 cannot be adopted
under this pre-registration: it would need a new one with B4 set at 0.80, and that is a clinical judgement
about which error is worse, not an analytical one. The commissioned value stands at 0.0207 until the author
decides otherwise, and moving it is a one-number change in
[`composition_guard_v1.json`](../chain/Runtime%20Matrices/A_Scoring_Module/composition_guard_v1.json).