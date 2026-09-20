# OUTCOME — LAB-ZERO-02: the fourth lab decides the lab-zero route

**Run 2026-09-20.** GSE111629 UCLA, 237 `PD-free control` arrays (PD arrays not opened), per-sample IDAT fetch (1.9 GB, 10 min), Stage 1 six-wide (237/237, 43 min), Walther presence gate, identity-loci gauge, map `stage1_noob_450K` frozen, control-probe features and ridge λ = 1 exactly as LAB-ZERO-01. **Against PREREG.md as sealed.**

## Verdict: **P4 FAIL, P4b FAIL, P6 FAIL. The control-probe route carries direction but not magnitude; the per-lab healthy-control panel (CLSI EP28) is the lab zero.**

| test | bar | observed | verdict |
|---|---|---|---|
| P1 | ≥ 95 % | 237/237 | PASS |
| P2 presence | ≥ 90 % | 86.1 % (n = 204) — older cohort (median ~70), more samples below the immune+haem-prog ≥ 0.85 gate | miss, recorded |
| P3 median mapped immune A | report | **0.9401** (unmapped 0.7570) | map moved it +0.18; cohort constant **−0.0457** vs Uppsala |
| **P4** UCLA offset predicted from Uppsala + Karolinska + Munich | \|err\| ≤ 0.010 | pred −0.0202 vs obs −0.0457, **err 0.0256** | **FAIL** |
| **P4b** leave-one-out over four labs | all ≤ 0.010 | Karolinska 0.0041 · Uppsala 0.0180 · UCLA 0.0256 · Munich 0.0517 | **FAIL** (1/4) |
| **P6** within-cohort correction (fit on three labs, applied to UCLA) | sd narrows ≥ 20 %, median shift ≤ 0.005 | sd 0.0203 → 0.0177 (**12.6 %**), shift −0.0011; in-band 5.2 % → 2.1 % (v2 band, UCLA sits below it) | **FAIL** |
| N-plate | report | 38 Sentrix chips, F = 1.79, p = 0.0089 | modest chip effect |
| N-ethnicity | descriptive | Hispanic (n = 18) − Caucasian: signed d −0.33 | recorded; not interpretable at n = 18 |
| N-sex signed d(f−m) | report | 40s −0.10 · 60s −0.18 · 70s −0.07 · 80s −0.55 — women LOWER (as Uppsala; opposite to Karolinska) | sign remains lab-dependent; no split |

## Four labs on one scale (mapped immune identity-loci A, cohort median vs Uppsala)
| Uppsala SE | Karolinska SE | Munich DE | UCLA US |
|---|---|---|---|
| 0 | +0.024 | −0.021 | −0.046 |

Each constant is flat across age within its cohort. The pipeline map (Stage 1s) is common to all four and remains COMMISSIONED; the constants are what the map does not remove.

## Reading
1. **Control probes see direction, not size.** Four labs, four correct signs; only the two Swedish labs — the most alike — predict each other within the bar. Adding a fourth lab widened the error spread rather than tightening the model. The housekeeping channels record part of a lab's chemistry (normalisation and conversion controls) but not the pre-analytical part (DNA input, bisulfite batch, storage) that evidently carries much of the constant. This is the honest end of the route for now.
2. **The within-cohort narrowing is real but small** (13 % on an unseen lab vs the 20 % bar). Not worth a chain stage; noted for a future revisit with more labs.
3. **The LAB ZERO is the healthy-control panel**, as CLSI EP28 prescribes: 20–30 healthy arrays per lab, run once through the same Stage 1, median mapped immune A subtracted, the offset printed on every report. The author chose to try the control-probe route first and fall back to the standard if it failed; it failed cleanly and in the direction that leaves the floor and the map untouched.
4. The P2 miss (86 %) in an older cohort says the presence gate needs an age-aware look before the band is finalised — recorded as a commissioning note, not changed here.

---
**SEALED** sha256 `b9cff800569de4b946d7f48019621ea1423856a18840fc2ef4d6d08c790c5648` · 2026-09-20
