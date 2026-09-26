# PROC-TARE-01 — pre-registration: can the array's own known-value probes tare the instrument, so that no healthy panel defines where A = 1.0 sits?

**Written 2026-09-26, before any SNP or control probe was read for this purpose.** Nothing below moves after results
are visible.

## The question, in the author's words

*"We are looking more for a scale tare, before we weigh an object … rather than a calibration by comparison."*

The laboratory zero as built is `z_lab = median(A − c(age)) − 1.0` over a panel of ≥ 40 healthy arrays from one
laboratory (lab_zero.compute_lab_zero). Read literally, that construction **assumes the panel's median person reads
exactly 1.0** and shifts the laboratory until they do — a population defining zero, which is the cohort methodology
this instrument is meant to replace. The four commissioned zeros are small (GSE87571 −0.0117, GSE42861 +0.0084,
GSE111629 −0.0673, GSE125105 −0.0346; three of four already inside NORMAL 0.95–1.04 with no zero), but the logic is
the problem, not the size.

A tare reads the instrument on a **known input** at the moment of measurement. The array carries two:

1. **SNP probes** (`rs…`, 65 on 450K, 59 on EPIC v1). They read genotype, not methylation, so their β is exactly
   **0, 0.5 or 1** by construction. Where they land on a given chip is that chip's error at three known points —
   offset and scale — with no person's health involved.
2. **Control probes** (bisulfite conversion, extension, hybridisation, negative, non-polymorphic). Stage 1 (`noob`)
   already uses the out-of-band probes for background and dye bias; this procedure asks whether the *result* sits where
   it should.

## What is measured

On every array of the four commissioned laboratories' healthy panels (the 48-array null used by PROC-MF-01/02/03,
plus the full 732 GSE87571 arrays where the tare is cheap to compute), from the raw IDATs through the chain's own
Stage 1:

- **T_offset** = median over the SNP probes assigned to the 0.5 cluster of (β − 0.5)
- **T_scale** = (median of the "1" cluster − median of the "0" cluster), the instrument's gain against the ideal 1.0
- cluster assignment by nearest ideal value after a first pass, then re-assigned once (two iterations, fixed here)
- the per-array tare applied to the identity-locus betas as β′ = (β − T_offset·k(β)) / T_scale, with k the
  standard linear correction toward the cluster centres; the exact form is fixed in PROC_TARE_01.py before the
  first run and not changed after
- **A on the identity surface, per array, with and without the tare, with no laboratory zero applied**

## Bars

| bar | what must be true | why |
|---|---|---|
| B1 | on the four panels, the **laboratory medians of untared A** sit where the four commissioned zeros predict: median(A − c(age)) − 1 within ± 0.005 of z_lab for each laboratory | confirms the measurement reproduces the zeros it is trying to explain before explaining them |
| B2 | applying the per-array tare moves each laboratory's median A **toward 1.0**, and the largest offset (GSE111629, −0.067) shrinks by **≥ 50 %** | the claim: the offset is instrument, and the array's own controls see it |
| B3 | after taring, **all four** laboratory medians sit inside NORMAL (0.95–1.04) with no zero applied | the practical claim: a single array can be read on the physics scale with no panel |
| B4 | the tare **tightens or holds** the within-laboratory spread of healthy A: post-tare SD ≤ pre-tare SD in ≥ 3 of 4 laboratories | a correction that adds noise is not a tare |
| B5 | the tare is **specific to the instrument**: T_offset and T_scale do not correlate with age or sex (|r| < 0.15 on the 732 GSE87571 arrays) | if the "tare" tracks biology it is not a tare |
| B6 | the **Sentrix-chip term**: within one laboratory, the chip-to-chip spread of median A falls after taring (≥ 20 % reduction on GSE87571's 732 arrays, which span many chips) | the chip term was named the next residual after the gauge switch; a tare should see it |
| B7 | the commissioned reading is **unchanged where it should be**: on the 11 commissioning arrays, the tared per-cell A differs from the untared by no more than the tare itself predicts (no other path touched) | the instrument-unchanged check |

**Decision rule.** B1–B4 met → the laboratory zero is **retired as a construction**: Stage 1s applies the per-array
tare, `stage_b_identity` receives `lab_zero=None` by design, [`lab_zero.py`](../chain/lab_zero.py) is retired, and the healthy panel of 40 is
required **only** for the Stage 2d detection noise floor. A laboratory whose panel median falls outside NORMAL after
taring is **flagged** (`LAB_OFFSET_UNEXPLAINED`) and never re-centred. B2 met but B3 failed → the tare is adopted as a
partial correction and the remainder is reported per laboratory as an unexplained offset, not corrected. B2 failed →
the offset is not visible to the array's own controls; the zero stays, its cohort-anchored construction is stated in
the SOP as a known limitation, and a physical reference material becomes the named route. Any bar not run is
recorded NOT ASSESSED, never passed by argument. B5–B7 failing with B1–B4 met → not adopted; the failing bar is
recorded and the author decides.

## What this does not do

It does not change what healthy is. A = 1.00 ± 5 % is the physics; this procedure only asks whether the scale can be
zeroed on itself before the object is placed on it.

## Evidence files (named before they exist)

In the kit folder: **PROC_TARE_01.py**. In the kit results folder: **PROC_TARE_01.json**, **PROC_TARE_01_per_array.parquet**. In the plates folder: **PROC_TARE_01.png**. Linked from the outcome once they exist; not linked here because a link to a file that does not yet exist is a broken link.
