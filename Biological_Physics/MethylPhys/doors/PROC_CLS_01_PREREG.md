# PROC-CLS-01 — does the residual sky have scale structure worth reporting?

**Pre-registered 2026-09-25, before any spectrum was computed.** Enhancement B2, the second item on
[`ENHANCEMENTS.md`](ENHANCEMENTS.md) and the first borrowed method on the shelf: *"is a departure locally
clustered along the genome, or spread across it? A focal lesion and a systemic process should look different,
and nothing in the chain currently asks."*

## The question, and why an angular power spectrum is the right object

The chain already builds a residual sky per specimen — `z = (β − Σ_c f_c μ_c − m_lab) / s_lab` per CpG,
projected to 196,608 HEALPix pixels in genomic order. Today it is **looked at**: rendered as a plate and
summarised by the fraction of pixels beyond |z| = 2. Neither of those distinguishes a departure concentrated
in a few genomic neighbourhoods from the same total departure spread evenly across the genome, and that
distinction is the difference between a focal process and a systemic one.

The angular power spectrum is what cosmology built for exactly this question: it decomposes a map into the
scales its structure lives on. Two properties of this sky make it applicable rather than decorative, and both
were measured before today: the projection is **genomically local** (196,608 of 196,608 pixels hold
contiguous CpGs from one chromosome, median span 511 bp) and neighbouring pixels are genomically adjacent
(median gap 1,256 CpGs, a quarter within 10). So multipole is a genomic scale, statistically.

**The honest caveat, stated first.** The sky is masked — pixels with no assessable CpG carry nothing — so what
is computed here is a **pseudo-spectrum**, not a mask-deconvolved one. No MASTER-style deconvolution is
attempted; the mask's effect is instead controlled by the null in B2 and the confound test in B5, both of
which use the same mask as the data.

## What enters

| | |
|---|---|
| Arrays | the **318 healthy whole-blood arrays** of PROC-BAND-01, four laboratories, from [`reference_data/`](../reference_data/) |
| Sky | `stage_4_6_patient_cmb.patient_sky` as the chain calls it — the same residual, mask and projection a report draws |
| Residual scales | the four commissioned per-laboratory files in `Runtime Matrices/Patient_CMB/`. **Not refitted** |
| Transform | `healpy.anafast` on the masked map, ℓ = 2 … 255 (ℓmax = 2·nside). Binned into six logarithmic bands |
| Nothing from a patient | this procedure measures healthy spectra only. It makes no disease claim and produces no reading |

## The bars, fixed now

**B1 — the spectrum is computable on real specimens.** Every array must yield a finite `C_ℓ` across the band
range with an unmasked sky fraction **f_sky ≥ 0.5**. Fewer than 95 % of arrays meeting that is a failure.

**B2 — there is structure to measure at all.** For each array, the spectrum is compared with a **within-mask
permutation null**: the same pixel values shuffled among the unmasked pixels, which destroys spatial
structure while keeping the mask, the pixel count and the one-point distribution exactly. In **at least one ℓ
band**, ≥ 95 % of arrays must fall outside the [2.5, 97.5] percentile range of their own null (20 shuffles
per array). If no band clears this, the residual sky has **no detectable scale structure** at this resolution
and the spectrum is not worth reporting — which is a publishable answer, not a disappointment.

**B3 — it reproduces across laboratories.** Leave-one-laboratory-out on band powers: build the healthy
p10–p90 interval per band from three laboratories, measure the held-out laboratory's coverage. The mean
coverage across bands must be in **[0.70, 0.90]** and no single band below **0.60** — the same interval the
immune band was held to, with a floor added because six bands give six chances to fail.

**B4 — it is not the scalar in disguise.** The adopted summary statistic must have **|Pearson r| < 0.9**
against |z_immune| across the 318 arrays. A spectrum that tracks the reading already on the page adds a
picture, not a measurement.

**B5 — the mask is not making the shape.** The adopted summary must have **|Pearson r| < 0.5 against f_sky**.
A statistic that tracks how much sky was visible is measuring coverage.

**B6 — nothing in service moves.** This adds an observable and changes no existing number; immune A″ on these
arrays must be unchanged from PROC-BAND-01 to 1e-9.

## Decision rule

- **B1–B5 met:** the healthy band-power reference is adopted, the register gains a row, and the report gains
  a spectrum panel — *subject to the dependency question below, which is a cost, not a result.*
- **B2 fails:** **NOT COMMISSIONED**, and the published statement is that the residual sky is consistent with
  spatially unstructured noise at nside 128 — which tells the next reader not to spend a week on the
  higher-order statistics (bispectrum, Minkowski functionals) that all assume structure at this scale.
- **B3 fails:** the spectrum is real but not yet transferable; it is published as a diagnostic and not as a
  reference, exactly as PROC-BAND-01's band was.
- **B4 or B5 fails:** the statistic is redundant or is measuring the mask. Published, not adopted.

## A cost to name before it is a surprise

The chain has **no healpy dependency today** — `stage_4_6_patient_cmb` implements its own pixel geometry
specifically to avoid one. A spherical-harmonic transform is not something to hand-roll: if this is adopted
into the report, either healpy joins `chain/requirements.txt` (it pulls a compiled FITS library with it) or
the transform is precomputed. That decision is the author's and belongs after the measurement, not before.
