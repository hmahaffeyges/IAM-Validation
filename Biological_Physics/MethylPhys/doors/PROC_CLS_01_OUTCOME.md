# PROC-CLS-01 — outcome: the residual sky HAS large-scale structure. The reference is NOT COMMISSIONED.

**Sealed 2026-09-25** against the bars in [`PROC_CLS_01_PREREG.md`](PROC_CLS_01_PREREG.md), fixed before any
spectrum was computed. 318 healthy whole-blood arrays, four laboratories, each one's residual sky built by
`stage_4_6_patient_sky` exactly as a report builds it, then `healpy.anafast` to ℓ = 255 and 20 within-mask
permutations as its own null. Evidence: [`PROC_CLS_01.json`](../kit/results/PROC_CLS_01.json).
Scripts: [`PROC_CLS_01_measure.py`](../kit/PROC_CLS_01_measure.py) ·
[`PROC_CLS_01_analyse.py`](../kit/PROC_CLS_01_analyse.py).

## The finding, which is the part worth reading

**The methylome's residual sky is not spatially random, and its structure is large-scale.**

| ℓ band | median C_b ÷ its own permutation null | arrays outside their null |
|---|---|---|
| **2–8** | **3.509** | 94.3 % |
| 9–24 | 1.397 | 92.8 % |
| **25–64** | 1.165 | **96.9 %** |
| **65–128** | 1.086 | **97.5 %** |
| 129–191 | 1.026 | 78.9 % |
| 192–255 | 0.992 | 31.1 % |

The null shuffles each specimen's own pixel values among its own unmasked pixels: same mask, same pixel
count, same one-point distribution, spatial arrangement destroyed. So this excess is spatial structure and
cannot be an artefact of coverage or of heavy tails. **B2 is met** — two bands put more than 95 % of arrays
outside their own null.

**And it is the opposite of what I expected.** Because the projection is genomically local — neighbouring
pixels hold neighbouring CpGs, median 511 bp within a pixel — I expected the excess at *small* angular
scales, where a pixel's neighbours are its genomic neighbours. The measurement says the excess is at the
*largest* scales and has died out by ℓ ≈ 200. Whatever organises the residual is organised over very long
genomic ranges, not over kilobases.

**The leading interpretation, which this procedure did not test.** The projection walks the genome in order,
so a whole-chromosome offset — sex chromosomes being the obvious candidate in a mixed-sex cohort, and
chromosome-scale compartment structure the next — appears as power at the lowest multipoles. That is a
specific, cheap test (mask chrX and chrY, re-run, see whether the ℓ 2–8 excess survives) and it is **not run
here**, because choosing a new configuration after seeing the spectrum is how a result gets talked into
existence. It is the obvious next procedure.

## The verdict

| bar | result | |
|---|---|---|
| **B1** f_sky ≥ 0.5 on ≥ 95 % of arrays | **0 % of arrays**; median f_sky 0.433, range 0.302–0.447 | **NOT MET** |
| **B2** structure beyond the permutation null | two bands above 95 %, ℓ 2–8 at 3.5× its null | MET |
| **B3** leave-one-laboratory-out coverage | Uppsala 0.754 · Karolinska 0.765 · **UCLA 0.906** · **Munich 0.650**; worst band 0.500 | **NOT MET** |
| **B4** not the scalar in disguise (\|r\| < 0.9 vs \|z_immune\|) | every candidate 0.18–0.58 | MET |
| **B5** not measuring the mask (\|r\| < 0.5 vs f_sky) | band powers 0.01–0.29; **high/low ratio −0.478**, inside the bar but close to it | MET, narrowly for the ratio |
| **B6** nothing in service moves | no chain file touched; no reading recomputed | MET |

**Decision rule, applied.** B3 fails, so: *"the spectrum is real but not yet transferable; it is published as
a diagnostic and not as a reference."* No band-power reference is adopted, the report gains no spectrum
panel, and `CLS` stays `NOT_BUILT` in the CMB tool register — now with a measurement behind that state
instead of an absence.

## On B1, which failed for a reason that is mine

The f_sky floor of 0.5 was set without checking what the platform can cover, and **no array could have met
it**: these are 450K arrays projected onto a pixel grid built for the EPIC∪450K union, so roughly 44 % is
the ceiling, not a defect of any specimen. The bar is wrong, not the data.

I am not moving it. A threshold that moves after the numbers are visible is worth nothing, and the one thing
that makes these procedures worth reading is that they cannot be edited into agreement. What should happen
instead: a new pre-registration with a **platform-aware** floor — f_sky as a fraction of the platform's own
maximum, not of the whole sphere — and the rest of the bars unchanged. That is the author's call.

## What would make the reference transferable

B3 failed in **both directions**, which is informative. Munich is under-covered (0.650: its spectra fall
outside a band built on the other three more often than they should) and UCLA is over-covered (0.906: its
spectra sit inside more often than they should). A band that is simultaneously too tight for one laboratory
and too loose for another is a **per-laboratory scale problem**, not a noise problem — the four residual
scales were each built on their own 40-array panel, and the spectrum is quadratic in the residual, so a 10 %
error in `s_lab` moves every band power by 20 %.

So the honest next step for the reference is not more arrays: it is **per-laboratory band powers**, or a
common residual scale, exactly as enhancement A3 proposes for the scalar band. Worth doing after A3, not
before — the two share a cause.
