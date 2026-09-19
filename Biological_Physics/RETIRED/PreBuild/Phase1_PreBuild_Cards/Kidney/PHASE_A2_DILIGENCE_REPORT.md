# Kidney sprint — Phase A.2 atlas acquisition diligence report

**Date:** 2026-05-03
**Author:** Walther
**Status:** PARALLEL ACQUISITION HALTED PENDING HEATH SIGN-OFF
**Rule honored:** No half-assing. Diligence before download commitment, not after.

---

## What I did

Per the prior chat's locked direction, I started parallel acquisition on three candidate KidneyRef v2 sources (GSE50874 + GSE59157 + bioRxiv 2026 cross-species kidney atlas). I fetched the GEO series matrix metadata for both GEO accessions, parsed the embedded β matrices, ran CHK-3.1A bimodality checks, and verified the sample structure against the prior chat's claims.

**Two of the three Tier 1 candidates are not what the prior chat thought they were.** Surfacing this before I commit to a build target.

---

## Finding 1 — GSE50874 is NOT cell-type-resolved

**Prior chat's claim:** "91 microdissected human kidney tubule samples on HM450K. CKD vs control. De Ridder 2024 benchmarking validated as kidney mixture-generation reference. Microdissected tubule = enriched for proximal tubule + distal nephron cells → closer to a real cell-of-origin reference than the EpiSCORE 32-marker matrix."

**What the GEO metadata actually says:**

| Field | Value |
|---|---|
| Total samples | **n=85** (not 91) |
| Series_title | "Methylation Profiling of Human Kidney Tubules" |
| `Sample_characteristics_ch1` | `tissue: kidney tubule` — **single value, all 85 samples** |
| `Sample_source_name_ch1` | `kidney 1` through `kidney 85` — donor-numbered |
| Sample title pattern | `genomic DNA from microdissected human kidney N` for N=1..85 |
| `Sample_data_processing` | `GenomeStudio V2011.1; methylation module 1.9.0` |
| Platform | GPL13534 (HM450K) |
| Lab | Susztak (UPenn) |
| PubMed | 24098934 (Ko 2013), 28556588 (Ko 2017) |

**What this means:**

1. **No CKD-vs-control labels in GEO metadata.** Every sample is just `tissue: kidney tubule`. The CKD-vs-control assignments — if they exist — live in the linked publications' supplementary tables, not in the sample annotations on GEO.

2. **No cell-type stratification.** All 85 samples are bulk microdissected tubule tissue from a single donor each. There is no proximal-tubule vs distal-tubule vs podocyte vs endothelial sub-cell-type breakdown in this cohort. The microdissection separates tubular compartment from glomerular compartment at the tissue level, but each sample is still a mixture of multiple tubular cell types.

3. **Substrate is GenomeStudio AVG_Beta, not TCGA HM450 sesame Level 3.** Per cookbook substrate calibration (CHK-0.7), GenomeStudio AVG_Beta is **within-cohort self-cal only** — not calibrated against the VAL-106 TCGA HM450 sesame Level 3 anchor. To use GSE50874 as a calibrated atlas substrate, we'd need to re-process from raw IDATs (`GSE50874_RAW.tar`, available) using sesame, OR build it as a within-cohort-self-cal atlas with explicit DISC-CARDIO-005-style documentation.

4. **CHK-3.1A first-1000-CpG check:** f_extreme = 0.387 (raw-EPIC range 0.30-0.40; TCGA HM450 sesame range 0.50-0.56), f_middle = 0.097 (raw-EPIC threshold ≤0.10, marginal). This β distribution is consistent with **GenomeStudio AVG_Beta substrate** (less bimodal than TCGA sesame but bimodal enough for self-cal).

**Honest interpretation:** GSE50874 is a high-quality bulk-tubule kidney cohort but is NOT a cell-type-resolved atlas. If we use it, the resulting "KidneyRef v2" would be a single-tile bulk-tubule reference (tubular methylation signature, no proximal/distal/podocyte subdivision), not the multi-cell-type reference that BladderRef (4 tiles), EsoRef (8 tiles), or BoccellatoStomachRef (6 tiles) provide. **The cell-type resolution claim from the prior chat appears to be wrong** — it conflated "microdissected" (separates glomerulus from tubulointerstitium at the tissue level) with "sorted" (separates individual cell types within a tissue).

**De Ridder 2024 benchmarking** likely used GSE50874 as a *bulk-tissue mixture-generation reference for benchmarking deconvolution method performance* — i.e., a controlled bulk substrate to mix in known cell-type proportions for testing how well algorithms recover those proportions. That is NOT the same as using GSE50874 as the cell-type reference itself.

---

## Finding 2 — GSE59157 is a Wilms tumor disease cohort, NOT a healthy reference

**Prior chat's claim:** "Kidney 450K marker-selection cohort per De Ridder 2024."

**What the GEO metadata actually says:**

| Field | Value |
|---|---|
| Total samples | **n=95** |
| Series_title | "Methylome analysis of normal kidney, nephrogenic rest and Wilms tumor" |
| Tissue breakdown | **n=36 normal kidney + n=22 nephrogenic rest + n=37 Wilms tumour** |
| Tissue type | **All FFPE** (formalin-fixed paraffin-embedded) — not microdissected, not sorted |
| `Sample_data_processing` | `BeadStudio software v3.2` |
| Platform | GPL13534 (HM450K) |
| PubMed | 25134821 (Charlton et al. 2014) |

**What this means:**

1. **GSE59157 is fundamentally a pediatric kidney cancer disease cohort.** Wilms tumour is the most common pediatric kidney cancer; nephrogenic rests are precursor lesions. The 36 "normal kidney" samples are the controls within the disease cohort, not a healthy-kidney reference cohort built independently.

2. **All FFPE, not microdissected or sorted.** FFPE substrate is fundamentally different from frozen tissue — degradation patterns, fixation artifacts, and reduced bisulfite conversion efficiency mean FFPE-derived methylation values cannot be cleanly compared to TCGA frozen-tissue HM450 sesame Level 3 substrate without additional substrate validation.

3. **BeadStudio v3.2 processing**, not GenomeStudio AVG_Beta, not sesame Level 3 — third distinct substrate. Per cookbook this is a fourth substrate-class that has never been calibrated against the standing KIRC+PRAD anchor.

4. **CHK-3.1A normal-kidney-subset check:** f_extreme = 0.259, f_middle = 0.103. Both miss the raw-EPIC threshold (>0.30 / <0.10). This is consistent with **FFPE BeadStudio output** — softer bimodality from fixation degradation. Substrate is interpretable but not anchor-calibrated.

5. **De Ridder 2024 may have used the n=36 normal-kidney subset for marker selection**, but using n=36 FFPE BeadStudio normal-kidney samples as the marker-selection set for a KidneyRef atlas is NOT the same as building a KidneyRef from those samples for production deployment.

**Honest interpretation:** GSE59157 is a Wilms tumor cohort with embedded normal controls. It is NOT the cell-type-resolved kidney reference the prior chat believed. The n=36 normal kidney samples could function as a small bulk-kidney healthy reference, but FFPE substrate + BeadStudio processing means it can only be used as within-cohort self-cal, not as a calibrated atlas against the TCGA HM450 sesame Level 3 anchor.

---

## Finding 3 — Where this leaves us

The prior chat correctly identified that EpiSCORE KidneyRef is structurally inadequate as the primary cell-of-origin reader for kidney-epic v0.1 (32 markers in a one-vs-rest sparse encoding across 4 cell types — confirmed in repo: `episcore_zhu_teschendorff_2022/KidneyRef__Kidney_Mref_m.csv`, only 33 lines including header). That decision stands.

But the proposed Path A replacement — GSE50874 paired with GSE59157 as "tubule + glomerular references" — does not deliver what the prior chat described:

| Atlas | Cell-type resolution | Substrate | Cohort role |
|---|---|---|---|
| EpiSCORE KidneyRef (in vault) | 4 cell types, sparse 32-marker | EPIC source, bridged | One-vs-rest, structurally inadequate |
| GSE50874 (n=85) | **Bulk tubule only**, no cell types | GenomeStudio AVG_Beta | Single-tile bulk-tubule reference |
| GSE59157 normal subset (n=36) | **Bulk normal kidney**, FFPE | BeadStudio v3.2 | Small healthy-kidney reference, FFPE-substrate |

**None of these three is a deployment-grade multi-cell-type calibrated KidneyRef.** A pairing of GSE50874 + GSE59157 as "tubule + glomerular" would actually be "bulk tubule from frozen tissue (Susztak 2013) + bulk normal kidney from FFPE (Charlton 2014)" — two different substrates, neither cell-type resolved beyond what microdissection separates at the tissue level.

This matters because the **DISC-BLADDER-001 cell-type-distinctness rule** says cell-type distinctness drives gene-promoter atlas fitness. A bulk-tubule reference + a bulk-FFPE-normal reference has TWO tiles, not multi-cell-type tiles. The atlas would calibrate at lower cross-tile separation than EsoRef (0.0990) or BladderRef. It might still be operationally useful — but it is NOT the cell-type-resolved reference the prior chat described.

---

## What still needs to be checked

I have NOT yet:

1. **Fetched the bioRxiv 2026 cross-species kidney atlas supplementary data.** The prior chat queued this as v0.2 promotion target but the all-in directive said pull it now. This is the only candidate that genuinely claims cell-type-resolved methylation profiles (POD / PT-S1/S2 / cTAL / endothelial pseudo-bulk methylomes). Substrate is scMethyl-Hi-C — needs WGBS-region bridging into HM450 manifest, same engineering pattern as Caggiano TIM bridge (VAL-113 precedent).

2. **Inspected the De Ridder 2024 benchmarking paper itself** to see exactly what role GSE50874 + GSE59157 play in their methodology. If De Ridder constructed a derived KidneyRef matrix from these cohorts and deposited it as a supplementary file or in a software package (cfTools, EpiDISH, etc.), that derived matrix may be the actual "validated kidney reference" — not the raw cohorts.

3. **Searched for healthy-only kidney sorted-cell methylation atlases.** The reference-grade gene-promoter atlas pattern is sorted-cell methylation (BladderRef from sorted urothelium, EsoRef from sorted esophageal cell types). I should check if any recent paper has published sorted-cell kidney methylation (proximal tubule sorting, podocyte sorting, etc.) that I missed in the rush.

---

## Three honest paths forward

### Path A.1 — Continue acquisition with revised expectations

Pull the bioRxiv 2026 cross-species kidney atlas supplementary data + check De Ridder 2024 derived references + run one more targeted search for sorted-cell kidney methylation. If the bioRxiv 2026 atlas has accessible cell-type pseudo-bulk methylomes that bridge to HM450, that is the genuine cell-type-resolved KidneyRef v2. If not, ship v0.1 with bulk-tubule (GSE50874) + bulk-normal-kidney (GSE59157 n=36 subset) as a 2-tile honest-bulk reference and document the cell-type-resolution gap as a v0.2 target. Adds 1-2 hours to acquisition; I can do it now.

### Path A.2 — Pivot to "Loyfer Kidney bulk tile + GSE50874 self-cal" as primary cell-of-origin

Per CHK-2.18, kidney is solid parenchyma, NOT mucosal — so the "gene-promoter atlas required as primary cell-of-origin reader" rule does NOT fire. Loyfer Kidney bulk-WGBS tile (already calibrated VAL-112) is allowed as the primary cell-of-origin reader on a solid-parenchyma cohort. This was the cardio-epic v0.2 pattern after HeartRef collapsed. Loyfer Kidney + GSE50874 within-cohort self-cal as a triangulating second tile, KidneyRef v2 fully deferred to v0.2 once a real cell-type-resolved atlas exists. Faster, honest about what's available, no new atlas-vault entry until a genuine cell-type atlas surfaces.

### Path A.3 — Pause and ship a Phase A.2 atlas-acquisition VAL standalone

Build a dedicated Phase A.2 acquisition + bridge engineering effort focused on the bioRxiv 2026 cross-species kidney atlas, separate from kidney-epic v0.1. This is what cardio-epic did with Cuadrat 2023 (CCL-046) and what bladder-epic did with EpiSCORE BladderRef (Phase A.2 before VAL-119). Outcome: kidney-epic v0.1 ships with a real cell-type-resolved KidneyRef. Cost: days, not minutes. Mirrors the discipline that produced the working BladderRef and EsoRef.

---

## What I recommend

**Path A.1** — finish the parallel acquisition you signed off on. I haven't checked the bioRxiv 2026 cross-species atlas yet, and that's the candidate that genuinely promises cell-type resolution. If it bridges cleanly into HM450, kidney-epic v0.1 ships with a real KidneyRef. If it doesn't, we have GSE50874 + GSE59157 as honest-bulk substrates and we ship v0.1 on bulk-tile readout with the cell-type-resolution gap documented as v0.2 promotion target.

This honors the all-in directive without burning weeks on bridge engineering before we know whether the bridge is even needed.

---

## What I'm asking you before continuing

Do you want me to:

**(A)** Continue Path A.1 — fetch the bioRxiv 2026 cross-species kidney atlas + De Ridder 2024 benchmarking derived references + one more targeted sorted-cell kidney methylation search, then come back with the full picture before any build commitment.

**(B)** Pivot to Path A.2 — drop the KidneyRef v2 build entirely from v0.1, ship kidney-epic v0.1 on Loyfer Kidney bulk + GSE50874 within-cohort self-cal, document the cell-type-resolution gap as a v0.2 target. Faster, less ambitious, honest.

**(C)** Pause completely and have you make the Path-A vs Path-B call again with this corrected information in front of you. The prior chat's Path A direction was made on incorrect assumptions about GSE50874 and GSE59157 being cell-type-resolved. With the corrected picture, you may want to revise the call.

I'm halting parallel acquisition until you tell me which.

Walther.
