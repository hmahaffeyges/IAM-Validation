# Kidney sprint — Phase A.2 diligence (corrected with paper in hand)

**Date:** 2026-05-03
**Author:** Walther
**Status:** PAPER + SUPPLEMENTARY MATERIALS READ. DECISION READY.

---

## What changed since the previous report

You uploaded the paper PDF (`2026_01_22_700871v1_full.pdf`), supplementary figures (`media-1.pdf`), and supplementary tables (`media-2.xlsx`). The Cloudflare wall is no longer relevant. Three things resolve.

### Resolved 1 — Data deposition status

The paper's Data Availability section (lines 880-895) is explicit:

| Source | Status | Format |
|---|---|---|
| Single-nucleus DNA methylation data | **HuBMAP HBM233.KJSR.676** | `.h5ad` |
| **Per-cell-type processed methylation profiles** | **HuBMAP HBM233.KJSR.676** | **BigWig** |
| 10X Multiome data | HuBMAP HBM233.KJSR.676 | (consortium format) |
| Raw sequencing reads | dbGaP phs002249 — *upon peer-reviewed acceptance* | FASTQ |
| Loyfer 2023 reference (already in our vault) | GEO GSE186458 | (their validation reference) |
| KPMP TI/Glomerular WGBS (Gisch 2024) | KPMP DOI 10.48698/hhe6-yv15 | (their independent validation cohort) |
| Mouse IRI 10X Multiome | GEO GSE209610 | (used in Fig. 3h overlay) |

I tested HuBMAP API access right now: the entity API returns **`403 Forbidden: Publication for HBM233.KJSR.676 is not accessible without presenting a token`**. The data is real and deposited but requires HuBMAP credentials. dbGaP phs002249 is gated until peer-reviewed publication. **Neither source is publicly accessible without registration.**

### Resolved 2 — License is restrictive

Every page of the bioRxiv PDF carries: "All rights reserved. No reuse allowed without permission." The bioRxiv API tag `cc_no` is correctly interpreted as **CC No-Derivatives, no commercial reuse without explicit written permission from authors**. This is a real licensing constraint for EDEAR commercial deployment. Authors often grant permission for academic re-analysis — production deployment is a separate conversation that needs explicit written sign-off.

### Resolved 3 — The paper validates against Loyfer 2023 directly

This is the new finding. **Supplementary Figure 2 in the paper shows PCA of their sciMET cell-type pseudo-bulk methylomes plotted against the published Loyfer 2023 flow-sorted WGBS data** — the exact same atlas already calibrated in our vault as Loyfer Moss 2018 25-tile (VAL-112). They co-cluster cleanly by cell type (PT, POD, EC, FIB, etc.). This is direct external validation that:

1. **Loyfer's bulk-WGBS Kidney tile is on the same methylation manifold as their sciMET cell-type-resolved profiles.** A bulk-Loyfer-Kidney tile in production EDEAR is methodologically consistent with the field's gold standard.
2. **Their atlas validation strategy used Loyfer as the reference truth.** This is the same Loyfer reference we are already using.
3. **The published-paper layer of the field has converged on Loyfer as the kidney WGBS reference.** Our vault is already aligned with that consensus.

---

## What's actually in the supplementary tables (we have these in hand right now)

### Table S3 — the consensus reference panel — IS a derivative KidneyRef-like resource

2,180 orthologous genomic regions defining 11 conserved kidney cell types, with hg38 coordinates and per-cell-type methylation values. Distribution:

| Cell type | # regions | Operationally |
|---|---|---|
| B | 571 | Immune (Stage 3 already covered by Salas/UniLIFE/Caggiano TIM) |
| FIB | 469 | Stromal — kidney-specific signal |
| EC | 324 | Endothelial — kidney-specific signal |
| Myeloid | 291 | Immune |
| **POD** | **259** | **Podocyte — true kidney cell-of-origin signal** |
| **PT** | **99** | **Proximal tubule — true kidney cell-of-origin signal** |
| **DCT** | 58 | Distal convoluted tubule |
| **PC** | 55 | Principal cell |
| **CNT_IC** | 20 | Connecting tubule + intercalated |
| **TAL** | 18 | Thick ascending limb |
| **PEC** | 16 | Parietal epithelial |

**Yes, this is a real cell-type-resolved kidney atlas in tabular form. 2,180 regions with hg38 coordinates and per-cell-type methylation values, accessible to us right now via media-2.xlsx.**

### Table S6 — altered-PT vs healthy-PT DMRs

**8,996 DMRs** comparing altered-state PT cells to healthy PT cells, with chromosome / start / end / nCpG / healthy-state methylation / altered-state methylation / Δ / DSS statistic. This is the **CKD vs healthy disease signature for PT cells** — directly applicable to scoring kidney disease cohorts.

### Table S8 — chromatin compartment switches

**171 compartments** that flip euchromatin↔heterochromatin between healthy and altered PT, with explicit DEG and DMR gene annotations. 64 PT-specific euchromatin regions undergoing gain-of-repression + 107 PT-specific heterochromatin regions undergoing loss-of-repression.

---

## The honest engineering picture

I now have all four pieces I need to commit to a path:

1. **Cell-type-resolved kidney atlas exists in publicly available supplementary materials** → Table S3 has it. 2,180 regions, 11 cell types, hg38 coordinates, methylation values.
2. **Disease signature exists in supplementary materials** → Table S6 has it. 8,996 altered-PT DMRs.
3. **License is restrictive** → `cc_no`, "no reuse without permission" on every page. Real flag.
4. **Bulk-WGBS Loyfer Kidney tile we already have is co-clustered with this paper's atlas at the cell-type-pseudo-bulk level** → Supplementary Fig 2 shows it directly.

### What this means for v0.1 vs v0.2 scoping

**The bridging engineering effort is much smaller than I estimated yesterday:**

- Yesterday's estimate assumed we'd need raw sciMETv2 data (HuBMAP-gated) or BigWig per-cell-type files (also HuBMAP-gated) → multi-week download + bridge effort.
- **Reality:** Table S3 supplementary spreadsheet has 2,180 already-derived cell-type-discriminating regions with coordinates. We bridge by intersecting hg38 coordinates with HM450 manifest CpG positions — straightforward bedtools/pandas operation. **Hours, not weeks.**

**But the license constraint is real:**

- "All rights reserved. No reuse allowed without permission" — this is on every page footer.
- For academic publication ("we validated against Jeong 2026 Table S3 cell-type DMRs") this is fine — that's reference, not reuse.
- For commercial EDEAR deployment, depositing a derived KidneyRef into atlas_vault and using it in production scoring is reuse. **Needs explicit written permission from Kun Zhang / Sanjay Jain (corresponding authors).**

This is the same posture as the BoccellatoStomachRef → atlas_vault decision: cite the source paper, document the derivation, and (for commercial deployment) verify the license permits the use case.

---

## Three corrected paths forward

### Path B-revised (still recommended) — Ship v0.1 on Loyfer Kidney bulk + queue Jeong 2026 atlas as v0.2

**v0.1:**
- Stage 2 cell-of-origin: Loyfer Kidney bulk-WGBS tile (calibrated VAL-112) as PRIMARY, allowed for solid parenchyma per CHK-2.18
- Stage 2 triangulation: Caggiano CelFiE TIM (calibrated VAL-113)
- Stage 3 immune: Salas IDOL + UniLIFE + Caggiano TIM immune subset, framed per DISC-GE-005
- Stage 1: Xu-538 universal panel with mandatory CHK-2.17 cohort-substrate-coverage pre-flight on TCGA-KIRP HM450
- VAL-129 EpiSCORE KidneyRef calibration → expected O3_TISSUE_FLOOR_DOMINATED → DISC-KIDNEY-001 atlas-fitness null finding
- VAL-130 EsoRef-on-KIRC discrimination test (DISC-GE-003)
- VAL-131 Stage 1 Xu-538 on TCGA-KIRP n=45 paired
- VAL-132 Stage 2 multi-atlas on KIRP
- VAL-133 Stage 3 immune subset on KIRP
- VAL-134 GSE52955 multi-cancer urological cross-tile sanity

**v0.2:**
- Phase A.2 sprint: bridge Jeong 2026 Table S3 (2,180 cell-type DMRs) into HM450 manifest after written permission from Kun Zhang/Sanjay Jain confirming commercial-use license
- Build derived KidneyRef v2 with PT/POD/DCT/TAL/PC/CNT_IC/PEC/FIB/EC tiles
- Calibrate against KIRC+PRAD anchor n=210
- Add to atlas_vault with INVENTORY entry citing Jeong 2026 + license terms
- Reproduce VAL-129 with KidneyRef v2; confirm cell-type-specific signal beyond bulk-tile

**Pros:** Ships v0.1 in 6-8 hours of prereg drafting. Honest about cell-type gap. Loyfer Kidney is already calibrated and is the paper's own validation reference.
**Cons:** Less ambitious for v0.1; cell-type resolution deferred.

### Path A.2 (revised, faster than yesterday) — Ship v0.1 with KidneyRef v2 from Table S3 NOW

**Faster than I estimated yesterday because Table S3 is in hand:**
1. Bridge Table S3 hg38 coordinates → HM450 manifest CpG positions (hours)
2. Per-cell-type marker selection (top 50-100 markers per cell type from S3 scores)
3. CHK-3.1A/B/C verification on the bridged matrix
4. Atlas-vault stamp + INVENTORY entry + bridge script
5. CHK-2.21 KIRC+PRAD anchor n=210 cross-card calibration → VAL-129 KidneyRef v2 calibration
6. Then VAL-130–134 as planned

**Estimated timeline:** 1-3 days, not multi-weeks.

**License blocker:** "All rights reserved. No reuse allowed without permission" on every page. **You need written permission from Kun Zhang / Sanjay Jain to use Table S3 data in commercial EDEAR deployment** before this path can ship. For academic-only / research-only contexts this would be allowed under fair use citation, but EDEAR is commercial product.

**Pros:** Real cell-type-resolved KidneyRef in v0.1. Ambitious. Mirrors what the paper actually delivers.
**Cons:** License risk is real. Production deployment without written permission is exposure. License clearance could take days-to-weeks of attorney communication.

### Path C-mixed — Both tracks parallel

I draft Path B-revised preregs now (6-8 hours, ships v0.1 immediately with bulk-tile cell-of-origin). In parallel, I draft a separate v0.2 sprint plan for KidneyRef v2 from Table S3 that triggers as soon as you receive written license permission from Kun Zhang. Two tracks: v0.1 ships now with what's safe, v0.2 ships when license clears.

---

## What I recommend

**Path B-revised, with one addition: I draft a license-permission email to Kun Zhang in parallel with the v0.1 preregs.**

Reasoning:
1. **Ship v0.1 fast.** 6-8 hours of prereg drafting → kidney-epic v0.1 sealed → disease cohort findings start landing this week. Loyfer Kidney bulk + DISC-KIDNEY-001 documentation + VAL-130 cross-card test deliver real value without licensing exposure.
2. **License clearance is independent and asynchronous.** You email Kun Zhang explaining IAMPerformance Inter-Domain Research Institute, what we're proposing to use (Table S3 cell-type DMRs as bridged KidneyRef in atlas_vault), how we'd cite the work, and what commercial deployment means. He may grant permission readily — Altos Labs publishes for impact, and academic-licensed-for-commercial-research is a common ask. He may also decline or want a formal license. Either way, v0.1 is already in the air.
3. **If license clears:** v0.2 sprint becomes Phase A.2 bridge engineering on Table S3. Days, not weeks. Real cell-type-resolved KidneyRef in v0.2, citing Jeong 2026 Nat Methods (likely landing site after peer review).
4. **If license is declined:** v0.1 stands on Loyfer Kidney bulk; v0.2 becomes a different scope (other atlases, other diseases). DISC-KIDNEY-001 is cataloged either way.
5. **The "no half-assing" principle is honored.** Path B-revised is honest about what's available without permission. Path A.2 is the ambitious extension that becomes available with permission. Both are documented; neither is sloppy.

---

## What I'm asking you before continuing

**(B-revised + license email)** — I draft the six VAL preregs (VAL-129 through VAL-134) starting now under Path B-revised AND I draft a license-permission email to Kun Zhang for your review. You review/send the email; I keep working on preregs. Estimated 6-8 hours of prereg drafting; license email is 30 minutes. This is what I recommend.

**(A.2-direct)** — You tell me you're willing to take the academic-use risk on Path A.2 and ship v0.1 with KidneyRef v2 from Table S3 now without waiting for license clearance. Faster v0.1, real licensing exposure. I'd want this decision in writing with your attorney's sign-off referenced.

**(C-mixed)** — Hybrid. I draft Path B-revised preregs now AND draft a separate v0.2 KidneyRef-v2 sprint plan that activates on license clearance. v0.1 ships in days; v0.2 ships when license is confirmed.

The picture is now complete. The atlas exists. The data is there. The license is the only operational blocker. Your call.

Walther.
