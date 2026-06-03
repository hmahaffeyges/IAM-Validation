# Cross-Card Calibration To-Do List — v0.5 Discipline Wave

**Created:** 2026-04-29 evening, after VAL-112 + VAL-113 cardio sprint completion
**Updated:** 2026-04-30 — v0.5 surgically adds 11 fixes after re-audit of cardio v0.3 truth-state README + card JSON: full CCL/CHK enumerated checklist, substrate-platform variation handling, structured `atlases_deferred` format, DISC-CARD-NNN per-card discoveries format, CHK-5.13 web-verification of canonical citations in pre-flight, validation evidence summary template, atlas_vault INVENTORY.json discipline, cookbook-wide CCL cross-reference template, per-disease scoring policy template, Phase A sub-tasks (acquisition / bridge engineering), and rewritten cardio reference example tracking all five VALs (108/109/110/111/112+113) in order with their distinct outcome classes. v0.4 content preserved verbatim.
**Purpose:** Apply the same VAL-112-style calibration discipline (CCL-041 + CCL-048 + CHK-3.1A/B/C + CHK-0.7) to every existing EDEAR card before any new cards are built.
**Estimated total effort:** 8-10 cards × ~3 hours per card calibration sprint = roughly 25-30 hours of execution time. Most steps run in background; effective wall-clock is shorter.
**Owner:** Heath W. Mahaffey + Walther
**Track via:** This document. Update with completion checkmarks as each card finishes. Each card produces its own VAL-NNN sealed outputs that go to GitHub repo + Heath-only deliverables (cookbook IP).

---

## READ THIS FIRST — guardrails that override everything else

These rules are absolute. Any next chat picking up this document must read this section before touching any card.

1. **No-fabrication rule.** Never invent class names, biological categories, numerical values, citations, cohort sizes, validation results, method details, or product specs. Mirror source files exactly. If the source files don't have it, ask Heath or leave it out. Never fill in plausible-looking numbers.

2. **Per-card workflow is absolute.** Finish ONE card completely before starting the next. The seven files that update every card:
   - TESTING_CHECKLIST.md (Heath-only)
   - PIPELINE_REFERENCE_v2 (Heath-only)
   - README_MASTER (Heath-only)
   - LESSONS_LEARNED.md (Heath-only)
   - card README (Heath-only)
   - card JSON (Heath-only)
   - Evidence Report HTML (Heath-only, with Python inline + source links)

3. **GitHub push vs Heath-only delivery.** Hard split. After every card sprint:

   **PUSH to GitHub** (Biological_Physics/ tree, public repo `hmahaffeyges/IAM-Validation`):
   - Sealed VAL-NNN Python script(s)
   - Pre-registration document (prereg.md)
   - Outcome document (outcome.md)
   - Results JSON (the headline numbers)
   - Stratified-results JSON (per-subgroup breakdowns)
   - Cohort manifest (sample IDs + group assignments + provenance)
   - Clinical metadata (de-identified, study-of-origin)
   - Biological_Physics local README.md update (the public-facing per-card README in the repo)

   **DELIVER to Heath after the push** — Heath gets the same scripts and JSONs that were pushed to the repo, so he has a local working copy. Specifically: every VAL Python script + every results JSON + every stratified JSON that was pushed gets sent to Heath in the same delivery as the Heath-only files below.

   **NEVER push — deliver to Heath only** (cookbook IP):
   - Evidence Report HTML (currently `GAPE_Evidence_Report_UPDATED.html` — the live evidence report, updated with each new VAL block)
   - README_MASTER (currently `README_MASTER_v2_4.md` — the cookbook master spec; bump version every cycle)
   - Card README (e.g. `cardio_epic_README_v0_2_2.md` — bump card version per Phase E)
   - Card JSON (e.g. `cardio_epic_card_v0_2_2.json` — bump card version per Phase E)
   - LESSONS_LEARNED.md
   - TESTING_CHECKLIST.md
   - EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md (operational pipeline spec; update if pipeline changes)
   - GAPE_Reproduction_Paper_v1.md (the reproduction-grade documentation; updates when methodology evolves)
   - This TODO document

4. **Surgical edits only.** Always work from the most recent uploaded files. When in doubt — ask Heath first. Never delete paragraphs, sections, or material without explicit sign-off.

5. **TESTING_CHECKLIST.md is absolute.** First call when starting any new card or VAL is `view /mnt/user-data/outputs/cookbook_v2.1/TESTING_CHECKLIST.md`. Order of checks: data integrity → biology → framework. Pre-scoring requirements:
   - Beta distribution check (raw beta >30% extremes <10% in [0.4, 0.6]; flat distribution = residuals not raw)
   - Cross-cohort healthy baseline >1 SD = mismatch
   - Saturation flag must be checked
   - Never publish null without data both interpretable AND consistent with clinical-grade disease panels

6. **Reproducibility triple (CHK-7.6, absolute).** Every Evidence Report VAL block and every VAL outcome.md needs all four items:
   - Inline source code (HTML-escaped pre/code)
   - Inputs list with download URL + size + SHA per file
   - Environment (Python version, package versions, runtime, memory)
   - Expected headline output

7. **Language discipline (absolute).** Never use: resolves, confirms, validates, proves. Always use: consistent with, tested against, predictions within the framework, the data are consistent with. Never use the phrase "it matters" or "why it matters" in Heath-facing output — use the actual substantive point instead.

8. **Companion paper context.** The Floor Breach paper (DOI 10.5281/zenodo.18702042) establishes the structural identity between black hole formation and cellular malignant transformation. The cellular floor `H_min(class)` is calibrated via MCMC against published reference cell measurements with R-hat < 1.001 and zero free parameters. Do not introduce H_min recalibration during this discipline wave. The eight class floors are frozen 2026-04-06 and stay frozen. This wave is about ATLAS calibration discipline (Stage 1 / Stage 2 / Stage 3 reference atlases used inside the pipeline), NOT about touching the architectural floors themselves.

9. **EDEAR commercial deployment is unaffected throughout this discipline wave per CCL-037.** Production deployment runs customer-specific calibration regardless of cookbook-side audit state.

10. **EXHAUSTIVE COHORT SURVEY before any card sprint begins.** This is the absolute first step of every card audit and every new card build. Before claiming a VAL ID, before opening a calibration script, before touching any atlas — survey what is actually publicly available for this disease. Search GEO, ArrayExpress, TCGA / GDC, dbGaP (where DUA is feasible), SRA, EBI, the published-cohort lists in recent reviews of the disease, and any consortium repositories relevant to the disease (UK Biobank with restricted access, FinnGen, Million Veterans, Sister Study, EPIC, etc.). Build an enumerated list of every plausibly-relevant cohort with: GSE/EGAS/dbGaP ID, sample size, disease+control breakdown, substrate (HM450 / EPIC v1 / EPIC v2 / WGBS / RRBS), tissue (whole blood / plasma cfDNA / tumor / adjacent-normal / sorted cells), accession status (open / restricted / DUA-required), and the published reference. The point is to know what the universe of testable data actually contains BEFORE deciding what to score against. Cards built without an exhaustive cohort survey miss obvious validation opportunities and surface as gaps later.

11. **CALIBRATION BEFORE TESTING is the inviolable order.** Phase B (calibration on a structurally-separated healthy cohort) MUST seal before Phase C (re-scoring disease cohorts) begins. The CHK-3.1B q5 thresholds and per-tile healthy-floor distributions get sealed first, with their own VAL ID and SHA-256 hash, BEFORE any disease-cohort A-score is computed. This is not a ceremony — it is the difference between a falsifiable cookbook product and within-cohort self-cal artifacts. If anyone is tempted to fold calibration and disease scoring into the same script, that is the discipline gap CCL-041 was created to close. Stop and re-architect.

12. **RUN EVERYTHING through ALL atlases — every IDAT, every cohort, every time.** Per the run-everything-every-time architecture (signed off 2026-04-26, EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2). When Phase C re-scores a disease cohort, it scores against EVERY atlas in the card's atlases_run block. Not the atlas the card thinks is most relevant. Not the atlas the disease was originally scored against. Every one of them. The compute cost is seconds per IDAT; the cost of NOT running everything is missed signals (e.g. cardio-epic's PAH cohort surfaced cardiac signal across all three atlases, but BAV cohort needed multi-atlas triangulation to clean up Loyfer's small-n confounder). The output is per-cohort × per-atlas × per-tile A-scores AND per-atlas Cohen's d. Stratification preserved. No shortcuts.

13. **The CCL / CHK enumerated checklist — every card audit must apply ALL of these.** This is the operational discipline list that produced cardio v0.3. Every card audit honors all of them; cards that violate any of them surface as the audit gaps the cookbook keeps catching. Each item below is followed by what cardio sprint did with it.

    **CCL (Cookbook Cross-card Lessons) — propagate from one card to all subsequent cards:**
    - **CCL-040 LL-PROCESSED-OUTPUT-DEFERRAL** — when a public cohort's β panel is processed differently (GenomeStudio AVG_Beta vs minfi vs sesame), the substrate envelope is cohort-specific. *Cardio applied this with substrate-specific CHK-3.1A self-cal envelopes per substrate.*
    - **CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION** — different substrates require different CHK-3.1A thresholds; calibration must precede scoring (the inviolable order, also Guardrail #11). *Cardio applied this with VAL-106 (TCGA HM450K sesame Level 3 baseline) + VAL-107 (cardio-epic CHK-3.1B subset) sealed before VAL-108/109/110 disease cohort scoring.*
    - **CCL-042 LL-CHK-3.1-A/B-SPLIT** — CHK-3.1A is the full-genome baseline gate; CHK-3.1B is the atlas-subset coverage gate. They are distinct gates with distinct thresholds. *Cardio v0.1 was the first card built natively under the split convention.*
    - **CCL-043 LL-CARDIO biology lessons** — formalized 2026-04-28 with the four cardio-specific lessons (LL-CARDIO-001 Stage 1 immune workhorse / LL-CARDIO-002 whole blood does not stratify stroke etiology / LL-CARDIO-003 hPAH > iPAH framework signal / LL-CARDIO-004 aortic pathology Stage 1 immune-detectable Stage 2 vascular-tile-resistant); LL-CARDIO-005 added in v0.2 from VAL-111. *Each subsequent card produces its own LL-XXXXX biology lessons block.*
    - **CCL-046 LL-CANONICAL-DOC-FACTUAL-ERROR** — documents-of-record can themselves contain factual errors. CHK-5.12 protects against picking the wrong atlas from atlas_vault but does NOT protect against following an incorrect citation in the canonical document. Web-verification is required. *Cardio v0.2.2 caught Konigsberg 2023 → actual paper is Cuadrat 2023; Liu 2023 → actual paper is Tian et al. 2023.*
    - **CCL-047 LL-ATLAS-DEDUPE** — atlas reference matrices can contain duplicate CpG IDs that silently bias A-score scoring. *Cardio v0.2.2 caught Layered Moss+Loyfer reference_atlas.csv had 7,890 rows but only 6,105 unique CpGs (1,785 identical-value duplicates); deduped, original preserved as `_with_duplicates.csv` for audit.*
    - **CCL-048 LL-SUBSTRATE-NORMALIZATION-REQUIRED** — raw IDAT files cannot be scored directly. They must first go through sesame normalization (recommended) or customer-specific calibration on within-cohort self-cal substrate. CHK-0.7 substrate-normalization gate enforces this. *Cardio v0.3 added CHK-0.7 to TESTING_CHECKLIST.*
    - **CCL-049 LL-MULTI-ATLAS-MANDATE** — small-n single-atlas |d| > 2 not replicated by other atlases is a confounder candidate; multi-atlas reporting mandated. *Cardio v0.3 BAV cohort surfaced as the first card-level instance: Loyfer normal_vs_BAV |d| > 2 on Colon_epithelial / Hepatocytes / Pancreatic — cardiac-specialized atlases (Caggiano, HeartRef) cleaned up the noise.*

    **CHK (Checklist gates) — every card publish or VAL seal must clear these:**
    - **CHK-3.1A** — full-genome substrate baseline gate (per-sample f_extreme + f_middle pass within substrate envelope)
    - **CHK-3.1B** — atlas-subset coverage gate (per-sample n_subset_valid ≥ threshold per atlas)
    - **CHK-3.1C** — atlas-deduplication gate (added 2026-04-29 per CCL-047). Every atlas reference matrix must have unique CpG IDs before scoring. Bias from duplicates is uniform within-cohort but corrupts absolute A-score magnitudes.
    - **CHK-5.7** — universal_reference block structural-parity gate (every card's JSON has the 14 substantive sub-keys)
    - **CHK-5.8** — atlases_used_and_deferred block structural-parity gate (atlases_run + atlases_deferred with structured target/dependency for every deferred atlas)
    - **CHK-5.9** — substrate_roadmap block structural-parity gate (DNAm validated + four non-DNAm substrates deferred to v0.4+ with explicit anchor and target)
    - **CHK-5.10** — chk_3_1_thresholds_per_substrate block structural-parity gate (3.1A + 3.1B per-substrate thresholds with calibration_anchor_val_id and calibration_anchor_cohort_n)
    - **CHK-5.12** — atlas-canonical-source-check gate. Atlas selection must trace to a canonical-document section name. *Cardio v0.2.1 added this after VAL-111 tested an off-critical-path atlas (EpiSCORE HeartRef) that was in atlas_vault but not on PIPELINE_REFERENCE Part 2.4's named critical path.*
    - **CHK-5.13** — documents-of-record citation-verification gate (added 2026-04-29 per CCL-046). Before sealing a card publish or VAL, every external citation introduced in new content (canonical-document quotes, atlas attributions, cohort accessions, prior-art references) must have at least one web-verification pass: DOI loads, authors match, content matches, accessions resolve.
    - **CHK-0.7** — substrate-normalization gate (added 2026-04-29 per CCL-048). Confirm the cohort β panel was processed through sesame Level 3 or equivalent normalization before any A-score scoring; raw IDAT cannot be scored directly.
    - **CHK-7.6** — reproducibility triple (already Guardrail #6): inline source code, inputs list with download URL + size + SHA per file, environment, expected headline output.

---

## Why we're doing this before any new cards

Before VAL-112 + VAL-113, every existing card's sealed VAL outcomes (VAL-001 through VAL-111) used Stage 2 atlases that scored against disease cohorts using **within-cohort self-cal** as the operational fallback. CCL-041 logged this as the discipline gap. VAL-112 + VAL-113 demonstrated the correction template: structurally-separated TCGA n=210 calibration cohort + per-atlas CHK-3.1B q5 thresholds + per-tile healthy-floor A-score distributions sealed BEFORE any disease-cohort scoring.

Every card needs this audit. Some cards will pass (their atlases happen to be calibrated already, or their sealed outcomes are robust to within-cohort vs structurally-separated calibration). Some cards will need new VAL-NNN calibration runs and may shift sealed outcomes when re-scored under proper discipline. The point is to do this systematically, BEFORE new cards introduce new atlases that compound the problem.

---

## Pre-flight checklist (do these BEFORE the first card sprint)

These are one-time setup verifications. The next chat should run through this list at the start of any session and confirm each item before touching a card.

### File availability — confirm each is on disk and accessible
- [ ] `/home/claude/edear_working/VAL-106/calibration_betas/` (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 calibration cohort — substrate HM450 sesame Level 3)
- [ ] GSE40279 Hannum n=656 whole-blood healthy cohort (already on disk for VAL-006 — verify path)
- [ ] `validation_runs/VAL-112_run_everything/val_112_calibrate.py` (calibration script template)
- [ ] `validation_runs/VAL-112_run_everything/val_112_phaseC.py` (Phase C scoring template)
- [ ] `validation_runs/VAL-112_113_unified/outcome.md` (outcome.md template — what every Phase C output should look like)
- [ ] `atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/bridge_caggiano_to_array.py` (atlas-bridging template if any new atlas needs bridging)
- [ ] `atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/extract_manifest.R` (HM450 hg19 manifest extraction template)
- [ ] Most recent EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md (operational pipeline spec)
- [ ] Most recent README_MASTER for cookbook v2.1
- [ ] Most recent TESTING_CHECKLIST.md
- [ ] Floor Breach companion paper PDF (`/mnt/user-data/uploads/floor_breach_derivation.pdf` or equivalent on disk) — read for context, do not modify

### VAL ID claiming — claim the IDs before starting work
- [ ] Next available VAL ID after VAL-113 is **VAL-114**. Reserve a contiguous block: VAL-114 through approximately VAL-130 for this discipline wave.
- [ ] Document the VAL-ID-to-task map at the top of this TODO before starting any sprint. (See "VAL ID reservations" section below.)

### Atlas integrity — CHK-3.1C dedup audit
- [ ] Xu-538 atlas file: confirm no duplicate CpG IDs
- [ ] Salas Blood.EPIC IDOL atlas file: confirm no duplicate CpG IDs
- [ ] UniLIFE 19-cell atlas file: confirm no duplicate CpG IDs
- [ ] Layered Moss+Loyfer Stage 2 atlas: already audited via VAL-112 ✓
- [ ] Caggiano CelFiE TIM: already audited via VAL-113 ✓
- [ ] EpiSCORE HeartRef: already audited via VAL-112 ✓

### Beta distribution sanity check
- [ ] For each calibration cohort, run the pre-scoring beta distribution check (per TESTING_CHECKLIST.md): raw beta should have >30% extremes and <10% values in [0.4, 0.6]. Flat distribution = the file is residuals or transformed values, not raw beta. STOP if flat.

### Cross-cohort healthy baseline check
- [ ] Confirm calibration cohort healthy baseline is within 1 SD of GSE40279 Hannum baseline (cross-cohort sanity). >1 SD = substrate / preprocessing mismatch — STOP and investigate before proceeding.

### Substrate inventory per cohort (CCL-040 + CCL-041)
- [ ] For every disease cohort the card will score against, document the methylation preprocessing pipeline. Different pipelines produce different f_extreme distributions — this is real and substantial. Cardio v0.3 sprint hit four substrates with a 24-percentage-point distribution gap:
  - TCGA HM450K sesame Level 3: f_extreme 55.87% ± 2.44%
  - GenomeStudio AVG_Beta HM450K (raw, no normalization): f_extreme 31.81% ± 3.54%
  - GenomeStudio V2011.1 HM450K raw: f_extreme 33.95% ± 2.21%
  - minfi `preprocessFunnorm` HM450K: f_extreme 52.82% ± 2.33%
- [ ] Each substrate gets its own CHK-3.1A self-cal envelope. Within-cohort self-cal envelopes work for v0.2-style validation; structurally-separated calibration VALs are required to make a substrate envelope generalizable as a platform threshold (deferred work tracked in card's `chk_3_1_thresholds_per_substrate` block).
- [ ] If a cohort uses a substrate not yet documented in the card's `chk_3_1_thresholds_per_substrate` block, add a new substrate entry with `calibration_anchor_val_id` and `calibration_anchor_cohort_n` before scoring.

### Web-verification of canonical citations (CHK-5.13, per CCL-046)
- [ ] For every external citation introduced in new content this sprint (canonical-document quotes, atlas attributions, cohort accessions, prior-art references), run at least one web-verification pass: DOI loads, authors match the cited names, content matches the description, accessions resolve.
- [ ] Cardio v0.2.2 cited as the worked example: PIPELINE_REFERENCE_v2.md Part 2.4 cited "Konigsberg 2023" — web-verification of the cited DOI (10.1093/nargab/lqad061) found the actual paper is **Cuadrat, Kratzer, Giral Arnal et al. 2023** (no Konigsberg in the author list); content was bulk ENCODE heart tissues, not sorted cardiomyocytes. A second audit pass caught "Liu 2023" → actual paper is **Tian et al. 2023**. Both errors caught only via web-verification of the cited DOI.
- [ ] One web search per citation. Cheap; catches an entire class of compounding errors.

---

## VAL ID reservations (claim before sprint)

Update this block at the start of each sprint. Once a VAL ID is claimed, it is not reusable.

| VAL ID | Purpose | Card | Status |
|---|---|---|---|
| VAL-114 | Xu-538 immune panel calibration on GSE40279 | (Shared Task A) | ▢ Reserved |
| VAL-115 | Salas IDOL 6-cell calibration on GSE40279 | (Shared Task B) | ▢ Reserved |
| VAL-116 | UniLIFE 19-cell calibration on GSE40279 | (Shared Task B) | ▢ Reserved |
| VAL-117+ | AD-immune card audit | AD-immune | ▢ |
| VAL-118+ | CRC-epic card audit | CRC-epic | ▢ |
| VAL-119+ | breast-epic card audit | breast-epic | ▢ |
| (continue as cards are picked up) | | | |

---

## Master template (the VAL-112 / VAL-113 pattern)

Every card calibration sprint follows this seven-phase template. The cardio card sprint is the worked example — see "Reference example: cardio-epic card" below for what each phase actually looked like in execution.

**Phase 0: Exhaustive cohort survey.** Before anything else, build the universe of testable data.
1. Search GEO (Gene Expression Omnibus) by disease keyword + methylation array. Record every result with sample size and substrate.
2. Search ArrayExpress by the same terms. Cross-reference with GEO results.
3. Search TCGA / GDC for tumor + adjacent-normal cohorts of the disease (and adjacent tissues if applicable for cell-of-origin Stage 2 work).
4. Check dbGaP for restricted-access cohorts; note which require DUA.
5. Search SRA / EBI for sequencing-based methylation cohorts (WGBS, RRBS) if applicable.
6. Check disease-specific consortia: UK Biobank, FinnGen, Million Veterans Program, Sister Study, EPIC, ROSMAP / AIBL / ADNI / AddNeuroMed for neurodegeneration, etc.
7. Pull recent (last 5 years) review articles and meta-analyses of methylation in this disease — their cohort tables are pre-curated lists.
8. Output: a `cohort_survey.md` document at `/cards/<card_name>/cohort_survey.md` with every plausibly-relevant cohort enumerated. Columns: GSE/EGAS/dbGaP ID, citation, sample size (cases / controls), substrate, tissue/cell type, accession status (open / restricted / DUA), pre-diagnostic vs at-diagnosis, ancestry composition if reported, status notes (already used in prior VAL / candidate for this sprint / deferred to v0.4+ / inaccessible).
9. Heath reviews the cohort_survey.md and signs off on which cohorts go into Phase C of this sprint vs which are deferred. No Phase A begins until Phase 0 sign-off lands.

**Phase A: Inventory the card's atlas list and acquire/bridge any new atlases.**

*A.1 — Inventory.*
1. Open card JSON `atlases_used_and_deferred.atlases_run` block
2. List every Stage 2 + Stage 3 + Stage 1 atlas the card scores against
3. For each atlas: check if a calibration JSON already exists at `validation_runs/VAL-XXX/calibration_results.json` — if yes, note the calibration VAL ID; if no, mark as needing new calibration
4. Check atlas file in atlas_vault for CHK-3.1C (no duplicate CpGs); if duplicates found, deduplicate with CCL-047 audit-trail preservation (keep original as `_with_duplicates.csv`)
5. Verify CHK-5.12: every atlas in `atlases_run` traces to a canonical-document section name (PIPELINE_REFERENCE Part 2.x). If an atlas in atlas_vault doesn't have a canonical-document anchor, do not score against it without explicit Heath sign-off.

*A.2 — Atlas acquisition (if any new atlases identified in Phase 0 or A.1).*
1. For each atlas to acquire, identify: paper DOI, license (CC-BY / MIT / GPL-2 / etc.), source URL (GitHub / Bioconductor / supplementary data / accession), atlas form (signature matrix CSV / .rda R object / WGBS regions / nanopore / etc.), target cookbook version (v0.3 / v0.4+ / v1.0+).
2. Run CHK-5.13 web-verification on the canonical-document citation BEFORE acquiring: DOI loads, authors match, content matches the description, accessions resolve. Catches Konigsberg→Cuadrat / Liu→Tian class errors.
3. Update `atlas_vault/INVENTORY.json` with: source URL, citation, license, SHA-256 of acquired file, atlas family classification (tile-coverage WGBS-derived / gene-promoter / sorted-cell / training-scaffold), calibration anchor when sealed.
4. Push atlas_vault commit to GitHub alongside VAL commits. Cardio sprint commit: `57beb38` (deduped Loyfer + Caggiano TIM bridged + INVENTORY.json updated with calibration anchors).

*A.3 — Bridge engineering (reusable infrastructure, not card-specific).*
1. **HM450 hg19 manifest extraction** — required to bridge WGBS-derived atlases (Caggiano CelFiE TIM, Cuadrat 2023 ENCODE EPIC) to array-CpG resolution. Template: `atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/extract_manifest.R`. Output: array-CpG-in-region intersection (e.g. Caggiano 1,581 → 254 array CpGs × 19 cell types).
2. **R rpy2 bridge** — required for atlases distributed only as R packages (EpiSCORE, deconvR). Existing infrastructure in cookbook; adds the atlas-specific Python entry point.
3. **Nanopore→array CpG bridge** — required for Tanaka 2025 6-cell-type neural and any future nanopore atlases. Engineering work; deferred to v0.4+ unless explicitly prioritized.
4. **scMCodes→array projection** — required for Tian et al. 2023 single-cell brain atlas. Engineering work; deferred to v0.4+.
5. Each bridge produces array-CpG-resolution input for Phase B calibration. Bridge engineering is reusable infrastructure: build once, reuse for every card with overlapping cell types.

*A.4 — Integration testing.*
1. For each acquired or bridged atlas, run a smoke test: score against the calibration cohort to verify the atlas reads sensibly (no all-NaN tiles, no all-1.0 A-scores, no negative-CpG-count failures).
2. If smoke test passes → proceed to Phase B (calibration).
3. If smoke test fails → log as DISC-CARD-NNN finding (atlas family unfit / bridge engineering gap / substrate mismatch), defer atlas to next version with explicit unblock dependency. Cardio reference: VAL-111 EpiSCORE HeartRef sealed at `O3_TISSUE_FLOOR_DOMINATED` after smoke-test failure pattern (gene-promoter atlas family does not transfer to A-score tile reading on heterogeneous β panels) — atlas → atlases_deferred for next version with explicit dependency.

**Phase B: Calibrate each uncalibrated atlas on a structurally-separated healthy cohort.**
1. Pick the calibration cohort that matches the card's primary substrate. For HM450 sesame Level 3 cards, use TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (same as VAL-106/107/112/113). For EPIC v1/v2 cards, identify a substrate-matched healthy cohort (typically a TCGA solid-tumor adjacent-normal subset).
2. Run the VAL-112-style calibration script on the cohort: per-sample CHK-3.1A (full-genome) + per-sample CHK-3.1B (atlas subset) + per-tile A-scores
3. Seal: CHK-3.1B q5 threshold + per-tile healthy-floor distribution (mean, sd, n, q2.5, q5, q50, q95, q97.5)
4. Output: `validation_runs/VAL-XXX_calibrate_<atlas_name>/calibration_results.json` + `per_sample_calibration.csv`
5. **Calibration is sealed and SHA-256 hashed BEFORE Phase C begins.** This is the inviolable order — guardrail #11.

**Phase C: Re-score the card's existing disease cohorts under run-everything discipline.**
1. For each disease cohort the card has already scored (VAL-XXX sealed outputs) AND every new cohort approved out of Phase 0, re-score against EVERY atlas in the card's atlases_run (the run-everything pass per guardrail #12)
2. Per-cohort: per-sample CHK-3.1A + per-sample CHK-3.1B per atlas + per-tile per-atlas A-scores
3. Per-cohort group-pair Cohen's d per atlas per tile
4. Output: `validation_runs/VAL-XXX_re_score/cohort_per_sample_run_everything.csv` + `cohen_d_per_atlas.json`

**Phase D: Compare v0.2.x vs v0.3 sealed outcomes.**
1. For each cohort, compare the original within-cohort self-cal outcome vs the new calibrated outcome
2. If sealed direction + magnitude are unchanged → v0.2.x outcome stands; v0.3 adds calibration metadata
3. If direction reversed or magnitude shifted by > 0.5 Cohen's d → log as outcome-shift finding, draft new outcome.md, decide whether to supersede or note as new finding
4. If small-n confounders surfaced (single-atlas |d| > 2 not replicated by other atlases) → flag per CCL-049 (mandate multi-atlas reporting for that cohort)

**Phase E: Card promotion — all structured blocks every card produces.**

This phase is where the card's documentation discipline lives. The cardio v0.3 README + card JSON established the templates below. Every card produces all of them. Skipping any of them is the discipline gap subsequent audits will catch.

*E.1 — Bump card version.*
1. Card JSON header: `card_version`, `card_date`, `supersedes` block listing all prior versions
2. Card README header: version, build date, validation tier, "Built under" CCL/CHK list, "Supersedes" prior versions
3. Card filename: bump version (e.g. `cardio_epic_card_v0_2_2.json` → `cardio_epic_card_v0_3.json`)

*E.2 — Update `atlases_used_and_deferred` block (CHK-5.8).*
1. `atlases_run`: every atlas calibrated this sprint, with `calibration_anchor_val_id`, `chk_3_1b_q5_threshold`, `chk_3_1c_passed: true`
2. `atlases_deferred`: structured table format. Every deferred atlas has THREE fields:
   - **Atlas name** (canonical-document name, e.g. "Cuadrat 2023 cardiovascular extended", "Sorted cardiomyocyte array-CpG atlas (open published-literature gap)")
   - **Target version** (v0.3 / v0.4+ / v1.0+)
   - **Unblock dependency** (e.g. "HM450 hg19 manifest acquisition", "R rpy2 bridge", "Nanopore→array CpG bridge engineering", "External: published-literature gap — monthly literature surveillance pass")
3. Cardio v0.2.1 cited as worked example: nine deferred atlases enumerated with explicit target/dependency for each.

*E.3 — Update `chk_3_1_thresholds_per_substrate` block (CHK-5.10).*
1. For every measurement substrate the card's cohorts encountered, document both 3.1A and 3.1B thresholds
2. Each substrate entry: `substrate_name`, `f_extreme_baseline`, `f_extreme_sd`, `f_middle_baseline`, `f_middle_sd`, `chk_3_1b_subset_threshold`, `calibration_anchor_val_id`, `calibration_anchor_cohort_n`
3. Cardio cited as worked example: four substrates documented (TCGA HM450K sesame Level 3, GenomeStudio AVG_Beta HM450K, GenomeStudio V2011.1 HM450K raw, minfi preprocessFunnorm HM450K)

*E.4 — Add `vX_run_everything_phase_c_results` block.*
1. Per-cohort headline findings for the card's Phase C run
2. Each cohort entry: cohort name + n + per-atlas Cohen's d for the strongest tile + per-atlas convergence note (e.g. "convergent across 3 atlases" / "single-atlas confounder cleaned by multi-atlas triangulation" / "convergent null across 3 atlases — biology-correct")
3. Cardio v0.3 cited: PAH (convergent strong cardiac signal across 3 atlases, Caggiano `heart` d=+1.42 strongest), BAV (multi-atlas triangulation cleaned Loyfer's small-n confounder), stroke (convergent null across 3 atlases, max |d| = 0.19).

*E.5 — Add Per-disease scoring policy block.*

Every card produces this. For each disease subdomain or substrate the card scores, explicitly state what the card claims and what it does NOT claim. This is the honesty discipline that makes the cookbook defensible. Cardio v0.3 cited as worked example:

```
### <disease_subdomain> (<substrate>)
- **What the card claims:** <pooled signature / specific tile differentiation / Stage 1 immune flag / etc.>
- **What the card does NOT claim:** <subtype stratification / cross-platform discrimination / etc.>
- **Stage 1 immune:** <claim or null>
- **Stage 2 cell-of-origin:** <claim or null>
- **Stage 3 immune subcomposition:** <claim or null>
- **Operational implication:** <one sentence>
```

Cardio v0.3 example coverage: ischemic stroke whole blood (pooled report only, no subtype stratification), PAH cultured PECs (direct vascular-tile differentiation operational, hPAH-vs-iPAH framework-equivalent — not claimed), aortic pathology (pooled-pathology report only, dissection-vs-BAV framework-equivalent — not claimed; Stage 2 vascular tile does NOT discriminate bulk aorta).

*E.6 — Add "What we discovered" / DISC-CARD-NNN section.*

Every card produces a discoveries section with structured DISC-CARD-NNN entries. Each entry has:
- **DISC-CARD-NNN — <one-line title>**
- Body paragraph explaining what was found
- **Implication:** one paragraph stating what changes operationally for this card and what propagates to future cards

These discoveries also propagate to LESSONS_LEARNED.md as LL-CARD-NNN entries. Cardio v0.3 cited as worked example: eight discoveries logged (DISC-CARDIO-001 Stage 1 immune workhorse / DISC-CARDIO-002 substrate-cell match critical / DISC-CARDIO-003 biology-correct nulls are first-class outcomes / DISC-CARDIO-004 atlas family matters at Stage 2 / DISC-CARDIO-005 substrate envelopes work but not generalizable / DISC-CARDIO-006 CHK-3.1A/B split convention exercised end-to-end / DISC-CARDIO-007 atlas selection traces to canonical document — CHK-5.12 added / DISC-CARDIO-008 atlas reference structural validation — CHK-3.1C + CCL-047 added).

*E.7 — Add "What we chose not to claim" + "What remains open" sections.*

These two sections are the honesty discipline beyond outcomes. Cardio v0.3 cited as worked example:
- **What we chose not to claim** — explicit list of stratifications the card does NOT claim, with one-line rationale per item (e.g. "did not claim stroke etiology stratification — VAL-108 demonstrated etiology-equivalence in whole blood — biology-correct null per LL-CARDIO-002")
- **What remains open** — explicit numbered list of subdomains, atlases, cohorts, or methodologies still pending (e.g. "Coronary heart disease / MI subdomain — target cohort GSE56046 MESA n=1,202 EPIC-era ROS/MAP", "Caggiano CelFiE TIM cardiac blocked at HM450 hg19 manifest acquisition")

*E.8 — Add Validation evidence summary per VAL.*

Every sealed VAL gets a structured entry in the card README's `## Validation evidence summary` section. Each VAL entry has:

```
### VAL-NNN — <cohort name + tissue/substrate type>

**Cohort:** <GSE/cohort ID + description + n>
**Substrate:** <preprocessing pipeline + array platform>
**Design:** <within-cohort case-control / pre-diagnostic / atlas integration test / etc.>
**QC pass rate:** <n_passed / n_total (%)>
**Outcome:** `<O2_/O3_/O4_/O5_OUTCOME_CLASS>` (sealed)

**Key Cohen's d values:**
- <Stage 1 / Stage 2 tile / Stage 3 contrast>: <d-value with 95% CI when applicable>
- ...

**Interpretation:** <one paragraph framing what the data say and why it says it>

**Prereg SHA-256:** `<hash>`
```

Cardio v0.3 cited as worked example: VAL-108 stroke null O3, VAL-109 PAH O2, VAL-110 aortic O2, VAL-111 EpiSCORE atlas-fitness deferral O3, VAL-112 + VAL-113 calibration-only entries plus per-cohort scoring entries.

*E.9 — Add Cookbook-wide CCL cross-references section.*

Every card README has a `## Cookbook-wide lessons referenced` section listing every CCL the card inherits, applies, or formalizes. Each entry: CCL-NNN + one-line description + one-line statement of how the card honored / formalized / inherited the lesson. Cardio v0.3 cited four CCLs (CCL-040, CCL-041, CCL-042, CCL-043) plus the three new ones formalized in cardio (CCL-046, CCL-047, CCL-048, CCL-049).

*E.10 — Update card README.md prose body.*
1. Add v0.X header section with Phase C findings + operational implications
2. Append the `vX-1 → vX changes` section (additive — sealed prior outcomes preserved unchanged in audit-trail)
3. Update the validation tier (e.g. `multi_modal_validated + multi_atlas_calibrated` after a multi-atlas Phase C run)

**Phase F: Push + deliver — the seven-files protocol.**

This phase is the operational discipline of the per-card workflow rule. Three sub-steps in this exact order.

**F.1 — PUSH to GitHub** (Biological_Physics/ tree of `hmahaffeyges/IAM-Validation`):
1. Sealed VAL-NNN Python script(s) for both Phase B (calibration) and Phase C (re-scoring)
2. Pre-registration document (prereg.md) — sealed BEFORE Phase C ran
3. Outcome document (outcome.md) — drafted from Phase C results
4. Results JSON (headline numbers per cohort per atlas)
5. Stratified-results JSON (per-subgroup breakdowns)
6. Cohort manifest (sample IDs + group assignments + provenance)
7. Clinical metadata (de-identified, study-of-origin, what stratification variables were used)
8. Biological_Physics/README.md update — the public per-card README in the repo, updated with the new VAL findings + Phase C summary

**F.2 — DELIVER to Heath** the same scripts and JSONs that were just pushed:
- Every VAL Python script that went to GitHub (Heath gets a local working copy)
- Every results JSON that went to GitHub
- Every stratified JSON that went to GitHub
- The cohort manifest and clinical metadata files
- The Biological_Physics/README.md update preview

This is so Heath always has a local copy of what is in the public repo and can refer to it without pulling from GitHub.

**F.3 — DELIVER to Heath only** (cookbook IP, NEVER pushed). All seven of these update every card sprint:
1. **Evidence Report HTML** (currently `GAPE_Evidence_Report_UPDATED.html`) — append the new VAL block with Python inline + source links + reproducibility triple per CHK-7.6
2. **README_MASTER** (currently `README_MASTER_v2_4.md`) — bump version, add the new VAL findings to the per-card status table, update the Phase summary
3. **Card README** (e.g. `cardio_epic_README_v0_2_2.md`) — bump card version per Phase E, add v0.3 section with Phase C findings
4. **Card JSON** (e.g. `cardio_epic_card_v0_2_2.json`) — bump card version per Phase E, update atlases_run with calibration anchors
5. **LESSONS_LEARNED.md** — append any lessons from this card sprint (small-n confounders surfaced, atlas behavior surprises, cohort-specific gotchas)
6. **TESTING_CHECKLIST.md** — append any new pre-flight checks the card sprint surfaced
7. **EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md** — update if this card sprint introduced any new operational pipeline behavior; otherwise leave unchanged
8. **GAPE_Reproduction_Paper_v1.md** — update if methodology evolved this sprint; otherwise leave unchanged

**F.4 — Update this TODO document** with ✓ next to the completed card + commit hash from F.1 + delivery timestamp from F.2/F.3.

---

## ✅ Reference example: cardio-epic card — DONE 2026-04-26

This is the worked example. Read this section first if you've never run a card sprint before. Every step below is what was actually done for cardio-epic. Use this as your template for every other card.

**Important context:** the cardio sprint produced **eight sealed VALs** across three months. Not every VAL was a calibration run — some were atlas-fitness tests, some were biology-correct nulls, some documented platform divergence. The next chat will produce different outcome classes (O2 differentiating, O3 floor-dominated, O3 undifferentiated, O5) for the same kinds of reasons. Do not collapse all VALs into "the calibration sprint" — they are distinct sealed outcomes with distinct outcome classes.

**Phase 0 executed (cohort survey):**
- Searched GEO for `methylation cardiovascular`, `methylation heart`, `methylation cardiac`, `cell-free DNA cardiac`, plus disease-specific terms (PAH, BAV, stroke, MI, HF, atherosclerosis).
- Searched ArrayExpress + dbGaP + SRA + EBI for the same terms.
- Pulled cohort tables from recent (2020-2025) reviews of methylation in cardiovascular disease.
- Output: `cards/cardio_epic/cohort_survey.md` enumerated 14 plausibly-relevant cohorts. Three were selected for Phase C (PAH GSE84395, BAV/dissection GSE84274, stroke etiology GSE69138). Eleven were deferred to v0.4+ with reasons (substrate mismatch, restricted access pending DUA, sample size below threshold, etc.).
- Heath signed off on the three-cohort selection. Phase A began.

**Phase A executed (A.1 inventory + A.2 acquisition + A.3 bridge engineering + A.4 integration testing):**

*A.1 inventory.* Card JSON `atlases_run` and `atlases_deferred` blocks opened. atlases_run at v0.2.1: Layered Moss+Loyfer (Stage 2), UniLIFE 19-cell (Stage 3), Salas Blood.EPIC IDOL 6-cell (Stage 3). atlases_deferred initially: 8 entries identified after CHK-5.12 trace to PIPELINE_REFERENCE Part 2.4.

*A.2 acquisition + CHK-5.13 web-verification.* Phase A acquisition pass on PIPELINE_REFERENCE Part 2.4's named cardio Stage 2 atlas hit a CCL-046 wall: web-verification of the cited DOI `10.1093/nargab/lqad061` found the canonical document had two factual errors. The "Konigsberg 2023" citation was actually **Cuadrat, Kratzer, Giral Arnal et al. 2023** (no Konigsberg in author list). Atlas content was **bulk ENCODE heart tissues** (right atrium, left ventricle, coronary artery), not sorted cardiomyocytes. A second audit pass caught "Liu 2023 Science adf5357" → actually **Tian et al. 2023**. Both errors propagated through atlas_full_name, deferral_rationale, and atlases_deferred table; corrected in v0.2.2 expanded patch. Atlas vault commit logged the corrections.

*A.2 atlas vault discipline.* Three atlases promoted from atlases_deferred to atlases_run during the sprint: Layered Moss+Loyfer (deduped via CCL-047 from 7,890 → 6,105 unique CpGs after CHK-3.1C dedup gate fired; original preserved as `reference_atlas_v0.2_with_duplicates.csv`), EpiSCORE HeartRef sub-panel bridged via gene-promoter mapping (3,727 CpGs × 5 cardiac cell types: CM/EC/FB/MP/SMC), Caggiano CelFiE TIM array-bridged via HM450 hg19 manifest CpG-in-region intersection (1,581 → 254 array CpGs × 19 cell types). atlas_vault commit `57beb38` pushed to GitHub with all three deduped/bridged + INVENTORY.json updated with calibration anchors.

*A.3 bridge engineering.* HM450 hg19 manifest extraction script written for Caggiano CelFiE TIM (template at `atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/extract_manifest.R`). Reusable for every future card with WGBS-derived atlases.

*A.4 integration testing.* All three Stage 2 atlases CHK-3.1C dedup audit passed. Smoke test on calibration cohort verified no all-NaN tiles, no all-1.0 A-scores.

**Phase B executed (calibration anchors VAL-106 + VAL-107, then atlas calibration VAL-112 + VAL-113):**

*VAL-106 — TCGA HM450K sesame Level 3 CHK-3.1A baseline.* Cohorts: TCGA-KIRC adjacent-normal n=160 + TCGA-PRAD adjacent-normal n=50 (combined n=210). QC pass: 194/210. CHK-3.1A baseline established: f_extreme 55.87% ± 2.44%, f_middle 7.42% ± 0.75%. Outcome `O3_CALIBRATION_DEGENERATE` under sealed prereg's bounds (which were derived from CpG-subset prior data points); reclassified post-hoc as the substrate baseline anchor for TCGA HM450K sesame Level 3 under the CHK-3.1A/B split convention. Prereg SHA-256: `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`.

*VAL-107 — Cardio-epic CHK-3.1B subset on TCGA HM450K sesame Level 3.* Same 210-sample cohort. Cardio CHK-3.1B subset: 8,100 unique CpGs (Loyfer 25-tile 6,105 + UniLIFE 1,906 + Salas 350). Subset SHA: `5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511`. Coverage pass: 210/210 (n_subset_valid always > 7,000 of 8,100). Outcome `O2_PLATFORM_DIVERGENCE_DOCUMENTED` (Mann-Whitney p=0.034 on f_extreme; practical Δ only 0.7 percentage points between KIRC and PRAD). Established threshold: extreme ≥ 55.0%, middle ≤ 8.5%, n_subset_valid ≥ 7,000.

*VAL-112 — Layered Moss+Loyfer (deduped) + EpiSCORE HeartRef Stage 2 calibration on TCGA n=210.* Per-tile healthy-floor distributions sealed for both atlases. CHK-3.1B q5 thresholds: 0.6839 (Loyfer 25 tiles), 0.4283 (HeartRef 5 tiles). Calibration sealed and SHA-256 hashed BEFORE any disease-cohort scoring per Guardrail #11.

*VAL-113 — Caggiano CelFiE TIM array-bridged Stage 2 calibration on TCGA n=210.* Per-tile healthy-floor distributions sealed (19 cell types, 254 array CpGs). CHK-3.1B q5 threshold: 0.5779. Calibration sealed BEFORE any disease-cohort scoring.

**Phase C executed (run-everything across three cardio cohorts; every atlas every cohort per Guardrail #12):**

*VAL-108 — GSE69138 ischemic stroke 3-subtype on whole blood (n=404; substrate GenomeStudio AVG_Beta HM450K).* QC pass 383/404 (94.8%) under cohort-specific CHK-3.1A self-cal envelope. Outcome **O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED**. Maximum |d| across all stages and pair contrasts: 0.167 (Stage 2 Monocytes_EPIC, small-vessel vs cardioembolic). Three independent atlases later confirmed convergent null (max |d| = 0.19 across all atlases in Phase C re-scoring). Whole-blood DNA methylation does not stratify ischemic stroke by TOAST etiology — this is a biology-correct null, not a framework failure (post-stroke systemic inflammatory response homogenizes the immune signature). Prereg SHA-256: `6f40ebd9d30bb10242b245d7bde280607f1170e3c7993a8284e2852ad1f69e7a`.

*VAL-109 — GSE84395 PAH cultured pulmonary endothelial cells (n=39; substrate minfi preprocessFunnorm HM450K).* QC pass 37/39 (94.9%) under cohort-specific CHK-3.1A self-cal envelope. Outcome **O2_PAH_VASCULAR_TILE_DIFFERENTIATING**. Stage 1 immune control vs hPAH d=+0.65 / control vs iPAH d=+0.65; Stage 2 Vascular_endothelial_cells control vs hPAH d=+0.79; Stage 2 Left_atrium control vs hPAH d=+0.65. hPAH-vs-iPAH framework-equivalent (all |d|<0.5) — not claimed as discriminator. Three-atlas Phase C re-scoring: Caggiano `heart` tile control vs iPAH d=+1.42 (strongest single-tile signal of any atlas tested); HeartRef CM control vs iPAH d=−0.80; Loyfer Vascular_endothelial_cells control vs hPAH d=+0.83. Convergent strong cardiac signal across 3 atlases. Prereg SHA-256: `f6450b4cf5d384d2ea27b349c101b3f167a6a549d276e670e68fb2232b45f21e`.

*VAL-110 — GSE84274 ascending aorta dissection/BAV (n=24; substrate GenomeStudio V2011.1 HM450K raw).* QC pass 23/24 (95.8%). Outcome **O2_AORTIC_ANY_TILE_DIFFERENTIATING**. Stage 1 immune normal vs BAV+dilation d=+1.08 (strongest aortic signal in card); Stage 1 normal vs dissection d=+0.56; Stage 2 Vascular_endothelial_cells |d|≤0.15 (does NOT discriminate — bulk aorta dominated by SMC/fibroblast); Stage 2 Adipocytes normal vs BAV d=−0.88 (peri-aortic adipose contamination). Three-atlas Phase C re-scoring on n=6 vs n=6 BAV exposed the CCL-049 confounder pattern: Loyfer normal_vs_BAV |d|>2 on Colon_epithelial / Hepatocytes / Pancreatic — these tiles have no biological connection to BAV; with n=6 vs n=6 the within-group sd is artificially small and Cohen's d inflates spuriously. Cardiac-specialized atlases (Caggiano `heart` d=+1.40, `endothelial` d=+1.52, `fibroblast` d=+2.10; HeartRef CM d=−0.60) showed the real cardiac-cell-type signal cleanly. Prereg SHA-256: `1041738ccc8bcdd45a4754d599a28ad80fde3a7b37b6c18b4d528f4fe0271bc8`.

*VAL-111 — EpiSCORE HeartRef atlas integration test on three cardio cohorts.* Atlas SHA-256: `bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83`. Three cohorts (β panels reused from VAL-108/109/110): GSE69138 stroke whole blood (negative-control substrate; non-cardiac tissue should produce all five cardiac tiles below A=0.10 floor), GSE84395 PAH cultured PECs (vascular substrate; EC tile expected to dominate), GSE84274 ascending aorta (smooth-muscle-rich substrate; SMC tile expected to dominate). All three cohorts cleared >500 atlas CpG intersection (no O4 bridge failure: 3,727/3,727/3,408 atlas∩cohort intersections). Outcome **O3_TISSUE_FLOOR_DOMINATED**. All five cardiac tile A-scores read 0.46-0.51 across all three substrates; maximum within-cohort tissue discrimination 0.0152 (vs 0.10 threshold required) — discrimination ratio 15% of threshold; blood-floor breach on 5/5 tiles in GSE69138 (cohort means 0.48-0.51 well above 0.10). Direction was biologically sensible (dissection > BAV+dilation > normal monotonic in GSE84274; SMC tile always highest in aortic samples; iPAH > hPAH > control on EC tile in GSE84395) but A-score magnitude is set by gene-promoter average methylation (~0.5 in heterogeneous β panels) rather than substrate-specific cell-of-origin contrast. EpiSCORE HeartRef methodologically sound for its design purpose (EpiDISH proportion estimation in heart tissue) but does not transfer to A-score tile reading on heterogeneous β panels at the resolution required for cardio-epic Stage 2. Atlas → atlases_deferred for next version with explicit unblock dependency. Prereg SHA-256: `172c6ae2a11345935c176b4a1fc57d30009ad4bac9bb9cdeeb9c8226035b78a6`.

**Phase D executed:**
- Compared v0.2 single-atlas outcomes (VAL-108/109/110) vs v0.3 multi-atlas Phase C re-scoring outcomes
- VAL-108 stroke null held across all three atlases — biology-correct null confirmed
- VAL-109 PAH cardiac signal strengthened with multi-atlas triangulation (convergent across 3 atlases; strongest Caggiano `heart` d=+1.42 control vs iPAH)
- VAL-110 BAV cohort surfaced as the first card-level instance of CCL-049: small-n single-atlas |d|>2 not replicated by other atlases is a confounder candidate; cardiac-specialized atlases cleaned up Loyfer's spurious tissue signals; multi-atlas reporting mandated for v0.3+ cardio scoring
- Sealed VAL outcomes preserved unchanged in audit-trail; v0.3 Phase C results ADD to the same cohorts under correct calibration discipline
- 31,948 calibrated A-score readings sealed at VAL-112_113_unified

**Phase E executed (E.1 through E.10):**

*E.1 — Card version bump.* `cardio_epic_card_v0_2_2.json` → `cardio_epic_card_v0_3.json`. Card README similarly bumped. supersedes block listed all prior versions (v0.1, v0.2, v0.2.1, v0.2.2 → v0.3). Validation tier updated to `multi_modal_validated + multi_atlas_calibrated`.

*E.2 — atlases_used_and_deferred block.* atlases_run promoted to three calibrated Stage 2 atlases: Layered Moss+Loyfer (deduped), EpiSCORE HeartRef, Caggiano CelFiE TIM array-bridged. atlases_deferred at v0.3 contained 9 structured entries: Cuadrat 2023 cardiovascular extended (target v0.3+, dependency = atlas acquisition), Sorted cardiomyocyte array-CpG atlas (target v1.0+, dependency = external published-literature gap), Caggiano CelFiE TIM cardiac (target v0.3, dependency = HM450 hg19 manifest acquisition — RESOLVED in v0.3), EpiSCORE Zhu/Teschendorff 2022 pan-tissue (target v0.3, dependency = R rpy2 bridge), Tanaka 2025 6-cell-type neural (target v0.3+, dependency = nanopore→array CpG bridge engineering), Tian et al. 2023 scMCodes brain (target v0.4+, dependency = scMCodes→array projection), MARLIN Capper 2025 training scaffold (target v0.3+, dependency = leukemia matrix build-out), Sabedot GeLB 2021 (target v0.3+, dependency = GSE150289 cohort acquisition + R→Python integration).

*E.3 — chk_3_1_thresholds_per_substrate block.* Four substrates documented: TCGA HM450K sesame Level 3 (anchor VAL-106), GenomeStudio AVG_Beta HM450K (anchor VAL-108 within-cohort self-cal), GenomeStudio V2011.1 HM450K raw (anchor VAL-110 within-cohort self-cal), minfi preprocessFunnorm HM450K (anchor VAL-109 within-cohort self-cal). 24-percentage-point f_extreme distribution gap documented; structurally-separated platform calibration deferred per substrate.

*E.4 — v0_3_run_everything_phase_c_results block.* PAH (convergent strong cardiac signal across 3 atlases, Caggiano `heart` d=+1.42 strongest), BAV (multi-atlas triangulation cleaned Loyfer's small-n confounder; cardiac-specialized atlases d=+1.40 to +2.10), stroke (convergent null across 3 atlases, max |d| = 0.19).

*E.5 — Per-disease scoring policy.* Stroke (whole blood): pooled report only; no subtype stratification; no healthy-vs-stroke contrast at v0.2 (cohort had no healthy controls); Stage 1 immune is primary scoring target. PAH (cultured PECs): direct vascular-tile differentiation operational; hPAH-vs-iPAH framework-equivalent — NOT claimed; Stage 1 immune also discriminates. Aortic pathology (ascending aorta tissue): pooled-pathology report only; dissection-vs-BAV framework-equivalent — NOT claimed; Stage 1 immune is the strongest signal; Stage 2 vascular tile does NOT discriminate aortic pathology (bulk aorta dominated by SMC/fibroblast). EpiSCORE HeartRef cardiac-tile discrimination: NOT claimed at v0.2 (atlas → atlases_deferred for v0.3 with explicit dependency).

*E.6 — DISC-CARDIO-NNN discoveries section.* Eight discoveries logged (each propagated to LESSONS_LEARNED.md):
- **DISC-CARDIO-001** — Stage 1 immune A-score is the workhorse for cardio-epic across all substrates tested (whole blood, cultured PECs, aortic tissue)
- **DISC-CARDIO-002** — Substrate-cell match is the single most important cardio biology consideration (cultured PEC Vascular_endothelial_cells d=+0.79 vs bulk aorta same tile d=−0.04)
- **DISC-CARDIO-003** — Biology-correct nulls are first-class outcomes (VAL-108 stroke etiology homogenization)
- **DISC-CARDIO-004** — Atlas family matters at Stage 2: tile-coverage atlases ≠ gene-promoter atlases (VAL-111 EpiSCORE HeartRef O3 floor-dominated)
- **DISC-CARDIO-005** — Substrate-specific CHK-3.1A self-cal envelopes work for cardio at v0.2 — they are NOT a generalizable platform threshold yet (24-percentage-point f_extreme gap)
- **DISC-CARDIO-006** — Cardio sprint exercised the entire CHK-3.1A/B split convention end-to-end for the first time
- **DISC-CARDIO-007** — Always read PIPELINE_REFERENCE Part 2 first; atlas selection must trace to a canonical-document name (CHK-5.12 added in v0.2.1; corrected in v0.2.2 after CCL-046 finding)
- **DISC-CARDIO-008** — Atlas reference matrices need structural validation (duplicate-CpG check) before A-score scoring (CHK-3.1C added in v0.2.2 from CCL-047 finding)

*E.7 — What we chose not to claim + what remains open.* Six explicit non-claims (stroke etiology stratification, hPAH-vs-iPAH discrimination, aortic dissection-vs-BAV discrimination, EpiSCORE HeartRef cardiac-tile discrimination, generalizable platform thresholds for non-sesame substrates, retroactive threshold accommodation). Ten subdomains/atlases remaining open (CHD/MI subdomain target GSE56046 MESA n=1,202, stroke vs healthy contrast target GSE128235, heart failure subdomain, hypertensive heart disease, hemorrhagic stroke, pulmonary embolism, EpiSCORE HeartRef re-bridging, Caggiano CelFiE TIM unblocked at HM450 manifest — RESOLVED in v0.3, sorted cardiomyocyte array-CpG gap pending external publication, Tanaka 2025 nanopore bridge engineering).

*E.8 — Validation evidence summary per VAL.* Eight VAL entries (VAL-106, VAL-107, VAL-108, VAL-109, VAL-110, VAL-111, VAL-112, VAL-113) each formatted with cohort/n/substrate/design/QC/outcome/Cohen's d/interpretation/prereg SHA-256.

*E.9 — Cookbook-wide CCL cross-references.* CCL-040, CCL-041, CCL-042, CCL-043 inherited; CCL-046, CCL-047, CCL-048, CCL-049 formalized in cardio sprint.

*E.10 — README prose body update.* v0.3 header section + v0.2.2 → v0.3 changes section appended (additive); validation tier updated to `multi_modal_validated + multi_atlas_calibrated`.

**Phase F executed (the seven-files protocol):**

*F.1 — PUSHED to GitHub:*
- VAL-106 + VAL-107 + VAL-108 + VAL-109 + VAL-110 + VAL-111 + VAL-112 + VAL-113 Python scripts (all eight VALs)
- prereg.md + outcome.md for each VAL with reproducibility triple per CHK-7.6
- results JSON + stratified JSON + cohort manifests + clinical metadata for stroke / PAH / aortic cohorts plus calibration cohort
- Biological_Physics/README.md updated with cardio-epic v0.3 multi-atlas summary
- atlas_vault commit `57beb38` (deduped Loyfer + Caggiano TIM bridged + INVENTORY.json with calibration anchors)

*F.2 — DELIVERED to Heath* the same scripts and JSONs that were pushed to GitHub: local working copies of all eight VAL Python scripts + results JSONs + stratified JSONs + manifests + clinical metadata.

*F.3 — DELIVERED to Heath only* (cookbook IP, NEVER pushed):
- `GAPE_Evidence_Report_UPDATED.html` — appended VAL-106 through VAL-113 blocks with reproducibility triple
- `README_MASTER_v2_4.md` — bumped from prior version, added cardio-epic v0.3 to per-card status table
- `cardio_epic_README_v0_3.md` — Phase E version bump + v0.3 findings section + 8 DISC-CARDIO discoveries section
- `cardio_epic_card_v0_3.json` — Phase E version bump + atlases_run with three calibrated Stage 2 atlases + structured atlases_deferred with 9 entries + chk_3_1_thresholds_per_substrate with 4 substrates + v0_3_run_everything_phase_c_results block
- `LESSONS_LEARNED.md` — appended LL-CARDIO-001 through LL-CARDIO-005 plus the eight DISC-CARDIO discoveries; appended CCL-046, CCL-047, CCL-048, CCL-049 lessons formalized in cardio sprint
- `TESTING_CHECKLIST.md` — appended CHK-3.1C atlas-deduplication gate + CHK-5.12 atlas-canonical-source-check + CHK-5.13 documents-of-record citation-verification + CHK-0.7 substrate-normalization gate + multi-atlas-triangulation pre-flight check
- `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` — Part 2.4 fully rewritten with corrected Cuadrat 2023 description + correction note + sorted-cardiomyocyte gap acknowledgment; Part 21 added documenting the v0.2.2 honesty patch; Part 22 substrate normalization pipeline architecture added
- `GAPE_Reproduction_Paper_v1.md` — §7.24 + §7.25 added with cardio-epic Phase A-F as worked example

*F.4 — Updated this TODO* with ✓ next to cardio-epic + commit hash `57beb38` (atlas vault) + commit hashes from VAL-NNN GitHub pushes.

**Outcome:** Cardio-epic is the template. Every other card follows this exact structure. Lessons learned that propagate to every subsequent card:
- Multi-atlas triangulation catches small-n confounders that single-atlas scoring misses (CCL-049)
- Substrate-cell match is the single most important biology consideration at Stage 2 (DISC-CARDIO-002)
- Biology-correct nulls are first-class outcomes — the framework correctly does NOT stratify what biology has homogenized (DISC-CARDIO-003)
- Atlas family matters at Stage 2: tile-coverage WGBS-derived atlases work; gene-promoter atlases (EpiSCORE-class) do not transfer to A-score tile reading on heterogeneous β panels (DISC-CARDIO-004)
- Substrate-specific CHK-3.1A self-cal envelopes work within-cohort but are NOT generalizable platform thresholds (DISC-CARDIO-005)
- Bridge-engineering work (HM450 hg19 manifest extraction for Caggiano) produces lasting infrastructure usable by any future card with overlapping cell types
- Atlas selection must trace to canonical-document name (CHK-5.12) AND the canonical document itself must be web-verified (CHK-5.13) — Konigsberg→Cuadrat and Liu→Tian are the worked examples
- Atlas reference matrices need structural validation (CHK-3.1C dedup gate) before A-score scoring — duplicates silently bias all downstream A-score computations
- Phase 0 (exhaustive cohort survey) caught two cohorts that would have been missed by going straight to known/obvious sources — the survey is not optional
- Stage 1 immune A-score is the workhorse across all substrates tested; Stage 2 cardio-tiles add localization context but are not load-bearing for the headline call

---

## Per-card audit table — execute in this order

Order chosen by: (1) card maturity / commercial-deployment priority, (2) substrate diversity (do similar substrates together), (3) atlas reuse (cards sharing atlases get done after their shared atlases are calibrated once).

Each card section below has the same structure as the cardio reference example. Follow Phase 0 → A → B → C → D → E → F in order. Do not skip phases. Do not begin a new card before the current card's Phase F is complete.

### Standard Phase 0 (every card): exhaustive cohort survey

For every card below, before any other phase begins, run the cohort survey per guardrail #10 and master-template Phase 0. Output a `cards/<card_name>/cohort_survey.md` with every plausibly-relevant cohort enumerated. Heath signs off on the cohort selection before Phase A begins. Card-specific search keywords are noted in each card's section.

### Standard Phase F (every card): seven-files protocol

For every card below, Phase F follows the master template's F.1 / F.2 / F.3 / F.4 sub-steps:
- **F.1 PUSH GitHub:** VAL-NNN py + prereg + outcome + results JSON + stratified JSON + cohort manifest + clinical metadata + Biological_Physics/README.md
- **F.2 DELIVER to Heath same files pushed:** local copies of all VAL Python + results JSON + stratified JSON + manifest + metadata
- **F.3 DELIVER to Heath only (cookbook IP, NEVER pushed):** updated `GAPE_Evidence_Report_UPDATED.html`, `README_MASTER_v2_4.md` (bump version), card README (bump version), card JSON (bump version), `LESSONS_LEARNED.md`, `TESTING_CHECKLIST.md`, `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` if changed, `GAPE_Reproduction_Paper_v1.md` if methodology evolved
- **F.4 Update this TODO** with ✓ + commit hash + delivery timestamp

---

### ▢ 1. AD-immune card — HIGHEST PRIORITY (commercial deployment candidate)

**Status before audit:** Card has VAL-050 (sealed AD null pooled-entropy d=+0.08), VAL-051 (sealed AD-directional panel 7 CpGs Rule A d=+0.62 AIBL holdout, AUC 0.68), VAL-052 (AddNeuroMed cross-platform d=+0.33 AUC 0.60), VAL-053 (sex-specific panels do NOT outperform unified), VAL-054b (HC-permutation p=0.003).

**Atlas list to audit:**
- Xu-538 immune panel (Stage 1) — calibration anchor unknown
- Salas Blood.EPIC IDOL 6-cell (Stage 3) — calibration anchor unknown
- UniLIFE 19-cell (Stage 3) — calibration anchor unknown

**Step-by-step execution plan:**

0. **Phase 0 (cohort survey).** Build `cards/ad_immune/cohort_survey.md`. Search GEO/ArrayExpress/dbGaP/SRA/EBI for: methylation Alzheimer's, methylation dementia, methylation cognitive impairment, methylation MCI, blood DNA methylation neurodegeneration. Pull cohort tables from recent (2020-2025) reviews of methylation in AD. Check ROSMAP, AIBL, ADNI, AddNeuroMed accessions and DUA status. Enumerate every plausibly-relevant cohort. Heath signs off on which cohorts go into Phase C this sprint.

1. **Phase A (inventory).** Open `cards/ad_immune/card_v2_x.json`. Confirm atlases_run lists exactly the three atlases above (and only those). If additional atlases appear, add them to the audit list. Run CHK-3.1C dedup audit on each.

2. **Phase B (calibrate).** Use Shared Task A's VAL-114 (Xu-538 calibration) + Shared Task B's VAL-115 (Salas IDOL) + VAL-116 (UniLIFE) as the calibration anchors. Calibration cohort: GSE40279 Hannum n=656 whole-blood healthy. Substrate: HM450. If shared tasks are not yet complete, do them first (see Wave 1 below). All calibrations sealed and SHA-256 hashed BEFORE Phase C.

3. **Phase C (re-score, run everything every cohort every atlas per guardrail #12).** For each AD disease cohort approved out of Phase 0, run the VAL-112-style Phase C script with the new calibration anchors:
   - **AIBL holdout** (the original VAL-051 cohort) — re-score with all three atlases + the 7-CpG directional Rule A panel
   - **AddNeuroMed** (the VAL-052 cohort) — re-score; this is the cross-platform replication that read d=+0.33 originally
   - **ROSMAP** if accessible — likely needs DUA verification before scoring
   - **VAL-054b HC-permutation set** — re-run permutation under new calibration anchors
   - Plus any new cohorts surfaced in Phase 0
   - For each cohort, output per-sample CHK-3.1A and CHK-3.1B per atlas, per-tile A-scores per atlas, and per-cohort Cohen's d per atlas per tile.

4. **Phase D (compare).** Compare the new sealed outcomes against VAL-050/051/052/053/054b originals. Specifically watch for: does the AD-instance bidirectional cancellation pattern (VAL-050 pooled null vs VAL-051 directional positive on the same 7 CpGs) hold under proper Stage 1+3 calibration? If yes, the AD-instance pattern is robust. If the directional signal weakens, that is a finding that needs new outcome.md.

5. **Phase E (promote).** Bump card to v0.3. Update card JSON with the three new calibration anchors. Add v0_3_run_everything block with per-cohort findings.

6. **Phase F (push + deliver — seven-files protocol).** Per F.1 / F.2 / F.3 / F.4 in the master template. Push VAL-117+ Python scripts, prereg, outcome, results JSON, stratified JSON, manifest, metadata + Biological_Physics/README.md to GitHub. Deliver same Python + JSONs locally to Heath. Deliver Heath-only updates to all seven cookbook IP files (`GAPE_Evidence_Report_UPDATED.html`, `README_MASTER_v2_4.md` → bumped, `ad_immune_README_v0_3.md`, `ad_immune_card_v0_3.json`, `LESSONS_LEARNED.md`, `TESTING_CHECKLIST.md`, `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` if changed, `GAPE_Reproduction_Paper_v1.md` if methodology evolved).

**Estimated effort:** 4-5 hours. AD-immune is the most validated Stage 1 product; getting calibration discipline right here is highest-leverage.

**Sealed deliverables expected:** v0.3 promotion of card with three calibrated Stage 1+3 atlases. Confirmation or revision of the AD-instance bidirectional cancellation finding.

**Critical question this card will answer:** Does the directional Rule A 7-CpG panel hold up under structurally-separated calibration, or was VAL-051's d=+0.624 partly a within-cohort self-cal artifact?

---

### ▢ 2. CRC-epic card — HIGH PRIORITY (Tightening v2 + anatomy stratification)

**Status before audit:** Multiple VALs sealed including immune-class anatomy stratification (C18 + C19 light up, C20 stays silent), secretory-class variance findings, pre-diagnostic temporal structure on Italian Caltagirone cohorts.

**Atlas list to audit:**
- Layered Moss+Loyfer (Stage 2) — **NOW CALIBRATED** via VAL-112 ✓ (use VAL-112 thresholds directly)
- Xu-538 immune panel (Stage 1) — calibration anchor unknown
- Salas IDOL 6-cell + UniLIFE 19-cell (Stage 3) — calibration anchor unknown

**Step-by-step execution plan:**

1. **Phase A (inventory).** Open `cards/crc_epic/card_v2_x.json`. Confirm atlases_run. Note that Stage 2 Layered Moss+Loyfer is already calibrated via VAL-112 — reuse those thresholds. Stage 1 Xu-538 and Stage 3 Salas+UniLIFE need calibration anchors from VAL-114/115/116.

2. **Phase B (calibrate).** Stage 2 done. Stage 1+3 use the shared-task calibrations. No new B-phase work specific to this card.

3. **Phase C (re-score).** Re-score each cohort below under run-everything across all four atlases (Layered Moss+Loyfer + Xu-538 + Salas + UniLIFE):
   - **GSE51032** (n=235 breast + n=166 CRC vs n=424 controls — note: this cohort overlaps with breast-epic, so coordinate to score once if possible)
   - **TCGA-COAD pre-diagnostic anatomy stratification cohorts** — preserve the C18 / C19 / C20 anatomy stratification structure
   - **Anatomy-stratified contrasts:** C18 ascending colon, C19 sigmoid, C20 rectum — re-run each under the new calibration anchors

4. **Phase D (compare).** Critical question: does the C18+C19 light-up vs C20 silence pattern hold under proper Stage 1+3 calibration, or was it partly within-cohort self-cal artifact? Document direction and magnitude shifts.

5. **Phase E (promote).** Bump CRC-epic to v0.3.

6. **Phase F (push + deliver).** Standard split.

**Estimated effort:** 3-4 hours (Stage 2 already calibrated saves 1 hour).

**Critical question this card will answer:** Does CRC anatomy stratification (proximal vs distal vs rectal) hold up under multi-atlas triangulation?

---

### ▢ 3. Breast-epic card — HIGH PRIORITY (Phase 9 + 12 dancer cohorts)

**Status before audit:** VAL-047 secured pre-diagnostic Cohen's d up to +1.78 at >10yr pre-dx. Phase 9 + 12 cohorts (GSE51057, GSE51032).

**Atlas list to audit:** Same as CRC-epic (Stage 2 calibrated by VAL-112; Stage 1 + 3 need calibration via VAL-114/115/116).

**Step-by-step execution plan:**

1. **Phase A (inventory).** Open `cards/breast_epic/card_v2_x.json`. Confirm atlases_run matches CRC-epic. CHK-3.1C dedup audit is already done (Stage 2 via VAL-112; Stage 1+3 via shared tasks).

2. **Phase B (calibrate).** All atlases calibrated via shared tasks. No new B work.

3. **Phase C (re-score).** Run-everything re-score on:
   - **GSE51057** (n=146 cases + n=177 controls) — preserve TtD window stratification (>10yr / 5-10yr / 2-5yr / 0-2yr)
   - **GSE51032** (n=235 cases vs n=424 controls — coordinate with CRC-epic) — preserve TtD window stratification
   - **The dancer pre-diagnostic temporal contrasts** (>10yr / 5-10yr / <5yr to diagnosis)
   - For each window, output per-atlas per-tile Cohen's d.

4. **Phase D (compare).** Critical question — also Heath's recurring concern: do the dancer findings replicate under multi-atlas triangulation? Expected outcome based on the cardio reference: large n + coherent multi-tile patterns + cross-cohort replication should replicate. But running it is what matters. Document any direction or magnitude shifts.

5. **Phase E (promote).** Bump breast-epic to v0.3.

6. **Phase F (push + deliver).** Standard split.

**Estimated effort:** 3-4 hours (shares Stage 1 + 3 calibrations with CRC-epic if done together).

**Critical question this card will answer:** Does the d=+1.78 at >10yr pre-dx survive multi-atlas triangulation? This is the headline finding of EDEAR's pre-diagnostic story.

---

### ▢ 4. Lung-epic card — MEDIUM PRIORITY

**Status before audit:** VAL-063 sealed TCGA-LUAD HM450 n=29 paired tissue d=+1.020 (largest paired tissue effect in catalog). Smoking-stratified per CCL-009: ever-smoker d=+1.283, never-smoker underpowered.

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), Xu-538 Stage 1 + Stage 3 panels (need calibration via VAL-114/115/116); EpiSCORE LungRef pan-tissue (deferred — needs gene→CpG bridging).

**Step-by-step execution plan:**

1. **Phase A.** Open card JSON. Confirm atlases_run. EpiSCORE LungRef is deferred — note this; do not attempt to bridge it during this discipline wave (defer to v0.4+ separate sprint).

2. **Phase B.** Stage 2 + 1 + 3 calibrations all from shared tasks.

3. **Phase C re-score targets:**
   - **TCGA-LUAD adjacent-normal vs tumor** (n=29 paired) — preserve smoking stratification (ever-smoker n=22 vs never-smoker n=2)
   - **GSE63704** if accessible
   - **Smoking-discordant twin cohorts** if accessible

4. **Phase D.** Critical question: does the +1.020 paired tissue effect hold under multi-atlas re-scoring? Does smoking stratification (CCL-009 mandatory covariate) survive?

5. **Phase E + F.** Standard.

**Estimated effort:** 3 hours.

---

### ▢ 5. HCC-epic card — MEDIUM PRIORITY

**Status before audit:** VAL-101 sealed at O5 on TCGA-LIHC 25-tile etiology stratification. Has open question about whether the etiology stratification holds under multi-atlas re-scoring.

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), Xu-538 + Salas + UniLIFE.

**Step-by-step execution plan:**

1. **Phase A.** Open card JSON. Confirm atlases_run.

2. **Phase B.** Calibrations from shared tasks.

3. **Phase C re-score targets:**
   - **TCGA-LIHC etiology subgroups** (HBV, HCV, alcohol, NAFLD, mixed, unknown) — 25-tile stratification preserved
   - **GSE54503** cirrhosis vs HCC — preserve dose-response structure (healthy 0 → fibrosis +0.44 → cirrhosis +0.45 → HCC +0.63)

4. **Phase D.** Critical questions: (a) does the dose-response monotonicity hold? (b) does the viral-vs-non-viral substrate-restricted finding (CCL-010: ccfDNA plasma validated, whole-blood leukocyte NULL) hold?

5. **Phase E + F.** Standard.

**Estimated effort:** 3 hours.

---

### ▢ 6. Kidney-epic card — MEDIUM PRIORITY

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), Xu-538 + Salas + UniLIFE.

**Step-by-step execution plan:**

1. **Phase A.** Open card JSON.

2. **Phase B.** Calibrations from shared tasks.

3. **Phase C re-score targets:** TCGA-KIRC (which is part of the calibration cohort itself — see structural-separation caveat below); TCGA-KICH; consider TCGA-KIRP holdout.

4. **Phase D.** Standard comparison.

5. **Phase E + F.** Standard.

**Caveat (read carefully):** TCGA-KIRC adjacent-normal is part of the VAL-106/107/112/113 calibration cohort. Re-scoring the kidney-epic disease arm against KIRC tumor samples is fine (tumor and adjacent-normal are different samples), but cardio-epic-style "structurally separate calibration vs disease cohort" requires care here. The honest framing: kidney-epic disease cohorts get scored against the same atlases other cards use; the atlas calibration uses adjacent-normal kidney + prostate; the calibration cohort and the disease cohort are NOT the same samples even though they come from the same TCGA project. This is acceptable structural separation for cookbook purposes (per CCL-041 guidance: "samples-not-shared between calibration and scoring"), but **must be flagged in the kidney-epic v0.3 prereg as a caveat**.

**Estimated effort:** 3 hours + careful structural-separation documentation.

---

### ▢ 7. Cervical-epic card — MEDIUM PRIORITY

**Status before audit:** VAL-076/077 sealed null on LBC pathway with panel-transferability caveat (CHK-0.5).

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), Xu-538 + Salas + UniLIFE.

**Step-by-step execution plan:**

1. **Phase A.** Open card JSON. Note CHK-0.5 panel-transferability caveat for LBC substrate.

2. **Phase B.** Calibrations from shared tasks.

3. **Phase C re-score targets:** WID-CIN (n=2,254) — CIN1/2/3 and invasive contrasts, the original VAL-009 cohort. Preserve CIN stratification.

4. **Phase D.** Critical question: does the CIN3 monotonic Normal < CIN3 < SCC pattern (VAL-073 Verlaat Amsterdam tissue arm) hold under multi-atlas re-scoring? Cohort heterogeneity between Verlaat and Stockholm/Oslo cohorts is under investigation (HPV-stratification of normals leading explanation) — if multi-atlas triangulation cleans this up, that is a finding.

5. **Phase E + F.** Standard.

**Estimated effort:** 3 hours.

---

### ▢ 8. Glioma-epic card — MEDIUM-LOW PRIORITY

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), MARLIN training scaffold (deferred), Sabedot GeLB (deferred), Tanaka 2025 neural (deferred — acquisition + bridge engineering).

**Step-by-step execution plan:**

1. **Phase A.** Most of glioma-epic's atlas list is in atlases_deferred — actual atlases_run is mostly layered Moss+Loyfer (Stage 2) plus the standard Stage 1+3. Calibration sprint will be lighter on engineering for this card. Phase A (acquisition + bridging of MARLIN / Tanaka 2025 / Sabedot) is its own project — defer to v0.4+.

2. **Phase B.** Calibrations from shared tasks. No new bridge engineering.

3. **Phase C re-score targets:** TCGA-LGG + TCGA-GBM blood + tissue cohorts. Preserve the LGG-louder-than-GBM finding from VAL-090.

4. **Phase D.** Critical question: does the cortical-neuron fraction d=+1.96 [+1.62, +2.31] at array resolution (VAL-090) hold under run-everything?

5. **Phase E + F.** Standard. Defer the deferred-atlas engineering (MARLIN / Tanaka / Sabedot) to v0.4+ as a separate sprint with its own VAL ID block.

**Estimated effort:** 2 hours (light because most atlases are still deferred and won't be calibrated this round).

---

### ▢ 9. Pancreatic-epic card — MEDIUM-LOW PRIORITY

**Status before audit:** VAL-066/067/068 sealed (TCGA-PAAD, GSE49149, GSE74071) plus VAL-069 324-CpG directional fallback panel. PDAC confirmed as second bidirectional-cancellation disease.

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), Xu-538 + Salas + UniLIFE.

**Step-by-step execution plan:**

1. **Phase A.** Open card JSON. Note the 324-CpG directional fallback panel is a separate scoring layer that does NOT depend on H_min calibration (z-score normalized). Document this distinction in the card audit — the directional panel's anchor is its own and is not part of this calibration sprint.

2. **Phase B.** Standard atlas calibrations from shared tasks.

3. **Phase C re-score targets:** TCGA-PAAD, GSE49149, GSE74071. Preserve the bidirectional-cancellation Stage 1 pooled null + 324-CpG directional positive structure (the AD-instance pattern repeated).

4. **Phase D.** Critical question: does PDAC's bidirectional-cancellation pattern hold under multi-atlas re-scoring? Does the Rotterdam Study cohort signal at 2-5yr pre-dx survive?

5. **Phase E + F.** Standard.

**Estimated effort:** 3 hours.

---

### ▢ 10. Prostate-epic card — LOW PRIORITY

**Status before audit:** Less validated than other cards; VAL inventory smaller. Card built tissue-first because no public blood cohort available.

**Atlas list:** Layered Moss+Loyfer Stage 2 (calibrated VAL-112), Xu-538 + Salas + UniLIFE.

**Step-by-step execution plan:**

1. **Phase A.** Open card JSON.

2. **Phase B.** Calibrations from shared tasks.

3. **Phase C re-score targets:** TCGA-PRAD adjacent-normal vs tumor (note: PRAD adjacent-normal is part of VAL-106/107/112/113 calibration cohort — same caveat as kidney-epic). GSE269244 ($n=238$ African-American men, EPIC 850K) for the African-American cohort tissue arm.

4. **Phase D.** Critical question: does the GSE269244 paired d=+0.497 (VAL-058) hold under multi-atlas re-scoring?

5. **Phase E + F.** Standard. Document the structural-separation caveat in the v0.3 prereg same as kidney-epic.

**Estimated effort:** 3 hours + careful structural-separation documentation.

---

## Shared sub-tasks (do once, benefits multiple cards)

### ▢ Shared Task A: Calibrate Xu-538 immune panel on a structurally-separated whole-blood healthy cohort — VAL-114

**Atlas:** Stage 1 Xu-538 immune panel (538 CpGs)
**Calibration cohort candidate:** GSE40279 Hannum n=656 (already on disk for VAL-006)
**Substrate:** HM450
**Output:** `validation_runs/VAL-114_xu538_calibrate/calibration_results.json` + per_sample CSV
**Used by:** AD-immune, CRC-epic, breast-epic, lung-epic, hcc-epic, kidney-epic, cervical-epic, glioma-epic, pancreatic-epic, prostate-epic — basically every card. **DO THIS FIRST AFTER THE TODO IS APPROVED.**
**Estimated effort:** 2 hours (calibration cohort already on disk; same script template as VAL-112).

**Step-by-step:**
1. Pre-flight: confirm GSE40279 path. Run beta distribution sanity check. Run cross-cohort baseline check.
2. Copy `val_112_calibrate.py` to `validation_runs/VAL-114_xu538_calibrate/val_114_calibrate.py`. Edit atlas path to point to Xu-538.
3. Run on GSE40279 n=656. Output: per-sample CHK-3.1A + CHK-3.1B + per-tile A-scores.
4. Seal results: q5 threshold + per-tile distribution.
5. Push GitHub: VAL-114 py + prereg + calibration_results.json + per_sample.csv + Biological_Physics/README.md update.
6. Deliver to Heath: VAL-114 Python + results JSON.

### ▢ Shared Task B: Calibrate Salas Blood.EPIC IDOL 6-cell + UniLIFE 19-cell on whole-blood healthy cohort — VAL-115 + VAL-116

**Atlases:** Stage 3 Salas IDOL (350 CpGs) + UniLIFE (1,906 CpGs)
**Calibration cohort:** Same as Shared Task A
**Output:** `validation_runs/VAL-115_salas_idol_calibrate/` + `validation_runs/VAL-116_unilife_calibrate/`
**Used by:** Every card with a Stage 3 component
**Estimated effort:** 2 hours each (4 hours combined; can run in parallel).

**Step-by-step:**
Same structure as Shared Task A, run twice (once per atlas). Both can run in parallel since they don't share input/output paths.

### ▢ Shared Task C: Audit + deduplicate Xu-538 / Salas IDOL / UniLIFE atlas files

**Check:** Do any of these have duplicate CpGs (CHK-3.1C)? CCL-047 was specifically about Loyfer, but the same dedup discipline applies to every atlas.
**Output:** Per-atlas dedup audit results; deduplicate if needed with audit-trail preservation.
**Estimated effort:** 30 minutes total.

### ▢ Shared Task D: Build the run-everything framework as a reusable script

Currently each card needs its own Phase B + Phase C scripts. The VAL-112 + VAL-113 sprint produced templates that can be generalized. Build a parameterized `run_everything.py` that takes:
- A card name
- A list of (atlas_path, calibration_VAL_id) pairs
- A list of (cohort_path, cohort_name, group_assigner_function) tuples
- Output directory

And produces all Phase B calibrations + Phase C re-scoring outputs in the standardized format. This will make subsequent cards' calibration audits 10× faster.

**Estimated effort:** 3 hours.

---

## Execution order (recommended)

**Wave 1 (do before any cards): Shared Tasks A + B + C + D.** These unlock every card's Stage 1 + 3 calibration in one pass. ~10 hours.

**Wave 2 (top-priority cards): AD-immune + CRC-epic + breast-epic.** These are most-validated, most-commercially-relevant, and benefit most from the shared Stage 1 + 3 calibrations. ~10-12 hours.

**Wave 3 (medium-priority cards): lung-epic + hcc-epic + cervical-epic + pancreatic-epic.** ~12 hours.

**Wave 4 (medium-low / low-priority cards): kidney-epic + glioma-epic + ~~prostate-epic~~ ✓ DONE 2026-04-30.** ~7-8 hours (kidney + prostate need careful structural-separation documentation; glioma is light because most atlases are still deferred).

**✅ prostate-epic — DONE 2026-04-30.**
- VAL-117 ProstateRef Phase B calibration on TCGA n=210 (KIRC + PRAD adjacent-normal HM450K sesame Level 3): commit `40ce175`
- VAL-118 first execution sealed `O5_LE_DIRECTION_FLIP_UNANTICIPATED`: commit `edf6229` (preserved as discipline-discovery record)
- VAL-118 amendment sealed `O1_MULTI_ATLAS_CONVERGENT + O2_LE_TILE_DIFFERENTIATING (LE_NEGATIVE) + O4_STAGE_3_IMMUNE_SHIFT_PROMINENT`: commit `58ecd16`
- Phase D v0.2-vs-v0.3 outcome comparison: commit `c5ee9d5`
- F.1 deliverables (cohort manifest + clinical metadata + stratified-results JSON + Biological_Physics/README.md update): commit `388e5b0`
- Heath delivery: 2026-04-30 (re-delivered with full F.3 completion 2026-04-30 evening after audit)
- Headline finding: ProstateRef LE tile reads tumor at d_paired = −0.767 (luminal dedifferentiation); five-vs-one ProstateRef direction split is the v0.3 prostate cancer methylation-architecture signature
- Tier promoted: stage_2_only_validated → multi_modal_validated_plus_multi_atlas_calibrated
- DISC-PROSTATE-001/002/003 sealed; CHK-2.7 (magnitude-based |d| with direction labels for cell-of-origin atlas preregs) + CHK-2.8 (substrate-floor-based CHK-3.1B coverage thresholds) formalized as cookbook-wide rules
- LESSONS_LEARNED.md gains prostate-LL-006/007/008
- Atlas vault: ProstateRef CpG-bridged matrix added as third successful EpiSCORE bridge alongside HeartRef and BreastRef

**Total estimated effort:** ~40 hours of execution time spread across however many sessions Heath wants. Most steps run in background; effective wall-clock is shorter.

---

## Tracking + accountability

- **Per-card sealed deliverables go to GitHub:** VAL-NNN py + prereg + outcome + results JSON + stratified JSON + cohort manifest + clinical metadata + Biological_Physics/README.md updates. Per the per-card workflow rule: PUSH these to repo.
- **Heath-only deliverables (cookbook IP, NEVER push):** Evidence Report HTML, README_MASTER, card README + card JSON, LESSONS_LEARNED.md, TESTING_CHECKLIST.md, PIPELINE_REFERENCE updates, and this TODO document.
- **Update this document** with ✓ next to each completed task + commit hash. When all cards are at v0.3, this document gets archived as the v0.3 calibration discipline wave audit-trail.
- **No new cards** until at least Wave 1 + Wave 2 are complete. Wave 3 and Wave 4 can proceed in parallel with new card work IF the new card uses ONLY atlases that are already calibrated.

---

## Reference

### Templates and scripts (next chat must locate these before sprint 1)
- VAL-112 + VAL-113 outcome.md template: `/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-112_113_unified/outcome.md`
- Calibration script template: `validation_runs/VAL-112_run_everything/val_112_calibrate.py`
- Phase C scoring template: `validation_runs/VAL-112_run_everything/val_112_phaseC.py`
- Caggiano-style atlas bridging template: `atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/bridge_caggiano_to_array.py`
- HM450 hg19 manifest extraction template: `atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/extract_manifest.R`

### Calibration cohorts
- HM450 sesame Level 3 cards: TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 at `/home/claude/edear_working/VAL-106/calibration_betas/`
- Whole-blood cards: GSE40279 Hannum n=656 at the VAL-006 working directory location

### Heath-only cookbook IP files (current versions; bump per sprint)
- `cardio_epic_card_v0_2_2.json` — example card JSON, the cardio worked example
- `cardio_epic_README_v0_2_2.md` — example card README, the cardio worked example
- `TESTING_CHECKLIST.md` — pre-flight checklist, updated as new checks are surfaced
- `LESSONS_LEARNED.md` — running log of cross-sprint lessons
- `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` — operational pipeline spec
- `README_MASTER_v2_4.md` — cookbook master spec; bump version every cycle
- `GAPE_Reproduction_Paper_v1.md` — reproduction-grade methodology documentation
- `GAPE_Evidence_Report_UPDATED.html` — live evidence report; append a VAL block per sprint
- This TODO document (CROSS_CARD_CALIBRATION_TODO_vX_Y.md)

### Companion published material (read for context)
- Floor Breach companion paper: DOI 10.5281/zenodo.18702042 — read for context on why H_min(class) values stay frozen during this wave
- EDEAR Comprehensive Paper (current public-facing version): `/mnt/user-data/outputs/EDEAR_Comprehensive_Paper_AUDITED.tex` — context for what the world sees about each card

### GitHub repository (push targets)
- Public repo: `https://github.com/hmahaffeyges/IAM-Validation`
- Push tree: `Biological_Physics/`
- Per push: VAL-NNN py, prereg.md, outcome.md, results JSON, stratified JSON, cohort manifest, clinical metadata, Biological_Physics/README.md update
- Heath gets local copies of every pushed Python + JSON in F.2

---

**EDEAR commercial deployment unaffected throughout this discipline wave per CCL-037.** Production deployment runs customer-specific calibration regardless of cookbook-side audit state.

---

## Changelog

- **v0.3 (2026-04-29):** Initial document after VAL-112 + VAL-113 cardio sprint completion. Established master template, per-card audit table, shared sub-tasks, execution order.
- **v0.4 (2026-04-30 morning):** Added (1) READ THIS FIRST guardrails section with no-fabrication rule, per-card workflow, GitHub vs Heath-only split, surgical edits rule, TESTING_CHECKLIST primacy, reproducibility triple, language discipline, Floor Breach companion context. (2) Pre-flight checklist (file availability, VAL ID claiming, atlas integrity, beta distribution sanity, cross-cohort baseline). (3) VAL ID reservations table. (4) Cardio-epic reference example marked DONE with full Phase A-F walkthrough as the worked template. (5) Per-card explicit step-by-step execution plans for all 10 cards. (6) Critical question per card stating what hypothesis the audit is testing. (7) Step-by-step instructions for Shared Tasks A and B. No content removed from v0.3.
- **v0.4 (2026-04-30 afternoon revision):** Added (1) three new top-level guardrails: #10 EXHAUSTIVE COHORT SURVEY before any card sprint begins, #11 CALIBRATION BEFORE TESTING is the inviolable order, #12 RUN EVERYTHING through ALL atlases. (2) Phase 0 (cohort survey) added to master template — now seven phases: 0/A/B/C/D/E/F. (3) Phase F restructured into F.1 (PUSH GitHub) / F.2 (DELIVER Heath same-as-pushed locally) / F.3 (DELIVER Heath-only cookbook IP) / F.4 (update TODO). (4) F.3 enumerated as eight named files: GAPE_Evidence_Report_UPDATED.html, README_MASTER_v2_4.md, card README (e.g. cardio_epic_README_v0_2_2.md), card JSON (e.g. cardio_epic_card_v0_2_2.json), LESSONS_LEARNED.md, TESTING_CHECKLIST.md, EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md, GAPE_Reproduction_Paper_v1.md. (5) F.2 explicitly states Heath gets local copies of every Python script and JSON pushed to GitHub. (6) Cardio reference example expanded with Phase 0 worked example (14 cohorts surveyed, 3 selected, 11 deferred) and Phase F broken into F.1/F.2/F.3/F.4 with all named files cited. (7) AD-immune card execution plan extended with explicit Step 0 (cohort survey) as the template; standard Phase 0 + Phase F blocks added at the top of the per-card section so the next chat doesn't have to repeat the protocol for every card. (8) Reference section reorganized into templates / calibration cohorts / Heath-only cookbook IP files / companion material / GitHub repo subsections, with the named file roster Heath specified. No content removed from prior v0.4.
- **v0.5 (2026-04-30 evening):** Audit-driven fixes after re-reading actual cardio v0.3 README + card JSON truth-state. Added (1) Guardrail #13: full CCL/CHK enumerated checklist that every card audit must apply (CCL-040/041/042/043/046/047/048/049 + CHK-3.1A/B/C, CHK-5.7/5.8/5.9/5.10/5.12/5.13, CHK-0.7, CHK-7.6 all listed with one-line descriptions and citing cardio sprint as worked example for each). (2) Pre-flight: substrate inventory per cohort (different preprocessing pipelines need different CHK-3.1A envelopes — cardio surfaced four substrates with 24-percentage-point f_extreme distribution gap). (3) Pre-flight: web-verify every external citation introduced in new content (CHK-5.13 — catches Konigsberg→Cuadrat and Liu→Tian factual errors). (4) Phase A restructured into A.1 (inventory) / A.2 (atlas acquisition with target version per atlas) / A.3 (bridge engineering as reusable infrastructure) / A.4 (integration testing). (5) Phase E expanded with structured templates that every card produces: structured `atlases_deferred` block (atlas / target version / unblock dependency), Per-disease scoring policy block (what claimed / what NOT claimed), DISC-CARD-NNN discoveries section, Validation evidence summary per VAL (cohort/n/substrate/design/QC/outcome/Cohen's d/interpretation/prereg SHA-256), atlas_vault INVENTORY.json update with calibration anchor, Cookbook-wide CCL cross-reference section. (6) Cardio reference example rewritten to track all five sealed VALs in order (VAL-108 stroke null O3, VAL-109 PAH O2, VAL-110 aortic Stage 1 +1.08 O2, VAL-111 EpiSCORE atlas-fitness deferral O3, VAL-112 Loyfer+HeartRef calibration on TCGA n=210, VAL-113 Caggiano TIM array-bridged calibration) with their distinct outcome classes (O2 differentiating, O3 floor-dominated, O3 undifferentiated). (7) Cardio sprint's 8 DISC-CARDIO discoveries enumerated in the reference example so the next chat sees what discoveries look like for a real card. (8) Atlas vault GitHub commit discipline noted (commit 57beb38 cited as the cardio example). No content removed from prior v0.4.

