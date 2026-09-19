# VAL-119 — Outcome

**Sealed:** 2026-05-01T03:46:00Z
**Outcome class:** `O1_BLADDERREF_CALIBRATION_SEALED`
**Pre-registration chain:**
- `prereg.md` SHA-256: `04d9d0d36faaf3bc6051c5f852f3bcb9d2ab48c437982b714ff8573d4cad178a` (sealed 2026-05-01T03:35:46Z)
- `prereg_amendment.md` SHA-256: `c3015ca3ba25f6c13f4f93fec85edea8506f64472657d03b59ed9ccda8355787` (sealed 2026-05-01T03:38:56Z; atlas SHA correction for NaN serialization fix; no β data observed under original prereg)

---

## Headline

EpiSCORE BladderRef CpG-bridged matrix (2,696 unique 450K CpGs × 4 bladder cell types) is calibrated against TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (substrate: TCGA HM450K sesame Level 3). All three CHK gates clear:

- **CHK-3.1A** (full-genome substrate baseline): 206/210 pass (98.1%); observed f_extreme 55.9% ± 2.4% (consistent with VAL-106 sesame Level 3 sealed baseline 55.87% ± 2.44%); observed f_middle 7.37% ± 0.75%
- **CHK-3.1B** (atlas-subset coverage, threshold ≥80% per CHK-2.8 substrate-floor for TCGA HM450K small atlas subsets): 210/210 pass (100%); observed q5 = 86.15%
- **CHK-3.1C** (atlas dedup): 2,696 unique probes, 0 duplicates; passed

Per-tile healthy-floor distributions seal as the BladderRef calibration anchor for bladder-epic v0.1 production scoring on HM450K sesame Level 3 substrate.

---

## Per-tile healthy-floor distributions (n=206 QC-passed)

| Tile | Cell type | Class | H_min | Mean A | SD | q5 | q50 | q95 | Within-cohort range |
|------|-----------|-------|-------|--------|------|------|------|------|---------------------|
| **EC**  | Vascular endothelial            | stromal   | 0.862950 | 0.4087 | 0.0100 | 0.3972 | — | 0.4265 | 0.0565 |
| **Epi** | Urothelial epithelium *(BC cell of origin)* | secretory | 0.843264 | **0.4135** | **0.0066** | 0.4004 | — | 0.4219 | **0.0410** |
| **Fib** | Fibroblasts (stromal)           | stromal   | 0.862950 | 0.4875 | 0.0090 | 0.4770 | — | 0.5020 | 0.0694 |
| **IC**  | Immune cells (intra-bladder)    | immune    | 0.838889 | 0.4106 | 0.0086 | 0.4001 | — | 0.4263 | 0.0504 |

**Epi tile observation.** Lowest within-cohort variance (sd=0.0066, range=0.0410) of all four tiles. Tight healthy-floor distribution means small disease-driven A-score shifts on the bladder cancer cell of origin (urothelial epithelium) are detectable above the calibration noise floor. This is the operationally most-important tile for bladder-epic v0.1 disease scoring — the bladder analog of prostate's LE tile (which had sd=0.0041, range=0.0293; both are the cell-of-origin tile with tightest within-cohort variance in their respective atlases).

**Fib tile observation.** Highest mean A (0.4875) of all four tiles, well separated from the other three (EC, Epi, IC all cluster 0.41±0.005). This separation is the operational signal that BladderRef's 4-cell-type set spans markedly different gene-promoter methylation profiles for the marker genes in bladder tissue — the prerequisite for gene-promoter atlas family success per DISC-CARDIO-004 + DISC-PROSTATE-001.

---

## Pre-locked outcomes — what fired

| Outcome | Pre-locked criterion | Observed | Status |
|---|---|---|---|
| **O1_BLADDERREF_CALIBRATION_SEALED** | CHK-3.1A pass ≥90%, CHK-3.1B pass ≥95%, CHK-3.1C pass, max within-cohort tile range ≥ 0.02 | 98.1% / 100% / pass / 0.0694 | **FIRED** |
| O2_BLADDERREF_CALIBRATION_PARTIAL | CHK-3.1A 75-90% OR CHK-3.1B 85-95% | n/a (O1 fired first) | not fired |
| O3_BLADDERREF_TISSUE_FLOOR_DOMINATED | All tiles within-cohort range < 0.02 | min 0.0410, max 0.0694 — all tiles clear 0.02 floor | not fired |
| O4_BLADDERREF_BRIDGE_FAILURE | CHK-3.1C dedup fails or all-NaN tiles | 0 duplicates, all 4 tiles produce valid A-scores | not fired |
| O5_BLADDERREF_UNEXPECTED | Anything else | n/a | not fired |

---

## Comparison to cardio sprint VAL-111 / prostate sprint VAL-117 precedent

EpiSCORE atlas family behavior across three tissues now:

| Atlas | n_CpGs | n_cell_types | Atlas family | Calibration cohort | Outcome | Max within-cohort tile range |
|---|---|---|---|---|---|---|
| EpiSCORE HeartRef (VAL-111) | 3,727 | 5 (CM/EC/FB/MP/SMC) | gene-promoter | 3 cardio cohorts (n=652) | O3_TISSUE_FLOOR_DOMINATED | 0.0152 |
| EpiSCORE ProstateRef (VAL-117) | 2,603 | 6 (BE/EC/Fib/LE/Leu/SM) | gene-promoter | TCGA n=210 | O1 sealed | 0.0597 |
| **EpiSCORE BladderRef (VAL-119)** | **2,696** | **4 (EC/Epi/Fib/IC)** | **gene-promoter** | **TCGA n=210** | **O1 sealed** | **0.0694** |

**DISC-BLADDER-001 candidate (propagates to LESSONS_LEARNED.md):** Gene-promoter atlas family fitness is NOT a function of cell-type count alone. BladderRef's 4 cell types produced larger within-cohort tile range (0.0694) than ProstateRef's 6 cell types (0.0597) and far larger than HeartRef's 5 cell types (0.0152). The discriminating variable is per-tissue cell-type distinctness at the gene-promoter level for the marker genes used. Bladder's 4 compartments (urothelial barrier-secretory epithelium, intra-bladder vasculature, fibroblast stroma, intra-bladder immune) ARE markedly distinct gene-promoter programs even though there are fewer of them. Cardiac cell types (CM, EC, FB, MP, SMC), despite being 5 in number, have similar gene-promoter methylation profiles for the marker genes Zhu/Teschendorff selected for that tissue. Per CHK-5.12 logic, this finding extends DISC-CARDIO-004 + DISC-PROSTATE-001 by clarifying that cell-type COUNT is not a useful predictor; cell-type DISTINCTNESS is.

**Operational takeaway for atlas selection across remaining EpiSCORE tissues:** Each tissue's BridgedRef calibration must be smoke-tested independently per VAL-094/111/117/119 protocol. Tissue distinctness at the gene-promoter level is not predictable a priori from the EpiSCORE source matrix dimensions.

---

## What VAL-119 unblocks

BladderRef enters bladder-epic v0.1 `atlases_run` block with calibration anchor VAL-119. Phase C scoring on bladder cohorts (TCGA-BLCA tumor + adjacent-normal at minimum, plus any secondary cohorts surfaced in the Phase 0 cohort survey) can now run multi-atlas:

- Stage 1: Xu-538 immune panel (within-cohort self-cal at v0.1; Wave 1 VAL-114 v0.X+1 promotion path)
- Stage 2: Layered Moss+Loyfer (calibrated VAL-112) + EpiSCORE BladderRef (calibrated VAL-119) + Caggiano CelFiE TIM (calibrated VAL-113) — three calibrated atlases for run-everything Phase C
- Stage 3: Salas Blood.EPIC IDOL + UniLIFE 19-cell + Caggiano TIM immune subset (within-cohort self-cal at v0.1)

This is the patient-flow-natural sprint structure: Stage 1 red flag → Stage 2 cell-of-origin localization with three calibrated atlases → Stage 3 immune fine-tune.

---

## Reproducibility triple (CHK-7.6)

### Source code
`val119_bladderref_calibrate.py` — Python 3.12 stdlib + numpy. Embedded inline in this directory; SHA-256 will be recorded post-script-freeze. Runtime: 162.1 sec on 210-sample cohort.

### Inputs
1. **Bridged atlas matrix:** `episcore_bladderref_cpg_bridged.csv`
   - Path: `Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv`
   - SHA-256: `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`
   - Size: ~165 KB; 2,696 CpGs × 4 cell types + weight + EID
2. **Calibration cohort:** TCGA-KIRC adjacent-normal n=160 + TCGA-PRAD adjacent-normal n=50
   - Source: GDC API `https://api.gdc.cancer.gov/data/{file_id}`
   - Manifest path: `/home/claude/edear_working/bladder_epic/cohort_acquisition_manifest.json`
   - Substrate: TCGA HM450K sesame Level 3
   - File format: tab-separated (CpG_id, β_value); ~12-13 MB each

### Environment
- Python 3.12.3
- numpy 2.4.4
- No pandas dependency (csv stdlib only)
- Runtime: 162.1 sec
- Memory peak ~500 MB

### Headline output
- `VAL-119_calibration_results.json` — per-tile mean, sd, n, q2.5, q5, q50, q95, q97.5, min, max, within_cohort_range
- `VAL-119_per_sample_calibration.csv` — 210 rows × 24 columns (sample_id, project, n_cpgs_genome, f_extreme, f_middle, median, chk_3_1a_passed, n_atlas_cpgs_present, coverage, chk_3_1b_passed, A_EC, A_Epi, A_Fib, A_IC, n_cpgs_EC, n_cpgs_Epi, n_cpgs_Fib, n_cpgs_IC)

---

## Sealed-against audit chain

This outcome seals against:
- The bladder-epic v0.1 Phase 0 cohort survey signed off 2026-04-30
- Calibration TODO v0.5 Phase B requirement
- Guardrail #11 (calibration before testing is the inviolable order)
- CCL-041 (prereg locked before β read; amendment locked before re-execution; no β data observed before either seal)
- CCL-046 (atlas selection traces to canonical-document-named candidates per CHK-5.12)
- CHK-2.7 (magnitude-based thresholds for direction-ambiguous outcomes)
- CHK-2.8 (TCGA HM450K substrate-floor for atlas-subset coverage threshold ≥80%)
- DISC-CARDIO-004 + DISC-PROSTATE-001 (gene-promoter atlas family per-tissue distinctness rule)

**Outcome sealed. No outcome added post-hoc. No threshold relaxed post-hoc.**
