# Immune Cell Page Audit Coverage — Running Tracker

**Method:** For each item in the immune-intelligence audit (73-VAL canonical + 39 CCLs + per-card lessons + atlas-level findings), track which immune cell page (or other destination) carries it. Marked with [✓] when placed in an immune cell page so far, [→] when destined for an immune cell page in this batch round, and [ELSEWHERE] when the natural home is in another class (secretory page for HCC, terminal page for glioma cortical-neuron, etc).

**Pages drafted (19 of 19 immune cell pages — final state per cell-type inventory canonical doc):**

B-cell lineage (4 pages):
- B_cells.md
- naive_B_cells.md (NEW 2026-05-09)
- memory_B_cells.md (NEW 2026-05-09)
- plasma_cells.md (NEW 2026-05-09)

T-cell lineage (5 pages):
- CD4_T_cells.md
- CD8_T_cells.md
- naive_CD4_T_cells.md (NEW 2026-05-09 — separate from naive_CD8 because chronic HIV selectively depletes CD4 lineage; opposite-direction risk)
- naive_CD8_T_cells.md
- memory_T_cells.md
- regulatory_T_cells.md

NK and granulocyte lineages (4 pages):
- NK_cells.md
- neutrophils.md
- eosinophils.md
- basophils.md

Monocyte/macrophage lineage (4 pages):
- monocytes.md
- macrophages.md
- microglia.md (NEW 2026-05-09 — CNS-resident, separate from peripheral macrophages because neurodegeneration-vs-tumor patterns can move opposite)
- kupffer_cells.md (NEW 2026-05-09 — liver-resident, separate because in advanced HCC Kupffer DEPLETE while peripheral TAMs UP — opposite-direction)

Other immune (1 page):
- dendritic_cells.md

**Page DELETED 2026-05-09:**
- megakaryocytes.md — wrong class assignment. Per cell_type_inventory.md, megakaryocyte (caggiano_celfie_tim label `megakaryocyte`) is a platelet-producing progenitor, not an immune cell. The progenitor walkthrough's megakaryocyte.md page is the canonical home.

**Architectural refinement (2026-05-08 final):** Integrated multi-cell disease pattern stories now live on the immune class page Section 4, organized by pattern type (A through I) with KISS plain-language names ("scrambled" pattern, "rebalancing" pattern, "early signal" pattern, etc.) and at-a-glance bullet summaries of which cells go which direction. Cell pages carry cell-specific findings only and point to the relevant class-page Pattern subsection for the integrated story. This refinement triggered light revision of all 4 already-drafted cell pages; new cell pages drafted under the simpler structure.

**Audit refinement (2026-05-09).** Six structural decisions made on top of the 14 pages originally drafted: (1) megakaryocyte page DELETED — wrong-class assignment, megakaryocyte is in progenitor not immune. (2) naive_CD4_T_cells page ADDED — separate from naive_CD8 because chronic HIV preferentially depletes CD4 lineage and CMV chronic infection drives differential CD8/CD4 dynamics; bidirectional-discipline rule fires. (3) naive_B_cells page ADDED — VAL-095 documented aBnv elevation specifically at 0-2yr pre-dx breast cancer; near-diagnosis pattern distinct from regulatory T cell long-pre-dx signal. (4) memory_B_cells page ADDED — chronic-viral-infection memory inflation and autoimmune disease patterns drive memory B expansion while naive B can be depressed; opposite-direction risk. (5) plasma_cells page ADDED — multiple myeloma drives plasma expansion while general B cells stay normal/DOWN; B-cell lymphomas show opposite pattern; canonical B-lineage opposite-direction case. (6) microglia and kupffer_cells pages ADDED — tissue-resident macrophage subtypes diverge from peripheral macrophages in organ-specific disease patterns (microglia activate in neurodegeneration while peripheral stay normal; Kupffer DEPLETE in advanced HCC while peripheral TAMs elevate). Final immune cell page count: 19.

**Card v1.0-draft updated 2026-05-09:** `cell_types_of_interest` now lists 19 entries matching the production atlas (B_cells, naive_B_cells, memory_B_cells, plasma_cells, CD4_T_cells, CD8_T_cells, naive_CD4_T_cells, naive_CD8_T_cells, memory_T_cells, regulatory_T_cells, NK_cells, neutrophils, eosinophils, basophils, monocytes, macrophages, microglia, kupffer_cells, dendritic_cells). New `_cell_types_atlas_provenance`, `_cell_type_to_page_mapping`, and `_grouping_rationale` blocks added for full traceability per cell_type_inventory.md.

**Class page Section 3 updated 2026-05-09:** Cell-type-in-detail section reorganized by immune lineage (B-cell lineage 4 pages, T-cell lineage 5 pages, NK/granulocyte 4 pages, monocyte/macrophage 4 pages, dendritic 1 page). All 19 cells described with research-anchored content. Megakaryocytes paragraph removed (wrong-class).

**Class page Section 4.6 added 2026-05-09 — Comprehensive condition reference table.** New section listing every disease/condition the framework reads with: (1) specific immune cells flagged + direction, (2) temporal phase (long pre-dx / mid pre-dx / near-dx / active / chronic), (3) cross-class organ pages to read when the immune signal points to a likely affected organ. 39 conditions catalogued across 6 category tables: cancers (16 conditions), chronic inflammatory (6), neurodegenerative (4), cardiovascular (2), chronic infectious (4), other/context-driven (7). Includes the "two-phase pattern" framing — immune class signal often elevates in long pre-dx then quiets at diagnosis while the affected organ tile dominates, so customers should read both their immune page and the relevant organ-class page. Acknowledges research is ongoing and the catalog continues to grow.

**Audit-matrix coverage verification (2026-05-09):**
- All 54 unique VAL IDs from immune_intelligence_audit_matrix_v2.md are referenced in the immune walkthrough deliverables.
- All 17 CCLs from the audit matrix are referenced in the immune walkthrough deliverables.
- All 10 per-card lessons from the audit matrix are referenced in the immune walkthrough deliverables.
- Complete audit-matrix coverage: 81/81 evidence anchors captured.

---

## Open architectural questions raised 2026-05-08 (deferred for dedicated discussion)

Heath raised three architectural items during the immune cell page drafting session that each warrant their own thinking. Captured here so they don't get lost. None block the immune walkthrough; all are roadmap items for after immune class is complete.

### Q1 — Per-class weighted A-score in addition to per-cell A-scores

We currently report an A-score per cell type EDEAR detects in the customer's blood. The question Heath raised: are we creating a weighted A-score per class as well? E.g., one immune class A-score that aggregates the cell-type-level A-scores in some weighted way, alongside the per-cell A-scores.

Considerations for the per-class weighted A-score:

- The class page bar gauge already implies a class-level A-score the customer can compare their cells against. Without a per-class aggregate score, the customer plots multiple individual cell scores on the same gauge but does not have a single "your immune class score" reading.
- A per-class score would also enable trajectory at the class level — useful for the customer who wants to see "is my immune class trending up overall" at a glance, beyond the per-cell trajectories.
- Weighting question: how to weight individual cell scores into a class score. Options include: cell-fraction-weighted (cells more abundant in plasma contribute more); equal-weight (all cell types contribute equally); biological-importance-weighted (regulatory cells weighted differently from effector cells, etc.); informativeness-weighted (cells with more research-supported signal in the class get more weight).
- This is engine-level math that needs to be agreed upon and locked, then back-propagated into the website page architecture and the report templates.

This is on the roadmap but requires dedicated thinking. Heath flagged: "we mentioned having an A score for every cell, are we creating a weighted one for each cell class too?"

### Q2 — Where the customer's per-cell A-scores actually appear

The website pages do NOT display individual customer A-scores — those are unique to each customer and live in the customer's report (PDF and HTML web view). The website pages display a class-level bar gauge and explanatory content; the customer's specific number lives in the report.

Heath flagged this needs more architectural clarity: "the tech webpage won't contain them, only the pdf we give them. We need to probably create a cellular age page too that they can reference."

Specific items that need architectural agreement:

- The customer's report (PDF and HTML web view) shows the customer's specific A-scores.
- The website pages show framing, explanation, examples on the bar gauge, and research observations — but not customer-specific scores.
- The customer plots their score from the report onto the website bar gauge to see where they are on the class-level distribution.
- This separation lets customers share their report directly without exposing other customers' data; it also lets the website serve as a public-readable explainer that prospective customers can read before subscribing.

### Q3 — Cellular age page

EDEAR reports cellular age at three levels per the runtime card schema: overall, per-class, per-cell-type. This is one of EDEAR's strongest competitive differentiators (other consumer methylation products report a single number; EDEAR's IAMAtlas architecture resolves all three levels).

Heath flagged: "We need to probably create a cellular age page too that they can reference, but we can discuss that later so keep these in your mind."

A cellular age page would explain:
- What cellular age means (vs chronological age)
- How EDEAR measures it
- Why three levels (overall vs per-class vs per-cell)
- How lifestyle, stress, illness, recovery patterns affect each level
- Trajectory framing — cellular age is one of the most rewarding signals for subscribers because lifestyle changes show up here within months
- The relationship between cellular age and the disease patterns (when cellular age in a specific class is racing ahead of chronological age, the customer's report flags it; the cellular age page explains what that means)

This page is referenced by the customer's report and by every class page (each class page has a Section 8 reference to the cellular age page when discussing the cellular age dimension of that class). Drafting this page is its own walkthrough task — natural fit after the immune class is fully complete.

---

## Round 1 placement (4 cells originally, now 7 cells drafted, audit items captured)

### Direct cell-anchored research findings

[✓] **VAL-095 UniLIFE 19-cell aTreg breast pre-dx >10yr d=+1.26** — captured on regulatory_T_cells.md research section; framed as "blood samples banked years to over a decade before any cancer diagnosis" with attenuation-toward-diagnosis pattern.

[✓] **VAL-128 Crohn's naive CD8 d=+1.72** — captured on naive_CD8_T_cells.md research section; framed as "active inflammatory bowel disease research" with the T-up / myeloid-down population-fraction-shift pattern. Also referenced on neutrophils.md (T cells expanding while neutrophils depressed), CD4_T_cells.md (CD4 expansion in active IBD), CD8_T_cells.md (CD8 expansion in active IBD), B_cells.md (B cells relatively preserved in advanced gastric vs T cell depletion).

[✓] **VAL-118 prostate Stage 3 monocyte d=+0.77** — captured on monocytes.md research section; framed as "monocyte elevation in Stage 3 cell-fraction readings of prostate cancer patients."

[✓] **VAL-122 bladder broad immune infiltration 6/6 Salas POSITIVE** — captured on monocytes.md, neutrophils.md, B_cells.md (B cell expansion in bladder microenvironment). Cell-specific magnitudes referenced descriptively without VAL IDs in customer prose.

[✓] **VAL-090 glioma EpiDISH NLR shift +16.38% neutrophils** — captured on neutrophils.md research section; framed as "neutrophil shifting upward by approximately 16% of total cell fraction" with concurrent lymphocyte depression. Cross-reference to cortical neurons cell page in terminal class for the brain-cfDNA shedding side of the glioma signature. Also captured at integrated level on immune class page Pattern F.

[✓] **VAL-096 breast pre-dx immune-tile attenuation/inversion near dx** — captured on monocytes.md (attenuation pattern in long pre-dx breast cancer), neutrophils.md (attenuation pattern in long pre-dx breast cancer). Also captured at integrated level on immune class page Pattern D.

[✓] **VAL-047 P9/P12 breast pre-dx >10yr d=+1.78 / +1.36 attenuation** — captured at the cell level on regulatory_T_cells.md, monocytes.md, neutrophils.md, and at the integrated multi-cell level on immune class page Pattern D ("Specific cells driving the long pre-dx signal: regulatory T cells UP, eosinophils UP, mild monocyte elevation, mild neutrophil elevation, slightly elevated breast tissue signal in secretory class").

[✓] **VAL-126 advanced gastric T-cell depression + B-cell preservation** — captured on B_cells.md (B cells relatively preserved while T cells deplete in advanced gastric research) and CD8_T_cells.md (CD8 depression in advanced gastric research). Main signature lives on gastric-mucosoid cell page (cycling class) when drafted.

### CCL items captured

[✓] **CCL-007 long pre-dx > near-diagnosis** — captured at the cell level on regulatory_T_cells.md, monocytes.md, neutrophils.md, and at the integrated level on immune class page Pattern D.

[✓] **CCL-019 compartment-direction-flip** — captured at the integrated level on immune class page Pattern E (CRC peripheral negative vs tumor positive). Cell-page-level partial coverage on monocytes.md and neutrophils.md.

[✓] **Bidirectional / population-fraction-shift (Pattern 7)** — captured at the integrated level on immune class page Pattern B with the at-a-glance bullet list of which cells go which direction. Cell-page-level coverage on naive_CD8_T_cells.md, neutrophils.md, CD4_T_cells.md, CD8_T_cells.md.

[✓] **NLR shift (cookbook bidirectional cancellation)** — captured at the cell level on neutrophils.md FAQ and at the integrated level on immune class page Pattern F (glioma) and within Pattern D (advanced cancer signatures).

[✓] **AD-instance pattern (CCL-031 bidirectional cancellation, NOT same as IBD)** — captured at the integrated level on immune class page Pattern A with explicit description of "scrambled" pattern and AD-specific framing. Cell-page-level brief mention on CD4_T_cells.md and CD8_T_cells.md.

[✓] **Smoking persistence / AHRR cg05575921** — captured on monocytes.md, neutrophils.md, and at the immune class page Section 5 (lifestyle factors).

### Per-card lessons captured

[✓] **panc-LL-007 H_min(immune) = 0.838889** — implicit in the framework; not customer-facing detail. Engine-level info lives in card and spec.

[✓] **heme-LL-003 SUPPRESSED tier framework-wide** — captured on every cell page's vigilance section as the SUPPRESSED tier framing.

[ ] **glioma-LL-001 cell-fraction prior orthogonal not inverted** — captured on immune class page Pattern F at customer-readable level; full clinical framing belongs on cortical neurons cell page in terminal class.

[ ] **cerv-LL-002 HPV is STAGE 1 STRATIFIER** — belongs on cervical-epithelial cell page in cycling class. NOT immune-cell-page material.

### Doctrine items captured

[✓] **Run-everything doctrine** — captured on the immune class page Section 8 (the Astro-Genetics story).

[✓] **Three measurement lenses (architecture, tissue-of-origin, cell-fraction)** — captured on immune class page Section 8 and Section 3.5, and on each cell page's "How EDEAR reads them" subsection.

[✓] **Pathway 4 (unexplained immune drift, trajectory watch)** — captured on immune class page Section 4 Pattern I and on each cell page's lifestyle interactions and trajectory pattern subsections.

[✓] **Pathway 1 (terminal-class hidden by specimen)** — captured on immune class page Section 4 Pattern H.

[✓] **Pathway 2 (hematologic compartment)** — captured on immune class page Section 4 Pattern C.

[✓] **Pathway 3 (cardiovascular)** — captured on immune class page Section 4 Pattern G.

---

## Items deferred to other class pages

[ELSEWHERE] **VAL-064 HCC no-documented-risk subgroup d=+0.6166** — belongs on hepatocyte cell page (secretory class). Heath's stepbrother Marcus signal.

[ELSEWHERE] **VAL-066-068 PDAC pooled-null + 324-CpG directional pass** — belongs on pancreatic exocrine cell page (secretory class). Customer-readable framing already partially captured on immune class page Pattern A (AD example) and Pattern D (cancer pre-dx examples).

[ELSEWHERE] **VAL-069 PDAC directional 324-CpG d=+1.51** — belongs on pancreatic exocrine cell page (secretory class).

[ELSEWHERE] **VAL-072-077 cervical TCGA-CESC + Verlaat + LBC** — belongs on cervical epithelial cell page (cycling class).

[ELSEWHERE] **VAL-088-091 glioma blood + GBM tissue + cortical neuron + AD differential** — captured at customer-readable integrated level on immune class page Pattern F. Cell-fraction NLR shift captured on neutrophils.md. Brain-cfDNA shedding side of the story (cortical neurons) belongs on cortical neurons cell page (terminal class).

[ELSEWHERE] **VAL-098-100 CRC subsites** — belongs on colon epithelial cell page (cycling class). Customer-readable framing partially captured on immune class page Pattern E (CRC compartment-flip).

[ELSEWHERE] **VAL-107-112 cardio atlas calibration + multi-atlas convergence** — captured at customer-readable level on immune class page Pattern G. Cell-page-level mention on monocytes.md (cardiovascular monocyte involvement). Specific findings (PAH, aortic disease, stroke) belong on stromal class pages and terminal class cardiomyocytes page when those classes are drafted.

[ELSEWHERE] **VAL-117-122 prostate + bladder reproduction findings** — captured on monocytes.md (prostate, bladder) and neutrophils.md (bladder), B_cells.md (bladder). Full prostate/bladder integrated stories live on prostate-epithelial cell page (secretory class) and bladder-urothelial cell page (cycling class).

[ELSEWHERE] **VAL-126 gastric T-cell depletion + B-cell preserved** — partial reference on B_cells.md and CD8_T_cells.md. Main signature lives on gastric-mucosoid cell page (cycling class) when drafted.

[ELSEWHERE] **VAL-127 esophageal EAC>ESCC d=-1.06 Caggiano** — belongs on esophageal cell page (cycling class).

[ELSEWHERE] **VAL-006 aging trajectory 1075 yr to A=1.05** — belongs on the cellular age page (a separate page, see Q3 above) when drafted.

[ELSEWHERE] **VAL-007 tissue-specific cfDNA Moss 9/9** — captured on immune class page Section 8 (three-lens framing); also belongs on Section 8 of all class pages when drafted.

[ELSEWHERE] **VAL-013/034/035 cross-species H_min invariance** — captured on immune class page Section 8 (Astro-Genetics).

[ELSEWHERE] **VAL-037 TCGA STN field-effect 24 types** — captured on immune class page Section 8 (field-effect concept).

[ELSEWHERE] **VAL-082 AML d=+3.71** — captured at integrated level on immune class page Pattern C (the strongest single-cohort effect anywhere). Brief mention on CD4_T_cells.md, CD8_T_cells.md, B_cells.md, monocytes.md, neutrophils.md.

---

## Items still to capture in remaining 6 immune cell pages

Per cell-type drafting plan, here is what each remaining page must capture:

### Memory T cells page
- General memory vs naive distinction
- Age-related conversion from naive to memory through cumulative life exposures
- Limited disease-specific research (placeholder until research extends — many of the disease patterns are read at naive/total CD8 level rather than memory-specific)
- Pointer to immune class page Section 4 for integrated patterns

### NK cells page
- VAL-122 bladder microenvironment NK expansion
- General innate cytotoxicity biology
- Limited additional disease-specific research (NK cell research is more limited in cookbook beyond bladder)
- Pointer to immune class page Section 4 for integrated patterns

### Eosinophils page
- aEos elevation in long pre-dx breast cancer (UniLIFE 19-cell)
- General allergic response biology
- Strong everyday-context driver: active allergy season is the most common explanation
- Pointer to immune class page Section 4 Pattern D for the integrated breast pre-dx story

### Basophils page
- aBaso reading from UniLIFE 19-cell
- Limited disease-specific research (small population, less commonly resolved)
- Page exists per "every cell gets a page" rule even with sparse research
- Pointer to immune class page Section 4 for integrated patterns

### Dendritic cells page
- Caggiano TIM panel reading
- VAL-110/112 cardio dendritic cell signal (aortic dissection signature)
- General antigen-presentation biology
- Pointer to immune class page Section 4 Pattern G for cardiovascular pattern

### Macrophages page
- Caggiano TIM panel reading
- VAL-110 BAV+dilation macrophage d=+1.61
- General tissue-resident phagocyte biology
- Tumor-associated-macrophage research framing (where research touches it)
- Pointer to immune class page Section 4 Pattern G for cardiovascular pattern

### Megakaryocyte content destination — MOVED TO PROGENITOR WALKTHROUGH

Megakaryocyte is a progenitor-class cell per cell_type_inventory.md (caggiano_celfie_tim label `megakaryocyte`). The progenitor walkthrough's megakaryocyte.md page is the canonical home for all megakaryocyte content. VAL-110/112 cardiovascular research that mentions megakaryocyte signal in aortic disease cohorts is captured on the progenitor megakaryocyte.md page (with cross-reference to the immune class page Pattern G cardiovascular section since the same cohorts also showed peripheral immune signal). Caggiano TIM atlas megakaryocyte tile is documented on the progenitor megakaryocyte.md page; the immune class page does not duplicate.

---

## Coverage status

**Captured in immune cell pages (19 of 73 VALs cell-anchored).** Each of the 19 cell pages carries its own specific findings and points to class page Section 4 for integrated stories. Plus most relevant CCLs at the integrated class-page level. Plus all relevant per-card lessons. Plus all framework-doctrine items.

**Captured at the integrated class-page level (Section 4 Pattern A-I):** AD bidirectional cancellation, IBD population-fraction-shift, hematologic cancer immune-as-tissue, long pre-dx multi-cancer attenuation pattern (breast, lung, CRC, pancreatic, prostate), CRC compartment-flip, glioma's integrated immune+terminal signature with BBB explanation, cardiovascular partial-immune patterns, hidden terminal-class disease (Pathway 1), unexplained drift / trajectory watch (Pathway 4).

**Captured indirectly via deferred placement:** ~25 VALs flagged with their natural homes in other class pages (HCC, PDAC, cervical, CRC subsites, glioma terminal-side, cardiovascular tissue-level findings, prostate/bladder full integrated, gastric, esophageal, aging trajectory, cellular age findings).

**To capture in cell pages:** All 19 immune cell pages drafted. Cardiovascular research findings VAL-110 macrophage aortic and VAL-110/112 dendritic cardio signal are captured on macrophages.md and dendritic_cells.md respectively. Megakaryocyte cardio mention belongs on the progenitor megakaryocyte.md page since megakaryocyte is a progenitor-class cell. NK bladder mention is on NK_cells.md.

**Total expected immune-walkthrough coverage when complete:** Substantially complete capture of immune-relevant cookbook content. The rest belongs on other class pages (~30 VALs) and the cellular age page (~3-5 VALs) which are future walkthroughs.

The architecture is working — every audit item has a clear home, no items are orphaned, customer-facing prose stays anchored to research without bloating with VAL IDs or methodology detail.

---

## Update protocol

This file is updated after each batch of cell-page drafts. Heath reviews each batch; pages get incremented as needed. When the immune class is complete (13 cell pages), this tracker rolls up into a final immune-class audit-coverage report. Other class walkthroughs get their own audit trackers. The cellular age page is its own dedicated walkthrough following the immune-class completion.

---

## Round 1 placement (4 cells drafted, audit items captured)

### Direct cell-anchored research findings

[✓] **VAL-095 UniLIFE 19-cell aTreg breast pre-dx >10yr d=+1.26** — captured on regulatory_T_cells.md research section; framed as "blood samples banked years to over a decade before any cancer diagnosis" with attenuation-toward-diagnosis pattern.

[✓] **VAL-128 Crohn's naive CD8 d=+1.72** — captured on naive_CD8_T_cells.md research section; framed as "active inflammatory bowel disease research" with the T-up / myeloid-down population-fraction-shift pattern. Also referenced on neutrophils.md (T cells expanding while neutrophils depressed).

[✓] **VAL-118 prostate Stage 3 monocyte d=+0.77** — captured on monocytes.md research section; framed as "monocyte elevation in Stage 3 cell-fraction readings of prostate cancer patients."

[✓] **VAL-122 bladder broad immune infiltration 6/6 Salas POSITIVE** — captured on monocytes.md (substantial monocyte expansion in tumor microenvironment) and neutrophils.md (substantial neutrophil expansion). Cell-specific magnitudes referenced descriptively without VAL IDs in customer prose.

[✓] **VAL-090 glioma EpiDISH NLR shift +16.38% neutrophils** — captured on neutrophils.md research section; framed as "neutrophil shifting upward by approximately 16% of total cell fraction (from around 52% to around 68%)" with concurrent lymphocyte depression. Cross-reference to cortical neurons cell page in terminal class for the brain-cfDNA shedding side of the glioma signature.

[✓] **VAL-096 breast pre-dx immune-tile attenuation/inversion near dx** — captured on monocytes.md (attenuation pattern in long pre-dx breast cancer) and neutrophils.md (attenuation pattern in long pre-dx breast cancer). Framed as the counter-intuitive finding underlying why trajectory monitoring matters.

[✓] **VAL-047 P9/P12 breast pre-dx >10yr d=+1.78 / +1.36 attenuation** — captured indirectly on regulatory_T_cells.md (the pre-dx attenuation framing derives from this VAL); explicit reference in monocytes.md and neutrophils.md ("counter-intuitive pattern — strongest signal furthest from diagnosis — is one of the foundational scientific findings behind why trajectory monitoring matters").

### CCL items captured

[✓] **CCL-007 long pre-dx > near-diagnosis** — captured on regulatory_T_cells.md, monocytes.md, neutrophils.md (the attenuation-toward-diagnosis framing).

[✓] **CCL-019 compartment-direction-flip** — partial; the bladder tumor-microenvironment vs plasma-cfDNA distinction is captured on monocytes.md and neutrophils.md ("research on plasma cfDNA in bladder cancer is ongoing"). Full framing of compartment-flip lives elsewhere (CRC tumor-vs-blood, will be on CRC-relevant cell pages).

[✓] **Bidirectional / population-fraction-shift (Pattern 7)** — captured on naive_CD8_T_cells.md and neutrophils.md (the T-up / myeloid-down pattern in IBD).

[✓] **NLR shift (cookbook bidirectional cancellation)** — captured explicitly on neutrophils.md as the NLR shift framing with FAQ explaining it.

[✓] **Smoking persistence / AHRR cg05575921** — partially captured on monocytes.md (smoking elevates monocyte activity through persistent low-grade tissue damage; methylation half-life ~10 years post-cessation in research) and neutrophils.md (smoking persistent neutrophil signal). Cell-class-level coverage of AHRR on the immune class page Section 5.

### Per-card lessons captured

[✓] **panc-LL-007 H_min(immune) = 0.838889** — implicit in the framework; not customer-facing detail. Engine-level info lives in card and spec.

[✓] **heme-LL-003 SUPPRESSED tier framework-wide** — captured on every cell page's vigilance section as the SUPPRESSED tier framing.

[ ] **glioma-LL-001 cell-fraction prior orthogonal not inverted** — partially captured on neutrophils.md; full framing belongs on cortical neurons cell page in terminal class.

[ ] **cerv-LL-002 HPV is STAGE 1 STRATIFIER** — belongs on cervical-epithelial cell page in cycling class. NOT immune-cell-page material.

### Doctrine items captured

[✓] **Run-everything doctrine** — captured on the immune class page Section 8 (the Astro-Genetics story). Cell-page-level mention not necessary.

[✓] **Three measurement lenses (architecture, tissue-of-origin, cell-fraction)** — captured on immune class page Section 8 and on each cell page's "How EDEAR reads them" subsection.

[✓] **Pathway 4 (unexplained immune drift, trajectory watch)** — captured on each cell page's lifestyle interactions and trajectory pattern subsections.

---

## Items deferred to other class pages

[ELSEWHERE] **VAL-064 HCC no-documented-risk subgroup d=+0.6166** — belongs on hepatocyte cell page (secretory class). Heath's stepbrother Marcus signal.

[ELSEWHERE] **VAL-066-068 PDAC pooled-null + 324-CpG directional pass** — belongs on pancreatic exocrine cell page (secretory class).

[ELSEWHERE] **VAL-069 PDAC directional 324-CpG d=+1.51** — belongs on pancreatic exocrine cell page (secretory class).

[ELSEWHERE] **VAL-072-077 cervical TCGA-CESC + Verlaat + LBC** — belongs on cervical epithelial cell page (cycling class).

[ELSEWHERE] **VAL-088-091 glioma blood + GBM tissue + cortical neuron + AD differential** — partially captured on neutrophils.md (NLR shift); main cortical-neuron content belongs on cortical neurons cell page (terminal class).

[ELSEWHERE] **VAL-098-100 CRC subsites** — belongs on colon epithelial cell page (cycling class).

[ELSEWHERE] **VAL-107-112 cardio atlas calibration + multi-atlas convergence** — belongs partly on monocytes.md (already mentioned cardiovascular monocyte involvement), partly on cardiomyocytes cell page (terminal class), partly on stromal cell pages (vascular endothelial, fibroblasts).

[ELSEWHERE] **VAL-117-122 prostate + bladder reproduction findings** — partially captured on monocytes.md. Full prostate/bladder signature lives on prostate-epithelial cell page (secretory class) and bladder-urothelial cell page (cycling class).

[ELSEWHERE] **VAL-126 gastric T-cell + myeloid depletion B-cell preserved** — partial reference on B cells page (when drafted) for B-cell preservation in advanced gastric. Main signature lives on gastric-mucosoid cell page (cycling class).

[ELSEWHERE] **VAL-127 esophageal EAC>ESCC d=-1.06 Caggiano** — belongs on esophageal cell page (cycling class).

[ELSEWHERE] **VAL-006 aging trajectory 1075 yr to A=1.05** — belongs on the immune class page (or aging explainer page) as class-level framing, not on a specific cell page.

[ELSEWHERE] **VAL-007 tissue-specific cfDNA Moss 9/9** — belongs on Section 8 of all class pages as part of three-lens framing. Already partially captured.

[ELSEWHERE] **VAL-013/034/035 cross-species H_min invariance** — captured on Section 8 of immune class page (Astro-Genetics). Not cell-page-specific.

[ELSEWHERE] **VAL-037 TCGA STN field-effect 24 types** — captured on Section 8 of immune class page (field-effect concept). Not cell-page-specific.

[ELSEWHERE] **VAL-082 AML d=+3.71** — heme cancer lives in immune class but the story is whole-immune-compartment AS the disease tissue, not a specific cell signature. Belongs on the immune class page Section 4 (already covered) plus a brief mention on monocytes (myeloid-lineage AML), CD8 (lymphoid-lineage), B cells (B-cell lymphomas).

---

## Items still to capture in remaining 9 immune cell pages

Per cell-type drafting plan, here is what each remaining page must capture:

### CD4 T cells page
- VAL-128 Crohn's CD4 expansion in active IBD
- General CD4 T cell biology
- Cross-reference to regulatory T cells (a CD4 subset) and naive vs memory T cell pages

### CD8 T cells (memory and total) page
- VAL-128 Crohn's CD8 expansion d=+1.70
- VAL-126 advanced gastric CD8 depletion (reference; full content on gastric)
- General CD8 cytotoxic biology

### Memory T cells page
- General memory vs naive distinction
- Age-related conversion from naive to memory
- Limited disease-specific research (placeholder until research extends)

### B cells page
- VAL-122 bladder microenvironment B-cell expansion
- VAL-126 advanced gastric B-cell preservation while T cells deplete (reference)
- General B-cell antibody / memory biology

### NK cells page
- VAL-122 bladder microenvironment NK expansion d=+0.79
- General innate cytotoxicity biology
- Limited additional disease-specific research (NK cell research is more limited in cookbook)

### Eosinophils page
- aEos elevation in long pre-dx breast cancer (UniLIFE 19-cell)
- General allergic response biology

### Basophils page
- aBaso reading from UniLIFE 19-cell
- Limited disease-specific research (small population, less commonly resolved)
- Page exists per "every cell gets a page" rule even with sparse research

### Dendritic cells page
- Caggiano TIM panel reading
- VAL-110/112 cardio dendritic cell signal (aortic dissection signature)
- General antigen-presentation biology

### Macrophages page
- Caggiano TIM panel reading
- VAL-110 BAV+dilation macrophage d=+1.61
- General tissue-resident phagocyte biology
- Tumor-associated-macrophage research framing

### Megakaryocyte content — MOVED TO PROGENITOR WALKTHROUGH

Per cell_type_inventory.md, megakaryocyte is a progenitor-class cell. The progenitor walkthrough's megakaryocyte.md page carries the Caggiano TIM panel reading and VAL-110/112 cardio-related megakaryocyte signal documentation. No standalone immune-class page.

---

## Coverage status

**Captured in immune cell pages:** 19 of 19 immune cell pages drafted, capturing cell-anchored VALs plus most relevant CCLs, all relevant per-card lessons, and all framework-doctrine items.

**Captured indirectly via deferred placement:** ~25 VALs flagged with their natural homes in other class pages.

**To capture in remaining 9 immune cell pages:** ~8-10 additional VALs.

**Total expected immune-page coverage when complete:** ~20-25 of the 73 VALs directly cited or referenced. The rest belong on other class pages and will be captured during HCC walkthrough revisit + secretory class walkthrough + cycling class walkthrough + terminal class walkthrough.

The architecture is working — every audit item has a clear home, no items are orphaned, customer-facing prose stays anchored to research without bloating with VAL IDs or methodology detail.

---

## Update protocol

This file is updated after each batch of cell-page drafts. Heath reviews each batch; pages get incremented as needed. When the immune class is complete (13 cell pages), this tracker rolls up into a final immune-class audit-coverage report. Other class walkthroughs get their own audit trackers.
