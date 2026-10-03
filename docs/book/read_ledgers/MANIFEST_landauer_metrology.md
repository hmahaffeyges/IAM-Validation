# MANIFEST — line-for-line carriage of *Physics of Methylation: Landauer Metrology* into Part 4

Clone: `12d8fb1` (HEAD at the time of cloning). Nothing pushed or committed.

## Files delivered

| file | status | lines |
|---|---|---|
| `docs/book/part4/p4_02_landauer.tex` (ch:landauer) | rewritten in place; every earlier label kept; FB1 applied | 274 |
| `docs/book/part4/p4_15b_fixedorigin.tex` (ch:fixedorigin) | **new chapter** | 274 |
| `docs/book/main.tex` | one line added after `\input{part4/p4_15_identity}`: `\input{part4/p4_15b_fixedorigin}` | — |
| `docs/book/figscripts/fig_p4_landauer_metrology.py` | new; on `_bookstyle.py` | — |
| `docs/book/figures/part4/fig_cell_budget.pdf/.png` | redrawn (adds the holding energy; Landauer units) | — |
| `docs/book/figures/part4/fig_division_floor.pdf/.png` | redrawn with the corrected count (FB1) | — |
| `docs/book/figures/part4/fig_p4_02_operating.pdf/.png` | new (Table 1 of the paper) | — |
| `docs/book/figures/part4/fig_p4_15b_transfer.pdf/.png` | new (Table 3 of the paper, per array) | — |
| `docs/book/figures/part4/fig_p4_15b_lowsignal.pdf/.png` | new (low-signal laboratory) | — |
| `docs/verification/scripts/verify_landauer_metrology.py` + `_output.txt` | new; ALL CHECKS PASS | — |
| `docs/book/bib_landauer_metrology.bib` | CrossRef record of the 10 DOIs checked; no new entries needed (all keys already in iam.bib) | — |

Placement in `main.tex` (Part 4):
```
\input{part4/p4_15_identity}
\input{part4/p4_15b_fixedorigin}   % new
\input{part4/p4_16a_skytools}
```

## Reading ledger

| source | line count | ranges read | note |
|---|---|---|---|
| `docs/papers/Physics_of_Methylation__Landauer_Metrology.pdf` (pypdfium2 text, `=== PAGE` markers included) | 381 (9 pp) | 1–50, 51–100, 101–150, 151–200, 201–250, 251–300, 301–350, 351–381 — complete | matches `PAPER_LINE_COUNTS.md` row 45 (381); same MD5 as the copy in `Biological_Physics/MethylPhys/papers/` |
| `docs/book/coverage/wave2/45_Physics_of_Methylation_Landauer.tex` | 108 | 1–50, 51–108 — complete | used as the author's words; its FB1–FB4 pointers checked |
| old `part4/p4_02_landauer.tex` | 191 | 1–50, 51–100, 101–150, 151–191 | content kept (corrected) |
| `part4/p4_12_instrument.tex` | 152 | 1–152 | to avoid duplication |
| `PAPER_ERRATA.md` rows LM1, FB1–FB4, TR9, L16 | — | read | LM1 is the only row naming this paper |
| `app_B2_errata_cells.tex` | 27 | 1–27 | |
| `CELL_ITEMS_VERDICTS.md` | 42 | 1–42 | |
| outcome records | — | FINDING_GSE125105_LOW_SIGNAL (39, full), LABZERO01 (22, full), LABZERO02 (33, full), PHASE1 (79, full), PROC_HMIN_BOOT_01 (23, full), PROC_E2E_01 (72, full), PROC_DECONV_V2_01 (34, full), PROC_INTAKE_01 (38, full), PROC_SKY_01 (47, full), ATLAS_READABILITY (1–60), RUNBOOK §9–11 | |
| data | — | `kit/results/PROC_TARE_01_per_array.parquet` (768 rows), `kit/results/FINDING_GSE125105_controls.csv` (12 rows) | recomputed |

The paper has no LaTeX under `docs/papers/latex/`; `Biological_Physics/MethylPhys/papers/Landauer_Metrology_of_the_Methylome.tex` (191 lines) is the author's
source; the four numbered equations were retyped from the PDF and checked against it.

## Item table (every equation, table, figure and quantitative claim)

Paper location = PDF text line numbers. Book location = file:line in this delivery.

| paper lines | content | book location | verdict | correction source / note | status label |
|---|---|---|---|---|---|
| 2-9 | Title, subtitle, author line, date | missing (excluded) | EXCLUDED | stand-alone rule (no author/paper names) | - |
| 11-15 | Abstract: Sanchez-Mackenzie established CDM obeys Landauer; Weibull/generalised-gamma background | part4/p4_02_landauer.tex:202 | CARRIED | - | observed |
| 15-18 | Abstract: divergence from control pool, 'same origin of coordinates'; ask if origin fixed by physics | part4/p4_15b_fixedorigin.tex:5 | CARRIED | - | - |
| 18-21 | Abstract: per-cell-class H_min, A = H(beta-bar)/H_min, one number per cell type | part4/p4_15b_fixedorigin.tex:58 | CARRIED-CORRECTED | FB4, TR9 (class floor -> per-cell reference; class form kept only as the statistic of this 450K test) | calibrated |
| 21-23 | A = 1.00 healthy; NORMAL [0.95,1.05) is tolerance not range of people | part4/p4_15b_fixedorigin.tex:63 | CARRIED | - | - |
| 23-26 | Abstract: thermal noise as unit; M = 20.9; CMOS ~117; transmon 1 | part4/p4_02_landauer.tex:85 | CARRIED-CORRECTED | LM1 (transmon M = ln2); CMOS replaced by the book's verified chip value (ch:cmos, 9950X M=399-411); Apple M1 '~117' not traceable (Apple publishes no TDP) - author flag | calc |
| 26-36 | Abstract: three labs, one pipeline, affine map transfers; medians 0.99/1.02/0.96; 95% in NORMAL; per-lab constants +-0.03; control-probe prediction 0.004-0.018 | part4/p4_15b_fixedorigin.tex:10 | CARRIED | verified in script s.7, s.9 | measured |
| 36-42 | Abstract: what calibration discipline found (wrong locus set; low-signal lab); population layers recorded and removed | part4/p4_15b_fixedorigin.tex:217 | CARRIED | - | measured |
| 44-46 | No disease/diagnostic claim; code public | part4/p4_15b_fixedorigin.tex:259 | CARRIED | - | - |
| 48-50 | CDM stable, enzymatic; patterning carries identity information | part4/p4_02_landauer.tex:3 | CARRIED | - | - |
| 50-55 | Eq. (1) per-site Shannon entropy H(C_i) | part4/p4_02_landauer.tex:207 | CARRIED (was MISSING) | - | - |
| 56-59 | Information in region R = change of sum H; E_R = I_R kBT ln2 | part4/p4_02_landauer.tex:213 | CARRIED (was MISSING as equation) | - | derived |
| 59-63 | Weibull background; persistence length 39-67 nm; signal-detection theory | part4/p4_02_landauer.tex:217 | CARRIED | - | observed |
| 63-67 | MethylIT: Hellinger divergence, Weibull/gamma fit, classifier cutoff | part4/p4_02_landauer.tex:219 | CARRIED | - | observed |
| 68-73 | Three things taken as established (i)-(iii) | part4/p4_02_landauer.tex:223 | CARRIED | - | - |
| 74-77 | Origin = control centroid re-established per cohort; ask if fixed by physics | part4/p4_02_landauer.tex:228 | CARRIED | wording: 'study' for 'cohort' (author rule) | - |
| 77-82 | Reached independently; read after values frozen; why cited | part4/p4_02_landauer.tex:231 | CARRIED | - | - |
| 84-88 | Eight architecture classes; identity loci; H_min(c) = H of class mean | missing (excluded) | EXCLUDED | FB4, TR9; brief: no class floors (chain v3 reads per-cell references, ch:meta) | - |
| 90-98 | April 2026 MCMC calibration on 37 reference methylomes (32 walkers, 5 chains, 5,000 steps, R-hat<1.001); immune 0.795 -> 0.8389+-0.0012; readings revised by 0.055 | missing (excluded) | EXCLUDED | FB4/TR9 (class floors retired); app_B2 rows 'G-002 deposit README' (8 of 22 Roadmap IDs name another cell; 7 cite whole tissue) and 'G-002 reference values' contradict 'FACS-sorted only'; the 0.8389 value is carried only as the calibrated reference of the 450K test | - |
| 99-105 | Bootstrap cross-check 0.060% mean, 0.095% max, 8/8 in CI; earlier 0.168% covered other substrates | missing (excluded) | EXCLUDED | as above (class floors retired); record PROC_HMIN_BOOT_01_OUTCOME.md confirms the numbers | - |
| 105-108 | No disease sample entered calibration; code and Zenodo deposit | missing (excluded) | EXCLUDED | class calibration retired; author's deposit not cited (stand-alone rule) | - |
| 109-115 | Eq. (2) A_c = H(beta-bar_c)/H_min(c) | part4/p4_15b_fixedorigin.tex:59 | CARRIED-CORRECTED | FB4/TR9: carried as the statistic of the 450K test only; current reading is Met-A (eq:meta) | calibrated |
| 116-121 | A_c = 1 fixed point; per-cell-type reading; unimodal loci; Jensen on bimodal panels | part4/p4_15b_fixedorigin.tex:63 | CARRIED | - | - |
| 122-125 | Clarification 1: ratio not divergence; no locus information | part4/p4_02_landauer.tex:236 | CARRIED | - | - |
| 125-129 | Clarification 2: mixtures; NNLS against 115-cell, 483,092-CpG atlas; presence floor | part4/p4_15b_fixedorigin.tex:47 | CARRIED | also L (Two clarifications) | - |
| 132-138 | Eq. (3) M = E_drive/kBT; kBT as unit not background | part4/p4_02_landauer.tex:33 | CARRIED | - | - |
| 141-150 | Eq. (4) M_cell = 54,000/(8.314x310.15) = 20.9 | part4/p4_02_landauer.tex:29 | CARRIED | 20.94 to four figures (verify s.1) | calc |
| 151-160 | Table 1 (three substrates) | part4/p4_02_landauer.tex:85 | CARRIED-CORRECTED | LM1; CMOS row from ch:cmos; new figure fig_p4_02_operating | calc |
| 153-156 | Table 1 caption: transmon exact; gap and ln2 cancel; cell ~21 quanta above floor | part4/p4_02_landauer.tex:97 | CARRIED-CORRECTED | LM1 (M = ln2; 1 in Landauer units); 'above the floor' restated: 20.94 kBT = 30.2 floors | derived |
| 161-166 | M and H_min different quantities; modest arithmetical claim | part4/p4_02_landauer.tex:109 | CARRIED-CORRECTED | FB4 (per-class H_min -> per-cell-type reference) | calc |
| 166-168 | kBT as unit argued elsewhere; nothing depends on it | part4/p4_02_landauer.tex:113 | CARRIED | pointer to Part 1 instead of the author's document | - |
| 171-175 | Arrays: four labs, GEO series, counts; only control arms; per-sample fetch | part4/p4_15b_fixedorigin.tex:26 | CARRIED | source-study disease names and the age range 14-94 dropped (no cohort/population language) | - |
| 176-178 | Fourth lab reported separately; no transfer result rests on it | part4/p4_15b_fixedorigin.tex:29 | CARRIED | - | - |
| 179-187 | Processing: noob, detection, masking; NNLS; Stage A; 42,024 loci; presence gate 0.85/0.02 | part4/p4_15b_fixedorigin.tex:45 | CARRIED | - | - |
| 188-193 | Scale map beta_S1 = a beta_ref + b, a=1.0127, b=0.0662; frozen; entire calibration | part4/p4_15b_fixedorigin.tex:72 | CARRIED (was MISSING) | fit details from PHASE1_OUTCOME addendum (560 arrays, 32,688 loci, r=0.51, gain form 1.1023) | fitted |
| 194-197 | Intake: control probes, detection, QUARANTINE; wired 27 Sep 2026 | part4/p4_15b_fixedorigin.tex:186 | CARRIED | - | measured |
| 198-211 | Validation history: 119 pre-Atlas VALs (107 run, 12 not), six families, VAL-049, 22 post-Atlas, chain integrity within 1%; 175-row index; none could see the offset | part4/p4_15b_fixedorigin.tex:241 | CARRIED-CORRECTED | 'fifteen-cohort cross-population' -> 'a series in which a frozen site panel and frozen references were transferred' (no cohort language); Zenodo deposit not cited; index path kit/VAL_INDEX.csv | observed |
| 212-216 | Pre-registration; failed bar investigated; fixed, moved past, written down | part4/p4_15b_fixedorigin.tex:237 | CARRIED | - | - |
| 217-223 | Table 2 healthy arrays | part4/p4_15b_fixedorigin.tex:35 | CARRIED (was MISSING) | - | measured |
| 228-231 | Unmapped A 0.76-0.86; +0.066 offset largest term; cancels in difference | part4/p4_15b_fixedorigin.tex:84 | CARRIED-CORRECTED | recomputed: 0.066 is the map intercept; the shift at the identity mean is 0.075 (verify s.6); three-pipeline table from PHASE1 addendum | measured |
| 231-234 | Map transfers; medians 0.992, 1.016, 0.961; 94.6% in NORMAL; property of pipeline | part4/p4_15b_fixedorigin.tex:90 | CARRIED | recomputed from parquet (verify s.7): 94.6% of the 756 arrays of three labs | measured |
| 235-242 | Table 3 three labs on one scale | part4/p4_15b_fixedorigin.tex:100 | CARRIED (was MISSING) | recomputed; new figure fig_p4_15b_transfer | measured |
| 245-246 | Per-lab offset +-0.03, flat across age | part4/p4_15b_fixedorigin.tex:118 | CARRIED-CORRECTED | 'flat across age' dropped (age of donors is population language) | measured |
| 246-251 | 850 control probes; 33 features; ridge LOO; correct sign; 0.004-0.018; UCLA 0.026; fourth 0.052 | part4/p4_15b_fixedorigin.tex:128 | CARRIED (was MISSING) | table from LABZERO01/02 records | measured |
| 252-253 | Normalisation and bisulfite-conversion-II red-channel controls separate labs | part4/p4_15b_fixedorigin.tex:123 | CARRIED | - | measured |
| 253-257 | Sealed as failure at 0.010; 0.052 failure of input; model now the recorded route to absolute reading | part4/p4_15b_fixedorigin.tex:142 | CARRIED-CORRECTED | LABZERO02 reading: control probes carry direction not size (UCLA 0.026 with normal signal); route labelled open problem, not the instrument's; chain v3 tare is ch:instrument | interp/openprob |
| 257-259 | SNP-probe linear tare (65 probes) not the instrument's | part4/p4_15b_fixedorigin.tex:142 | CARRIED | also ch:instrument | measured |
| 262-266 | Fourth lab failed every procedure: band test, control-probe model, PROC-SKY-01 3/12 vs 12,11,10 | part4/p4_15b_fixedorigin.tex:149 | CARRIED | 'band test' -> 'two-laboratory comparison' | measured |
| 266 | Author's rule 'once is a result...' | part4/p4_15b_fixedorigin.tex:151 | CARRIED | attribution to 'the author' removed (stand-alone) | - |
| 267-272 | Controls one-sixth; negative 175 vs 311; 12.5% at background vs 0.9/1.5/5.4; SNP clusters 3-4x wider; 8 months | part4/p4_15b_fixedorigin.tex:158 | CARRIED-CORRECTED | table from FINDING record; ratio recomputed 6.6/6.4 ('one-sixth to one-seventh'); SNP clusters ~4x Uppsala's (4.1, 4.5) rather than 'three to four'; new figure fig_p4_15b_lowsignal | measured |
| 272 | One address in eight not a measurement | part4/p4_15b_fixedorigin.tex:182 | CARRIED | - | measured |
| 272-274 | Gate existed but never handed numbers; recorded deferred and advanced | part4/p4_15b_fixedorigin.tex:187 | CARRIED | - | measured |
| 274-278 | At poobah 0.05: 12/12 refused vs 0/12 Uppsala; UCLA 5/12 refused, 7 borderline, author's open decision | part4/p4_15b_fixedorigin.tex:189 | CARRIED-CORRECTED | PROC_INTAKE_01_OUTCOME: the decision was made - line 0.93 on the 48-array measurement; Munich 12/12 quarantined, Uppsala 100/100 advance, UCLA min 0.932 above the line | measured |
| 278-279 | Consequence: three labs not four | part4/p4_15b_fixedorigin.tex:191 | CARRIED | - | - |
| 282-283 | Two results bear on reliability; neither from group comparison | part4/p4_15b_fixedorigin.tex:195 | CARRIED | - | - |
| 284-292 | End-to-end simulation: every synthetic read large departure; marker-panel union; band same statistic; eleven weeks; identity loci 0.99 | part4/p4_15b_fixedorigin.tex:198 | CARRIED | 24 specimens and MAE<=0.015 added from RUNBOOK s.9; 'eleven weeks' not found in an outcome record - author to confirm | measured |
| 291-292 | A statistic agreeing with its own band measures nothing | part4/p4_15b_fixedorigin.tex:206 | CARRIED | - | - |
| 293-300 | Cross-method: unconstrained method cut; columns nearly parallel (r up to +0.99; condition number 31); constraint chooses split; both checks every release | part4/p4_15b_fixedorigin.tex:209 | CARRIED-CORRECTED | ATLAS_READABILITY.md: r +0.958 / +0.989 over 22,548 sites; condition number of the eight-class design 46 (31 not found in any record; author to confirm which design gave 31) | measured |
| 302-307 | Population layers 20-26 Sep: EP28 per-lab zero on 40-array panel, age curve on 1,379 arrays, per-decade percentile band; 75-84% vs nominal 80% | missing (excluded) | EXCLUDED (numbers); construction named in one sentence | author rule (no population/cohort language or math for cells); the ruling itself removes it from the instrument; record kept in RETIRED_2026-09/cohort_gauge_layers_2026-09-27/ | - |
| 307-314 | Removed 27 Sep by ruling; healthy A = 1 by physics; people never a correction; tier edge from group bound is population defining tolerance | part4/p4_15b_fixedorigin.tex:223 | CARRIED | tier-file example dropped (tiers retired, FB4) | - |
| 314-318 | Record kept; nothing live reads it; two surviving facts | part4/p4_15b_fixedorigin.tex:229 | CARRIED | - | measured |
| 320-324 | Both rest on kBT ln2; one filters, one calibrates | part4/p4_02_landauer.tex:242 | CARRIED | - | - |
| 325-328 | Complementary; their background model selects identity loci; first extension to test | part4/p4_02_landauer.tex:247 | CARRIED | label changed to open problem (a plan, not a prediction) | openprob |
| 328-330 | Transfer discipline applies to divergences; pool inherits lab offset | part4/p4_02_landauer.tex:248 | CARRIED | offset value 0.075 (recomputed) | measured |
| 331-334 | Three limitations: WGBS vs arrays; site vs class; binary language not evaluated | part4/p4_02_landauer.tex:253 | CARRIED | - | - |
| 335-338 | What instrument reports: fraction, A, tier NORMAL/ELEVATED/Warburg 1.07/SIGNIFICANTLY ELEVATED/BREACH 1.10/SUPPRESSED | part4/p4_15b_fixedorigin.tex:250 | CARRIED-CORRECTED | FB4: tiers beyond Normal withdrawn; chain v3 prints Normal / above Normal / below Normal (ch:gauge) | - |
| 339-342 | Residual per address on sphere; spread from atlas posterior and SNP-probe noise | part4/p4_15b_fixedorigin.tex:253 | CARRIED-CORRECTED | PROC-SKY-01 failed the SNP-noise spread; chain v3 sky uses spread from purified reference arrays (ch:sky) | - |
| 342-345 | Not an epigenetic clock; no age; names no condition; one axis measured | part4/p4_15b_fixedorigin.tex:255 | CARRIED | - | openprob |
| 347-354 | What is not claimed: metrology on healthy blood; no disease detection; research stage; clinical use needs validation, regulatory review, oversight | part4/p4_15b_fixedorigin.tex:259 | CARRIED | 'disease cohort' -> 'disease series'; 'tier file' dropped (FB4) | - |
| 356-364 | Reproducibility: repo, GEO, fetch tool, kit/results, doors, build_all.py, vocabulary guard | part4/p4_15b_fixedorigin.tex:264 | CARRIED | URL replaced by repository paths; 'companion SOP' named as operating procedure | - |
| 366-370 | Refs Sanchez 2016, 2019 | part4/p4_02_landauer.tex:218 | CARRIED | DOIs CrossRef-verified | - |
| 371-372 | Ref Landauer 1961 | part4/p4_02_landauer.tex:5 | CARRIED | DOI verified | - |
| 373-374 | Ref CLSI EP28 | missing (excluded) | EXCLUDED | only cited for the removed population construction | - |
| 375-380 | Refs Mahaffey 2026a,b | missing (excluded) | EXCLUDED | stand-alone rule (author's own documents) | - |

### Equations

| paper | book | verdict |
|---|---|---|
| (1) H(C_i) | `eq:sanchezH` | CARRIED |
| (in text) E_R = I_R k_BT ln2 | `eq:sanchezER` | CARRIED |
| (2) A_c = H(β̄_c)/H_min(c) | `eq:p4fo_A` | CARRIED-CORRECTED (statistic of the 450K test; FB4/TR9) |
| (3) M = E_drive/k_BT | Definition (Mahaffey number), ch:landauer | CARRIED |
| (4) M_cell = 20.9 | `eq:M` (20.94) | CARRIED |
| (in text) β_S1 = aβ_ref + b | `eq:p4fo_map` | CARRIED (new) |
| (Table 1 caption) transmon M | `eq:Mtransmon` | CARRIED-CORRECTED (LM1) |

### Tables and figures

| paper | book | verdict |
|---|---|---|
| Table 1 | `tab:p4operating` + `fig:p4operating` | CARRIED-CORRECTED (LM1; chip row from ch:cmos) |
| Table 2 | `tab:p4fo_arrays` | CARRIED |
| Table 3 | `tab:p4fo_transfer` + `fig:p4fo_transfer` | CARRIED (recomputed) |
| (none) | `tab:p4fo_ridge`, `tab:p4fo_lowsignal`, `fig:p4fo_lowsignal` | added from the outcome records the paper cites |
| The paper has no figures | — | — |

## Corrections applied

1. **LM1** (PAPER_ERRATA): transmon M = Δ ln2/Δ = ln 2 = 0.693; 1 in Landauer units. Applied in `tab:p4operating`, `eq:Mtransmon`, summary.
2. **FB1**: methylome floor counts all 28,217,448 CpGs (hg19 CpG index): E_floor = 8.37×10⁻¹⁴ J = 9.3×10⁵ ATP at 54 kJ/mol (1.0×10⁶ at 50). Applied in `eq:efloor`,
   figure `fig:divfloor`, the energy checkbox (the copy-to-floor ratio is now ≈21, not 30) and the summary.
3. **FB4 / TR9**: class H_min and tiers retired. The class calibration (8 classes, MCMC, bootstrap) is not carried; the tier list (ELEVATED, Warburg 1.07,
   SIGNIFICANTLY ELEVATED, BREACH 1.10, SUPPRESSED) is replaced by chain v3's three words; the paper's class statistic is kept only as the statistic of the
   450K transfer test, labelled calibrated.
4. Recomputed: the pipeline offset on the identity sites is 0.075 in β at the identity mean (0.066 is the intercept); the old chapter's "0.066 between two
   pipelines" corrected.
5. Recomputed: Munich signal ratio 6.6/6.4 ("one-sixth to one-seventh"); SNP clusters ≈4× Uppsala's.
6. PROC_INTAKE_01_OUTCOME: the call-rate line decision is made (0.93); the paper's "author's open decision" (UCLA 5/12) replaced by the current state.
7. LAB-ZERO-02: the control-probe model is an open route (direction, not size), not "the recorded route to an absolute reading".
8. PROC-SKY-01 + ch:sky: the sky's spread is from purified reference arrays, not the SNP-probe model.
9. ATLAS_READABILITY: correlation and condition number from the record (see flags).

## Exclusions (for the author)

| paper lines | item | basis |
|---|---|---|
| 84–108 | eight architecture classes, class H_min definition, April MCMC calibration, immune 0.795→0.8389 history, bootstrap cross-check, Zenodo deposit | FB4, TR9 (class floors retired; brief: no class floors); app_B2 G-002 rows contradict "FACS-sorted only" |
| 302–307 | numbers of the population construction (40-array panel, 1,379-array age curve, per-decade band, 75–84 % vs 80 %), CLSI EP28 citation | author rule: no population/cohort language or math for cells; the paper itself records the construction as removed |
| 312–313 | tier-file example (NORMAL upper edge at a group's central-95 % bound) | tiers retired (FB4); the ruling is carried |
| 172–174, 245 | source-study disease names, ages 14–94, "flat across age" | no cohort/population language |
| 26, 158 | Apple M1 "~117" | not carried: Apple publishes no TDP; the book's verified chip value (ch:cmos) used instead — **author flag** |
| 2–9, 375–380 | title block, author, own documents | stand-alone rule |

## Author flags (values not found in an outcome record)

- "eleven weeks" (paper l. 289) is carried as printed; no outcome record states it.
- "condition number 31" (l. 297): the record (ATLAS_READABILITY.md) gives 46 for the eight-class design; the book prints 46. If 31 was a three-column
  blood subset, say so and it can be added.
- Apple M1 ~117 (see exclusions).

## Cross-file blocks for the lead (files I do not own)

1. `part1/p1_01_encoding_surfaces.tex` lines 106–107: "for the $19.6\times10^{6}$ sites a cell holds methylated it is $5.82\times10^{-14}$ J" →
   "for all $28{,}217{,}448$ CpGs of the genome, each a binary decision, it is $8.37\times10^{-14}$ J" (FB1).
2. `appendices/app_E_formulas.tex` lines 192–193 (entry 129): $E_{\rm floor}=N\kB\Tb\ln2=2.822\times10^7\times2.968\times10^{-21}=8.37\times10^{-14}$ J,
   "Landauer floor for one rewrite of all 28,217,448 CpGs" (FB1).
3. `appendices/app_B2_errata_cells.tex` line 9: corrected value uses the old count ($1.96\times10^7$, $7.7\times10^{69}$); FB2 gives
   $1.5\times10^{77}/2.82\times10^7=5.4\times10^{69}$.
4. `docs/book/PAPER_LINE_COUNTS.md` row 45: ledger 1–381 confirmed 2026-10-03, status complete.

## Static checks (both chapter files)

Braces balanced; environments balanced and correctly nested; no label duplicated anywhere in the book; every `\ref`/`\eqref` (62) resolves against the
current tree; every `\cite` (13 keys) is in `iam.bib`; every figure file exists. Floats use the preamble's `[htbp]` override. Fixed after a first pass:
VAL index path (`kit/VAL_INDEX.csv`).

## Citations checked (CrossRef, 2026-10-03)

Sanchez2016 10.1371/journal.pone.0150427; Sanchez2019 10.3390/ijms20215343; Landauer1961 10.1147/rd.53.0183; Bostick2007 10.1126/science.1147939;
Sharif2007 10.1038/nature06397; Hopfield1974 10.1073/pnas.71.10.4135; Genereux2005 10.1073/pnas.0502036102; Loyfer2023 10.1038/s41586-022-05580-6;
Pradhan1999 10.1074/jbc.274.46.33002; Adam2023 10.1093/nar/gkad465 — all resolve, titles match. Nelson2017, Milo2015, Triche2013 (books / no DOI in iam.bib) not checked.

## Verification output

```
1. Landauer cost at body temperature
  [OK ] k_B T ln2 (J): 2.96811e-21 (book 2.968e-21, tol 5e-25)
  [OK ] per mole of bits (kJ/mol): 1.78744 (book 1.787, tol 0.001)
  [OK ] M = dG_ATP/RT: 20.9405 (book 20.94, tol 0.005)
  [OK ] M/ln2 (Landauer units per ATP): 30.2108 (book 30.21, tol 0.01)
2. Sanchez-Mackenzie information and energy (symbolic)
  H(1/2) = 1 bit; H(p) = H(1-p): True
  E_R = I_R k_B T ln2; for I_R = 1 bit at T_body: 2.9681136706905537e-21 J
3. The operating ratio M = E_drive/k_B T for three substrates
  transmon: E_drive = Delta ln2, k_B T_gap = Delta  ->  M = log(2) = 0.6931471805599453 ; in Landauer units M/ln2 = 1
  [OK ] transmon M: 0.693147 (book 0.693, tol 0.0005)
  CMOS 9950X N=2e+10: E_sw=1.977e-18 J, M=E_sw/k_B T_j=411, M/ln2=594
  CMOS 9950X N=2.06e+10: E_sw=1.919e-18 J, M=E_sw/k_B T_j=399, M/ln2=576
4. The floor for one copy of the methylome (errata FB1)
  [OK ] E_floor all CpGs (J): 8.37526e-14 (book 8.37e-14, tol 1e-16)
  E_floor = 8.3753e-14 J = 9.340e+05 ATP at 54 kJ/mol = 1.009e+06 ATP at 50 kJ/mol
  methylated CpGs at 70 %: 1.975e+07; chemical cost >= 1 ATP each -> 1.98e+07 ATP
  chemical cost of the copy / its Landauer floor = 21.1
  per mark: M/ln2 = 30.2; per copy the floor counts all 28,217,448 decisions, chemistry pays the methylated ones: 0.70*M/ln2 = 21.1
  ATP over 24 h at 1e9/s: 8.64e+13; copy cost fraction 2.3e-07
5. Fidelity per write
  ln(1/0.02) = 3.91 k_B T
  ln(1/0.1) = 2.30 k_B T
  Hopfield ln(7) = 1.95 k_B T
  Hopfield ln(21) = 3.04 k_B T
  Hopfield ln(80) = 4.38 k_B T
  [OK ] phi = 3.41/M: 0.162842 (book 0.163, tol 0.0005)
  sigma(phi) = 0.12/M = 0.0057; E_hold in Landauer units 3.41/ln2 = 4.92
6. The pipeline scale map [record PHASE1_OUTCOME.md addendum]: beta_S1 = 1.0127 beta_ref + 0.0662
  H(0.7318) = 0.8389 bits (reference beta of the test; reference entropy 0.8389)
  noob beta-bar 0.815 -> mapped 0.7394; A unmapped 0.824; A mapped 0.987
  offset at beta_ref = 0.737: beta_S1 - beta_ref = 0.0756
7. Three laboratories on one scale (Table 3) from PROC_TARE_01_per_array.parquet, column A_raw = mapped A
  Uppsala     n=732 median 0.992  p10-p90 0.963-1.021  offset +0.000
  Karolinska  n= 12 median 1.016  p10-p90 0.981-1.040  offset +0.024
  UCLA        n= 12 median 0.961  p10-p90 0.934-0.971  offset -0.031
  Munich      n= 12 median 1.004  p10-p90 0.991-1.022  offset +0.013
  [OK ] Uppsala median: 0.991842 (book 0.992, tol 0.0005)
  [OK ] Karolinska median: 1.0156 (book 1.016, tol 0.0005)
  [OK ] UCLA median: 0.961293 (book 0.961, tol 0.0005)
  [OK ] fraction of 756 arrays in [0.95,1.05): 0.945767 (book 0.946, tol 0.0005)
  arrays counted: 756
8. The low-signal laboratory (FINDING_GSE125105_controls.csv; three arrays per laboratory)
           nonpoly_G  nonpoly_R  neg_G  neg_R   bsII_R    hyb_G  poobah_fail%
lab                                                                          
GSE111629     4823.0     7251.5  215.0  280.0  16590.0  20392.0           5.4
GSE125105     1291.0     2268.0  175.0  275.0   7965.5  12494.0          12.5
GSE42861      5959.0    10772.5  165.0  270.0  20159.0  14726.0           1.5
GSE87571      8581.5    14426.5  311.0  370.0  32558.5  27410.0           0.9
  Uppsala / Munich non-polymorphic signal: G 6.65x, R 6.36x  (one-sixth to one-seventh)
  SNP homozygous-cluster SD [record FINDING_GSE125105_LOW_SIGNAL.md]: Munich 0.069/0.072 vs Uppsala 0.017/0.016 -> 4.1x, 4.5x
9. Control-probe ridge, leave-one-laboratory-out [record LABZERO02_OUTCOME.md P4b]
  Karolinska  |error| 0.0041  within 0.010: True
  Uppsala     |error| 0.0180  within 0.010: False
  UCLA        |error| 0.0256  within 0.010: False
  Munich      |error| 0.0517  within 0.010: False
10. Intake line 0.93 on the 48-array measurement [record PROC_INTAKE_01_OUTCOME.md]
  medians (min/max): Uppsala 0.985 (min 0.979), Karolinska 0.975 (min 0.891), UCLA 0.953 (min 0.932), Munich 0.878 (max 0.928)
  Munich 12/12 quarantined (0.764-0.928); Uppsala first 100: 100/100 advance

ALL CHECKS PASS
```
