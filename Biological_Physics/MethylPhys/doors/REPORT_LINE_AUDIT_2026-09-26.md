# Report interface — line-by-line accuracy audit, 2026-09-26

Every sentence of every tab of the rendered report (healthy blood array GSM…, the rendered blood report (RUN-20260926-14), 17 tabs, 1,252
sentences after tables are collapsed to header + first rows) was read and checked against the thing it describes: the
chain's code, the runtime matrix it names, the bundle it prints from, or the sealed record. Verdicts:

- **OK** — states what the code/file/record says.
- **FIXED** — was wrong or stale; corrected in [`build_methylphys.py`](../chain/MethylPhys_Interface/build_methylphys.py) this session, re-rendered and re-read.
- **REMOVED** — accurate or not, it is a cohort-methodology statement or an internal item; taken off by the author's decision.
- **UNVERIFIED** — could not be checked from this machine; left in place and *named here* rather than passed.

Sentence numbers refer to the audit's scratch sentence file (the extraction that drove the audit).

## Physics (154 sentences)

| # | claim | verdict | evidence |
|---|---|---|---|
| 1–8 | reference tab; audience; formulas optional | OK | prose |
| 9–13 | every cell spends energy to hold order; the margin is the reading | OK | the framework's statement |
| 17 | n = 54,000 / (8.314 × 310.15) = 20.94 | OK | recomputed: 20.942 |
| 19–21 | ΔG_ATP ~54 kJ/mol; R = 8.314 | OK | textbook |
| 23–43 | R·T is the thermal scale; ratio is dimensionless; not metabolism | OK | arithmetic and units |
| 47–55 | three parts; one floor per class | OK | consistent with [`iamatlas_gauge_identity_loci_v1_0.json`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json): 8 H_min values |
| 56–61 | eight sandcastles | OK | analogy, labelled as one |
| 62 | for each class we found addresses where every healthy cell sits at the same level | OK — **extended** | per-CELL identity loci also exist ([`iamatlas_percell_identity_loci_v1_1.json`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_percell_identity_loci_v1_1.json)) and are where the cell is read; sentence now says so |
| 63 | 42,134 immune / 29,181 cycling / 29,617 secretory identity loci | OK | read from the file: immune 42,134; cycling 29,181; secretory 29,617 |
| 64–65 | eight numbers measured once and frozen | OK | H_min values in the file; PROC-HMIN-BOOT-01 confirms unchanged |
| 67–70 | entropy 0→1; computed from array data alone | OK | binary entropy |
| 71–75 | "Take a sample. Go to the immune identity addresses… divide by 0.838889" | **FIXED** | described the pooled class gauge as *the* reading. Rewritten as the order the chain runs: calibrate → deconvolve (presence gate) → per present cell, H(mean β over that cell's identity loci)/H_min[class]. Pooled class A stated as an internal gate, never printed |
| 76–77 | A = 1.00 healthy; 1.10 = 10 % more scrambled | OK — extended | added NORMAL 0.95–1.04, SUPPRESSED < 0.95, Warburg 1.07, breach 1.10 from [`tier_breakpoints.json`](../chain/Runtime%20Matrices/Tier_breakpoints/tier_breakpoints.json) |
| 78 | "The middle 80 % of healthy donors land between ? and ?, and that band is drawn on every gauge" | **FIXED** | two literal `?` placeholders (band dict has no p10/p90 keys), and a cohort band. Removed |
| 79 | one sample can be read on its own — no cohort | OK | now true of the per-cell reading too |
| 81–94 | same floor in a chip and a qubit; Landauer 1961 | OK | framework statement; Landauer's principle correctly dated |
| 85 | "~117 (Apple M1)" | **UNVERIFIED** | number comes from the the semiconductor report issue; not recomputed here |
| 96–100 | eight class levels are measured not derived; 37 reference methylomes; MCMC; bootstrap; frozen | OK | record: G-002 MCMC, 37 cells, R-hat < 1.001; PROC-HMIN-BOOT-01 8/8 inside CI |
| 106 | "483,092 addresses × 115 cell types, each entry a level with an uncertainty" | **FIXED** | 483,092 rows confirmed; 115 cells confirmed; but the atlas is sparse per address (source families on different platforms). Sentence now says so, and that a cell with too few addresses is reported *not resolvable*, never absent |
| 110–111 | "…what the second solver weights by" | **FIXED** | the posterior-weighted solver ([`nilc_celltype_deconvolver.py`](../chain/nilc_celltype_deconvolver.py)) is called by the conductor as a cross-check on composition; it is not a reading. Sentence now says cross-check |
| 113–123 | MCMC and posterior in plain words | OK | correct description |
| 125 | floors fitted by MCMC, R-hat < 1.001 | OK | record |
| 126 | "a separate leave-one-out bootstrap agreed with every one" | **FIXED** | the record is a non-parametric bootstrap (PROC-HMIN-BOOT-01), not leave-one-out. Named correctly |
| 127 | "the sky weights each class panel by how well the atlas pinned that class down" | **FIXED (removed)** | no such weighting in [`stage_4_6_patient_cmb.py`](../chain/stage_4_6_patient_cmb.py) or the conductor; the sky subtracts Σ f_c μ_c and scales by the laboratory residual. Replaced by the one new verified use: the per-cell reference interval drawn from the posterior |
| 128 | second solver uses posterior SD as inverse-variance weight | OK as fact — **relabelled** | true of `nilc_celltype_deconvolver.py`; now stated as a cross-check |
| 130–133 | posterior SD is the precision of an average, never a normal range | OK | correct, and the reason the reference interval is labelled as the atlas's precision |
| 134 | "prints three quantities: reading uncertainty, healthy range, atlas posterior (weighting only)" | **FIXED** | the healthy range is no longer printed anywhere (author: no cohort ranges); the posterior IS now printed, as the interval on the cell's reference. Sentence now names the two intervals and says no healthy range exists on the report |
| 137–139 | M, H_min(c), symbol table | OK | definitions |
| 140–141 | H(β) formula; k_B T ln 2 = 2.968e-21 J | OK | recomputed: 2.968e-21 |
| 143–146 | H concave; marker loci read false disorder; PROC-N7-01; ceiling 1/H_min | OK | PROC-N7-01 in the record (CPG_PHASE1, PROC-TIER-02); ceiling asserted in `_score_one_identity` |
| 148–151 | Sanchez & Mackenzie 2016 filter vs ruler; cited not built upon | OK | as recorded |
| 153 | "Operations Manual (~300 pages …)" | **FIXED** | the OM as built today is 218 pages; the typed count removed |
| 153 | "short methods paper, Landauer Metrology of the Methylome" | OK | `Landauer_Metrology_of_the_Methylome.tex` in the tree |
| 154 | both at the same commit | OK | links carry `R['sha']` |

## The other sixteen tabs — read sentence by sentence from the render of 2026-09-27 (sentences extracted from the rendered report by the audit script into a scratch file - 1,100 sentences)

Verdicts as above. Sentences not listed read as what the code, file or record says (OK). Every FIXED item was re-rendered and re-read.

### Reading (35)
| # | claim | verdict | note |
|---|---|---|---|
| 5-9 | healthy is A = 1.00; NORMAL 0.95-1.04; no population defines any number | OK | matches tier_breakpoints.json and the physics statement |
| 11-13 | per-cell table: cell, class, fraction, A, 95 % interval, tier | **FIXED** (round 2) | reference / departure / offset columns removed; tier against 1.00 |
| 16-21 | footnote: nothing added or subtracted; tier against 1.00 | **FIXED** (round 2) | the muted-offset sentence removed |
| 23 | foreign cells: unspecific, 21 of 21 | OK | matches the bundle's foreign_detection status |
| 30 | laboratory zero on record, not applied | OK | on record -0.0117; TARE-01 decides |
| 31-32 | pooled gauge is an internal gate, not printed | OK | |

### How to read (129)
| # | claim | verdict | note |
|---|---|---|---|
| 9 | "drifted from the healthy reference for its class" | **FIXED** | now "fidelity against the minimum its identity requires - A = H/H_min" |
| 10 | "sits near the reference" | **FIXED** | "sits at 1.00" |
| 20-28 | levels table; no population band | OK | (round 1) |
| 30-32 | pooled gauge rises 0.47 mA/yr; no age term on a cell | OK | 1,379 donors, identity_band_v3 _meta |
| 46-48 | A = 1.00 row of the "three things" table called a "calibration point" checked against 1,379 donors | **FIXED** | now the fixed point; the 1,379 is an observation about people on the Instrument tab |
| 56-69 | 8x5 saturation wall chart, 15 SAT / 2 TGT | OK | reproduced from cpg_gauge_engine.H_MIN_TABLE; author's Issue 002 chart cited as source |
| 71-74 | senescent 1.24-1.27, malignant 1.28-1.32; glioma 1.2846 / GBM 1.256 | UNVERIFIED | typed from the Issue 002 corpus; not re-measured on this chain and says so |
| 76 | Issue 002: healthy reference cells ~0.97 under the raw ratio | OK | quoted from Issue 002 |
| 77 | per-cell atlas profile "typically 0.985-0.995" | **FIXED** | file says 0.936-1.020, median 0.990; now quoted from the file and PROC-UNMIX-01 named |
| 78 | "Earlier versions of this report moved healthy onto 1.00 with a laboratory zero" | **FIXED** | superseded-language removed; states only that no zero is applied |
| 80-92 | Warburg line at 1.07; not re-derived on this chain | OK | tier file carries it as a line with anchor flag; open item stated |
| 117-119 | "the design record contains an ordered series ... history, not a result" | **FIXED** | replaced by the named next test; pre-chain work is education, not something to defend |
| 122 | not a clock: 0.047 over a lifetime | OK | |

### Sky (131)
| # | claim | verdict | note |
|---|---|---|---|
| 3 | z = (beta - sum f mu - m_lab) / s_lab | OK | matches stage_4_6_patient_cmb |
| 4 | "the comparison is to a healthy cohort's statistics" | **FIXED** | z is now stated as departure from the specimen's own composition expectation in units of the laboratory's spread; the panel-measured zero is named as a queued chain change (0g) - a constructed atlas specimen reads 50 % beyond \|z\|=2 against it |
| 20-21 | monopole/dipole twin; nothing subtracted from a cell's A | OK | |
| 41 | "each cell's reference carries the atlas's 95 % interval" | **FIXED** | "atlas profile" |
| 44-45 | PROC-COV-01: no cross-class covariance; residual misfit measured | OK | PROC_COV_01_OUTCOME.md |
| 58-65 | between-person spread 0.029 (IQR 0.018-0.046), quietest 0.0089; detectable-change table | UNVERIFIED | Uppsala panel numbers from the 2026-09-22 working note; not re-measured today |
| 67 | "EPIC-Italy foundation cohort is 460 distinct participants" | **FIXED** | the arrays carry no subject identifier (measured 2026-09-25); the 460 was unverifiable |
| 75 | 196,608 pixels; median span 511 bp, p90 20 kb | UNVERIFIED | from the mapping's build note |
| 92-97 | literature search counts (Europe PMC, arXiv), 2026-09-22 | UNVERIFIED | recorded search; not re-run |
| 103, 112 | healthy skies 2.6-3.2 % beyond \|z\|=2 (PROC-CMB-04) | OK | sealed |
| 108 | this specimen 2.1 % beyond \|z\|=2 | OK | computed on this run |
| 120 | beam-smoothed spread 0.171 vs 0.131 shuffled, 57 sigma | UNVERIFIED | 2026-09-22 measurement, open item |

### Story (101)
| # | claim | verdict | note |
|---|---|---|---|
| 86 | "A = 1.00 is the healthy reference - the reading a healthy cell gives" | **FIXED** | the fixed point |
| 89 | PROC-HMIN-BOOT-01 exists and agrees | OK | record present |
| 90 | "on the Healthy reference tab" | **FIXED** | Instrument tab |
| 94 | "the fixed zero, the single-sample absolute reading, the three-layer reference" | **FIXED** | frozen floors, single-sample absolute reading |
| rest | the author's framing, in his words | OK (author's text) | not audited for physics claims beyond internal consistency |

### Cells (86)
| # | claim | verdict | note |
|---|---|---|---|
| 3-6 | trace-class detection: t~-16 healthy, threshold from 38 donors, 2 % detects / 5 % names | OK | PROC-SMALL-01 |
| legend | column legend | OK | author-approved 2026-09-26 |
| tables | present cells only carry A and tier; families print each member's A | OK | verified on the render |
| identifiability | "not in this platform's solve block" for the 76 unresolvable | **FIXED** (round 2) | was "solved alone" |
| coverage column | "identity loci on this platform" | **FIXED** (round 2) | was read as a finding |

### Instrument (47)
Every number checked against the file it names - pipeline map, panel n per laboratory, 80 per laboratory in reference_data, detection panel, tier lines, G-002 sentence, per-cell reference range. All match. **FIXED:** floor table read the five-substrate tuple (now 8x5 with the methylation column read and four RESERVED); presence floors rendered as a raw dict (now values + PROC-CMB-05 rule); detection panel described as "48 arrays" (now 12 per laboratory on the four commissioning laboratories, 36+ for a new one).

### Coverage (39)
| # | claim | verdict | note |
|---|---|---|---|
| 4-5 | each combination needs a pipeline map, presence floors, detection panel; healthy needs no layer | OK | (round 1) |
| 12 | methylation AUC 0.8663 | UNVERIFIED | engine weight from source literature; stated as not a result of this chain |
| 29 | "36 or more arrays for a line" | OK | commission_detection_lab.py refuses below 36 |
| 33 | nothing about healthy transfers because nothing about healthy is measured | OK | |

### Red flags (28) - OK. Three flags on this run (INTAKE_SKIPPED, NO_INTEGRITY_HASH, FOREIGN_UNSPECIFIC), each with a code and an action; the two about quantities not printed are filtered (round 1).

### Safeguards (45)
| # | claim | verdict | note |
|---|---|---|---|
| 13-14 | release check last run 2026-09-22 at 3b38e87; "re-run the release check" | OK, and acted on | the tab says so honestly; release_check.py re-run in this push |
| 23-24 | NILC vs legacy class L1 0.0628 AGREE | OK | computed on this run |
| 31 | NILC switched off July 2026, PROC-NILC-01, PROC-SEP-03 | OK | provenance of the check; procedures in the record |
| 43-44 | DETECT: limit 0.5-1 % (MF-02/03); full filter tied NNLS (MF-01) | OK | sealed |

### Troubleshooting (47)
| # | claim | verdict | note |
|---|---|---|---|
| 17-24 | 731 healthy arrays; bisulfite gate PROVISIONAL (refuses 731/731) | OK | PROC-STAGE0-02 |
| 26, 40 | commission on 36+ arrays | OK | |
| 33 | 0.737 on the reference scale vs 0.815 through Stage 1 | OK | LESSON-SCALE-01 in the SOP |

### Integrity (25) - OK after round 2 (refusals name only printed quantities; constants labelled gate-and-sky only).

### Chain (102)
| # | claim | verdict | note |
|---|---|---|---|
| 37 | "reads 1.0 on its own reference" | **FIXED** | 1.00 |
| 57, 74, 82 | rows 5 / 5d / 6d RUNS, NOT SHOWN | OK, interim | removed from run_full in the deferred chain patch after TARE-01 |
| 61-63 | row 7 Tiers "goes in: A'', reportable, class H_min" | **FIXED** | a present cell's A and its class H_min |
| 69-72 | row B headed "The reported gauge" | **FIXED** | "The class gauge - INTERNAL GATE" |
| 78-81 | row 6 age reference "subtracted from A before the reading is placed" | **FIXED** | gate only; no age term on a cell |
| 33 | NILC "VINDICATED, not yet re-wired" | **FIXED** | it runs (stage_2b in the live path) |
| 98 | stage_2c_trace_detection.py "called by nothing" | **FIXED (checker)** | run_full calls it through _trace_detect; the derivation now follows helpers called from run_full |

### Files (41) - OK. Generated; 0 undescribed. The Superseded group's four files are each still imported by a live module (disease_matching.py, switching_order.py, build_chain_inventory.py) and stay.

### Findings (23)
| # | claim | verdict | note |
|---|---|---|---|
| 15-20 | VAL-DRYRUN, 43 entries departed, "magnitude (healthy spreads)" | **REMOVED** | a marker-surface dry run from commit 77adaa9d - not a run of this chain; record moved to RETIRED_2026-09 |

### Record (17)
| # | claim | verdict | note |
|---|---|---|---|
| 8 | G-002 status "CONVERGED - values proprietary" | **FIXED (note)** | the sealed index row is not edited; a note beside the table says the floors are public at 10.5281/zenodo.22905819 |

### Run (78)
| # | claim | verdict | note |
|---|---|---|---|
| 4 | "the same atlas, the same band and the same age curve" | **FIXED** | atlas, floors, identity loci, pipeline map |
| 29 | test_gauge_switch docstring "REPORTS the identity-loci gauge with the three-layer reference" | **FIXED** | docstring in the kit test |
| 45-48 | any laboratory's arrays read with no panel; commissioning is the instrument around the reading | OK | |
| 50-55 | 14 files read, all match the tree | OK | negative control passed 2026-09-26 |
| 78 | "the [Issue 003](../manual/MethylPhys_CPG_Operations_Manual.pdf) manual" | **FIXED** | the Operations Manual |

**Totals across the sixteen:** 1,100 sentences read; 26 passages fixed this round on top of rounds 1-2; 1 record removed; 7 items left UNVERIFIED and named above (all are 2026-09-22 working-note measurements or literature-search counts, none of them a reading of this specimen). Nothing on any tab now reads a cell against a population.

**Physics summary:** 154 sentences; 9 passages fixed (the reading's order and per-cell nature, two `?` placeholders, a
cohort band, a claim about the sky the code does not make, a mislabelled bootstrap, a stale "three quantities", a stale
page count, an over-strong "each entry" about the atlas); 1 unverified (the M1 figure); the rest OK.
