# OM migration check — Issue 003 Operations Manual vs the full SOP v3

**Sources read** (`Biological_Physics/MethylPhys/manual/`, `main` at `d5873bd`): the rendered `MethylPhys_CPG_Operations_Manual.pdf` (Edition 003, 234 pages, built from `build_operations_manual.py`, `om_data.py`, `om_part3.py`, `intro_blocks.json`, `appendix_vi_vii.json`, `report_tabs.json`, `switching_order.py`, `val_index.json`, `toc_pages.json`), `PART_II_CHAPTER_NOTES.md`, `OM_v3_neutrophil_chain.md`.
**How:** text of all 234 pages extracted; every page searched for intake, calibration, custody, install and operating terms; the pages that carry chain-v3-relevant content were read in full (pp. 1–2, 10–11, 112–114, 127–130, 134, 148–150, 156, 233). Sections whose heading names a retired method (class floors, class cards, atlas classes, five substrates, sky/CMB, tiers, laboratory zero, age band) were classed outdated from the heading and a keyword scan, not read line by line.
**Test for "still true":** the statement describes what the current code does, or a rule/record that still holds for chain v3.

## 1. Still true for chain v3 and not yet in `MethylPhys_CPG_SOP_v3_full.md` (16 items)

| # | OM location | item (as it applies to v3) | where it should go in the SOP |
|---|---|---|---|
| 1 | p. 1 "What this manual is not"; p. 11 "Not claimed" | Not clinical validation; nothing here should inform patient care; any clinical use requires prospective validation, regulatory review and qualified clinical oversight. The instrument claims no detection of any condition and no clinical readiness. | header |
| 2 | p. 11 "What this document may say about detection" | A statement that the chain can or cannot detect something is made only from a procedure that ran the chain on that question; otherwise write "not yet tested", never "cannot". Recording a measured defect is a measurement and stays. | §8 preamble |
| 3 | p. 11 "What the report says" | The report names no condition and gives no age in years; it states where the cell sits on its gauge. (True of `report_v3.py`.) | §6 |
| 4 | §3b.5 intro (p. 129) | Meaning of the non-answers: QUARANTINE = Stage 0 refused the specimen, nothing scored, exit code 2; DEFERRED = a check could not be made, never a pass; PROVISIONAL = a threshold exists but was never measured on healthy specimens, value printed, not refused (exactly one: bisulfite conversion). | §5 Stage 0 / §7 |
| 5 | §3b.2 (pp. 127–128) | How the hand-off builds its numbers (`stage_0_1_qc_handoff.py`): Type II probe read at address A in both channels, Type I at addresses A and B in its own colour channel (naive sum gave detected fraction 0.979, design-aware 0.9998 on the same array); bisulfite conversion per matched control pair C/(C+U) in green, median over the six pairs; background = the array's own 613 NEGATIVE control probes (median/MAD). | §5 Stage 0, hand-off row |
| 6 | §3b.3 (p. 128) | Intake numbers measured on 732 healthy whole-blood arrays (PROC-STAGE0-02): detection median 0.9994 (lowest 5 % 0.9990, worst 0.9951); call rate 0.9983 / 0.9961 / 0.9889; bead 0.9988 / 0.9969 / 0.9900 (9 of 731 warn); bisulfite 0.7979 / 0.7400 / 0.6354 (731 of 731 below 0.95). Use: a value far from these is the array; a value at 0 or 1 is the configuration. Caveat: measured with the hand-off statistics, which the post-Stage-1 re-check overwrites (SOP_AUDIT A4). | §7 (reference values for the operator) |
| 7 | §3b.4 (p. 128) | Stage 0 run over all 732 pairs after wiring: 720 PROCEED, 8 PROCEED_WITH_PENALTY (bead borderline), 3 QUARANTINE (no published age), 1 decode error (file truncated inside its gzip stream while passing the 1 MB floor). Four defect patterns found and closed, each with a negative control: a gate that cannot open its input never fires (gzipped headers); a value lost between steps looks like a wrong value; a refusal that does not stop the run is reported downstream as something else; a failure logged as deferred is worse than no check. | §8 (Stage 0 development record) |
| 8 | §3b.5 sex row (p. 129) | Sex mismatch: check the paperwork first; the array's call agreed with the depositors' label on 729 of 731 arrays (the other 2 had no label). | §7 sex row |
| 9 | §3b.6 (p. 129) | Low detection / call rate with a correct configuration: check scanner signal; non-polymorphic controls over negatives should be about 25–30, not 10. Stage 1 records this as `signal_to_background_G` / `_R` in `intake.controls` (`stage_1_idat_calibration.py:161-162`). | §7 detection row |
| 10 | p. 127 (§3b intro); p. 10; §3b.6 | The low-signal laboratory example: GSE125105 (1 in 8 probes at background) is refused at intake; per-laboratory intake outcome on 12 arrays each (Uppsala 10/2/0, Karolinska 5/6/1, UCLA 0/7/5, Munich 0/0/12 PROCEED/PENALTY/QUARANTINE) — measured at the earlier 0.95 line; the 0.93 line in `intake_thresholds_v1.json` (`_meta.provenance`) rests on the same 48 arrays. | §8 / §3 provenance of 0.93 |
| 11 | §3b.7 (p. 130) | First run fetches the Illumina manifest once; after that calibration takes about 26 s per array (`run_sample.py` prints "about 25 s"). | §2 |
| 12 | §3b.7 (p. 130) | A batch script that dies with a process-pool error: some environments forbid process pools; use threads. | §4.2 |
| 13 | §9 Operating rules, THE TEST (p. 149) | A number goes on a reading only if it could exist had nobody else's sample ever been measured (physics or calibration: H, floor, A); a band, percentile, age curve or range of people never does. (v3: the purified-cell floor and profiles and the same-run references are calibration of this run.) | §1 or a short "rules" block |
| 14 | §9 SURF-RC4 (p. 149) | A runtime file the code does not resolve on its search path is not part of the instrument, whatever a document says. | §3 preamble |
| 15 | §9 FIREWALL (p. 150) | The chain subtracts no age, sex or smoking term from a reading. (v3 subtracts nothing.) | rules block |
| 16 | §9 PL-002 (p. 150) | Changes made after data are seen are labelled as such in the record ("after looking"). | §8 preamble |

**Cannot confirm (not counted):** p. 134 PROC-CAL-01 and p. 233 glossary — "Stage 1 bit-identical to the project cache on 11/11". Measured before Stage 1 began removing probes at poobah p > 0.05 (2026-09-27); not re-run on the current Stage 1.

**Already in the SOP (no action):** Stage 0 step list and thresholds (§3b.1, values corrected to the code); refusal actions in §3b.5 (manifest fields, hashed id, missing channel, truncated upload, array-type mismatch → trust the header, corrupt IDAT, re-transmission → another `--intake-log` for a planned re-run, borderline/bead warnings); `HOME` writable cache and manifest download (§3b.7); run `run_sample.py` from its own directory; methylprep 1.7.1 needs pandas < 2 (p. 134, p. 233, `PART_II_CHAPTER_NOTES.md` Stage 1); `--patient-id`, `--intake-log`, `--manifest-dir` (III.11); hand-off module purpose (III.10); all of `OM_v3_neutrophil_chain.md` that is still current.

## 2. Outdated — by heading only

Front matter
- Contents page (p. 2)
- The chain as it stands (p. 10)
- What this document claims, and what it does not — "When we seal", "What the instrument reports", "What this field is called", "Claimed" (p. 11)
- Prior art — the door into the conversation; What we agree on; What was missing; Thermal noise is the ruler (p. 13)
- Procedures sealed after the first printing (p. 14)

Part I
- s1 Reconciliation: Issue 002 to repository HEAD, incl. Closed in code; 1.5 Rulings on the two open decisions; What the cosmology tools found that cohorts could not; PROC-PLASMA-MIX; The reporting rule (pp. 15–25)
- s2 The IAM Atlas: 2.1 Cell types by class; 2.2 Identity loci (the gauge panel); 2.3 Discriminative markers; 2.4 The deconvolver (pp. 26–27)
- s3 Two instruments and the presence rule: 3.1–3.4; Laboratory zero; The test that commissioned it; The single-array case; What the layers do not remove (pp. 28–32)
- s4 Five-substrate framework — formulas and transformations; Combined A-score formula; Body temperature scaling & vertebrate lifespan; Clinical horizon; Clinical interpretation of saturated substrates (pp. 33–41)
- The eight architecture-class cards #1–#8 (each: Cell identity & clinical context; Five-substrate fidelity gauge; Substrate-by-substrate breakdown; Best testing method ranking; Healthy aging trajectory; Vertebrate lifespan context; Core metrics; Dated predictions) (pp. 42–101)
- s5 The physics — Landauer, the Mahaffey number, the reference, the gauge (pp. 123–126)
- s5A Where the tools come from (pp. 115–122)

Part III
- III.1 The chain, stage by stage
- III.2 The atlas, and the cells it can speak about
- III.3 The cosmology toolkit, tool by tool
- III.4 What the chain refuses, and the cosmology twin of each refusal
- III.5 The engine, exactly — every formula the chain computes
- III.6 The screens, in the order the report prints them
- III.7 The healthy reference — who it is, and what is not in it
- III.8 Coverage — what is lit, and what lighting one cell requires
- III.9 The guards — what each one refuses, and its last result
- III.10 The files of the chain, by role (except the hand-off / Stage 0 / Stage 1 descriptions, covered)
- III.11 Running it on your own sample (except the three custody flags, covered)

Section 3b (Stage 0)
- 3b.1 The ten steps, as the code applies them — detection and call-rate borderline lines
- 3b.6 It ran, but the chain will not place a number — sky, UNMAPPED, age, NOT ASSESSABLE rows
- 3b.7 It will not start — atlas not found; pyarrow; the four install checks; the healthy-specimen pipeline-map check

Record and appendices
- s7 Substrate characterisation: 7.1–7.4 (pp. 131–133)
- s8 Procedures (pp. 134–148), incl. PROC-QC-01 specification
- s9 Operating rules — CLASS, SURF-RC1, SURF-RC2, SURF-RC3, SURF-CAL, L-5, CCL-019/020, CCL-039, CCL-032/CHK-3.1, LESSON-DECONV-01, 2026-06-11, PRESENCE, SUBSTRATE, glioma-LL-002, PIPELINE, SEAL, SURFACE, L-10
- s10 Falsification record (pp. 151–153)
- s11 Engine map — what the running chain contains; The class gauge — internal gate only (pp. 154–155)
- s12 For the clinician — Margin, Floor, Reading, Your sample; The report this chain produces, tab by tab (pp. 156–192)
- Appendix V — validation index (pp. 193–204)
- Appendix VI — the CMB → methylome translation map, scored (pp. 205–214)
- Appendix VII — the completion sprint, scored (p. 215)
- Future goals — what is worth the effort, in order (pp. 216–218)
- Edition record — what changed from Issue 002 (pp. 219–220)
- Data sources — primary citations; Physics & derivation; Biology & substrates; Statistical & validation; Framework & clinical; Clinical & treatment protocols; Research cohorts & biobanks (pp. 221–229)
- Glossary — CMB and chain terms; Glossary — chain links (pp. 230–234)

Other sources
- `PART_II_CHAPTER_NOTES.md` — Three kinds of file (FLOOR / RULER / BAND) and the per-stage chapter notes
- `OM_v3_neutrophil_chain.md` — Reading the report (composition ≥ 50 % row); Faults ("not the dominant cell" row)

## 3. Result
Once the 16 items in §1 are added to the SOP, nothing in the Issue 003 Operations Manual that is true for chain v3 remains only in the OM, and the OM can be retired for the v3 chain (subject to the scope note on how the retired-heading sections were checked).
