# IAM's Law and Order — master TODO

> **Status 2026-10-04.** The book now has seven parts. Items below were written against the five-part draft; a ticked box was
> checked against the current files. Open book work before release: author's page-by-page sign-off, release tag and DOI, website
> pull request. Rows 0b, 4.4, 11.x and 17.x are repo, chain and errata-record work, not book text. Cell figures marked `COMMISSIONING-RETURN` and the tare method sentences return after commissioning.

One list for the whole book. Checked item by item against the repository at HEAD `5f5997b` (2026-10-03).
One book: Part 0 front matter, Parts 1–5, appendices, one bibliography (`iam.bib`), one figure folder, one `main.tex`.
Always "Part N"; never "the cellular book" or "the quantum book".

Key: [x] done (evidence) · [~] partly (what remains) · [ ] not started · [!] needs the author's decision (the decision, one line).
Evidence is `file:line` at HEAD, a commit, or the read ledgers in `docs/book/read_ledgers/` (MANIFESTs and `paper_vs_book_2026-10-03.csv`).
The author's decisions are collected in Section 19, one line each; items elsewhere point there.

## 0. The public tree and the build
- [x] 0.1 Clinical, treatment and lifestyle advice off the public repo: manual renderer, tier file wording, How-to and SOP text, CHANGELOG callout,
      7 register entries dropped (e3b7097); Operations Manual rebuilt with 0 directive hits (8ed577e); 39 files moved, not deleted, to
      `Biological_Physics/RETIRED_2026-10/` with `INDEX.md` — 20 Record files, the cell-thermodynamics and vertebrate-lifespan papers (their content goes into Part 4),
      the 8 v2 report pages and `reference.html` (236b146). Deleting the retired copies: 19.1.
- [x] 0.2 One book in one place: `docs/book/` holds `main.tex`, `preamble.tex`, `part0/`–`part5/`, `appendices/`, `figures/`, `figscripts/`, `tables/`,
      `iam.bib` (6ab55e5); earlier draft folders in `RETIRED_drafts_2026-10/` (6ab55e5). Product material off the repo (bc1559e, 3913bce).
- [~] 0.3 Compiles: 0 errors, 0 undefined references and citations (95867f3, 1808034; 791 pages at the lead's last compile). Static check at 5f5997b:
      96 included files, 1,315 labels none duplicated, 874 `\ref` targets all resolve (with `tables/` inputs), 469 cite keys all in `iam.bib`,
      every `\includegraphics` file present, braces and environments balanced. Remains: the Overleaf zip of the folder, compiled as uploaded.
- [x] 0.4 Three tracked files are not input by `main.tex`: `appendices/app_B_errata_physics.tex`, `appendices/app_B2_errata_cells.tex` (errata out of the
      build, c39b251) and `part3/p3_05_walls.tex` (Part 3 uses `p3_05_coherence_optimum`, main.tex:55). They still carry retired content
      (σ_crit at app_B:32, :49; the quantum-processor report rows at app_B:24–27; "five substrates" at app_B2:19). Move them to `RETIRED_drafts_2026-10/` with `git mv`.
- [x] 0.5 `read_ledgers/eg_MANIFEST.md` and `read_ledgers/eg_MANIFEST_entropic_gravity.md` are byte-identical (`cmp`): keep one.

## 0b. Runs that feed Part 4
- [x] DNMT Part B scored: IAM-A 1.65–1.97, Q1 8/8, Q2 8/8 (`doors/PROC_DNMT_01_PARTB_OUTCOME.md`, e3b7097).
- [x] Coho (Le Luyer, conversion-filtered copy error): 39/39 scored; D1 ICC 0.826 (bar 0.9) and D2 |ρ| 0.384 (bar 0.3) not met, groups not described (aff3c47).
- [x] Stool route: `doors/DEV_STOOL_01_OUTCOME.md` (f3214c8). Tumour pairs: `doors/PROC_TUMOUR_01_OUTCOME.md`.
- [~] Neutrophil commissioning: test records `doors/PROC_NEUT_TEST_01_OUTCOME.md`, `PROC_NEUT_TEST_01_T2_OUTCOME.md`; chain v3 is still a development build (part4/p4_19_chain.tex:6).
- [ ] 580-species atlas fish (PRJNA802599): no outcome record in `doors/`.
- [ ] Coho WGBS: no outcome record in `doors/`.
- [ ] Hourly box report while runs go (operations; no book file).

## 1. Part 0 — front matter
- [x] 1.1 Title: IAM's Law and Order: The Actualization of Reality — The Cost of Recording It, and the Price of Maintaining It (`docs/book/README.md:3`).
- [x] 1.2 Preface in the author's voice: operational origin in power-grid operations (part0/p0_preface.tex:40), "a messenger, not an inventor" (p0_preface.tex:21);
      Shoulders of Giants and GRF essay read in full (`read_ledgers/LEDGER_Shoulders_of_Giants.md`, `LEDGER_GRF_Essay.md`; f847b6a, 94feee7).
- [~] 1.3 How to read: every status label defined with its meaning (part0/p0_how_to_read.tex:24–38). Remains: one worked example per label.
- [x] 1.4 Notation and units: `appendices/app_N_notation.tex` (337 lines; `\chapter{Notation and units}` l.3; af81e71).
- [x] 1.5 The giants: `part0/p0_giants.tex` (207 lines; "The messenger, not the inventor" l.7; Landauer, Bennett, Bekenstein, Jacobson l.75–98; 8bdffd4).

## 2. Status labels
- [x] 2.1 One macro set in `preamble.tex`: `\derived \calc \calibrated \fitted \conjecture \prediction \observed \openprob` (l.58–65), `\measured \analogy \record` (l.93–95), `\interp` (l.99).
- [~] 2.2 Every claim labelled: 3,228 status macros across the 96 included files; every chapter of Parts 1–5 carries labels; lowest counts:
      p5_09_open 2, p4_11_translation 5, p0_preface 1 (counted at 5f5997b). The per-chapter label-count table is not in the book
      (p5_09_open.tex:126 counts open-problem labels only). Remains: the table, and a pass on the low-count chapters.
- [x] 2.3 Part 4's MEASURED is its own label: `\measured` (preamble.tex:93), defined at p0_how_to_read.tex:32.

## 3. Part 1 — Introduction to IAM's Law and the Virial Theorem
- [x] 3.1 Part 1 = p1_01 floor breach (opening, d73d3b7), p1_02 IAM's Law (line for line, 3a00e42), p1_03 virial law, p1_04 thermodynamic identity (8bdffd4); main.tex:13–16.
- [x] 3.2 Variational and holographic derivations in Part 1: `sec:law_variational` (part1/p1_02_iams_law.tex:395–406, minisuperspace with lapse),
      holographic saturation condition (p1_02:628).
- [~] 3.3 37-orders table: `tab:virial_domains` (part1/p1_03_virial_law.tex:152). Remains: each row recomputed in one script and the script named in the caption.
- [x] 3.4 Landauer, Bennett, Bekenstein, Jacobson, Zurek background: part0/p0_giants.tex:75–98; Zurek in part2/p2_14_quantum_records.tex (adf5d99).
- [~] 3.5 Virial theorem as one instance of the law in every domain (author ruling 2026-10-02): the 1/r half stated in Part 1 and Part 4, CMOS CV² halves (done);
      whether this is one statement or two: 19.2.

## 4. Part 2 — Informational Actualization of GR: The Cosmological Dynamic
- [x] 4.1 Part 2 assembled from the Quantum Order book ch. 5–14 and the Part 2 drafts (6ab55e5), then rebuilt line for line paper by paper (Section 12); main.tex:19–48.
- [x] 4.2 Chapters written: survey predictions p2_16, lensing dynamics p2_17, three-way clusters p2_18, missing satellites p2_19, w(z) far future p2_20,
      Quantum Darwinism p2_14, two faces of time p5_03, entanglement p2_21, electroweak p2_22, Higgs record p2_22b, measurement problem p5_04
      (32f661f, e6c184e, c8a60ec, adf5d99, 2f32f60, 12d8fb1); GRF essay carried into the framing (94feee7).
- [~] 4.3 Headline numbers re-run from the repo chains: late-time growth and Level 2 (`verify_late_time_level2.py` 31/31, 51208f8); dual-sector chapters
      (`verify_dual_sector_chapters.py`, 69e700a); sector tension and S8 (`verify_sector_tension.py`, `verify_s8_trend.py`, a0bca9d). Remains: the DESI χ² values
- DONE 2026-10-04: DESI/SDSS ShapeFit chi2 now scripted (docs/verification/scripts/verify_shapefit_chi2.py: 4.52/5.14, 6.20/6.96); the book prints the scripted values.
- [ ] 4.4 Level 2 open runs: Run D rerun with the IAM growth inside CAMB (errata P7, P12; `cg_MANIFEST.md` exclusion 1) and a Level 2b background chain
      with Eq. 13 coded as written (erratum P17; `cg_MANIFEST.md` finding 1).
- [x] 4.5 Free-µ0 value: PAPER_ERRATA T16 gives 0.039 ± 0.125; p2_04's table gives medians +0.059/+0.064 at the prior edge. Reconcile and print one
      (`MANIFEST_p2_03_theory.md` §7).
- [ ] 4.6 Theory §13 is carried in full in p2_03 (`sec:th:interp`) and in shorter form in part5/p5_01_interpretation.tex: reconcile (`MANIFEST_p2_03_theory.md` §7).
- [ ] 4.7 Virial left-undone list (`MANIFEST_virial.md` l.332–339): sector census of the 26 probes by the worldline rule; lensing time-delay H0 73.3 ± 1.8
      at 3.4σ above the photon-sector rate (open); LRG1 numbers untraced; DESI per-tracer Ω_m, 10 RSD and 32 cosmic-chronometer points not in the repo
      (panels not redrawn; also `val_MANIFEST.md` l.123–125); quantum Table 1 rows (CSL, graviton emission, semiclassical) need sources.
- [ ] 4.8 Three-way clusters: the four binned "observed" ratios are withheld until each bin is traced (TW3; `st_MANIFEST_clusters_satellites.md` l.71–79).

## 5. Part 3 — The Quantum Order of Informational Actualization
- [x] 5.1 Part 3 assembled from the Quantum Order book ch. 15, 17–21 (6ab55e5); one-gauge and three-instrument chapters moved to Part 5 after the synthesis
      (a7f4e3f; main.tex:97–98); x_qp carried line for line (c75708a, 346 lines).
- [x] 5.2 the quantum-processor report and the semiconductor report Issue 002 read in full, 2,551 and 2,432 lines (dae7ad8); physics and dated predictions only, no product material
      (bc1559e, 23db491, 01f385d).
- [~] 5.3 Encoding-surface notes ("bones", 3,116 lines) read in full by the Part 3 proofreader (2026-10-02 session notes); Part 3 merged with the proofread (c39b251).
      Remains: a read ledger for the notes in `read_ledgers/`.
- [x] 5.4 Platform demo pages: product material, removed from the repo and not carried (bc1559e).
- [x] 5.5 One-gauge figure `figures/part3/one_gauge.pdf` (p3_08_one_gauge.tex:46) has no script (`appendices/app_I_provenance.tex:188`) and plots
      Ryzen 7 1800X (2017) and Athlon 64 (2003) points whose inputs are not sourced in the chapter. Rebuild it from a script with sourced inputs.

## 6. Part 4 — Cellular Physics: Thermodynamics of the Methylome
- [~] 6.1 Physics of Methylation: Landauer Metrology, read 1–381 in full (`MANIFEST_landauer_metrology.md` l.32): p4_02 carried line for line (1b2867f; Sanchez &
      Mackenzie premise, M table, whole-genome floor 8.37×10⁻¹⁴ J). Remains: the fixed-origin chapter `p4_15b_fixedorigin.tex` (pipeline map, four labs on one scale,
      low-signal laboratory) is not in the tree; it was held because the pipeline map is fitted and its Normal-band statistics are group statistics (19.4).
- [~] 6.2 Astro-Genetics in the bridge chapter in the author's words (part4/p4_01_bridge.tex:3–15; d82044a). Remains: the Cellular Thermometer PDF has no read ledger in `read_ledgers/`.
- [!] 6.3 Pasted text of 2026-10-02 09-12 (methylation as Φ_IAM, the 50/50 partition in cells) is not in the book (Φ_IAM occurs only in p5_02_exploratory.tex:45–56);
      it conflicts with 08-51 ("cells aren't 1/r bound"). Proposed: Part 5 section labelled \conjecture with the conflict stated (19.3).
      The other three pasted texts are in (gauge chapter, 79-tool map `part4/16a_skytools_map79.tex`, sky-tools chapter p4_16a).
- [~] 6.4 Report-tab prose on the current method: partly in Part 4 (refusal states in 9 Part 4 files; posterior-SD category error p4_14_atlas; instrument fingerprint
      p4_20_report; ceiling guard p4_18_discipline, 74eefbe). Not found in Part 4 by phrase search: presence gate never a correction, spatial-shuffled null, residual matched filter,
      test a check with known-bad input. Carry them in the author's words.
- [~] 6.5 Reproduction paper (v3, 3,665 lines) read in full (2026-10-02 session notes; no ledger in `read_ledgers/`); per-cell ceiling A_max carried
      (p4_05_floorbreach.tex:63). Not found in Part 4 by phrase search: integrity gates,
      synthetic-mixture recovery test. Carry them.
- [x] 6.6 Salmonid chapter with the honest result: part4/p4_22b_salmonid.tex (main.tex:83; 8bdffd4).
- [~] 6.7 New results as they land: DNMT Part B, tumour pairs, IMR90 channels, coho are named in Part 4 (DNMT in 9 files, IMR90 in 2, coho in 6). Remains: neutrophil
      commissioning when it lands (Section 0b).
- [x] 6.8 Test Validation Compendium (1–856), Official Score Card (1–357) and Technical Reference (1–1,270) read in full; 13 items inserted (`sum_MANIFEST.md` §1, §5; 74eefbe).
      Overview Companion: read by an earlier pass (2026-10-02); its items are in (checked: TRGB 70.39 ± 1.94 at p2_04:35; Reyes 2010 E_G at p2_16:160; bulk flows at p2_16:185;
      no "independent check" left in p2_04). No read ledger for it in `read_ledgers/`.
- [ ] 6.9 `figscripts/fig_p4.py:8` names "Normal (0.95–1.05)" as the design tolerance. Check every Part 4 figure and caption against chain v3's three state words
      so no retired tier word is drawn.

## 7. Part 5 — 37 Orders of Magnitude and the Web that is IAM
- [x] 7.1 Synthesis: part5/p5_08_synthesis.tex "Law and Order: one accounting at five places" (636ddff); one gauge p3_08 and reach p3_09 follow it (main.tex:96–98).
- [x] 7.2 Falsifiable predictions: part5/p5_07_predictions.tex, 519 non-cell register entries triaged, about 40 predictions with value, test, date, falsifier
      (11d6d8f; `CANON/predictions_triage_2026-10-02.json`); cell predictions only after commissioning (897db28). Four early predictions removed from book and repo (09708e5).
- [x] 7.3 Exploratory: part5/p5_02_exploratory.tex, 361 lines, written for fun to show the reach of the law (33ca096; main.tex:101). The two repo PDFs are one paper
      (erratum GE5; the propulsion copy is docs/RETIRED_2026-10/top_level/Gravitational_Propulsion_and_IAM_duplicate.pdf (archived privately), e08384a).
- [x] 7.4 Interpretation chapter: part5/p5_01_interpretation.tex (main.tex:88); black-hole information p5_01b (3a00e42).
- [x] 7.5 Status table merged across all parts: part5/p5_11_status_all.tex (8bdffd4; rows added 74eefbe).
- [x] 7.6 Open problems with the plan for each: part5/p5_09_open.tex "What is open, and the plan to close it" (12 derivations, 14 cell-instrument items, 6 device items; 636ddff).
- [~] 7.7 Future tests: in p5_07 (test and date per prediction) and p5_09 (who settles each). Remains: one table splitting ours from those for others.
- [x] 7.8 Conclusion: part5/p5_10_conclusion.tex (636ddff; main.tex:103).

## 8. Appendices (as built, main.tex:106–115)
- [x] A Constants: `app_A_canon.tex` (generated from CANON, retitled "Constants and names", 06a2ac7) and `app_A2_frozen_values.tex`.
- [x] B Formula sheet: `app_E_formulas.tex` (164 displayed equations with status and location, af81e71).
- [x] C Derivations in full: `app_C3_derivations.tex` (372 lines, 8bdffd4).
- [x] D Errata: kept in `docs/verification/PAPER_ERRATA.md` (391 lines), not printed in the book (c39b251). Pending rows: Section 17.
- [x] E Reproduction: `app_C_reproduce_physics.tex`, `app_C2_reproduce_cells.tex`.
- [x] F Glossary: `app_F_glossary.tex` (579 lines).
- [x] G Predictions register: `app_G_predictions_register.tex` (generator `figscripts/make_app_G.py`).
- [x] H Open items and work plan: Part 5 chapter p5_09_open (not an appendix).
- [~] I Figure and data provenance: `app_I_provenance.tex` (generator `figscripts/make_app_I.py`); 13 figures marked "no script found"
      (app_I l.30, 104, 111, 115, 117, 118, 122, 124, 128, 139, 145, 154, 188). Give each a script or a recorded source.
- [~] Bibliography: one file `iam.bib`, 499 entries (bib_*.bib merged, 8bdffd4, c39b251), 0 self-entries. Remains: the duplicate key `Reyes2010`
      (iam.bib:3349 and :3686, same paper, doi 10.1038/nature08857) and DOI fields (Section 15).

## 9. Figures, tables, visuals
- [x] 9.1 Inventory: 204 PDF figures under `figures/part0`–`part5`, 44 `fig_*.py` scripts plus helpers in `figscripts/` (5f5997b).
- [~] 9.2 Per chapter at least two figures and one table: 40 of 86 Part 0–5 files fall short by float count at 5f5997b (e.g. p2_13_baryon 0/0 — its figures moved to p2_13b;
      p2_22_electroweak 0/0; p5_05c_virial_decoherence 0/0; p5_10_conclusion 0/0). Decide per chapter.
- [x] 9.3 Retired content in figures: no 1.10 breach line, tier, class floor or group band in the figure scripts (grep at 5f5997b; fig_p4.py:8 states none are drawn); see 6.9.
- [~] 9.4 New figures: one gauge across domains (p3_08, item 5.5), qubit and cell gauges (`figures/part3/fig_qubit_gauge.pdf`, `fig_cell_gauge.pdf`), encoding ladder
      (`fig_encoding_ladder.pdf`), virial domains (`fig_virial_domains.pdf`), µ(z) and fσ8, sky maps, C(d), IMR90 channels (66846be). Remains: law-to-domain map and atlas
      coverage (no figure file by those names at 5f5997b).
- [~] 9.5 Every figure rebuildable from a script: see 8.I (13 without a script).

## 10. Derivations audit
- [~] 10.1 Every derivation walked: no "it can be shown" in the book (grep at 5f5997b); verification scripts per chapter
      (e.g. `verify_theory_derivations.py` 45 pass, 0 fail; `verify_late_time_level2.py` 31/31; `verify_entropic_gravity.py`; `verify_xqp_book.py`).
      Remains: 5 DISCREPANCY lines in `verify_theory_derivations_output.txt` (T23–T25 and T5's D2 ratios) and erratum T17's wording ("within 5 %" → constant 7 %, 1/a 2 %).
- [x] 10.2 Incomplete derivations listed: Appendix C (`app_C3_derivations.tex`) and the 12 derivations in p5_09_open.

## 11. Repo
- [~] 11.1 Book folder README `docs/book/README.md` (23 lines) with the build command. Remains: its label list (l.11) lacks `\interp` and `\record`; its appendices line (l.18)
      still lists "corrections to the source papers"; status points to this file. Block in the MANIFEST.
- [ ] 11.2 `Biological_Physics/MethylPhys/chain/README.md` describes the v2 stages: per-cell A = H/H_min of its class "and its tier" (l.19), class gauge (l.20), orchestrator
      `cpg_conductor.py run_full` (retired with chain v2 on 2026-10-03, archived privately) (l.24), tier file among the runtime constants (l.28); last changed 3942f8b (2026-09-27), before `conductor_v3.py` (bc4a651, 2026-10-01).
      Rewrite it on chain v3 (Stages A, M, MC, T, IAM-A; median same-run tare, 03e15ad).
- [ ] 11.3 v3 report rebuilt with the tab prose: `chain/MethylPhys_Interface/report_v3.py` last changed a8a7637 (2026-10-01); tab prose not yet in it.
- [ ] 11.4 Everything shown to the author is in the repo (except private material).
- [ ] 11.5 Ledgers out of date: `PAPER_LINE_COUNTS.md` rows 38, 40, 45, 46, 47 say "not confirmed" and `BOOK_READING_TODO.md` rows 38, 40, 45, 46, 47 are unticked,
      though the Compendium, Score Card, Landauer Metrology and the exploratory paper were read in full (`sum_MANIFEST.md` §1; `MANIFEST_landauer_metrology.md` l.32;
      `MANIFEST_exploratory.md` l.25–27). Blocks in the MANIFEST.

## 12. Every paper carried
- [x] 12.1 Per-paper table `read_ledgers/paper_vs_book_2026-10-03.csv`: 28 paper groups; book carries 165,397 words against 159,362 paper words (5f5997b).
- [x] 12.2 Line-for-line rewrites: IAM's Law p1_02, Theory p2_03 (3a00e42); Koide p2_15a, electron mass p2_15b, Higgs record p2_22b (12d8fb1); six virial papers and
      quantum-level gravitational decoherence p1_03, p1_04, p2_02, p2_02b, p5_05, p5_05b, p5_05c (8bdffd4); Bekenstein and two black-hole papers p2_01, p2_01a, p5_01b (3a00e42).
- [x] 12.3 Floor Breach as the Part 1 opening chapter (d73d3b7).
- [~] 12.4 The wave under ratio 0.6: every group now at 0.56 or above (csv). Remains: Missing satellites 0.56 (failed side prediction left out by the 2026-10-03 ruling, c8a60ec, c637c4a);
      Landauer metrology 0.67 (item 6.1); Koide + electron 0.72; electroweak 0.79.
- [~] 12.5 Author's own words in every chapter: each line-for-line chapter carries his sentences, changed only where an erratum or rule requires (MANIFESTs: own-words
      method in `MANIFEST_exploratory.md` l.23; `val_MANIFEST.md` l.86–97; `wz_MANIFEST_darkenergy_surveys.md` l.11–19). Remains: Part 3 and Part 4 chapters not yet re-set from his texts.

## 13. Application chapters (after Section 12)
- [ ] 13.1 Qubit and chip chapters (Part 3) against the quantum-processor report and the semiconductor report Issue 002 read again in full; cell chapters (Part 4) against the cell issues, web.py, the reproduction paper and the
      Hubble-to-cell documents. Part 4 ch. 12–23 already rewritten on chain v3 as coded (2b775ba).
- [x] 13.2 Rule: everything a referee needs to confirm the physics and the readings; floor values, specific application methods and product names stay out.

## 14. Author's cellular items file
- [~] 14.1 Read in full (412 lines); verdicts `docs/book/CELL_ITEMS_VERDICTS.md` (94f8231). Applied: Astro-Genetics opening (p4_01_bridge.tex:3), cell as microchip and
      notebook (d82044a), "cannot be selectively right" (p4_01_bridge.tex:163), operational origin (p0_preface.tex:40), H(ε₀) = 0.2043 (p4_08_iama.tex:32, c39b251).
      Remains: the Φ_IAM section (verdict 4; item 6.3).

## 15. Citations
- [x] 15.1 Every DOI and its metadata checked against CrossRef (arXiv-only against the arXiv API) when merged (21e06bb, c39b251; each MANIFEST lists its checks).
      Anthony2022 now Nat. Commun. 15, 6444 (2024) (iam.bib:33–42); DESI2025 now PRD 112, 083515 (iam.bib:298–302); no Mahaffey entries remain.
- [~] 15.2 Whether each cited paper carries the number attached to it. Checked: Andrade 2024 µ0 − 1 = 0.02 ± 0.19; KiDS-Legacy S8 0.815 (erratum ST7); Cyburt 2016 Table IV
      (has no 6.137; `bl_MANIFEST.md` l.54); Stölzner, Nguyen, DESI 2024 V App. A (`ts_MANIFEST_sector_s8.md` l.26–33); Kurter, De Dominicis, Ristè, Burnett (`xqp_MANIFEST_xqp.md` l.17).
      Still to check: DESI2016 (Year 5 fσ8 precision), Kepler2007 (0.6 M☉ white-dwarf mean), Shemmer2004 (TON 618, p5_01:42), Pradhan1999 (DNMT1 7–21×, p4_02:193),
      Loyfer2023 (3.41 kT copy-error source, p4_14:127).
- [~] 15.3 Unsourced claims: sourced now — Euclid sensitivity (Albuquerque 2025, sec:lt_euclid), DGP ghost (Koyama2007, p1_02:543), Kelvin–Helmholtz (p1_03:89), TOV
      (Tolman1939, OppenheimerVolkoff1939, p2_01:318), m_p/m_e quasar bound (Ubachs2016), 9950X (AMD9950X, p3_08:21); removed — 25-of-54 census (ruling 2026-10-03).
      DESI Year 5: no σ(µ0) forecast value is printed now (p2_05:207, 215, 238 name the survey only). Still unsourced: island of stability (p5_02:244), Bulbul 2024 M_SZ/M_hydro 0.99 ± 0.04 (not printed), Planck SZ S8 ≈ 0.78,
      LSST ~200,000 clusters (`st_MANIFEST_clusters_satellites.md` l.90).
- [x] 15.4 Two S8/DESI papers with a private correspondent among the authors left uncited on purpose (naming rule).
- [x] 15.5 DOI fields missing in iam.bib: Riess2022 10.3847/2041-8213/ac5c5b, Heymans2021 10.1051/0004-6361/202039063, Kurter2022 10.1038/s41534-022-00542-2
      (all three CrossRef-checked 2026-10-03); DESY3, HSCY3, Abazajian2016, DESI2016, Einstein1915 still without a DOI field.
- [x] 15.6 Duplicate bib key Reyes2010 (iam.bib:3349, :3686): keep one.

## 16. Cross-file fixes from the read-ledger MANIFESTs (not yet applied at 5f5997b)
- [x] 16.1 `appendices/app_E_formulas.tex:16` lists ch:sectortension as having no displayed equation; it now has st_entropy, st_hm, st_mu, st_ode, st_sigma8, st_S8 (`ts_MANIFEST_sector_s8.md` l.37).
- [x] 16.2 Look-elsewhere count: the book's own count is 3 of 414 within 1 % (ch:lambda); erratum C17 (PAPER_ERRATA.md:202) still says 2 of 540 (`bl_MANIFEST.md` l.58;
      `app_C3_derivations.tex` no longer does).
- [ ] 16.3 `docs/verification/cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` l.69 says the record's η 6.1155 uses the full chain; the cause is the factor 2.74 vs 2.739 (`bl_MANIFEST.md` l.55).
- [ ] 16.4 `docs/verification/particle/HIGGS_DURATION_CHECK.md` does not exist; write it from errata HD1–HD13 (`MANIFEST_particle.md` l.542–543).
- [x] 16.5 Wording from the old particle chapter (`MANIFEST_particle.md` l.555–556): "at most three" at p5_07:145, p5_09:30, app_G:134 matches p2_15a:216;
      "6.6 ppm" only in app_B:18, which is out of the build (item 0.4).
- [x] 16.6 Unreferenced figure files: `figures/part2/fig_two_rulers_future.{pdf,png}`, `fig_w0wa.{pdf,png}` (no `\includegraphics` of either at 5f5997b;
      `wz_MANIFEST_darkenergy_surveys.md` l.237–238; `ts_MANIFEST_sector_s8.md` l.37): move to the retired folder. (`fig_sat_census` already removed, c8a60ec.)
- [x] 16.7 Applied already (checked at 5f5997b): app_I rows for fig:eta, fig:baryon_posterior, tab:baryon_chains → ch:baryon_chain (app_I l.78, 79, 294); x_qp figure rows (l.105–109);
      exploratory rows fig:twin_exploratory, fig:steering, fig:transit, fig:recession (l.197–200); eq:ms_mmin entry and fig:sat_census row removed (no hits);
      FB1 count in p1_01 (no "19.6" left in Part 1); "61.5, excluded" rewritten (51208f8); p5_03 electroweak time 9×10⁻¹² s (p5_03:41, 125); p2_03 sec:source first law (2385460).

## 17. Errata proposals not yet in `docs/verification/PAPER_ERRATA.md`
Ten sets of proposed rows in the MANIFESTs have no row at 5f5997b (grep of each correct value in PAPER_ERRATA.md). The book already carries each corrected form or leaves the item out (as each MANIFEST records).
Rows proposed in the x_qp, theory, particle and exploratory MANIFESTs are in (XQ9–XQ19, T19–T27, EM5–EM9, KO5–KO8, HD1–HD13, GE6–GE7).
- [ ] 17.1 Virial (11 rows): PRL "37 orders" atom→cluster is 33 decades; T/|V| = 1.0000 → 2T/|V| = 1; "unique ½"; DM/DE "ratio 2"; 6.4 % → 7.3 %; t_dyn √(4π/3); Rovelli Entropy 24, 1022;
      phonon ⟨n⟩ grows; P_IAM Eq. 11 vs Eq. 5; falsification reach at 1e13–1e14 amu; SBF value is Blakeslee 2021 (`MANIFEST_virial.md` l.317–330).
- [ ] 17.2 Cosmological constant and baryon (8 rows + close C10): Cyburt Table IV has no 6.137; CMB 6.136 matches no Planck entry; η factor 2.739; a_EW with g_*s;
      history-integrand form; look-elsewhere 3 of 414; nats vs bits; "about 123 orders"; Ω_mh² 0.1431 (`bl_MANIFEST.md` l.53–61).
- [ ] 17.3 Late-time growth and Level 2 (7 rows; P17 is in): BAO data are 6dF 2011, DR7 MGS, DR12 consensus; 20,000 samples not met; Table 1 z = 3, 5 and Table 6 z = 5 rows;
      HSC Y3 0.769; Level 2 shifts; z = 0.85 pull 1.39σ; Level 1 parameter values (`cg_MANIFEST.md` l.187–198).
- [ ] 17.4 Sector tension and S8 trend (8 rows): SX9 traced (fσ_s8); Fig. 2 used DESI fiducial distances; trend is RSD-based; γCDM differs at all z < 1; citation fixes; DES Y3 area;
      E < 0.1 for z > 2.30; ODE σ8; Level 2 shifts (`ts_MANIFEST_sector_s8.md` l.28–37).
- [ ] 17.5 Dual-sector validation and note (10 rows): Eq. 7 /10 pc vs +25; m_b^corr; Nelder–Mead; Table V β 1.5σ not 3.1σ; Δβ ± 0.015; optimisers; 1,701 light curves of 1,550 SNe;
      Amendola LRR 21, 2 (2018); Einstein–Infeld–Hoffmann 1938; diagonal vs full covariance (`val_MANIFEST.md` l.103–115).
- [ ] 17.6 w(z) far future and survey predictions (10 rows): H_∞ = 0.8275 H0; no-Big-Rip criterion; sirens 2 % (Chen 2018); Planck lensing 40σ; KiDS 1.6σ; w0–wa and Σ; ISW sign convention;
      f(R) ISW magnitude; two-ruler table on 67.16; 1.15 % (`wz_MANIFEST_darkenergy_surveys.md` l.32–49).
- [ ] 17.7 Lensing, three-way clusters and missing satellites (8 rows): t_dyn = (π/6) GM/σ³ (10^8.63 M☉); Ω_m + Ω_Λ = 1.0000; Read 2006 reference does not resolve; Benson 2002 II …05388.x;
      Pizzuti JCAP 07 (2017) 023; CCCP/WtG from Planck 2015 XXIV; turnover at z = 1.40; PS abundance +0.7 % at satellite masses
      (`st_MANIFEST_clusters_satellites.md` l.53–63; `sat_MANIFEST_satellites.md` l.76–82).
- [ ] 17.8 Measurement problem (2 rows): Table 1 dust-grain row (R = 5 µm: E_G 1.3×10⁻²⁹ J, τ_PD 7.9×10⁻⁶ s, τ_IAM 5.3×10¹¹ s); N = 111,612, 10^−33,599
      (`qr_MANIFEST_records_measurement_time.md` l.31–33).
- [ ] 17.9 Entropic gravity (3 rows): Luciano 2025 summary; Table 2 µ0 0.05 ± 0.22 and Σ0 0.008 ± 0.045 not in the cited sources; Eq. 1 sign/temperature (`eg_MANIFEST.md` l.105–109).
- [ ] 17.10 Summary papers (15 rows): CO1–CO4, SC1–SC7, TRa–TRc, BK1 (`sum_MANIFEST.md` l.228–246).
- [ ] 17.11 Rows still pending a trace or test inside PAPER_ERRATA: P11, P14, S5, V15, N2–N4, C10.

## 18. Layout overruns
- [x] 18.1 No compile log is in the repository or the artifact store at 5f5997b, so the overfull-box list is not given here: take it from the next compile
      (`grep -n 'Overfull .hbox' main.log`, boxes over 10 pt, by file and line). Earlier fixes: running heads capped (45c40a4), wide tables wrapped (ddb1cc4, 8bdffd4),
      lone-float pages (41f7646, 95867f3).

## 19. Author decisions (one line each)
- [!] 19.1 Delete the retired copies (`Biological_Physics/RETIRED_2026-09`, `RETIRED_2026-10`, `docs/book/RETIRED_drafts_2026-10`) once you confirm you hold copies?
- [!] 19.2 Virial half in every domain: one statement of the law, or two (the 1/r half and the law)? (item 3.5)
- [!] 19.3 Methylation as Φ_IAM (pasted text 09-12): carry in Part 5 as \conjecture with the conflict with "cells aren't 1/r bound" stated, or leave out? (items 6.3, 14.1)
- [!] 19.4 Fixed-origin chapter p4_15b: carry with the pipeline map labelled \fitted and the group statistics removed, or leave out? (item 6.1)
- [!] 19.5 Landauer metrology: "eleven weeks" (paper l.289) has no outcome record — keep or drop? (`MANIFEST_landauer_metrology.md` l.177)
- [!] 19.6 Landauer metrology: condition number 31 (paper l.297) vs 46 in ATLAS_READABILITY.md — was 31 a three-column blood subset? (l.178–179)
- [!] 19.7 Apple M1 "~117" left out (Apple publishes no TDP; the book uses the verified chip value) — agree? (l.172, 180)
- [!] 19.8 Three-channel β_m split, DM/DE halves, arrow-of-time sections (erratum V12): keep as \conjecture in p5_05b, or drop? (`MANIFEST_virial.md` l.339)
- [!] 19.9 Erratum D1: the SN tests cannot select H0 (χ² flat with M free) — approve the change to the Validation paper's title and abstract?
- [!] 19.10 Erratum M3 / Missing Satellites paper: with Mechanism B out of the book, is the paper withdrawn or corrected?
- [!] 19.11 Erratum M1: remove the M–σ material from the remaining LaTeX sources and scripts listed in the row?
- [!] 19.12 Erratum N1: regenerate the PDFs in `docs/papers/` without correspondent names (the LaTeX is done)?
- [!] 19.13 Late-time growth and Level 2 new findings (Section 17.3): approve the errata rows (`cg_MANIFEST.md` l.173).
- [!] 19.14 w(z) and survey corrections (Section 17.6): approve the ten corrections made without errata rows (`wz_MANIFEST_darkenergy_surveys.md` l.32).
- [!] 19.15 Measurement problem: confirm the recomputed dust-grain row and 10^−33,599 (Section 17.8).
- [!] 19.16 Quantum Darwinism §7.1 "black holes form when the local rate saturates the bound": not derived, no errata row — carry as \conjecture or leave out? (`qr_MANIFEST_records_measurement_time.md` l.185)
- [!] 19.17 Quantum Darwinism Fig. 4 DESI phantom-crossing redshifts: give the source, or leave out? (same line)
- [!] 19.18 Three-way clusters: give the sources of the four binned observed ratios (the printed Sereno 2017 and Grandis 2024 references do not match CrossRef)? (item 4.8)
- [!] 19.19 Score Card SC1: voids under µ < 1 are less evolved, not emptier — approve the correction?
- [!] 19.20 Score Card SC5: supply `tests/iam_derivation_tests.py` (not in the repo) or drop β = 1.058, χ² 41.43/10.20, 7.05 %?
- [!] 19.21 Prediction-paper placement: survey predictions, lensing dynamics and three-way clusters are chapters in Part 2 (p2_16–p2_18); move them to the predictions chapter, or keep?
      (`wz_MANIFEST_darkenergy_surveys.md` l.235–236; `BOOK_READING_TODO.md` rows 19–21)
- [!] 19.22 Electron mass: which expansion rate prices the electron's bit, photon-sector 67.16 or matter-sector 72.26? Printed as \openprob for now (`MANIFEST_particle.md` l.547–548).
- [!] 19.23 Order of particle chapters: electroweak (mass generation) before Koide and the electron mass? (`MANIFEST_particle.md` l.29–30)

## Put back at commissioning (open)
- [ ] Six Part VI figures taken out 2026-10-03 (development values): see `COMMISSIONING_RETURNS.md`; markers `% COMMISSIONING-RETURN` in the chapters. Redraw from the commissioned chain and restore the sentence that points to each.
- [ ] Replicate precision in the serial chapter: state the commissioned within-person spread (development notes DEV_REPL_V3_01, DEV_SELFTARE_01).
