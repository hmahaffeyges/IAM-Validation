# IAM's Law and Order — master TODO (one book, five parts)

Started 2026-10-02. One book: Part 0 front matter, Parts 1–5, appendices, one bibliography, one figure folder, one main.tex.
Always "Part N"; never "the cellular book" or "the quantum book".
Key: [x] done · [~] started · [ ] not started · [!] needs the author.

## 0. Now
- [~] 0.1 Clinical / treatment / lifestyle advice off the public repo
  - [x] manual renderer mphys002_lib.py: intervention texts, ranked levers, post-breach "therapeutic zones" section, glucose and
        end-of-life glossary entries, Z-score "clinical action" column, screening/surveillance directives, C3 "therapy" text
  - [x] tier_breakpoints.json ("metabolic support", "holistic intervention"), cpg_gauge label, v2 How-to text
  - [x] Alpha Omega 5 "Levers past the Warburg wall"; v2 SOP "adjust lifestyle"; CHANGELOG wellness callout
  - [x] predictions register: 7 treatment/care claims dropped (CEL-001, 029, 136-139, 142); Issue 002 glossary review: 8 rows
  - [x] Operations Manual rebuilt (8ed577e): the 235-page rebuild of 236b146 still had a testicular-cancer workup/fertility scenario and
        treatment-response glossary entries; class cards, Issue 002 glossary and end-of-life paragraph no longer rendered; 159 pp, 0 directive hits
  - [x] 39 files moved (nothing deleted) to Biological_Physics/RETIRED_2026-10 with INDEX.md; links unchanged at 16 broken (pushed 236b146)
  - [x] the 8 v2 report pages moved to RETIRED_2026-10
  - [x] author: move to a retired folder, delete later once he confirms copies (20 Record files moved; RETIRED_2026-09 left in place)
  - [x] both papers retired (author: superseded by the book; their content goes into Part 4)
  - [x] reference.html retired with the v2 pages
  - [x] pushed e3b7097 and 236b146
- [ ] 0.2 Boxes
  - [x] box 2 relaunched (m7a.32xlarge, 128 cores)
  - [~] coho Le Luyer resubmitted (99a3baca), running
  - [x] DNMT Part B scored: IAM-A 1.65-1.97, Q1 8/8, Q2 8/8 PASS (pushed e3b7097)
  - [ ] then, per the master order: 580-species atlas fish, coho WGBS, stool, neutrophil commissioning tests
  - [ ] hourly box report while runs go
- [ ] 0.3 One book in one place: repo docs/book/ = main.tex, preamble.tex, part0/ … part5/, appendices/, figures/, iam.bib,
      figscripts/. Part 4 and the Part 3 reach chapter moved in. Old part*_drafts retired. Private material never in the repo.
- [ ] 0.4 Overleaf zip of the whole book that compiles as-is (compile here first)

## 1. Part 0 — front matter
- [x] 1.1 Title: IAM's Law and Order: How Decoherence Writes the Classical World, and the Energy That Holds It Against Thermal Noise
- [ ] 1.2 Preface (exists) with the author's voice from Shoulders of Giants and the GRF essay
- [ ] 1.3 How to read this book: the status labels, what each means, one example each
- [ ] 1.4 Notation and units
- [ ] 1.5 Acknowledgement of the giants ("the messenger, not the inventor")

## 2. Status labels on every chapter (agreed)
- [ ] 2.1 One macro set in preamble.tex: \derived \calc \calibrated \fitted \conjecture \prediction \observed \openprob (+ \measured, \record)
- [ ] 2.2 Every claim labelled; per-chapter label count table
- [!] 2.3 Part 4 uses MEASURED: keep as its own label or fold into OBSERVED

## 3. Part 1 — Introduction to IAM's Law and the Virial Theorem
- [ ] 3.1 Merge Quantum Order book ch 1-4 with p1_iams_law, p1_virial_law, p1_encoding_surfaces
- [ ] 3.2 Variational derivation and holographic derivation papers into Part 1
- [ ] 3.3 37-orders table in full, every row recomputed
- [ ] 3.4 Landauer, Bennett, Bekenstein, Jacobson, Zurek background
- [~] 3.5 Virial theorem as one instance of the law, operating in every domain (author ruling 2026-10-02): atoms/molecules/bonds are 1/r,
      so the half holds inside every star, cell and transistor [x] Part 1 + Part 4 wording; CMOS CV^2 halves [x]; one statement or two? CONJECTURE

## 4. Part 2 — Informational Actualization of GR: The Cosmological Dynamic
- [ ] 4.1 Merge Quantum Order book ch 5-14 with the 15 Part 2 drafts
- [ ] 4.2 Chapters still to write (papers already read in full): survey predictions, lensing dynamics, three-way clusters (Level 1
      only, on hold), missing satellites, w(z) far future, Quantum Darwinism, two faces of time, entanglement and records,
      electroweak, measurement problem, GRF essay
- [ ] 4.3 Every headline number re-run from the repo chains

## 5. Part 3 — The Quantum Order of Informational Actualization
- [ ] 5.1 Merge Quantum Order book ch 15-21 with p3_one_gauge, p3_xqp, p3_reach
- [ ] 5.2 the quantum-processor report and the semiconductor report Issue 002: re-read the chunks that came back truncated; add the physics (no product material)
- [ ] 5.3 Encoding-surface "bones": re-read the truncated chunks before using any of it
- [ ] 5.4 Platform demo pages: figures only

## 6. Part 4 — Cellular Physics: Thermodynamics of the Methylome
- [ ] 6.1 Physics of Methylation: Landauer Metrology — read in full for the book and write it in: Sanchez & Mackenzie premise
      (cited), filter vs ruler, M table of substrates, pipeline map, four labs on one scale, CLSI EP28
- [ ] 6.2 What is AstroGenetics (tex) and Cellular Thermometer PDF: read in full; the author's language into the bridge and sky chapters
- [~] 6.3 Pasted texts of 2026-10-02: 08-51 [x] gauge chapter; 08-57-18 [x] 79-tool map; 08-57-49 [x] sky-tools chapter;
      09-12 (methylation as Φ_IAM, 50/50 partition in cells) [!] not used — conflicts with 08-51 ("cells aren't 1/r bound");
      proposed: Part 5 as CONJECTURE with the conflict stated
- [ ] 6.4 Report-tab prose restated on the current method; re-read the tab chunks that came back truncated
- [ ] 6.5 Reproduction paper: re-read the truncated chunks before using any of it
- [ ] 6.6 Salmonid chapter with the honest result
- [ ] 6.7 New results as they land (EM-seq Part B, tumour pairs, neutrophil commissioning); IMR90 channels [x]
- [ ] 6.8 Test Validation Compendium, Overview Companion, Official Score Card: read in full (record only)

## 7. Part 5 — 37 Orders of Magnitude and the Web that is IAM
- [ ] 7.1 Synthesis: one law, the encoding surfaces, one gauge in every domain, the black hole and the cancer cell (surface full)
- [ ] 7.2 Falsifiable predictions: every one from every part, value / test / date / falsified-if, checked against source;
      cell predictions only after commissioning
- [ ] 7.3 Exploratory: Gravitational Propulsion and IAM; Gravitational Engineering Exploration (read in full; CONJECTURE throughout)
- [ ] 7.4 Interpretation chapter (exists)
- [ ] 7.5 Status table merged across all parts
- [ ] 7.6 Open problems: still to derive (cell floor from first principles, 3/16, (2π)^(3/10), history integral in the
      horizon-entropy variable, Met-A floor, breach and cancer region on the gauge, …) with our plan for each
- [ ] 7.7 Future tests and experiments: ours vs ones for others
- [ ] 7.8 Conclusion

## 8. Appendices
- [ ] A Constants (generated from CANON)
- [ ] B Formula sheet: every equation, numbered, pointing to where it is derived
- [ ] C Derivations in full (anything a chapter shortens goes here)
- [ ] D Errata to the source papers
- [ ] E Reproduction: repo, commits, chains, scripts, how to rebuild every number and figure
- [ ] F Glossary (large): CANON/GLOSSARY.md + Quantum Order book + Part 4 + every term used
- [ ] G Predictions register (cleaned)
- [ ] H Open items and our work plan
- [ ] I Figure and data provenance
- [ ] Bibliography (large): merge all paper bibliographies (861 \bibitem in the repo .tex) + Quantum Order book + Part 4;
      dedupe by DOI; check each DOI

## 9. Figures, tables, visuals
- [ ] 9.1 Inventory: 401 repo images (outside retired/record), 17 figure scripts, every paper's figures, 37 Part 4, 74 book3/new
- [ ] 9.2 Per chapter: at least two figures and one table
- [ ] 9.3 Regenerate any figure with retired content (1.10 line, class floors, tiers, cohort bands)
- [ ] 9.4 New: one gauge across domains; encoding surfaces across 60 decades; 37-orders ladder; law-to-domain map; µ(z) and fσ8;
      qubit and chip gauges; atlas coverage; sky maps [x]; C(d) [x]; IMR90 channels
- [ ] 9.5 One figure script per part; every figure rebuildable

## 10. Derivations audit
- [ ] 10.1 Every derivation walked; nothing "it can be shown"; every number in a check file
- [ ] 10.2 Incomplete derivations listed -> Appendix C or open problems

## 11. Repo
- [ ] 11.1 Book folder README, build script, compile check
- [ ] 11.2 v3 report rebuilt with the tab prose
- [ ] 11.3 Everything shown to the author is in the repo (except private material)

## 12. Every paper carried (added 2026-10-02)
- [ ] 12.1 Per-paper table: paper lines/words vs chapter lines/words, equations, tables, figures (`paper_vs_book_2026-10-02.csv`; recompute after every merge). Done only when every row shows the paper carried.
- [ ] 12.2 Line-for-line rewrites running: IAM's Law (p1_02), Theory (p2_03), Koide / electron mass / Higgs (new p2_15a, p2_15b, Higgs chapter), six virial papers + quantum-level gravitational decoherence (p1_03, p2_02, p5_05), Bekenstein coefficient + two black-hole papers (p2_01 + new chapter).
- [x] 12.3 Floor Breach (black hole to the cell) as the Part 1 opening chapter, in the author's words (d73d3b7).
- [ ] 12.4 Next wave, every paper under ratio ~0.6: baryon asymmetry (2), dark energy or sector tension, measurement problem, missing satellites, Quantum Darwinism, cosmological constant, x_qp, three-way clusters, dual-sector note, lensing and dynamics, dual-sector validation, Landauer metrology, two faces of time, S8 trend, dual-sector perturbation, w(z), survey predictions, exploratory, late-time growth.
- [ ] 12.5 Author's own words carried verbatim in every chapter (corrected only where an erratum or rule requires); interpretive and speculative passages carried labelled, excluded only when a correction shows them wrong.

## 13. Application chapters (after section 12)
- [ ] 13.1 Qubit and chip chapters (Part 3) against the quantum-processor report Issue 002 and the semiconductor report Issue 002, read again in full; cell chapters (Part 4) against the the methylation report issues, web.py, the reproduction paper (the cell-reading engine rebuild) and the Hubble2Methyl documents.
- [ ] 13.2 Rule: carry everything a referee needs to confirm the physics and the readings; floor values, specific application methods and product names stay out.

## 14. Author's cellular items file (added 2026-10-02)
- [ ] 14.1 "Law and Order - IAM Cellular Physics: Thermodynamics of the Methylome - ITEMS TO CONSIDER" (67 KB): lead reads every line, confirms each item against chain code, frozen files, outcome records and canon, and adds confirmed items to Part 4 in the author's words; item-by-item verdict list to the author.
