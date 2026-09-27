# PLAN — what we do next, in order

One line per item. No history, no strikes. When an item is done it comes OFF this page and its record goes to
[`ENHANCEMENTS.md`](ENHANCEMENTS.md) (the long ledger) and the register. Rule for the order: chain before documents, documents before
launcher, and nothing edits the chain while a procedure is scoring.

## Now (in flight)

- (items 1, 2 and 3 done 2026-09-27: audit round 3 pushed a48230e; PROC-TARE-01 sealed NOT COMMISSIONED; tier_breakpoints v1.5 makes NORMAL [0.95, 1.05) about the fixed point)


## Chain — after TARE-01 finishes, one at a time

3. ~~Deferred chain patch~~ **DONE 2026-09-27, c09e9ef** — folded into the MEASURE-DON'T-COMPARE removal: stage 5/6 cohort stages deleted from the conductor (not just their calls), Patient_CMB on the search path, Percell_Reference / Cellular_Age / Mahalanobis_healthy_reference off it.
4. ~~PROC-UNMIX-01~~ **SEALED NOT ADOPTED 2026-09-27** — 5 of 6 bars failed; the inversion is exact with true fractions and reads the solver's ±0.03 error otherwise. Fraction stays a gate. Re-test only after a more precise solver (item 27).
5. **Sky zero and spread from the atlas posterior** — a constructed atlas specimen must read quiet; the four panels must stay at 2.6-3.2 %.  ← PROC-SKY-01 sealed 2026-09-27: B2 failed (36/48; GSE125105 3/12 — see FINDING_GSE125105_LOW_SIGNAL.md). Panel scales retired. **Author's decision pending: withhold the sky, or draw the residual in β units with no σ (plus the serial difference map).**
5a. **PROC-INTAKE-01** — the Stage 0 detection/call-rate gate runs on the array's own numbers (pre-registered 2026-09-27); failed probes masked before any mean; deferred never advances. Then re-run PROC-SKY-01's script with Munich's sub-threshold arrays flagged — the outcome stays sealed, this is the follow-up.
6. **Held-out Stage 2d** — per-array shards, run alone, then register row B-12 says verified or not.
7. **Stage 2d kit test** — commissioned panels never fire more than 1-in-n; no panel → "not commissioned", nothing else.
8. **Twin/family thresholds as a runtime matrix** — no constants in code.
9. **Chip term** — read TARE-01 B6; if the tare does not remove it, an on-chip reference (control probes) does.
10. **Serial mode** — `run_sample.py --prior <bundle>`: same patient, per-cell ΔA, Δfraction, difference sky; change floor pre-registered.

## Documents — written once, after the chain above is still

11. ~~SOP~~ DONE 2026-09-27 (build_all) — **SOP** — LESSON-DECON-01, Stage 2d and its commissioning rule, TARE/UNMIX outcomes, pre-registration conventions; every runtime file named from the tree.
12. ~~Operations Manual~~ DONE 2026-09-27 (build_all) — **Operations Manual** — engine section generated from [`chain_sequence.json`](../chain/chain_sequence.json); then every rendered page read against a checklist; per-tab figures regenerated from the audited report.
13. **The cells** — an OM section, one entry per scoreable cell, rewritten from the webpage drafts onto the physics (no cohort range, no wellness framing); welcome-page explanations folded into OM front matter and the Physics/How-to tabs where they add something.
14. ~~Documentation catch-up~~ DONE 2026-09-27 (build_all) — **Documentation catch-up** — manifest, component map, runbook, chain sequence, inventory naming the per-cell surface, the reference folder, the solve block and twin rules (mostly regenerates; read anyway).
15. ~~START_HERE.md~~ DONE 2026-09-27 (build_all) — **START_HERE.md** — what this is, which document is canonical for what, the one command that verifies the chain.

## The face of the chain

16. **Launcher** — local `run.py`: verify files by hash, take the IDAT pair, run stage by stage with live pass/fail, open the report; replication menu for sealed PROCs.

## Procedures — once the chain is fully commissioned

17. **PROC-BRAIN-01 redo** — clean single-provenance CSF run.
18. **Commission a solid-tissue laboratory** — pipeline map and floors; gates the glioma and progression re-tests.
19. **Gastric** — six stomach entries, two families, never tested.
20. **Breast shedding in real patient blood** — the question that started the detection work.
21. **Re-run the webpage-draft VALs (CRC, breast, HCC) as pre-registered PROCs** — education re-tested on the commissioned chain.

## Later chain work

22. CD4/CD8 separation — needs loci this block lacks (second blood block).
23. Second-block solve for the nine finer blood subsets.
24. EPIC platform block (865k loci).
25. Coverage floor on any class-selection rule.
26. **ENHANCEMENTS re-plan** — once 1-16 are done, rewrite the ledger as one ordered plan for the remaining CMB borrowings, atlas duplicates and the EPIC block.

27. **Atlas v2 — joint hierarchical build** (needs the November machine or a rented 128 GB box, ~1-2 days). One MCMC over all cells with a shared per-locus probe effect, a per-source effect and a per-cell effect; undefined loci imputed from class and nearest cells with wide SD; cross-cell covariance kept. Acceptance, written before the run: the PROC-COV-01 misfit (0.067) falls below the NORMAL tolerance; twins collapse to one cell effect; a 252-locus cell's imputed loci carry SD > 0.1; H_min unchanged (the floor is G-002, not the atlas). Adding a cell from a new source (Loyfer, Moss) becomes one script: scale map → coverage family → twin test → identity loci → MCMC reference → reads 1.00 on itself. Consequence: the cross-cell covariance is the noise model the matched filter lacked - MF-01 gets a second, pre-registered test on atlas-derived covariance once v2 exists.

- (2026-09-27, done: **build_all.py** - one command regenerates every document that reports the chain (SOP mirror + repoint, OM PDF, cell descriptions from the author's drafts, reviewer manifest, fresh reference report + tab reference, measured REPO_INVENTORY, RUNBOOK/README generated blocks, folder READMEs, GENERATED_MANIFEST.json) and gates on vocab_scan (report guards over SOP, OM, every report tab, the paper), SOP-mirror reconciliation and link_check; guarded_push runs it before propagate. **Item 13 done**: cell_descriptions_v1.json (52 of 115 cells with biology from 27 drafts; drafts stripped to biology, guarded). **SOP checked step by step** against chain_sequence.json: 131 sections bannered LIVE/RECORD/NOT IN CHAIN/NOT BUILT/KIT. **Paper** (Landauer_Metrology_of_the_Methylome.tex) revised to three laboratories + Munich as low-signal input, population layers moved to a record subsection, under the same guard; the report links the PDF once the author commits the Overleaf build.)

## Standing (not tasks)

- Every push carries a copy of the changed files. Data worth keeping is bundled before it can be lost.
- Healthy is A = 1.00; the tier scale is the tolerance. No cohort defines any number on a cell.
- A pre-registration is written before data is read and does not move afterwards.
- Verify by reading the render; write the verdict after.
