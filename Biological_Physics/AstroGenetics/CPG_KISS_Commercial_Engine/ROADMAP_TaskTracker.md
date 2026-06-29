# CPG_CMB_vKISS — Living Roadmap & Task Tracker
**Owner:** Heath W. Mahaffey · **Scribe:** Walther · **Updated:** 2026-06-29 · read at the top of every session.

Legend: [x] done · [~] in progress · [ ] open · [B] blocked-on-Heath-go

---
## TRACK 1 — vKISS chain (the lean clinical engine)
- [x] Map the v5 chain against real code; verify A-score core reproduces documented numbers.
- [x] Diagnose the false-fire (deconvolver-free per-cell = bulk-mixture artifact, sd~0.25).
- [x] Corrected design: Walther-alone presence + derived-hull verdict (gates no call). Verified healthy→within-band.
- [x] Conductor `cpg_cmb_kiss.py` rewritten + verified (healthy clean, departure flagged & named).
- [x] Presence rule locked: Mode-1 (3% composition gate, hull) vs Mode-2 (A-score-driven shed, NOT fraction-gated; 1.292% cortical neuron).
- [x] **Noise fix found + built**: low-rep cells = wide MCMC posterior sd → `atlas_cell_reliability.py` (Microglia/Kupffer/aliases rel≈0, clean cells rel=1.0).
- [B] Repoint pipeline v1_8 → v1_13: `walther_clinical.py:158`, `build_strawman.py:3`, `enrich_strawman.py:8` (one line each).

## TRACK 2 — Report builder → `cpg_report_builder_KISS.py` (built + verified end-to-end)
- [x] Wire CI/reliability noise fix into the per-cell readout. Per-cell 95% CI renders ([lo, hi]); a wide CI (>0.05) auto-tags "indicative · thin reference" (correctly quiet on well-constrained cells). Uses the bundle's existing brightness CI — no archive re-read.
- [x] Fold Cosmic Methylome Background back in as **Stage 4.6** — collapsible with the reference plate thumbnail + the per-patient projection note (full 8-panel sky map renders on the production box: healpy + the cpg→HEALPix mapping).
- [x] Surgical cuts verified clean: cfDNA (self-gates off for whole blood) · deconvolver-disagreement collapsible removed · NILC/two-deconvolver intro+footer reframed to **one deconvolver (composition/presence, gates no call)** · footer→vKISS. (Only "NILC" left in output is random chars inside a base64 gauge image — not text.)
- [x] Run end-to-end: built `CPG_report_DEMO_vKISS.html` (full readout, CI rendering, Stage 4.6, machine-readable snapshot) and `CPG_report_SAMPLE_healthy_43M_vKISS.html` (scale guard correctly withheld on the cached healthy β — see gap below).
- [x] **v1_13 repoint** (3 one-line edits): `walther_clinical.py:158`, `builders/build_strawman.py:3`, `builders/enrich_strawman.py:8` → v1_13.
- [x] **Strawman LIT UP**: fixed path resolution (find engine assets via `CPG_ENGINE_ROOT`) + the age/sex `None`→`""` bug (`html.escape(None)` crash). Patient wall (81 cols) + crown-jewel wall (81×81) render in iframes. `strawman_data_v2.json` (builders/) is the live one.
- [x] **Full whole-blood report**: `CPG_report_WHOLEBLOOD_RA_vKISS.html` — GSM1051525 (RA), passes scale guard (0/3 below floor), all sections, 28 per-cell CIs, Stage 4.6, strawman, honest Mode-1 resemblance (lung_cancer 100% shape resemblance — resemblance, not probability).
- [x] **healpix provenance cleaned**: removed external manifest/Zhou-lab mentions; hg19-genomic-order framing + explicit "no external atlas/reference/matrix, ever" statement. (Production copy on Heath's box needs the same one-time edit.)
- [ ] **DECISION (Heath):** embedded crown-jewel wall carries cfDNA/NILC mentions = honest provenance (HCC/glioma cfDNA-detected rows; one Walther-vs-NILC validation rho) in curated `strawman_data_v2.json`. Keep as-is (a, recommended) / filter to whole-blood diseases (b) / reword tooltips (c). Not touched without go.
- [ ] Cached healthy whole-blood β trips the scale guard (LESSON-DECONV-01, raw vs noob) — production IDAT path doesn't; verify "healthy reads healthy" once a noob-calibrated whole-blood sample is available.
- [ ] Assets to wire into the running dir: `cpg_gauge.py`, `A1_reference_gauge.png`, `star_gauge.png`, `iamatlas_cpg_to_healpix_nside128.npy`.
- Note: `atlas_cell_reliability.py` is the standalone proof of the noise finding; the report itself uses the bundle's per-cell CI (same brightness posteriors) for the "thin reference" tag, so the signal is already in the report. Wire the explicit reliability multiplier only if a separate number is wanted.

## TRACK 3 — Docs (after the report)
- [x] `flowchart_vKISS.html` (cfDNA removed, hull verdict, Mode-2 A-score, NILC→shelf, v1_13).
- [x] `CPG_Doctor_Workflow_KISS.html` (one tube, one tree, whole blood).
- [ ] Update flowchart + doctor files **if the report build changes anything** (e.g. Stage 4.6 added, CI shown).
- [ ] **Updated master roadmap + Lessons Learned** based on what the report generates + what's left.
      Document: the noise findings (low-rep → wide posterior → reliability fix), folding-not-cutting the CMB, the false-fire diagnosis, the presence-rule split.

## TRACK 4 — Validation (the Null Suite, N1–N8)
- [ ] Run **N7 once** against the vKISS chain (chain-recovery; last: Walther MAE≈0.008 PASS, end-to-end wired).
- [ ] N1–N8 per VAL as we re-test.
- [ ] Carry the N7 caveat: synthetic patients ≠ real n=601 HC distribution → use within-cohort/matched-arm reference for synthetic recovery, never the production hull.
- [ ] N7 v0.2 carry-forward: `restrict_panel_to_cpgs` default to marker substrate; matched arms / k-fold; extend R1+R3 → R1–R8.

## TRACK 5 — Disease coverage (the long arm)
- [ ] Finish the VAL tests; run every disease through the vKISS chain (directional sweep).
- [ ] Build all the residual maps we can (precision triad: breast, AD done; glioma + crc/pancreatic blood arms to check).
- [ ] Enhance the disease matrix by adding data collected from running diseases through the chain.
- [ ] Confirm every new disease **card** matches the **disease matrix**, and the matrix matches the **strawman**.
- [ ] Matrix audit follow-through (sign-convention sweep; cml/mds/mpn + lymphoid inversion risk; the 80-rows/53-labels duplicate flags: FTD ×2, PSP ×2, alzheimers placement).

## STANDING GUARDRAILS (do not regress)
- Whole blood only (cfDNA = GRAIL's lane). Deconvolver = composition/presence, gates no call.
- DERIVED-IAMAtlas-ONLY (no cohort pooling, no population comparison). Per-cell/per-patient is the unit.
- Surgical edits, before/after line counts, no deletions without agreement; preserve evidence.
- Push scope: `Biological_Physics/AstroGenetics/...` only; `atlas_vault` read-only. Version-bump, one canonical copy.
- Language: "consistent with / tested against / detectable"; never "proves/confirms/validates/resolves".
- Heath sets direction + says go; Walther sets up sealed/ready and awaits go. Nothing pushed without it.

---
## SESSION UPDATE 2026-06-29 (report fine-tooth-comb pass)
DONE + verified on the RA whole-blood report:
- [x] AND-gate fully cut (census + ranking): 1→45 cells, 0→67 CIs, strawman 3→41 mapped.
- [x] Reference bar gauge + star gauge collapsible render (CPG_ROOT vs CPG_ENGINE_ROOT path bug).
- [x] Mode 1 rewritten A-score-first vs the gauge; resemblance = pattern shape, "not a diagnosis/probability/stage."
- [x] AD = suppression toward H_min (20 neg/0 pos); -0.398 labeled directional-panel score, not an A-score.
- [x] Crown jewel → v3 wall.
- [x] **Disease-matrix specificity bug fixed**: disease_origin_cells.json was MISSING → solid cancers named off generic immune cells. Built origin map; lung/breast/rectal now NON_SPECIFIC; exec reads "non-specific systemic pattern, not flagged as a named malignancy." Root cause of the frightening reports.
- [x] "How CPG works" AstroGenetics explainer wired (collapsible); source CPG_AstroGenetics_explainer_section.html.
- [x] Mode-2 shedding framing aligned (elevated-amount = possible shedding, scored when abundant).

STILL OPEN:
- [ ] CMB patient panel: per-patient projection + Plate 3 + full-sky single CMB (patient sky-map renders on Heath's box).
- [ ] Literature-anchor section: wire to matrix evidence_anchors for flagged diseases.
- [ ] Heath red-pens the explainer prose.
- [ ] disease_origin_cells.json → repo Disease Matrix/DISEASE_MATRIX/ (pending push-go).
- [ ] Healthy whole-blood "reads healthy" full report once a noob-calibrated sample is available.
