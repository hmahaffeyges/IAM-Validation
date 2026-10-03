# DEV-ATLAS-EPIC-02 — stage 3 atlas deconvolution on chain v3: the held Stage A patch (development; check written 2026-10-03 before the data were read)

**Commissioning step 2.** Check: as for NILC on constructed mixtures, and agreement with the 8-group composition on whole blood where both
apply; each additional cell type is read only after its own floor and reference are commissioned ("no cell rather than part of one").

**Under test.** `doors/data/DEV_ATLAS_EPIC_01/STAGE_A_PROPOSAL.patch`, engine `atlas_e`: `deconv_v2.DeconvV2` with the frozen SOLVER settings
on the 12 atlas v2 cells that circulate, were measured on arrays, and survive the mixture-identifiability rule; neutrophil identity sites never
markers. Point estimates (n_boot 0), as in the patch. Also tested: (R) the variant-e rule itself — re-run on the atlas v2 array-measured blood
cells it must remove exactly `b cells`, `cd4 t cells`, `cd8 t cells` (reproducibility of the rule as recorded).

**Truth sets and bars.** H1 and H2 exactly as DEV-NILC-01 (held out: variant e was formed after DEV-ATLAS-EPIC-01 run 2 had been read on the
in-sample sets). Neutrophil RMSE <= 0.02, every other group RMSE <= 0.03; GSE250556 pooled-replicate within-person SD <= 0.010 per group.
Agreement: on every whole blood labelled healthy reference and read in DEV-BASE-CHAIN-01, |mean(atlas_e - NNLS8)| <= 0.02 for NEU and <= 0.03
for every other group. Pass = every bar.

**Wiring rule.** Passing: `atlas_composition` (cell fractions, 8-group sums, NNLS8 comparator) written to the bundle and the report; Met-A keeps
the stage 2 composition until the author decides; no new cell is read (stage 5 per cell is its own step). Failing: behind a flag, off by default.

---
## Outcome (recorded 2026-10-03 after the run; nothing above the line was changed)
Box job 91d7b32f (chain commit 63d55fa; beta vectors from DEV-BASE-CHAIN-01). Records: `data/DEV_TOOLKIT_01/` (`scores.csv`, `repeatability.csv`,
`agreement.csv`, `fractions_long.csv`, `summary.json`, script `toolkit_b.py`).

- (R) the variant-e rule reproduces: it removes exactly `b cells`, `cd4 t cells`, `cd8 t cells`; 12 cells remain. **PASS.**
- Truth: H1 NEU 0.083, MONO 0.025, B 0.026, NK 0.045, CD4T 0.024, CD8T 0.053; H2 NEU 0.122, MONO 0.059, B 0.007, T 0.103. **FAIL** (NEU both sets; NK, CD8T on H1; MONO, T on H2).
- Repeatability (GSE250556 pooled): every group <= 0.0065. **PASS.**
- Agreement with NNLS8 on 539 healthy whole bloods: mean differences NEU +0.012, EOS -0.016, BASO -0.008, MONO -0.014, B +0.012, NK +0.003, CD4T -0.014, CD8T +0.024. **PASS.**

**Verdict: FAIL (truth bars). Stage 3 is not wired; the patch stays held.** B cells pass every per-cell bar (H1 0.026, H2 0.007, repeatability 0.0012),
so B is the one cell that qualified for DEV-PERCELL-01. The H1/H2 caveats of DEV-NILC-01 apply (the running NNLS8 fails the same truth bars).
Also measured here, for the composition runtime file: the builder's own rule (chain_tests/blood_comp.py) re-run on the 91 Salas purified arrays gives
per-group marker counts B 150, BASO 150, CD4T 41, CD8T 22, EOS 150, MONO 150, NEU 150, NK 150; their union is exactly the file's 963 markers (no marker in
two groups) and the group means agree to 5e-6. Recorded in the file's `_meta` (data keys byte-identical).
