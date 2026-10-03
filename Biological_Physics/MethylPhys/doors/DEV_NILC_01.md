# DEV-NILC-01 — stage 4 NILC component separation on chain v3 (development; check written 2026-10-03 before the data were read)

**Commissioning step 1 (SOP v3 section 2b).** Check: on constructed mixtures of purified cells, NILC recovers the known fractions within a
pre-set error, and on the same-run replicates its fractions repeat within a pre-set spread. Outcome below the line; nothing above it changes.

**Methods under test.**
- N1 `chain/nilc_celltype_deconvolver.py` as built (the toolkit module): atlas `atlas/IAMAtlasREBUILD.csv.xz`, markers
  `chain/Runtime Matrices/Celltype_Marker/iamatlas_celltype_markers_v0_2.json`, module defaults; cell fractions summed to the 8 blood groups by
  cell name (NEU, EOS, BASO, MONO, B, NK, CD4T, CD8T; any other cell = other).
- N2 NILC-e, the second opinion of `doors/data/DEV_ATLAS_EPIC_01/STAGE_A_PROPOSAL.patch`: `comp_methods.NILC` on the 12 array-measured
  circulating atlas v2 cells (variant e), sites with template range >= 0.2, the 6,000 neutrophil identity sites excluded, covariance `atlas`
  (template variance + 0.02^2), no offset (D0). Settings as frozen in DEV-ATLAS-EPIC-01; nothing re-chosen.

**Truth sets (held out: neither was used by DEV-ATLAS-EPIC-01).**
- H1 GSE181034 umbilical-cord-blood DNA mixtures (12 physical mixtures; truth = the depositors' percentages cd4t, cd8t, bcell, nk, mono, neu).
- H2 constructed mixtures of GSE122244 purified healthy-control arrays (5 donors; neutrophils, monocytes, B lymphocytes, T lymphocytes from the
  same donor), beta = sum f_c beta_c at every site measured on all four arrays (linear in beta; a computational construction, stated as such).
  Six fraction sets per donor, (NEU, MONO, B, T): (0.60,0.10,0.10,0.20) (0.70,0.05,0.05,0.20) (0.50,0.15,0.10,0.25) (0.40,0.10,0.15,0.35)
  (0.30,0.20,0.20,0.30) (0.20,0.10,0.10,0.60); 30 mixtures. T = CD4T + CD8T.
- In-sample sets (GSE110554, GSE167998, GSE182379 Salas mixtures; GSE112618 flow-counted bloods) are reported, not scored.

**Bars (per method).** On H1 and on H2: neutrophil RMSE <= 0.02 and every other group in the truth RMSE <= 0.03.
Repeatability: GSE250556 pooled-DNA replicates (same DNA, 4 people), within-person SD of every group <= 0.010.
A method passes when every bar holds. Beta vectors: DEV-BASE-CHAIN-01 Stage 1 (`--save-betas`; the diagnostic arm where Stage 0 stopped
an array for a missing age or sex).

**Wiring rule.** A passing method is wired as stage 4: its fractions are written to the bundle (`nilc`) and the report; the Met-A reading is not
changed (its composition input stays stage 2 until the author decides otherwise). A failing method is wired only behind a flag, off by default.

---
## Outcome (recorded 2026-10-03 after the run; nothing above the line was changed)
Box job 91d7b32f (chain commit 63d55fa; beta vectors from DEV-BASE-CHAIN-01). Records: `data/DEV_TOOLKIT_01/` (`scores.csv`, `repeatability.csv`,
`agreement.csv`, `fractions_long.csv`, `summary.json`, script `toolkit_b.py`).

| method | H1 cord-blood mixtures (RMSE) | H2 constructed (RMSE) | GSE250556 pooled within-person SD | verdict |
|---|---|---|---|---|
| N1 toolkit module | NEU 0.050, MONO 0.073, B 0.095, NK 0.055, CD4T 0.125, CD8T 0.137 | NEU 0.262, MONO 0.119, B 0.103, T 0.142 | all <= 0.009 | **FAIL** (every truth bar) |
| N2 NILC-e | NEU 0.061, MONO 0.033, B 0.042, NK 0.022, CD4T 0.037, CD8T 0.129 | NEU 0.138, MONO 0.053, B 0.014, T 0.097 | CD8T 0.0127 (> 0.010), rest <= 0.006 | **FAIL** |

**Stage 4 is not wired.** Read beside it (not part of the verdict): the running stage 2 (NNLS8) reads the same truth sets with NEU RMSE 0.095 (H1)
and 0.132 (H2). Two observations recorded after reading, as development findings, not used to move anything: (1) on H1 every method reads
neutrophils low by 0.05-0.09 (cord-blood neutrophils against adult templates); (2) H2's premise that the GSE122244 purified arrays are pure does not
hold for every donor - stage 2 reads neutrophil fractions of 0.76 and 0.28 in two "monocyte" arrays and 0.54 in one "T lymphocyte" array - so H2 is
not a clean truth set. A cleaner held-out truth set is needed before stage 4 is tried again (author decision).
