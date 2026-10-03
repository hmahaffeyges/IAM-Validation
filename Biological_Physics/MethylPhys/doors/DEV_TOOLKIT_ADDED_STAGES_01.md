# DEV-TOOLKIT-ADDED-01 — stages 3b, 3c, 11b, 12b (development; checks written 2026-10-03 before any data were read)

Added toolkit stages, after the six commissioning steps (SOP v3 section 2b). One section per stage; outcome below the line.

**3b trace-cell detection** (`stage_2c_trace_detection.py`, `trace_detection_panel_v1.json`). TOOLKIT.md: its thresholds were a healthy-donor
percentile and it reads classes, not cells; "its line must be re-set without a population at commissioning". A check needs that line first.
**Not run; author decision**: the physics-only line (e.g. the score test's own standard error on this array) and the cell set.

**3c foreign-cell detection** (`toolkit_foreign_detection.py`, `detection_panel_v3.json`). Its line is the 0.99 quantile over 732 arrays known to
lack the cell (a population quantile) and its input was the class-era scale-mapped beta. **Not run; author decision**: a line without a
population (e.g. each template's own NNLS standard error on this array) and the beta scale.

**11b surface brightness** (`toolkit_surface_brightness.py`). Reads class-era fields and the class archives' per-CpG brightness CSVs; v3 has no
class. **Not run; author decision**: whether the interval is wanted on the per-cell Met-A, and from which per-site uncertainty (for EPIC
neutrophils the 6 floor arrays' per-site SD is the only per-cell source).

**12b difference map** (`serial_mode.py`). Built: the per-address difference of two draws (`delta_sky`) and the same-person check; not built:
the difference drawn as a sky; `delta_cells`/`trajectory` read class-era fields. Check on what is built: (i) `check_same_person` through a v3
adapter (patient hash = Stage 0's hashed id, array type, Stage 1 pipeline) accepts two bundles carrying the same hash and refuses different
hashes; (ii) on GSE250556, for every person, the q99 of |delta beta| between two pooled-DNA replicates of that person is below the q99 between that
replicate and every replicate of each other person, in >= 95 % of comparisons. Pass -> per-address difference wired behind `--prior-betas`;
the sky drawing stays not built.

---
## Outcome (recorded 2026-10-03 after the run; nothing above the line was changed)
Box job 91d7b32f (chain commit 63d55fa; beta vectors from DEV-BASE-CHAIN-01). Records: `data/DEV_TOOLKIT_01/` (`scores.csv`, `repeatability.csv`,
`agreement.csv`, `fractions_long.csv`, `summary.json`, script `toolkit_b.py`).

- 3b, 3c, 11b: not run (reasons above, recorded before data). Author decisions.
- **12b: PASS.** (i) the v3 adapter accepts two draws with the same identifier hash and refuses different hashes; (ii) GSE250556 pooled replicates:
  same-person q99 |delta beta| median **0.060** against **0.174** to other people; same person below in **348 of 348** comparisons (bar 95 %).
  **Wired** behind `--prior-betas` / `--prior-bundle` (bundle `difference_map`, report section, toolkit row; release check E5). Every bundle now
  records the identifier hash (Stage 0's hashed id, or `--patient-id` hashed for a beta table) and the pipeline, so a later draw can be compared.
  Not built: the difference drawn as a sky; `delta_cells`/`trajectory` still read class-era fields and are not called.
