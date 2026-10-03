# PROC-STAGE2D-03 — outcome: ADOPTED by the author's ruling. B1's unspecific bar failed as written and was superseded, not loosened.

Scored 2026-09-27 against [`PROC_STAGE2D_03_PREREG.md`](PROC_STAGE2D_03_PREREG.md). Runner kit/PROC_STAGE2D_03.py (archived privately); null on 732 arrays
kit/results/PROC_STAGE2D_03_null_732.csv (archived privately); spikes kit/results/PROC_STAGE2D_03_spikes.csv (archived privately); bars kit/results/PROC_STAGE2D_03.json (archived privately).

| bar | measured | |
|---|---|---|
| B1 per-template FP, leave-one-chip-out, 732 arrays | 0.0–1.23 % (bar 1.5 %) | MET |
| B1 UNSPECIFIC arrays (≥ 3 templates above floor) | **14 of 732 = 1.9 % (bar ≤ 1 %)** | **FAILED as written** |
| B2 Karolinska + UCLA under Uppsala floors | not yet scored on v3 (v2 scored 0 fires on 23) | pending |
| B3 detection, honest spikes, 12 new hosts | full-coverage 100 % at 5 %; thin 83 % at 5 %; all 96 % at 10 % | MET |
| B4 naming | 94 % at 10 %, 74 % at 5 % | MET |
| B5 composition unchanged, old vs new conductor | max diff 0 on 12 arrays | MET |
| B6 reference Glia not scored; no undetected non-blood cell scored on 100 arrays | Glia NOT_DETECTED, not scored; 0 of 100 | MET |
| B7 kit test rewritten to the contract | kit/test_stage2d_panels.py (archived privately) — K2 allows UNSPECIFIC ≤ 4 % of N, **looser than the bar**, for the reason below | written |

**The pre-registered decision rule alone says NOT ADOPTED** — B1 is not the "B3/B4 thin-only" exception. **The author ruled (2026-09-27):** "That doesnt mean automatically treating 1.9 percent as false." The 1.9 % is a measurement and is printed as measured; the 1 % bar is recorded as failed-as-written and superseded by that ruling.

## Why it is wired in anyway, and what that is
The 14 UNSPECIFIC arrays are not noise the bar can legislate away: they are the oldest arrays in the cohort (median age 86 against 47),
their immune A is 1.037 against 0.991, their total foreign-like mass is 5.5 % against 0.6 %, and foreign-like mass correlates with age
at r = 0.69 across all 732 (not chip, not tare, not sex). The bar of 1 % was written by me before this was measured. Under the author's
ruling — development stage; fix what is wrong, move on, write it down — the joint fit replaces a detector that false-alarmed on 48 % of
healthy arrays and could not see a real 5 % spike (PROC_STAGE2D_02_OUTCOME.md). **That is a development decision, not a bar pass, and it is
the author's to confirm or reverse.** Until he does: the page prints "epithelial-like material, cell not resolved, x %" on such arrays
and this file is linked from the Stage 2d section.

## Also measured
- The undifferentiated gastric family is not resolved from the differentiated one on this block (0 % at 5 %, 17 % at 10 %): NOT DETECTABLE.
- Glia and Thyroid carry a standing 0.2–0.8 % on healthy blood (printed beside the floor). The Glia BREACH the author caught on the reference
  array came from the solver's 1.03 % Glia clearing the 1 % cell floor; the B7 gate (a non-blood cell in whole blood is scored only when
  detected) closes that, and the joint fit does not detect Glia on that array.

## The author's ruling on the open question
The 1.9 % is not called false and not called a cell: where ≥ 3 templates fire the page prints "epithelial-like material, x %, cell
not resolved"; where one template fires above its floor the page names the cell. **Scoring A on a detected foreign cell** at ~2 % is a
theory from the early VALs that "we need to test" — until a pre-registered test decides the scoring floor (real spikes at 2 / 5 / 10 %,
A recovered against A true; PLAN item 9, solver precision on minority cells), a detected foreign cell prints its fraction and "A not read
below the scoring floor". That test is PROC-FOREIGNSCORE-01, to be written.
