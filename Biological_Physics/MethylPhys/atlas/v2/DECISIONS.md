# Atlas v2 — decisions

Every decision that shaped atlas v2, with who made it, when, and the author's words where he gave them. The full exchange is in
COMMUNICATION.md. A decision here is changed only by a new dated entry, never by editing an old one.

| # | date | decision | author's words / basis |
|---|---|---|---|
| D1 | 2026-09-27 | Find every public atlas that can extend the atlas; bring in each cell only through the same checks. | "do your diligence to locate any and all atlas's that could help expand and improve our atlas" |
| D2 | 2026-09-27 | Stream sources; keep only what is needed. | "Hopefully you can stream and only get what you need?" |
| D3 | 2026-09-28 | **The fit is cells only. No class enters the model** — no class prior, no class shrinkage, no brightness files. A class only names the floor a cell is divided by. | "The cell classes only need to be referenced by the A scoring module when its calculating h_min for a particular cell type"; "No cohort language, and no class scores" |
| D4 | 2026-09-28 | **The atlas sits on our own Stage 1 array scale.** Sequencing sources are mapped onto it by a source term; a patient array then needs no pipeline map. | "Def our own arry scale!" |
| D5 | 2026-09-28 | Pooled brain cells (Tian single-cell pseudobulk) enter flagged; v1-only cells stay legacy until a v2 source exists; embryonic stem lines fill the pluripotent class. | accepted the three recommendations |
| D6 | 2026-09-28 | Roster: mixtures OUT (CD3 T cells, granulocytes); podocyte OUT (no distinct identity from kidney tubule of the same kidneys, no hypomethylation at podocyte genes — likely sort impurity); cells with one sample WAIT. | twin test + marker check, `records/10_*`, `records/atlas_v2_roster_final.csv` |
| D7 | 2026-09-28 | Eight classes, by definition (one per dominant inversion); a future split must have its own inversion and a floor outside the parent's bars. | "8 trap doors, each with their own unique shape and size" — `doors/CLASS_HISTORY.md` |
| D8 | 2026-09-28 | **Sources whose term cannot be measured stay in, flagged ASSUMED with the reason and the estimated effect on A.** GSE63409 (HSC/progenitors) offset 0 ASSUMED. ENCODE stem lines were ASSUMED, then MEASURED the same day via GSE116754. | "Keep them in and notate the truth, its better than dealing with the noise again." |
| D9 | 2026-09-28 | **No partial cells.** A cell enters only if measured on the whole array (WGBS >= 10 reads at >= 90 % of array CpGs). Nothing is filled in; an unmeasured (cell, locus) is NOT MEASURED. GSE262275 liver/bile cells (27–59 % coverage) are held out for a later experiment. PROC-PARTIALCOV-01 not adopted. | "I'd rather not have a cell than only have part of one because I know what kind of headache that causes later." / "that is exactly how i ended up with misplaced, misnamed, and duplicate cells the first time." |
| D10 | 2026-09-28 | Judge the atlas by whether its cells come out distinct, never by R-hat alone (the v1 flatness lesson) — a pass/fail gate on the finished run. | [`IAMAtlas_FLATNESS_LESSON_v1.md`](IAMAtlas_FLATNESS_LESSON_v1.md) |
| D11 | 2026-09-28 | Keep 20 posterior draws per cell per locus and the per-locus prior, so later tools carry the atlas's uncertainty and a new cell can be appended without refitting. | agreed ("Awesome!") |
| D12 | 2026-09-28 | Compute: as many cores as possible. | "money isnt a problem…my time is more valuable. Lets get as many cores running as we can" |
| D13 | 2026-09-28 | A specimen is any vial the methylome can be read from; the atlas holds one cell type per entry. Tissue profiles are test specimens, not references. | "A cell is a cell is a cell"; "Every sample or specimen or vial is a means to the methylome we seek." |
| D14 | 2026-09-28 | Report every difference also in A terms, and say when a number does not touch A. | the author's request after conflating floor gaps with beta offsets |
