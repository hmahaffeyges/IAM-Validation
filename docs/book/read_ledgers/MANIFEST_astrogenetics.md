# MANIFEST — Astro-Genetics chapter (base HEAD 5f5997b)

## What this delivers
- NEW chapter `docs/book/part4/p4_00b_astrogenetics.tex` (407 lines), `\chapter{Astro-Genetics: the cell read as a thermometer}\label{ch:astrogenetics}`.
  **main.tex**: insert `\input{part4/p4_00b_astrogenetics}` directly AFTER line 60 `\input{part4/p4_01_bridge}` (Block B1), so it is the second chapter of Part 4.
- NEW bib `docs/book/bib_astrogenetics.bib` (1 entry: Warburg1956, DOI 10.1126/science.123.3191.309, CrossRef-checked 2026-10-03: "On the Origin of Cancer Cells", Science 123(3191) 309-314, 1956).
  main.tex line 119: `\bibliography{iam}` -> add `bib_astrogenetics` to the list (with the other wave bib files).
- NEW `docs/verification/scripts/verify_astrogenetics_book.py` + `_output.txt`: recomputes every number in the chapter and blocks (38 checks, 0 FAIL) from the records
  `Biological_Physics/MethylPhys/doors/PROC_TUMOUR_01_OUTCOME.md`, `PROC_DNMT_01_PARTB/dnmt_b_pairs.csv`, `PROC_LINES_02_channels/imr90_channels.csv`; constants traced to their book lines in comments.
- Six insertion blocks (B1-B6) below for main.tex, p4_01_bridge, p4_16_sky, p4_20_report, p3_08_one_gauge.

## Sources read (line counts confirmed first, then read in chunks of <= 50 lines)
| source | lines | read |
|---|---|---|
| What_Is_Astro_Genetics.tex | 389 | 1-389 in full (1-50, 51-100, 101-150, 151-200, 201-250, 251-300, 301-345, 346-389) |
| astro_thermometer.txt (text of 1_What_is_Astro_Genetics__Cellular_Thermometer.pdf, 23 pp) | 839 (838 newlines) | 1-839 in full (17 chunks of 50, last 801-839) |
| What_Is_Astro_Genetics.pdf (23 pp) | 838 (own extraction) | NOT read line by line: its letter stream was compared with the thermometer PDF and is identical except 3 'ffi' ligature extractions; it is the same paper |
Both PDFs carry the same text as the .tex; the PDF text adds only the text inside the figures (two ledgers, scale strip, cell gauge, cosmic gauge, toolkit, scorecard), mapped below.
Book files read in full for consistency: p4_01_bridge (181), p4_06_gauge (138), p4_20_report (95), p4_16_sky (90), p3_08_one_gauge (157), CELL_ITEMS_VERDICTS.md (42); parts of p4_14_atlas, p4_24_status, p2_01_blackholes, p4_11_translation, p4_16a_skytools, p2_12_lambda, p2_13b_baryon_chain, p3_09_reach, p4_23_reach (grep + context).

## Insertion blocks
Anchors are whole lines copied from HEAD 5f5997b; each occurs exactly once in its file (checked with `grep -cxF`).

### B1 — docs/book/main.tex | AFTER | `\input{part4/p4_01_bridge}`
```latex
\input{part4/p4_00b_astrogenetics}
```

### B2 — docs/book/part4/p4_01_bridge.tex | AFTER | `interpretation. \interp`  (line 78)
Source lines 96-106 and Fig. D (112-117). Adds Table tab:p4_twoledgers. Overlaps in theme with the bridge's own paragraph at l.73-78; merge the two paragraphs if preferred.
```latex

\paragraph{The receipt for every action.} There is an old idea, older than physics, that the world runs on balance: when something gains,
something gives. Modern physics gave it a name that hides how simple it is, the virial theorem: in a stable system bound by a $1/r$ force
the energy of motion is half the binding energy (Chapter~\ref{ch:virial_law}). \derived{} The framework takes the position that the same
accounting applies across roughly thirty-seven orders of magnitude, from a single atom to the cosmic horizon, not because something
enforces it but because a configuration that cannot pay its costs at finite price does not persist. What we see when we look around, from
galaxies to cells, is what the accounting allows to exist; everything that could not pay has already come and gone. \conjecture

Here is the part that connects the cosmos to the clinic. Every time a physical system does something irreversible, it pays a cost in
information. This is not a metaphor; it is Landauer's principle, Eq.~\eqref{eq:landauer}. \derived{} When a star shines, that cost is paid,
and in this reading it is paid in two halves: one half shows up as motion, which we observe as the star's heat and light; the other half as
the way the star bends space around itself, which we feel as gravity. A cell does the same thing. Every time it reads its genes, builds
proteins or divides, it pays the same kind of cost, split into the same two halves: one half is the cell's metabolism, the moment-to-moment
work; the other half is written into its DNA, not into the genes themselves but into the methyl groups placed at specific sites along the
strand (Table~\ref{tab:p4_twoledgers}). \conjecture{} The methylation pattern is the cell's running record of what it has been doing and
what it is meant to do next. In other words, the cell uses its DNA as a notebook, and writes the receipt for every action. The pattern of
receipts in a healthy cell looks one way; the pattern in a cell moving toward disease looks different. Reading those patterns is what this
Part does. \interp{} The balance that keeps a star from collapsing is, in the same reading, the balance that keeps a cell healthy. When a
star's core passes its limiting mass, or a cell's pattern moves too far from its healthy reading, the system has run out of capacity to
keep paying its costs in the normal way; we call the stellar failure a black hole and, in the cell, the regime where cancer lives. We did
not invent this rule; we learned to read it. \conjecture

\begin{table}[htbp]\centering\small
\caption{Two ledgers, one law: every irreversible event pays at least $\kB T\ln2$ per bit at the temperature of the surface written on. The split of
the cost into a kinetic half and a record half is exact for $1/r$-bound systems (Chapter~\ref{ch:virial_law}); in the cell it is the
picture of this Part, not a measurement (Chapter~\ref{ch:ledgers}). \conjecture}\label{tab:p4_twoledgers}
\begin{tabular}{@{}lll@{}}\toprule
 & a star & a cell \\\midrule
kinetic half & heat and light (observed luminosity) & metabolism (ATP, transcription) \\
record half & spacetime curvature (what we feel as gravity) & the methylation pattern (written into the DNA) \\
cost per bit & $\kB T\ln2$ & $\kB T\ln2$ at 310~K \\\bottomrule
\end{tabular}
\end{table}
```

### B3 — docs/book/part4/p4_01_bridge.tex | BEFORE | `Chapter~\ref{ch:landauer} puts numbers on the cost at body temperature and defines the Mahaffey`  (line 174)
```latex
Chapter~\ref{ch:astrogenetics} says the same for geneticists and clinicians in plain language: what the instrument reads, what a reading
does not mean, and how the toolkit moves to other questions in genetics.
```

### B4 — docs/book/part4/p4_16_sky.tex | AFTER | `$z$ centred on zero with no structure. A cell far from its healthy state, or a cell the reference does not hold, leaves structure.`  (line 7)
Source line 251 (brightening).
```latex

\paragraph{Brightening.} The reference for the cells that come next is made the way an astronomer makes a map of the microwave background,
and we call the step \emph{brightening}. The raw microwave map is noisy and unevenly covered; the analysis produces a posterior over the
true underlying field at every point on the sky, using the known statistical structure of the field, and faint, noisy pixels resolve into
a clean map you can read structure from. The hierarchical fit of atlas v2 does the same to the methylation reference, at every value a cell
was measured, with one difference the cell demands: where a cell was never measured, the value is left blank, not filled
(Chapter~\ref{ch:atlas}). Two sky maps, one of the early universe and one of the healthy cell, sharpened by the same inference. \analogy
```

### B5 — docs/book/part4/p4_20_report.tex | AFTER | `\derived{} for what the arithmetic contains; the rest is the rule of the report.`  (line 69)
Source line 226 (short form; full form in the new chapter).
```latex
The honest reading of a reading above Normal is a prompt for a clinician, not a verdict for a patient: the instrument flags and refers; it
does not diagnose. Chapter~\ref{ch:astrogenetics} sets out in plain language what such a reading does and does not mean, and why a benign
growth is expected to read near 1 (Section~\ref{sec:ag_benign}).
```

### B6 — docs/book/part3/p3_08_one_gauge.tex | AFTER | `band is less error than healthy; for cells what it means biologically has not yet been measured.`  (line 44)
Source lines 157-161 and 195.
```latex

\paragraph{The compartment, not the average.} A point of method carries over directly from the cell to the star. The cell instrument does
not score the whole person; it scores each cell type on its own. The star gauge does the same: it reads the core, not the whole star. A red
supergiant such as Betelgeuse is enormous and diffuse, so its overall structure looks unremarkable on any whole-body measure, exactly as a
person can look well while one cell type is in trouble. The signal is in the compartment, not the average. The supernova is a core event.
\analogy{} Stellar astrophysics already predicts these transitions with great precision by its own mature methods. The star gauge is not
offered as a new way to predict stars; it reproduces what astronomy already knows. Its purpose is to show that the cell and the star are
read by one instrument. The predictive novelty is on the biological side, where no such gauge existed before. \interp
```

## Carriage table — What_Is_Astro_Genetics.tex (source line -> book file:line, or EXCLUDED with the rule)
Book line numbers for the new chapter are lines of `docs/book/part4/p4_00b_astrogenetics.tex` in this zip.
| source lines | destination | note |
|---|---|---|
| 1-50 | EXCLUDED | front matter: title, author name, institute, e-mail, Zenodo DOI (stand-alone book rule) |
| 52-70 (abstract) | EXCLUDED | abstract: content carried where its body sections are carried; 0.36 %, 0.07 %, 27 of 28 TCGA, the methylation report/the cell-reading engine are retired (see 271-334) |
| 73-75 | carried part4/p4_00b_astrogenetics.tex:7-12 | "informed patient" dropped (no patient-facing text); "cosmology in this paper has been checked" -> "the tools have been checked" (IAM cosmology itself is not checked by the field) |
| 78-80 | carried part4/p4_00b_astrogenetics.tex:16-20 | "accounting is the same" given its derivation (k_BT ln2) and label |
| 83-85 | IN BOOK part4/p4_01_bridge.tex:14-18 | cartography/stargazers already carried; institute name EXCLUDED (name rule); "healthy floor of each cell class" retired (bridge carries the fixed wording) |
| 87 | carried part4/p4_00b_astrogenetics.tex:25 |  |
| 89 | carried part4/p4_00b_astrogenetics.tex:28-36 | "architectural drift signatures" -> "the methylation pattern" (class era) |
| 91 | carried part4/p4_00b_astrogenetics.tex:38-44 | "fixed multiple" restated as measured 3.41 k_BT = 4.9 Landauer units; "floor crossings"/"saturate" -> departures from the healthy reading (floor is on the left) |
| 93 | IN BOOK part4/p4_01_bridge.tex:17-18 |  |
| 96-98 | BLOCK B2 (p4_01_bridge after "interpretation. \interp") | "Greeks/Aristotle twenty-five centuries" dated history dropped per verdict; "thought to apply only to gravity" is false (Clausius, Fock) - dropped |
| 100 | BLOCK B2 | retired name of the balance rule EXCLUDED |
| 102 | BLOCK B2 | two halves labelled CONJECTURE (virial ruling) |
| 104 | BLOCK B2 (receipts)  | the methylation report/the cell-reading engine/product sentence EXCLUDED (names) |
| 106 | BLOCK B2 | labelled CONJECTURE |
| 108 | carried part4/p4_00b_astrogenetics.tex:48-56 | Bekenstein-bound saturation cited (Bekenstein1981); "identical threshold" labelled INTERPRETATION |
| 110 | carried part4/p4_00b_astrogenetics.tex:58-75 | full surface restated on current gauge (3.03, 4.45); tumour copy-error result added as the measured case; "black hole and tumour same event" CONJECTURE |
| 112-117 (Fig. D two ledgers) | BLOCK B2 table tab:p4_twoledgers | figure content as a table; CONJECTURE |
| 119-124 (Fig. F scale strip) | EXCLUDED | rows retired or errata: electron 6.6 ppm (H0 artefact), N-body 0.815 row not carried, "17 chains 0.3 %" is an identity; the current atom-to-cluster table is in ch:virial_law |
| 127-135 | carried part4/p4_00b_astrogenetics.tex:79-82 | A redefined on current method (reading/healthy reference); "floor for that class" and "breach line" retired; breach "to be measured" |
| 137-145 | carried part4/p4_00b_astrogenetics.tex:97-104 | class H_min, eight architectural classes, five-tier vocabulary, "fuel gauge" EXCLUDED (retired; report says "not a fuel gauge"); keybox "Measured against the cell's own healthy state" part4/p4_00b_astrogenetics.tex:106 added per task |
| 147-152 (Fig. B cell gauge) | EXCLUDED (replaced) | tiers 1.07/1.10, class floors; current gauge is fig:gauge (ch:gauge) and the fixed-points table part4/p4_00b_astrogenetics.tex:87 |
| 154-161 | BLOCK B6 (p3_08_one_gauge after line 44) | compartment/Betelgeuse paragraph; "most-loaded compartment" -> "each cell type on its own"; star gauge restated part4/p4_00b_astrogenetics.tex:115 |
| 163 | EXCLUDED | rescaling of the star ratio onto the cellular breach line (retired; book keeps the load ratio on its own axis, p4_11_translation.tex:121-123) |
| 165-184 (Table 1 stellar gauge) | EXCLUDED (replaced) | rescaled to 1.10 breach and tiers; current star values in p2_01_blackholes.tex:322-330 (fig:stargauge); cap logic IN BOOK p4_11_translation.tex:117-123 |
| 186-191 (Fig. C cosmic gauge) | EXCLUDED (replaced) | same reason; fig:stargauge is the current figure |
| 193-195 | BLOCK B6 |  |
| 198-200 | carried part4/p4_00b_astrogenetics.tex:122-125 | "flags and refers; it does not diagnose" added (task rule) |
| 202-204 | carried part4/p4_00b_astrogenetics.tex:129-133 | "architectural class" -> own cell type; "grade" kept as CONJECTURE, "its own axis" per report chapter |
| 206-208 | EXCLUDED | "three levels" are tier lines (retired) |
| 210 | EXCLUDED | recoverable range 1.0-1.07 (tier) and lifestyle/metabolic/hormonal/nutritional intervention (advice rule) |
| 212 | carried physics only part4/p4_00b_astrogenetics.tex:146-149 | 1.07 line, supplementation, "holistic strategy" EXCLUDED (tier + advice); Warburg effect stated as physics with Warburg1956 (DOI checked), place on gauge OPEN |
| 214 | carried restated part4/p4_00b_astrogenetics.tex:137-142 | 1.10 breach and senescent 1.24-1.27 / cancer 1.28-1.32 EXCLUDED (class era); breach restated "to be measured" part4/p4_00b_astrogenetics.tex:93; current IMR90 channel readings replace the senescent claim |
| 216-218 | carried part4/p4_00b_astrogenetics.tex:153-154 | "breach" -> "a reading above Normal" |
| 220 | carried part4/p4_00b_astrogenetics.tex:157-159 | "cell-free DNA" -> "a specimen's DNA" (chain v3 reads whole blood) |
| 222 | carried part4/p4_00b_astrogenetics.tex:161-162 | "same breach threshold crossed by senescent cells" restated: senescent cells read away from 1 (measured, below Normal) |
| 224 | carried part4/p4_00b_astrogenetics.tex:164 |  |
| 226 | carried part4/p4_00b_astrogenetics.tex:167-169 | also BLOCK B5 (p4_20_report) short form |
| 228-230 | carried part4/p4_00b_astrogenetics.tex:173-177 | PREDICTION, untested |
| 232 | carried restated part4/p4_00b_astrogenetics.tex:179-184 | colorectal/breast A values (0.983, 1.037, 1.050, 1.069, 1.147, 1.045, 1.097) EXCLUDED (class-era readings); ordering kept as PREDICTION; tissue series named test OPEN |
| 234 | carried part4/p4_00b_astrogenetics.tex:186-190 | "correlates strongly" -> "may correlate" (untested); colonoscopy miss-rate clause EXCLUDED (unsourced clinical claim); shape-independence moved to the PREDICTION paragraph |
| 237 | EXCLUDED | \label{sec:iamatlas} without a section (source markup) |
| 239 | carried part4/p4_00b_astrogenetics.tex:194-199 | "healthy floor" -> "floor" |
| 241 | EXCLUDED | two-phase history: "confirmed that the physics applied" was a class-era claim; clinical-grade and commercial sentences (commercial/name rules) |
| 243 | carried restated part4/p4_00b_astrogenetics.tex:201-204 | IAMAtlas name and "eight architectural classes" EXCLUDED; current reference (purified neutrophils) and atlas v2 stated |
| 245 | carried part4/p4_00b_astrogenetics.tex:204-210 | "borrowing strength from the class structure" and "fully populated map" corrected: pulled to the locus level only; unmeasured values blank, not filled (ch:atlas) |
| 247 | carried part4/p4_00b_astrogenetics.tex:212-219 | "recoverable range, metabolic transition, breach line" (tiers) -> three parts of a reading's interval (ch:gauge) |
| 249 | carried part4/p4_00b_astrogenetics.tex:221-225 | "two and a half weeks" EXCLUDED (not traceable to a record); 814,000 loci / 700 blocks from ch:atlas |
| 251 | BLOCK B4 (p4_16_sky after line 7) | brightening; blank-not-filled correction |
| 253 | carried part4/p4_00b_astrogenetics.tex:227-234 | IAMAtlas -> "the atlas"; "physics-derived floors" -> "the floor"; DESI spelled out |
| 255-257 | carried part4/p4_00b_astrogenetics.tex:238-245 | "floor for each architectural class derived" -> floor form derived, reading against the cell's own healthy state |
| 259 | carried part4/p4_00b_astrogenetics.tex:247-262 | IMR90 measured case added; timing claim labelled CONJECTURE + OPEN (untested) |
| 261 | carried in part part4/p4_00b_astrogenetics.tex:264-272 | 1961 Bell Labs / Penzias detail IN BOOK p4_16a_skytools.tex:23-26 and p0_giants.tex:134-136; EPIC-Italy immune-class signal ten years before diagnosis and "recovers signal analyses left behind" EXCLUDED (cohort evidence, class-era chain result); prospective one-person test kept as OPEN |
| 263-268 (Fig. G toolkit) | carried as table part4/p4_00b_astrogenetics.tex:319-331 | IAMAtlas and class floor wording replaced |
| 271-273 | carried part4/p4_00b_astrogenetics.tex:287-290 | "blind" -> "a check made in the right order"; pre-registration hashes |
| 275-283 | carried restated part4/p4_00b_astrogenetics.tex:292-298 | eta 6.115+/-0.037 vs 6.137 (0.36 %) and Yeh 2026 EXCLUDED as printed; current chain value 6.113+/-0.037 (ch:baryon_chain), MEASURED, constraint OPEN |
| 285-289 | carried restated part4/p4_00b_astrogenetics.tex:298-300 | 1.141 vs 1.14 (0.07 %, "no free parameters") -> 1.142 vs 1.133 (+0.79 %), FITTED (ch:lambda) |
| 291 | carried in part part4/p4_00b_astrogenetics.tex:293 | 0.1575 -> canon 0.15765; "seventeen-chain posterior 0.1583+/-0.0033, 0.24 sigma" EXCLUDED (Omega_m/2 of the posterior is an identity, not a test) |
| 293-295 | carried restated part4/p4_00b_astrogenetics.tex:303-317 | class floors frozen 6 Apr 2026 and 27 of 28 TCGA EXCLUDED (retired class floors); current per-person tests: DNMT1 8/8 IAM-A 1.65-1.97, tumour 6/6 median 1.148, oral 4/4, IMR90 |
| 297-306 (lifespan, Fig. 6) | EXCLUDED | A = H/H_min(class) across 34 species and the A = 1.05 line: class floors, tier line and a cross-species comparison (retired; MEASURE NOT COMPARE) |
| 308-310 | carried restated part4/p4_00b_astrogenetics.tex:300-301 | "one correction, two results" -> one observed relation read in two variables; "pinned blind" and the MPHYS-in-a-cell parallel EXCLUDED (not true on current method; name rule) |
| 312-327 (Table 2 scorecard) | EXCLUDED (replaced) | values restated in section sec:ag_both on current method |
| 329-334 (Fig. E scorecard) | EXCLUDED (replaced) | same |
| 337-339 | carried in part part4/p4_00b_astrogenetics.tex:335 | first three sentences IN BOOK p4_16a_skytools.tex:11-15 |
| 341 | carried part4/p4_00b_astrogenetics.tex:336-341 | "seals every prediction with a public timestamp" -> pass conditions written down with a hash; IAMAtlas EXCLUDED; last two sentences IN BOOK p4_16a_skytools.tex:14-15 |
| 344-348 | carried part4/p4_00b_astrogenetics.tex:345-359 | "one bit per four Planck areas" corrected to one k_B (one nat) per 4 l_P^2, one bit 4 ln2 l_P^2 (ch:surfaces); "Einstein's equations not fundamental" -> "follow as an equation of state"; floor height OPEN |
| 350 | EXCLUDED | names and paraphrases private correspondents (no-correspondent rule) |
| 353-355 | IN BOOK part4/p4_01_bridge.tex:30-32 | definition-of-life open question already carried |
| 358-360 | carried part4/p4_00b_astrogenetics.tex:387-391 | "lifestyle, supplementation, hormone therapy, diet, exercise, sleep, and stress" EXCLUDED (advice rule); trajectory OPEN (serial change floor not measured) |
| 362 | carried part4/p4_00b_astrogenetics.tex:394-398 | PREDICTION; "breach line set by physics" -> measured on per-person data; "supplement protocol" EXCLUDED (advice); "reserve" -> room before the full surface |
| 364-366 | carried part4/p4_00b_astrogenetics.tex:400-403 | "structural identity demonstrated" -> derived for the cost, analogy for the rest (ch:floorbreach); the cell-reading engine EXCLUDED (name) |
| 369-371 | carried part4/p4_00b_astrogenetics.tex:405-407 | repository pointer kept; "every prediction predates the data... years before" restated as pre-registrations with hashes |
| 374-387 (references) | bib keys | Planck2018VI, Bekenstein1973, Hawking1975, Landauer1961, Jacobson1995, CaiKim2005, Zurek2003 cited; Yeh 2026 EXCLUDED (comparator not used on current method); Peebles 1980 EXCLUDED (cited only for correspondence); repository/Zenodo item EXCLUDED (own record) |
| task: two jobs | carried part4/p4_00b_astrogenetics.tex:276-283 | not in either source paper; carried from the operations-manual rule L-5 (own words): "Physics measures; groups of people only point... enters only as a DIRECTION, never as a baseline"; plus THE TEST wording |
| task: transfer to genetics | carried part4/p4_00b_astrogenetics.tex:363-382 | not a section of either source; composed from source 253 (source-agnostic), 339-341 and current book items (p3_09_reach.tex:60-80, ch:temperature, ch:sky); every item labelled |

## PDF text (astro_thermometer.txt, same paper) -> tex line -> carriage row
| txt lines | matches | carriage |
|---|---|---|
| 1-9 | tex 3-41 | see source row 1-50 |
| 10-11 | tex 386 | see source row 374-387 (references) |
| 12-30 | tex 53-66 | see source row 52-70 (abstract) |
| 31 | page number | EXCLUDED: page footer |
| 32-34 | tex 66 | see source row 52-70 (abstract) |
| 35-47 | tex 73-75 | see source row 73-75 |
| 48-55 | tex 78-80 | see source row 78-80 |
| 56-64 | tex 83-85 | see source row 83-85 |
| 65 | page number | EXCLUDED: page footer |
| 66-67 | tex 87 | see source row 87 |
| 68-78 | tex 89 | see source row 89 |
| 79-85 | tex 91 | see source row 91 |
| 86-88 | tex 93 | see source row 93 |
| 89-94 | tex 96-98 | see source row 96-98 |
| 95-96 | tex 100 | see source row 100 |
| 97 | page number | EXCLUDED: page footer |
| 98-103 | tex 100 | see source row 100 |
| 104-116 | tex 102 | see source row 102 |
| 117-123 | tex 104 | see source row 104 |
| 124-129 | tex 106 | see source row 106 |
| 130-136 | tex 108 | see source row 108 |
| 137 | page number | EXCLUDED: page footer |
| 138-142 | tex 108 | see source row 108 |
| 143-155 | tex 110 | see source row 110 |
| 156-187 | tex 112-115 | see source row 112-117 (Fig. D two ledgers) |
| 188 | page number | EXCLUDED: page footer |
| 189-206 | tex 119-122 | see source row 119-124 (Fig. F scale strip) |
| 207-216 | tex 127-135 | see source row 127-135 |
| 217-224 | tex 137-139 | see source row 137-145 |
| 225 | page number | EXCLUDED: page footer |
| 226-230 | tex 145 | see source row 137-145 |
| 231-253 | tex 147-150 | see source row 147-152 (Fig. B cell gauge) |
| 254-267 | tex 154-161 | see source row 154-161 |
| 268-271 | tex 163 | see source row 163 |
| 272 | page number | EXCLUDED: page footer |
| 273-273 | tex 163 | see source row 163 |
| 274-311 | tex 165-182 | see source row 165-184 (Table 1 stellar gauge) |
| 312 | page number | EXCLUDED: page footer |
| 313-358 | tex 186-189 | see source row 186-191 (Fig. C cosmic gauge) |
| 359-364 | tex 193-195 | see source row 193-195 |
| 365-371 | tex 198-200 | see source row 198-200 |
| 372 | page number | EXCLUDED: page footer |
| 373-380 | tex 202-204 | see source row 202-204 |
| 381-383 | tex 206-208 | see source row 206-208 |
| 384-388 | tex 210 | see source row 210 |
| 389-398 | tex 212 | see source row 212 |
| 399-405 | tex 214 | see source row 214 |
| 406 | page number | EXCLUDED: page footer |
| 407-410 | tex 216-218 | see source row 216-218 |
| 411-415 | tex 220 | see source row 220 |
| 416-419 | tex 222 | see source row 222 |
| 420-423 | tex 224 | see source row 224 |
| 424-425 | tex 226 | see source row 226 |
| 426-433 | tex 228-230 | see source row 228-230 |
| 434-439 | tex 232 | see source row 232 |
| 440 | page number | EXCLUDED: page footer |
| 441-443 | tex 232 | see source row 232 |
| 444-453 | tex 234 | see source row 234 |
| 454-461 | tex 239 | see source row 239 |
| 462-475 | tex 241 | see source row 241 |
| 476-477 | tex 243 | see source row 243 |
| 478 | page number | EXCLUDED: page footer |
| 479-484 | tex 243 | see source row 243 |
| 485-493 | tex 245 | see source row 245 |
| 494-505 | tex 247 | see source row 247 |
| 506-513 | tex 249 | see source row 249 |
| 514 | page number | EXCLUDED: page footer |
| 515-522 | tex 251 | see source row 251 |
| 523-537 | tex 253 | see source row 253 |
| 538-545 | tex 255-257 | see source row 255-257 |
| 546-548 | tex 259 | see source row 259 |
| 549 | page number | EXCLUDED: page footer |
| 550-563 | tex 259 | see source row 259 |
| 564-585 | tex 261 | see source row 261 |
| 586 | page number | EXCLUDED: page footer |
| 587-592 | tex 261 | see source row 261 |
| 593-625 | tex 263-266 | see source row 263-268 (Fig. G toolkit) |
| 626-631 | tex 271-273 | see source row 271-273 |
| 632-633 | tex 275-277 | see source row 275-283 |
| 634 | page number | EXCLUDED: page footer |
| 635-645 | tex 279-283 | see source row 275-283 |
| 646-646 | tex 377 | see source row 374-387 (references) |
| 647-652 | tex 285 | see source row 285-289 |
| 653-654 | tex 320 | see source row 312-327 (Table 2 scorecard) |
| 655-655 | tex 289 | see source row 285-289 |
| 656-659 | tex 291 | see source row 291 |
| 660-667 | tex 293-295 | see source row 293-295 |
| 668-671 | tex 297-299 | see source row 297-306 (lifespan, Fig. 6) |
| 672 | page number | EXCLUDED: page footer |
| 673-682 | tex 299-304 | see source row 297-306 (lifespan, Fig. 6) |
| 683-691 | tex 308-310 | see source row 308-310 |
| 692 | page number | EXCLUDED: page footer |
| 693-693 | tex 310 | see source row 308-310 |
| 694-702 | tex 317-325 | see source row 312-327 (Table 2 scorecard) |
| 703-713 | tex 329-332 | see source row 329-334 (Fig. E scorecard) |
| 714-719 | tex 337-339 | see source row 337-339 |
| 720 | page number | EXCLUDED: page footer |
| 721-725 | tex 339 | see source row 337-339 |
| 726-736 | tex 341 | see source row 341 |
| 737-756 | tex 344-348 | see source row 344-348 |
| 757 | page number | EXCLUDED: page footer |
| 758-758 | tex 348 | see source row 344-348 |
| 759-773 | tex 350 | see source row 350 |
| 774-782 | tex 353-355 | see source row 353-355 |
| 783-789 | tex 358-360 | see source row 358-360 |
| 790 | page number | EXCLUDED: page footer |
| 791-794 | tex 360 | see source row 358-360 |
| 795-802 | tex 362 | see source row 362 |
| 803-810 | tex 364-366 | see source row 364-366 |
| 811-815 | tex 369-371 | see source row 369-371 |
| 816-817 | page number | EXCLUDED: page footer |
| 818-821 | tex 377-378 | see source row 374-387 (references) |
| 822-822 | tex 283 | see source row 275-283 |
| 823-838 | tex 379-386 | see source row 374-387 (references) |
| 839 | page number | EXCLUDED: page footer |

## Static checks (run on a copy of HEAD with B1-B6 applied and the chapter added; no TeX compile)
- braces balanced, \begin/\end matched, `$` even: new chapter and all five touched files, and every block: PASS.
- labels unique across all 98 \input files (1,328 labels): PASS (new labels ch:astrogenetics, sec:ag_who, sec:ag_one, sec:ag_two, sec:ag_surface, sec:ag_gauge, sec:ag_read, sec:ag_benign, sec:ag_reference, sec:ag_physics, sec:ag_twojobs, sec:ag_both, sec:ag_name, sec:ag_lineage, sec:ag_transfer, sec:ag_for, tab:ag_toolkit, tab:p4_twoledgers).
- every \ref/\eqref in the new chapter and blocks resolves (refs used: ch:bridge, ch:quantumrecords, ch:skytools, ch:landauer, ch:blackholes, ch:floorbreach, part4:ch:reach, eq:A, ch:gauge, ch:meta, ch:iama, ch:onegauge, ch:separation, fig:stargauge, sec:p4notmean, ch:ledgers, ch:report, ch:atlas, ch:serial, ch:discipline, ch:virial, ch:baryon_chain, ch:lambda, ch:firstreadings, ch:sky, ch:surfaces, ch:iams_law, ch:open, ch:reach, ch:temperature, ch:salmonid, ch:virial_law, eq:landauer, sec:ag_benign, ch:astrogenetics). The only unresolved refs in touched files are part:2/3/4, which are defined in main.tex itself (not an \input file).
- every \cite key present in iam.bib or bib_astrogenetics.bib: PASS (Zurek2009, Bekenstein1981, Warburg1956, Horvath2013, AlpherHerman1948, PenziasWilson1965, Dicke1965, Jacobson1995, CaiKim2005, Bekenstein1973, Hawking1975, Zurek2003, Landauer1961).
- retired-term scan of chapter and blocks (class floors, tiers, 1.10, 1.07, the methylation report/the cell-reading engine/the quantum-processor report/the semiconductor report, Aristotelian, IAMAtlas, TCGA, 0.36 %, cohort, percentile, AUC, lifestyle, supplement, fuel gauge, 'the author', own name): no hit except 'class' in the never-pool rule and 'population' in negations.

## Open items for the lead
1. `figures/part3/one_gauge.pdf/.png` (fig:onegauge, p3_08) still draws **'breach 1.10 (provisional)'** on the cell row: a retired tier line on a book figure. Regenerate without it ('breach: to be measured').
2. p3_08_one_gauge.tex l.78-80 still describes the noise-corrected tare regression (A = a + b f_neu + c N, >= 20 references), which was removed from chain v3 (median same-run tare only). The new chapter states the median tare only.
3. 'Two jobs' (sec:ag_twojobs) and 'Where the toolkit goes next in genetics' (sec:ag_transfer) are NOT passages of either source paper. Two jobs is carried from the operations-manual rule L-5 and THE TEST (own words); the transfer section is composed from source lines 253/339-341 and current book items. Author to confirm wording.
4. Warburg: the paper gives no physics of the Warburg effect, only a 1.07 tier with metabolic advice (both excluded). The chapter carries one physics-only statement (Warburg1956) with its place on the gauge OPEN. Drop it if the author prefers nothing.
5. The colonoscopy miss-rate clause (source 234) was dropped as an unsourced clinical claim; re-add only with a primary source.
6. B2 overlaps the bridge's existing 'Why your cells follow the same law as the stars' paragraph (l.73-78); merge at apply time if it reads twice.
7. Lineage paragraph (source 348): 'one bit per four Planck areas' corrected to one k_B per 4 l_P^2 (one bit 4 ln2 l_P^2), as in p1_01_encoding_surfaces.tex:38-40.
8. No TeX compile was run (static checks only).
