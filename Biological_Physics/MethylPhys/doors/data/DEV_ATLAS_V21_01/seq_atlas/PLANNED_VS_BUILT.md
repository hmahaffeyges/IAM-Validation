# The IAM Sequencing Atlas: planned against built

Every row of the atlas's checklist (`SEQ_ATLAS_CHECKLIST.md`) with four questions: is it built (from `CHECKLIST_STATUS.md`); does the chain use it
when it reads a sequenced sample (from `run_sample.py` and `stage_seq_atlas.py`: a tool called only by a `--build` step is a picture or a check, not
a reading input); is it on the website; is it in the book's sequencing-atlas chapter (`docs/book/part6/p6_14c_seq_atlas.tex`). Plan item 61.3.

| Row | Planned | Built | Used when the chain reads a sample | Website | Book |
|---|---|---|---|---|---|
| A1 | Each donor's read counts, never averaged away | Partly: at the identity and island sites, not every CpG | Yes: Met-A donor by donor, the islands from the reference's own reads | — | — |
| A2 | Each donor's measured disorder | Yes | Yes: the expected disorder Met-A reads against | Cell pages | Chapter |
| A3 | Level, donor spread, donors, reads per cell | Yes | Yes: the Deconvolver's columns, the globe's reference | Cell pages (numbers) | Chapter |
| A4 | Blank where unmeasured, counts beside every value | Yes | Yes: readings use only measured sites | — | Chapter |
| A5 | Each laboratory's offset per cell | Measured on BLUEPRINT only; not measurable inside the atlas (one laboratory per cell) | Replaced by the same-laboratory tare (development, under test) | — | Chapter (the floor) |
| A6 | The molecules themselves (read patterns) | Yes (held_reference_v1) | Yes: IAM-A in the sample's own mix, Coupling-A | — | Chapter |
| A7 | Donor traits | Yes | Annotation only, by design (foregrounds annotated, never subtracted) | Cell pages (sex) | — |
| A8 | 16-bit counts | Yes | Yes | — | — |
| B1–B9 | A healthy reference for every sequencing instrument | Yes, all nine | Yes: every reading on a sequenced sample (pass 5: all Normal) | Catalogue cards (in development) | Chapter |
| C1 | Every CpG on the sky sphere | Yes (57 cells, level and 95 % range) | Yes: a sequenced sample's globe is read against it | Globe Explorer, 57 cell pages | Chapter |
| C2 | Information sphere per cell, with its distance | Built as a picture (the information globe) and as each sample's distance from its own healthy self | Partly: a sequenced sample's information sphere still uses IAM-Atlas v2.1's layout of 191,458 sites (35.30.12 open) | Information globe page | One mention |
| C3 | Noise maps for both spheres | Yes, on the sky sphere | As a check that the 95 % ranges are right (nulls met in 56 of 56); not an input to any reading | Explorer and cell pages | Chapter |
| C4 | Three spheres compared (v2, v2.1, sequencing) | Against v2.1 (the bridge); not possible for v2 (not in the repository) | Platform offset per cell recorded | — | Chapter |
| D1 | Sealed donor halves | Yes | Yes: self-free healthy tests read against the other half | — | Chapter |
| D2 | Read halves | Yes | For the noise maps (a check) | — | Chapter |
| D3 | Noise maps per cell | Yes | Check (as C3) | Explorer | Chapter |
| D4 | 95 % range per value | Yes (synthetic version 3 met) | Yes: the globe's width at each site | Cell pages | Chapter |
| D5 | Masks and apodization | Yes | For the sky statistics only; readings checked without the sex chromosomes (no effect) | — | Chapter |
| D6 | Genome-distance spectrum | Yes | No reading uses it yet | — | Chapter |
| D7 | Cell-count floor (copies per site) | Yes (synthetic met) | Yes: an in-mix reading below one copy is flagged | — | Chapter |
| D8 | Covariance of healthy sites | Yes, from the spectrum | No reading uses it yet; it is what a whole-genome C-Score on sequencing needs (new 60.22) | — | Chapter (as the spectrum's correlation by distance) |
| D9 | Transfer function | Yes, at the islands (slope 1.0001) | Check | — | Chapter |
| D10 | Null tests | Yes | Check | — | Chapter |
| D11 | Bisulfite against enzymatic; bridge to v2.1 | Bridge yes; enzymatic not yet (no enzymatic inputs) | Bridge recorded per cell | — | Chapter |
| D12 | 5mC against 5hmC | Not possible: no oxidative data in the inputs | — | — | — |
| D13 | Minkowski functionals | Synthetic met; real maps not yet readable | No | — | Chapter |
| D14 | Isotropy | Yes | Check | — | Chapter |
| D15 | Blinding and held-out samples | Yes | Yes: outside healthy samples kept unread by the build | — | Chapter |
| D16 | Reproducibility, DOI | Yes (Zenodo) | — | DOI on the page | Chapter |
| D17 | The guidebook | Yes | — | Linked | Chapter |
| F1 | Fragment features (nucleosomes, WPS, DELFI) | Not built: needs healthy plasma read Normal first (Phase B) | — | — | — |

## What the table shows

1. **Built:** 33 of 38 rows fully, 3 partly (A1, C2, D11), 1 not possible (D12), 1 waiting on plasma (F1).
2. **Used in readings:** the counts, disorder, levels, molecules, the nine instrument references, the sky sphere with its ranges, the donor halves and the cell-count floor all feed readings. The noise maps, nulls, transfer function and isotropy are checks that the atlas is right, which is their job.
3. **Built but not yet improving any reading:** the genome-distance spectrum and the covariance (D6, D8). Their planned use is a whole-genome C-Score on sequencing, whose healthy limit needs the healthy correlation between neighboring sites: new item 60.22, synthetic first. The Minkowski functionals (D13) wait on real-map readability.
4. **The information sphere on sequencing** still places sites by IAM-Atlas v2.1's layout. A layout from the sequencing atlas itself is open (35.30.12); its seven sky tools (35.5), five uses (35.4) and the side-by-side view with the address globe (35.8) are open.
5. **The book's chapter** carried the sphere, masks, noise maps, spectrum, transfer function, isotropy, bridge and cell pages; it now also carries the Minkowski functionals, the guidebook, the Globe Explorer, Group-A, the distance to the ceiling and the cell-count floor on sequencing (plan 61.4). The covariance appears as the spectrum's correlation by distance.

