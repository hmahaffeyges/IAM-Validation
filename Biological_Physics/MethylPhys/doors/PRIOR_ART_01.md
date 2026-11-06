# PRIOR-ART-01: what others have published, before any public claim of novelty (plan 61.5)

**The job:** before any public text says IAM's work is new, check the literature for each claim. A claim found in the literature is
reworded to what IAM adds; a claim not found stays worded as "not found in our search", never "never done before". References are named by
their work and DOI, not by their authors. Literature only; no data read.

## First pass (web search, four claims)

| Claim | What the search found | Wording from here |
|---|---|---|
| (a) A whole-genome atlas of healthy cell types | **Exists.** The published sequencing atlas of human cell types (39 cell types, 205 healthy samples, deep WGBS; doi:10.1038/s41586-022-05580-6) is the source IAM's sequencing atlas is counted from. It reports average methylation in blocks and does not store a range at every value or pool laboratories with a posterior. | IAM-Atlas v2.1's joint fit across laboratories with a range at every value, and the sequencing atlas's per-donor counts and measured disorder at every CpG, are what IAM adds. Do not call a whole-genome cell-type atlas new. |
| (b) Per-sample methylation disorder | **Exists, widely.** Methylation entropy and related measures (proportion of intermediately methylated sites, Shannon entropy, normalized methylation entropy, read-level pattern clustering) are established; tumors read more disordered than matched normal tissue across cancer types (e.g. doi:10.1186/s12967-026-08028-x; PMC10909509; PMC10018346; PMC7329512). | Disorder as a measurement is not new. IAM's distinct parts are the physical derivation of the floor and ceiling, and reading disorder as a ratio against the same person's own healthy cell type (claim c). |
| (c) Disorder as a ratio against a floor set by the person's own healthy cell type, with derived ends | **Not found in this pass.** The nearest is tumor against matched normal from the same person (paired analysis), which compares two samples rather than reading against a derived floor and ceiling. | "Not found in our search so far." Needs a dedicated second pass (method papers on per-sample normalization of entropy, and patents). |
| (d) A methylome placed on a HEALPix sphere and read with microwave-sky tools | **Not found in this pass.** HEALPix and its statistics are standard in cosmology; no methylation use came up. | "Not found in our search so far." Needs a second pass (spherical embeddings of genomic data; Fisher-Rao geometry on methylation). |

## Next
1. Second pass on (c) and (d): literature databases and patents, with the exact terms of each method.
2. A third claim set from the book's novelty statements: every sentence that says "first", "new" or "never" listed and checked.
3. Findings feed the paper's introduction and the book; nothing public changes until each claim's row is settled.

## Second pass (web search: the physics, the self-reference, the disorder ratio, the sphere)

Read from abstracts, search excerpts and the papers' own method summaries; the full texts are to be read before any public wording changes.
References are named by their work and DOI.

| Claim | What the search found | Wording from here |
|---|---|---|
| (e) A methyl mark read as information held against heat, with Landauer's bound (k_B T ln 2 per bit) setting the energy of a methylation change | **Published.** An information-thermodynamics model of cytosine methylation built on Landauer's principle (doi:10.1371/journal.pone.0150427), extended to a thermodynamic model of the methylation process with the fluctuation theorem, applied to plant and human cancer methylomes (doi:10.1038/s41598-023-35166-9). | IAM does not introduce Landauer's principle to methylation. What IAM adds has to be stated against this work: the same law across 37 orders of magnitude, and instruments built on it. |
| (f) A person's methylation read against a healthy reference, with thermal background separated from signal | **Published.** A signal-detection method (an R package, Zenodo record 4041756) models the background as thermal noise, builds a reference individual from pooled healthy controls, and scores each sample by an information divergence (Hellinger) from it; used for human clinical diagnostics (doi:10.3390/ijms20215343) and on public cancer methylomes (doi:10.1038/s41598-023-30422-4). | IAM's self-reference is not new as a principle. What may be new is the form: the expected disorder of the person's own healthy cell type at the sample's own mix and read depth, from a counted atlas, with nothing fitted. Not found in this pass; needs the full texts. |
| (g) Methylation disorder higher in tumors than in matched normal tissue | **Published, widely.** Per-read disorder and entropy measures, tumor against matched normal (e.g. a 2017 epiallele method, PMC5526259; a 2026 preprint on methylation disorder across tumor types, doi:10.64898/2026.03.12.711303; a patent family on proportion of discordant reads in leukemia against normal B cells, US10801070 and US11834718). | Never claimed as new; the books and papers cite it as known. |
| (h) Methylation disorder adjusted for cell-type composition by deconvolution | **Published (preprint).** Shannon entropy adjusted for tissue and cell-type composition across human tissues (doi:10.64898/2026.08.19.745715). | The nearest published work to Met-A in the sample's own mix. Met-A's ratio to a counted healthy expectation per cell at the sample's own depth: not found in this pass. |
| (i) Fisher-Rao geometry on a sphere for cell types | **Published for gene expression, not methylation.** Single-cell RNA cells as multinomials, distances as arcs on a hypersphere (doi:10.1093/bib/bbaf631.055). | The information sphere for methylation: not found in this pass. |
| (d) A methylome on a HEALPix sphere read with microwave-sky statistics | **Not found** in two passes; HEALPix appears in astronomy, cartography and spherical neural networks only. | "Not found in our search." |

**What this pass changes.** The physics root (a held mark priced by Landauer's bound) and the principle of reading a person against a healthy
reference have published precedent. Public text must not say IAM is the first to read methylation as information under Landauer's
principle or to read against a healthy reference. What remains not found: the law carried across scales into methylation; the counted
expectation of disorder per cell at the sample's own mix and depth, between two derived ends; the information sphere for methylation; the
methylome on a HEALPix sphere with sky statistics. Each stays "not found in our search" until the full texts and a patent search are read.

**Next.** Read the five full texts; a patent search on the four not-found items; every "first", "new" or "never" sentence in the book listed and
checked against this table.

**Full text read: the methylation-thermodynamics paper (doi:10.1038/s41598-023-35166-9).** Confirmed from the full text: each methylation change is priced at no less than k_B T ln 2 per bit (Landauer), about 1.8 J per mole at body temperature in the paper's own figure;
a methylome's divergence is measured against a reference (the centroid of a control group, or an independent control subset); its
Gibbs entropy rises in cancer cells against their corresponding normal tissue, all against one shared reference individual. Not in it: any
cell-type deconvolution or purity term (no mention), any per-cell expected disorder at the sample's own mix and read depth, and any fixed
ends of a reading scale; its scale is fitted (a generalized gamma model of the divergence, by maximum likelihood). The clinical
signal-detection paper (doi:10.3390/ijms20215343): the open-access copy retrieved is a different paper by the same group (a plant methylome
preprint, bioRxiv 252106); the clinical paper's full text is still to be read.

**Expected profile from the cell mix (search 3).** A published accuracy score for reference-based deconvolution measures how far a sample's methylation levels sit from the levels
expected from its estimated cell proportions and the cell types' reference profiles (PMC9980651). It is the nearest published form of "the
sample against its own healthy mix", applied to levels as a check on the deconvolution, not to disorder and not as a health reading. Met-A's
ratio of the sample's own disorder to the disorder its healthy mix would hold at its own read depth: still not found.

**The book already records claims (e) and (f) (found after the second pass; disclosed).** `docs/book/front/p0_giants.tex` ("the bridge crossed before") credits the 2016 information-thermodynamics paper and the 2019
signal-detection method (iam.bib `Sanchez2016`, `Sanchez2019`) as the first to put Landauer's bound on methylation, says IAM read them only
after its reference values were frozen, and states the difference: they use k_B T as the scale of a background to be filtered; IAM uses it as
the unit of measure (one filters, the other calibrates). What it names as IAM's addition is the floor: a holding energy measured on single
molecules of healthy cells, the floor's form from a Boltzmann factor, and each cell's held error read against it. This pass should have read
that chapter first (the agent's error, against the known-information rule). One addition for the book: the 2023 extension to human cancer
methylomes (doi:10.1038/s41598-023-35166-9) is not yet cited there.

**The book's novelty sentences.** Every "first", "never", "no one has" and "new" in the book's sources was listed (23 lines): none claims IAM
is first. The "never" sentences describe data an instrument has not seen (laboratories the atlas has never held); "no one has announced" is a
prediction's wording; "not a new law" says the opposite of a claim. The book's one "first" credits other researchers.
