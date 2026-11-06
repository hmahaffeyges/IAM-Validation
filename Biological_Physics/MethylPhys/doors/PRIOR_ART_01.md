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
