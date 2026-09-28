# PROC-PARTIALCOV-01 — pre-registration: entering a cell measured on part of the array

> **NOT ADOPTED (author, 2026-09-28):** "I'd rather not have a cell than only have part of one because I know what kind of headache that causes later. We can experiment later with them and see if they even benefit the atlas and if so we can splice it in later." The atlas takes whole-array cells only. The GSE262275 cells stay on the box (atlas_sources/mcnamara2025) as an experiment for later; this rule is kept as the record of what was proposed and why it was declined.

**Written 2026-09-28, after GSE262275's depth was read (all 14 fail the v2 WGBS rule: 27–59 % of array CpGs with >= 10 reads; 13 of 14 at 48–59 %, keratinocyte HKER-B at 27 % with median depth 3, so under rule 1 below the keratinocyte has one qualifying sample and does not enter) and
before any new cell is fitted, scored or tested.** Why it exists: GSE262275 holds hepatic stellate, Kupffer cells and biliary
epithelium from three sites — the cell of bile-duct cancer — none in the atlas, all distinct from every Loyfer cell (twin test, Q3),
but each measured on only ~55 % of the array (a targeted capture panel). The author wants liver and bile cells in, on quality.

## Rule
1. **Entry.** A cell measured on part of the array enters if >= 2 samples each have >= 10 reads at >= 50 % of array CpGs, and it
   passes the sample-level twin test against every atlas cell of its family. A pair that reads as one donor (Q2) enters flagged
   *one donor — donor SD unmeasured*.
2. **Fit.** Appended to atlas v2 after acceptance, per locus, with v2's per-locus prior and source terms held fixed (the append fit).
   Where the cell has data, its mean, SD and interval come from its own samples. **Where it has none, nothing is imputed**: the locus
   is marked NOT MEASURED for that cell — no prior-only mean enters any calculation.
3. **Identity loci.** Chosen only from addresses the cell measured. Its A is computed only there.
4. **Composition and detection.** The cell's column in the composition fit and the out-of-span projector is restricted to its measured
   addresses (the fit runs on the intersection when this cell is a candidate).
5. **Source term.** GSE262275 shares no cell with our arrays; it is WGBS-format capture sequencing in Loyfer's lab format. It takes
   Loyfer's transfer, status **ASSUMED**, until a bridge is measured.
6. **Flags on every reading of the cell:** PARTIAL COVERAGE (fraction measured), ASSUMED (source term), and CULTURED if the paper's
   methods state culture.

## Acceptance (the same test every v2 cell takes)
- **P1** each cell reads A within NORMAL on its own samples (the v2 A1 self-read), held-out sample where there are two donors.
- **P2** a constructed whole-blood specimen spiked with the cell at 5 % names it, with every other v2 cell's reading unchanged
  beyond 0.005 in A.
- **P3** no other v2 cell's identity loci move when the cell is appended.
A cell that fails any of P1–P3 does not enter; its failure is recorded.
