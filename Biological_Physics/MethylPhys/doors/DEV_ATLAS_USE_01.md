# DEV-ATLAS-USE-01: atlas v2.1 on the jobs Loyfer and Moss use an atlas for (ATL-060), SEALED 2026-10-15 before the atlas exists

**In plain words.** A reference atlas is used for two jobs: finding the DNA spots that mark each cell type, and working out which cells a
mixed sample (such as blood) came from. These tests ask whether atlas v2.1 does both jobs at least as well as Moss's published reference on
the same held-out samples, and whether, because it holds more cell types, it can account for cells the smaller reference cannot see,
the way a grid operator sees past the edge of his own control area. They are a showcase, not a gate: a miss is reported and never blocks
the release.

**The tests and bars** (each with its reason; full text in `data/DEV_ATLAS_USE_01/use_tests_v21.py`):
- UT1 markers: 100 marker sites per cell from each reference, confirmed on held-out arrays when the cell differs from every other shared
  cell by at least 0.20; v2.1 must confirm at least Moss's share minus 0.05, per cell.
- UT2A mixtures on shared ground: 200 mixtures of the eight shared cells; v2.1's median error at most 1.25 times Moss's.
- UT2B wide area: the same with 10-30 percent from cells only v2.1 holds (microglia, oligodendrocytes, astrocytes, lung alveolar
  macrophages); v2.1 must recover that share within 0.05 in at least 90 percent of mixtures, with median error at most 0.03.
- UT3 extras: shown on one blood cell, not scored.

**Sealed inputs:** `data/DEV_ATLAS_USE_01/sealed_use_v21.json` (sample lists by cell, the scorer's and Moss reference's checksums, the
disclosures). The real run is refused in code until the v2.1 release. Amendment (same day, before any data): GSE132547 and GSE131989 are scored only after
their carried checks D13 and D14 are recorded, as in the L15 bars.

**Synthetic test:** DEV_ATLAS_USE_01/use_tests_v21.py, DEV_ATLAS_USE_01/use_tests_synthetic.txt

The synthetic ran twice. Run 1 (kept as use_tests_synthetic_run1_NOTMET.txt) showed an absolute mixture bar (Moss plus 0.01) passed a planted
v2.1 four times noisier than Moss, so UT2A became a ratio before any real value was read. Run 2 read MET: equal references pass; a noisier
v2.1 fails UT1 and UT2A; the wide-area arm recovers the outside cells and fails when their profiles are wrong.

What it would take to succeed: v2.1 matching Moss's reference on markers and shared-ground mixtures, and recovering the cells Moss's reference
does not hold; finish line, not a promise.
