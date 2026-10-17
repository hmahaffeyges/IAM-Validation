# DEV-ATLAS-USE-02: atlas v2.1 head to head against Loyfer's own reference (ATL-067)

**The instrument's job:** for the jobs a reference atlas is used for, read held-out arrays at least as well as Loyfer's own reference.

The Moss half of this comparison met all three bars on 2026-10-16 (DEV-ATLAS-USE-01). This is the Loyfer half, on the same held-out
arrays, the same sealed scorer (unchanged, checked by its SHA-256) and the same bars.

**Loyfer's reference**, built his way: his 205 samples from GEO in his own 39 groups (his Supplementary Table S1), projected onto array
probes with his tool's rule (coverage at least 3), then the mean of each group (`data/DEV_ATLAS_USE_02/build_loyfer_array_ref.py`).

| bar | rule |
|---|---|
| UT1 | markers that hold on held-out arrays: v2.1's share at least Loyfer's minus 0.05 |
| UT2A | mixtures of shared cells: v2.1's median error at most 1.5 x Loyfer's |
| UT2B | mixtures including cells Loyfer lacks (microglia, astrocytes): v2.1's median error at most half of Loyfer's |

**Sealed 2026-10-16** (`data/DEV_ATLAS_USE_02/sealed_use_loyfer_v21.json`), before Loyfer's projected reference was built.

**Synthetic test:** DEV_ATLAS_USE_02/use_tests_loyfer_v21.py

Run with `planted`: a planted opponent, and a v2.1 with scrambled outside-cell profiles that must fail UT2B.

**Sources:** public: Loyfer et al. 2023, Nature 613:355, GEO GSE186458 (.beta files) and his Supplementary Tables S1 and S12; the held-out arrays named in DEV_ATLAS_USE_01/sealed_use_v21.json.

**Disclosed:** v2.1 contains Loyfer's sequencing (it was built from it plus arrays); Loyfer pools CD4 and CD8 T cells, so both atlases use one T-cell profile; his monocyte group includes macrophages; the held-out arrays were read before (U1, Moss); his reference has never been built or read.

**What it would take to succeed:** all three bars met.

**In plain words:** Loyfer and Moss each published a reference of what healthy cells look like. Our atlas already beat Moss's on his own jobs; this runs the same contest against Loyfer's, built exactly the way his paper builds it.
