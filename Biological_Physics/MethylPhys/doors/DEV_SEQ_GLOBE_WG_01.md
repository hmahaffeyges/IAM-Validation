# DEV-SEQ-GLOBE-WG-01: the Sequencing Atlas Globe at every CpG (design; top priority beside commissioning in the plan)

**The instrument's job:** every one of the IAM Sequencing Atlas's 28,217,448 CpGs is placed on the globe, for each of its 57 cell types,
with its healthy level and 95 % range, so a sequenced sample can be compared with its healthy self at every CpG the sample was read at.

**Why.** The Sequencing Atlas Globe now in use was drawn by the array atlas's builder on the array atlas's 813,951-site layout: it shows the
sequencing atlas's values at the array's sites only, about 3 % of the CpGs the atlas holds. The globe is a comparison tool; on a layout of
array sites it cannot show a change that sits between them, which is what a sequencing atlas exists to see.

**The layout.** The hg19 CpG numbering is already in genome order (chr1 to chr22, X, Y), so CpG i of n goes to HEALPix pixel
floor(i x NPIX / n), nested ordering, the rule the array globes use (`sky_maps_v21.to_sky`). Two resolutions are measured in the synthetic
before one is chosen:
- NSIDE 1024 (12,582,912 pixels): each pixel averages two or three neighboring CpGs; no pixel empty by construction.
- NSIDE 2048 (50,331,648 pixels): each CpG its own pixel; about 44 % of pixels empty in a regular pattern, masked.

**The values.** Each cell's level is the pooled count (methylated over all reads of its donors), and its 95 % range the rule the current
globe already passed, computed at
every CpG from the atlas's own pieces (`seq_atlas/pieces57`), not from the 813,951-site columns.

The range rule's own test: version 3 of the range synthetic in DEV_ATLAS_V21_01.md (script `doors/data/DEV_ATLAS_V21_01/seq_atlas/sphere_range_synthetic.py`, output `sphere_range_synthetic_v3.txt`: stated variance 1.010 and 1.007 of the true scatter, coverage 0.955 and 0.948).

**The synthetic, to be written and sealed with the code** (constructed atlas of a few cell types at every CpG, known truth):
1. Every CpG with a value lands in the map: the count of CpGs contributing equals the count measured, for every cell.
2. A planted change of one CpG island's size (about 10 neighboring CpGs) in one cell is seen at the planted place and nowhere else
   (the change test the Sample Globe already uses: blocks beyond the 5 % family-wise line).
3. At the 813,951 array-layout sites, the new globe's values equal the current globe's (same counts, same rule): nothing changes
   where both exist.

**The machine.** One Google spot machine (memory sized from the pieces: 57 cells x 28.2 million values x 4 bytes, about 6.4 GB per map
kind, built cell by cell), the textures for the Globe Viewer drawn at a size a browser can hold, the full-resolution maps deposited with the
atlas on Zenodo. Then the Sample Globe on sequencing is read on it, synthetic first, and the cell pages and Globe Viewer move to it.

**When:** the design now; the build as soon as the B and T and Johns Hopkins machines free a slot. Status reported in every progress update.
