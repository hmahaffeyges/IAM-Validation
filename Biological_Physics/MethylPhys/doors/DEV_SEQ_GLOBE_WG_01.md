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

**The builder and its synthetic, written and committed before any atlas value is read.** `Biological_Physics/MethylPhys/atlas/tools/seq_globe_wg.py`. It builds the globe from the published atlas (the 113 pieces of the Zenodo
deposit), one piece at a time: each piece gives, per cell, the sums and counts of its CpGs' level and range in each pixel it touches, and these
add exactly across pieces, so the build splits over machines and the joined map is the same as one built whole. The level and range are
`stage_seq_atlas.range_of_average` with each cell's number of healthy inputs from `seq_atlas_inputs.csv`, the current globe's rule and counts.
The synthetic builds a constructed three-cell atlas at all 28,217,448 CpGs in pieces of 250,000, exactly as the real build, at NSIDE 2048 and
1024, against the design's bars: 1. every CpG with reads lands in the map, for every cell; 2. a change planted in one cell at 10 neighboring
CpGs changes that cell's map at exactly those CpGs' pixels and nothing else anywhere; 3. at NSIDE 2048 no pixel holds two CpGs, and at NSIDE
1024 each pixel is the mean of the CpGs placed in it, within rounding. The design's third check (the new globe equals the current one at the
813,951 array-layout sites) is made on the real build, before the globe is used. A one-million-CpG run of the same code met all three (a code
check only, not the test); the full-size run is on GitHub (explore workflow).

**The synthetic, run 1 (all 28,217,448 CpGs, GitHub): not met** (`doors/data/DEV_SEQ_GLOBE_WG_01/seq_globe_wg_synthetic_run1.txt`). Bars 1 and 2 met: every CpG with reads landed in the map for each
cell, and the change planted at 10 CpGs changed exactly those 10 pixels in that cell and nothing else. Bar 3 not met by a rounding margin: at
NSIDE 2048 no pixel holds two CpGs (met), but at NSIDE 1024 the largest gap between a pixel and the mean of its CpGs printed 0.000050 against
the 0.00005 that rounding to 0.0001 allows. **Cause (my code):** the builder kept its per-pixel sums in 32-bit numbers, whose error of about
one part in ten million can tip a value lying on a rounding boundary to the next step. **Fix:** the sums are kept in 64-bit; the bars are
unchanged. Run 2 is the same synthetic on the fixed builder.

**The synthetic, run 2 (all 28,217,448 CpGs, GitHub, builder with 64-bit sums): met** (`doors/data/DEV_SEQ_GLOBE_WG_01/seq_globe_wg_synthetic_run2.txt`). All three bars met: every CpG with reads lands; the planted change
shows at exactly its 10 pixels in its own cell only; one CpG to a pixel at NSIDE 2048, and at NSIDE 1024 each pixel the mean of its CpGs
within rounding. **Resolution chosen: NSIDE 2048**, where every CpG keeps its own pixel and nothing is averaged (47 % of pixels hold no CpG by
the layout, masked). **The build:** workflow `globe_wg.yml`, one published piece set per GitHub job (12 jobs), every piece checked against the
deposit's SHA256 manifest before it is read; each piece writes its own stretch of the globe (`seq_globe_wg.py slice`), with NSIDE 256 sums
for the Globe Viewer's textures. Before the globe is used: the design's third check, the new globe against the current one at the 813,951
array-layout sites.

**The build and the check against the current globe: met** (`doors/data/DEV_SEQ_GLOBE_WG_01/globe_wg_check.json`; build run 42317003484, 12 of 12 published piece sets, every piece checked against the
deposit's SHA256 manifest; check run 42320588128). At every one of the 813,951 array-layout sites, for all 57 cells, the new globe's level, low
and high equal the current Sequencing Atlas Globe's rule applied to the committed array-site columns: 139,185,168 values compared, 139,185,168
equal, none on one globe only, largest gap 0. The Sequencing Atlas Globe at every CpG (NSIDE 2048, all 28,217,448 CpGs, 57 cells) is built.
The Globe Viewer textures (NSIDE 256) are joined from the same slices. **Next:** the full-resolution globe deposited with the atlas on Zenodo
(GitHub keeps run artifacts 30 days), then the Globe Viewer and the website page.
