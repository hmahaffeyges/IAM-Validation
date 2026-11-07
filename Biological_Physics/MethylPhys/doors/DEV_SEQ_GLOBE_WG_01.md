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

**Published on Zenodo: 10.5281/zenodo.24025624** (author's yes). The globe was rebuilt on a Google machine from the published atlas by the committed deposit job (`deposit/zenodo_globe_upload.py`), each piece
checked against the atlas deposit's SHA256 manifest, one set at a time, each set deposited as it finished; the job rebuilt the textures and the check
against the current globe there and stops before depositing them if the check is not met. The record holds the 12 set tars (the NSIDE 2048 slices,
stored compressed), the NSIDE 256 textures, the check, the synthetic, the builder, the README and the checksums.

**The CMB tools at every CpG (written before they read the atlas).** Until now the globe's noise maps, mask, null test and spectrum were computed at the 813,951 array-layout sites only. The published pieces
hold, at every CpG, each cell's two donor halves and two read halves, so both noise maps are made at all 28,217,448 CpGs from the Zenodo
deposit alone (`seq_globe_wg.py`: `noise-part`, `noise-join`): map 1, donor half A minus donor half B; map 2, read half A minus read half B.
**Mask:** the sex chromosomes, as on the array layout. The array layout's second mask, probes that cannot be placed uniquely, is a list of
array probes; a whole-genome mappability mask is not in hand and is recorded as not yet applied. **Per cell and map:** the mean and mean
square (the donor part, map 1's minus map 2's); the genome-distance spectrum (`stage_seq_atlas.genome_spectrum`'s pairs, taken within each
250,000-CpG piece so the sums add across machines); the sphere null test (NSIDE 256 coarse map, each NSIDE 128 pixel the mean of its four,
`sky_statistics.sky_spectrum_record`, apodised mask, 20 shuffles), as `stage_seq_atlas.sky_null` does on the array layout. **Synthetic first**
(`noise-synthetic`, all 28,217,448 CpGs): two constructed cells, one with structure planted in its donor halves only. Bars: 1. X and Y left out
and each map's sums from the pieces equal the whole map's; 2. the planted map shows structure on the sphere (a band above all shuffles, power
over 3 times their mean) and along the genome (correlation above 0.05 at 1 to 10 kb and 10 to 100 kb); 3. the other three maps show none (no
band over 1.5 times the shuffles' mean, every correlation within 0.01 of 0); 4. the pieces' spectrum within 0.01 of the whole map's. **Not yet at
every CpG (recorded):** Minkowski functionals and the information sphere.

**The noise tools' synthetic: met** (`doors/data/DEV_SEQ_GLOBE_WG_01/seq_globe_wg_noise_synthetic.txt`, GitHub run 42338862102; all 28,217,448 constructed CpGs). All four bars met: X
and Y left out and each map's sums from the pieces equal the whole map's; the planted donor-halves map showed structure on the sphere (bands
above all 20 shuffles, the largest 86.6 times their mean) and along the genome (correlation 0.09 from under 1 kb to 1 to 10 Mb); the three other
maps showed none (every band within 1.5 times the shuffles' mean, every correlation within 0.01 of 0); the pieces' spectrum within 0.00152 of
the whole map's. **Recorded:** one pure-noise map (the planted cell's read halves) put two bands just above all 20 shuffles (ratios 1.0126 and
1.0264): above-every-shuffle alone flags by chance, so every band is reported with its ratio to the shuffles' mean beside it. The real run is
next (`globe_noise.yml`, one published set per machine, then the join).

**The noise maps at every CpG, measured on the published atlas.** (`doors/data/DEV_SEQ_GLOBE_WG_01/globe_wg_noise_summary.txt`, `globe_wg_noise.json`; GitHub run 42339899387, all 113 pieces.) **Read halves
(counting only):** all 57 cells; no cell shows structure, no sphere band above all 20 shuffles in any cell and a genome-distance
correlation of 0.0005 (median, under 1 kb): what is left after the donors are taken out is counting, as the physics says.
**Donor halves:** all 49 cells with two or more donors show structure on the sphere and along the genome (correlation 0.0838
under 1 kb, falling with distance): healthy people differ from one another in regions, not CpG by CpG. **Open, before the page shows it:** in the
read-halves maps every band's power is below the shuffles' mean (median of each cell's lowest band 0.5705); the synthetic's
pure-noise maps sat near 1. The synthetic had even depth everywhere, the atlas does not, so the cause is likely the uneven depth along the genome
meeting the apodised mask's weights; it is checked (a synthetic with the atlas's own depth along the genome) before this ratio is shown or used.
The whole-sphere maps at NSIDE 256 (259 MB) go to Zenodo with the page.

**The read-halves ratio below 1: the cause found, and the sign-flip null (its synthetic written before it runs on GitHub).** With the plain spectrum the B-cell read-halves map matches its shuffles from l 9 to 383 (1.021, 1.006, 0.998): pure counting noise.
The low ratio comes from the null, not the map: the pixels the apodised mask tapers, beside the mask and the coverage gaps, carry about three
times the noise of the rest (0.0136 against 0.0048 a pixel; CpGs beside centromeres and unmappable stretches hold fewer reads). The mask
weighs them down in the map, but a shuffle carries their large values into full-weight places, so the shuffles hold more power than the map.
The CMB meets the same thing (noise highest where the scan is thinnest) and answers it with noise simulations that keep each pixel's own noise.
For a half-difference map that is exact: flipping each pixel's sign at random draws a fresh noise map with every pixel's level kept
(`seq_globe_wg.py` `flip_null`; on the same B-cell map, apodised: 0.894 to 1.348). The join now records the flips beside the shuffles; the
shuffles stay as a record only.

**Range of the data:** real (the read-halves maps, `globe_wg_noise_maps_n256.npz`): local noise level 5.3 to 5.8 times from the 1st to the
99th percentile of each map, 2.1 to 2.2 times from the 5th to the 95th, 94.5 % of pixels holding data, the tapered pixels about 2.8 times
louder (0.0136 against 0.0048); synthetic (`flip-synthetic`, NSIDE 128): noise 1 to 6 times its quietest level with a long tail, 94.5 % of
pixels holding data plus 1 % in gaps, the tapered pixels 3 times louder.

Bars: 1. pure noise (even, uneven, louder at the edges): every band within 3 standard deviations of the flips' own ratios, none above all 20
flips at over 1.5 times their mean; 2. planted structure above all 20 flips at over 3 times their mean. **Disclosed:** a first local code check
held bar 1 to a fixed 0.85 to 1.15 and failed at the lowest band (l 2 to 8, about 60 modes) at 0.74 on even noise: a band of few modes
scatters that widely by itself (cosmic variance, flips' own scatter 0.185), so the bar is set by the flips' scatter; the local check with it
was met. The run on GitHub is the record.

**The sign-flip null's synthetic on GitHub: met** (`doors/data/DEV_SEQ_GLOBE_WG_01/seq_globe_wg_flip_synthetic.txt`, run 42344172326): pure noise, even, uneven or louder at the edges, within the flips' own scatter in every band and nothing flagged; the planted structure above all 20 flips (613.8 and 23.7 times their mean at the two largest scales). On the edge-loud noise the shuffles read 0.61 to 0.70, as the real read-halves maps did: the cause is reproduced. **One story source** for the page, the book, the SOP and the catalogue: `CANON/globe_wg_story.json` (13 parts, each with its CMB name, the name a geneticist uses, its physics, its result and the record behind it, and the rebuild steps from Zenodo and GitHub alone).
